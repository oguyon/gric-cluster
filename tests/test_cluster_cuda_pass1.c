/**
 * @file test_cluster_cuda_pass1.c
 * @brief Unit and end-to-end verification tests for GPU Anchor Store and GPU Pass 1.
 */

#include "cuda_anchor_store.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void create_test_spiral_data(
    const char *filename,
    int         npoints)
{
    FILE *fp = fopen(filename, "w");
    assert(fp != NULL);

    for (int i = 0; i < npoints; i++)
    {
        double theta = (double)i * 0.05;
        double r = 0.01 + 0.0005 * (double)i;
        double x = r * cos(theta);
        double y = r * sin(theta);
        fprintf(fp, "%.6f %.6f\n", x, y);
    }

    fclose(fp);
}

static const char *get_binary_path(void)
{
    if (access("./gric-cluster", X_OK) == 0)
    {
        return "./gric-cluster";
    }
    if (access("./build/gric-cluster", X_OK) == 0)
    {
        return "./build/gric-cluster";
    }
    return "gric-cluster";
}

/**
 * test_gpu_anchor_store_unit() - Direct numerical unit tests of GpuAnchorStore.
 */
static void test_gpu_anchor_store_unit(void)
{
    printf("Testing GpuAnchorStore unit correctness...\n");

    const int dim = 32;
    const int max_anchors = 64;
    const int num_anchors = 8;
    const int num_queries = 16;

    GpuAnchorStoreConfig config;
    config.max_clusters = max_anchors;
    config.dim = dim;
    config.device_id = 0;
    config.max_batch_size = num_queries;

    GpuAnchorStore *store = gpu_anchor_store_create(&config);
    assert(store != NULL);
    assert(gpu_anchor_store_get_count(store) == 0);

    /* Generate synthetic anchors */
    float *anchors = (float *)malloc((size_t)num_anchors * (size_t)dim * sizeof(float));
    assert(anchors != NULL);

    for (int k = 0; k < num_anchors; k++)
    {
        float *a = anchors + (size_t)k * (size_t)dim;
        for (int d = 0; d < dim; d++)
        {
            a[d] = ((float)(k + 1) / (float)num_anchors) * ((float)d / (float)dim);
        }
        int appended = gpu_anchor_store_append_anchor(store, a, 0);
        assert(appended == k);
    }
    assert(gpu_anchor_store_get_count(store) == num_anchors);

    /* Generate synthetic queries */
    float *queries = (float *)malloc((size_t)num_queries * (size_t)dim * sizeof(float));
    assert(queries != NULL);

    for (int q = 0; q < num_queries; q++)
    {
        float *qv = queries + (size_t)q * (size_t)dim;
        for (int d = 0; d < dim; d++)
        {
            qv[d] = ((float)(q % num_anchors + 1) / (float)num_anchors) *
                    ((float)d / (float)dim) + 0.02f;
        }
    }

    /* Test find_nearest */
    int   *best_cl = (int *)malloc((size_t)num_queries * sizeof(int));
    float *best_dist = (float *)malloc((size_t)num_queries * sizeof(float));
    assert(best_cl != NULL && best_dist != NULL);

    int res = gpu_anchor_store_find_nearest(store, queries, num_queries, 0,
                                            best_cl, best_dist);
    assert(res == 0);

    /* Verify against CPU ground truth */
    for (int q = 0; q < num_queries; q++)
    {
        const float *qv = queries + (size_t)q * (size_t)dim;
        float ref_min_dist = 1e30f;
        int   ref_best_cl = -1;

        for (int k = 0; k < num_anchors; k++)
        {
            const float *a = anchors + (size_t)k * (size_t)dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = qv[d] - a[d];
                sum_sq += diff * diff;
            }
            float dist = sqrtf(sum_sq);
            if (dist < ref_min_dist)
            {
                ref_min_dist = dist;
                ref_best_cl = k;
            }
        }

        printf("q=%d: best_cl=%d, ref_cl=%d, best_dist=%.6f, ref_dist=%.6f, diff=%.6e\n",
               q, best_cl[q], ref_best_cl, best_dist[q], ref_min_dist,
               fabsf(best_dist[q] - ref_min_dist));
        assert(best_cl[q] == ref_best_cl);
        assert(fabsf(best_dist[q] - ref_min_dist) < 1e-3f);
    }

    /* Test find_top_m for m = 3 */
    const int m = 3;
    int   *top_cl = (int *)malloc((size_t)num_queries * (size_t)m * sizeof(int));
    float *top_dist = (float *)malloc((size_t)num_queries * (size_t)m * sizeof(float));
    assert(top_cl != NULL && top_dist != NULL);

    res = gpu_anchor_store_find_top_m(store, queries, num_queries, 0, m,
                                      top_cl, top_dist);
    assert(res == 0);

    for (int q = 0; q < num_queries; q++)
    {
        /* Nearest in top_m must equal best_cl */
        assert(top_cl[q * m] == best_cl[q]);
        assert(fabsf(top_dist[q * m] - best_dist[q]) < 1e-4f);

        /* Distances must be sorted non-decreasing */
        for (int j = 0; j < m - 1; j++)
        {
            assert(top_dist[q * m + j] <= top_dist[q * m + j + 1] + 1e-5f);
        }
    }

    gpu_anchor_store_destroy(store);
    free(anchors);
    free(queries);
    free(best_cl);
    free(best_dist);
    free(top_cl);
    free(top_dist);

    printf("GpuAnchorStore unit tests passed successfully.\n");
}

/**
 * test_gpu_pass1_e2e() - End-to-end test of gric-cluster --gpu-brute-force.
 */
static void test_gpu_pass1_e2e(void)
{
    printf("Testing gric-cluster --gpu-brute-force end-to-end...\n");

    const char *test_file = "/tmp/test_gpu_pass1_data.txt";
    const char *out_dir = "/tmp/test_gpu_pass1_out";
    const char *bin = get_binary_path();
    const int   npoints = 800;
    const double rlim = 0.05;

    create_test_spiral_data(test_file, npoints);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s %s -rlim %.4f --gpu-brute-force --gpu-batch-size 64 -txt -outdir %s > /dev/null",
             bin, test_file, rlim, out_dir);
    int res = system(cmd);
    assert(res == 0);

    /* Read frame_membership.txt and verify every frame satisfies d <= rlim */
    char path_mem[512];
    snprintf(path_mem, sizeof(path_mem), "%s/frame_membership.txt", out_dir);
    FILE *fp = fopen(path_mem, "r");
    assert(fp != NULL);

    int count = 0;
    long frame_id;
    int cluster_id;
    double dist;
    int max_cluster = -1;

    while (fscanf(fp, "%ld %d %lf", &frame_id, &cluster_id, &dist) == 3)
    {
        assert(frame_id == count);
        assert(cluster_id >= 0);
        assert(dist <= rlim + 1e-4);
        if (cluster_id > max_cluster)
        {
            max_cluster = cluster_id;
        }
        count++;
    }
    fclose(fp);

    assert(count == npoints);
    assert(max_cluster > 0);
    printf("GPU Pass 1 created %d clusters for %d frames; all dist <= rlim.\n",
           max_cluster + 1, count);
}

/**
 * test_gpu_pass1_and_pass2() - Verify combined Pass 1 + Pass 2 on GPU.
 */
static void test_gpu_pass1_and_pass2(void)
{
    printf("Testing combined GPU Pass 1 + Pass 2 (--gpu-brute-force -pass2nearest)...\n");

    const char *test_file = "/tmp/test_gpu_p1p2_data.txt";
    const char *out_dir = "/tmp/test_gpu_p1p2_out";
    const char *bin = get_binary_path();
    const int   npoints = 500;
    const double rlim = 0.04;

    create_test_spiral_data(test_file, npoints);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s %s -rlim %.4f --gpu-brute-force -pass2nearest -txt -outdir %s > /dev/null",
             bin, test_file, rlim, out_dir);
    int res = system(cmd);
    assert(res == 0);

    char path_mem[512];
    snprintf(path_mem, sizeof(path_mem), "%s/frame_membership.txt", out_dir);
    FILE *fp = fopen(path_mem, "r");
    assert(fp != NULL);

    int count = 0;
    long frame_id;
    int cluster_id;
    double dist;

    while (fscanf(fp, "%ld %d %lf", &frame_id, &cluster_id, &dist) == 3)
    {
        assert(frame_id == count);
        assert(cluster_id >= 0);
        assert(dist <= rlim + 1e-4);
        count++;
    }
    fclose(fp);
    assert(count == npoints);

    printf("Combined GPU Pass 1 + Pass 2 test passed successfully.\n");
}

int main(void)
{
    test_gpu_anchor_store_unit();
    test_gpu_pass1_e2e();
    test_gpu_pass1_and_pass2();

    printf("\nAll GPU Pass 1 tests passed!\n");
    return 0;
}

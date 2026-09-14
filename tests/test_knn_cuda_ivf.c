/**
 * @file test_knn_cuda_ivf.c
 * @brief Tests verifying GPU Inverted-File (IVF) hierarchical metric pruned k-NN search.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void create_spiral_dataset(
    const char *path,
    int         nframes)
{
    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (int i = 0; i < nframes; i++)
    {
        double theta = (double)i * 0.05;
        double r = 0.01 + 0.0005 * (double)i;
        double x = r * cos(theta);
        double y = r * sin(theta);
        fprintf(fp, "%.6f %.6f\n", x, y);
    }

    fclose(fp);
}

static void get_binary_path(
    const char *name,
    char       *out,
    size_t      sz)
{
    snprintf(out, sz, "./%s", name);
    if (access(out, X_OK) == 0)
    {
        return;
    }
    snprintf(out, sz, "./build/%s", name);
    if (access(out, X_OK) == 0)
    {
        return;
    }
    snprintf(out, sz, "%s", name);
}

static int compare_binary_indices(
    const char *file1,
    const char *file2,
    int         nqueries,
    int         k)
{
    FILE *f1 = fopen(file1, "rb");
    FILE *f2 = fopen(file2, "rb");
    assert(f1 != NULL && f2 != NULL);

    size_t total = (size_t)nqueries * (size_t)k;
    int *idx1 = (int *)malloc(total * sizeof(int));
    int *idx2 = (int *)malloc(total * sizeof(int));
    assert(idx1 != NULL && idx2 != NULL);

    size_t r1 = fread(idx1, sizeof(int), total, f1);
    size_t r2 = fread(idx2, sizeof(int), total, f2);
    assert(r1 == total && r2 == total);

    fclose(f1);
    fclose(f2);

    int matches = 0;
    for (size_t i = 0; i < total; i++)
    {
        if (idx1[i] == idx2[i])
        {
            matches++;
        }
    }

    free(idx1);
    free(idx2);
    return matches;
}

static void test_knn_cuda_ivf_features(void)
{
    const char *data_file = "/tmp/test_knn_ivf_data.txt";
    const char *cluster_dir = "/tmp/test_knn_ivf_clusters";
    char cluster_bin[256];
    char knn_bin[256];
    get_binary_path("gric-cluster", cluster_bin, sizeof(cluster_bin));
    get_binary_path("gric-knn", knn_bin, sizeof(knn_bin));

    int nframes = 1000;
    int k = 10;
    char cmd[1024];

    create_spiral_dataset(data_file, nframes);

    /* Generate Pass 1 clusters */
    snprintf(cmd, sizeof(cmd),
             "%s %s -rlim 0.10 -outdir %s > /dev/null 2>&1",
             cluster_bin, data_file, cluster_dir);
    int rc = system(cmd);
    assert(rc == 0);

    /* Test 1: CPU vs GPU IVF Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --cpu -o /tmp/ivf_test_cpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --gpu -o /tmp/ivf_test_gpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/ivf_test_cpu_indices.bin",
            "/tmp/ivf_test_gpu_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  GPU IVF vs CPU parity: %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    /* Test 2: GPU IVF vs GPU Brute-Force (--gpu-bf) Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --gpu --gpu-bf -o /tmp/ivf_test_bf.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/ivf_test_gpu_indices.bin",
            "/tmp/ivf_test_bf_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  GPU IVF vs GPU Brute-Force parity: %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    /* Test 3: -gpu-nprobe parameter verification */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --gpu -gpu-nprobe 8 -o "
                 "/tmp/ivf_test_nprobe.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/ivf_test_cpu_indices.bin",
            "/tmp/ivf_test_nprobe_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  GPU IVF (-gpu-nprobe 8) parity: %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 95.0);
    }

    printf("  PASS: test_knn_cuda_ivf_features\n");
}

int main(void)
{
    printf("Running GPU IVF k-NN Tests...\n");
    test_knn_cuda_ivf_features();
    printf("All GPU IVF k-NN tests passed!\n");
    return 0;
}

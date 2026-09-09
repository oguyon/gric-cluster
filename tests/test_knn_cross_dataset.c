/**
 * @file test_knn_cross_dataset.c
 * @brief Integration tests for cross-dataset k-NN search (gric-knn -query).
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gric_bin_io.h"

#define TEST_A_TXT "/tmp/test_cross_A.txt"
#define TEST_C_TXT "/tmp/test_cross_C.txt"
#define TEST_CLUSTER_DIR "/tmp/test_cross_cluster"
#define TEST_KNN_OUT "/tmp/test_cross_knn_out.txt"

int main(void)
{
    printf("=== Starting Cross-Dataset k-NN Tests ===\n");

    // 1. Generate Dataset A (500 samples, 2D spiral)
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "./gric-mktxtseq 500 %s 2Dspiral", TEST_A_TXT);
    int ret = system(cmd);
    assert(ret == 0);

    // 2. Cluster Dataset A
    snprintf(cmd, sizeof(cmd), "./gric-cluster 0.1 %s -outdir %s", TEST_A_TXT, TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // 3. Create Dataset C (50 query samples derived from A with slight shift)
    FILE *fa = fopen(TEST_A_TXT, "r");
    assert(fa != NULL);
    FILE *fc = fopen(TEST_C_TXT, "w");
    assert(fc != NULL);

    double a_samples[500][2];
    int count_a = 0;
    char line[1024];

    while (fgets(line, sizeof(line), fa) != NULL && count_a < 500)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }
        if (sscanf(line, "%lf %lf", &a_samples[count_a][0], &a_samples[count_a][1]) == 2)
        {
            count_a++;
        }
    }
    fclose(fa);
    assert(count_a == 500);

    double c_samples[50][2];
    for (int i = 0; i < 50; i++)
    {
        // Select every 10th sample and add small perturbation
        c_samples[i][0] = a_samples[i * 10][0] + 0.0005;
        c_samples[i][1] = a_samples[i * 10][1] - 0.0005;
        fprintf(fc, "%.8f %.8f\n", c_samples[i][0], c_samples[i][1]);
    }
    fclose(fc);

    // 4. Run gric-knn with -query
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -query %s -k 5 -o %s",
             TEST_A_TXT, TEST_CLUSTER_DIR, TEST_C_TXT, TEST_KNN_OUT);
    ret = system(cmd);
    assert(ret == 0);

    // 5. Verify results against brute force
    FILE *fout = fopen(TEST_KNN_OUT, "r");
    assert(fout != NULL);

    int queries_verified = 0;
    while (fgets(line, sizeof(line), fout) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        long q_id = -1;
        int n_ids[5];
        double n_dists[5];

        int scanned = sscanf(line, "%ld %d %lf %d %lf %d %lf %d %lf %d %lf",
                             &q_id,
                             &n_ids[0], &n_dists[0],
                             &n_ids[1], &n_dists[1],
                             &n_ids[2], &n_dists[2],
                             &n_ids[3], &n_dists[3],
                             &n_ids[4], &n_dists[4]);
        assert(scanned == 11);
        assert(q_id == queries_verified);

        // Compute brute-force distances for this query
        double all_dists[500];
        int sorted_ids[500];
        for (int j = 0; j < count_a; j++)
        {
            double dx = c_samples[q_id][0] - a_samples[j][0];
            double dy = c_samples[q_id][1] - a_samples[j][1];
            all_dists[j] = sqrt(dx * dx + dy * dy);
            sorted_ids[j] = j;
        }

        // Partial sort for top 5
        for (int p = 0; p < 5; p++)
        {
            int min_idx = p;
            for (int j = p + 1; j < count_a; j++)
            {
                if (all_dists[j] < all_dists[min_idx])
                {
                    min_idx = j;
                }
            }
            double tmp_d = all_dists[p];
            all_dists[p] = all_dists[min_idx];
            all_dists[min_idx] = tmp_d;

            int tmp_id = sorted_ids[p];
            sorted_ids[p] = sorted_ids[min_idx];
            sorted_ids[min_idx] = tmp_id;
        }

        // Compare top 5 distances
        for (int p = 0; p < 5; p++)
        {
            double diff = fabs(n_dists[p] - all_dists[p]);
            assert(diff < 1e-5);
        }

        queries_verified++;
    } // while reading output

    fclose(fout);
    assert(queries_verified == 50);

    // 6. Pre-compute k-NN graph of Dataset A (k=10)
    printf("--- Pre-computing k-NN graph on Dataset A ---\n");
    snprintf(cmd, sizeof(cmd), "./gric-knn %s %s -k 10 -txt", TEST_A_TXT, TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // 7. Run graph-accelerated cross-dataset query search
    printf("--- Running graph-accelerated cross-dataset search ---\n");
    char graph_out_txt[256];
    snprintf(graph_out_txt, sizeof(graph_out_txt), "/tmp/test_cross_knn_graph_out.txt");
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -query %s -k 5 -o %s",
             TEST_A_TXT, TEST_CLUSTER_DIR, TEST_C_TXT, graph_out_txt);
    ret = system(cmd);
    assert(ret == 0);

    // 8. Verify graph-accelerated results match brute-force exactly
    FILE *f_graph = fopen(graph_out_txt, "r");
    assert(f_graph != NULL);

    queries_verified = 0;
    while (fgets(line, sizeof(line), f_graph) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        long q_id = -1;
        int n_ids[5];
        double n_dists[5];

        int scanned = sscanf(line, "%ld %d %lf %d %lf %d %lf %d %lf %d %lf",
                             &q_id,
                             &n_ids[0], &n_dists[0],
                             &n_ids[1], &n_dists[1],
                             &n_ids[2], &n_dists[2],
                             &n_ids[3], &n_dists[3],
                             &n_ids[4], &n_dists[4]);
        assert(scanned == 11);
        assert(q_id == queries_verified);

        // Compute brute-force distances for this query
        double all_dists[500];
        int sorted_ids[500];
        for (int j = 0; j < count_a; j++)
        {
            double dx = c_samples[q_id][0] - a_samples[j][0];
            double dy = c_samples[q_id][1] - a_samples[j][1];
            all_dists[j] = sqrt(dx * dx + dy * dy);
            sorted_ids[j] = j;
        }

        // Partial sort for top 5
        for (int p = 0; p < 5; p++)
        {
            int min_idx = p;
            for (int j = p + 1; j < count_a; j++)
            {
                if (all_dists[j] < all_dists[min_idx])
                {
                    min_idx = j;
                }
            }
            double tmp_d = all_dists[p];
            all_dists[p] = all_dists[min_idx];
            all_dists[min_idx] = tmp_d;

            int tmp_id = sorted_ids[p];
            sorted_ids[p] = sorted_ids[min_idx];
            sorted_ids[min_idx] = tmp_id;
        }

        // Compare top 5 distances
        for (int p = 0; p < 5; p++)
        {
            double diff = fabs(n_dists[p] - all_dists[p]);
            assert(diff < 1e-5);
        }

        queries_verified++;
    }

    fclose(f_graph);
    assert(queries_verified == 50);

    // 8. Run cross-dataset query search with -sq8
    printf("--- Running cross-dataset search with -sq8 ---\n");
    char sq8_out_txt[1024];
    snprintf(sq8_out_txt, sizeof(sq8_out_txt), "/tmp/test_cross_knn_sq8.txt");
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -query %s -k 5 -sq8 -o %s",
             TEST_A_TXT, TEST_CLUSTER_DIR, TEST_C_TXT, sq8_out_txt);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_sq8 = fopen(sq8_out_txt, "r");
    assert(f_sq8 != NULL);
    queries_verified = 0;
    while (fgets(line, sizeof(line), f_sq8) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        long q_id = -1;
        int n_ids[5];
        double n_dists[5];

        int scanned = sscanf(line, "%ld %d %lf %d %lf %d %lf %d %lf %d %lf",
                             &q_id,
                             &n_ids[0], &n_dists[0],
                             &n_ids[1], &n_dists[1],
                             &n_ids[2], &n_dists[2],
                             &n_ids[3], &n_dists[3],
                             &n_ids[4], &n_dists[4]);
        assert(scanned == 11);
        assert(q_id == queries_verified);

        double all_dists[500];
        int sorted_ids[500];
        for (int j = 0; j < count_a; j++)
        {
            double dx = c_samples[q_id][0] - a_samples[j][0];
            double dy = c_samples[q_id][1] - a_samples[j][1];
            all_dists[j] = sqrt(dx * dx + dy * dy);
            sorted_ids[j] = j;
        }

        for (int p = 0; p < 5; p++)
        {
            int min_idx = p;
            for (int j = p + 1; j < count_a; j++)
            {
                if (all_dists[j] < all_dists[min_idx])
                {
                    min_idx = j;
                }
            }
            double tmp_d = all_dists[p];
            all_dists[p] = all_dists[min_idx];
            all_dists[min_idx] = tmp_d;

            int tmp_id = sorted_ids[p];
            sorted_ids[p] = sorted_ids[min_idx];
            sorted_ids[min_idx] = tmp_id;
        }

        for (int p = 0; p < 5; p++)
        {
            double diff = fabs(n_dists[p] - all_dists[p]);
            assert(diff < 1e-5);
        }

        queries_verified++;
    }
    fclose(f_sq8);
    assert(queries_verified == 50);

    // 9. Outlier Query Refusal Test
    printf("--- Testing Out-of-Cluster Query Refusal ---\n");
    char outlier_txt[256];
    snprintf(outlier_txt, sizeof(outlier_txt), "/tmp/test_cross_outlier.txt");
    FILE *f_outlier = fopen(outlier_txt, "w");
    assert(f_outlier != NULL);
    // Query 0: In-cluster point (near A[0])
    fprintf(f_outlier, "%.8f %.8f\n", a_samples[0][0] + 0.001, a_samples[0][1] + 0.001);
    // Query 1: Far away outlier 1
    fprintf(f_outlier, "100.0 100.0\n");
    // Query 2: Far away outlier 2
    fprintf(f_outlier, "-80.0 -80.0\n");
    fclose(f_outlier);

    char outlier_out_txt[256];
    snprintf(outlier_out_txt, sizeof(outlier_out_txt), "/tmp/test_cross_outlier_out.txt");
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -query %s -k 5 -o %s",
             TEST_A_TXT, TEST_CLUSTER_DIR, outlier_txt, outlier_out_txt);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_out_res = fopen(outlier_out_txt, "r");
    assert(f_out_res != NULL);
    int outlier_q_count = 0;
    while (fgets(line, sizeof(line), f_out_res) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }
        long q_id = -1;
        int n_ids[5];
        double n_dists[5];
        int scanned = sscanf(line, "%ld %d %lf %d %lf %d %lf %d %lf %d %lf",
                             &q_id,
                             &n_ids[0], &n_dists[0],
                             &n_ids[1], &n_dists[1],
                             &n_ids[2], &n_dists[2],
                             &n_ids[3], &n_dists[3],
                             &n_ids[4], &n_dists[4]);
        assert(scanned == 11);
        if (q_id == 0)
        {
            // In-cluster query: must be accepted with valid indices >= 0
            assert(n_ids[0] >= 0);
            assert(n_dists[0] >= 0.0);
        }
        else
        {
            // Out-of-cluster queries: must be refused with -1 sentinels
            for (int p = 0; p < 5; p++)
            {
                assert(n_ids[p] == -1);
                assert(n_dists[p] == -1.0);
            }
        }
        outlier_q_count++;
    }
    fclose(f_out_res);
    assert(outlier_q_count == 3);

    // Validate binary outputs for query refusal
    char outlier_bin_idx[256];
    char outlier_bin_dst[256];
    snprintf(outlier_bin_idx, sizeof(outlier_bin_idx),
             "/tmp/test_cross_outlier_out_indices.bin");
    snprintf(outlier_bin_dst, sizeof(outlier_bin_dst),
             "/tmp/test_cross_outlier_out_distances.bin");

    FILE *f_bidx = fopen(outlier_bin_idx, "rb");
    assert(f_bidx != NULL);
    gric_bin_header_t hdr_i;
    char *comm_i = NULL;
    assert(gric_bin_read_header(f_bidx, &hdr_i, &comm_i) == 0);
    assert(hdr_i.dims[0] == 3 && hdr_i.dims[1] == 5);
    uint32_t b_indices[15];
    assert(fread(b_indices, sizeof(uint32_t), 15, f_bidx) == 15);
    fclose(f_bidx);
    if (comm_i != NULL)
    {
        free(comm_i);
    }

    FILE *f_bdst = fopen(outlier_bin_dst, "rb");
    assert(f_bdst != NULL);
    gric_bin_header_t hdr_d;
    char *comm_d = NULL;
    assert(gric_bin_read_header(f_bdst, &hdr_d, &comm_d) == 0);
    assert(hdr_d.dims[0] == 3 && hdr_d.dims[1] == 5);
    float b_distances[15];
    assert(fread(b_distances, sizeof(float), 15, f_bdst) == 15);
    fclose(f_bdst);
    if (comm_d != NULL)
    {
        free(comm_d);
    }

    for (int p = 0; p < 5; p++)
    {
        assert(b_indices[p] != UINT32_MAX);
        assert(b_distances[p] >= 0.0f);
    }
    for (int q = 1; q < 3; q++)
    {
        for (int p = 0; p < 5; p++)
        {
            assert(b_indices[q * 5 + p] == UINT32_MAX);
            assert(b_distances[q * 5 + p] == -1.0f);
        }
    }

    // 10. Test --all-queries flag to ensure refusal can be disabled
    printf("--- Testing --all-queries override flag ---\n");
    char all_q_out_txt[256];
    snprintf(all_q_out_txt, sizeof(all_q_out_txt), "/tmp/test_cross_all_queries_out.txt");
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -query %s -k 5 --all-queries -o %s",
             TEST_A_TXT, TEST_CLUSTER_DIR, outlier_txt, all_q_out_txt);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_all_q = fopen(all_q_out_txt, "r");
    assert(f_all_q != NULL);
    outlier_q_count = 0;
    while (fgets(line, sizeof(line), f_all_q) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }
        long q_id = -1;
        int n_ids[5];
        double n_dists[5];
        int scanned = sscanf(line, "%ld %d %lf %d %lf %d %lf %d %lf %d %lf",
                             &q_id,
                             &n_ids[0], &n_dists[0],
                             &n_ids[1], &n_dists[1],
                             &n_ids[2], &n_dists[2],
                             &n_ids[3], &n_dists[3],
                             &n_ids[4], &n_dists[4]);
        assert(scanned == 11);
        // With --all-queries, every query must be accepted
        assert(n_ids[0] >= 0);
        assert(n_dists[0] >= 0.0);
        outlier_q_count++;
    }
    fclose(f_all_q);
    assert(outlier_q_count == 3);

    remove(outlier_txt);
    remove(outlier_out_txt);
    remove(outlier_bin_idx);
    remove(outlier_bin_dst);
    remove(all_q_out_txt);

    // Cleanup temporary test files
    remove(TEST_A_TXT);
    remove(TEST_C_TXT);
    remove(TEST_KNN_OUT);
    remove(graph_out_txt);
    remove(sq8_out_txt);

    printf("=== All 50 Cross-Dataset k-NN queries verified successfully! ===\n");
    return 0;
}

/**
 * @file test_knn_cross_dataset.c
 * @brief Integration tests for cross-dataset k-NN search (gric-knn -query).
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    // Cleanup temporary test files
    remove(TEST_A_TXT);
    remove(TEST_C_TXT);
    remove(TEST_KNN_OUT);
    remove(graph_out_txt);

    printf("=== All 50 Cross-Dataset k-NN queries verified successfully! ===\n");
    return 0;
}

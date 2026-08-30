/**
 * @file test_dimdensity.c
 * @brief Integration tests for gric-dimdensity (MLE LID & Mack-Rosenblatt density).
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gric_bin_io.h"

#define TEST_SPIRAL_TXT   "/tmp/test_dimdens_spiral.txt"
#define TEST_CLUSTER_DIR  "/tmp/test_dimdens_cluster"
#define TEST_KNN_TXT      "/tmp/test_dimdens_knn.txt"
#define TEST_KNN_BIN      "/tmp/test_dimdens_cluster/knn_distances.bin"
#define TEST_DIMDENS_TXT  "/tmp/test_dimdens_out.txt"
#define TEST_DIMDENS_BIN  "/tmp/test_dimdens_out.bin"

int main(void)
{
    printf("=== Starting gric-dimdensity Integration Tests ===\n");

    char cmd[2048];

    // 1. Generate 2D synthetic spiral dataset (1D intrinsic manifold embedded in 2D)
    snprintf(cmd, sizeof(cmd), "./gric-mktxtseq 1000 %s 2Dspiral", TEST_SPIRAL_TXT);
    int ret = system(cmd);
    assert(ret == 0);

    // 2. Cluster dataset with gric-cluster
    snprintf(cmd, sizeof(cmd), "./gric-cluster 0.1 %s -maxim 1000 -outdir %s",
             TEST_SPIRAL_TXT, TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // 3. Solve k-NN with gric-knn (k = 20)
    snprintf(cmd, sizeof(cmd), "./gric-knn %s %s -k 20 -dtmin 1 -o %s",
             TEST_SPIRAL_TXT, TEST_CLUSTER_DIR, TEST_KNN_TXT);
    ret = system(cmd);
    assert(ret == 0);

    // 4. Run gric-dimdensity on the binary k-NN output matrix
    snprintf(cmd, sizeof(cmd),
             "./gric-dimdensity %s -k 15 -o %s",
             TEST_KNN_BIN, TEST_DIMDENS_TXT);
    ret = system(cmd);
    assert(ret == 0);

    // 5. Verify ASCII results output
    FILE *fp_txt = fopen(TEST_DIMDENS_TXT, "r");
    assert(fp_txt != NULL);
    char line[4096];
    int line_count = 0;
    double mean_dim = 0.0;
    double mean_dens = 0.0;

    while (fgets(line, sizeof(line), fp_txt) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        unsigned long id;
        double d, dens, log_dens, rk;
        int n_scanned = sscanf(line, "%lu %lf %lf %lf %lf", &id, &d, &dens, &log_dens, &rk);
        assert(n_scanned == 5);
        assert(!isnan(d) && !isinf(d));
        assert(!isnan(dens) && !isinf(dens));
        assert(d > 0.0);
        assert(dens > 0.0);
        assert(rk > 0.0);

        mean_dim += d;
        mean_dens += dens;
        line_count++;
    }
    fclose(fp_txt);

    assert(line_count == 1000);
    mean_dim /= (double)line_count;
    mean_dens /= (double)line_count;

    printf("  Verified 1000 sample records from %s\n", TEST_DIMDENS_TXT);
    printf("  Mean Intrinsic Dimension: %.4f (Expected ~1.0 for 1D spiral manifold)\n", mean_dim);
    // For a 1D spiral manifold, estimated intrinsic dimension should be close to 1.0
    assert(mean_dim >= 0.70 && mean_dim <= 1.45);

    // 6. Verify Binary results output
    FILE *fp_bin = fopen(TEST_DIMDENS_BIN, "rb");
    assert(fp_bin != NULL);
    gric_bin_header_t hdr;
    char *comment = NULL;
    assert(gric_bin_read_header(fp_bin, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_GENERIC);
    assert(hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
    assert(hdr.ndim == 2);
    assert(hdr.dims[0] == 1000);
    assert(hdr.dims[1] == 4);
    fclose(fp_bin);
    if (comment != NULL)
    {
        free(comment);
    }
    printf("  Verified binary matrix %s [1000 x 4]\n", TEST_DIMDENS_BIN);

    // 7. Test JSON output mode
    snprintf(cmd, sizeof(cmd),
             "./gric-dimdensity %s -k 15 -json > /tmp/test_dimdens.json",
             TEST_KNN_BIN);
    ret = system(cmd);
    assert(ret == 0);

    FILE *fp_json = fopen("/tmp/test_dimdens.json", "r");
    assert(fp_json != NULL);
    size_t json_bytes = fread(line, 1, sizeof(line) - 1, fp_json);
    line[json_bytes] = '\0';
    fclose(fp_json);
    assert(strstr(line, "\"intrinsic_dimension\"") != NULL);
    assert(strstr(line, "\"local_density\"") != NULL);
    assert(strstr(line, "\"percentiles\"") != NULL);
    printf("  Verified JSON report formatting\n");

    // 8. Test multi-scale range smoothing and Epanechnikov kernel
    snprintf(cmd, sizeof(cmd),
             "./gric-dimdensity %s -kmin 5 -kmax 18 -range -kernel epanechnikov -o %s",
             TEST_KNN_BIN, TEST_DIMDENS_TXT);
    ret = system(cmd);
    assert(ret == 0);

    // 9. Test directory ingestion fallback
    snprintf(cmd, sizeof(cmd),
             "./gric-dimdensity %s -k 10 -o %s",
             TEST_CLUSTER_DIR, TEST_DIMDENS_TXT);
    ret = system(cmd);
    assert(ret == 0);

    // Cleanup temporary files
    remove(TEST_SPIRAL_TXT);
    remove(TEST_KNN_TXT);
    remove(TEST_KNN_BIN);
    remove(TEST_DIMDENS_TXT);
    remove(TEST_DIMDENS_BIN);
    remove("/tmp/test_dimdens.json");

    printf("=== All gric-dimdensity Integration Tests Passed Successfully! ===\n");
    return 0;
}

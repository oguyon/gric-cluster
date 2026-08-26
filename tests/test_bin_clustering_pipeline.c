/**
 * @file test_bin_clustering_pipeline.c
 * @brief Integration tests for binary dataset input, binary cluster outputs, and KNN loader.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "gric_bin_io.h"

#define TEST_SPIRAL_TXT "/tmp/test_pipeline_spiral.txt"
#define TEST_SPIRAL_BIN "/tmp/test_pipeline_spiral.bin"
#define TEST_CLUSTER_DIR "/tmp/test_pipeline_clusterdat"

int main(void)
{
    printf("=== Starting End-to-End Binary Clustering Pipeline Tests ===\n");

    // 1. Generate 2D synthetic dataset using gric-mktxtseq
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "./gric-mktxtseq 500 %s 2Dspiral", TEST_SPIRAL_TXT);
    int ret = system(cmd);
    assert(ret == 0);

    // 2. Convert ASCII dataset to .bin using gric-ascii2bin
    snprintf(cmd, sizeof(cmd), "./gric-ascii2bin %s %s -type coords",
             TEST_SPIRAL_TXT, TEST_SPIRAL_BIN);
    ret = system(cmd);
    assert(ret == 0);

    // Verify .bin header
    FILE *f_bin = fopen(TEST_SPIRAL_BIN, "rb");
    assert(f_bin != NULL);
    gric_bin_header_t hdr;
    char *comment = NULL;
    assert(gric_bin_read_header(f_bin, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_COORDINATES);
    assert(hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
    assert(hdr.dims[0] == 500);
    assert(hdr.dims[1] == 2);
    fclose(f_bin);
    if (comment != NULL)
    {
        free(comment);
    }

    // 3. Run gric-cluster directly on binary input file
    snprintf(cmd, sizeof(cmd),
             "./gric-cluster 0.15 %s -outdir %s",
             TEST_SPIRAL_BIN, TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // 4. Verify binary output files were generated in clusterdir
    char bin_path[1024];

    // 4a. anchors.bin
    snprintf(bin_path, sizeof(bin_path), "%s/anchors.bin", TEST_CLUSTER_DIR);
    FILE *f_anchors = fopen(bin_path, "rb");
    assert(f_anchors != NULL);
    assert(gric_bin_read_header(f_anchors, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_ANCHORS);
    assert(hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
    assert(hdr.ndim == 2);
    assert(hdr.dims[1] == 2);
    uint64_t num_clusters = hdr.dims[0];
    assert(num_clusters > 0);
    fclose(f_anchors);
    if (comment != NULL)
    {
        free(comment);
    }

    // 4b. dcc.bin
    snprintf(bin_path, sizeof(bin_path), "%s/dcc.bin", TEST_CLUSTER_DIR);
    FILE *f_dcc = fopen(bin_path, "rb");
    assert(f_dcc != NULL);
    assert(gric_bin_read_header(f_dcc, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_DCC);
    assert(hdr.dims[0] == num_clusters);
    assert(hdr.dims[1] == num_clusters);
    fclose(f_dcc);
    if (comment != NULL)
    {
        free(comment);
    }

    // 4c. frame_membership.bin
    snprintf(bin_path, sizeof(bin_path), "%s/frame_membership.bin", TEST_CLUSTER_DIR);
    FILE *f_memb = fopen(bin_path, "rb");
    assert(f_memb != NULL);
    assert(gric_bin_read_header(f_memb, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_MEMBERSHIP);
    assert(hdr.data_type == GRIC_BIN_DTYPE_UINT32);
    assert(hdr.dims[0] == 500);
    fclose(f_memb);
    if (comment != NULL)
    {
        free(comment);
    }

    // 4d. cluster_counts.bin
    snprintf(bin_path, sizeof(bin_path), "%s/cluster_counts.bin", TEST_CLUSTER_DIR);
    FILE *f_cnt = fopen(bin_path, "rb");
    assert(f_cnt != NULL);
    assert(gric_bin_read_header(f_cnt, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_COUNTS);
    assert(hdr.dims[0] == num_clusters);
    fclose(f_cnt);
    if (comment != NULL)
    {
        free(comment);
    }

    // 5. Test gric-bin2ascii -info inspection tool on outputs
    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s/anchors.bin -info", TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s/dcc.bin -info", TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // 6. Test gric-knn loading binary artifacts directly
    snprintf(cmd, sizeof(cmd),
             "./gric-knn %s %s -k 5 -o /tmp/test_pipeline_knn.txt",
             TEST_SPIRAL_BIN, TEST_CLUSTER_DIR);
    ret = system(cmd);
    assert(ret == 0);

    // Cleanup temporary files
    remove(TEST_SPIRAL_TXT);
    remove(TEST_SPIRAL_BIN);
    remove("/tmp/test_pipeline_knn.txt");

    printf("=== All End-to-End Binary Clustering Pipeline Tests Passed! ===\n");
    return 0;
}

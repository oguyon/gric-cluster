/**
 * @file test_bin_ascii_roundtrip.c
 * @brief Integration tests for gric-ascii2bin and gric-bin2ascii conversion fidelity.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "gric_bin_io.h"

#define TMP_TXT_IN  "/tmp/test_gric_roundtrip_in.txt"
#define TMP_BIN_OUT "/tmp/test_gric_roundtrip.bin"
#define TMP_TXT_OUT "/tmp/test_gric_roundtrip_out.txt"

/**
 * test_coordinates_roundtrip() - Test encoding and decoding 2D coordinate stream.
 */
static void test_coordinates_roundtrip(void)
{
    printf("[TEST] Testing 2D coordinates ASCII -> BIN (float32) -> ASCII...\n");

    const int npts = 100;
    FILE *f_in = fopen(TMP_TXT_IN, "w");
    assert(f_in != NULL);

    fprintf(f_in, "# Test 2D Spiral points\n");
    for (int i = 0; i < npts; i++)
    {
        double t = i * 0.1;
        double x = t * cos(t);
        double y = t * sin(t);
        fprintf(f_in, "%.6f %.6f\n", x, y);
    }
    fclose(f_in);

    // Encode to binary using command or direct library
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "./gric-ascii2bin %s %s -type coords", TMP_TXT_IN, TMP_BIN_OUT);
    int ret = system(cmd);
    assert(ret == 0);

    // Validate binary header
    FILE *f_bin = fopen(TMP_BIN_OUT, "rb");
    assert(f_bin != NULL);
    gric_bin_header_t hdr;
    char *comment = NULL;
    assert(gric_bin_read_header(f_bin, &hdr, &comment) == 0);
    assert(hdr.file_type == GRIC_BIN_TYPE_COORDINATES);
    assert(hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
    assert(hdr.ndim == 2);
    assert(hdr.dims[0] == (uint64_t)npts);
    assert(hdr.dims[1] == 2);
    assert(hdr.num_elements == (uint64_t)(npts * 2));
    fclose(f_bin);
    if (comment != NULL) free(comment);

    // Decode back to ASCII
    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s", TMP_BIN_OUT, TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    // Verify values
    FILE *f_out = fopen(TMP_TXT_OUT, "r");
    assert(f_out != NULL);
    for (int i = 0; i < npts; i++)
    {
        double t = i * 0.1;
        double expected_x = (float)(t * cos(t));
        double expected_y = (float)(t * sin(t));
        double read_x = 0, read_y = 0;
        assert(fscanf(f_out, "%lf %lf", &read_x, &read_y) == 2);
        assert(fabs(read_x - expected_x) < 1e-4);
        assert(fabs(read_y - expected_y) < 1e-4);
    }
    fclose(f_out);

    remove(TMP_TXT_IN);
    remove(TMP_BIN_OUT);
    remove(TMP_TXT_OUT);
    printf("  [PASS] 2D coordinates roundtrip verified.\n");
}

/**
 * test_dcc_matrix_roundtrip() - Test encoding and decoding distance matrix (float64).
 */
static void test_dcc_matrix_roundtrip(void)
{
    printf("[TEST] Testing DCC distance matrix ASCII -> BIN (float64) -> ASCII...\n");

    const int k = 15;
    FILE *f_in = fopen(TMP_TXT_IN, "w");
    assert(f_in != NULL);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double dist = (i == j) ? 0.0 : (double)(abs(i - j) * 1.25);
            fprintf(f_in, "%.8f%s", dist, (j + 1 < k) ? " " : "\n");
        }
    }
    fclose(f_in);

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "./gric-ascii2bin %s %s -type dcc -double", TMP_TXT_IN, TMP_BIN_OUT);
    int ret = system(cmd);
    assert(ret == 0);

    // Decode back to ASCII
    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s -fmt %%.8f", TMP_BIN_OUT, TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_out = fopen(TMP_TXT_OUT, "r");
    assert(f_out != NULL);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double expected = (i == j) ? 0.0 : (double)(abs(i - j) * 1.25);
            double actual = 0.0;
            assert(fscanf(f_out, "%lf", &actual) == 1);
            assert(fabs(actual - expected) < 1e-7);
        }
    }
    fclose(f_out);

    remove(TMP_TXT_IN);
    remove(TMP_BIN_OUT);
    remove(TMP_TXT_OUT);
    printf("  [PASS] DCC distance matrix roundtrip verified.\n");
}

/**
 * test_membership_roundtrip() - Test encoding and decoding integer membership list.
 */
static void test_membership_roundtrip(void)
{
    printf("[TEST] Testing frame membership ASCII -> BIN (uint32) -> ASCII...\n");

    const int nframes = 50;
    FILE *f_in = fopen(TMP_TXT_IN, "w");
    assert(f_in != NULL);

    for (int i = 0; i < nframes; i++)
    {
        uint32_t cluster_id = (uint32_t)(i % 7);
        fprintf(f_in, "%u\n", cluster_id);
    }
    fclose(f_in);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "./gric-ascii2bin %s %s -type membership -uint32",
             TMP_TXT_IN, TMP_BIN_OUT);
    int ret = system(cmd);
    assert(ret == 0);

    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s", TMP_BIN_OUT, TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_out = fopen(TMP_TXT_OUT, "r");
    assert(f_out != NULL);
    for (int i = 0; i < nframes; i++)
    {
        uint32_t expected = (uint32_t)(i % 7);
        uint32_t actual = 0;
        assert(fscanf(f_out, "%u", &actual) == 1);
        assert(actual == expected);
    }
    fclose(f_out);

    remove(TMP_TXT_IN);
    remove(TMP_BIN_OUT);
    remove(TMP_TXT_OUT);
    printf("  [PASS] Frame membership roundtrip verified.\n");
}

/**
 * test_bin2ascii_header_option() - Test decoding with explanatory header comments.
 */
static void test_bin2ascii_header_option(void)
{
    printf("[TEST] Testing gric-bin2ascii with -header option...\n");

    FILE *f_in = fopen(TMP_TXT_IN, "w");
    assert(f_in != NULL);
    fprintf(f_in, "1.0 2.0\n");
    fprintf(f_in, "3.0 4.0\n");
    fprintf(f_in, "5.0 6.0\n");
    fclose(f_in);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "./gric-ascii2bin %s %s -type anchors -comment \"Cluster centroids\"",
             TMP_TXT_IN, TMP_BIN_OUT);
    int ret = system(cmd);
    assert(ret == 0);

    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s -header", TMP_BIN_OUT, TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_out = fopen(TMP_TXT_OUT, "r");
    assert(f_out != NULL);
    char line[256];
    int found_header_delim = 0;
    int found_content_desc = 0;
    int found_anchor_idx_col = 0;
    int data_rows = 0;

    while (fgets(line, sizeof(line), f_out) != NULL)
    {
        if (strstr(line, "# ========================================================") != NULL)
        {
            found_header_delim++;
        }
        if (strstr(line, "# Content     : Cluster centroids") != NULL)
        {
            found_content_desc = 1;
        }
        if (strstr(line, "# Columns     : anchor_idx dim_0 .. dim_1") != NULL)
        {
            found_anchor_idx_col = 1;
        }
        if (line[0] != '#' && strlen(line) > 1)
        {
            int idx = -1;
            float x = 0.0f, y = 0.0f;
            assert(sscanf(line, "%d %f %f", &idx, &x, &y) == 3);
            assert(idx == data_rows);
            if (data_rows == 0)
            {
                assert(fabsf(x - 1.0f) < 1e-5f && fabsf(y - 2.0f) < 1e-5f);
            }
            else if (data_rows == 1)
            {
                assert(fabsf(x - 3.0f) < 1e-5f && fabsf(y - 4.0f) < 1e-5f);
            }
            else if (data_rows == 2)
            {
                assert(fabsf(x - 5.0f) < 1e-5f && fabsf(y - 6.0f) < 1e-5f);
            }
            data_rows++;
        }
    }
    fclose(f_out);

    assert(found_header_delim >= 2);
    assert(found_content_desc == 1);
    assert(found_anchor_idx_col == 1);
    assert(data_rows == 3);

    // Verify gric-ascii2bin can auto-detect anchor_idx from comments and recreate bin
    snprintf(cmd, sizeof(cmd),
             "./gric-ascii2bin %s /tmp/test_rt_header.bin -type anchors",
             TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    FILE *f_rt = fopen("/tmp/test_rt_header.bin", "rb");
    assert(f_rt != NULL);
    gric_bin_header_t rt_hdr;
    char *rt_comment = NULL;
    assert(gric_bin_read_header(f_rt, &rt_hdr, &rt_comment) == 0);
    assert(rt_hdr.file_type == GRIC_BIN_TYPE_ANCHORS);
    assert(rt_hdr.ndim == 2);
    assert(rt_hdr.dims[0] == 3);
    assert(rt_hdr.dims[1] == 2);
    assert(rt_hdr.num_elements == 6);
    float rt_payload[6];
    assert(fread(rt_payload, sizeof(float), 6, f_rt) == 6);
    assert(fabsf(rt_payload[0] - 1.0f) < 1e-5f);
    assert(fabsf(rt_payload[1] - 2.0f) < 1e-5f);
    assert(fabsf(rt_payload[2] - 3.0f) < 1e-5f);
    assert(fabsf(rt_payload[3] - 4.0f) < 1e-5f);
    assert(fabsf(rt_payload[4] - 5.0f) < 1e-5f);
    assert(fabsf(rt_payload[5] - 6.0f) < 1e-5f);
    fclose(f_rt);
    if (rt_comment != NULL) free(rt_comment);

    // Also verify decoding with -no-index omits index
    snprintf(cmd, sizeof(cmd),
             "./gric-bin2ascii /tmp/test_rt_header.bin %s -no-index",
             TMP_TXT_OUT);
    ret = system(cmd);
    assert(ret == 0);

    f_out = fopen(TMP_TXT_OUT, "r");
    assert(f_out != NULL);
    float no_idx_x = 0.0f, no_idx_y = 0.0f;
    assert(fscanf(f_out, "%f %f", &no_idx_x, &no_idx_y) == 2);
    assert(fabsf(no_idx_x - 1.0f) < 1e-5f && fabsf(no_idx_y - 2.0f) < 1e-5f);
    fclose(f_out);

    remove(TMP_TXT_IN);
    remove(TMP_BIN_OUT);
    remove(TMP_TXT_OUT);
    remove("/tmp/test_rt_header.bin");
    printf("  [PASS] Decoded ASCII header generation, anchor_idx, and roundtrip verified.\n");
}

/**
 * test_radii_index_roundtrip() - Test cluster radii indexing and auto-detection roundtrip.
 */
static void test_radii_index_roundtrip(void)
{
    printf("[TEST] Testing cluster_radii indexing, header, and roundtrip...\n");

    const char *bin_path = "/tmp/test_cluster_radii.bin";
    const char *txt_path = "/tmp/test_cluster_radii.txt";
    const char *rt_bin_path = "/tmp/test_cluster_radii_rt.bin";

    FILE *f_bin = fopen(bin_path, "wb");
    assert(f_bin != NULL);

    gric_bin_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr.ndim = 1;
    hdr.dims[0] = 4;
    hdr.num_elements = 4;
    hdr.data_bytes = 4 * sizeof(float);

    assert(gric_bin_write_header(f_bin, &hdr, "Cluster max radii") == 0);
    float radii[4] = {0.25f, 0.35f, 0.45f, 0.55f};
    assert(fwrite(radii, sizeof(float), 4, f_bin) == 4);
    fclose(f_bin);

    // Decode with -header (should auto-enable cluster_idx)
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s -header", bin_path, txt_path);
    assert(system(cmd) == 0);

    FILE *f_txt = fopen(txt_path, "r");
    assert(f_txt != NULL);
    char line[256];
    int found_header_col = 0;
    int data_rows = 0;

    while (fgets(line, sizeof(line), f_txt) != NULL)
    {
        if (strstr(line, "# Columns     : cluster_idx radius") != NULL)
        {
            found_header_col = 1;
        }
        if (line[0] != '#' && strlen(line) > 1)
        {
            int idx = -1;
            float r = 0.0f;
            assert(sscanf(line, "%d %f", &idx, &r) == 2);
            assert(idx == data_rows);
            assert(fabsf(r - radii[data_rows]) < 1e-5f);
            data_rows++;
        }
    }
    fclose(f_txt);

    assert(found_header_col == 1);
    assert(data_rows == 4);

    // Roundtrip back to binary (auto-detecting cluster_idx)
    snprintf(cmd, sizeof(cmd), "./gric-ascii2bin %s %s", txt_path, rt_bin_path);
    assert(system(cmd) == 0);

    FILE *f_rt = fopen(rt_bin_path, "rb");
    assert(f_rt != NULL);
    gric_bin_header_t rt_hdr;
    char *rt_comment = NULL;
    assert(gric_bin_read_header(f_rt, &rt_hdr, &rt_comment) == 0);
    assert(rt_hdr.ndim == 1);
    assert(rt_hdr.dims[0] == 4);
    assert(rt_hdr.num_elements == 4);
    float rt_payload[4];
    assert(fread(rt_payload, sizeof(float), 4, f_rt) == 4);
    for (int i = 0; i < 4; i++)
    {
        assert(fabsf(rt_payload[i] - radii[i]) < 1e-5f);
    }
    fclose(f_rt);
    if (rt_comment != NULL) free(rt_comment);

    // Verify -no-index omits index
    snprintf(cmd, sizeof(cmd), "./gric-bin2ascii %s %s -no-index", bin_path, txt_path);
    assert(system(cmd) == 0);

    f_txt = fopen(txt_path, "r");
    assert(f_txt != NULL);
    float no_idx_r = 0.0f;
    assert(fscanf(f_txt, "%f", &no_idx_r) == 1);
    assert(fabsf(no_idx_r - radii[0]) < 1e-5f);
    fclose(f_txt);

    remove(bin_path);
    remove(txt_path);
    remove(rt_bin_path);
    printf("  [PASS] Cluster radii indexing and roundtrip verified.\n");
}

int main(void)
{
    printf("Running GRIC Binary <-> ASCII Roundtrip Tests...\n");
    test_coordinates_roundtrip();
    test_dcc_matrix_roundtrip();
    test_membership_roundtrip();
    test_bin2ascii_header_option();
    test_radii_index_roundtrip();
    printf("All GRIC Binary <-> ASCII tests passed successfully!\n");
    return 0;
}

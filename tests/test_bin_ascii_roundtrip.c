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

int main(void)
{
    printf("Running GRIC Binary <-> ASCII Roundtrip Tests...\n");
    test_coordinates_roundtrip();
    test_dcc_matrix_roundtrip();
    test_membership_roundtrip();
    printf("All GRIC Binary <-> ASCII tests passed successfully!\n");
    return 0;
}

/**
 * @file test_product_quant.c
 * @brief Unit tests for Product Quantization and FastScan SIMD operations.
 */

#include "product_quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_pq_alloc_free(void)
{
    printf("[TEST] Testing PQ codebook allocation and deallocation...\n");

    PQCodebook *cb = pq_codebook_alloc(128, 32, 16);
    assert(cb != NULL);
    assert(cb->dim == 128);
    assert(cb->m == 32);
    assert(cb->d_sub == 4);
    assert(cb->k_centroids == 16);
    assert(cb->is_4bit == 1);
    assert(cb->centroids != NULL);
    assert(cb->sub_radius != NULL);

    pq_codebook_free(cb);

    /* Invalid arguments */
    assert(pq_codebook_alloc(0, 16, 16) == NULL);
    assert(pq_codebook_alloc(128, 0, 16) == NULL);
    assert(pq_codebook_alloc(128, 30, 16) == NULL); /* 128 % 30 != 0 */
    assert(pq_codebook_alloc(128, 16, 32) == NULL); /* K must be 16 or 256 */

    printf("[PASS] PQ codebook alloc/free verified.\n");
}

static void test_pq_training_and_quantization(void)
{
    printf("[TEST] Testing PQ training and quantization...\n");

    long dim = 64;
    int m = 16;
    int K = 16;
    long num_frames = 200;

    PQCodebook *cb = pq_codebook_alloc(dim, m, K);
    assert(cb != NULL);

    float *train_data = (float *)malloc((size_t)num_frames * (size_t)dim * sizeof(float));
    assert(train_data != NULL);

    for (long i = 0; i < num_frames * dim; i++)
    {
        train_data[i] = (float)sin((double)i * 0.05);
    } // for (long i = 0; i < num_frames * dim; i++)

    int ret = pq_train_codebook(cb, train_data, num_frames, 10);
    assert(ret == 0);
    assert(cb->total_radius >= 0.0f);

    /* Test quantization of a frame */
    uint8_t codes[16];
    pq_quantize_frame_float(train_data, codes, cb);

    for (int s = 0; s < m; s++)
    {
        assert(codes[s] < K);
    } // for (int s = 0; s < m; s++)

    /* Test double frame quantization */
    double *double_frame = (double *)malloc((size_t)dim * sizeof(double));
    assert(double_frame != NULL);
    for (long j = 0; j < dim; j++)
    {
        double_frame[j] = (double)train_data[j];
    } // for (long j = 0; j < dim; j++)

    uint8_t d_codes[16];
    pq_quantize_frame_double(double_frame, d_codes, cb);
    for (int s = 0; s < m; s++)
    {
        assert(d_codes[s] == codes[s]);
    } // for (int s = 0; s < m; s++)

    free(double_frame);
    free(train_data);
    pq_codebook_free(cb);

    printf("[PASS] PQ training and quantization verified.\n");
}

static void test_pq_fastscan_simd_vs_scalar(void)
{
    printf("[TEST] Testing FastScan SIMD kernel against scalar reference (%s)...\n",
           pq_get_simd_mode_str());

    long dim = 128;
    int m = 32;
    int K = 16;
    PQCodebook *cb = pq_codebook_alloc(dim, m, K);
    assert(cb != NULL);

    long num_train = 100;
    float *train_data = (float *)malloc((size_t)num_train * (size_t)dim * sizeof(float));
    assert(train_data != NULL);

    for (long i = 0; i < num_train * dim; i++)
    {
        train_data[i] = (float)((i % 37) - 18) * 0.1f;
    } // for (long i = 0; i < num_train * dim; i++)

    pq_train_codebook(cb, train_data, num_train, 10);

    /* Generate 32 candidate vectors */
    uint8_t *raw_codes = (uint8_t *)malloc(32 * (size_t)m * sizeof(uint8_t));
    assert(raw_codes != NULL);

    for (int i = 0; i < 32; i++)
    {
        float *vec = train_data + (i % num_train) * dim;
        pq_quantize_frame_float(vec, raw_codes + i * m, cb);
    } // for (int i = 0; i < 32; i++)

    /* Transpose into 32-vector FastScan layout */
    uint8_t *block = (uint8_t *)malloc((size_t)m * 32 * sizeof(uint8_t));
    assert(block != NULL);
    pq_transpose_block_codes(raw_codes, block, m, 32);

    /* Build query LUT */
    float *query = (float *)malloc((size_t)dim * sizeof(float));
    assert(query != NULL);
    for (long j = 0; j < dim; j++)
    {
        query[j] = 0.5f;
    } // for (long j = 0; j < dim; j++)

    PQLookupTable lut;
    lut.lut_u8 = (uint8_t *)malloc((size_t)m * (size_t)K * sizeof(uint8_t));
    assert(lut.lut_u8 != NULL);
    pq_build_query_lut_float(query, cb, &lut, 0.0);

    /* Compare FastScan SIMD vs Scalar across all 256 possible cutoffs */
    for (int cutoff = 0; cutoff < 256; cutoff++)
    {
        uint32_t mask_scalar = pq_fastscan_32x_scalar(lut.lut_u8, block, m, (uint8_t)cutoff);
        uint32_t mask_simd   = pq_fastscan_32x(lut.lut_u8, block, m, (uint8_t)cutoff);
        if (mask_scalar != mask_simd)
        {
            fprintf(stderr,
                    "Mismatch at cutoff %d: scalar=0x%08X simd=0x%08X\n",
                    cutoff, mask_scalar, mask_simd);
            assert(mask_scalar == mask_simd);
        }
    } // for (int cutoff = 0; cutoff < 256; cutoff++)

    free(lut.lut_u8);
    free(query);
    free(block);
    free(raw_codes);
    free(train_data);
    pq_codebook_free(cb);

    printf("[PASS] FastScan SIMD matches scalar reference for all 256 cutoffs.\n");
}

static void test_pq_sidecar_roundtrip(void)
{
    printf("[TEST] Testing PQ sidecar save and load roundtrip...\n");

    long dim = 32;
    int m = 8;
    int K = 16;
    long num_frames = 64;

    PQCodebook *cb = pq_codebook_alloc(dim, m, K);
    assert(cb != NULL);

    float *train_data = (float *)malloc((size_t)num_frames * (size_t)dim * sizeof(float));
    assert(train_data != NULL);
    for (long i = 0; i < num_frames * dim; i++)
    {
        train_data[i] = (float)(i % 13) * 0.25f;
    } // for (long i = 0; i < num_frames * dim; i++)

    pq_train_codebook(cb, train_data, num_frames, 5);

    /* Quantize all frames */
    long num_blocks = (num_frames + 31) / 32;
    uint8_t *transposed = (uint8_t *)malloc((size_t)num_blocks * (size_t)m * 32);
    assert(transposed != NULL);

    uint8_t *raw_codes = (uint8_t *)malloc((size_t)num_frames * (size_t)m);
    assert(raw_codes != NULL);

    for (long i = 0; i < num_frames; i++)
    {
        pq_quantize_frame_float(train_data + i * dim, raw_codes + i * m, cb);
    } // for (long i = 0; i < num_frames; i++)

    for (long b = 0; b < num_blocks; b++)
    {
        int valid = (int)(num_frames - b * 32);
        if (valid > 32)
        {
            valid = 32;
        }
        pq_transpose_block_codes(
            raw_codes + b * 32 * m,
            transposed + b * m * 32,
            m,
            valid
        );
    } // for (long b = 0; b < num_blocks; b++)

    const char *sidecar_path = "/tmp/test_gric_pq.pq";
    int save_ret = pq_save_sidecar(sidecar_path, cb, transposed, num_frames);
    assert(save_ret == 0);

    PQCodebook *loaded_cb = NULL;
    uint8_t *loaded_trans = NULL;
    long loaded_frames = 0;
    int load_ret = pq_load_sidecar(sidecar_path, &loaded_cb, &loaded_trans, &loaded_frames);
    assert(load_ret == 0);
    assert(loaded_frames == num_frames);
    assert(loaded_cb->dim == dim);
    assert(loaded_cb->m == m);
    assert(loaded_cb->k_centroids == K);

    size_t c_size = (size_t)m * (size_t)K * (size_t)cb->d_sub;
    assert(memcmp(cb->centroids, loaded_cb->centroids, c_size * sizeof(float)) == 0);
    assert(memcmp(transposed, loaded_trans, (size_t)num_blocks * (size_t)m * 32) == 0);

    remove(sidecar_path);

    free(raw_codes);
    free(transposed);
    free(loaded_trans);
    pq_codebook_free(cb);
    pq_codebook_free(loaded_cb);
    free(train_data);

    printf("[PASS] PQ sidecar save and load roundtrip verified.\n");
}

int main(void)
{
    printf("==================================================\n");
    printf("  RUNNING PRODUCT QUANTIZATION (PQ / ADC) TESTS   \n");
    printf("==================================================\n");

    test_pq_alloc_free();
    test_pq_training_and_quantization();
    test_pq_fastscan_simd_vs_scalar();
    test_pq_sidecar_roundtrip();

    printf("==================================================\n");
    printf("  ALL PRODUCT QUANTIZATION TESTS PASSED!          \n");
    printf("==================================================\n");
    return 0;
}

/**
 * @file test_scalar_quant.c
 * @brief Unit tests for 8-bit scalar quantization and lower-bound invariance.
 */

#include "scalar_quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double reference_euclidean_distance(
    const float *a,
    const float *b,
    long         dim)
{
    double sum = 0.0;
    for (long i = 0; i < dim; i++)
    {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static uint64_t reference_ssd_u8(
    const uint8_t *a,
    const uint8_t *b,
    long           dim)
{
    uint64_t sum = 0;
    for (long i = 0; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint64_t)(diff * diff);
    }
    return sum;
}

static void test_initialization_and_clamping()
{
    printf("[TEST] Testing initialization and clamping...\n");
    SQ8Params params;
    sq8_init_params(&params, -10.0f, 10.0f, 128);

    assert(fabsf(params.min_val - (-10.0f)) < 1e-5f);
    assert(fabsf(params.max_val - 10.0f) < 1e-5f);
    assert(fabsf(params.scale - (20.0f / 255.0f)) < 1e-5f);
    assert(params.dim == 128);

    float src[4] = {-15.0f, -10.0f, 10.0f, 25.0f};
    uint8_t dst[4];
    params.dim = 4;
    sq8_quantize_float(src, dst, &params);

    assert(dst[0] == 0);   // Clamped below min
    assert(dst[1] == 0);   // Min
    assert(dst[2] == 255); // Max
    assert(dst[3] == 255); // Clamped above max

    printf("  -> Initialization and clamping passed.\n");
}

static void test_simd_bit_exactness()
{
    printf("[TEST] Testing SIMD vs scalar bit-exactness...\n");
    long test_dims[] = {1, 7, 16, 31, 32, 33, 63, 64, 65, 128, 255, 1024, 4097};
    int num_dims = sizeof(test_dims) / sizeof(test_dims[0]);

    for (int d = 0; d < num_dims; d++)
    {
        long dim = test_dims[d];
        uint8_t *a = (uint8_t *)malloc((size_t)dim);
        uint8_t *b = (uint8_t *)malloc((size_t)dim);

        for (long i = 0; i < dim; i++)
        {
            a[i] = (uint8_t)(rand() % 256);
            b[i] = (uint8_t)(rand() % 256);
        }

        uint64_t ref = reference_ssd_u8(a, b, dim);
        uint64_t simd = sq8_dist_squared_u8(a, b, dim);

        assert(ref == simd);

        free(a);
        free(b);
    }

    printf("  -> SIMD bit-exactness passed across all tested dimensions.\n");
}

static void test_metric_lower_bound_invariance()
{
    printf("[TEST] Testing metric lower-bound invariance (100,000 random vector pairs)...\n");
    long test_dims[] = {2, 8, 32, 128, 512, 1024};
    int num_dims = sizeof(test_dims) / sizeof(test_dims[0]);
    int pairs_per_dim = 20000;

    for (int d = 0; d < num_dims; d++)
    {
        long dim = test_dims[d];
        SQ8Params params;
        sq8_init_params(&params, -50.0f, 150.0f, dim);

        float *fa = (float *)malloc((size_t)dim * sizeof(float));
        float *fb = (float *)malloc((size_t)dim * sizeof(float));
        uint8_t *qa = (uint8_t *)malloc((size_t)dim);
        uint8_t *qb = (uint8_t *)malloc((size_t)dim);

        for (int p = 0; p < pairs_per_dim; p++)
        {
            for (long i = 0; i < dim; i++)
            {
                fa[i] = -50.0f + (float)rand() / (float)(RAND_MAX / 200.0f);
                fb[i] = -50.0f + (float)rand() / (float)(RAND_MAX / 200.0f);
            }

            sq8_quantize_float(fa, qa, &params);
            sq8_quantize_float(fb, qb, &params);

            double d_exact = reference_euclidean_distance(fa, fb, dim);
            double d_lb = sq8_compute_lower_bound(qa, qb, &params, 0.0);

            // Lower bound MUST be <= exact distance (allowing tiny float epsilon)
            if (d_lb > d_exact + 1e-6)
            {
                fprintf(stderr, "VIOLATION at dim %ld, pair %d: d_lb=%.6f > d_exact=%.6f\n",
                        dim, p, d_lb, d_exact);
                assert(d_lb <= d_exact + 1e-6);
            }
        }

        free(fa);
        free(fb);
        free(qa);
        free(qb);
    }

    printf("  -> Metric lower-bound invariance passed: zero violations observed.\n");
}

static void test_sidecar_roundtrip()
{
    printf("[TEST] Testing .sq8 sidecar file save and load round-trip...\n");
    const char *tmp_path = "/tmp/test_sq8_sidecar.sq8";
    long dim = 256;
    long num_frames = 10;

    SQ8Params p_save;
    sq8_init_params(&p_save, -12.5f, 87.5f, dim);

    size_t total_bytes = (size_t)dim * (size_t)num_frames;
    uint8_t *data_save = (uint8_t *)malloc(total_bytes);
    for (size_t i = 0; i < total_bytes; i++)
    {
        data_save[i] = (uint8_t)(i % 256);
    }

    int rc = sq8_save_sidecar(tmp_path, &p_save, data_save, num_frames);
    assert(rc == 0);

    SQ8Params p_load;
    uint8_t *data_load = NULL;
    long loaded_frames = 0;

    rc = sq8_load_sidecar(tmp_path, &p_load, &data_load, &loaded_frames);
    assert(rc == 0);
    assert(loaded_frames == num_frames);
    assert(p_load.dim == dim);
    assert(fabsf(p_load.min_val - p_save.min_val) < 1e-5f);
    assert(fabsf(p_load.max_val - p_save.max_val) < 1e-5f);
    assert(fabsf(p_load.scale - p_save.scale) < 1e-5f);
    assert(memcmp(data_save, data_load, total_bytes) == 0);

    free(data_save);
    free(data_load);
    remove(tmp_path);

    printf("  -> Sidecar file round-trip passed.\n");
}

static uint64_t reference_ssd_i16(
    const int16_t *a,
    const int16_t *b,
    long           dim)
{
    uint64_t sum = 0;
    for (long i = 0; i < dim; i++)
    {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i];
        sum += (uint64_t)(diff * diff);
    }
    return sum;
}

static void test_sq16_initialization_and_clamping()
{
    printf("[TEST] Testing SQ16 initialization and clamping...\n");
    SQ16Params params;
    sq16_init_params(&params, -10.0f, 10.0f, 128);

    assert(fabsf(params.min_val - (-10.0f)) < 1e-5f);
    assert(fabsf(params.max_val - 10.0f) < 1e-5f);
    assert(fabsf(params.scale - (20.0f / 32767.0f)) < 1e-5f);
    assert(params.dim == 128);

    float src[4] = {-15.0f, -10.0f, 10.0f, 25.0f};
    int16_t dst[4];
    params.dim = 4;
    sq16_quantize_float(src, dst, &params);

    assert(dst[0] == 0);     // Clamped below min
    assert(dst[1] == 0);     // Min
    assert(dst[2] == 32767); // Max
    assert(dst[3] == 32767); // Clamped above max

    printf("  -> SQ16 initialization and clamping passed.\n");
}

static void test_sq16_simd_bit_exactness()
{
    printf("[TEST] Testing SQ16 SIMD vs scalar bit-exactness...\n");
    long test_dims[] = {1, 7, 16, 31, 32, 33, 63, 64, 65, 128, 255, 1024, 4097};
    int num_dims = sizeof(test_dims) / sizeof(test_dims[0]);

    for (int d = 0; d < num_dims; d++)
    {
        long dim = test_dims[d];
        int16_t *a = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        int16_t *b = (int16_t *)malloc((size_t)dim * sizeof(int16_t));

        for (long i = 0; i < dim; i++)
        {
            a[i] = (int16_t)(rand() % 32768);
            b[i] = (int16_t)(rand() % 32768);
        }

        uint64_t ref = reference_ssd_i16(a, b, dim);
        uint64_t simd = sq16_dist_squared_i16(a, b, dim);

        assert(ref == simd);

        free(a);
        free(b);
    }

    printf("  -> SQ16 SIMD bit-exactness passed across all tested dimensions.\n");
}

static void test_sq16_metric_lower_bound_invariance()
{
    printf("[TEST] Testing SQ16 metric lower-bound invariance (100,000 random vector pairs)...\n");
    long test_dims[] = {2, 8, 32, 128, 512, 1024};
    int num_dims = sizeof(test_dims) / sizeof(test_dims[0]);
    int pairs_per_dim = 20000;

    for (int d = 0; d < num_dims; d++)
    {
        long dim = test_dims[d];
        SQ16Params params;
        sq16_init_params(&params, -50.0f, 150.0f, dim);

        float *fa = (float *)malloc((size_t)dim * sizeof(float));
        float *fb = (float *)malloc((size_t)dim * sizeof(float));
        int16_t *qa = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        int16_t *qb = (int16_t *)malloc((size_t)dim * sizeof(int16_t));

        for (int p = 0; p < pairs_per_dim; p++)
        {
            for (long i = 0; i < dim; i++)
            {
                fa[i] = -50.0f + (float)rand() / (float)(RAND_MAX / 200.0f);
                fb[i] = -50.0f + (float)rand() / (float)(RAND_MAX / 200.0f);
            }

            sq16_quantize_float(fa, qa, &params);
            sq16_quantize_float(fb, qb, &params);

            double d_exact = reference_euclidean_distance(fa, fb, dim);
            double d_lb = sq16_compute_lower_bound(qa, qb, &params, 0.0);

            // Lower bound MUST be <= exact distance (allowing tiny float epsilon)
            if (d_lb > d_exact + 1e-6)
            {
                fprintf(stderr, "VIOLATION at dim %ld, pair %d: d_lb=%.6f > d_exact=%.6f\n",
                        dim, p, d_lb, d_exact);
                assert(d_lb <= d_exact + 1e-6);
            }
        }

        free(fa);
        free(fb);
        free(qa);
        free(qb);
    }

    printf("  -> SQ16 metric lower-bound invariance passed: zero violations observed.\n");
}

static void test_sq16_sidecar_roundtrip()
{
    printf("[TEST] Testing .sq16 sidecar file save and load round-trip...\n");
    const char *tmp_path = "/tmp/test_sq16_sidecar.sq16";
    long dim = 256;
    long num_frames = 10;

    SQ16Params p_save;
    sq16_init_params(&p_save, -12.5f, 87.5f, dim);

    size_t total_elements = (size_t)dim * (size_t)num_frames;
    int16_t *data_save = (int16_t *)malloc(total_elements * sizeof(int16_t));
    for (size_t i = 0; i < total_elements; i++)
    {
        data_save[i] = (int16_t)(i % 32768);
    }

    int rc = sq16_save_sidecar(tmp_path, &p_save, data_save, num_frames);
    assert(rc == 0);

    SQ16Params p_load;
    int16_t *data_load = NULL;
    long loaded_frames = 0;

    rc = sq16_load_sidecar(tmp_path, &p_load, &data_load, &loaded_frames);
    assert(rc == 0);
    assert(loaded_frames == num_frames);
    assert(p_load.dim == dim);
    assert(fabsf(p_load.min_val - p_save.min_val) < 1e-5f);
    assert(fabsf(p_load.max_val - p_save.max_val) < 1e-5f);
    assert(fabsf(p_load.scale - p_save.scale) < 1e-5f);
    assert(memcmp(data_save, data_load, total_elements * sizeof(int16_t)) == 0);

    free(data_save);
    free(data_load);
    remove(tmp_path);

    printf("  -> SQ16 sidecar file round-trip passed.\n");
}

static void test_sq16_batch_and_filter()
{
    printf("[TEST] Testing SQ16 batch 1x4 and bulk filter bit-exactness...\n");
    long dim = 128;
    int num_candidates = 16;
    int16_t query[128];
    int16_t anchors[16][128];
    const int16_t *anchor_ptrs[16];
    int cand_indices[16];
    int clmembflag[16];

    for (long j = 0; j < dim; j++)
    {
        query[j] = (int16_t)(rand() % 32768);
    }
    for (int k = 0; k < num_candidates; k++)
    {
        cand_indices[k] = k;
        clmembflag[k] = 1;
        anchor_ptrs[k] = anchors[k];
        for (long j = 0; j < dim; j++)
        {
            anchors[k][j] = (int16_t)(rand() % 32768);
        }
    }

    // 1. Verify 1x4 batch SSD matches scalar SSD bitwise
    for (int b = 0; b < 4; b++)
    {
        uint64_t batch_out[4];
        sq16_dist_squared_batch_1x4_i16(query, &anchor_ptrs[b * 4], batch_out, dim);
        for (int k = 0; k < 4; k++)
        {
            uint64_t ref_ssd = sq16_dist_squared_i16(query, anchor_ptrs[b * 4 + k], dim);
            assert(batch_out[k] == ref_ssd);
        }
    }

    // 2. Verify bulk filter correctness
    SQ16Params params;
    sq16_init_params(&params, -1.0f, 1.0f, dim);
    double cutoff_dist = 0.5;

    int pruned = sq16_batch_filter_candidates(
        query, anchor_ptrs, cand_indices, num_candidates,
        cutoff_dist, &params, clmembflag);

    int manual_pruned = 0;
    for (int k = 0; k < num_candidates; k++)
    {
        double lb = sq16_compute_lower_bound(query, anchor_ptrs[k], &params, 0.0);
        if (lb > cutoff_dist)
        {
            assert(clmembflag[k] == 0);
            manual_pruned++;
        }
        else
        {
            assert(clmembflag[k] == 1);
        }
    }
    assert(pruned == manual_pruned);

    printf("  -> SQ16 batch 1x4 and bulk filter passed successfully (%d pruned).\n", pruned);
}

static void test_sq16_cutoff()
{
    printf("[TEST] Testing SQ16 early cutoff correctness...\n");
    const long dim = 128;
    int16_t a[128];
    int16_t b[128];

    for (int rep = 0; rep < 1000; rep++)
    {
        for (long j = 0; j < dim; j++)
        {
            a[j] = (int16_t)(rand() % 32768);
            b[j] = (int16_t)(rand() % 32768);
        }

        uint64_t full_ssd = sq16_dist_squared_i16(a, b, dim);

        // Test with cutoff > full_ssd (must return exact full_ssd)
        uint64_t high_cutoff = full_ssd + 1000;
        uint64_t res_high = sq16_dist_squared_cutoff_i16(a, b, dim, high_cutoff);
        assert(res_high == full_ssd);

        // Test with cutoff < full_ssd (must return > low_cutoff)
        if (full_ssd > 100)
        {
            uint64_t low_cutoff = full_ssd / 2;
            uint64_t res_low = sq16_dist_squared_cutoff_i16(a, b, dim, low_cutoff);
            assert(res_low > low_cutoff);
        }
    }
    printf("  -> SQ16 early cutoff passed across 1,000 random vectors.\n");
}

int main()
{
    printf("=== Running Scalar Quantization (SQ8 & SQ16) Unit Tests ===\n");
    test_initialization_and_clamping();
    test_simd_bit_exactness();
    test_metric_lower_bound_invariance();
    test_sidecar_roundtrip();

    test_sq16_initialization_and_clamping();
    test_sq16_simd_bit_exactness();
    test_sq16_metric_lower_bound_invariance();
    test_sq16_sidecar_roundtrip();
    test_sq16_batch_and_filter();
    test_sq16_cutoff();
    printf("=== All SQ8 & SQ16 Unit Tests Passed Successfully ===\n");
    return 0;
}

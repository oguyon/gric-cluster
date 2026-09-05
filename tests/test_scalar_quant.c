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

int main()
{
    printf("=== Running Scalar Quantization (SQ8) Unit Tests ===\n");
    test_initialization_and_clamping();
    test_simd_bit_exactness();
    test_metric_lower_bound_invariance();
    test_sidecar_roundtrip();
    printf("=== All SQ8 Unit Tests Passed Successfully ===\n");
    return 0;
}

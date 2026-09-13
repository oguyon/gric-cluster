/**
 * @file test_residual_quant.c
 * @brief Unit tests for 8-bit residual quantization and lower-bound metric safety.
 */

#include "residual_quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double reference_euclidean_dist(
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

static uint64_t reference_rq8_ssd(
    const int16_t *q_res,
    const int8_t  *cand_res,
    long           dim)
{
    uint64_t sum = 0;
    for (long i = 0; i < dim; i++)
    {
        int32_t diff = (int32_t)q_res[i] - (int32_t)cand_res[i];
        sum += (uint64_t)(diff * diff);
    }
    return sum;
}

static void test_initialization_and_clamping(void)
{
    printf("[TEST] Testing RQ8 initialization and clamping...\n");
    RQ8Params params;
    rq8_init_params(&params, 0.5f, 128);

    assert(fabsf(params.rlim - 0.5f) < 1e-5f);
    assert(fabsf(params.scale - (0.5f / 127.0f)) < 1e-5f);
    assert(params.dim == 128);

    float anchor[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float src[4] = {1.0f, 2.5f, 2.0f, 5.0f}; // deltas: 0.0, +0.5, -1.0 (clamped), +1.0 (clamped)
    int8_t dst[4];
    params.dim = 4;
    params.rlim = 0.5f;
    params.scale = 0.5f / 127.0f;
    params.inv_scale = 127.0f / 0.5f;

    rq8_quantize_residual_float(src, anchor, dst, &params);
    assert(dst[0] == 0);
    assert(dst[1] == 127);
    assert(dst[2] == -127); // Clamped at -127
    assert(dst[3] == 127);  // Clamped at +127
    printf("  -> Clamping passed.\n");
}

static void test_quantization_precision(void)
{
    printf("[TEST] Testing RQ8 quantization precision...\n");
    long dim = 128;
    RQ8Params params;
    rq8_init_params(&params, 0.2f, dim);

    float *anchor = (float *)malloc((size_t)dim * sizeof(float));
    float *src = (float *)malloc((size_t)dim * sizeof(float));
    int8_t *dst = (int8_t *)malloc((size_t)dim * sizeof(int8_t));

    srand(12345);
    for (long d = 0; d < dim; d++)
    {
        anchor[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // Generate member within rlim ball:
        float rand_val = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float delta = rand_val * (0.2f / sqrtf((float)dim));
        src[d] = anchor[d] + delta;
    }

    rq8_quantize_residual_float(src, anchor, dst, &params);

    for (long d = 0; d < dim; d++)
    {
        float delta_true = src[d] - anchor[d];
        float delta_dequant = (float)dst[d] * params.scale;
        float err = fabsf(delta_true - delta_dequant);
        assert(err <= params.scale * 0.5001f);
    }

    free(anchor);
    free(src);
    free(dst);
    printf("  -> Precision within scale/2 passed.\n");
}

static void test_metric_lower_bound_safety(void)
{
    printf("[TEST] Testing metric lower bound safety (zero false dismissals)...\n");
    long dims[] = {3, 16, 64, 128};
    int num_dims = 4;
    int num_trials = 20000;

    for (int di = 0; di < num_dims; di++)
    {
        long dim = dims[di];
        float rlim = 0.15f;
        RQ8Params params;
        rq8_init_params(&params, rlim, dim);

        float *anchor = (float *)malloc((size_t)dim * sizeof(float));
        float *query = (float *)malloc((size_t)dim * sizeof(float));
        float *cand = (float *)malloc((size_t)dim * sizeof(float));
        int8_t *cand_res = (int8_t *)malloc((size_t)dim * sizeof(int8_t));
        int16_t *query_res = (int16_t *)malloc((size_t)dim * sizeof(int16_t));

        for (int t = 0; t < num_trials; t++)
        {
            // Random anchor
            for (long d = 0; d < dim; d++)
            {
                anchor[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }

            // Member inside cluster ball: ||cand - anchor|| <= rlim
            float cand_norm = 0.0f;
            for (long d = 0; d < dim; d++)
            {
                cand[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                cand_norm += cand[d] * cand[d];
            }
            cand_norm = sqrtf(cand_norm);
            float scale_cand = (((float)rand() / (float)RAND_MAX) * rlim) / (cand_norm + 1e-9f);
            for (long d = 0; d < dim; d++)
            {
                cand[d] = anchor[d] + cand[d] * scale_cand;
            }

            // Random query within nearby space
            float q_norm = 0.0f;
            for (long d = 0; d < dim; d++)
            {
                query[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                q_norm += query[d] * query[d];
            }
            q_norm = sqrtf(q_norm);
            float scale_q = (((float)rand() / (float)RAND_MAX) * rlim * 2.5f) / (q_norm + 1e-9f);
            for (long d = 0; d < dim; d++)
            {
                query[d] = anchor[d] + query[d] * scale_q;
            }

            rq8_quantize_residual_float(cand, anchor, cand_res, &params);
            rq8_quantize_query_residual_float(query, anchor, query_res, &params);

            uint64_t ssd = reference_rq8_ssd(query_res, cand_res, dim);
            double d_lb = rq8_compute_lower_bound(ssd, &params, 0.0);
            double d_true = reference_euclidean_dist(query, cand, dim);

            // True distance must be >= metric lower bound
            assert(d_lb <= d_true + 1e-5);

            // Test cutoff threshold: if cutoff is d_true, it must NEVER be pruned!
            uint64_t ssd_thresh = rq8_compute_cutoff_thresh(d_true, &params, 0.0);
            assert(ssd <= ssd_thresh);
        }

        free(anchor);
        free(query);
        free(cand);
        free(cand_res);
        free(query_res);
    }
    printf("  -> 80,000 randomized metric trials passed with 0 false dismissals.\n");
}

static void test_cutoff_kernel_consistency(void)
{
    printf("[TEST] Testing SIMD cutoff kernel vs reference scalar...\n");
    long dims[] = {3, 16, 32, 64, 128, 256};
    int num_dims = 6;

    for (int di = 0; di < num_dims; di++)
    {
        long dim = dims[di];
        int16_t *q_res = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        int8_t *cand_res = (int8_t *)malloc((size_t)dim * sizeof(int8_t));

        for (int trial = 0; trial < 100; trial++)
        {
            for (long d = 0; d < dim; d++)
            {
                q_res[d] = (int16_t)((rand() % 500) - 250);
                cand_res[d] = (int8_t)((rand() % 255) - 127);
            }

            uint64_t ref_ssd = reference_rq8_ssd(q_res, cand_res, dim);

            // Test cutoff > ref_ssd
            uint64_t ssd_high = rq8_dist_squared_cutoff_i8(q_res, cand_res, dim, ref_ssd + 100);
            assert(ssd_high == ref_ssd);

            // Test cutoff < ref_ssd (early exit)
            if (ref_ssd > 50)
            {
                uint64_t ssd_low = rq8_dist_squared_cutoff_i8(q_res, cand_res, dim, ref_ssd - 10);
                assert(ssd_low > ref_ssd - 10);
            }
        }

        free(q_res);
        free(cand_res);
    }
    printf("  -> Cutoff kernel consistency verified across all dimensions.\n");
}

static void test_fastscan_32x(void)
{
    printf("[TEST] Testing FastScan 32x kernel vs scalar bitmask...\n");
    long dims[] = {3, 16, 64, 128};
    int num_dims = 4;

    for (int di = 0; di < num_dims; di++)
    {
        long dim = dims[di];
        int16_t *query_res = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        int8_t *block_coords = (int8_t *)malloc((size_t)dim * 32 * sizeof(int8_t));

        for (int trial = 0; trial < 50; trial++)
        {
            for (long d = 0; d < dim; d++)
            {
                query_res[d] = (int16_t)((rand() % 400) - 200);
                for (int i = 0; i < 32; i++)
                {
                    block_coords[d * 32 + i] = (int8_t)((rand() % 255) - 127);
                }
            }

            // Pick arbitrary cutoff
            uint64_t cutoff = (uint64_t)(dim * 2000);

            uint32_t ref_mask = rq8_fastscan_32x_generic_scalar(
                query_res, block_coords, dim, cutoff
            );
            uint32_t simd_mask = rq8_fastscan_32x(query_res, block_coords, dim, cutoff);

            assert(ref_mask == simd_mask);
        }

        free(query_res);
        free(block_coords);
    }
    printf("  -> FastScan 32x bitmask exact match verified.\n");
}

static void test_sidecar_roundtrip(void)
{
    printf("[TEST] Testing .rq8 sidecar serialization...\n");
    const char *tmp_path = "test_sidecar.rq8";
    long dim = 32;
    long num_frames = 100;
    RQ8Params params;
    rq8_init_params(&params, 0.25f, dim);

    size_t total_elements = (size_t)num_frames * (size_t)dim;
    int8_t *data_out = (int8_t *)malloc(total_elements * sizeof(int8_t));
    for (size_t i = 0; i < total_elements; i++)
    {
        data_out[i] = (int8_t)((rand() % 255) - 127);
    }

    int ret_save = rq8_save_sidecar(tmp_path, &params, data_out, num_frames);
    assert(ret_save == 0);

    RQ8Params loaded_params;
    int8_t *data_in = NULL;
    long loaded_frames = 0;
    int ret_load = rq8_load_sidecar(tmp_path, &loaded_params, &data_in, &loaded_frames);
    assert(ret_load == 0);
    assert(loaded_frames == num_frames);
    assert(loaded_params.dim == params.dim);
    assert(fabsf(loaded_params.rlim - params.rlim) < 1e-6f);
    assert(memcmp(data_out, data_in, total_elements * sizeof(int8_t)) == 0);

    remove(tmp_path);
    free(data_out);
    free(data_in);
    printf("  -> Sidecar serialization round-trip passed.\n");
}

int main(void)
{
    printf("=== Starting Residual Quantization (RQ8) Unit Tests ===\n");
    printf("SIMD Mode detected: %s\n", rq8_get_simd_mode_str());

    test_initialization_and_clamping();
    test_quantization_precision();
    test_metric_lower_bound_safety();
    test_cutoff_kernel_consistency();
    test_fastscan_32x();
    test_sidecar_roundtrip();

    printf("=== All Residual Quantization (RQ8) Unit Tests Passed! ===\n");
    return 0;
}

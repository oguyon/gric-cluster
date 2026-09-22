/**
 * @file test_rq8_e8.c
 * @brief Unit tests for E8 lattice residual vector quantization and metric bounding.
 */

#include "residual_quant.h"
#include "e8_lattice.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void test_e8_rq8_params(void)
{
    printf("Testing RQ8 E8 parameter initialization...\n");

    RQ8Params params_cubic;
    RQ8Params params_e8;

    float rlim = 2.0f;
    long dim = 128;

    rq8_init_params_ex(&params_cubic, rlim, dim, RQ8_LATTICE_CUBIC);
    rq8_init_params_ex(&params_e8, rlim, dim, RQ8_LATTICE_E8);

    assert(params_cubic.lattice_mode == RQ8_LATTICE_CUBIC);
    assert(params_e8.lattice_mode == RQ8_LATTICE_E8);

    /* Scales must match */
    assert(fabsf(params_cubic.scale - params_e8.scale) < 1e-7f);

    /* E8 covering radius must be exactly 29.29% smaller */
    float ratio = params_e8.err_radius / params_cubic.err_radius;
    printf("  dim=%ld: err_radius(Cubic)=%.6f, err_radius(E8)=%.6f (ratio=%.4f)\n",
           dim, params_cubic.err_radius, params_e8.err_radius, ratio);
    assert(fabsf(ratio - (float)E8_COVERING_RATIO) < 1e-4f);
}

static void test_e8_rq8_bounding(void)
{
    printf("Testing RQ8 E8 metric lower bounding guarantees...\n");

    long dim = 64;
    float rlim = 1.5f;

    RQ8Params params_cubic;
    RQ8Params params_e8;
    rq8_init_params_ex(&params_cubic, rlim, dim, RQ8_LATTICE_CUBIC);
    rq8_init_params_ex(&params_e8, rlim, dim, RQ8_LATTICE_E8);

    float anchor[64];
    for (int i = 0; i < 64; i++)
    {
        anchor[i] = 0.0f;
    }

    srand(12345);
    int trials = 10000;
    int tighter_count = 0;

    for (int t = 0; t < trials; t++)
    {
        float cand[64];
        float query[64];

        /* Generate candidate inside cluster ball */
        float c_norm_sq = 0.0f;
        for (int i = 0; i < 64; i++)
        {
            cand[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            c_norm_sq += cand[i] * cand[i];
        }
        float c_norm = sqrtf(c_norm_sq);
        if (c_norm > rlim * 0.9f)
        {
            float s = (rlim * 0.9f) / c_norm;
            for (int i = 0; i < 64; i++)
            {
                cand[i] *= s;
            }
        }

        /* Generate query near cluster */
        for (int i = 0; i < 64; i++)
        {
            query[i] = ((float)rand() / (float)RAND_MAX) * 0.4f - 0.2f;
        }

        /* Compute true Euclidean distance */
        double true_dist_sq = 0.0;
        for (int i = 0; i < 64; i++)
        {
            double diff = (double)query[i] - (double)cand[i];
            true_dist_sq += diff * diff;
        }
        double true_dist = sqrt(true_dist_sq);

        /* Cubic Quantization */
        int8_t cand_i8_c[64];
        int16_t query_i16_c[64];
        rq8_quantize_residual_float(cand, anchor, cand_i8_c, &params_cubic);
        rq8_quantize_query_residual_float(query, anchor, query_i16_c, &params_cubic);

        uint64_t ssd_c = 0;
        for (int i = 0; i < 64; i++)
        {
            int32_t diff = (int32_t)query_i16_c[i] - (int32_t)cand_i8_c[i];
            ssd_c += (uint64_t)(diff * diff);
        }
        double lb_cubic = rq8_compute_lower_bound(ssd_c, &params_cubic, 0.0);

        /* E8 Lattice Quantization */
        int8_t cand_i8_e8[64];
        int16_t query_i16_e8[64];
        rq8_quantize_residual_float(cand, anchor, cand_i8_e8, &params_e8);
        rq8_quantize_query_residual_float(query, anchor, query_i16_e8, &params_e8);

        uint64_t ssd_e8 = 0;
        for (int i = 0; i < 64; i++)
        {
            int32_t diff = (int32_t)query_i16_e8[i] - (int32_t)cand_i8_e8[i];
            ssd_e8 += (uint64_t)(diff * diff);
        }
        double lb_e8 = rq8_compute_lower_bound(ssd_e8, &params_e8, 0.0);

        /* Metric bounding guarantee: lower bound must NEVER exceed true distance */
        assert(lb_e8 <= true_dist + 1e-4);
        assert(lb_cubic <= true_dist + 1e-4);

        if (lb_e8 > lb_cubic)
        {
            tighter_count++;
        }
    } // for (int t = 0; t < trials; t++)

    printf("  %d / %d trials (%.1f%%) produced strictly tighter lower bounds with E8!\n",
           tighter_count, trials, ((double)tighter_count / (double)trials) * 100.0);
}

static void test_e8_rq8_adc(void)
{
    printf("Testing RQ8 E8 Asymmetric Distance Computation (ADC)...\n");

    long dim = 64;
    float rlim = 1.5f;

    RQ8Params params_e8;
    rq8_init_params_ex(&params_e8, rlim, dim, RQ8_LATTICE_E8);

    float anchor[64];
    for (int i = 0; i < 64; i++)
    {
        anchor[i] = 0.0f;
    }

    srand(54321);
    int trials = 10000;
    int adc_tighter_count = 0;

    for (int t = 0; t < trials; t++)
    {
        float cand[64];
        float query[64];

        for (int i = 0; i < 64; i++)
        {
            cand[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            query[i] = ((float)rand() / (float)RAND_MAX) * 0.4f - 0.2f;
        }

        double true_dist_sq = 0.0;
        for (int i = 0; i < 64; i++)
        {
            double diff = (double)query[i] - (double)cand[i];
            true_dist_sq += diff * diff;
        }
        double true_dist = sqrt(true_dist_sq);

        int8_t cand_i8[64];
        rq8_quantize_residual_float(cand, anchor, cand_i8, &params_e8);

        /* SDC */
        int16_t query_i16[64];
        rq8_quantize_query_residual_float(query, anchor, query_i16, &params_e8);
        uint64_t ssd_sdc = 0;
        for (int i = 0; i < 64; i++)
        {
            int32_t diff = (int32_t)query_i16[i] - (int32_t)cand_i8[i];
            ssd_sdc += (uint64_t)(diff * diff);
        }
        double lb_sdc = rq8_compute_lower_bound(ssd_sdc, &params_e8, 0.0);

        /* ADC */
        float query_adc[64];
        rq8_prepare_query_residual_float(query, anchor, query_adc, &params_e8);
        float dist_sq_adc = rq8_dist_asym_cutoff_f32(query_adc, cand_i8, dim, 1e30f);
        double lb_adc = rq8_compute_lower_bound_adc(dist_sq_adc, &params_e8, 0.0);

        /* ADC metric bounding guarantee: lb_adc <= true_dist */
        assert(lb_adc <= true_dist + 1e-4);

        if (lb_adc > lb_sdc)
        {
            adc_tighter_count++;
        }
    }

    printf("  %d / %d trials (%.1f%%) produced strictly tighter lower bounds with ADC!\n",
           adc_tighter_count, trials, ((double)adc_tighter_count / (double)trials) * 100.0);
}

static void test_e8_rq8_fastscan_and_sparse(void)
{
    printf("Testing RQ8 FastScan ADC and SparseCache 1x4 batching...\n");

    long dim = 64;
    float rlim = 1.0f;
    RQ8Params params;
    rq8_init_params_ex(&params, rlim, dim, RQ8_LATTICE_E8);

    int8_t cand_block[32 * 64];
    int8_t transposed[32 * 64];

    for (int m = 0; m < 32; m++)
    {
        for (int d = 0; d < 64; d++)
        {
            cand_block[m * 64 + d] = (int8_t)((rand() % 41) - 20);
        }
    }

    /* Transpose to 32-wide FastScan layout */
    for (int d = 0; d < 64; d++)
    {
        for (int lane = 0; lane < 32; lane++)
        {
            transposed[d * 32 + lane] = cand_block[lane * 64 + d];
        }
    }

    float q_adc[64];
    for (int d = 0; d < 64; d++)
    {
        q_adc[d] = ((float)rand() / (float)RAND_MAX) * 20.0f - 10.0f;
    }

    /* Verify 1x4 batching against scalar */
    for (int i = 0; i < 32; i += 4)
    {
        const int8_t *cands[4] = {
            cand_block + (i + 0) * 64,
            cand_block + (i + 1) * 64,
            cand_block + (i + 2) * 64,
            cand_block + (i + 3) * 64
        };
        float dsq[4];
        rq8_dist_asym_cutoff_batch_1x4(q_adc, cands, dim, 1e30f, dsq);

        for (int k = 0; k < 4; k++)
        {
            float expected = rq8_dist_asym_cutoff_f32(q_adc, cands[k], dim, 1e30f);
            assert(fabsf(dsq[k] - expected) < 1e-3f);
        }
    }

    /* Verify FastScan 32x bitmask against scalar */
    float cutoff = 500.0f;
    uint32_t expected_mask = 0;
    for (int m = 0; m < 32; m++)
    {
        float d = rq8_dist_asym_cutoff_f32(q_adc, cand_block + m * 64, dim, cutoff);
        if (d <= cutoff)
        {
            expected_mask |= (1U << m);
        }
    }

    uint32_t fastscan_mask = rq8_fastscan_32x_adc(q_adc, transposed, dim, cutoff);
    assert(fastscan_mask == expected_mask);

    printf("  Verified 1x4 batching and FastScan ADC bitmask against scalar!\n");
}

int main(void)
{
    printf("===========================================\n");
    printf("Running RQ8 E8 Lattice Quantization Tests\n");
    printf("===========================================\n");

    test_e8_rq8_params();
    test_e8_rq8_bounding();
    test_e8_rq8_adc();
    test_e8_rq8_fastscan_and_sparse();

    printf("\nAll RQ8 E8 quantization tests passed successfully!\n");
    return 0;
}

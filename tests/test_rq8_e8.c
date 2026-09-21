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

int main(void)
{
    printf("===========================================\n");
    printf("Running RQ8 E8 Lattice Quantization Tests\n");
    printf("===========================================\n");

    test_e8_rq8_params();
    test_e8_rq8_bounding();

    printf("\nAll RQ8 E8 quantization tests passed successfully!\n");
    return 0;
}

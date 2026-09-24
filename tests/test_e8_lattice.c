/**
 * @file test_e8_lattice.c
 * @brief Unit tests for E8 lattice algebra, Conway-Sloane quantizer, and root geometry.
 */

#include "e8_lattice.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * is_in_e8() - Check if an 8D vector belongs to the E8 lattice.
 * @v: 8D coordinate vector.
 *
 * Vector is in E8 if either:
 * 1) All coordinates are integers and their sum is even (D8 lattice).
 * 2) All coordinates are half-integers and sum of minus signs is even.
 */
static bool is_in_e8(
    const double *v)
{
    bool all_int = true;
    bool all_half = true;
    int int_sum = 0;
    int minus_count = 0;

    for (int i = 0; i < 8; i++)
    {
        double diff_int = fabs(v[i] - round(v[i]));
        if (diff_int > 1e-5)
        {
            all_int = false;
        }
        else
        {
            int_sum += (int)round(v[i]);
        }

        double diff_half = fabs(v[i] - (floor(v[i]) + 0.5));
        if (diff_half > 1e-5)
        {
            all_half = false;
        }
        else
        {
            if (v[i] < 0.0)
            {
                minus_count++;
            }
        }
    }

    if (all_int && ((int_sum & 1) == 0))
    {
        return true;
    }

    if (all_half)
    {
        /* In 2*E8, all coordinates are odd integers with sum = 0 mod 4 */
        int doubled_sum = 0;
        for (int i = 0; i < 8; i++)
        {
            doubled_sum += (int)round(v[i] * 2.0);
        }
        if ((doubled_sum % 4) == 0)
        {
            return true;
        }
    }

    return false;
}

static void test_e8_roots(void)
{
    printf("Testing E8 240 root vectors...\n");
    e8_init_root_table();

    const float (*roots_f)[8] = e8_get_roots_float();
    const int8_t (*roots_i)[8] = e8_get_roots_doubled_int8();

    /* 1. Verify norms and integer representation */
    for (int r = 0; r < E8_NUM_ROOTS; r++)
    {
        double norm_sq = 0.0;
        double d_root[8];
        for (int i = 0; i < 8; i++)
        {
            d_root[i] = (double)roots_f[r][i];
            norm_sq += d_root[i] * d_root[i];

            /* Check that doubled integer matches float * 2 */
            int doubled = (int)round(d_root[i] * 2.0);
            assert(roots_i[r][i] == doubled);
        }

        /* Standard norm squared of root must be 2.0 */
        assert(fabs(norm_sq - 2.0) < 1e-6);

        /* Root must belong to E8 */
        assert(is_in_e8(d_root));
    } // for (int r = 0; r < E8_NUM_ROOTS; r++)

    /* 2. Verify all 240 roots are unique */
    for (int r1 = 0; r1 < E8_NUM_ROOTS - 1; r1++)
    {
        for (int r2 = r1 + 1; r2 < E8_NUM_ROOTS; r2++)
        {
            bool identical = true;
            for (int i = 0; i < 8; i++)
            {
                if (roots_i[r1][i] != roots_i[r2][i])
                {
                    identical = false;
                    break;
                }
            }
            assert(!identical);
        }
    }

    printf("  All 240 roots validated: norm^2 = 2.0, unique, and valid E8 points.\n");
}

static void test_conway_sloane_quantizer(void)
{
    printf("Testing Conway-Sloane E8 quantizer...\n");

    const float (*roots_f)[8] = e8_get_roots_float();

    /* 1. Quantizing any root must return itself */
    for (int r = 0; r < E8_NUM_ROOTS; r++)
    {
        float out[8];
        e8_quantize_point_float(roots_f[r], out);

        for (int i = 0; i < 8; i++)
        {
            assert(fabsf(out[i] - roots_f[r][i]) < 1e-6f);
        }
    }

    /* 2. Test 10,000 random 8D points */
    srand(42);
    int num_random_tests = 10000;
    double max_err_sq = 0.0;

    for (int t = 0; t < num_random_tests; t++)
    {
        double pt[8];
        for (int i = 0; i < 8; i++)
        {
            pt[i] = ((double)rand() / (double)RAND_MAX) * 10.0 - 5.0;
        }

        double quantized[8];
        e8_quantize_point_double(pt, quantized);

        /* Check that quantized point is in E8 */
        assert(is_in_e8(quantized));

        /* Check that distance is within covering radius (R_c = 1.0, R_c^2 = 1.0) */
        double err_sq = 0.0;
        for (int i = 0; i < 8; i++)
        {
            double diff = pt[i] - quantized[i];
            err_sq += diff * diff;
        }

        if (err_sq > max_err_sq)
        {
            max_err_sq = err_sq;
        }

        /* Strict mathematical guarantee: distance <= 1.0 */
        assert(err_sq <= 1.000001);
    } // for (int t = 0; t < num_random_tests; t++)

    printf("  Conway-Sloane quantizer passed %d trials (max error^2 = %.5f <= 1.0).\n",
           num_random_tests, max_err_sq);
}

static void test_covering_radius_gain(void)
{
    printf("Testing E8 vs Z^D covering radius gain...\n");

    long dims[] = {8, 16, 64, 128, 512};
    int num_dims = (int)(sizeof(dims) / sizeof(dims[0]));
    float scale = 0.25f;

    for (int d = 0; d < num_dims; d++)
    {
        long dim = dims[d];
        float r_e8 = e8_covering_radius(dim, scale);
        float r_cubic = sqrtf((float)dim) * scale * 0.5f;

        float ratio = r_e8 / r_cubic;
        float reduction_pct = (1.0f - ratio) * 100.0f;

        printf("  Dim=%3ld: R_c(E8)=%.4f vs R_c(Cubic)=%.4f -> %.2f%% tighter bounds\n",
               dim, r_e8, r_cubic, reduction_pct);

        /* Ratio must match 1 / sqrt(2) = 0.707107 (29.29% reduction) */
        assert(fabsf(ratio - (float)E8_COVERING_RATIO) < 1e-4f);
    }
}

static void test_e8_find_nearest_root(void)
{
    printf("Testing E8 find nearest root vector...\n");

    const float (*roots_f)[8] = e8_get_roots_float();

    /* 1. Finding nearest root to each root must return the root itself */
    for (int r = 0; r < E8_NUM_ROOTS; r++)
    {
        int   best_idx = -1;
        float cos_sim = e8_find_nearest_root_float(roots_f[r], &best_idx);

        assert(best_idx == r);
        assert(fabsf(cos_sim - 1.0f) < 1e-5f);

        /* Double precision */
        double d_root[8];
        for (int i = 0; i < 8; i++)
        {
            d_root[i] = (double)roots_f[r][i];
        } // for (int i = 0; i < 8; i++)

        int    best_d_idx = -1;
        double d_cos_sim = e8_find_nearest_root_double(d_root, &best_d_idx);

        assert(best_d_idx == r);
        assert(fabs(d_cos_sim - 1.0) < 1e-6);
    } // for (int r = 0; r < E8_NUM_ROOTS; r++)

    /* 2. Test 1,000 random direction vectors */
    srand(12345);
    for (int t = 0; t < 1000; t++)
    {
        float  f_dir[8];
        double d_dir[8];
        for (int i = 0; i < 8; i++)
        {
            f_dir[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            d_dir[i] = (double)f_dir[i];
        } // for (int i = 0; i < 8; i++)

        int   best_f = -1;
        float sim_f = e8_find_nearest_root_float(f_dir, &best_f);
        assert(best_f >= 0 && best_f < E8_NUM_ROOTS);
        assert(sim_f >= -1.0f && sim_f <= 1.0001f);

        int    best_d = -1;
        double sim_d = e8_find_nearest_root_double(d_dir, &best_d);
        assert(best_d == best_f);
        assert(fabs((double)sim_f - sim_d) < 1e-5);
    } // for (int t = 0; t < 1000; t++)

    printf("  Nearest root search validated on all 240 roots and 1000 random vectors.\n");
}

int main(void)
{
    printf("========================================\n");
    printf("Running E8 Lattice Algebraic Unit Tests\n");
    printf("========================================\n");

    test_e8_roots();
    test_conway_sloane_quantizer();
    test_covering_radius_gain();
    test_e8_find_nearest_root();

    printf("\nAll E8 lattice unit tests passed successfully!\n");
    return 0;
}

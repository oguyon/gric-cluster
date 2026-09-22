/**
 * @file test_framedist_cutoff.c
 * @brief Unit tests for 1x4 batched distance kernels with early-cutoff checkpoints.
 */

#include "framedistance.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ref_dist_float(
    const float *a,
    const float *b,
    long         dim)
{
    double sum = 0.0;
    for (long d = 0; d < dim; d++)
    {
        double diff = (double)a[d] - (double)b[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double ref_dist_double(
    const double *a,
    const double *b,
    long          dim)
{
    double sum = 0.0;
    for (long d = 0; d < dim; d++)
    {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static void test_cutoff_1x4_float_unconstrained(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x4_float (unconstrained cutoff_sq=0)...\n");
    long test_dims[] = {2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256, 500, 512, 1024};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        float *q = (float *)malloc((size_t)dim * sizeof(float));
        float *anchors_mem = (float *)malloc((size_t)4 * (size_t)dim * sizeof(float));
        const float *anchors[4];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 4; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        }

        for (int k = 0; k < 4; k++)
        {
            float *a_ptr = (float *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
            }
        }

        double out_dists[4];
        int mask = framedist_batch_cutoff_1x4_float(q, anchors, dim, 0.0, out_dists);
        assert(mask == 0);

        for (int k = 0; k < 4; k++)
        {
            double ref = ref_dist_float(q, anchors[k], dim);
            double diff = fabs(out_dists[k] - ref);
            if (diff > 1e-4)
            {
                fprintf(stderr, "FAIL: dim=%ld k=%d dist=%f ref=%f diff=%f\n",
                        dim, k, out_dists[k], ref, diff);
                assert(diff <= 1e-4);
            }
        }

        free(q);
        free(anchors_mem);
    } // for (int idx = 0; idx < num_dims; idx++)
}

static void test_cutoff_1x4_float_early_exit(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x4_float (early cutoff abort)...\n");
    long test_dims[] = {64, 128, 256, 512, 1024};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        float *q = (float *)malloc((size_t)dim * sizeof(float));
        float *anchors_mem = (float *)malloc((size_t)4 * (size_t)dim * sizeof(float));
        const float *anchors[4];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 4; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = 0.0f;
        }

        /* Set anchors far away so they all exceed a small cutoff */
        for (int k = 0; k < 4; k++)
        {
            float *a_ptr = (float *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = 10.0f;
            }
        }

        double out_dists[4];
        double cutoff = 1.0;
        double cutoff_sq = cutoff * cutoff;
        int mask = framedist_batch_cutoff_1x4_float(q, anchors, dim, cutoff_sq, out_dists);
        assert(mask == 0xF);

        for (int k = 0; k < 4; k++)
        {
            assert(out_dists[k] >= cutoff);
        }

        free(q);
        free(anchors_mem);
    } // for (int idx = 0; idx < num_dims; idx++)
}

static void test_cutoff_1x4_float_mixed(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x4_float (mixed cutoff survival)...\n");
    long dim = 512;
    float *q = (float *)malloc((size_t)dim * sizeof(float));
    float *anchors_mem = (float *)malloc((size_t)4 * (size_t)dim * sizeof(float));
    const float *anchors[4];
    assert(q != NULL && anchors_mem != NULL);

    for (int k = 0; k < 4; k++)
    {
        anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
    }

    for (long i = 0; i < dim; i++)
    {
        q[i] = 0.0f;
    }

    /* Candidate 0: close (dist ~ 0.5) */
    float *a0 = (float *)anchors[0];
    for (long i = 0; i < dim; i++)
    {
        a0[i] = 0.5f / sqrtf((float)dim);
    }

    /* Candidate 1: far (dist ~ 10.0) */
    float *a1 = (float *)anchors[1];
    for (long i = 0; i < dim; i++)
    {
        a1[i] = 10.0f / sqrtf((float)dim);
    }

    /* Candidate 2: close (dist ~ 0.8) */
    float *a2 = (float *)anchors[2];
    for (long i = 0; i < dim; i++)
    {
        a2[i] = 0.8f / sqrtf((float)dim);
    }

    /* Candidate 3: far (dist ~ 20.0) */
    float *a3 = (float *)anchors[3];
    for (long i = 0; i < dim; i++)
    {
        a3[i] = 20.0f / sqrtf((float)dim);
    }

    double cutoff = 2.0;
    double cutoff_sq = cutoff * cutoff;
    double out_dists[4];
    int mask = framedist_batch_cutoff_1x4_float(q, anchors, dim, cutoff_sq, out_dists);

    /* Bits 0 and 2 should NOT be pruned (mask & 1 == 0, mask & 4 == 0) */
    assert((mask & 0x1) == 0);
    assert((mask & 0x4) == 0);

    /* Bits 1 and 3 SHOULD be pruned (mask & 2 != 0, mask & 8 != 0) */
    assert((mask & 0x2) != 0);
    assert((mask & 0x8) != 0);

    /* Surviving candidates must match exact reference distance */
    double ref0 = ref_dist_float(q, anchors[0], dim);
    double ref2 = ref_dist_float(q, anchors[2], dim);
    assert(fabs(out_dists[0] - ref0) < 1e-4);
    assert(fabs(out_dists[2] - ref2) < 1e-4);

    free(q);
    free(anchors_mem);
}

static void test_cutoff_1x4_double_unconstrained(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x4_double (unconstrained cutoff_sq=0)...\n");
    long test_dims[] = {2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256, 500, 512};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        double *q = (double *)malloc((size_t)dim * sizeof(double));
        double *anchors_mem = (double *)malloc((size_t)4 * (size_t)dim * sizeof(double));
        const double *anchors[4];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 4; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
        }

        for (int k = 0; k < 4; k++)
        {
            double *a_ptr = (double *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
            }
        }

        double out_dists[4];
        int mask = framedist_batch_cutoff_1x4_double(q, anchors, dim, 0.0, out_dists);
        assert(mask == 0);

        for (int k = 0; k < 4; k++)
        {
            double ref = ref_dist_double(q, anchors[k], dim);
            double diff = fabs(out_dists[k] - ref);
            assert(diff <= 1e-7);
        }

        free(q);
        free(anchors_mem);
    } // for (int idx = 0; idx < num_dims; idx++)
}

static void test_cutoff_1x4_double_mixed(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x4_double (mixed cutoff survival)...\n");
    long dim = 256;
    double *q = (double *)malloc((size_t)dim * sizeof(double));
    double *anchors_mem = (double *)malloc((size_t)4 * (size_t)dim * sizeof(double));
    const double *anchors[4];
    assert(q != NULL && anchors_mem != NULL);

    for (int k = 0; k < 4; k++)
    {
        anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
    }

    for (long i = 0; i < dim; i++)
    {
        q[i] = 0.0;
    }

    double *a0 = (double *)anchors[0];
    double *a1 = (double *)anchors[1];
    double *a2 = (double *)anchors[2];
    double *a3 = (double *)anchors[3];

    for (long i = 0; i < dim; i++)
    {
        a0[i] = 0.5 / sqrt((double)dim);
        a1[i] = 10.0 / sqrt((double)dim);
        a2[i] = 0.8 / sqrt((double)dim);
        a3[i] = 20.0 / sqrt((double)dim);
    }

    double cutoff = 2.0;
    double cutoff_sq = cutoff * cutoff;
    double out_dists[4];
    int mask = framedist_batch_cutoff_1x4_double(q, anchors, dim, cutoff_sq, out_dists);

    assert((mask & 0x1) == 0);
    assert((mask & 0x4) == 0);
    assert((mask & 0x2) != 0);
    assert((mask & 0x8) != 0);

    double ref0 = ref_dist_double(q, anchors[0], dim);
    double ref2 = ref_dist_double(q, anchors[2], dim);
    assert(fabs(out_dists[0] - ref0) < 1e-7);
    assert(fabs(out_dists[2] - ref2) < 1e-7);

    free(q);
    free(anchors_mem);
}

static void test_cutoff_1x8_float_unconstrained(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x8_float (unconstrained cutoff_sq=0)...\n");
    long test_dims[] = {2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256, 500, 512, 1024};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        float *q = (float *)malloc((size_t)dim * sizeof(float));
        float *anchors_mem = (float *)malloc((size_t)8 * (size_t)dim * sizeof(float));
        const float *anchors[8];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 8; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        }

        for (int k = 0; k < 8; k++)
        {
            float *a_ptr = (float *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
            }
        }

        double out_dists[8];
        int mask = framedist_batch_cutoff_1x8_float(q, anchors, dim, 0.0, out_dists);
        assert(mask == 0);

        for (int k = 0; k < 8; k++)
        {
            double ref = ref_dist_float(q, anchors[k], dim);
            double diff = fabs(out_dists[k] - ref);
            if (diff > 1e-4)
            {
                fprintf(stderr, "FAIL 1x8: dim=%ld k=%d dist=%f ref=%f diff=%f\n",
                        dim, k, out_dists[k], ref, diff);
                assert(diff <= 1e-4);
            }
        }

        free(q);
        free(anchors_mem);
    }
}

static void test_cutoff_1x8_float_early_exit(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x8_float (early cutoff abort)...\n");
    long test_dims[] = {64, 128, 256, 512, 1024};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        float *q = (float *)malloc((size_t)dim * sizeof(float));
        float *anchors_mem = (float *)malloc((size_t)8 * (size_t)dim * sizeof(float));
        const float *anchors[8];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 8; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = 0.0f;
        }

        for (int k = 0; k < 8; k++)
        {
            float *a_ptr = (float *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = 10.0f;
            }
        }

        double cutoff = 1.0;
        double cutoff_sq = cutoff * cutoff;
        double out_dists[8];
        int mask = framedist_batch_cutoff_1x8_float(q, anchors, dim, cutoff_sq, out_dists);
        assert(mask == 0xFF);

        for (int k = 0; k < 8; k++)
        {
            assert(out_dists[k] > cutoff);
        }

        free(q);
        free(anchors_mem);
    }
}

static void test_cutoff_1x8_float_mixed(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x8_float (mixed cutoff)...\n");
    long dim = 512;
    float *q = (float *)malloc((size_t)dim * sizeof(float));
    float *anchors_mem = (float *)malloc((size_t)8 * (size_t)dim * sizeof(float));
    const float *anchors[8];
    assert(q != NULL && anchors_mem != NULL);

    for (int k = 0; k < 8; k++)
    {
        anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
    }

    for (long i = 0; i < dim; i++)
    {
        q[i] = 0.0f;
    }

    for (int k = 0; k < 8; k++)
    {
        float *a_ptr = (float *)anchors[k];
        float val = ((k % 2) == 0) ? 0.5f : 10.0f;
        for (long i = 0; i < dim; i++)
        {
            a_ptr[i] = val / sqrtf((float)dim);
        }
    }

    double cutoff = 2.0;
    double cutoff_sq = cutoff * cutoff;
    double out_dists[8];
    int mask = framedist_batch_cutoff_1x8_float(q, anchors, dim, cutoff_sq, out_dists);

    for (int k = 0; k < 8; k++)
    {
        if ((k % 2) == 0)
        {
            assert((mask & (1 << k)) == 0);
            double ref = ref_dist_float(q, anchors[k], dim);
            assert(fabs(out_dists[k] - ref) < 1e-4);
        }
        else
        {
            assert((mask & (1 << k)) != 0);
            assert(out_dists[k] > cutoff);
        }
    }

    free(q);
    free(anchors_mem);
}

static void test_cutoff_1x8_double_unconstrained(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x8_double (unconstrained cutoff_sq=0)...\n");
    long test_dims[] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
    int num_dims = (int)(sizeof(test_dims) / sizeof(test_dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        long dim = test_dims[idx];
        double *q = (double *)malloc((size_t)dim * sizeof(double));
        double *anchors_mem = (double *)malloc((size_t)8 * (size_t)dim * sizeof(double));
        const double *anchors[8];
        assert(q != NULL && anchors_mem != NULL);

        for (int k = 0; k < 8; k++)
        {
            anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
        }

        for (long i = 0; i < dim; i++)
        {
            q[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
        }

        for (int k = 0; k < 8; k++)
        {
            double *a_ptr = (double *)anchors[k];
            for (long i = 0; i < dim; i++)
            {
                a_ptr[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
            }
        }

        double out_dists[8];
        int mask = framedist_batch_cutoff_1x8_double(q, anchors, dim, 0.0, out_dists);
        assert(mask == 0);

        for (int k = 0; k < 8; k++)
        {
            double ref = ref_dist_double(q, anchors[k], dim);
            double diff = fabs(out_dists[k] - ref);
            assert(diff <= 1e-7);
        }

        free(q);
        free(anchors_mem);
    }
}

static void test_cutoff_1x8_double_mixed(void)
{
    printf("[TEST] Testing framedist_batch_cutoff_1x8_double (mixed cutoff)...\n");
    long dim = 256;
    double *q = (double *)malloc((size_t)dim * sizeof(double));
    double *anchors_mem = (double *)malloc((size_t)8 * (size_t)dim * sizeof(double));
    const double *anchors[8];
    assert(q != NULL && anchors_mem != NULL);

    for (int k = 0; k < 8; k++)
    {
        anchors[k] = anchors_mem + (size_t)k * (size_t)dim;
    }

    for (long i = 0; i < dim; i++)
    {
        q[i] = 0.0;
    }

    for (int k = 0; k < 8; k++)
    {
        double *a_ptr = (double *)anchors[k];
        double val = ((k % 2) == 0) ? 0.5 : 10.0;
        for (long i = 0; i < dim; i++)
        {
            a_ptr[i] = val / sqrt((double)dim);
        }
    }

    double cutoff = 2.0;
    double cutoff_sq = cutoff * cutoff;
    double out_dists[8];
    int mask = framedist_batch_cutoff_1x8_double(q, anchors, dim, cutoff_sq, out_dists);

    for (int k = 0; k < 8; k++)
    {
        if ((k % 2) == 0)
        {
            assert((mask & (1 << k)) == 0);
            double ref = ref_dist_double(q, anchors[k], dim);
            assert(fabs(out_dists[k] - ref) < 1e-7);
        }
        else
        {
            assert((mask & (1 << k)) != 0);
            assert(out_dists[k] > cutoff);
        }
    }

    free(q);
    free(anchors_mem);
}

int main(void)
{
    printf("==================================================\n");
    printf("Running Batched Early-Cutoff Framedist Unit Tests\n");
    printf("==================================================\n");

    test_cutoff_1x4_float_unconstrained();
    test_cutoff_1x4_float_early_exit();
    test_cutoff_1x4_float_mixed();
    test_cutoff_1x4_double_unconstrained();
    test_cutoff_1x4_double_mixed();

    test_cutoff_1x8_float_unconstrained();
    test_cutoff_1x8_float_early_exit();
    test_cutoff_1x8_float_mixed();
    test_cutoff_1x8_double_unconstrained();
    test_cutoff_1x8_double_mixed();

    printf("All Batched Cutoff Framedist Unit Tests PASSED!\n");
    return 0;
}

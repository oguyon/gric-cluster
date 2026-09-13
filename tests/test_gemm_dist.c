/**
 * @file test_gemm_dist.c
 * @brief Unit tests for batched matrix GEMM distance engine.
 */

#include "cluster_gemm_dist.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ref_dist_float(
    const float *a,
    const float *b,
    int          dim)
{
    double sum = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = (double)a[d] - (double)b[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double ref_dist_double(
    const double *a,
    const double *b,
    int           dim)
{
    double sum = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static void test_norms_float(void)
{
    printf("[TEST] Testing cluster_compute_l2_norms_float...\n");
    int count = 25;
    int dims[] = {1, 2, 7, 8, 15, 16, 33, 128, 512};
    int num_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        int dim = dims[idx];
        float *mat = (float *)malloc((size_t)count * (size_t)dim * sizeof(float));
        float *norms = (float *)malloc((size_t)count * sizeof(float));
        assert(mat && norms);

        for (int i = 0; i < count * dim; i++)
        {
            mat[i] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;
        }

        int rc = cluster_compute_l2_norms_float(mat, count, dim, norms);
        assert(rc == 0);

        for (int i = 0; i < count; i++)
        {
            double ref = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double v = (double)mat[i * dim + d];
                ref += v * v;
            }
            double err = fabs((double)norms[i] - ref);
            assert(err < 1e-3);
        }

        free(mat);
        free(norms);
    }
    printf("  -> Passed norms float tests\n");
}

static void test_norms_double(void)
{
    printf("[TEST] Testing cluster_compute_l2_norms_double...\n");
    int count = 25;
    int dims[] = {1, 2, 4, 7, 8, 16, 64};
    int num_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    for (int idx = 0; idx < num_dims; idx++)
    {
        int dim = dims[idx];
        double *mat = (double *)malloc((size_t)count * (size_t)dim * sizeof(double));
        double *norms = (double *)malloc((size_t)count * sizeof(double));
        assert(mat && norms);

        for (int i = 0; i < count * dim; i++)
        {
            mat[i] = (double)rand() / (double)RAND_MAX * 10.0 - 5.0;
        }

        int rc = cluster_compute_l2_norms_double(mat, count, dim, norms);
        assert(rc == 0);

        for (int i = 0; i < count; i++)
        {
            double ref = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double v = mat[i * dim + d];
                ref += v * v;
            }
            double err = fabs(norms[i] - ref);
            assert(err < 1e-9);
        }

        free(mat);
        free(norms);
    }
    printf("  -> Passed norms double tests\n");
}

static void test_matrix_dist_float(
    int M,
    int N,
    int D)
{
    printf("[TEST] Matrix GEMM dist float: M=%d, N=%d, D=%d\n", M, N, D);

    float *Q = (float *)malloc((size_t)M * (size_t)D * sizeof(float));
    float *X = (float *)malloc((size_t)N * (size_t)D * sizeof(float));
    float *q_norms = (float *)malloc((size_t)M * sizeof(float));
    float *x_norms = (float *)malloc((size_t)N * sizeof(float));
    float *dist_sq = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    double *dists = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
    const float **cand_ptrs = (const float **)malloc((size_t)N * sizeof(const float *));

    assert(Q && X && q_norms && x_norms && dist_sq && dists && cand_ptrs);

    for (int i = 0; i < M * D; i++)
    {
        Q[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    for (int j = 0; j < N * D; j++)
    {
        X[j] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    cluster_compute_l2_norms_float(Q, M, D, q_norms);
    cluster_compute_l2_norms_float(X, N, D, x_norms);

    /* 1. Test contiguous matrix */
    int rc = cluster_gemm_dist_float(Q, X, q_norms, x_norms, M, N, D, dist_sq, dists);
    assert(rc == 0);

    double max_err_d = 0.0;
    double max_err_sq = 0.0;

    for (int i = 0; i < M; i++)
    {
        const float *q_ptr = Q + (size_t)i * (size_t)D;
        for (int j = 0; j < N; j++)
        {
            const float *x_ptr = X + (size_t)j * (size_t)D;
            double ref = ref_dist_float(q_ptr, x_ptr, D);
            double ref_sq = ref * ref;

            double err_d = fabs(dists[i * N + j] - ref);
            double err_sq = fabs((double)dist_sq[i * N + j] - ref_sq);

            if (err_d > max_err_d)
            {
                max_err_d = err_d;
            }
            if (err_sq > max_err_sq)
            {
                max_err_sq = err_sq;
            }

            assert(err_d < 1e-3);
        }
    }

    /* 2. Test ptrs variant */
    memset(dist_sq, 0, (size_t)M * (size_t)N * sizeof(float));
    memset(dists, 0, (size_t)M * (size_t)N * sizeof(double));

    rc = cluster_gemm_dist_ptrs_float(
        Q, cand_ptrs, q_norms, x_norms, M, N, D, dist_sq, dists
    );
    assert(rc == 0);

    for (int i = 0; i < M; i++)
    {
        const float *q_ptr = Q + (size_t)i * (size_t)D;
        for (int j = 0; j < N; j++)
        {
            double ref = ref_dist_float(q_ptr, cand_ptrs[j], D);
            double err_d = fabs(dists[i * N + j] - ref);
            assert(err_d < 1e-3);
        }
    }

    free(Q);
    free(X);
    free(q_norms);
    free(x_norms);
    free(dist_sq);
    free(dists);
    free(cand_ptrs);

    printf("  -> Max err: dist=%.2e, dist_sq=%.2e (OK)\n", max_err_d, max_err_sq);
}

static void test_matrix_dist_double(
    int M,
    int N,
    int D)
{
    printf("[TEST] Matrix GEMM dist double: M=%d, N=%d, D=%d\n", M, N, D);

    double *Q = (double *)malloc((size_t)M * (size_t)D * sizeof(double));
    double *X = (double *)malloc((size_t)N * (size_t)D * sizeof(double));
    double *q_norms = (double *)malloc((size_t)M * sizeof(double));
    double *x_norms = (double *)malloc((size_t)N * sizeof(double));
    double *dist_sq = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
    double *dists = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
    const double **cand_ptrs = (const double **)malloc((size_t)N * sizeof(const double *));

    assert(Q && X && q_norms && x_norms && dist_sq && dists && cand_ptrs);

    for (int i = 0; i < M * D; i++)
    {
        Q[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    }
    for (int j = 0; j < N * D; j++)
    {
        X[j] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    }
    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    cluster_compute_l2_norms_double(Q, M, D, q_norms);
    cluster_compute_l2_norms_double(X, N, D, x_norms);

    int rc = cluster_gemm_dist_double(Q, X, q_norms, x_norms, M, N, D, dist_sq, dists);
    assert(rc == 0);

    double max_err_d = 0.0;

    for (int i = 0; i < M; i++)
    {
        const double *q_ptr = Q + (size_t)i * (size_t)D;
        for (int j = 0; j < N; j++)
        {
            const double *x_ptr = X + (size_t)j * (size_t)D;
            double ref = ref_dist_double(q_ptr, x_ptr, D);
            double err_d = fabs(dists[i * N + j] - ref);

            if (err_d > max_err_d)
            {
                max_err_d = err_d;
            }
            assert(err_d < 1e-9);
        }
    }

    free(Q);
    free(X);
    free(q_norms);
    free(x_norms);
    free(dist_sq);
    free(dists);
    free(cand_ptrs);

    printf("  -> Max err double: %.2e (OK)\n", max_err_d);
}

static void test_self_distance_zero(void)
{
    printf("[TEST] Testing self-distance zero and clamping...\n");
    int N = 32;
    int D = 256;

    float *X = (float *)malloc((size_t)N * (size_t)D * sizeof(float));
    float *norms = (float *)malloc((size_t)N * sizeof(float));
    float *dist_sq = (float *)malloc((size_t)N * (size_t)N * sizeof(float));
    double *dists = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
    assert(X && norms && dist_sq && dists);

    for (int i = 0; i < N * D; i++)
    {
        X[i] = (float)rand() / (float)RAND_MAX * 100.0f;
    }

    cluster_compute_l2_norms_float(X, N, D, norms);
    int rc = cluster_gemm_dist_float(X, X, norms, norms, N, N, D, dist_sq, dists);
    assert(rc == 0);

    for (int i = 0; i < N; i++)
    {
        /* Relative error against squared norm (~8.5e5) must be < 1e-5 */
        assert(dist_sq[i * N + i] >= 0.0f);
        assert(dist_sq[i * N + i] < 1.0f);
        assert(dists[i * N + i] >= 0.0);
        assert(dists[i * N + i] < 1.0);
    }

    /* Test normalized vectors in [-1, 1] */
    for (int i = 0; i < N * D; i++)
    {
        X[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    cluster_compute_l2_norms_float(X, N, D, norms);
    rc = cluster_gemm_dist_float(X, X, norms, norms, N, N, D, dist_sq, dists);
    assert(rc == 0);

    for (int i = 0; i < N; i++)
    {
        assert(dist_sq[i * N + i] >= 0.0f);
        assert(dist_sq[i * N + i] < 1e-4f);
        assert(dists[i * N + i] >= 0.0);
        assert(dists[i * N + i] < 1e-2);
    }

    free(X);
    free(norms);
    free(dist_sq);
    free(dists);
    printf("  -> Passed self-distance zero tests\n");
}

int main(void)
{
    printf("=== Running Batched GEMM Distance Unit Tests ===\n");

    test_norms_float();
    test_norms_double();

    /* Test square shapes */
    test_matrix_dist_float(64, 64, 1024);
    test_matrix_dist_float(32, 32, 256);

    /* Test non-multiple-of-4 / odd shapes */
    test_matrix_dist_float(17, 23, 37);
    test_matrix_dist_float(1, 1, 128);
    test_matrix_dist_float(3, 5, 2);
    test_matrix_dist_float(64, 4, 1024);
    test_matrix_dist_float(4, 64, 1024);

    /* Test double precision */
    test_matrix_dist_double(16, 16, 128);
    test_matrix_dist_double(7, 11, 23);

    /* Test self distance zero clamping */
    test_self_distance_zero();

    printf("\nAll GEMM distance tests PASSED successfully!\n");
    return 0;
}

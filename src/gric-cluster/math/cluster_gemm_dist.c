/**
 * @file cluster_gemm_dist.c
 * @brief High-throughput batched matrix Euclidean distance engine (BLAS Level 3).
 *
 * Implements the FAISS-style matrix multiplication distance formulation:
 *   ||Q_i - X_j||^2 = ||Q_i||^2 + ||X_j||^2 - 2 * (Q * X^T)_ij
 * using an in-tree register-tiled fused micro-kernel (AVX2/AVX-512) and optional
 * external BLAS (cblas_sgemm/cblas_dgemm).
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_gemm_dist.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#ifdef USE_BLAS
#include <cblas.h>
#endif

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline float hadd_m256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(s, s);
    __m128 squad = _mm_add_ps(s, shuf);
    shuf = _mm_shuffle_ps(squad, squad, 1);
    return _mm_cvtss_f32(_mm_add_ss(squad, shuf));
}

static inline double hadd_m256d_pd(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
#endif

/**
 * cluster_compute_l2_norms_float() - Precompute squared L2 norms of float vectors.
 * @mat:       Contiguous input matrix [count x dim].
 * @count:     Number of vectors in matrix.
 * @dim:       Dimensionality of each vector.
 * @out_norms: Output array of size count receiving squared L2 norms.
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_compute_l2_norms_float(
    const float *restrict mat,
    int                   count,
    int                   dim,
    float       *restrict out_norms)
{
    if (mat == NULL || out_norms == NULL || count <= 0 || dim <= 0)
    {
        return -1;
    }

    for (int i = 0; i < count; i++)
    {
        const float *v = mat + (size_t)i * (size_t)dim;
        float sum = 0.0f;
        int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        if (dim >= 8)
        {
            __m256 acc = _mm256_setzero_ps();
            for (; d <= dim - 8; d += 8)
            {
                __m256 val = _mm256_loadu_ps(&v[d]);
#ifdef __FMA__
                acc = _mm256_fmadd_ps(val, val, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(val, val));
#endif
            }
            sum += hadd_m256_ps(acc);
        }
#endif

        for (; d < dim; d++)
        {
            sum += v[d] * v[d];
        }

        out_norms[i] = sum;
    } // for (int i = 0; i < count; i++)

    return 0;
}

/**
 * cluster_compute_l2_norms_double() - Precompute squared L2 norms of double vectors.
 * @mat:       Contiguous input matrix [count x dim].
 * @count:     Number of vectors in matrix.
 * @dim:       Dimensionality of each vector.
 * @out_norms: Output array of size count receiving squared L2 norms.
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_compute_l2_norms_double(
    const double *restrict mat,
    int                    count,
    int                    dim,
    double       *restrict out_norms)
{
    if (mat == NULL || out_norms == NULL || count <= 0 || dim <= 0)
    {
        return -1;
    }

    for (int i = 0; i < count; i++)
    {
        const double *v = mat + (size_t)i * (size_t)dim;
        double sum = 0.0;
        int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        if (dim >= 4)
        {
            __m256d acc = _mm256_setzero_pd();
            for (; d <= dim - 4; d += 4)
            {
                __m256d val = _mm256_loadu_pd(&v[d]);
#ifdef __FMA__
                acc = _mm256_fmadd_pd(val, val, acc);
#else
                acc = _mm256_add_pd(acc, _mm256_mul_pd(val, val));
#endif
            }
            sum += hadd_m256d_pd(acc);
        }
#endif

        for (; d < dim; d++)
        {
            sum += v[d] * v[d];
        }

        out_norms[i] = sum;
    } // for (int i = 0; i < count; i++)

    return 0;
}

/**
 * gemm_dist_scalar_float() - Compute single vector-to-vector distance with norms.
 */
static inline void gemm_dist_scalar_float(
    const float *restrict q,
    const float *restrict x,
    float                 qn,
    float                 xn,
    int                   dim,
    float       *restrict out_sq,
    double      *restrict out_d)
{
    float dot = 0.0f;
    int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (dim >= 8)
    {
        __m256 acc = _mm256_setzero_ps();
        for (; d <= dim - 8; d += 8)
        {
            __m256 vq = _mm256_loadu_ps(&q[d]);
            __m256 vx = _mm256_loadu_ps(&x[d]);
#ifdef __FMA__
            acc = _mm256_fmadd_ps(vq, vx, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vq, vx));
#endif
        }
        dot += hadd_m256_ps(acc);
    }
#endif

    for (; d < dim; d++)
    {
        dot += q[d] * x[d];
    }

    float d2 = qn + xn - 2.0f * dot;
    if (d2 < 0.0f)
    {
        d2 = 0.0f;
    }

    if (out_sq != NULL)
    {
        *out_sq = d2;
    }
    if (out_d != NULL)
    {
        *out_d = (double)sqrtf(d2);
    }
}

/**
 * gemm_dist_scalar_double() - Compute single vector-to-vector distance with norms.
 */
static inline void gemm_dist_scalar_double(
    const double *restrict q,
    const double *restrict x,
    double                 qn,
    double                 xn,
    int                    dim,
    double       *restrict out_sq,
    double       *restrict out_d)
{
    double dot = 0.0;
    int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (dim >= 4)
    {
        __m256d acc = _mm256_setzero_pd();
        for (; d <= dim - 4; d += 4)
        {
            __m256d vq = _mm256_loadu_pd(&q[d]);
            __m256d vx = _mm256_loadu_pd(&x[d]);
#ifdef __FMA__
            acc = _mm256_fmadd_pd(vq, vx, acc);
#else
            acc = _mm256_add_pd(acc, _mm256_mul_pd(vq, vx));
#endif
        }
        dot += hadd_m256d_pd(acc);
    }
#endif

    for (; d < dim; d++)
    {
        dot += q[d] * x[d];
    }

    double d2 = qn + xn - 2.0 * dot;
    if (d2 < 0.0)
    {
        d2 = 0.0;
    }

    if (out_sq != NULL)
    {
        *out_sq = d2;
    }
    if (out_d != NULL)
    {
        *out_d = sqrt(d2);
    }
}

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * gemm_dist_tile_4x4_float() - 4x4 register-tiled fused distance microkernel.
 */
static void gemm_dist_tile_4x4_float(
    const float *const *restrict q_ptrs,
    const float *const *restrict x_ptrs,
    const float        *restrict qn,
    const float        *restrict xn,
    int                          dim,
    int                          out_stride,
    float              *restrict out_dist_sq,
    double             *restrict out_dists)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps();
    __m256 c03 = _mm256_setzero_ps();

    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c12 = _mm256_setzero_ps();
    __m256 c13 = _mm256_setzero_ps();

    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c22 = _mm256_setzero_ps();
    __m256 c23 = _mm256_setzero_ps();

    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c32 = _mm256_setzero_ps();
    __m256 c33 = _mm256_setzero_ps();

    int d = 0;
    for (; d <= dim - 8; d += 8)
    {
        __m256 vx0 = _mm256_loadu_ps(&x_ptrs[0][d]);
        __m256 vx1 = _mm256_loadu_ps(&x_ptrs[1][d]);
        __m256 vx2 = _mm256_loadu_ps(&x_ptrs[2][d]);
        __m256 vx3 = _mm256_loadu_ps(&x_ptrs[3][d]);

        __m256 vq0 = _mm256_loadu_ps(&q_ptrs[0][d]);
#ifdef __FMA__
        c00 = _mm256_fmadd_ps(vq0, vx0, c00);
        c01 = _mm256_fmadd_ps(vq0, vx1, c01);
        c02 = _mm256_fmadd_ps(vq0, vx2, c02);
        c03 = _mm256_fmadd_ps(vq0, vx3, c03);
#else
        c00 = _mm256_add_ps(c00, _mm256_mul_ps(vq0, vx0));
        c01 = _mm256_add_ps(c01, _mm256_mul_ps(vq0, vx1));
        c02 = _mm256_add_ps(c02, _mm256_mul_ps(vq0, vx2));
        c03 = _mm256_add_ps(c03, _mm256_mul_ps(vq0, vx3));
#endif

        __m256 vq1 = _mm256_loadu_ps(&q_ptrs[1][d]);
#ifdef __FMA__
        c10 = _mm256_fmadd_ps(vq1, vx0, c10);
        c11 = _mm256_fmadd_ps(vq1, vx1, c11);
        c12 = _mm256_fmadd_ps(vq1, vx2, c12);
        c13 = _mm256_fmadd_ps(vq1, vx3, c13);
#else
        c10 = _mm256_add_ps(c10, _mm256_mul_ps(vq1, vx0));
        c11 = _mm256_add_ps(c11, _mm256_mul_ps(vq1, vx1));
        c12 = _mm256_add_ps(c12, _mm256_mul_ps(vq1, vx2));
        c13 = _mm256_add_ps(c13, _mm256_mul_ps(vq1, vx3));
#endif

        __m256 vq2 = _mm256_loadu_ps(&q_ptrs[2][d]);
#ifdef __FMA__
        c20 = _mm256_fmadd_ps(vq2, vx0, c20);
        c21 = _mm256_fmadd_ps(vq2, vx1, c21);
        c22 = _mm256_fmadd_ps(vq2, vx2, c22);
        c23 = _mm256_fmadd_ps(vq2, vx3, c23);
#else
        c20 = _mm256_add_ps(c20, _mm256_mul_ps(vq2, vx0));
        c21 = _mm256_add_ps(c21, _mm256_mul_ps(vq2, vx1));
        c22 = _mm256_add_ps(c22, _mm256_mul_ps(vq2, vx2));
        c23 = _mm256_add_ps(c23, _mm256_mul_ps(vq2, vx3));
#endif

        __m256 vq3 = _mm256_loadu_ps(&q_ptrs[3][d]);
#ifdef __FMA__
        c30 = _mm256_fmadd_ps(vq3, vx0, c30);
        c31 = _mm256_fmadd_ps(vq3, vx1, c31);
        c32 = _mm256_fmadd_ps(vq3, vx2, c32);
        c33 = _mm256_fmadd_ps(vq3, vx3, c33);
#else
        c30 = _mm256_add_ps(c30, _mm256_mul_ps(vq3, vx0));
        c31 = _mm256_add_ps(c31, _mm256_mul_ps(vq3, vx1));
        c32 = _mm256_add_ps(c32, _mm256_mul_ps(vq3, vx2));
        c33 = _mm256_add_ps(c33, _mm256_mul_ps(vq3, vx3));
#endif
    } // for (; d <= dim - 8; d += 8)

    float dots[4][4] = {
        { hadd_m256_ps(c00), hadd_m256_ps(c01), hadd_m256_ps(c02), hadd_m256_ps(c03) },
        { hadd_m256_ps(c10), hadd_m256_ps(c11), hadd_m256_ps(c12), hadd_m256_ps(c13) },
        { hadd_m256_ps(c20), hadd_m256_ps(c21), hadd_m256_ps(c22), hadd_m256_ps(c23) },
        { hadd_m256_ps(c30), hadd_m256_ps(c31), hadd_m256_ps(c32), hadd_m256_ps(c33) }
    };

    /* Handle remainder dimensions for the tile */
    for (; d < dim; d++)
    {
        for (int r = 0; r < 4; r++)
        {
            float qval = q_ptrs[r][d];
            for (int c = 0; c < 4; c++)
            {
                dots[r][c] += qval * x_ptrs[c][d];
            }
        }
    }

    /* Assemble and store distances */
    for (int r = 0; r < 4; r++)
    {
        float q_norm = qn[r];
        float *row_sq = out_dist_sq ? (out_dist_sq + (size_t)r * (size_t)out_stride) : NULL;
        double *row_d = out_dists   ? (out_dists   + (size_t)r * (size_t)out_stride) : NULL;

        for (int c = 0; c < 4; c++)
        {
            float d2 = q_norm + xn[c] - 2.0f * dots[r][c];
            if (d2 < 0.0f)
            {
                d2 = 0.0f;
            }

            if (row_sq != NULL)
            {
                row_sq[c] = d2;
            }
            if (row_d != NULL)
            {
                row_d[c] = (double)sqrtf(d2);
            }
        }
    }
}
#endif // defined(__AVX__)

/**
 * cluster_gemm_dist_ptrs_float() - Batched Euclidean distance with non-contiguous candidates.
 * @Q:           Contiguous query matrix [M x D].
 * @cand_ptrs:   Array of pointers to N candidate vectors, each of length D.
 * @q_norms:     Array of M squared L2 norms for queries (computed if NULL).
 * @x_norms:     Array of N squared L2 norms for candidates (required).
 * @M:           Number of query vectors.
 * @N:           Number of candidate vectors.
 * @D:           Dimensionality of vectors.
 * @out_dist_sq: Optional output matrix [M x N] of squared Euclidean distances (float).
 * @out_dists:   Optional output matrix [M x N] of Euclidean distances (double).
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_gemm_dist_ptrs_float(
    const float *restrict        Q,
    const float *const *restrict cand_ptrs,
    const float *restrict        q_norms,
    const float *restrict        x_norms,
    int                          M,
    int                          N,
    int                          D,
    float       *restrict        out_dist_sq,
    double      *restrict        out_dists)
{
    if (Q == NULL || cand_ptrs == NULL || x_norms == NULL ||
        M <= 0 || N <= 0 || D <= 0 || (out_dist_sq == NULL && out_dists == NULL))
    {
        return -1;
    }

    float *local_q_norms = NULL;
    const float *active_q_norms = q_norms;

    if (active_q_norms == NULL)
    {
        local_q_norms = (float *)malloc((size_t)M * sizeof(float));
        if (local_q_norms == NULL)
        {
            return -1;
        }
        cluster_compute_l2_norms_float(Q, M, D, local_q_norms);
        active_q_norms = local_q_norms;
    }

    int i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    for (; i <= M - 4; i += 4)
    {
        const float *q_ptrs[4] = {
            Q + (size_t)(i + 0) * (size_t)D,
            Q + (size_t)(i + 1) * (size_t)D,
            Q + (size_t)(i + 2) * (size_t)D,
            Q + (size_t)(i + 3) * (size_t)D
        };
        const float cur_qn[4] = {
            active_q_norms[i + 0],
            active_q_norms[i + 1],
            active_q_norms[i + 2],
            active_q_norms[i + 3]
        };

        int j = 0;
        for (; j <= N - 4; j += 4)
        {
            const float *x_sub_ptrs[4] = {
                cand_ptrs[j + 0],
                cand_ptrs[j + 1],
                cand_ptrs[j + 2],
                cand_ptrs[j + 3]
            };
            const float cur_xn[4] = {
                x_norms[j + 0],
                x_norms[j + 1],
                x_norms[j + 2],
                x_norms[j + 3]
            };

            float *sq_dest = out_dist_sq ? (out_dist_sq + (size_t)i * (size_t)N + j) : NULL;
            double *d_dest = out_dists   ? (out_dists   + (size_t)i * (size_t)N + j) : NULL;

            gemm_dist_tile_4x4_float(q_ptrs, x_sub_ptrs, cur_qn, cur_xn, D, N, sq_dest, d_dest);
        } // for (; j <= N - 4; j += 4)

        /* Handle remainder candidate columns */
        for (; j < N; j++)
        {
            for (int r = 0; r < 4; r++)
            {
                size_t out_idx = (size_t)(i + r) * (size_t)N + (size_t)j;
                float *sq_dest = out_dist_sq ? &out_dist_sq[out_idx] : NULL;
                double *d_dest = out_dists   ? &out_dists[out_idx]   : NULL;
                gemm_dist_scalar_float(
                    q_ptrs[r], cand_ptrs[j], cur_qn[r], x_norms[j], D, sq_dest, d_dest
                );
            }
        }
    } // for (; i <= M - 4; i += 4)
#endif

    /* Handle remainder query rows */
    for (; i < M; i++)
    {
        const float *q_vec = Q + (size_t)i * (size_t)D;
        float qn = active_q_norms[i];

        for (int j = 0; j < N; j++)
        {
            size_t out_idx = (size_t)i * (size_t)N + (size_t)j;
            float *sq_dest = out_dist_sq ? &out_dist_sq[out_idx] : NULL;
            double *d_dest = out_dists   ? &out_dists[out_idx]   : NULL;
            gemm_dist_scalar_float(
                q_vec, cand_ptrs[j], qn, x_norms[j], D, sq_dest, d_dest
            );
        }
    } // for (; i < M; i++)

    if (local_q_norms != NULL)
    {
        free(local_q_norms);
    }

    return 0;
}

/**
 * cluster_gemm_dist_float() - Batched Euclidean distance between contiguous matrices.
 * @Q:           Contiguous query matrix [M x D].
 * @X:           Contiguous candidate matrix [N x D].
 * @q_norms:     Array of M squared L2 norms for queries (computed if NULL).
 * @x_norms:     Array of N squared L2 norms for candidates (required).
 * @M:           Number of query vectors.
 * @N:           Number of candidate vectors.
 * @D:           Dimensionality of vectors.
 * @out_dist_sq: Optional output matrix [M x N] of squared Euclidean distances (float).
 * @out_dists:   Optional output matrix [M x N] of Euclidean distances (double).
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_gemm_dist_float(
    const float *restrict Q,
    const float *restrict X,
    const float *restrict q_norms,
    const float *restrict x_norms,
    int                   M,
    int                   N,
    int                   D,
    float       *restrict out_dist_sq,
    double      *restrict out_dists)
{
    if (Q == NULL || X == NULL || x_norms == NULL ||
        M <= 0 || N <= 0 || D <= 0 || (out_dist_sq == NULL && out_dists == NULL))
    {
        return -1;
    }

#ifdef USE_BLAS
    /* When BLAS is available and matrices are large enough, use cblas_sgemm */
    if (M >= 16 && N >= 16)
    {
        float *local_q_norms = NULL;
        const float *active_q_norms = q_norms;

        if (active_q_norms == NULL)
        {
            local_q_norms = (float *)malloc((size_t)M * sizeof(float));
            if (local_q_norms == NULL)
            {
                return -1;
            }
            cluster_compute_l2_norms_float(Q, M, D, local_q_norms);
            active_q_norms = local_q_norms;
        }

        float *dist_sq_buf = out_dist_sq;
        int free_dist_sq = 0;
        if (dist_sq_buf == NULL)
        {
            dist_sq_buf = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
            if (dist_sq_buf == NULL)
            {
                if (local_q_norms != NULL)
                {
                    free(local_q_norms);
                }
                return -1;
            }
            free_dist_sq = 1;
        }

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, D,
            1.0f, Q, D,
            X, D,
            0.0f, dist_sq_buf, N
        );

        /* Broadcast assemble distance matrix */
        for (int i = 0; i < M; i++)
        {
            float qn = active_q_norms[i];
            float *row_sq = dist_sq_buf + (size_t)i * (size_t)N;
            double *row_d = out_dists ? (out_dists + (size_t)i * (size_t)N) : NULL;

            for (int j = 0; j < N; j++)
            {
                float d2 = qn + x_norms[j] - 2.0f * row_sq[j];
                if (d2 < 0.0f)
                {
                    d2 = 0.0f;
                }
                row_sq[j] = d2;
                if (row_d != NULL)
                {
                    row_d[j] = (double)sqrtf(d2);
                }
            }
        } // for (int i = 0; i < M; i++)

        if (free_dist_sq)
        {
            free(dist_sq_buf);
        }
        if (local_q_norms != NULL)
        {
            free(local_q_norms);
        }
        return 0;
    } // if (M >= 16 && N >= 16)
#endif // USE_BLAS

    /* Default in-tree microkernel path via ptrs wrapper */
    const float *stack_ptrs[128];
    const float **cand_ptrs = stack_ptrs;

    if (N > 128)
    {
        cand_ptrs = (const float **)malloc((size_t)N * sizeof(const float *));
        if (cand_ptrs == NULL)
        {
            return -1;
        }
    }

    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    int res = cluster_gemm_dist_ptrs_float(
        Q, cand_ptrs, q_norms, x_norms, M, N, D, out_dist_sq, out_dists
    );

    if (cand_ptrs != stack_ptrs)
    {
        free(cand_ptrs);
    }

    return res;
}

/**
 * cluster_gemm_dist_ptrs_double() - Batched Euclidean distance with non-contiguous candidates.
 * @Q:           Contiguous query matrix [M x D].
 * @cand_ptrs:   Array of pointers to N candidate vectors, each of length D.
 * @q_norms:     Array of M squared L2 norms for queries (computed if NULL).
 * @x_norms:     Array of N squared L2 norms for candidates (required).
 * @M:           Number of query vectors.
 * @N:           Number of candidate vectors.
 * @D:           Dimensionality of vectors.
 * @out_dist_sq: Optional output matrix [M x N] of squared Euclidean distances (double).
 * @out_dists:   Optional output matrix [M x N] of Euclidean distances (double).
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_gemm_dist_ptrs_double(
    const double *restrict        Q,
    const double *const *restrict cand_ptrs,
    const double *restrict        q_norms,
    const double *restrict        x_norms,
    int                           M,
    int                           N,
    int                           D,
    double       *restrict        out_dist_sq,
    double       *restrict        out_dists)
{
    if (Q == NULL || cand_ptrs == NULL || x_norms == NULL ||
        M <= 0 || N <= 0 || D <= 0 || (out_dist_sq == NULL && out_dists == NULL))
    {
        return -1;
    }

    double *local_q_norms = NULL;
    const double *active_q_norms = q_norms;

    if (active_q_norms == NULL)
    {
        local_q_norms = (double *)malloc((size_t)M * sizeof(double));
        if (local_q_norms == NULL)
        {
            return -1;
        }
        cluster_compute_l2_norms_double(Q, M, D, local_q_norms);
        active_q_norms = local_q_norms;
    }

    for (int i = 0; i < M; i++)
    {
        const double *q_vec = Q + (size_t)i * (size_t)D;
        double qn = active_q_norms[i];

        for (int j = 0; j < N; j++)
        {
            size_t out_idx = (size_t)i * (size_t)N + (size_t)j;
            double *sq_dest = out_dist_sq ? &out_dist_sq[out_idx] : NULL;
            double *d_dest  = out_dists   ? &out_dists[out_idx]   : NULL;
            gemm_dist_scalar_double(
                q_vec, cand_ptrs[j], qn, x_norms[j], D, sq_dest, d_dest
            );
        }
    } // for (int i = 0; i < M; i++)

    if (local_q_norms != NULL)
    {
        free(local_q_norms);
    }

    return 0;
}

/**
 * cluster_gemm_dist_double() - Batched Euclidean distance between contiguous matrices.
 * @Q:           Contiguous query matrix [M x D].
 * @X:           Contiguous candidate matrix [N x D].
 * @q_norms:     Array of M squared L2 norms for queries (computed if NULL).
 * @x_norms:     Array of N squared L2 norms for candidates (required).
 * @M:           Number of query vectors.
 * @N:           Number of candidate vectors.
 * @D:           Dimensionality of vectors.
 * @out_dist_sq: Optional output matrix [M x N] of squared Euclidean distances (double).
 * @out_dists:   Optional output matrix [M x N] of Euclidean distances (double).
 *
 * Return: 0 on success, -1 on invalid argument.
 */
int cluster_gemm_dist_double(
    const double *restrict Q,
    const double *restrict X,
    const double *restrict q_norms,
    const double *restrict x_norms,
    int                    M,
    int                    N,
    int                    D,
    double       *restrict out_dist_sq,
    double       *restrict out_dists)
{
    if (Q == NULL || X == NULL || x_norms == NULL ||
        M <= 0 || N <= 0 || D <= 0 || (out_dist_sq == NULL && out_dists == NULL))
    {
        return -1;
    }

#ifdef USE_BLAS
    if (M >= 16 && N >= 16)
    {
        double *local_q_norms = NULL;
        const double *active_q_norms = q_norms;

        if (active_q_norms == NULL)
        {
            local_q_norms = (double *)malloc((size_t)M * sizeof(double));
            if (local_q_norms == NULL)
            {
                return -1;
            }
            cluster_compute_l2_norms_double(Q, M, D, local_q_norms);
            active_q_norms = local_q_norms;
        }

        double *dist_sq_buf = out_dist_sq;
        int free_dist_sq = 0;
        if (dist_sq_buf == NULL)
        {
            dist_sq_buf = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
            if (dist_sq_buf == NULL)
            {
                if (local_q_norms != NULL)
                {
                    free(local_q_norms);
                }
                return -1;
            }
            free_dist_sq = 1;
        }

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, D,
            1.0, Q, D,
            X, D,
            0.0, dist_sq_buf, N
        );

        for (int i = 0; i < M; i++)
        {
            double qn = active_q_norms[i];
            double *row_sq = dist_sq_buf + (size_t)i * (size_t)N;
            double *row_d = out_dists ? (out_dists + (size_t)i * (size_t)N) : NULL;

            for (int j = 0; j < N; j++)
            {
                double d2 = qn + x_norms[j] - 2.0 * row_sq[j];
                if (d2 < 0.0)
                {
                    d2 = 0.0;
                }
                row_sq[j] = d2;
                if (row_d != NULL)
                {
                    row_d[j] = sqrt(d2);
                }
            }
        } // for (int i = 0; i < M; i++)

        if (free_dist_sq)
        {
            free(dist_sq_buf);
        }
        if (local_q_norms != NULL)
        {
            free(local_q_norms);
        }
        return 0;
    } // if (M >= 16 && N >= 16)
#endif // USE_BLAS

    const double *stack_ptrs[128];
    const double **cand_ptrs = stack_ptrs;

    if (N > 128)
    {
        cand_ptrs = (const double **)malloc((size_t)N * sizeof(const double *));
        if (cand_ptrs == NULL)
        {
            return -1;
        }
    }

    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    int res = cluster_gemm_dist_ptrs_double(
        Q, cand_ptrs, q_norms, x_norms, M, N, D, out_dist_sq, out_dists
    );

    if (cand_ptrs != stack_ptrs)
    {
        free(cand_ptrs);
    }

    return res;
}

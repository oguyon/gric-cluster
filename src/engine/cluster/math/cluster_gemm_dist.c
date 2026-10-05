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
#include "gric_simd.h"
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

#if (defined(__AVX__) || GRIC_HAVE_AVX512_TARGET) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * hadd_m256_ps() - Horizontal sum of 8 float lanes in a 256-bit AVX register.
 * @v: 256-bit vector holding 8 single-precision floats.
 *
 * Purpose & Context ("What is this used for?"):
 * Static helper used in AVX2 inner loops to reduce accumulator vectors into a single scalar sum.
 *
 * Return: Sum of all 8 vector lanes.
 */
GRIC_TARGET_AVX2
static inline float hadd_m256_ps(
    __m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(s, s);
    __m128 squad = _mm_add_ps(s, shuf);
    shuf = _mm_shuffle_ps(squad, squad, 1);
    return _mm_cvtss_f32(_mm_add_ss(squad, shuf));
}

/**
 * hadd_m256d_pd() - Horizontal sum of 4 double lanes in a 256-bit AVX register.
 * @v: 256-bit vector holding 4 double-precision floats.
 *
 * Purpose & Context ("What is this used for?"):
 * Static helper used in AVX2 double-precision inner loops to reduce accumulator vectors.
 *
 * Return: Sum of all 4 vector lanes.
 */
GRIC_TARGET_AVX2
static inline double hadd_m256d_pd(
    __m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
#endif

#if GRIC_HAVE_AVX512_TARGET
/**
 * compute_l2_norms_float_avx512() - Compute vector L2 squared norms using AVX-512.
 * @mat:       Pointer to row-major float matrix [count x dim].
 * @count:     Number of vectors.
 * @dim:       Vector dimensionality.
 * @out_norms: Output array [count] populated with ||v||^2 values.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked by cluster_compute_l2_norms_float() on AVX-512 hardware to precompute squared
 * vector norms for GEMM distance expansion (||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>).
 *
 * Return: 0 on success.
 */
GRIC_TARGET_AVX512
static int compute_l2_norms_float_avx512(
    const float *restrict mat,
    int                   count,
    int                   dim,
    float       *restrict out_norms)
{
    for (int i = 0; i < count; i++)
    {
        const float *v = mat + (size_t)i * (size_t)dim;
        float sum = 0.0f;
        int d = 0;

        if (dim >= 64)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            for (; d <= dim - 64; d += 64)
            {
                __m512 v0 = _mm512_loadu_ps(&v[d]);
                __m512 v1 = _mm512_loadu_ps(&v[d + 16]);
                __m512 v2 = _mm512_loadu_ps(&v[d + 32]);
                __m512 v3 = _mm512_loadu_ps(&v[d + 48]);
                acc0 = _mm512_fmadd_ps(v0, v0, acc0);
                acc1 = _mm512_fmadd_ps(v1, v1, acc1);
                acc2 = _mm512_fmadd_ps(v2, v2, acc2);
                acc3 = _mm512_fmadd_ps(v3, v3, acc3);
            }
            __m512 s01 = _mm512_add_ps(acc0, acc1);
            __m512 s23 = _mm512_add_ps(acc2, acc3);
            sum += _mm512_reduce_add_ps(_mm512_add_ps(s01, s23));
        }
        for (; d <= dim - 16; d += 16)
        {
            __m512 v0 = _mm512_loadu_ps(&v[d]);
            sum += _mm512_reduce_add_ps(_mm512_mul_ps(v0, v0));
        }
        for (; d < dim; d++)
        {
            sum += v[d] * v[d];
        }
        out_norms[i] = sum;
    }
    return 0;
}

/**
 * compute_l2_norms_double_avx512() - Compute double vector L2 squared norms using AVX-512.
 * @mat:       Pointer to row-major double matrix [count x dim].
 * @count:     Number of vectors.
 * @dim:       Vector dimensionality.
 * @out_norms: Output array [count] populated with ||v||^2 values.
 *
 * Purpose & Context ("What is this used for?"):
 * Double-precision counterpart to compute_l2_norms_float_avx512().
 *
 * Return: 0 on success.
 */
GRIC_TARGET_AVX512
static int compute_l2_norms_double_avx512(
    const double *restrict mat,
    int                    count,
    int                    dim,
    double       *restrict out_norms)
{
    for (int i = 0; i < count; i++)
    {
        const double *v = mat + (size_t)i * (size_t)dim;
        double sum = 0.0;
        int d = 0;

        if (dim >= 32)
        {
            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();
            __m512d acc2 = _mm512_setzero_pd();
            __m512d acc3 = _mm512_setzero_pd();
            for (; d <= dim - 32; d += 32)
            {
                __m512d v0 = _mm512_loadu_pd(&v[d]);
                __m512d v1 = _mm512_loadu_pd(&v[d + 8]);
                __m512d v2 = _mm512_loadu_pd(&v[d + 16]);
                __m512d v3 = _mm512_loadu_pd(&v[d + 24]);
                acc0 = _mm512_fmadd_pd(v0, v0, acc0);
                acc1 = _mm512_fmadd_pd(v1, v1, acc1);
                acc2 = _mm512_fmadd_pd(v2, v2, acc2);
                acc3 = _mm512_fmadd_pd(v3, v3, acc3);
            }
            __m512d s01 = _mm512_add_pd(acc0, acc1);
            __m512d s23 = _mm512_add_pd(acc2, acc3);
            sum += _mm512_reduce_add_pd(_mm512_add_pd(s01, s23));
        }
        for (; d <= dim - 8; d += 8)
        {
            __m512d v0 = _mm512_loadu_pd(&v[d]);
            sum += _mm512_reduce_add_pd(_mm512_mul_pd(v0, v0));
        }
        for (; d < dim; d++)
        {
            sum += v[d] * v[d];
        }
        out_norms[i] = sum;
    }
    return 0;
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

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 64)
    {
        return compute_l2_norms_float_avx512(mat, count, dim, out_norms);
    }
#endif

    for (int i = 0; i < count; i++)
    {
        const float *v = mat + (size_t)i * (size_t)dim;
        float sum = 0.0f;
        int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        if (dim >= 32)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (; d <= dim - 32; d += 32)
            {
                __m256 val0 = _mm256_loadu_ps(&v[d]);
                __m256 val1 = _mm256_loadu_ps(&v[d + 8]);
                __m256 val2 = _mm256_loadu_ps(&v[d + 16]);
                __m256 val3 = _mm256_loadu_ps(&v[d + 24]);
#ifdef __FMA__
                acc0 = _mm256_fmadd_ps(val0, val0, acc0);
                acc1 = _mm256_fmadd_ps(val1, val1, acc1);
                acc2 = _mm256_fmadd_ps(val2, val2, acc2);
                acc3 = _mm256_fmadd_ps(val3, val3, acc3);
#else
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(val0, val0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(val1, val1));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(val2, val2));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(val3, val3));
#endif
            }
            __m256 sum01 = _mm256_add_ps(acc0, acc1);
            __m256 sum23 = _mm256_add_ps(acc2, acc3);
            sum += hadd_m256_ps(_mm256_add_ps(sum01, sum23));
        }
        for (; d <= dim - 8; d += 8)
        {
            __m256 val = _mm256_loadu_ps(&v[d]);
#ifdef __FMA__
            __m256 a = _mm256_fmadd_ps(val, val, _mm256_setzero_ps());
#else
            __m256 a = _mm256_mul_ps(val, val);
#endif
            sum += hadd_m256_ps(a);
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

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 32)
    {
        return compute_l2_norms_double_avx512(mat, count, dim, out_norms);
    }
#endif

    for (int i = 0; i < count; i++)
    {
        const double *v = mat + (size_t)i * (size_t)dim;
        double sum = 0.0;
        int d = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        if (dim >= 16)
        {
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd();
            __m256d acc3 = _mm256_setzero_pd();
            for (; d <= dim - 16; d += 16)
            {
                __m256d val0 = _mm256_loadu_pd(&v[d]);
                __m256d val1 = _mm256_loadu_pd(&v[d + 4]);
                __m256d val2 = _mm256_loadu_pd(&v[d + 8]);
                __m256d val3 = _mm256_loadu_pd(&v[d + 12]);
#ifdef __FMA__
                acc0 = _mm256_fmadd_pd(val0, val0, acc0);
                acc1 = _mm256_fmadd_pd(val1, val1, acc1);
                acc2 = _mm256_fmadd_pd(val2, val2, acc2);
                acc3 = _mm256_fmadd_pd(val3, val3, acc3);
#else
                acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(val0, val0));
                acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(val1, val1));
                acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(val2, val2));
                acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(val3, val3));
#endif
            }
            __m256d sum01 = _mm256_add_pd(acc0, acc1);
            __m256d sum23 = _mm256_add_pd(acc2, acc3);
            sum += hadd_m256d_pd(_mm256_add_pd(sum01, sum23));
        }
        for (; d <= dim - 4; d += 4)
        {
            __m256d val = _mm256_loadu_pd(&v[d]);
#ifdef __FMA__
            __m256d a = _mm256_fmadd_pd(val, val, _mm256_setzero_pd());
#else
            __m256d a = _mm256_mul_pd(val, val);
#endif
            sum += hadd_m256d_pd(a);
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

#if (defined(__AVX__) || GRIC_HAVE_AVX512_TARGET) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * gemm_dist_tile_4x4_float() - 4x4 register-tiled fused distance microkernel.
 */
GRIC_TARGET_AVX2
/**
 * gemm_dist_tile_4x4_float() - AVX2 4x4 matrix-multiply microkernel for float distances.
 * @queries:     Array of 4 query vector pointers.
 * @anchors:     Array of 4 anchor vector pointers.
 * @q_norms:     Array of 4 precomputed squared query norms.
 * @a_norms:     Array of 4 precomputed squared anchor norms.
 * @dists_sq:    Output 4x4 distance squared matrix.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Register-blocked microkernel evaluating 16 pairwise distances simultaneously via FMA
 * dot products in AVX2 registers.
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
        if (dim >= 1024 && (d & 15) == 0)
        {
            _mm_prefetch((const char *)&x_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[1][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[2][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[3][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[1][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[2][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[3][d + 64], _MM_HINT_T0);
        }

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

    /* Assemble and store distances using vector operations */
    __m128 vxn = _mm_loadu_ps(xn);
    __m128 vzero_f = _mm_setzero_ps();
    __m128 vtwo_f = _mm_set1_ps(2.0f);

    for (int r = 0; r < 4; r++)
    {
        float *row_sq = out_dist_sq ? (out_dist_sq + (size_t)r * (size_t)out_stride) : NULL;
        double *row_d = out_dists   ? (out_dists   + (size_t)r * (size_t)out_stride) : NULL;

        __m128 vqn = _mm_set1_ps(qn[r]);
        __m128 vdot = _mm_loadu_ps(dots[r]);
        __m128 vd2 = _mm_max_ps(vzero_f,
            _mm_sub_ps(_mm_add_ps(vqn, vxn), _mm_mul_ps(vtwo_f, vdot)));

        if (row_sq != NULL)
        {
            _mm_storeu_ps(row_sq, vd2);
        }
        if (row_d != NULL)
        {
            __m128 vsqrt = _mm_sqrt_ps(vd2);
            __m128d d_lo = _mm_cvtps_pd(vsqrt);
            __m128d d_hi = _mm_cvtps_pd(_mm_movehl_ps(vsqrt, vsqrt));
            _mm_storeu_pd(&row_d[0], d_lo);
            _mm_storeu_pd(&row_d[2], d_hi);
        }
    }
}

/**
 * gemm_dist_tile_2x4_double() - 2x4 register-tiled fused distance microkernel (double).
 */
GRIC_TARGET_AVX2
/**
 * gemm_dist_tile_2x4_double() - AVX2 2x4 matrix-multiply microkernel for double distances.
 * @queries:     Array of 2 query vector pointers.
 * @anchors:     Array of 4 anchor vector pointers.
 * @q_norms:     Array of 2 precomputed squared query norms.
 * @a_norms:     Array of 4 precomputed squared anchor norms.
 * @dists_sq:    Output 2x4 distance squared matrix.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Double-precision register-blocked microkernel evaluating 8 pairwise distances simultaneously.
 */
static void gemm_dist_tile_2x4_double(
    const double *const *restrict q_ptrs,
    const double *const *restrict x_ptrs,
    const double        *restrict qn,
    const double        *restrict xn,
    int                           dim,
    int                           out_stride,
    double              *restrict out_dist_sq,
    double              *restrict out_dists)
{
    __m256d c00 = _mm256_setzero_pd();
    __m256d c01 = _mm256_setzero_pd();
    __m256d c02 = _mm256_setzero_pd();
    __m256d c03 = _mm256_setzero_pd();

    __m256d c10 = _mm256_setzero_pd();
    __m256d c11 = _mm256_setzero_pd();
    __m256d c12 = _mm256_setzero_pd();
    __m256d c13 = _mm256_setzero_pd();

    int d = 0;
    for (; d <= dim - 4; d += 4)
    {
        if (dim >= 512 && (d & 7) == 0)
        {
            _mm_prefetch((const char *)&x_ptrs[0][d + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[1][d + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[2][d + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[3][d + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[0][d + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[1][d + 32], _MM_HINT_T0);
        }

        __m256d vx0 = _mm256_loadu_pd(&x_ptrs[0][d]);
        __m256d vx1 = _mm256_loadu_pd(&x_ptrs[1][d]);
        __m256d vx2 = _mm256_loadu_pd(&x_ptrs[2][d]);
        __m256d vx3 = _mm256_loadu_pd(&x_ptrs[3][d]);

        __m256d vq0 = _mm256_loadu_pd(&q_ptrs[0][d]);
#ifdef __FMA__
        c00 = _mm256_fmadd_pd(vq0, vx0, c00);
        c01 = _mm256_fmadd_pd(vq0, vx1, c01);
        c02 = _mm256_fmadd_pd(vq0, vx2, c02);
        c03 = _mm256_fmadd_pd(vq0, vx3, c03);
#else
        c00 = _mm256_add_pd(c00, _mm256_mul_pd(vq0, vx0));
        c01 = _mm256_add_pd(c01, _mm256_mul_pd(vq0, vx1));
        c02 = _mm256_add_pd(c02, _mm256_mul_pd(vq0, vx2));
        c03 = _mm256_add_pd(c03, _mm256_mul_pd(vq0, vx3));
#endif

        __m256d vq1 = _mm256_loadu_pd(&q_ptrs[1][d]);
#ifdef __FMA__
        c10 = _mm256_fmadd_pd(vq1, vx0, c10);
        c11 = _mm256_fmadd_pd(vq1, vx1, c11);
        c12 = _mm256_fmadd_pd(vq1, vx2, c12);
        c13 = _mm256_fmadd_pd(vq1, vx3, c13);
#else
        c10 = _mm256_add_pd(c10, _mm256_mul_pd(vq1, vx0));
        c11 = _mm256_add_pd(c11, _mm256_mul_pd(vq1, vx1));
        c12 = _mm256_add_pd(c12, _mm256_mul_pd(vq1, vx2));
        c13 = _mm256_add_pd(c13, _mm256_mul_pd(vq1, vx3));
#endif
    } // for (; d <= dim - 4; d += 4)

    double dots[2][4] = {
        { hadd_m256d_pd(c00), hadd_m256d_pd(c01), hadd_m256d_pd(c02), hadd_m256d_pd(c03) },
        { hadd_m256d_pd(c10), hadd_m256d_pd(c11), hadd_m256d_pd(c12), hadd_m256d_pd(c13) }
    };

    for (; d < dim; d++)
    {
        for (int r = 0; r < 2; r++)
        {
            double qval = q_ptrs[r][d];
            for (int c = 0; c < 4; c++)
            {
                dots[r][c] += qval * x_ptrs[c][d];
            }
        }
    }

    __m256d vxn = _mm256_loadu_pd(xn);
    __m256d vzero = _mm256_setzero_pd();
    __m256d vtwo = _mm256_set1_pd(2.0);

    for (int r = 0; r < 2; r++)
    {
        double *row_sq = out_dist_sq ? (out_dist_sq + (size_t)r * (size_t)out_stride) : NULL;
        double *row_d  = out_dists   ? (out_dists   + (size_t)r * (size_t)out_stride) : NULL;

        __m256d vqn = _mm256_set1_pd(qn[r]);
        __m256d vdot = _mm256_loadu_pd(dots[r]);
        __m256d vd2 = _mm256_max_pd(vzero,
            _mm256_sub_pd(_mm256_add_pd(vqn, vxn), _mm256_mul_pd(vtwo, vdot)));

        if (row_sq != NULL)
        {
            _mm256_storeu_pd(row_sq, vd2);
        }
        if (row_d != NULL)
        {
            _mm256_storeu_pd(row_d, _mm256_sqrt_pd(vd2));
        }
    }
}

/**
 * gemm_dist_vec_candidates_float_avx2() - 1-query-to-N-candidates distance microkernel (float).
 */
GRIC_TARGET_AVX2
/**
 * gemm_dist_vec_candidates_float_avx2() - AVX2 1-query vs 8-candidate GEMM distance kernel.
 * @query:       Pointer to query float array [dim].
 * @q_norm:      Precomputed squared query norm.
 * @anchors:     Array of 8 candidate anchor pointers.
 * @a_norms:     Array of 8 precomputed squared anchor norms.
 * @dists:       Output array [8] populated with Euclidean distances.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Evaluates distances from 1 query to 8 candidate clusters in parallel using AVX2 FMA.
 */
static void gemm_dist_vec_candidates_float_avx2(
    const float *restrict        q,
    const float *restrict        X,
    const float *const *restrict cand_ptrs,
    float                        qn,
    const float *restrict        x_norms,
    int                          N,
    int                          dim,
    float       *restrict        out_dist_sq,
    double      *restrict        out_dists)
{
    int j = 0;
    for (; j <= N - 8; j += 8)
    {
        const float *x0 = X ? (X + (size_t)(j + 0) * (size_t)dim) : cand_ptrs[j + 0];
        const float *x1 = X ? (X + (size_t)(j + 1) * (size_t)dim) : cand_ptrs[j + 1];
        const float *x2 = X ? (X + (size_t)(j + 2) * (size_t)dim) : cand_ptrs[j + 2];
        const float *x3 = X ? (X + (size_t)(j + 3) * (size_t)dim) : cand_ptrs[j + 3];
        const float *x4 = X ? (X + (size_t)(j + 4) * (size_t)dim) : cand_ptrs[j + 4];
        const float *x5 = X ? (X + (size_t)(j + 5) * (size_t)dim) : cand_ptrs[j + 5];
        const float *x6 = X ? (X + (size_t)(j + 6) * (size_t)dim) : cand_ptrs[j + 6];
        const float *x7 = X ? (X + (size_t)(j + 7) * (size_t)dim) : cand_ptrs[j + 7];

        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        __m256 c4 = _mm256_setzero_ps();
        __m256 c5 = _mm256_setzero_ps();
        __m256 c6 = _mm256_setzero_ps();
        __m256 c7 = _mm256_setzero_ps();

        int d = 0;
        for (; d <= dim - 8; d += 8)
        {
            __m256 vq = _mm256_loadu_ps(&q[d]);
#ifdef __FMA__
            c0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x0[d]), c0);
            c1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x1[d]), c1);
            c2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x2[d]), c2);
            c3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x3[d]), c3);
            c4 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x4[d]), c4);
            c5 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x5[d]), c5);
            c6 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x6[d]), c6);
            c7 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(&x7[d]), c7);
#else
            c0 = _mm256_add_ps(c0, _mm256_mul_ps(vq, _mm256_loadu_ps(&x0[d])));
            c1 = _mm256_add_ps(c1, _mm256_mul_ps(vq, _mm256_loadu_ps(&x1[d])));
            c2 = _mm256_add_ps(c2, _mm256_mul_ps(vq, _mm256_loadu_ps(&x2[d])));
            c3 = _mm256_add_ps(c3, _mm256_mul_ps(vq, _mm256_loadu_ps(&x3[d])));
            c4 = _mm256_add_ps(c4, _mm256_mul_ps(vq, _mm256_loadu_ps(&x4[d])));
            c5 = _mm256_add_ps(c5, _mm256_mul_ps(vq, _mm256_loadu_ps(&x5[d])));
            c6 = _mm256_add_ps(c6, _mm256_mul_ps(vq, _mm256_loadu_ps(&x6[d])));
            c7 = _mm256_add_ps(c7, _mm256_mul_ps(vq, _mm256_loadu_ps(&x7[d])));
#endif
        }

        float dots[8] = {
            hadd_m256_ps(c0), hadd_m256_ps(c1), hadd_m256_ps(c2), hadd_m256_ps(c3),
            hadd_m256_ps(c4), hadd_m256_ps(c5), hadd_m256_ps(c6), hadd_m256_ps(c7)
        };

        for (; d < dim; d++)
        {
            float q_val = q[d];
            dots[0] += q_val * x0[d];
            dots[1] += q_val * x1[d];
            dots[2] += q_val * x2[d];
            dots[3] += q_val * x3[d];
            dots[4] += q_val * x4[d];
            dots[5] += q_val * x5[d];
            dots[6] += q_val * x6[d];
            dots[7] += q_val * x7[d];
        }

        for (int k = 0; k < 8; k++)
        {
            float d2 = qn + x_norms[j + k] - 2.0f * dots[k];
            if (d2 < 0.0f)
            {
                d2 = 0.0f;
            }
            if (out_dist_sq != NULL)
            {
                out_dist_sq[j + k] = d2;
            }
            if (out_dists != NULL)
            {
                out_dists[j + k] = (double)sqrtf(d2);
            }
        }
    } // for (; j <= N - 8; j += 8)

    for (; j < N; j++)
    {
        const float *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_float(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}

/**
 * gemm_dist_vec_candidates_double_avx2() - 1-query-to-N-candidates distance microkernel (double).
 */
GRIC_TARGET_AVX2
/**
 * gemm_dist_vec_candidates_double_avx2() - AVX2 1-query vs 4-candidate double GEMM distance kernel.
 * @query:       Pointer to query double array [dim].
 * @q_norm:      Precomputed squared query norm.
 * @anchors:     Array of 4 candidate anchor pointers.
 * @a_norms:     Array of 4 precomputed squared anchor norms.
 * @dists:       Output array [4] populated with Euclidean distances.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Double-precision counterpart to gemm_dist_vec_candidates_float_avx2().
 */
static void gemm_dist_vec_candidates_double_avx2(
    const double *restrict        q,
    const double *restrict        X,
    const double *const *restrict cand_ptrs,
    double                        qn,
    const double *restrict        x_norms,
    int                           N,
    int                           dim,
    double       *restrict        out_dist_sq,
    double       *restrict        out_dists)
{
    int j = 0;
    for (; j <= N - 4; j += 4)
    {
        const double *x0 = X ? (X + (size_t)(j + 0) * (size_t)dim) : cand_ptrs[j + 0];
        const double *x1 = X ? (X + (size_t)(j + 1) * (size_t)dim) : cand_ptrs[j + 1];
        const double *x2 = X ? (X + (size_t)(j + 2) * (size_t)dim) : cand_ptrs[j + 2];
        const double *x3 = X ? (X + (size_t)(j + 3) * (size_t)dim) : cand_ptrs[j + 3];

        __m256d c0 = _mm256_setzero_pd();
        __m256d c1 = _mm256_setzero_pd();
        __m256d c2 = _mm256_setzero_pd();
        __m256d c3 = _mm256_setzero_pd();

        int d = 0;
        for (; d <= dim - 4; d += 4)
        {
            __m256d vq = _mm256_loadu_pd(&q[d]);
#ifdef __FMA__
            c0 = _mm256_fmadd_pd(vq, _mm256_loadu_pd(&x0[d]), c0);
            c1 = _mm256_fmadd_pd(vq, _mm256_loadu_pd(&x1[d]), c1);
            c2 = _mm256_fmadd_pd(vq, _mm256_loadu_pd(&x2[d]), c2);
            c3 = _mm256_fmadd_pd(vq, _mm256_loadu_pd(&x3[d]), c3);
#else
            c0 = _mm256_add_pd(c0, _mm256_mul_pd(vq, _mm256_loadu_pd(&x0[d])));
            c1 = _mm256_add_pd(c1, _mm256_mul_pd(vq, _mm256_loadu_pd(&x1[d])));
            c2 = _mm256_add_pd(c2, _mm256_mul_pd(vq, _mm256_loadu_pd(&x2[d])));
            c3 = _mm256_add_pd(c3, _mm256_mul_pd(vq, _mm256_loadu_pd(&x3[d])));
#endif
        }

        double dots[4] = {
            hadd_m256d_pd(c0), hadd_m256d_pd(c1), hadd_m256d_pd(c2), hadd_m256d_pd(c3)
        };

        for (; d < dim; d++)
        {
            double q_val = q[d];
            dots[0] += q_val * x0[d];
            dots[1] += q_val * x1[d];
            dots[2] += q_val * x2[d];
            dots[3] += q_val * x3[d];
        }

        for (int k = 0; k < 4; k++)
        {
            double d2 = qn + x_norms[j + k] - 2.0 * dots[k];
            if (d2 < 0.0)
            {
                d2 = 0.0;
            }
            if (out_dist_sq != NULL)
            {
                out_dist_sq[j + k] = d2;
            }
            if (out_dists != NULL)
            {
                out_dists[j + k] = sqrt(d2);
            }
        }
    } // for (; j <= N - 4; j += 4)

    for (; j < N; j++)
    {
        const double *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_double(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}
#endif // defined(__AVX__)

#if GRIC_HAVE_AVX512_TARGET
/**
 * gemm_dist_tile_4x4_float_avx512() - 4x4 register-tiled distance microkernel (AVX-512).
 */
GRIC_TARGET_AVX512
/**
 * gemm_dist_tile_4x4_float_avx512() - AVX-512 4x4 matrix-multiply microkernel for float distances.
 * @queries:     Array of 4 query vector pointers.
 * @anchors:     Array of 4 anchor vector pointers.
 * @q_norms:     Array of 4 precomputed squared query norms.
 * @a_norms:     Array of 4 precomputed squared anchor norms.
 * @dists_sq:    Output 4x4 distance squared matrix.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * AVX-512 512-bit register-blocked microkernel evaluating 16 pairwise distances simultaneously.
 */
static void gemm_dist_tile_4x4_float_avx512(
    const float *const *restrict q_ptrs,
    const float *const *restrict x_ptrs,
    const float        *restrict qn,
    const float        *restrict xn,
    int                          dim,
    int                          out_stride,
    float              *restrict out_dist_sq,
    double             *restrict out_dists)
{
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c02 = _mm512_setzero_ps();
    __m512 c03 = _mm512_setzero_ps();

    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c12 = _mm512_setzero_ps();
    __m512 c13 = _mm512_setzero_ps();

    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c22 = _mm512_setzero_ps();
    __m512 c23 = _mm512_setzero_ps();

    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c32 = _mm512_setzero_ps();
    __m512 c33 = _mm512_setzero_ps();

    int d = 0;
    for (; d <= dim - 32; d += 32)
    {
        if (dim >= 1024 && (d & 63) == 0)
        {
            _mm_prefetch((const char *)&x_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[1][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[2][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[3][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[1][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[2][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[3][d + 64], _MM_HINT_T0);
        }

        /* First 16 elements */
        {
            __m512 vx0 = _mm512_loadu_ps(&x_ptrs[0][d]);
            __m512 vx1 = _mm512_loadu_ps(&x_ptrs[1][d]);
            __m512 vx2 = _mm512_loadu_ps(&x_ptrs[2][d]);
            __m512 vx3 = _mm512_loadu_ps(&x_ptrs[3][d]);

            __m512 vq0 = _mm512_loadu_ps(&q_ptrs[0][d]);
            c00 = _mm512_fmadd_ps(vq0, vx0, c00);
            c01 = _mm512_fmadd_ps(vq0, vx1, c01);
            c02 = _mm512_fmadd_ps(vq0, vx2, c02);
            c03 = _mm512_fmadd_ps(vq0, vx3, c03);

            __m512 vq1 = _mm512_loadu_ps(&q_ptrs[1][d]);
            c10 = _mm512_fmadd_ps(vq1, vx0, c10);
            c11 = _mm512_fmadd_ps(vq1, vx1, c11);
            c12 = _mm512_fmadd_ps(vq1, vx2, c12);
            c13 = _mm512_fmadd_ps(vq1, vx3, c13);

            __m512 vq2 = _mm512_loadu_ps(&q_ptrs[2][d]);
            c20 = _mm512_fmadd_ps(vq2, vx0, c20);
            c21 = _mm512_fmadd_ps(vq2, vx1, c21);
            c22 = _mm512_fmadd_ps(vq2, vx2, c22);
            c23 = _mm512_fmadd_ps(vq2, vx3, c23);

            __m512 vq3 = _mm512_loadu_ps(&q_ptrs[3][d]);
            c30 = _mm512_fmadd_ps(vq3, vx0, c30);
            c31 = _mm512_fmadd_ps(vq3, vx1, c31);
            c32 = _mm512_fmadd_ps(vq3, vx2, c32);
            c33 = _mm512_fmadd_ps(vq3, vx3, c33);
        }

        /* Second 16 elements */
        {
            __m512 vx0 = _mm512_loadu_ps(&x_ptrs[0][d + 16]);
            __m512 vx1 = _mm512_loadu_ps(&x_ptrs[1][d + 16]);
            __m512 vx2 = _mm512_loadu_ps(&x_ptrs[2][d + 16]);
            __m512 vx3 = _mm512_loadu_ps(&x_ptrs[3][d + 16]);

            __m512 vq0 = _mm512_loadu_ps(&q_ptrs[0][d + 16]);
            c00 = _mm512_fmadd_ps(vq0, vx0, c00);
            c01 = _mm512_fmadd_ps(vq0, vx1, c01);
            c02 = _mm512_fmadd_ps(vq0, vx2, c02);
            c03 = _mm512_fmadd_ps(vq0, vx3, c03);

            __m512 vq1 = _mm512_loadu_ps(&q_ptrs[1][d + 16]);
            c10 = _mm512_fmadd_ps(vq1, vx0, c10);
            c11 = _mm512_fmadd_ps(vq1, vx1, c11);
            c12 = _mm512_fmadd_ps(vq1, vx2, c12);
            c13 = _mm512_fmadd_ps(vq1, vx3, c13);

            __m512 vq2 = _mm512_loadu_ps(&q_ptrs[2][d + 16]);
            c20 = _mm512_fmadd_ps(vq2, vx0, c20);
            c21 = _mm512_fmadd_ps(vq2, vx1, c21);
            c22 = _mm512_fmadd_ps(vq2, vx2, c22);
            c23 = _mm512_fmadd_ps(vq2, vx3, c23);

            __m512 vq3 = _mm512_loadu_ps(&q_ptrs[3][d + 16]);
            c30 = _mm512_fmadd_ps(vq3, vx0, c30);
            c31 = _mm512_fmadd_ps(vq3, vx1, c31);
            c32 = _mm512_fmadd_ps(vq3, vx2, c32);
            c33 = _mm512_fmadd_ps(vq3, vx3, c33);
        }
    }

    for (; d <= dim - 16; d += 16)
    {
        __m512 vx0 = _mm512_loadu_ps(&x_ptrs[0][d]);
        __m512 vx1 = _mm512_loadu_ps(&x_ptrs[1][d]);
        __m512 vx2 = _mm512_loadu_ps(&x_ptrs[2][d]);
        __m512 vx3 = _mm512_loadu_ps(&x_ptrs[3][d]);

        __m512 vq0 = _mm512_loadu_ps(&q_ptrs[0][d]);
        c00 = _mm512_fmadd_ps(vq0, vx0, c00);
        c01 = _mm512_fmadd_ps(vq0, vx1, c01);
        c02 = _mm512_fmadd_ps(vq0, vx2, c02);
        c03 = _mm512_fmadd_ps(vq0, vx3, c03);

        __m512 vq1 = _mm512_loadu_ps(&q_ptrs[1][d]);
        c10 = _mm512_fmadd_ps(vq1, vx0, c10);
        c11 = _mm512_fmadd_ps(vq1, vx1, c11);
        c12 = _mm512_fmadd_ps(vq1, vx2, c12);
        c13 = _mm512_fmadd_ps(vq1, vx3, c13);

        __m512 vq2 = _mm512_loadu_ps(&q_ptrs[2][d]);
        c20 = _mm512_fmadd_ps(vq2, vx0, c20);
        c21 = _mm512_fmadd_ps(vq2, vx1, c21);
        c22 = _mm512_fmadd_ps(vq2, vx2, c22);
        c23 = _mm512_fmadd_ps(vq2, vx3, c23);

        __m512 vq3 = _mm512_loadu_ps(&q_ptrs[3][d]);
        c30 = _mm512_fmadd_ps(vq3, vx0, c30);
        c31 = _mm512_fmadd_ps(vq3, vx1, c31);
        c32 = _mm512_fmadd_ps(vq3, vx2, c32);
        c33 = _mm512_fmadd_ps(vq3, vx3, c33);
    }

    float dots[4][4] = {
        { _mm512_reduce_add_ps(c00), _mm512_reduce_add_ps(c01),
          _mm512_reduce_add_ps(c02), _mm512_reduce_add_ps(c03) },
        { _mm512_reduce_add_ps(c10), _mm512_reduce_add_ps(c11),
          _mm512_reduce_add_ps(c12), _mm512_reduce_add_ps(c13) },
        { _mm512_reduce_add_ps(c20), _mm512_reduce_add_ps(c21),
          _mm512_reduce_add_ps(c22), _mm512_reduce_add_ps(c23) },
        { _mm512_reduce_add_ps(c30), _mm512_reduce_add_ps(c31),
          _mm512_reduce_add_ps(c32), _mm512_reduce_add_ps(c33) }
    };

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

    __m128 vxn = _mm_loadu_ps(xn);
    __m128 vzero_f = _mm_setzero_ps();
    __m128 vtwo_f = _mm_set1_ps(2.0f);

    for (int r = 0; r < 4; r++)
    {
        float *row_sq = out_dist_sq ? (out_dist_sq + (size_t)r * (size_t)out_stride) : NULL;
        double *row_d  = out_dists   ? (out_dists   + (size_t)r * (size_t)out_stride) : NULL;

        __m128 vqn = _mm_set1_ps(qn[r]);
        __m128 vdot = _mm_loadu_ps(dots[r]);
        __m128 vd2 = _mm_max_ps(vzero_f,
            _mm_sub_ps(_mm_add_ps(vqn, vxn), _mm_mul_ps(vtwo_f, vdot)));

        if (row_sq != NULL)
        {
            _mm_storeu_ps(row_sq, vd2);
        }
        if (row_d != NULL)
        {
            __m128 vsqrt = _mm_sqrt_ps(vd2);
            __m128d d_lo = _mm_cvtps_pd(vsqrt);
            __m128d d_hi = _mm_cvtps_pd(_mm_movehl_ps(vsqrt, vsqrt));
            _mm_storeu_pd(&row_d[0], d_lo);
            _mm_storeu_pd(&row_d[2], d_hi);
        }
    }
}

/**
 * gemm_dist_tile_2x4_double_avx512() - AVX-512 2x4 matrix-multiply kernel for double distances.
 * @queries:     Array of 2 query vector pointers.
 * @anchors:     Array of 4 anchor vector pointers.
 * @q_norms:     Array of 2 precomputed squared query norms.
 * @a_norms:     Array of 4 precomputed squared anchor norms.
 * @dists_sq:    Output 2x4 distance squared matrix.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Double-precision AVX-512 register-blocked microkernel evaluating 8 pairwise distances.
 */
GRIC_TARGET_AVX512
static void gemm_dist_tile_2x4_double_avx512(
    const double *const *restrict q_ptrs,
    const double *const *restrict x_ptrs,
    const double        *restrict qn,
    const double        *restrict xn,
    int                           dim,
    int                           out_stride,
    double              *restrict out_dist_sq,
    double              *restrict out_dists)
{
    __m512d c00 = _mm512_setzero_pd();
    __m512d c01 = _mm512_setzero_pd();
    __m512d c02 = _mm512_setzero_pd();
    __m512d c03 = _mm512_setzero_pd();

    __m512d c10 = _mm512_setzero_pd();
    __m512d c11 = _mm512_setzero_pd();
    __m512d c12 = _mm512_setzero_pd();
    __m512d c13 = _mm512_setzero_pd();

    int d = 0;
    for (; d <= dim - 16; d += 16)
    {
        if (dim >= 512 && (d & 31) == 0)
        {
            _mm_prefetch((const char *)&x_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[1][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[2][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&x_ptrs[3][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[0][d + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&q_ptrs[1][d + 64], _MM_HINT_T0);
        }

        /* First 8 doubles */
        {
            __m512d vx0 = _mm512_loadu_pd(&x_ptrs[0][d]);
            __m512d vx1 = _mm512_loadu_pd(&x_ptrs[1][d]);
            __m512d vx2 = _mm512_loadu_pd(&x_ptrs[2][d]);
            __m512d vx3 = _mm512_loadu_pd(&x_ptrs[3][d]);

            __m512d vq0 = _mm512_loadu_pd(&q_ptrs[0][d]);
            c00 = _mm512_fmadd_pd(vq0, vx0, c00);
            c01 = _mm512_fmadd_pd(vq0, vx1, c01);
            c02 = _mm512_fmadd_pd(vq0, vx2, c02);
            c03 = _mm512_fmadd_pd(vq0, vx3, c03);

            __m512d vq1 = _mm512_loadu_pd(&q_ptrs[1][d]);
            c10 = _mm512_fmadd_pd(vq1, vx0, c10);
            c11 = _mm512_fmadd_pd(vq1, vx1, c11);
            c12 = _mm512_fmadd_pd(vq1, vx2, c12);
            c13 = _mm512_fmadd_pd(vq1, vx3, c13);
        }

        /* Second 8 doubles */
        {
            __m512d vx0 = _mm512_loadu_pd(&x_ptrs[0][d + 8]);
            __m512d vx1 = _mm512_loadu_pd(&x_ptrs[1][d + 8]);
            __m512d vx2 = _mm512_loadu_pd(&x_ptrs[2][d + 8]);
            __m512d vx3 = _mm512_loadu_pd(&x_ptrs[3][d + 8]);

            __m512d vq0 = _mm512_loadu_pd(&q_ptrs[0][d + 8]);
            c00 = _mm512_fmadd_pd(vq0, vx0, c00);
            c01 = _mm512_fmadd_pd(vq0, vx1, c01);
            c02 = _mm512_fmadd_pd(vq0, vx2, c02);
            c03 = _mm512_fmadd_pd(vq0, vx3, c03);

            __m512d vq1 = _mm512_loadu_pd(&q_ptrs[1][d + 8]);
            c10 = _mm512_fmadd_pd(vq1, vx0, c10);
            c11 = _mm512_fmadd_pd(vq1, vx1, c11);
            c12 = _mm512_fmadd_pd(vq1, vx2, c12);
            c13 = _mm512_fmadd_pd(vq1, vx3, c13);
        }
    }

    for (; d <= dim - 8; d += 8)
    {
        __m512d vx0 = _mm512_loadu_pd(&x_ptrs[0][d]);
        __m512d vx1 = _mm512_loadu_pd(&x_ptrs[1][d]);
        __m512d vx2 = _mm512_loadu_pd(&x_ptrs[2][d]);
        __m512d vx3 = _mm512_loadu_pd(&x_ptrs[3][d]);

        __m512d vq0 = _mm512_loadu_pd(&q_ptrs[0][d]);
        c00 = _mm512_fmadd_pd(vq0, vx0, c00);
        c01 = _mm512_fmadd_pd(vq0, vx1, c01);
        c02 = _mm512_fmadd_pd(vq0, vx2, c02);
        c03 = _mm512_fmadd_pd(vq0, vx3, c03);

        __m512d vq1 = _mm512_loadu_pd(&q_ptrs[1][d]);
        c10 = _mm512_fmadd_pd(vq1, vx0, c10);
        c11 = _mm512_fmadd_pd(vq1, vx1, c11);
        c12 = _mm512_fmadd_pd(vq1, vx2, c12);
        c13 = _mm512_fmadd_pd(vq1, vx3, c13);
    }

    double dots[2][4] = {
        { _mm512_reduce_add_pd(c00), _mm512_reduce_add_pd(c01),
          _mm512_reduce_add_pd(c02), _mm512_reduce_add_pd(c03) },
        { _mm512_reduce_add_pd(c10), _mm512_reduce_add_pd(c11),
          _mm512_reduce_add_pd(c12), _mm512_reduce_add_pd(c13) }
    };

    for (; d < dim; d++)
    {
        for (int r = 0; r < 2; r++)
        {
            double qval = q_ptrs[r][d];
            for (int c = 0; c < 4; c++)
            {
                dots[r][c] += qval * x_ptrs[c][d];
            }
        }
    }

    __m256d vxn = _mm256_loadu_pd(xn);
    __m256d vzero = _mm256_setzero_pd();
    __m256d vtwo = _mm256_set1_pd(2.0);

    for (int r = 0; r < 2; r++)
    {
        double *row_sq = out_dist_sq ? (out_dist_sq + (size_t)r * (size_t)out_stride) : NULL;
        double *row_d  = out_dists   ? (out_dists   + (size_t)r * (size_t)out_stride) : NULL;

        __m256d vqn = _mm256_set1_pd(qn[r]);
        __m256d vdot = _mm256_loadu_pd(dots[r]);
        __m256d vd2 = _mm256_max_pd(vzero,
            _mm256_sub_pd(_mm256_add_pd(vqn, vxn), _mm256_mul_pd(vtwo, vdot)));

        if (row_sq != NULL)
        {
            _mm256_storeu_pd(row_sq, vd2);
        }
        if (row_d != NULL)
        {
            _mm256_storeu_pd(row_d, _mm256_sqrt_pd(vd2));
        }
    }
}

/**
 * gemm_dist_vec_candidates_float_avx512() - Vector-candidate distance (AVX-512, float).
 */
GRIC_TARGET_AVX512
/**
 * gemm_dist_vec_candidates_float_avx512() - AVX-512 1-query vs 16-candidate GEMM distance kernel.
 * @query:       Pointer to query float array [dim].
 * @q_norm:      Precomputed squared query norm.
 * @anchors:     Array of 16 candidate anchor pointers.
 * @a_norms:     Array of 16 precomputed squared anchor norms.
 * @dists:       Output array [16] populated with Euclidean distances.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Evaluates distances from 1 query to 16 candidate clusters simultaneously in AVX-512 registers.
 */
static void gemm_dist_vec_candidates_float_avx512(
    const float *restrict        q,
    const float *restrict        X,
    const float *const *restrict cand_ptrs,
    float                        qn,
    const float *restrict        x_norms,
    int                          N,
    int                          dim,
    float       *restrict        out_dist_sq,
    double      *restrict        out_dists)
{
    int rem_start = (N / 8) * 8;
    for (int j = 0; j < rem_start; j += 8)
    {
        const float *x0 = X ? (X + (size_t)(j + 0) * (size_t)dim) : cand_ptrs[j + 0];
        const float *x1 = X ? (X + (size_t)(j + 1) * (size_t)dim) : cand_ptrs[j + 1];
        const float *x2 = X ? (X + (size_t)(j + 2) * (size_t)dim) : cand_ptrs[j + 2];
        const float *x3 = X ? (X + (size_t)(j + 3) * (size_t)dim) : cand_ptrs[j + 3];
        const float *x4 = X ? (X + (size_t)(j + 4) * (size_t)dim) : cand_ptrs[j + 4];
        const float *x5 = X ? (X + (size_t)(j + 5) * (size_t)dim) : cand_ptrs[j + 5];
        const float *x6 = X ? (X + (size_t)(j + 6) * (size_t)dim) : cand_ptrs[j + 6];
        const float *x7 = X ? (X + (size_t)(j + 7) * (size_t)dim) : cand_ptrs[j + 7];

        __m512 c0 = _mm512_setzero_ps();
        __m512 c1 = _mm512_setzero_ps();
        __m512 c2 = _mm512_setzero_ps();
        __m512 c3 = _mm512_setzero_ps();
        __m512 c4 = _mm512_setzero_ps();
        __m512 c5 = _mm512_setzero_ps();
        __m512 c6 = _mm512_setzero_ps();
        __m512 c7 = _mm512_setzero_ps();

        /* Index d carries over into remainder dimension loop */
        int d = 0;
        for (; d <= dim - 16; d += 16)
        {
            __m512 vq = _mm512_loadu_ps(&q[d]);
            c0 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x0[d]), c0);
            c1 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x1[d]), c1);
            c2 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x2[d]), c2);
            c3 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x3[d]), c3);
            c4 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x4[d]), c4);
            c5 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x5[d]), c5);
            c6 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x6[d]), c6);
            c7 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(&x7[d]), c7);
        }

        float dots[8] = {
            _mm512_reduce_add_ps(c0), _mm512_reduce_add_ps(c1),
            _mm512_reduce_add_ps(c2), _mm512_reduce_add_ps(c3),
            _mm512_reduce_add_ps(c4), _mm512_reduce_add_ps(c5),
            _mm512_reduce_add_ps(c6), _mm512_reduce_add_ps(c7)
        };

        for (; d < dim; d++)
        {
            float q_val = q[d];
            dots[0] += q_val * x0[d];
            dots[1] += q_val * x1[d];
            dots[2] += q_val * x2[d];
            dots[3] += q_val * x3[d];
            dots[4] += q_val * x4[d];
            dots[5] += q_val * x5[d];
            dots[6] += q_val * x6[d];
            dots[7] += q_val * x7[d];
        }

        for (int k = 0; k < 8; k++)
        {
            float d2 = qn + x_norms[j + k] - 2.0f * dots[k];
            if (d2 < 0.0f)
            {
                d2 = 0.0f;
            }
            if (out_dist_sq != NULL)
            {
                out_dist_sq[j + k] = d2;
            }
            if (out_dists != NULL)
            {
                out_dists[j + k] = (double)sqrtf(d2);
            }
        }
    } // for (int j = 0; j < rem_start; j += 8)

    for (int j = rem_start; j < N; j++)
    {
        const float *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_float(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}

/**
 * gemm_dist_vec_candidates_double_avx512() - Vector-candidate distance (AVX-512, double).
 */
GRIC_TARGET_AVX512
/**
 * gemm_dist_vec_candidates_double_avx512() - AVX-512 1-query vs 8-candidate double GEMM kernel.
 * @query:       Pointer to query double array [dim].
 * @q_norm:      Precomputed squared query norm.
 * @anchors:     Array of 8 candidate anchor pointers.
 * @a_norms:     Array of 8 precomputed squared anchor norms.
 * @dists:       Output array [8] populated with Euclidean distances.
 * @dim:         Vector dimensionality.
 *
 * Purpose & Context ("What is this used for?"):
 * Evaluates distances from 1 query to 8 candidate clusters simultaneously in double precision.
 */
static void gemm_dist_vec_candidates_double_avx512(
    const double *restrict        q,
    const double *restrict        X,
    const double *const *restrict cand_ptrs,
    double                        qn,
    const double *restrict        x_norms,
    int                           N,
    int                           dim,
    double       *restrict        out_dist_sq,
    double       *restrict        out_dists)
{
    int rem_start = (N / 4) * 4;
    for (int j = 0; j < rem_start; j += 4)
    {
        const double *x0 = X ? (X + (size_t)(j + 0) * (size_t)dim) : cand_ptrs[j + 0];
        const double *x1 = X ? (X + (size_t)(j + 1) * (size_t)dim) : cand_ptrs[j + 1];
        const double *x2 = X ? (X + (size_t)(j + 2) * (size_t)dim) : cand_ptrs[j + 2];
        const double *x3 = X ? (X + (size_t)(j + 3) * (size_t)dim) : cand_ptrs[j + 3];

        __m512d c0 = _mm512_setzero_pd();
        __m512d c1 = _mm512_setzero_pd();
        __m512d c2 = _mm512_setzero_pd();
        __m512d c3 = _mm512_setzero_pd();

        /* Index d carries over into remainder dimension loop */
        int d = 0;
        for (; d <= dim - 8; d += 8)
        {
            __m512d vq = _mm512_loadu_pd(&q[d]);
            c0 = _mm512_fmadd_pd(vq, _mm512_loadu_pd(&x0[d]), c0);
            c1 = _mm512_fmadd_pd(vq, _mm512_loadu_pd(&x1[d]), c1);
            c2 = _mm512_fmadd_pd(vq, _mm512_loadu_pd(&x2[d]), c2);
            c3 = _mm512_fmadd_pd(vq, _mm512_loadu_pd(&x3[d]), c3);
        }

        double dots[4] = {
            _mm512_reduce_add_pd(c0), _mm512_reduce_add_pd(c1),
            _mm512_reduce_add_pd(c2), _mm512_reduce_add_pd(c3)
        };

        for (; d < dim; d++)
        {
            double q_val = q[d];
            dots[0] += q_val * x0[d];
            dots[1] += q_val * x1[d];
            dots[2] += q_val * x2[d];
            dots[3] += q_val * x3[d];
        }

        for (int k = 0; k < 4; k++)
        {
            double d2 = qn + x_norms[j + k] - 2.0 * dots[k];
            if (d2 < 0.0)
            {
                d2 = 0.0;
            }
            if (out_dist_sq != NULL)
            {
                out_dist_sq[j + k] = d2;
            }
            if (out_dists != NULL)
            {
                out_dists[j + k] = sqrt(d2);
            }
        }
    } // for (int j = 0; j < rem_start; j += 4)

    for (int j = rem_start; j < N; j++)
    {
        const double *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_double(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * gemm_dist_vector_general_float() - Dispatch 1-query-to-N-candidates distance (float).
 */
static void gemm_dist_vector_general_float(
    const float *restrict        q,
    const float *restrict        X,
    const float *const *restrict cand_ptrs,
    float                        qn,
    const float *restrict        x_norms,
    int                          N,
    int                          dim,
    float       *restrict        out_dist_sq,
    double      *restrict        out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 16)
    {
        gemm_dist_vec_candidates_float_avx512(
            q, X, cand_ptrs, qn, x_norms, N, dim, out_dist_sq, out_dists
        );
        return;
    }
#endif
#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (dim >= 8)
    {
        gemm_dist_vec_candidates_float_avx2(
            q, X, cand_ptrs, qn, x_norms, N, dim, out_dist_sq, out_dists
        );
        return;
    }
#endif
    for (int j = 0; j < N; j++)
    {
        const float *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_float(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}

/**
 * gemm_dist_vector_general_double() - Dispatch 1-query-to-N-candidates distance (double).
 */
static void gemm_dist_vector_general_double(
    const double *restrict        q,
    const double *restrict        X,
    const double *const *restrict cand_ptrs,
    double                        qn,
    const double *restrict        x_norms,
    int                           N,
    int                           dim,
    double       *restrict        out_dist_sq,
    double       *restrict        out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 8)
    {
        gemm_dist_vec_candidates_double_avx512(
            q, X, cand_ptrs, qn, x_norms, N, dim, out_dist_sq, out_dists
        );
        return;
    }
#endif
#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (dim >= 4)
    {
        gemm_dist_vec_candidates_double_avx2(
            q, X, cand_ptrs, qn, x_norms, N, dim, out_dist_sq, out_dists
        );
        return;
    }
#endif
    for (int j = 0; j < N; j++)
    {
        const double *xj = X ? (X + (size_t)j * (size_t)dim) : cand_ptrs[j];
        gemm_dist_scalar_double(
            q, xj, qn, x_norms[j], dim,
            out_dist_sq ? &out_dist_sq[j] : NULL,
            out_dists ? &out_dists[j] : NULL
        );
    }
}

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

#if (defined(__AVX__) || GRIC_HAVE_AVX512_TARGET) && \
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

#if GRIC_HAVE_AVX512_TARGET
            if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && D >= 16)
            {
                gemm_dist_tile_4x4_float_avx512(
                    q_ptrs, x_sub_ptrs, cur_qn, cur_xn, D, N, sq_dest, d_dest);
            }
            else
#endif
            {
                gemm_dist_tile_4x4_float(
                    q_ptrs, x_sub_ptrs, cur_qn, cur_xn, D, N, sq_dest, d_dest);
            }
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
        size_t row_off = (size_t)i * (size_t)N;
        float *sq_dest = out_dist_sq ? (out_dist_sq + row_off) : NULL;
        double *d_dest = out_dists   ? (out_dists   + row_off) : NULL;

        gemm_dist_vector_general_float(
            q_vec, NULL, cand_ptrs, qn, x_norms, N, D, sq_dest, d_dest
        );
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
    if (Q == NULL || X == NULL ||
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

    float *local_x_norms = NULL;
    const float *active_x_norms = x_norms;

    if (active_x_norms == NULL)
    {
        local_x_norms = (float *)malloc((size_t)N * sizeof(float));
        if (local_x_norms == NULL)
        {
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            return -1;
        }
        cluster_compute_l2_norms_float(X, N, D, local_x_norms);
        active_x_norms = local_x_norms;
    }

    if (M == 1)
    {
#ifdef USE_BLAS
        if (N >= 16)
        {
            float qn = active_q_norms[0];
            float *dist_sq_buf = out_dist_sq;
            int free_dist_sq = 0;
            if (dist_sq_buf == NULL)
            {
                dist_sq_buf = (float *)malloc((size_t)N * sizeof(float));
                if (dist_sq_buf == NULL)
                {
                    if (local_q_norms != NULL)
                    {
                        free(local_q_norms);
                    }
                    if (local_x_norms != NULL)
                    {
                        free(local_x_norms);
                    }
                    return -1;
                }
                free_dist_sq = 1;
            }

            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                N, D,
                1.0f, X, D,
                Q, 1,
                0.0f, dist_sq_buf, 1
            );

            int j = 0;
#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
            __m256 v_qn = _mm256_set1_ps(qn);
            __m256 vzero = _mm256_setzero_ps();
            __m256 vtwo = _mm256_set1_ps(2.0f);
            for (; j <= N - 8; j += 8)
            {
                __m256 v_xn = _mm256_loadu_ps(&active_x_norms[j]);
                __m256 v_dot = _mm256_loadu_ps(&dist_sq_buf[j]);
#ifdef __FMA__
                __m256 vd2 = _mm256_max_ps(vzero,
                    _mm256_fnmadd_ps(vtwo, v_dot, _mm256_add_ps(v_qn, v_xn)));
#else
                __m256 vd2 = _mm256_max_ps(vzero,
                    _mm256_sub_ps(_mm256_add_ps(v_qn, v_xn), _mm256_mul_ps(vtwo, v_dot)));
#endif
                _mm256_storeu_ps(&dist_sq_buf[j], vd2);
                if (out_dists != NULL)
                {
                    __m256 vsqrt = _mm256_sqrt_ps(vd2);
                    __m128 lo = _mm256_castps256_ps128(vsqrt);
                    __m128 hi = _mm256_extractf128_ps(vsqrt, 1);
                    _mm_storeu_pd(&out_dists[j], _mm_cvtps_pd(lo));
                    _mm_storeu_pd(&out_dists[j + 2], _mm_cvtps_pd(_mm_movehl_ps(lo, lo)));
                    _mm_storeu_pd(&out_dists[j + 4], _mm_cvtps_pd(hi));
                    _mm_storeu_pd(&out_dists[j + 6], _mm_cvtps_pd(_mm_movehl_ps(hi, hi)));
                }
            }
#endif
            for (; j < N; j++)
            {
                float d2 = qn + active_x_norms[j] - 2.0f * dist_sq_buf[j];
                if (d2 < 0.0f)
                {
                    d2 = 0.0f;
                }
                dist_sq_buf[j] = d2;
                if (out_dists != NULL)
                {
                    out_dists[j] = (double)sqrtf(d2);
                }
            }

            if (free_dist_sq)
            {
                free(dist_sq_buf);
            }
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            if (local_x_norms != NULL)
            {
                free(local_x_norms);
            }
            return 0;
        }
#endif // USE_BLAS

        gemm_dist_vector_general_float(
            Q, X, NULL, active_q_norms[0], active_x_norms, N, D, out_dist_sq, out_dists
        );
        if (local_q_norms != NULL)
        {
            free(local_q_norms);
        }
        if (local_x_norms != NULL)
        {
            free(local_x_norms);
        }
        return 0;
    } // if (M == 1)

#ifdef USE_BLAS
    /* When BLAS is available and matrices are large enough, use cblas_sgemm */
    if (M >= 16 && N >= 16)
    {
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
                if (local_x_norms != NULL)
                {
                    free(local_x_norms);
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
            int j = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
            __m256 v_qn = _mm256_set1_ps(qn);
            __m256 vzero = _mm256_setzero_ps();
            __m256 vtwo = _mm256_set1_ps(2.0f);
            for (; j <= N - 8; j += 8)
            {
                __m256 v_xn = _mm256_loadu_ps(&active_x_norms[j]);
                __m256 v_dot = _mm256_loadu_ps(&row_sq[j]);
#ifdef __FMA__
                __m256 vd2 = _mm256_max_ps(vzero,
                    _mm256_fnmadd_ps(vtwo, v_dot, _mm256_add_ps(v_qn, v_xn)));
#else
                __m256 vd2 = _mm256_max_ps(vzero,
                    _mm256_sub_ps(_mm256_add_ps(v_qn, v_xn), _mm256_mul_ps(vtwo, v_dot)));
#endif
                _mm256_storeu_ps(&row_sq[j], vd2);
                if (row_d != NULL)
                {
                    __m256 vsqrt = _mm256_sqrt_ps(vd2);
                    __m128 lo = _mm256_castps256_ps128(vsqrt);
                    __m128 hi = _mm256_extractf128_ps(vsqrt, 1);
                    _mm_storeu_pd(&row_d[j], _mm_cvtps_pd(lo));
                    _mm_storeu_pd(&row_d[j + 2], _mm_cvtps_pd(_mm_movehl_ps(lo, lo)));
                    _mm_storeu_pd(&row_d[j + 4], _mm_cvtps_pd(hi));
                    _mm_storeu_pd(&row_d[j + 6], _mm_cvtps_pd(_mm_movehl_ps(hi, hi)));
                }
            }
#endif

            for (; j < N; j++)
            {
                float d2 = qn + active_x_norms[j] - 2.0f * row_sq[j];
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
        if (local_x_norms != NULL)
        {
            free(local_x_norms);
        }
        return 0;
    } // if (M >= 16 && N >= 16)
#endif // USE_BLAS

    /* Default in-tree microkernel path via ptrs wrapper */
    const float *stack_ptrs[512];
    const float **cand_ptrs = stack_ptrs;

    if (N > 512)
    {
        cand_ptrs = (const float **)malloc((size_t)N * sizeof(const float *));
        if (cand_ptrs == NULL)
        {
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            if (local_x_norms != NULL)
            {
                free(local_x_norms);
            }
            return -1;
        }
    }

    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    int res = cluster_gemm_dist_ptrs_float(
        Q, cand_ptrs, active_q_norms, active_x_norms, M, N, D, out_dist_sq, out_dists
    );

    if (cand_ptrs != stack_ptrs)
    {
        free(cand_ptrs);
    }
    if (local_q_norms != NULL)
    {
        free(local_q_norms);
    }
    if (local_x_norms != NULL)
    {
        free(local_x_norms);
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

#if (defined(__AVX__) || GRIC_HAVE_AVX512_TARGET) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    int i = 0;
    for (; i <= M - 2; i += 2)
    {
        const double *q_block[2] = {
            Q + (size_t)i * (size_t)D,
            Q + (size_t)(i + 1) * (size_t)D
        };
        const double qn_block[2] = {
            active_q_norms[i],
            active_q_norms[i + 1]
        };

        int j = 0;
        for (; j <= N - 4; j += 4)
        {
            size_t row_off = (size_t)i * (size_t)N + (size_t)j;
            double *sq_dest = out_dist_sq ? (out_dist_sq + row_off) : NULL;
            double *d_dest  = out_dists   ? (out_dists   + row_off) : NULL;
#if GRIC_HAVE_AVX512_TARGET
            if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && D >= 8)
            {
                gemm_dist_tile_2x4_double_avx512(
                    q_block, &cand_ptrs[j], qn_block, &x_norms[j], D, N, sq_dest, d_dest
                );
            }
            else
#endif
            {
                gemm_dist_tile_2x4_double(
                    q_block, &cand_ptrs[j], qn_block, &x_norms[j], D, N, sq_dest, d_dest
                );
            }
        }

        for (; j < N; j++)
        {
            for (int r = 0; r < 2; r++)
            {
                size_t out_idx = (size_t)(i + r) * (size_t)N + (size_t)j;
                double *sq_dest = out_dist_sq ? &out_dist_sq[out_idx] : NULL;
                double *d_dest  = out_dists   ? &out_dists[out_idx]   : NULL;
                gemm_dist_scalar_double(
                    q_block[r], cand_ptrs[j], qn_block[r], x_norms[j], D, sq_dest, d_dest
                );
            }
        }
    }

    for (; i < M; i++)
    {
        const double *q_vec = Q + (size_t)i * (size_t)D;
        double qn = active_q_norms[i];
        size_t row_off = (size_t)i * (size_t)N;
        double *sq_dest = out_dist_sq ? (out_dist_sq + row_off) : NULL;
        double *d_dest  = out_dists   ? (out_dists   + row_off) : NULL;

        gemm_dist_vector_general_double(
            q_vec, NULL, cand_ptrs, qn, x_norms, N, D, sq_dest, d_dest
        );
    }
#else
    for (int i = 0; i < M; i++)
    {
        const double *q_vec = Q + (size_t)i * (size_t)D;
        double qn = active_q_norms[i];
        size_t row_off = (size_t)i * (size_t)N;
        double *sq_dest = out_dist_sq ? (out_dist_sq + row_off) : NULL;
        double *d_dest  = out_dists   ? (out_dists   + row_off) : NULL;

        gemm_dist_vector_general_double(
            q_vec, NULL, cand_ptrs, qn, x_norms, N, D, sq_dest, d_dest
        );
    } // for (int i = 0; i < M; i++)
#endif

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
    if (Q == NULL || X == NULL ||
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

    double *local_x_norms = NULL;
    const double *active_x_norms = x_norms;

    if (active_x_norms == NULL)
    {
        local_x_norms = (double *)malloc((size_t)N * sizeof(double));
        if (local_x_norms == NULL)
        {
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            return -1;
        }
        cluster_compute_l2_norms_double(X, N, D, local_x_norms);
        active_x_norms = local_x_norms;
    }

    if (M == 1)
    {
#ifdef USE_BLAS
        if (N >= 16)
        {
            double qn = active_q_norms[0];
            double *dist_sq_buf = out_dist_sq;
            int free_dist_sq = 0;
            if (dist_sq_buf == NULL)
            {
                dist_sq_buf = (double *)malloc((size_t)N * sizeof(double));
                if (dist_sq_buf == NULL)
                {
                    if (local_q_norms != NULL)
                    {
                        free(local_q_norms);
                    }
                    if (local_x_norms != NULL)
                    {
                        free(local_x_norms);
                    }
                    return -1;
                }
                free_dist_sq = 1;
            }

            cblas_dgemv(
                CblasRowMajor, CblasNoTrans,
                N, D,
                1.0, X, D,
                Q, 1,
                0.0, dist_sq_buf, 1
            );

            int j = 0;
#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
            __m256d v_qn = _mm256_set1_pd(qn);
            __m256d vzero = _mm256_setzero_pd();
            __m256d vtwo = _mm256_set1_pd(2.0);
            for (; j <= N - 4; j += 4)
            {
                __m256d v_xn = _mm256_loadu_pd(&active_x_norms[j]);
                __m256d v_dot = _mm256_loadu_pd(&dist_sq_buf[j]);
#ifdef __FMA__
                __m256d vd2 = _mm256_max_pd(vzero,
                    _mm256_fnmadd_pd(vtwo, v_dot, _mm256_add_pd(v_qn, v_xn)));
#else
                __m256d vd2 = _mm256_max_pd(vzero,
                    _mm256_sub_pd(_mm256_add_pd(v_qn, v_xn), _mm256_mul_pd(vtwo, v_dot)));
#endif
                _mm256_storeu_pd(&dist_sq_buf[j], vd2);
                if (out_dists != NULL)
                {
                    _mm256_storeu_pd(&out_dists[j], _mm256_sqrt_pd(vd2));
                }
            }
#endif
            for (; j < N; j++)
            {
                double d2 = qn + active_x_norms[j] - 2.0 * dist_sq_buf[j];
                if (d2 < 0.0)
                {
                    d2 = 0.0;
                }
                dist_sq_buf[j] = d2;
                if (out_dists != NULL)
                {
                    out_dists[j] = sqrt(d2);
                }
            }

            if (free_dist_sq)
            {
                free(dist_sq_buf);
            }
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            if (local_x_norms != NULL)
            {
                free(local_x_norms);
            }
            return 0;
        }
#endif // USE_BLAS

        gemm_dist_vector_general_double(
            Q, X, NULL, active_q_norms[0], active_x_norms, N, D, out_dist_sq, out_dists
        );
        if (local_q_norms != NULL)
        {
            free(local_q_norms);
        }
        if (local_x_norms != NULL)
        {
            free(local_x_norms);
        }
        return 0;
    } // if (M == 1)

#ifdef USE_BLAS
    if (M >= 16 && N >= 16)
    {
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
                if (local_x_norms != NULL)
                {
                    free(local_x_norms);
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
            int j = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
            __m256d v_qn = _mm256_set1_pd(qn);
            __m256d vzero = _mm256_setzero_pd();
            __m256d vtwo = _mm256_set1_pd(2.0);
            for (; j <= N - 4; j += 4)
            {
                __m256d v_xn = _mm256_loadu_pd(&active_x_norms[j]);
                __m256d v_dot = _mm256_loadu_pd(&row_sq[j]);
#ifdef __FMA__
                __m256d vd2 = _mm256_max_pd(vzero,
                    _mm256_fnmadd_pd(vtwo, v_dot, _mm256_add_pd(v_qn, v_xn)));
#else
                __m256d vd2 = _mm256_max_pd(vzero,
                    _mm256_sub_pd(_mm256_add_pd(v_qn, v_xn), _mm256_mul_pd(vtwo, v_dot)));
#endif
                _mm256_storeu_pd(&row_sq[j], vd2);
                if (row_d != NULL)
                {
                    _mm256_storeu_pd(&row_d[j], _mm256_sqrt_pd(vd2));
                }
            }
#endif

            for (; j < N; j++)
            {
                double d2 = qn + active_x_norms[j] - 2.0 * row_sq[j];
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
        if (local_x_norms != NULL)
        {
            free(local_x_norms);
        }
        return 0;
    } // if (M >= 16 && N >= 16)
#endif // USE_BLAS

    const double *stack_ptrs[512];
    const double **cand_ptrs = stack_ptrs;

    if (N > 512)
    {
        cand_ptrs = (const double **)malloc((size_t)N * sizeof(const double *));
        if (cand_ptrs == NULL)
        {
            if (local_q_norms != NULL)
            {
                free(local_q_norms);
            }
            if (local_x_norms != NULL)
            {
                free(local_x_norms);
            }
            return -1;
        }
    }

    for (int j = 0; j < N; j++)
    {
        cand_ptrs[j] = X + (size_t)j * (size_t)D;
    }

    int res = cluster_gemm_dist_ptrs_double(
        Q, cand_ptrs, active_q_norms, active_x_norms, M, N, D, out_dist_sq, out_dists
    );

    if (cand_ptrs != stack_ptrs)
    {
        free(cand_ptrs);
    }
    if (local_q_norms != NULL)
    {
        free(local_q_norms);
    }
    if (local_x_norms != NULL)
    {
        free(local_x_norms);
    }

    return res;
}

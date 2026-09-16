/**
 * @file framedistance_batch.c
 * @brief Batched 1-query to N-anchor Euclidean distance calculations.
 */

#include "framedistance.h"
#include "common.h"
#include "gric_simd.h"
#include <math.h>
#include <stdlib.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * framedist_batch_1x4_float() - Vectorized 1-query vs 4-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @out_dists: Array of 4 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x4_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    long i = 0;

    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];

#if defined(__SSE__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size == 3)
    {
        __m128 vqx = _mm_set1_ps(q[0]);
        __m128 vqy = _mm_set1_ps(q[1]);
        __m128 vqz = _mm_set1_ps(q[2]);
        __m128 va0 = _mm_set_ps(a3[0], a2[0], a1[0], a0[0]);
        __m128 va1 = _mm_set_ps(a3[1], a2[1], a1[1], a0[1]);
        __m128 va2 = _mm_set_ps(a3[2], a2[2], a1[2], a0[2]);
        __m128 dx = _mm_sub_ps(vqx, va0);
        __m128 dy = _mm_sub_ps(vqy, va1);
        __m128 dz = _mm_sub_ps(vqz, va2);
#ifdef __FMA__
        __m128 dsq = _mm_fmadd_ps(dz, dz, _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx)));
#else
        __m128 dsq = _mm_add_ps(_mm_mul_ps(dx, dx),
                                _mm_add_ps(_mm_mul_ps(dy, dy), _mm_mul_ps(dz, dz)));
#endif
        __m128 vd = _mm_sqrt_ps(dsq);
        out_dists[0] = (double)_mm_cvtss_f32(vd);
        out_dists[1] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(1, 1, 1, 1)));
        out_dists[2] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(2, 2, 2, 2)));
        out_dists[3] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(3, 3, 3, 3)));
        return;
    }
    if (size == 2)
    {
        __m128 vqx = _mm_set1_ps(q[0]);
        __m128 vqy = _mm_set1_ps(q[1]);
        __m128 va0 = _mm_set_ps(a3[0], a2[0], a1[0], a0[0]);
        __m128 va1 = _mm_set_ps(a3[1], a2[1], a1[1], a0[1]);
        __m128 dx = _mm_sub_ps(vqx, va0);
        __m128 dy = _mm_sub_ps(vqy, va1);
#ifdef __FMA__
        __m128 dsq = _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx));
#else
        __m128 dsq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
#endif
        __m128 vd = _mm_sqrt_ps(dsq);
        out_dists[0] = (double)_mm_cvtss_f32(vd);
        out_dists[1] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(1, 1, 1, 1)));
        out_dists[2] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(2, 2, 2, 2)));
        out_dists[3] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(3, 3, 3, 3)));
        return;
    }
#endif

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 16)
    {
        __m256 a0_0 = _mm256_setzero_ps();
        __m256 a0_1 = _mm256_setzero_ps();
        __m256 a1_0 = _mm256_setzero_ps();
        __m256 a1_1 = _mm256_setzero_ps();
        __m256 a2_0 = _mm256_setzero_ps();
        __m256 a2_1 = _mm256_setzero_ps();
        __m256 a3_0 = _mm256_setzero_ps();
        __m256 a3_1 = _mm256_setzero_ps();

        for (; i <= size - 16; i += 16)
        {
            /* Chunk 0 (8 floats) -> accumulates into *_0 */
            __m256 vq0 = _mm256_loadu_ps(&q[i]);
            __m256 d0_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a0[i]));
            __m256 d1_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a1[i]));
            __m256 d2_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a2[i]));
            __m256 d3_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_ps(d0_0, d0_0, a0_0);
            a1_0 = _mm256_fmadd_ps(d1_0, d1_0, a1_0);
            a2_0 = _mm256_fmadd_ps(d2_0, d2_0, a2_0);
            a3_0 = _mm256_fmadd_ps(d3_0, d3_0, a3_0);
#else
            a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(d0_0, d0_0));
            a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(d1_0, d1_0));
            a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(d2_0, d2_0));
            a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(d3_0, d3_0));
#endif

            /* Chunk 1 (8 floats) -> accumulates into *_1 */
            __m256 vq1 = _mm256_loadu_ps(&q[i + 8]);
            __m256 d0_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a0[i + 8]));
            __m256 d1_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a1[i + 8]));
            __m256 d2_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a2[i + 8]));
            __m256 d3_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a3[i + 8]));

#ifdef __FMA__
            a0_1 = _mm256_fmadd_ps(d0_1, d0_1, a0_1);
            a1_1 = _mm256_fmadd_ps(d1_1, d1_1, a1_1);
            a2_1 = _mm256_fmadd_ps(d2_1, d2_1, a2_1);
            a3_1 = _mm256_fmadd_ps(d3_1, d3_1, a3_1);
#else
            a0_1 = _mm256_add_ps(a0_1, _mm256_mul_ps(d0_1, d0_1));
            a1_1 = _mm256_add_ps(a1_1, _mm256_mul_ps(d1_1, d1_1));
            a2_1 = _mm256_add_ps(a2_1, _mm256_mul_ps(d2_1, d2_1));
            a3_1 = _mm256_add_ps(a3_1, _mm256_mul_ps(d3_1, d3_1));
#endif
        }

        for (; i <= size - 8; i += 8)
        {
            __m256 vq = _mm256_loadu_ps(&q[i]);
            __m256 d0 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a0[i]));
            __m256 d1 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a1[i]));
            __m256 d2 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a2[i]));
            __m256 d3 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_ps(d0, d0, a0_0);
            a1_0 = _mm256_fmadd_ps(d1, d1, a1_0);
            a2_0 = _mm256_fmadd_ps(d2, d2, a2_0);
            a3_0 = _mm256_fmadd_ps(d3, d3, a3_0);
#else
            a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(d0, d0));
            a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(d1, d1));
            a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(d2, d2));
            a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(d3, d3));
#endif
        }

        __m256 acc0 = _mm256_add_ps(a0_0, a0_1);
        __m256 acc1 = _mm256_add_ps(a1_0, a1_1);
        __m256 acc2 = _mm256_add_ps(a2_0, a2_1);
        __m256 acc3 = _mm256_add_ps(a3_0, a3_1);

        __m128 lo0 = _mm256_castps256_ps128(acc0);
        __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
        __m128 s0 = _mm_add_ps(lo0, hi0);

        __m128 lo1 = _mm256_castps256_ps128(acc1);
        __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1);

        __m128 lo2 = _mm256_castps256_ps128(acc2);
        __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 s2 = _mm_add_ps(lo2, hi2);

        __m128 lo3 = _mm256_castps256_ps128(acc3);
        __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
        __m128 s3 = _mm_add_ps(lo3, hi3);

        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        __m128 sum01 = _mm_add_ps(s0, s1);
        __m128 sum23 = _mm_add_ps(s2, s3);
        __m128 final_sums = _mm_add_ps(sum01, sum23);

        float fsums[4];
        _mm_storeu_ps(fsums, final_sums);
        sum0 += fsums[0];
        sum1 += fsums[1];
        sum2 += fsums[2];
        sum3 += fsums[3];
    }
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i];
        float diff1 = q_val - a1[i];
        float diff2 = q_val - a2[i];
        float diff3 = q_val - a3[i];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    out_dists[0] = (double)sqrtf(sum0);
    out_dists[1] = (double)sqrtf(sum1);
    out_dists[2] = (double)sqrtf(sum2);
    out_dists[3] = (double)sqrtf(sum3);
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static void calc_dist8_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    double              *restrict out_dists,
    long                         size)
{
    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];
    const float *restrict a4 = anchors[4];
    const float *restrict a5 = anchors[5];
    const float *restrict a6 = anchors[6];
    const float *restrict a7 = anchors[7];

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();

    long i = 0;
    for (; i <= size - 16; i += 16)
    {
        if (size >= 1024 && (i & 31) == 0)
        {
            _mm_prefetch((const char *)&q[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a0[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a1[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a2[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a3[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a4[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a5[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a6[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a7[i + 64], _MM_HINT_T0);
        }

        __m512 vq = _mm512_loadu_ps(&q[i]);
        __m512 d0 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a0[i]));
        __m512 d1 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a1[i]));
        __m512 d2 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a2[i]));
        __m512 d3 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a3[i]));
        __m512 d4 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a4[i]));
        __m512 d5 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a5[i]));
        __m512 d6 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a6[i]));
        __m512 d7 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a7[i]));

        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        acc4 = _mm512_fmadd_ps(d4, d4, acc4);
        acc5 = _mm512_fmadd_ps(d5, d5, acc5);
        acc6 = _mm512_fmadd_ps(d6, d6, acc6);
        acc7 = _mm512_fmadd_ps(d7, d7, acc7);
    }

    float s0 = _mm512_reduce_add_ps(acc0);
    float s1 = _mm512_reduce_add_ps(acc1);
    float s2 = _mm512_reduce_add_ps(acc2);
    float s3 = _mm512_reduce_add_ps(acc3);
    float s4 = _mm512_reduce_add_ps(acc4);
    float s5 = _mm512_reduce_add_ps(acc5);
    float s6 = _mm512_reduce_add_ps(acc6);
    float s7 = _mm512_reduce_add_ps(acc7);

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        float diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        float diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        float diff3 = q_val - a3[i]; s3 += diff3 * diff3;
        float diff4 = q_val - a4[i]; s4 += diff4 * diff4;
        float diff5 = q_val - a5[i]; s5 += diff5 * diff5;
        float diff6 = q_val - a6[i]; s6 += diff6 * diff6;
        float diff7 = q_val - a7[i]; s7 += diff7 * diff7;
    }

    out_dists[0] = (double)sqrtf(s0);
    out_dists[1] = (double)sqrtf(s1);
    out_dists[2] = (double)sqrtf(s2);
    out_dists[3] = (double)sqrtf(s3);
    out_dists[4] = (double)sqrtf(s4);
    out_dists[5] = (double)sqrtf(s5);
    out_dists[6] = (double)sqrtf(s6);
    out_dists[7] = (double)sqrtf(s7);
}
#endif

/**
 * framedist_batch_1x8_float() - Vectorized 1-query vs 8-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @out_dists: Array of 8 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x8_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        calc_dist8_f32_avx512(q, anchors, out_dists, size);
        return;
    }
#endif
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    long i = 0;

    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];
    const float *restrict a4 = anchors[4];
    const float *restrict a5 = anchors[5];
    const float *restrict a6 = anchors[6];
    const float *restrict a7 = anchors[7];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__AVX2__)
    if (size == 2)
    {
        __m256 vq0 = _mm256_set1_ps(q[0]);
        __m256 va0 = _mm256_set_ps(a7[0], a6[0], a5[0], a4[0],
                                   a3[0], a2[0], a1[0], a0[0]);
        __m256 diff0 = _mm256_sub_ps(vq0, va0);
        __m256 sum_vec = _mm256_mul_ps(diff0, diff0);

        __m256 vq1 = _mm256_set1_ps(q[1]);
        __m256 va1 = _mm256_set_ps(a7[1], a6[1], a5[1], a4[1],
                                   a3[1], a2[1], a1[1], a0[1]);
        __m256 diff1 = _mm256_sub_ps(vq1, va1);
#ifdef __FMA__
        sum_vec = _mm256_fmadd_ps(diff1, diff1, sum_vec);
#else
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff1, diff1));
#endif
        __m256 vdist = _mm256_sqrt_ps(sum_vec);
        __m128 vlo = _mm256_castps256_ps128(vdist);
        __m128 vhi = _mm256_extractf128_ps(vdist, 1);
        _mm256_storeu_pd(&out_dists[0], _mm256_cvtps_pd(vlo));
        _mm256_storeu_pd(&out_dists[4], _mm256_cvtps_pd(vhi));
        return;
    }
#endif

    if (size >= 8)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        for (; i <= size - 8; i += 8)
        {
            if (size >= 1024 && (i & 15) == 0)
            {
                _mm_prefetch((const char *)&q[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a0[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a1[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a2[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a3[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a4[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a5[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a6[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char *)&a7[i + 64], _MM_HINT_T0);
            }

            __m256 vq = _mm256_loadu_ps(&q[i]);
            __m256 d0 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a0[i]));
            __m256 d1 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a1[i]));
            __m256 d2 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a2[i]));
            __m256 d3 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a3[i]));
            __m256 d4 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a4[i]));
            __m256 d5 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a5[i]));
            __m256 d6 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a6[i]));
            __m256 d7 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a7[i]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            acc4 = _mm256_fmadd_ps(d4, d4, acc4);
            acc5 = _mm256_fmadd_ps(d5, d5, acc5);
            acc6 = _mm256_fmadd_ps(d6, d6, acc6);
            acc7 = _mm256_fmadd_ps(d7, d7, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4, d4));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5, d5));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6, d6));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7, d7));
#endif
        }

        __m128 s0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
        __m128 s1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
        __m128 s2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
        __m128 s3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
        __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc4), _mm256_extractf128_ps(acc4, 1));
        __m128 s5 = _mm_add_ps(_mm256_castps256_ps128(acc5), _mm256_extractf128_ps(acc5, 1));
        __m128 s6 = _mm_add_ps(_mm256_castps256_ps128(acc6), _mm256_extractf128_ps(acc6, 1));
        __m128 s7 = _mm_add_ps(_mm256_castps256_ps128(acc7), _mm256_extractf128_ps(acc7, 1));

        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        _MM_TRANSPOSE4_PS(s4, s5, s6, s7);

        __m128 f0123 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
        __m128 f4567 = _mm_add_ps(_mm_add_ps(s4, s5), _mm_add_ps(s6, s7));

        float f03[4];
        float f47[4];
        _mm_storeu_ps(f03, f0123);
        _mm_storeu_ps(f47, f4567);

        sum0 += f03[0];
        sum1 += f03[1];
        sum2 += f03[2];
        sum3 += f03[3];
        sum4 += f47[0];
        sum5 += f47[1];
        sum6 += f47[2];
        sum7 += f47[3];
    }
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float d0 = q_val - a0[i];
        float d1 = q_val - a1[i];
        float d2 = q_val - a2[i];
        float d3 = q_val - a3[i];
        float d4 = q_val - a4[i];
        float d5 = q_val - a5[i];
        float d6 = q_val - a6[i];
        float d7 = q_val - a7[i];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
    }

    out_dists[0] = (double)sqrtf(sum0);
    out_dists[1] = (double)sqrtf(sum1);
    out_dists[2] = (double)sqrtf(sum2);
    out_dists[3] = (double)sqrtf(sum3);
    out_dists[4] = (double)sqrtf(sum4);
    out_dists[5] = (double)sqrtf(sum5);
    out_dists[6] = (double)sqrtf(sum6);
    out_dists[7] = (double)sqrtf(sum7);
}

/**
 * framedist_batch_1x4_double() - Vectorized 1-query vs 4-anchor Euclidean distance (double).
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @out_dists: Array of 4 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x4_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size)
{
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    long i = 0;

    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 8)
    {
        __m256d a0_0 = _mm256_setzero_pd();
        __m256d a0_1 = _mm256_setzero_pd();
        __m256d a1_0 = _mm256_setzero_pd();
        __m256d a1_1 = _mm256_setzero_pd();
        __m256d a2_0 = _mm256_setzero_pd();
        __m256d a2_1 = _mm256_setzero_pd();
        __m256d a3_0 = _mm256_setzero_pd();
        __m256d a3_1 = _mm256_setzero_pd();

        for (; i <= size - 8; i += 8)
        {
            /* Chunk 0 (4 doubles) */
            __m256d vq0 = _mm256_loadu_pd(&q[i]);
            __m256d d0_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a0[i]));
            __m256d d1_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a1[i]));
            __m256d d2_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a2[i]));
            __m256d d3_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_pd(d0_0, d0_0, a0_0);
            a1_0 = _mm256_fmadd_pd(d1_0, d1_0, a1_0);
            a2_0 = _mm256_fmadd_pd(d2_0, d2_0, a2_0);
            a3_0 = _mm256_fmadd_pd(d3_0, d3_0, a3_0);
#else
            a0_0 = _mm256_add_pd(a0_0, _mm256_mul_pd(d0_0, d0_0));
            a1_0 = _mm256_add_pd(a1_0, _mm256_mul_pd(d1_0, d1_0));
            a2_0 = _mm256_add_pd(a2_0, _mm256_mul_pd(d2_0, d2_0));
            a3_0 = _mm256_add_pd(a3_0, _mm256_mul_pd(d3_0, d3_0));
#endif

            /* Chunk 1 (4 doubles) */
            __m256d vq1 = _mm256_loadu_pd(&q[i + 4]);
            __m256d d0_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a0[i + 4]));
            __m256d d1_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a1[i + 4]));
            __m256d d2_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a2[i + 4]));
            __m256d d3_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a3[i + 4]));

#ifdef __FMA__
            a0_1 = _mm256_fmadd_pd(d0_1, d0_1, a0_1);
            a1_1 = _mm256_fmadd_pd(d1_1, d1_1, a1_1);
            a2_1 = _mm256_fmadd_pd(d2_1, d2_1, a2_1);
            a3_1 = _mm256_fmadd_pd(d3_1, d3_1, a3_1);
#else
            a0_1 = _mm256_add_pd(a0_1, _mm256_mul_pd(d0_1, d0_1));
            a1_1 = _mm256_add_pd(a1_1, _mm256_mul_pd(d1_1, d1_1));
            a2_1 = _mm256_add_pd(a2_1, _mm256_mul_pd(d2_1, d2_1));
            a3_1 = _mm256_add_pd(a3_1, _mm256_mul_pd(d3_1, d3_1));
#endif
        }

        for (; i <= size - 4; i += 4)
        {
            __m256d vq = _mm256_loadu_pd(&q[i]);
            __m256d d0 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a0[i]));
            __m256d d1 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a1[i]));
            __m256d d2 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a2[i]));
            __m256d d3 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_pd(d0, d0, a0_0);
            a1_0 = _mm256_fmadd_pd(d1, d1, a1_0);
            a2_0 = _mm256_fmadd_pd(d2, d2, a2_0);
            a3_0 = _mm256_fmadd_pd(d3, d3, a3_0);
#else
            a0_0 = _mm256_add_pd(a0_0, _mm256_mul_pd(d0, d0));
            a1_0 = _mm256_add_pd(a1_0, _mm256_mul_pd(d1, d1));
            a2_0 = _mm256_add_pd(a2_0, _mm256_mul_pd(d2, d2));
            a3_0 = _mm256_add_pd(a3_0, _mm256_mul_pd(d3, d3));
#endif
        }

        __m256d acc0 = _mm256_add_pd(a0_0, a0_1);
        __m256d acc1 = _mm256_add_pd(a1_0, a1_1);
        __m256d acc2 = _mm256_add_pd(a2_0, a2_1);
        __m256d acc3 = _mm256_add_pd(a3_0, a3_1);

        __m128d lo0 = _mm256_castpd256_pd128(acc0);
        __m128d hi0 = _mm256_extractf128_pd(acc0, 1);
        __m128d s0 = _mm_add_pd(lo0, hi0);

        __m128d lo1 = _mm256_castpd256_pd128(acc1);
        __m128d hi1 = _mm256_extractf128_pd(acc1, 1);
        __m128d s1 = _mm_add_pd(lo1, hi1);

        __m128d lo2 = _mm256_castpd256_pd128(acc2);
        __m128d hi2 = _mm256_extractf128_pd(acc2, 1);
        __m128d s2 = _mm_add_pd(lo2, hi2);

        __m128d lo3 = _mm256_castpd256_pd128(acc3);
        __m128d hi3 = _mm256_extractf128_pd(acc3, 1);
        __m128d s3 = _mm_add_pd(lo3, hi3);

        sum0 += _mm_cvtsd_f64(_mm_add_sd(s0, _mm_unpackhi_pd(s0, s0)));
        sum1 += _mm_cvtsd_f64(_mm_add_sd(s1, _mm_unpackhi_pd(s1, s1)));
        sum2 += _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
        sum3 += _mm_cvtsd_f64(_mm_add_sd(s3, _mm_unpackhi_pd(s3, s3)));
    }
#endif

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i];
        double diff1 = q_val - a1[i];
        double diff2 = q_val - a2[i];
        double diff3 = q_val - a3[i];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    out_dists[0] = sqrt(sum0);
    out_dists[1] = sqrt(sum1);
    out_dists[2] = sqrt(sum2);
    out_dists[3] = sqrt(sum3);
}

/**
 * framedist_batch_float() - Batch Euclidean distance from query to N candidate anchors.
 * @q:         Pointer to query array.
 * @anchors:   Array of N pointers to candidate anchor arrays.
 * @n_anchors: Number of candidates in this batch.
 * @out_dists: Array of N doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size)
{
    int k = 0;
    while (k + 8 <= n_anchors)
    {
        framedist_batch_1x8_float(q, &anchors[k], &out_dists[k], size);
        k += 8;
    }

    while (k + 4 <= n_anchors)
    {
        framedist_batch_1x4_float(q, &anchors[k], &out_dists[k], size);
        k += 4;
    }

    for (; k < n_anchors; k++)
    {
        out_dists[k] = framedist_float(q, anchors[k], size);
    }
}

/**
 * framedist_batch_1x8_double() - Vectorized 1-query vs 8-anchor Euclidean distance (double).
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @out_dists: Array of 8 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x8_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size)
{
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    double sum5 = 0.0;
    double sum6 = 0.0;
    double sum7 = 0.0;
    long i = 0;

    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];
    const double *restrict a4 = anchors[4];
    const double *restrict a5 = anchors[5];
    const double *restrict a6 = anchors[6];
    const double *restrict a7 = anchors[7];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__AVX2__)
    if (size == 2)
    {
        __m256d vq0 = _mm256_set1_pd(q[0]);
        __m256d va0_lo = _mm256_set_pd(a3[0], a2[0], a1[0], a0[0]);
        __m256d va0_hi = _mm256_set_pd(a7[0], a6[0], a5[0], a4[0]);
        __m256d d0_lo = _mm256_sub_pd(vq0, va0_lo);
        __m256d d0_hi = _mm256_sub_pd(vq0, va0_hi);
        __m256d s_lo = _mm256_mul_pd(d0_lo, d0_lo);
        __m256d s_hi = _mm256_mul_pd(d0_hi, d0_hi);

        __m256d vq1 = _mm256_set1_pd(q[1]);
        __m256d va1_lo = _mm256_set_pd(a3[1], a2[1], a1[1], a0[1]);
        __m256d va1_hi = _mm256_set_pd(a7[1], a6[1], a5[1], a4[1]);
        __m256d d1_lo = _mm256_sub_pd(vq1, va1_lo);
        __m256d d1_hi = _mm256_sub_pd(vq1, va1_hi);
#ifdef __FMA__
        s_lo = _mm256_fmadd_pd(d1_lo, d1_lo, s_lo);
        s_hi = _mm256_fmadd_pd(d1_hi, d1_hi, s_hi);
#else
        s_lo = _mm256_add_pd(s_lo, _mm256_mul_pd(d1_lo, d1_lo));
        s_hi = _mm256_add_pd(s_hi, _mm256_mul_pd(d1_hi, d1_hi));
#endif
        _mm256_storeu_pd(&out_dists[0], _mm256_sqrt_pd(s_lo));
        _mm256_storeu_pd(&out_dists[4], _mm256_sqrt_pd(s_hi));
        return;
    }
#endif

    if (size >= 4)
    {
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();
        __m256d acc4 = _mm256_setzero_pd();
        __m256d acc5 = _mm256_setzero_pd();
        __m256d acc6 = _mm256_setzero_pd();
        __m256d acc7 = _mm256_setzero_pd();

        for (; i <= size - 4; i += 4)
        {
            if (size >= 512 && (i & 7) == 0)
            {
                _mm_prefetch((const char *)&q[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a0[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a1[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a2[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a3[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a4[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a5[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a6[i + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&a7[i + 32], _MM_HINT_T0);
            }

            __m256d vq = _mm256_loadu_pd(&q[i]);
            __m256d d0 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a0[i]));
            __m256d d1 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a1[i]));
            __m256d d2 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a2[i]));
            __m256d d3 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a3[i]));
            __m256d d4 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a4[i]));
            __m256d d5 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a5[i]));
            __m256d d6 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a6[i]));
            __m256d d7 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a7[i]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_pd(d0, d0, acc0);
            acc1 = _mm256_fmadd_pd(d1, d1, acc1);
            acc2 = _mm256_fmadd_pd(d2, d2, acc2);
            acc3 = _mm256_fmadd_pd(d3, d3, acc3);
            acc4 = _mm256_fmadd_pd(d4, d4, acc4);
            acc5 = _mm256_fmadd_pd(d5, d5, acc5);
            acc6 = _mm256_fmadd_pd(d6, d6, acc6);
            acc7 = _mm256_fmadd_pd(d7, d7, acc7);
#else
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(d2, d2));
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(d3, d3));
            acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(d4, d4));
            acc5 = _mm256_add_pd(acc5, _mm256_mul_pd(d5, d5));
            acc6 = _mm256_add_pd(acc6, _mm256_mul_pd(d6, d6));
            acc7 = _mm256_add_pd(acc7, _mm256_mul_pd(d7, d7));
#endif
        }

        __m128d lo0 = _mm256_castpd256_pd128(acc0);
        __m128d hi0 = _mm256_extractf128_pd(acc0, 1);
        __m128d s0 = _mm_add_pd(lo0, hi0);

        __m128d lo1 = _mm256_castpd256_pd128(acc1);
        __m128d hi1 = _mm256_extractf128_pd(acc1, 1);
        __m128d s1 = _mm_add_pd(lo1, hi1);

        __m128d lo2 = _mm256_castpd256_pd128(acc2);
        __m128d hi2 = _mm256_extractf128_pd(acc2, 1);
        __m128d s2 = _mm_add_pd(lo2, hi2);

        __m128d lo3 = _mm256_castpd256_pd128(acc3);
        __m128d hi3 = _mm256_extractf128_pd(acc3, 1);
        __m128d s3 = _mm_add_pd(lo3, hi3);

        __m128d lo4 = _mm256_castpd256_pd128(acc4);
        __m128d hi4 = _mm256_extractf128_pd(acc4, 1);
        __m128d s4 = _mm_add_pd(lo4, hi4);

        __m128d lo5 = _mm256_castpd256_pd128(acc5);
        __m128d hi5 = _mm256_extractf128_pd(acc5, 1);
        __m128d s5 = _mm_add_pd(lo5, hi5);

        __m128d lo6 = _mm256_castpd256_pd128(acc6);
        __m128d hi6 = _mm256_extractf128_pd(acc6, 1);
        __m128d s6 = _mm_add_pd(lo6, hi6);

        __m128d lo7 = _mm256_castpd256_pd128(acc7);
        __m128d hi7 = _mm256_extractf128_pd(acc7, 1);
        __m128d s7 = _mm_add_pd(lo7, hi7);

        sum0 += _mm_cvtsd_f64(_mm_add_sd(s0, _mm_unpackhi_pd(s0, s0)));
        sum1 += _mm_cvtsd_f64(_mm_add_sd(s1, _mm_unpackhi_pd(s1, s1)));
        sum2 += _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
        sum3 += _mm_cvtsd_f64(_mm_add_sd(s3, _mm_unpackhi_pd(s3, s3)));
        sum4 += _mm_cvtsd_f64(_mm_add_sd(s4, _mm_unpackhi_pd(s4, s4)));
        sum5 += _mm_cvtsd_f64(_mm_add_sd(s5, _mm_unpackhi_pd(s5, s5)));
        sum6 += _mm_cvtsd_f64(_mm_add_sd(s6, _mm_unpackhi_pd(s6, s6)));
        sum7 += _mm_cvtsd_f64(_mm_add_sd(s7, _mm_unpackhi_pd(s7, s7)));
    }
#endif

    for (; i < size; i++)
    {
        double q_val = q[i];
        double d0 = q_val - a0[i];
        double d1 = q_val - a1[i];
        double d2 = q_val - a2[i];
        double d3 = q_val - a3[i];
        double d4 = q_val - a4[i];
        double d5 = q_val - a5[i];
        double d6 = q_val - a6[i];
        double d7 = q_val - a7[i];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
    }

    out_dists[0] = sqrt(sum0);
    out_dists[1] = sqrt(sum1);
    out_dists[2] = sqrt(sum2);
    out_dists[3] = sqrt(sum3);
    out_dists[4] = sqrt(sum4);
    out_dists[5] = sqrt(sum5);
    out_dists[6] = sqrt(sum6);
    out_dists[7] = sqrt(sum7);
}

/**
 * framedist_batch_double() - Batch Euclidean distance from query to N candidate anchors.
 * @q:         Pointer to query array.
 * @anchors:   Array of N pointers to candidate anchor arrays.
 * @n_anchors: Number of candidates in this batch.
 * @out_dists: Array of N doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size)
{
    int k = 0;
    while (k + 8 <= n_anchors)
    {
        framedist_batch_1x8_double(q, &anchors[k], &out_dists[k], size);
        k += 8;
    }

    while (k + 4 <= n_anchors)
    {
        framedist_batch_1x4_double(q, &anchors[k], &out_dists[k], size);
        k += 4;
    }

    for (; k < n_anchors; k++)
    {
        out_dists[k] = framedist_double(q, anchors[k], size);
    }
}

/**
 * framedist_batch() - Computes Euclidean distances from query Frame to N candidate Frames.
 * @q:         Pointer to query Frame.
 * @anchors:   Array of pointers to candidate Frames.
 * @n_anchors: Number of candidate Frames.
 * @out_dists: Array of N doubles to receive computed distances.
 */
void framedist_batch(
    const Frame  *q,
    const Frame **anchors,
    int           n_anchors,
    double       *out_dists)
{
    if (n_anchors <= 0)
    {
        return;
    }

    long size = q->width * q->height;
    const void *ptrs_stack[64];
    const void **ptrs = ptrs_stack;

    if (n_anchors > 64)
    {
        ptrs = (const void **)malloc((size_t)n_anchors * sizeof(const void *));
        if (!ptrs)
        {
            return;
        }
    }

    for (int i = 0; i < n_anchors; i++)
    {
        ptrs[i] = anchors[i]->data;
    }

    if (q->is_double)
    {
        framedist_batch_double(
            (const double *)q->data,
            (const double *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }
    else
    {
        framedist_batch_float(
            (const float *)q->data,
            (const float *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }

    if (ptrs != ptrs_stack)
    {
        free(ptrs);
    }
}


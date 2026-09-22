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
static int calc_dist4_cutoff_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    long                         size,
    double                       cutoff_sq,
    double *restrict             out_dists)
{
    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];

    __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
    __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
    __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
    __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();

    long i = 0;
    for (; i <= size - 32; i += 32)
    {
        if (size >= 1024 && (i & 31) == 0)
        {
            _mm_prefetch((const char *)&q[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a0[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a1[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a2[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a3[i + 64], _MM_HINT_T0);
        }

        __m512 vq0 = _mm512_loadu_ps(&q[i]);
        __m512 d0_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a0[i]));
        __m512 d1_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a1[i]));
        __m512 d2_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a2[i]));
        __m512 d3_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a3[i]));

        acc0_0 = _mm512_fmadd_ps(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_ps(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_ps(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_ps(d3_0, d3_0, acc3_0);

        __m512 vq1 = _mm512_loadu_ps(&q[i + 16]);
        __m512 d0_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a0[i + 16]));
        __m512 d1_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a1[i + 16]));
        __m512 d2_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a2[i + 16]));
        __m512 d3_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a3[i + 16]));

        acc0_1 = _mm512_fmadd_ps(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_ps(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_ps(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_ps(d3_1, d3_1, acc3_1);

        if (cutoff_sq > 0.0 && ((i & 63) == 32))
        {
            float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_0, acc0_1));
            float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_0, acc1_1));
            float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_0, acc2_1));
            float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_0, acc3_1));
            float fcut = (float)cutoff_sq;
            if (s0 >= fcut && s1 >= fcut && s2 >= fcut && s3 >= fcut)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                out_dists[0] = dist_exceeded;
                out_dists[1] = dist_exceeded;
                out_dists[2] = dist_exceeded;
                out_dists[3] = dist_exceeded;
                return 0xF;
            }
        }
    } // for (; i <= size - 32; i += 32)

    float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_0, acc0_1));
    float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_0, acc1_1));
    float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_0, acc2_1));
    float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_0, acc3_1));

    for (; i <= size - 16; i += 16)
    {
        __m512 vq = _mm512_loadu_ps(&q[i]);
        __m512 d0 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a0[i]));
        __m512 d1 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a1[i]));
        __m512 d2 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a2[i]));
        __m512 d3 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a3[i]));

        s0 += _mm512_reduce_add_ps(_mm512_mul_ps(d0, d0));
        s1 += _mm512_reduce_add_ps(_mm512_mul_ps(d1, d1));
        s2 += _mm512_reduce_add_ps(_mm512_mul_ps(d2, d2));
        s3 += _mm512_reduce_add_ps(_mm512_mul_ps(d3, d3));
    } // for (; i <= size - 16; i += 16)

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        float diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        float diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        float diff3 = q_val - a3[i]; s3 += diff3 * diff3;
    }

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        float fcut = (float)cutoff_sq;
        if (s0 >= fcut) { pruned_mask |= 1; out_dists[0] = dist_exceeded; }
        else            { out_dists[0] = (double)sqrtf(s0); }

        if (s1 >= fcut) { pruned_mask |= 2; out_dists[1] = dist_exceeded; }
        else            { out_dists[1] = (double)sqrtf(s1); }

        if (s2 >= fcut) { pruned_mask |= 4; out_dists[2] = dist_exceeded; }
        else            { out_dists[2] = (double)sqrtf(s2); }

        if (s3 >= fcut) { pruned_mask |= 8; out_dists[3] = dist_exceeded; }
        else            { out_dists[3] = (double)sqrtf(s3); }
    }
    else
    {
        out_dists[0] = (double)sqrtf(s0);
        out_dists[1] = (double)sqrtf(s1);
        out_dists[2] = (double)sqrtf(s2);
        out_dists[3] = (double)sqrtf(s3);
    }
    return pruned_mask;
}

GRIC_TARGET_AVX512
static int calc_dist8_cutoff_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    long                         size,
    double                       cutoff_sq,
    double *restrict             out_dists)
{
    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];
    const float *restrict a4 = anchors[4];
    const float *restrict a5 = anchors[5];
    const float *restrict a6 = anchors[6];
    const float *restrict a7 = anchors[7];

    __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
    __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
    __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
    __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
    __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps();
    __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps();
    __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps();
    __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps();

    long i = 0;
    for (; i <= size - 32; i += 32)
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

        __m512 vq0 = _mm512_loadu_ps(&q[i]);
        __m512 d0_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a0[i]));
        __m512 d1_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a1[i]));
        __m512 d2_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a2[i]));
        __m512 d3_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a3[i]));
        __m512 d4_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a4[i]));
        __m512 d5_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a5[i]));
        __m512 d6_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a6[i]));
        __m512 d7_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a7[i]));

        acc0_0 = _mm512_fmadd_ps(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_ps(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_ps(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_ps(d3_0, d3_0, acc3_0);
        acc4_0 = _mm512_fmadd_ps(d4_0, d4_0, acc4_0);
        acc5_0 = _mm512_fmadd_ps(d5_0, d5_0, acc5_0);
        acc6_0 = _mm512_fmadd_ps(d6_0, d6_0, acc6_0);
        acc7_0 = _mm512_fmadd_ps(d7_0, d7_0, acc7_0);

        __m512 vq1 = _mm512_loadu_ps(&q[i + 16]);
        __m512 d0_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a0[i + 16]));
        __m512 d1_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a1[i + 16]));
        __m512 d2_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a2[i + 16]));
        __m512 d3_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a3[i + 16]));
        __m512 d4_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a4[i + 16]));
        __m512 d5_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a5[i + 16]));
        __m512 d6_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a6[i + 16]));
        __m512 d7_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a7[i + 16]));

        acc0_1 = _mm512_fmadd_ps(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_ps(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_ps(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_ps(d3_1, d3_1, acc3_1);
        acc4_1 = _mm512_fmadd_ps(d4_1, d4_1, acc4_1);
        acc5_1 = _mm512_fmadd_ps(d5_1, d5_1, acc5_1);
        acc6_1 = _mm512_fmadd_ps(d6_1, d6_1, acc6_1);
        acc7_1 = _mm512_fmadd_ps(d7_1, d7_1, acc7_1);

        if (cutoff_sq > 0.0 && ((i & 63) == 32))
        {
            float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_0, acc0_1));
            float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_0, acc1_1));
            float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_0, acc2_1));
            float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_0, acc3_1));
            float s4 = _mm512_reduce_add_ps(_mm512_add_ps(acc4_0, acc4_1));
            float s5 = _mm512_reduce_add_ps(_mm512_add_ps(acc5_0, acc5_1));
            float s6 = _mm512_reduce_add_ps(_mm512_add_ps(acc6_0, acc6_1));
            float s7 = _mm512_reduce_add_ps(_mm512_add_ps(acc7_0, acc7_1));
            float fcut = (float)cutoff_sq;
            if (s0 >= fcut && s1 >= fcut && s2 >= fcut && s3 >= fcut &&
                s4 >= fcut && s5 >= fcut && s6 >= fcut && s7 >= fcut)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                for (int k = 0; k < 8; k++)
                {
                    out_dists[k] = dist_exceeded;
                }
                return 0xFF;
            }
        }
    } // for (; i <= size - 32; i += 32)

    float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_0, acc0_1));
    float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_0, acc1_1));
    float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_0, acc2_1));
    float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_0, acc3_1));
    float s4 = _mm512_reduce_add_ps(_mm512_add_ps(acc4_0, acc4_1));
    float s5 = _mm512_reduce_add_ps(_mm512_add_ps(acc5_0, acc5_1));
    float s6 = _mm512_reduce_add_ps(_mm512_add_ps(acc6_0, acc6_1));
    float s7 = _mm512_reduce_add_ps(_mm512_add_ps(acc7_0, acc7_1));

    for (; i <= size - 16; i += 16)
    {
        __m512 vq = _mm512_loadu_ps(&q[i]);
        __m512 d0 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a0[i]));
        __m512 d1 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a1[i]));
        __m512 d2 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a2[i]));
        __m512 d3 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a3[i]));
        __m512 d4 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a4[i]));
        __m512 d5 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a5[i]));
        __m512 d6 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a6[i]));
        __m512 d7 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a7[i]));

        s0 += _mm512_reduce_add_ps(_mm512_mul_ps(d0, d0));
        s1 += _mm512_reduce_add_ps(_mm512_mul_ps(d1, d1));
        s2 += _mm512_reduce_add_ps(_mm512_mul_ps(d2, d2));
        s3 += _mm512_reduce_add_ps(_mm512_mul_ps(d3, d3));
        s4 += _mm512_reduce_add_ps(_mm512_mul_ps(d4, d4));
        s5 += _mm512_reduce_add_ps(_mm512_mul_ps(d5, d5));
        s6 += _mm512_reduce_add_ps(_mm512_mul_ps(d6, d6));
        s7 += _mm512_reduce_add_ps(_mm512_mul_ps(d7, d7));
    }

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

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        float fcut = (float)cutoff_sq;
        float sums[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int k = 0; k < 8; k++)
        {
            if (sums[k] >= fcut)
            {
                pruned_mask |= (1 << k);
                out_dists[k] = dist_exceeded;
            }
            else
            {
                out_dists[k] = (double)sqrtf(sums[k]);
            }
        }
    }
    else
    {
        out_dists[0] = (double)sqrtf(s0);
        out_dists[1] = (double)sqrtf(s1);
        out_dists[2] = (double)sqrtf(s2);
        out_dists[3] = (double)sqrtf(s3);
        out_dists[4] = (double)sqrtf(s4);
        out_dists[5] = (double)sqrtf(s5);
        out_dists[6] = (double)sqrtf(s6);
        out_dists[7] = (double)sqrtf(s7);
    }
    return pruned_mask;
}

GRIC_TARGET_AVX512
static void calc_dist8_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
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

    __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
    __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
    __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
    __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
    __m512 acc4_0 = _mm512_setzero_ps(), acc4_1 = _mm512_setzero_ps();
    __m512 acc5_0 = _mm512_setzero_ps(), acc5_1 = _mm512_setzero_ps();
    __m512 acc6_0 = _mm512_setzero_ps(), acc6_1 = _mm512_setzero_ps();
    __m512 acc7_0 = _mm512_setzero_ps(), acc7_1 = _mm512_setzero_ps();

    long i = 0;
    for (; i <= size - 32; i += 32)
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

        __m512 vq0 = _mm512_loadu_ps(&q[i]);
        __m512 d0_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a0[i]));
        __m512 d1_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a1[i]));
        __m512 d2_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a2[i]));
        __m512 d3_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a3[i]));
        __m512 d4_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a4[i]));
        __m512 d5_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a5[i]));
        __m512 d6_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a6[i]));
        __m512 d7_0 = _mm512_sub_ps(vq0, _mm512_loadu_ps(&a7[i]));

        acc0_0 = _mm512_fmadd_ps(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_ps(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_ps(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_ps(d3_0, d3_0, acc3_0);
        acc4_0 = _mm512_fmadd_ps(d4_0, d4_0, acc4_0);
        acc5_0 = _mm512_fmadd_ps(d5_0, d5_0, acc5_0);
        acc6_0 = _mm512_fmadd_ps(d6_0, d6_0, acc6_0);
        acc7_0 = _mm512_fmadd_ps(d7_0, d7_0, acc7_0);

        __m512 vq1 = _mm512_loadu_ps(&q[i + 16]);
        __m512 d0_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a0[i + 16]));
        __m512 d1_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a1[i + 16]));
        __m512 d2_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a2[i + 16]));
        __m512 d3_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a3[i + 16]));
        __m512 d4_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a4[i + 16]));
        __m512 d5_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a5[i + 16]));
        __m512 d6_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a6[i + 16]));
        __m512 d7_1 = _mm512_sub_ps(vq1, _mm512_loadu_ps(&a7[i + 16]));

        acc0_1 = _mm512_fmadd_ps(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_ps(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_ps(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_ps(d3_1, d3_1, acc3_1);
        acc4_1 = _mm512_fmadd_ps(d4_1, d4_1, acc4_1);
        acc5_1 = _mm512_fmadd_ps(d5_1, d5_1, acc5_1);
        acc6_1 = _mm512_fmadd_ps(d6_1, d6_1, acc6_1);
        acc7_1 = _mm512_fmadd_ps(d7_1, d7_1, acc7_1);
    } // for (; i <= size - 32; i += 32)

    float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_0, acc0_1));
    float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_0, acc1_1));
    float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_0, acc2_1));
    float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_0, acc3_1));
    float s4 = _mm512_reduce_add_ps(_mm512_add_ps(acc4_0, acc4_1));
    float s5 = _mm512_reduce_add_ps(_mm512_add_ps(acc5_0, acc5_1));
    float s6 = _mm512_reduce_add_ps(_mm512_add_ps(acc6_0, acc6_1));
    float s7 = _mm512_reduce_add_ps(_mm512_add_ps(acc7_0, acc7_1));

    for (; i <= size - 16; i += 16)
    {
        __m512 vq = _mm512_loadu_ps(&q[i]);
        __m512 d0 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a0[i]));
        __m512 d1 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a1[i]));
        __m512 d2 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a2[i]));
        __m512 d3 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a3[i]));
        __m512 d4 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a4[i]));
        __m512 d5 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a5[i]));
        __m512 d6 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a6[i]));
        __m512 d7 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a7[i]));

        s0 += _mm512_reduce_add_ps(_mm512_mul_ps(d0, d0));
        s1 += _mm512_reduce_add_ps(_mm512_mul_ps(d1, d1));
        s2 += _mm512_reduce_add_ps(_mm512_mul_ps(d2, d2));
        s3 += _mm512_reduce_add_ps(_mm512_mul_ps(d3, d3));
        s4 += _mm512_reduce_add_ps(_mm512_mul_ps(d4, d4));
        s5 += _mm512_reduce_add_ps(_mm512_mul_ps(d5, d5));
        s6 += _mm512_reduce_add_ps(_mm512_mul_ps(d6, d6));
        s7 += _mm512_reduce_add_ps(_mm512_mul_ps(d7, d7));
    } // for (; i <= size - 16; i += 16)

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

GRIC_TARGET_AVX512
static void calc_dist16_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
    const float *restrict a0  = anchors[0];
    const float *restrict a1  = anchors[1];
    const float *restrict a2  = anchors[2];
    const float *restrict a3  = anchors[3];
    const float *restrict a4  = anchors[4];
    const float *restrict a5  = anchors[5];
    const float *restrict a6  = anchors[6];
    const float *restrict a7  = anchors[7];
    const float *restrict a8  = anchors[8];
    const float *restrict a9  = anchors[9];
    const float *restrict a10 = anchors[10];
    const float *restrict a11 = anchors[11];
    const float *restrict a12 = anchors[12];
    const float *restrict a13 = anchors[13];
    const float *restrict a14 = anchors[14];
    const float *restrict a15 = anchors[15];

    __m512 acc0  = _mm512_setzero_ps(), acc1  = _mm512_setzero_ps();
    __m512 acc2  = _mm512_setzero_ps(), acc3  = _mm512_setzero_ps();
    __m512 acc4  = _mm512_setzero_ps(), acc5  = _mm512_setzero_ps();
    __m512 acc6  = _mm512_setzero_ps(), acc7  = _mm512_setzero_ps();
    __m512 acc8  = _mm512_setzero_ps(), acc9  = _mm512_setzero_ps();
    __m512 acc10 = _mm512_setzero_ps(), acc11 = _mm512_setzero_ps();
    __m512 acc12 = _mm512_setzero_ps(), acc13 = _mm512_setzero_ps();
    __m512 acc14 = _mm512_setzero_ps(), acc15 = _mm512_setzero_ps();

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
            _mm_prefetch((const char *)&a8[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a9[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a10[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a11[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a12[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a13[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a14[i + 64], _MM_HINT_T0);
            _mm_prefetch((const char *)&a15[i + 64], _MM_HINT_T0);
        }

        __m512 vq = _mm512_loadu_ps(&q[i]);
        __m512 d0  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a0[i]));
        __m512 d1  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a1[i]));
        __m512 d2  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a2[i]));
        __m512 d3  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a3[i]));
        __m512 d4  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a4[i]));
        __m512 d5  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a5[i]));
        __m512 d6  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a6[i]));
        __m512 d7  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a7[i]));
        __m512 d8  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a8[i]));
        __m512 d9  = _mm512_sub_ps(vq, _mm512_loadu_ps(&a9[i]));
        __m512 d10 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a10[i]));
        __m512 d11 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a11[i]));
        __m512 d12 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a12[i]));
        __m512 d13 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a13[i]));
        __m512 d14 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a14[i]));
        __m512 d15 = _mm512_sub_ps(vq, _mm512_loadu_ps(&a15[i]));

        acc0  = _mm512_fmadd_ps(d0,  d0,  acc0);
        acc1  = _mm512_fmadd_ps(d1,  d1,  acc1);
        acc2  = _mm512_fmadd_ps(d2,  d2,  acc2);
        acc3  = _mm512_fmadd_ps(d3,  d3,  acc3);
        acc4  = _mm512_fmadd_ps(d4,  d4,  acc4);
        acc5  = _mm512_fmadd_ps(d5,  d5,  acc5);
        acc6  = _mm512_fmadd_ps(d6,  d6,  acc6);
        acc7  = _mm512_fmadd_ps(d7,  d7,  acc7);
        acc8  = _mm512_fmadd_ps(d8,  d8,  acc8);
        acc9  = _mm512_fmadd_ps(d9,  d9,  acc9);
        acc10 = _mm512_fmadd_ps(d10, d10, acc10);
        acc11 = _mm512_fmadd_ps(d11, d11, acc11);
        acc12 = _mm512_fmadd_ps(d12, d12, acc12);
        acc13 = _mm512_fmadd_ps(d13, d13, acc13);
        acc14 = _mm512_fmadd_ps(d14, d14, acc14);
        acc15 = _mm512_fmadd_ps(d15, d15, acc15);
    } // for (; i <= size - 16; i += 16)

    float s0  = _mm512_reduce_add_ps(acc0);
    float s1  = _mm512_reduce_add_ps(acc1);
    float s2  = _mm512_reduce_add_ps(acc2);
    float s3  = _mm512_reduce_add_ps(acc3);
    float s4  = _mm512_reduce_add_ps(acc4);
    float s5  = _mm512_reduce_add_ps(acc5);
    float s6  = _mm512_reduce_add_ps(acc6);
    float s7  = _mm512_reduce_add_ps(acc7);
    float s8  = _mm512_reduce_add_ps(acc8);
    float s9  = _mm512_reduce_add_ps(acc9);
    float s10 = _mm512_reduce_add_ps(acc10);
    float s11 = _mm512_reduce_add_ps(acc11);
    float s12 = _mm512_reduce_add_ps(acc12);
    float s13 = _mm512_reduce_add_ps(acc13);
    float s14 = _mm512_reduce_add_ps(acc14);
    float s15 = _mm512_reduce_add_ps(acc15);

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0  = q_val - a0[i];  s0  += diff0  * diff0;
        float diff1  = q_val - a1[i];  s1  += diff1  * diff1;
        float diff2  = q_val - a2[i];  s2  += diff2  * diff2;
        float diff3  = q_val - a3[i];  s3  += diff3  * diff3;
        float diff4  = q_val - a4[i];  s4  += diff4  * diff4;
        float diff5  = q_val - a5[i];  s5  += diff5  * diff5;
        float diff6  = q_val - a6[i];  s6  += diff6  * diff6;
        float diff7  = q_val - a7[i];  s7  += diff7  * diff7;
        float diff8  = q_val - a8[i];  s8  += diff8  * diff8;
        float diff9  = q_val - a9[i];  s9  += diff9  * diff9;
        float diff10 = q_val - a10[i]; s10 += diff10 * diff10;
        float diff11 = q_val - a11[i]; s11 += diff11 * diff11;
        float diff12 = q_val - a12[i]; s12 += diff12 * diff12;
        float diff13 = q_val - a13[i]; s13 += diff13 * diff13;
        float diff14 = q_val - a14[i]; s14 += diff14 * diff14;
        float diff15 = q_val - a15[i]; s15 += diff15 * diff15;
    } // for (; i < size; i++)

    out_dists[0]  = (double)sqrtf(s0);
    out_dists[1]  = (double)sqrtf(s1);
    out_dists[2]  = (double)sqrtf(s2);
    out_dists[3]  = (double)sqrtf(s3);
    out_dists[4]  = (double)sqrtf(s4);
    out_dists[5]  = (double)sqrtf(s5);
    out_dists[6]  = (double)sqrtf(s6);
    out_dists[7]  = (double)sqrtf(s7);
    out_dists[8]  = (double)sqrtf(s8);
    out_dists[9]  = (double)sqrtf(s9);
    out_dists[10] = (double)sqrtf(s10);
    out_dists[11] = (double)sqrtf(s11);
    out_dists[12] = (double)sqrtf(s12);
    out_dists[13] = (double)sqrtf(s13);
    out_dists[14] = (double)sqrtf(s14);
    out_dists[15] = (double)sqrtf(s15);
}
#endif

/**
 * framedist_batch_1x16_float() - Vectorized 1-query vs 16-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 16 pointers to candidate anchor arrays.
 * @out_dists: Array of 16 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x16_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        calc_dist16_f32_avx512(q, anchors, out_dists, size);
        return;
    }
#endif
    framedist_batch_1x8_float(q, anchors, out_dists, size);
    framedist_batch_1x8_float(q, anchors + 8, out_dists + 8, size);
}

/**
 * framedist_batch_cutoff_1x4_float() - 1 query vs 4 anchors with early-cutoff checkpoints.
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @size:      Number of elements in each array.
 * @cutoff_sq: Squared distance threshold for early exit (<= 0.0 disables cutoff).
 * @out_dists: Array of 4 doubles to receive computed distances.
 *
 * Checks partial squared distances periodically (every 64 elements). If all 4
 * candidates exceed @cutoff_sq, aborts immediately.
 *
 * Return: Bitmask (0x0 to 0xF) where bit (1 << b) is 1 if candidate b exceeded cutoff_sq.
 */
int framedist_batch_cutoff_1x4_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    long                         size,
    double                       cutoff_sq,
    double *restrict             out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        return calc_dist4_cutoff_f32_avx512(q, anchors, size, cutoff_sq, out_dists);
    }
#endif
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    long i = 0;

    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 16)
    {
        __m256 a0_0 = _mm256_setzero_ps(), a0_1 = _mm256_setzero_ps();
        __m256 a1_0 = _mm256_setzero_ps(), a1_1 = _mm256_setzero_ps();
        __m256 a2_0 = _mm256_setzero_ps(), a2_1 = _mm256_setzero_ps();
        __m256 a3_0 = _mm256_setzero_ps(), a3_1 = _mm256_setzero_ps();

        for (; i <= size - 16; i += 16)
        {
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

            if (cutoff_sq > 0.0 && ((i & 63) == 48))
            {
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

                __m128 vcut = _mm_set1_ps((float)cutoff_sq);
                __m128 cmp = _mm_cmpgt_ps(final_sums, vcut);
                if (_mm_movemask_ps(cmp) == 0xF)
                {
                    double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                    out_dists[0] = dist_exceeded;
                    out_dists[1] = dist_exceeded;
                    out_dists[2] = dist_exceeded;
                    out_dists[3] = dist_exceeded;
                    return 0xF;
                }
            } // if (cutoff_sq > 0.0 && ((i & 63) == 48))
        } // for (; i <= size - 16; i += 16)

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
        } // for (; i <= size - 8; i += 8)

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
    } // if (size >= 16)
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i]; sum0 += diff0 * diff0;
        float diff1 = q_val - a1[i]; sum1 += diff1 * diff1;
        float diff2 = q_val - a2[i]; sum2 += diff2 * diff2;
        float diff3 = q_val - a3[i]; sum3 += diff3 * diff3;

        if (cutoff_sq > 0.0 && ((i & 63) == 63))
        {
            float fcut = (float)cutoff_sq;
            if (sum0 >= fcut && sum1 >= fcut && sum2 >= fcut && sum3 >= fcut)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                out_dists[0] = dist_exceeded;
                out_dists[1] = dist_exceeded;
                out_dists[2] = dist_exceeded;
                out_dists[3] = dist_exceeded;
                return 0xF;
            }
        }
    } // for (; i < size; i++)

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        if ((double)sum0 >= cutoff_sq)
        {
            pruned_mask |= (1 << 0);
            out_dists[0] = dist_exceeded;
        }
        else
        {
            out_dists[0] = (double)sqrtf(sum0);
        }

        if ((double)sum1 >= cutoff_sq)
        {
            pruned_mask |= (1 << 1);
            out_dists[1] = dist_exceeded;
        }
        else
        {
            out_dists[1] = (double)sqrtf(sum1);
        }

        if ((double)sum2 >= cutoff_sq)
        {
            pruned_mask |= (1 << 2);
            out_dists[2] = dist_exceeded;
        }
        else
        {
            out_dists[2] = (double)sqrtf(sum2);
        }

        if ((double)sum3 >= cutoff_sq)
        {
            pruned_mask |= (1 << 3);
            out_dists[3] = dist_exceeded;
        }
        else
        {
            out_dists[3] = (double)sqrtf(sum3);
        }
    }
    else
    {
        out_dists[0] = (double)sqrtf(sum0);
        out_dists[1] = (double)sqrtf(sum1);
        out_dists[2] = (double)sqrtf(sum2);
        out_dists[3] = (double)sqrtf(sum3);
    }
    return pruned_mask;
}

/**
 * framedist_batch_cutoff_1x8_float() - 1 query vs 8 anchors with early-cutoff checkpoints.
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @size:      Number of elements in each array.
 * @cutoff_sq: Squared distance threshold for early exit (<= 0.0 disables cutoff).
 * @out_dists: Array of 8 doubles to receive computed distances.
 *
 * Checks partial squared distances periodically (every 64 elements). If all 8
 * candidates exceed @cutoff_sq, aborts immediately.
 *
 * Return: Bitmask (0x00 to 0xFF) where bit (1 << b) is 1 if candidate b exceeded cutoff_sq.
 */
int framedist_batch_cutoff_1x8_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    long                         size,
    double                       cutoff_sq,
    double *restrict             out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        return calc_dist8_cutoff_f32_avx512(q, anchors, size, cutoff_sq, out_dists);
    }
#endif
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
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

            __m256 vq0 = _mm256_loadu_ps(&q[i]);
            __m256 d0_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a0[i]));
            __m256 d1_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a1[i]));
            __m256 d2_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a2[i]));
            __m256 d3_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a3[i]));
            __m256 d4_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a4[i]));
            __m256 d5_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a5[i]));
            __m256 d6_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a6[i]));
            __m256 d7_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a7[i]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0_0, d0_0, acc0);
            acc1 = _mm256_fmadd_ps(d1_0, d1_0, acc1);
            acc2 = _mm256_fmadd_ps(d2_0, d2_0, acc2);
            acc3 = _mm256_fmadd_ps(d3_0, d3_0, acc3);
            acc4 = _mm256_fmadd_ps(d4_0, d4_0, acc4);
            acc5 = _mm256_fmadd_ps(d5_0, d5_0, acc5);
            acc6 = _mm256_fmadd_ps(d6_0, d6_0, acc6);
            acc7 = _mm256_fmadd_ps(d7_0, d7_0, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0_0, d0_0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1_0, d1_0));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2_0, d2_0));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3_0, d3_0));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4_0, d4_0));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5_0, d5_0));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6_0, d6_0));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7_0, d7_0));
#endif

            __m256 vq1 = _mm256_loadu_ps(&q[i + 8]);
            __m256 d0_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a0[i + 8]));
            __m256 d1_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a1[i + 8]));
            __m256 d2_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a2[i + 8]));
            __m256 d3_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a3[i + 8]));
            __m256 d4_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a4[i + 8]));
            __m256 d5_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a5[i + 8]));
            __m256 d6_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a6[i + 8]));
            __m256 d7_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a7[i + 8]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0_1, d0_1, acc0);
            acc1 = _mm256_fmadd_ps(d1_1, d1_1, acc1);
            acc2 = _mm256_fmadd_ps(d2_1, d2_1, acc2);
            acc3 = _mm256_fmadd_ps(d3_1, d3_1, acc3);
            acc4 = _mm256_fmadd_ps(d4_1, d4_1, acc4);
            acc5 = _mm256_fmadd_ps(d5_1, d5_1, acc5);
            acc6 = _mm256_fmadd_ps(d6_1, d6_1, acc6);
            acc7 = _mm256_fmadd_ps(d7_1, d7_1, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0_1, d0_1));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1_1, d1_1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2_1, d2_1));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3_1, d3_1));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4_1, d4_1));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5_1, d5_1));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6_1, d6_1));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7_1, d7_1));
#endif

            if (cutoff_sq > 0.0 && ((i & 63) == 48))
            {
                __m128 s0 = _mm_add_ps(
                    _mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1)
                );
                __m128 s1 = _mm_add_ps(
                    _mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1)
                );
                __m128 s2 = _mm_add_ps(
                    _mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1)
                );
                __m128 s3 = _mm_add_ps(
                    _mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1)
                );
                __m128 s4 = _mm_add_ps(
                    _mm256_castps256_ps128(acc4), _mm256_extractf128_ps(acc4, 1)
                );
                __m128 s5 = _mm_add_ps(
                    _mm256_castps256_ps128(acc5), _mm256_extractf128_ps(acc5, 1)
                );
                __m128 s6 = _mm_add_ps(
                    _mm256_castps256_ps128(acc6), _mm256_extractf128_ps(acc6, 1)
                );
                __m128 s7 = _mm_add_ps(
                    _mm256_castps256_ps128(acc7), _mm256_extractf128_ps(acc7, 1)
                );

                _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
                __m128 sum03 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));

                _MM_TRANSPOSE4_PS(s4, s5, s6, s7);
                __m128 sum47 = _mm_add_ps(_mm_add_ps(s4, s5), _mm_add_ps(s6, s7));

                __m128 vcut = _mm_set1_ps((float)cutoff_sq);
                __m128 cmp_lo = _mm_cmpgt_ps(sum03, vcut);
                __m128 cmp_hi = _mm_cmpgt_ps(sum47, vcut);
                if (_mm_movemask_ps(cmp_lo) == 0xF && _mm_movemask_ps(cmp_hi) == 0xF)
                {
                    double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                    for (int k = 0; k < 8; k++)
                    {
                        out_dists[k] = dist_exceeded;
                    }
                    return 0xFF;
                }
            } // if (cutoff_sq > 0.0 && ((i & 63) == 48))
        } // for (; i <= size - 16; i += 16)

        for (; i <= size - 8; i += 8)
        {
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

        sum0 += f03[0]; sum1 += f03[1]; sum2 += f03[2]; sum3 += f03[3];
        sum4 += f47[0]; sum5 += f47[1]; sum6 += f47[2]; sum7 += f47[3];
    } // if (size >= 8)
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i]; sum0 += diff0 * diff0;
        float diff1 = q_val - a1[i]; sum1 += diff1 * diff1;
        float diff2 = q_val - a2[i]; sum2 += diff2 * diff2;
        float diff3 = q_val - a3[i]; sum3 += diff3 * diff3;
        float diff4 = q_val - a4[i]; sum4 += diff4 * diff4;
        float diff5 = q_val - a5[i]; sum5 += diff5 * diff5;
        float diff6 = q_val - a6[i]; sum6 += diff6 * diff6;
        float diff7 = q_val - a7[i]; sum7 += diff7 * diff7;

        if (cutoff_sq > 0.0 && ((i & 63) == 63))
        {
            float fcut = (float)cutoff_sq;
            if (sum0 >= fcut && sum1 >= fcut && sum2 >= fcut && sum3 >= fcut &&
                sum4 >= fcut && sum5 >= fcut && sum6 >= fcut && sum7 >= fcut)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                for (int k = 0; k < 8; k++)
                {
                    out_dists[k] = dist_exceeded;
                }
                return 0xFF;
            }
        }
    } // for (; i < size; i++)

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        float fcut = (float)cutoff_sq;
        float sums[8] = {sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7};
        for (int k = 0; k < 8; k++)
        {
            if (sums[k] >= fcut)
            {
                pruned_mask |= (1 << k);
                out_dists[k] = dist_exceeded;
            }
            else
            {
                out_dists[k] = (double)sqrtf(sums[k]);
            }
        }
    }
    else
    {
        out_dists[0] = (double)sqrtf(sum0);
        out_dists[1] = (double)sqrtf(sum1);
        out_dists[2] = (double)sqrtf(sum2);
        out_dists[3] = (double)sqrtf(sum3);
        out_dists[4] = (double)sqrtf(sum4);
        out_dists[5] = (double)sqrtf(sum5);
        out_dists[6] = (double)sqrtf(sum6);
        out_dists[7] = (double)sqrtf(sum7);
    }
    return pruned_mask;
}

/**
 * framedist_batch_1x8_float() - Vectorized 1-query vs 8-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @out_dists: Array of 8 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
static inline __attribute__((always_inline)) void calc_dist8_f32_core(
    const float *restrict q,
    const float *restrict a0,
    const float *restrict a1,
    const float *restrict a2,
    const float *restrict a3,
    const float *restrict a4,
    const float *restrict a5,
    const float *restrict a6,
    const float *restrict a7,
    double *restrict      out_dists,
    long                  size)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    long i = 0;

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

            __m256 vq0 = _mm256_loadu_ps(&q[i]);
            __m256 d0_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a0[i]));
            __m256 d1_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a1[i]));
            __m256 d2_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a2[i]));
            __m256 d3_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a3[i]));
            __m256 d4_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a4[i]));
            __m256 d5_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a5[i]));
            __m256 d6_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a6[i]));
            __m256 d7_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a7[i]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0_0, d0_0, acc0);
            acc1 = _mm256_fmadd_ps(d1_0, d1_0, acc1);
            acc2 = _mm256_fmadd_ps(d2_0, d2_0, acc2);
            acc3 = _mm256_fmadd_ps(d3_0, d3_0, acc3);
            acc4 = _mm256_fmadd_ps(d4_0, d4_0, acc4);
            acc5 = _mm256_fmadd_ps(d5_0, d5_0, acc5);
            acc6 = _mm256_fmadd_ps(d6_0, d6_0, acc6);
            acc7 = _mm256_fmadd_ps(d7_0, d7_0, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0_0, d0_0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1_0, d1_0));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2_0, d2_0));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3_0, d3_0));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4_0, d4_0));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5_0, d5_0));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6_0, d6_0));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7_0, d7_0));
#endif

            __m256 vq1 = _mm256_loadu_ps(&q[i + 8]);
            __m256 d0_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a0[i + 8]));
            __m256 d1_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a1[i + 8]));
            __m256 d2_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a2[i + 8]));
            __m256 d3_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a3[i + 8]));
            __m256 d4_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a4[i + 8]));
            __m256 d5_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a5[i + 8]));
            __m256 d6_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a6[i + 8]));
            __m256 d7_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a7[i + 8]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0_1, d0_1, acc0);
            acc1 = _mm256_fmadd_ps(d1_1, d1_1, acc1);
            acc2 = _mm256_fmadd_ps(d2_1, d2_1, acc2);
            acc3 = _mm256_fmadd_ps(d3_1, d3_1, acc3);
            acc4 = _mm256_fmadd_ps(d4_1, d4_1, acc4);
            acc5 = _mm256_fmadd_ps(d5_1, d5_1, acc5);
            acc6 = _mm256_fmadd_ps(d6_1, d6_1, acc6);
            acc7 = _mm256_fmadd_ps(d7_1, d7_1, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0_1, d0_1));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1_1, d1_1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2_1, d2_1));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3_1, d3_1));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4_1, d4_1));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5_1, d5_1));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6_1, d6_1));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7_1, d7_1));
#endif
        }

        for (; i <= size - 8; i += 8)
        {
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
    calc_dist8_f32_core(
        q,
        anchors[0], anchors[1], anchors[2], anchors[3],
        anchors[4], anchors[5], anchors[6], anchors[7],
        out_dists, size
    );
}

void framedist_batch_1x8_contiguous_float(
    const float *restrict q,
    const float *restrict anchors_matrix,
    double *restrict      out_dists,
    long                  size)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        const float *a_ptrs[8];
        for (int k = 0; k < 8; k++)
        {
            a_ptrs[k] = anchors_matrix + (size_t)k * (size_t)size;
        }
        calc_dist8_f32_avx512(q, a_ptrs, out_dists, size);
        return;
    }
#endif
    calc_dist8_f32_core(
        q,
        anchors_matrix,
        anchors_matrix + size,
        anchors_matrix + 2 * size,
        anchors_matrix + 3 * size,
        anchors_matrix + 4 * size,
        anchors_matrix + 5 * size,
        anchors_matrix + 6 * size,
        anchors_matrix + 7 * size,
        out_dists, size
    );
}

void framedist_batch_1x16_contiguous_float(
    const float *restrict q,
    const float *restrict anchors_matrix,
    double *restrict      out_dists,
    long                  size)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        const float *a_ptrs[16];
        for (int k = 0; k < 16; k++)
        {
            a_ptrs[k] = anchors_matrix + (size_t)k * (size_t)size;
        }
        calc_dist16_f32_avx512(q, a_ptrs, out_dists, size);
        return;
    }
#endif
    framedist_batch_1x8_contiguous_float(q, anchors_matrix, out_dists, size);
    framedist_batch_1x8_contiguous_float(
        q, anchors_matrix + 8 * size, out_dists + 8, size
    );
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static int calc_dist4_cutoff_d64_avx512(
    const double *restrict        q,
    const double *const *restrict anchors,
    long                          size,
    double                        cutoff_sq,
    double *restrict              out_dists)
{
    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];

    __m512d acc0_0 = _mm512_setzero_pd(), acc0_1 = _mm512_setzero_pd();
    __m512d acc1_0 = _mm512_setzero_pd(), acc1_1 = _mm512_setzero_pd();
    __m512d acc2_0 = _mm512_setzero_pd(), acc2_1 = _mm512_setzero_pd();
    __m512d acc3_0 = _mm512_setzero_pd(), acc3_1 = _mm512_setzero_pd();

    long i = 0;
    for (; i <= size - 16; i += 16)
    {
        if (size >= 512 && (i & 31) == 0)
        {
            _mm_prefetch((const char *)&q[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a0[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a1[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a2[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a3[i + 32], _MM_HINT_T0);
        }

        __m512d vq0 = _mm512_loadu_pd(&q[i]);
        __m512d d0_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a0[i]));
        __m512d d1_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a1[i]));
        __m512d d2_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a2[i]));
        __m512d d3_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a3[i]));

        acc0_0 = _mm512_fmadd_pd(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_pd(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_pd(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_pd(d3_0, d3_0, acc3_0);

        __m512d vq1 = _mm512_loadu_pd(&q[i + 8]);
        __m512d d0_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a0[i + 8]));
        __m512d d1_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a1[i + 8]));
        __m512d d2_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a2[i + 8]));
        __m512d d3_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a3[i + 8]));

        acc0_1 = _mm512_fmadd_pd(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_pd(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_pd(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_pd(d3_1, d3_1, acc3_1);

        if (cutoff_sq > 0.0 && ((i & 63) == 48))
        {
            double s0 = _mm512_reduce_add_pd(_mm512_add_pd(acc0_0, acc0_1));
            double s1 = _mm512_reduce_add_pd(_mm512_add_pd(acc1_0, acc1_1));
            double s2 = _mm512_reduce_add_pd(_mm512_add_pd(acc2_0, acc2_1));
            double s3 = _mm512_reduce_add_pd(_mm512_add_pd(acc3_0, acc3_1));
            if (s0 >= cutoff_sq && s1 >= cutoff_sq && s2 >= cutoff_sq && s3 >= cutoff_sq)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                out_dists[0] = dist_exceeded;
                out_dists[1] = dist_exceeded;
                out_dists[2] = dist_exceeded;
                out_dists[3] = dist_exceeded;
                return 0xF;
            }
        }
    } // for (; i <= size - 16; i += 16)

    double s0 = _mm512_reduce_add_pd(_mm512_add_pd(acc0_0, acc0_1));
    double s1 = _mm512_reduce_add_pd(_mm512_add_pd(acc1_0, acc1_1));
    double s2 = _mm512_reduce_add_pd(_mm512_add_pd(acc2_0, acc2_1));
    double s3 = _mm512_reduce_add_pd(_mm512_add_pd(acc3_0, acc3_1));

    for (; i <= size - 8; i += 8)
    {
        __m512d vq = _mm512_loadu_pd(&q[i]);
        __m512d d0 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a0[i]));
        __m512d d1 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a1[i]));
        __m512d d2 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a2[i]));
        __m512d d3 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a3[i]));

        s0 += _mm512_reduce_add_pd(_mm512_mul_pd(d0, d0));
        s1 += _mm512_reduce_add_pd(_mm512_mul_pd(d1, d1));
        s2 += _mm512_reduce_add_pd(_mm512_mul_pd(d2, d2));
        s3 += _mm512_reduce_add_pd(_mm512_mul_pd(d3, d3));
    } // for (; i <= size - 8; i += 8)

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; s3 += diff3 * diff3;
    }

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        if (s0 >= cutoff_sq) { pruned_mask |= 1; out_dists[0] = dist_exceeded; }
        else                 { out_dists[0] = sqrt(s0); }

        if (s1 >= cutoff_sq) { pruned_mask |= 2; out_dists[1] = dist_exceeded; }
        else                 { out_dists[1] = sqrt(s1); }

        if (s2 >= cutoff_sq) { pruned_mask |= 4; out_dists[2] = dist_exceeded; }
        else                 { out_dists[2] = sqrt(s2); }

        if (s3 >= cutoff_sq) { pruned_mask |= 8; out_dists[3] = dist_exceeded; }
        else                 { out_dists[3] = sqrt(s3); }
    }
    else
    {
        out_dists[0] = sqrt(s0);
        out_dists[1] = sqrt(s1);
        out_dists[2] = sqrt(s2);
        out_dists[3] = sqrt(s3);
    }
    return pruned_mask;
}

GRIC_TARGET_AVX512
static int calc_dist8_cutoff_d64_avx512(
    const double *restrict        q,
    const double *const *restrict anchors,
    long                          size,
    double                        cutoff_sq,
    double *restrict              out_dists)
{
    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];
    const double *restrict a4 = anchors[4];
    const double *restrict a5 = anchors[5];
    const double *restrict a6 = anchors[6];
    const double *restrict a7 = anchors[7];

    __m512d acc0_0 = _mm512_setzero_pd(), acc0_1 = _mm512_setzero_pd();
    __m512d acc1_0 = _mm512_setzero_pd(), acc1_1 = _mm512_setzero_pd();
    __m512d acc2_0 = _mm512_setzero_pd(), acc2_1 = _mm512_setzero_pd();
    __m512d acc3_0 = _mm512_setzero_pd(), acc3_1 = _mm512_setzero_pd();
    __m512d acc4_0 = _mm512_setzero_pd(), acc4_1 = _mm512_setzero_pd();
    __m512d acc5_0 = _mm512_setzero_pd(), acc5_1 = _mm512_setzero_pd();
    __m512d acc6_0 = _mm512_setzero_pd(), acc6_1 = _mm512_setzero_pd();
    __m512d acc7_0 = _mm512_setzero_pd(), acc7_1 = _mm512_setzero_pd();

    long i = 0;
    for (; i <= size - 16; i += 16)
    {
        if (size >= 512 && (i & 31) == 0)
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

        __m512d vq0 = _mm512_loadu_pd(&q[i]);
        __m512d d0_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a0[i]));
        __m512d d1_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a1[i]));
        __m512d d2_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a2[i]));
        __m512d d3_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a3[i]));
        __m512d d4_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a4[i]));
        __m512d d5_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a5[i]));
        __m512d d6_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a6[i]));
        __m512d d7_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a7[i]));

        acc0_0 = _mm512_fmadd_pd(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_pd(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_pd(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_pd(d3_0, d3_0, acc3_0);
        acc4_0 = _mm512_fmadd_pd(d4_0, d4_0, acc4_0);
        acc5_0 = _mm512_fmadd_pd(d5_0, d5_0, acc5_0);
        acc6_0 = _mm512_fmadd_pd(d6_0, d6_0, acc6_0);
        acc7_0 = _mm512_fmadd_pd(d7_0, d7_0, acc7_0);

        __m512d vq1 = _mm512_loadu_pd(&q[i + 8]);
        __m512d d0_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a0[i + 8]));
        __m512d d1_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a1[i + 8]));
        __m512d d2_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a2[i + 8]));
        __m512d d3_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a3[i + 8]));
        __m512d d4_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a4[i + 8]));
        __m512d d5_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a5[i + 8]));
        __m512d d6_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a6[i + 8]));
        __m512d d7_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a7[i + 8]));

        acc0_1 = _mm512_fmadd_pd(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_pd(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_pd(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_pd(d3_1, d3_1, acc3_1);
        acc4_1 = _mm512_fmadd_pd(d4_1, d4_1, acc4_1);
        acc5_1 = _mm512_fmadd_pd(d5_1, d5_1, acc5_1);
        acc6_1 = _mm512_fmadd_pd(d6_1, d6_1, acc6_1);
        acc7_1 = _mm512_fmadd_pd(d7_1, d7_1, acc7_1);

        if (cutoff_sq > 0.0 && ((i & 63) == 48))
        {
            double s0 = _mm512_reduce_add_pd(_mm512_add_pd(acc0_0, acc0_1));
            double s1 = _mm512_reduce_add_pd(_mm512_add_pd(acc1_0, acc1_1));
            double s2 = _mm512_reduce_add_pd(_mm512_add_pd(acc2_0, acc2_1));
            double s3 = _mm512_reduce_add_pd(_mm512_add_pd(acc3_0, acc3_1));
            double s4 = _mm512_reduce_add_pd(_mm512_add_pd(acc4_0, acc4_1));
            double s5 = _mm512_reduce_add_pd(_mm512_add_pd(acc5_0, acc5_1));
            double s6 = _mm512_reduce_add_pd(_mm512_add_pd(acc6_0, acc6_1));
            double s7 = _mm512_reduce_add_pd(_mm512_add_pd(acc7_0, acc7_1));
            if (s0 >= cutoff_sq && s1 >= cutoff_sq && s2 >= cutoff_sq && s3 >= cutoff_sq &&
                s4 >= cutoff_sq && s5 >= cutoff_sq && s6 >= cutoff_sq && s7 >= cutoff_sq)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                for (int k = 0; k < 8; k++)
                {
                    out_dists[k] = dist_exceeded;
                }
                return 0xFF;
            }
        }
    } // for (; i <= size - 16; i += 16)

    double s0 = _mm512_reduce_add_pd(_mm512_add_pd(acc0_0, acc0_1));
    double s1 = _mm512_reduce_add_pd(_mm512_add_pd(acc1_0, acc1_1));
    double s2 = _mm512_reduce_add_pd(_mm512_add_pd(acc2_0, acc2_1));
    double s3 = _mm512_reduce_add_pd(_mm512_add_pd(acc3_0, acc3_1));
    double s4 = _mm512_reduce_add_pd(_mm512_add_pd(acc4_0, acc4_1));
    double s5 = _mm512_reduce_add_pd(_mm512_add_pd(acc5_0, acc5_1));
    double s6 = _mm512_reduce_add_pd(_mm512_add_pd(acc6_0, acc6_1));
    double s7 = _mm512_reduce_add_pd(_mm512_add_pd(acc7_0, acc7_1));

    for (; i <= size - 8; i += 8)
    {
        __m512d vq = _mm512_loadu_pd(&q[i]);
        __m512d d0 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a0[i]));
        __m512d d1 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a1[i]));
        __m512d d2 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a2[i]));
        __m512d d3 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a3[i]));
        __m512d d4 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a4[i]));
        __m512d d5 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a5[i]));
        __m512d d6 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a6[i]));
        __m512d d7 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a7[i]));

        s0 += _mm512_reduce_add_pd(_mm512_mul_pd(d0, d0));
        s1 += _mm512_reduce_add_pd(_mm512_mul_pd(d1, d1));
        s2 += _mm512_reduce_add_pd(_mm512_mul_pd(d2, d2));
        s3 += _mm512_reduce_add_pd(_mm512_mul_pd(d3, d3));
        s4 += _mm512_reduce_add_pd(_mm512_mul_pd(d4, d4));
        s5 += _mm512_reduce_add_pd(_mm512_mul_pd(d5, d5));
        s6 += _mm512_reduce_add_pd(_mm512_mul_pd(d6, d6));
        s7 += _mm512_reduce_add_pd(_mm512_mul_pd(d7, d7));
    }

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; s3 += diff3 * diff3;
        double diff4 = q_val - a4[i]; s4 += diff4 * diff4;
        double diff5 = q_val - a5[i]; s5 += diff5 * diff5;
        double diff6 = q_val - a6[i]; s6 += diff6 * diff6;
        double diff7 = q_val - a7[i]; s7 += diff7 * diff7;
    }

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        double sums[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int k = 0; k < 8; k++)
        {
            if (sums[k] >= cutoff_sq)
            {
                pruned_mask |= (1 << k);
                out_dists[k] = dist_exceeded;
            }
            else
            {
                out_dists[k] = sqrt(sums[k]);
            }
        }
    }
    else
    {
        out_dists[0] = sqrt(s0);
        out_dists[1] = sqrt(s1);
        out_dists[2] = sqrt(s2);
        out_dists[3] = sqrt(s3);
        out_dists[4] = sqrt(s4);
        out_dists[5] = sqrt(s5);
        out_dists[6] = sqrt(s6);
        out_dists[7] = sqrt(s7);
    }
    return pruned_mask;
}

GRIC_TARGET_AVX512
static void calc_dist4_d64_avx512(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size)
{
    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];

    __m512d acc0_0 = _mm512_setzero_pd(), acc0_1 = _mm512_setzero_pd();
    __m512d acc1_0 = _mm512_setzero_pd(), acc1_1 = _mm512_setzero_pd();
    __m512d acc2_0 = _mm512_setzero_pd(), acc2_1 = _mm512_setzero_pd();
    __m512d acc3_0 = _mm512_setzero_pd(), acc3_1 = _mm512_setzero_pd();

    long i = 0;
    for (; i <= size - 16; i += 16)
    {
        if (size >= 512 && (i & 31) == 0)
        {
            _mm_prefetch((const char *)&q[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a0[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a1[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a2[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&a3[i + 32], _MM_HINT_T0);
        }

        __m512d vq0 = _mm512_loadu_pd(&q[i]);
        __m512d d0_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a0[i]));
        __m512d d1_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a1[i]));
        __m512d d2_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a2[i]));
        __m512d d3_0 = _mm512_sub_pd(vq0, _mm512_loadu_pd(&a3[i]));

        acc0_0 = _mm512_fmadd_pd(d0_0, d0_0, acc0_0);
        acc1_0 = _mm512_fmadd_pd(d1_0, d1_0, acc1_0);
        acc2_0 = _mm512_fmadd_pd(d2_0, d2_0, acc2_0);
        acc3_0 = _mm512_fmadd_pd(d3_0, d3_0, acc3_0);

        __m512d vq1 = _mm512_loadu_pd(&q[i + 8]);
        __m512d d0_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a0[i + 8]));
        __m512d d1_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a1[i + 8]));
        __m512d d2_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a2[i + 8]));
        __m512d d3_1 = _mm512_sub_pd(vq1, _mm512_loadu_pd(&a3[i + 8]));

        acc0_1 = _mm512_fmadd_pd(d0_1, d0_1, acc0_1);
        acc1_1 = _mm512_fmadd_pd(d1_1, d1_1, acc1_1);
        acc2_1 = _mm512_fmadd_pd(d2_1, d2_1, acc2_1);
        acc3_1 = _mm512_fmadd_pd(d3_1, d3_1, acc3_1);
    } // for (; i <= size - 16; i += 16)

    double s0 = _mm512_reduce_add_pd(_mm512_add_pd(acc0_0, acc0_1));
    double s1 = _mm512_reduce_add_pd(_mm512_add_pd(acc1_0, acc1_1));
    double s2 = _mm512_reduce_add_pd(_mm512_add_pd(acc2_0, acc2_1));
    double s3 = _mm512_reduce_add_pd(_mm512_add_pd(acc3_0, acc3_1));

    for (; i <= size - 8; i += 8)
    {
        __m512d vq = _mm512_loadu_pd(&q[i]);
        __m512d d0 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a0[i]));
        __m512d d1 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a1[i]));
        __m512d d2 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a2[i]));
        __m512d d3 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a3[i]));

        s0 += _mm512_reduce_add_pd(_mm512_mul_pd(d0, d0));
        s1 += _mm512_reduce_add_pd(_mm512_mul_pd(d1, d1));
        s2 += _mm512_reduce_add_pd(_mm512_mul_pd(d2, d2));
        s3 += _mm512_reduce_add_pd(_mm512_mul_pd(d3, d3));
    } // for (; i <= size - 8; i += 8)

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; s3 += diff3 * diff3;
    }

    out_dists[0] = sqrt(s0);
    out_dists[1] = sqrt(s1);
    out_dists[2] = sqrt(s2);
    out_dists[3] = sqrt(s3);
}
#endif

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
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 8)
    {
        calc_dist4_d64_avx512(q, anchors, out_dists, size);
        return;
    }
#endif
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
 * framedist_batch_cutoff_1x4_double() - 1 query vs 4 anchors with early-cutoff for double.
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @size:      Number of elements in each array.
 * @cutoff_sq: Squared distance threshold for early exit (<= 0.0 disables cutoff).
 * @out_dists: Array of 4 doubles to receive computed distances.
 *
 * Checks partial squared distances periodically (every 64 elements). If all 4
 * candidates exceed @cutoff_sq, aborts immediately.
 *
 * Return: Bitmask (0x0 to 0xF) where bit (1 << b) is 1 if candidate b exceeded cutoff_sq.
 */
int framedist_batch_cutoff_1x4_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    long                          size,
    double                        cutoff_sq,
    double *restrict              out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 8)
    {
        return calc_dist4_cutoff_d64_avx512(q, anchors, size, cutoff_sq, out_dists);
    }
#endif
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
        __m256d a0_0 = _mm256_setzero_pd(), a0_1 = _mm256_setzero_pd();
        __m256d a1_0 = _mm256_setzero_pd(), a1_1 = _mm256_setzero_pd();
        __m256d a2_0 = _mm256_setzero_pd(), a2_1 = _mm256_setzero_pd();
        __m256d a3_0 = _mm256_setzero_pd(), a3_1 = _mm256_setzero_pd();

        for (; i <= size - 8; i += 8)
        {
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

            if (cutoff_sq > 0.0 && ((i & 63) == 56))
            {
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

                double cur0 = _mm_cvtsd_f64(_mm_add_sd(s0, _mm_unpackhi_pd(s0, s0)));
                double cur1 = _mm_cvtsd_f64(_mm_add_sd(s1, _mm_unpackhi_pd(s1, s1)));
                double cur2 = _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
                double cur3 = _mm_cvtsd_f64(_mm_add_sd(s3, _mm_unpackhi_pd(s3, s3)));

                if (cur0 >= cutoff_sq && cur1 >= cutoff_sq &&
                    cur2 >= cutoff_sq && cur3 >= cutoff_sq)
                {
                    double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                    out_dists[0] = dist_exceeded;
                    out_dists[1] = dist_exceeded;
                    out_dists[2] = dist_exceeded;
                    out_dists[3] = dist_exceeded;
                    return 0xF;
                }
            } // if (cutoff_sq > 0.0 && ((i & 63) == 56))
        } // for (; i <= size - 8; i += 8)

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
        } // for (; i <= size - 4; i += 4)

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
    } // if (size >= 8)
#endif

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i]; sum0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; sum1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; sum2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; sum3 += diff3 * diff3;

        if (cutoff_sq > 0.0 && ((i & 63) == 63))
        {
            if (sum0 >= cutoff_sq && sum1 >= cutoff_sq &&
                sum2 >= cutoff_sq && sum3 >= cutoff_sq)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                out_dists[0] = dist_exceeded;
                out_dists[1] = dist_exceeded;
                out_dists[2] = dist_exceeded;
                out_dists[3] = dist_exceeded;
                return 0xF;
            }
        }
    } // for (; i < size; i++)

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        if (sum0 >= cutoff_sq)
        {
            pruned_mask |= (1 << 0);
            out_dists[0] = dist_exceeded;
        }
        else
        {
            out_dists[0] = sqrt(sum0);
        }

        if (sum1 >= cutoff_sq)
        {
            pruned_mask |= (1 << 1);
            out_dists[1] = dist_exceeded;
        }
        else
        {
            out_dists[1] = sqrt(sum1);
        }

        if (sum2 >= cutoff_sq)
        {
            pruned_mask |= (1 << 2);
            out_dists[2] = dist_exceeded;
        }
        else
        {
            out_dists[2] = sqrt(sum2);
        }

        if (sum3 >= cutoff_sq)
        {
            pruned_mask |= (1 << 3);
            out_dists[3] = dist_exceeded;
        }
        else
        {
            out_dists[3] = sqrt(sum3);
        }
    }
    else
    {
        out_dists[0] = sqrt(sum0);
        out_dists[1] = sqrt(sum1);
        out_dists[2] = sqrt(sum2);
        out_dists[3] = sqrt(sum3);
    }
    return pruned_mask;
}

/**
 * framedist_batch_cutoff_1x8_double() - 1 query vs 8 anchors with early-cutoff for double.
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @size:      Number of elements in each array.
 * @cutoff_sq: Squared distance threshold for early exit (<= 0.0 disables cutoff).
 * @out_dists: Array of 8 doubles to receive computed distances.
 *
 * Return: Bitmask (0x00 to 0xFF) where bit (1 << b) is 1 if candidate b exceeded cutoff_sq.
 */
int framedist_batch_cutoff_1x8_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    long                          size,
    double                        cutoff_sq,
    double *restrict              out_dists)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 8)
    {
        return calc_dist8_cutoff_d64_avx512(q, anchors, size, cutoff_sq, out_dists);
    }
#endif
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    double sum4 = 0.0, sum5 = 0.0, sum6 = 0.0, sum7 = 0.0;
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
            if (size >= 512 && (i & 31) == 0)
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

            if (cutoff_sq > 0.0 && ((i & 63) == 60))
            {
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

                double c0 = _mm_cvtsd_f64(_mm_add_sd(s0, _mm_unpackhi_pd(s0, s0)));
                double c1 = _mm_cvtsd_f64(_mm_add_sd(s1, _mm_unpackhi_pd(s1, s1)));
                double c2 = _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
                double c3 = _mm_cvtsd_f64(_mm_add_sd(s3, _mm_unpackhi_pd(s3, s3)));
                double c4 = _mm_cvtsd_f64(_mm_add_sd(s4, _mm_unpackhi_pd(s4, s4)));
                double c5 = _mm_cvtsd_f64(_mm_add_sd(s5, _mm_unpackhi_pd(s5, s5)));
                double c6 = _mm_cvtsd_f64(_mm_add_sd(s6, _mm_unpackhi_pd(s6, s6)));
                double c7 = _mm_cvtsd_f64(_mm_add_sd(s7, _mm_unpackhi_pd(s7, s7)));

                if (c0 >= cutoff_sq && c1 >= cutoff_sq && c2 >= cutoff_sq && c3 >= cutoff_sq &&
                    c4 >= cutoff_sq && c5 >= cutoff_sq && c6 >= cutoff_sq && c7 >= cutoff_sq)
                {
                    double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                    for (int k = 0; k < 8; k++)
                    {
                        out_dists[k] = dist_exceeded;
                    }
                    return 0xFF;
                }
            }
        } // for (; i <= size - 4; i += 4)

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
        double diff0 = q_val - a0[i]; sum0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; sum1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; sum2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; sum3 += diff3 * diff3;
        double diff4 = q_val - a4[i]; sum4 += diff4 * diff4;
        double diff5 = q_val - a5[i]; sum5 += diff5 * diff5;
        double diff6 = q_val - a6[i]; sum6 += diff6 * diff6;
        double diff7 = q_val - a7[i]; sum7 += diff7 * diff7;

        if (cutoff_sq > 0.0 && ((i & 63) == 63))
        {
            if (sum0 >= cutoff_sq && sum1 >= cutoff_sq &&
                sum2 >= cutoff_sq && sum3 >= cutoff_sq &&
                sum4 >= cutoff_sq && sum5 >= cutoff_sq &&
                sum6 >= cutoff_sq && sum7 >= cutoff_sq)
            {
                double dist_exceeded = sqrt(cutoff_sq) + 1.0;
                for (int k = 0; k < 8; k++)
                {
                    out_dists[k] = dist_exceeded;
                }
                return 0xFF;
            }
        }
    } // for (; i < size; i++)

    int pruned_mask = 0;
    if (cutoff_sq > 0.0)
    {
        double dist_exceeded = sqrt(cutoff_sq) + 1.0;
        double sums[8] = {sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7};
        for (int k = 0; k < 8; k++)
        {
            if (sums[k] >= cutoff_sq)
            {
                pruned_mask |= (1 << k);
                out_dists[k] = dist_exceeded;
            }
            else
            {
                out_dists[k] = sqrt(sums[k]);
            }
        }
    }
    else
    {
        out_dists[0] = sqrt(sum0);
        out_dists[1] = sqrt(sum1);
        out_dists[2] = sqrt(sum2);
        out_dists[3] = sqrt(sum3);
        out_dists[4] = sqrt(sum4);
        out_dists[5] = sqrt(sum5);
        out_dists[6] = sqrt(sum6);
        out_dists[7] = sqrt(sum7);
    }
    return pruned_mask;
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
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 16)
    {
        while (k + 16 <= n_anchors)
        {
            calc_dist16_f32_avx512(q, &anchors[k], &out_dists[k], size);
            k += 16;
        }
    }
#endif
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

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static void calc_dist8_d64_avx512(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size)
{
    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];
    const double *restrict a4 = anchors[4];
    const double *restrict a5 = anchors[5];
    const double *restrict a6 = anchors[6];
    const double *restrict a7 = anchors[7];

    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();
    __m512d acc4 = _mm512_setzero_pd();
    __m512d acc5 = _mm512_setzero_pd();
    __m512d acc6 = _mm512_setzero_pd();
    __m512d acc7 = _mm512_setzero_pd();

    long i = 0;
    for (; i <= size - 8; i += 8)
    {
        if (size >= 512 && (i & 31) == 0)
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

        __m512d vq = _mm512_loadu_pd(&q[i]);
        __m512d d0 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a0[i]));
        __m512d d1 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a1[i]));
        __m512d d2 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a2[i]));
        __m512d d3 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a3[i]));
        __m512d d4 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a4[i]));
        __m512d d5 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a5[i]));
        __m512d d6 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a6[i]));
        __m512d d7 = _mm512_sub_pd(vq, _mm512_loadu_pd(&a7[i]));

        acc0 = _mm512_fmadd_pd(d0, d0, acc0);
        acc1 = _mm512_fmadd_pd(d1, d1, acc1);
        acc2 = _mm512_fmadd_pd(d2, d2, acc2);
        acc3 = _mm512_fmadd_pd(d3, d3, acc3);
        acc4 = _mm512_fmadd_pd(d4, d4, acc4);
        acc5 = _mm512_fmadd_pd(d5, d5, acc5);
        acc6 = _mm512_fmadd_pd(d6, d6, acc6);
        acc7 = _mm512_fmadd_pd(d7, d7, acc7);
    } // for (; i <= size - 8; i += 8)

    double s0 = _mm512_reduce_add_pd(acc0);
    double s1 = _mm512_reduce_add_pd(acc1);
    double s2 = _mm512_reduce_add_pd(acc2);
    double s3 = _mm512_reduce_add_pd(acc3);
    double s4 = _mm512_reduce_add_pd(acc4);
    double s5 = _mm512_reduce_add_pd(acc5);
    double s6 = _mm512_reduce_add_pd(acc6);
    double s7 = _mm512_reduce_add_pd(acc7);

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i]; s0 += diff0 * diff0;
        double diff1 = q_val - a1[i]; s1 += diff1 * diff1;
        double diff2 = q_val - a2[i]; s2 += diff2 * diff2;
        double diff3 = q_val - a3[i]; s3 += diff3 * diff3;
        double diff4 = q_val - a4[i]; s4 += diff4 * diff4;
        double diff5 = q_val - a5[i]; s5 += diff5 * diff5;
        double diff6 = q_val - a6[i]; s6 += diff6 * diff6;
        double diff7 = q_val - a7[i]; s7 += diff7 * diff7;
    }

    out_dists[0] = sqrt(s0);
    out_dists[1] = sqrt(s1);
    out_dists[2] = sqrt(s2);
    out_dists[3] = sqrt(s3);
    out_dists[4] = sqrt(s4);
    out_dists[5] = sqrt(s5);
    out_dists[6] = sqrt(s6);
    out_dists[7] = sqrt(s7);
}
#endif

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
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && size >= 8)
    {
        calc_dist8_d64_avx512(q, anchors, out_dists, size);
        return;
    }
#endif
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


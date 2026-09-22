/**
 * @file eq16_dist.c
 * @brief Asymmetric Distance Computation (ADC) and SSD distance kernels for EQ16.
 */

#include "eq16_quant.h"
#include "gric_simd.h"
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif
#include <math.h>
#include <string.h>

uint64_t eq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim)
{
    long i = 0;
    uint64_t total = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2 && dim >= 16)
    {
        __m256i sum_lo = _mm256_setzero_si256();
        __m256i sum_hi = _mm256_setzero_si256();

        for (; i <= dim - 16; i += 16)
        {
            __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));

            __m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
            __m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
            __m256i d_lo = _mm256_sub_epi32(a_lo, b_lo);

            __m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
            __m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
            __m256i d_hi = _mm256_sub_epi32(a_hi, b_hi);

            __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
            __m256i d_lo_o = _mm256_srli_si256(d_lo, 4);
            __m256i p_lo_o = _mm256_mul_epi32(d_lo_o, d_lo_o);

            __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
            __m256i d_hi_o = _mm256_srli_si256(d_hi, 4);
            __m256i p_hi_o = _mm256_mul_epi32(d_hi_o, d_hi_o);

            sum_lo = _mm256_add_epi64(sum_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
            sum_hi = _mm256_add_epi64(sum_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
        } // for (; i <= dim - 16; i += 16)

        __m256i s = _mm256_add_epi64(sum_lo, sum_hi);
        __m128i r = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        total = (uint64_t)_mm_cvtsi128_si64(r) + (uint64_t)_mm_extract_epi64(r, 1);
    }
#endif

    for (; i < dim; i++)
    {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i];
        total += (uint64_t)(diff * diff);
    } // for (; i < dim; i++)

    return total;
}

/**
 * eq16_dist_squared_cutoff_i16() - Compute squared difference with early cutoff.
 * @a:          First vector [dim].
 * @b:          Second vector [dim].
 * @dim:        Vector dimension.
 * @ssd_cutoff: Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences, or ssd_cutoff + 1 if threshold exceeded.
 */
uint64_t eq16_dist_squared_cutoff_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    long i = 0;
    uint64_t total = 0;

    if (dim < 16)
    {
        for (long k = 0; k < dim; k++)
        {
            int64_t diff = (int64_t)a[k] - (int64_t)b[k];
            total += (uint64_t)(diff * diff);
            if (total > ssd_cutoff)
            {
                return ssd_cutoff + 1;
            }
        } // for (long k = 0; k < dim; k++)
        return total;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        __m256i sum_lo = _mm256_setzero_si256();
        __m256i sum_hi = _mm256_setzero_si256();

        for (; i <= dim - 16; i += 16)
        {
            __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));

            __m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
            __m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
            __m256i d_lo = _mm256_sub_epi32(a_lo, b_lo);

            __m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
            __m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
            __m256i d_hi = _mm256_sub_epi32(a_hi, b_hi);

            __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
            __m256i d_lo_o = _mm256_srli_si256(d_lo, 4);
            __m256i p_lo_o = _mm256_mul_epi32(d_lo_o, d_lo_o);

            __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
            __m256i d_hi_o = _mm256_srli_si256(d_hi, 4);
            __m256i p_hi_o = _mm256_mul_epi32(d_hi_o, d_hi_o);

            sum_lo = _mm256_add_epi64(sum_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
            sum_hi = _mm256_add_epi64(sum_hi, _mm256_add_epi64(p_hi_e, p_hi_o));

            if ((i & 31) == 16)
            {
                __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
                __m128i slo = _mm256_castsi256_si128(sum);
                __m128i shi = _mm256_extracti128_si256(sum, 1);
                __m128i s128 = _mm_add_epi64(slo, shi);
                uint64_t partial = (uint64_t)_mm_cvtsi128_si64(s128) +
                                   (uint64_t)_mm_extract_epi64(s128, 1);
                if (partial > ssd_cutoff)
                {
                    return ssd_cutoff + 1;
                }
            }
        } // for (; i <= dim - 16; i += 16)

        __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
        __m128i slo = _mm256_castsi256_si128(sum);
        __m128i shi = _mm256_extracti128_si256(sum, 1);
        __m128i s128 = _mm_add_epi64(slo, shi);
        total = (uint64_t)_mm_cvtsi128_si64(s128) +
                (uint64_t)_mm_extract_epi64(s128, 1);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    }
#endif

    for (; i < dim; i++)
    {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i];
        total += (uint64_t)(diff * diff);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i < dim; i++)

    return total;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline float hsum256_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(vlow, vhigh);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    return _mm_cvtss_f32(sum1);
}
#endif

static float eq16_dist_asym_cutoff_scalar(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    float total = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if ((i & 63) == 63 && total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (long i = 0; i < dim; i++)

    return total;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
GRIC_TARGET_AVX2
static float eq16_dist_asym_cutoff_avx2(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    long i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (; i <= dim - 16; i += 16)
    {
        __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i));
        __m256i c32_0 = _mm256_cvtepi16_epi32(c16_0);
        __m256  cf_0  = _mm256_cvtepi32_ps(c32_0);
        __m256  q_0   = _mm256_loadu_ps(q_scaled + i);
        __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i + 8));
        __m256i c32_1 = _mm256_cvtepi16_epi32(c16_1);
        __m256  cf_1  = _mm256_cvtepi32_ps(c32_1);
        __m256  q_1   = _mm256_loadu_ps(q_scaled + i + 8);
        __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        if ((i & 31) == 16)
        {
            float partial = hsum256_ps(_mm256_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    float total = hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}
#endif // x86 / AVX2

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static float eq16_dist_asym_cutoff_avx512(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    long i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    for (; i <= dim - 32; i += 32)
    {
        __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i));
        __m512i c32_0 = _mm512_cvtepi16_epi32(c16_0);
        __m512  cf_0  = _mm512_cvtepi32_ps(c32_0);
        __m512  q_0   = _mm512_loadu_ps(q_scaled + i);
        __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i + 16));
        __m512i c32_1 = _mm512_cvtepi16_epi32(c16_1);
        __m512  cf_1  = _mm512_cvtepi32_ps(c32_1);
        __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);
        __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        float partial = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        if (partial > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i <= dim - 32; i += 32)

    float total = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_dist_asym_cutoff_f32() - Asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cand_eq16: Quantized candidate int16 vector [dim].
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
float eq16_dist_asym_cutoff_f32(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    if (dim < 16)
    {
        return eq16_dist_asym_cutoff_scalar(q_scaled, cand_eq16, dim, cutoff);
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return eq16_dist_asym_cutoff_avx512(q_scaled, cand_eq16, dim, cutoff);
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return eq16_dist_asym_cutoff_avx2(q_scaled, cand_eq16, dim, cutoff);
    }
#endif

    return eq16_dist_asym_cutoff_scalar(q_scaled, cand_eq16, dim, cutoff);
}

/**
 * eq16_dist_asym_cutoff_batch_1x4_scalar() - Scalar 1x4 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 4 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 4 float squared distances.
 */
static void eq16_dist_asym_cutoff_batch_1x4_scalar(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        d0 += diff0 * diff0;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
        d3 += diff3 * diff3;

        if ((i & 63) == 63)
        {
            if (d0 > cutoff && d1 > cutoff && d2 > cutoff && d3 > cutoff)
            {
                out_dists[0] = d0;
                out_dists[1] = d1;
                out_dists[2] = d2;
                out_dists[3] = d3;
                return;
            }
        }
    } // for (long i = 0; i < dim; i++)

    out_dists[0] = d0;
    out_dists[1] = d1;
    out_dists[2] = d2;
    out_dists[3] = d3;
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_scalar() - Scalar 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
static void eq16_dist_asym_cutoff_batch_1x8_scalar(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float d4 = 0.0f;
    float d5 = 0.0f;
    float d6 = 0.0f;
    float d7 = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        d0 += diff0 * diff0;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
        d3 += diff3 * diff3;
        d4 += diff4 * diff4;
        d5 += diff5 * diff5;
        d6 += diff6 * diff6;
        d7 += diff7 * diff7;

        if ((i & 63) == 63)
        {
            if (d0 > cutoff && d1 > cutoff && d2 > cutoff && d3 > cutoff &&
                d4 > cutoff && d5 > cutoff && d6 > cutoff && d7 > cutoff)
            {
                out_dists[0] = d0;
                out_dists[1] = d1;
                out_dists[2] = d2;
                out_dists[3] = d3;
                out_dists[4] = d4;
                out_dists[5] = d5;
                out_dists[6] = d6;
                out_dists[7] = d7;
                return;
            }
        }
    } // for (long i = 0; i < dim; i++)

    out_dists[0] = d0;
    out_dists[1] = d1;
    out_dists[2] = d2;
    out_dists[3] = d3;
    out_dists[4] = d4;
    out_dists[5] = d5;
    out_dists[6] = d6;
    out_dists[7] = d7;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline __m128 eq16_reduce_4x256_ps(
    __m256 acc0,
    __m256 acc1,
    __m256 acc2,
    __m256 acc3)
{
    __m128 lo0 = _mm256_castps256_ps128(acc0);
    __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
    __m128 s0  = _mm_add_ps(lo0, hi0);

    __m128 lo1 = _mm256_castps256_ps128(acc1);
    __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
    __m128 s1  = _mm_add_ps(lo1, hi1);

    __m128 lo2 = _mm256_castps256_ps128(acc2);
    __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
    __m128 s2  = _mm_add_ps(lo2, hi2);

    __m128 lo3 = _mm256_castps256_ps128(acc3);
    __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
    __m128 s3  = _mm_add_ps(lo3, hi3);

    __m128 h01 = _mm_hadd_ps(s0, s1);
    __m128 h23 = _mm_hadd_ps(s2, s3);
    return _mm_hadd_ps(h01, h23);
}

/**
 * eq16_dist_asym_resume_cutoff_avx2() - Resume single candidate asymmetric distance.
 * @q_scaled:  Normalized query float vector.
 * @cand_eq16: Quantized candidate int16 vector.
 * @start_dim: Starting dimension index (multiple of 16).
 * @dim:       Vector dimension.
 * @init_sum:  Accumulated distance from earlier dimensions.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
GRIC_TARGET_AVX2
static float eq16_dist_asym_resume_cutoff_avx2(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    start_dim,
    long                    dim,
    float                   init_sum,
    float                   cutoff)
{
    long i = start_dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float total = init_sum;

    for (; i <= dim - 16; i += 16)
    {
        __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i));
        __m256i c32_0 = _mm256_cvtepi16_epi32(c16_0);
        __m256  cf_0  = _mm256_cvtepi32_ps(c32_0);
        __m256  q_0   = _mm256_loadu_ps(q_scaled + i);
        __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i + 8));
        __m256i c32_1 = _mm256_cvtepi16_epi32(c16_1);
        __m256  cf_1  = _mm256_cvtepi32_ps(c32_1);
        __m256  q_1   = _mm256_loadu_ps(q_scaled + i + 8);
        __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        if ((i & 31) == 16)
        {
            float partial = total + hsum256_ps(_mm256_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    total += hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}

GRIC_TARGET_AVX2
static void eq16_dist_asym_cutoff_batch_1x4_avx2(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 16; i += 16)
    {
        /* First 8 dimensions */
        {
            __m256 q_0 = _mm256_loadu_ps(q_scaled + i);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_0, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_0, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_0, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        /* Second 8 dimensions */
        {
            __m256 q_1 = _mm256_loadu_ps(q_scaled + i + 8);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i + 8));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_1, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i + 8));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i + 8));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_1, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i + 8));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_1, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        /* Periodic cutoff checkpoint (every 32 dimensions) */
        if ((i & 31) == 16)
        {
            __m128 sums = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
            __m128 cmp = _mm_cmpgt_ps(sums, vcutoff);
            int dead_mask = _mm_movemask_ps(cmp);
            if (dead_mask == 0xF)
            {
                _mm_storeu_ps(out_dists, sums);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 3 && i + 16 < dim)
            {
                _mm_storeu_ps(out_dists, sums);
                for (int k = 0; k < 4; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx2(
                            q_scaled, cands[k], i + 16, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    __m128 sums = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
    _mm_storeu_ps(out_dists, sums);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
    } // for (; i < dim; i++)
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_avx2() - AVX2 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
GRIC_TARGET_AVX2
static void eq16_dist_asym_cutoff_batch_1x8_avx2(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 16; i += 16)
    {
        /* First 8 dimensions */
        {
            __m256 q_0 = _mm256_loadu_ps(q_scaled + i);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_0, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_0, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_0, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

            __m128i c16_4 = _mm_loadu_si128((const __m128i *)(const void *)(c4 + i));
            __m256  cf_4  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_4));
            __m256  diff4 = _mm256_sub_ps(q_0, cf_4);
            acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);

            __m128i c16_5 = _mm_loadu_si128((const __m128i *)(const void *)(c5 + i));
            __m256  cf_5  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_5));
            __m256  diff5 = _mm256_sub_ps(q_0, cf_5);
            acc5 = _mm256_fmadd_ps(diff5, diff5, acc5);

            __m128i c16_6 = _mm_loadu_si128((const __m128i *)(const void *)(c6 + i));
            __m256  cf_6  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_6));
            __m256  diff6 = _mm256_sub_ps(q_0, cf_6);
            acc6 = _mm256_fmadd_ps(diff6, diff6, acc6);

            __m128i c16_7 = _mm_loadu_si128((const __m128i *)(const void *)(c7 + i));
            __m256  cf_7  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_7));
            __m256  diff7 = _mm256_sub_ps(q_0, cf_7);
            acc7 = _mm256_fmadd_ps(diff7, diff7, acc7);
        }

        /* Second 8 dimensions */
        {
            __m256 q_1 = _mm256_loadu_ps(q_scaled + i + 8);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i + 8));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_1, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i + 8));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i + 8));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_1, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i + 8));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_1, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

            __m128i c16_4 = _mm_loadu_si128((const __m128i *)(const void *)(c4 + i + 8));
            __m256  cf_4  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_4));
            __m256  diff4 = _mm256_sub_ps(q_1, cf_4);
            acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);

            __m128i c16_5 = _mm_loadu_si128((const __m128i *)(const void *)(c5 + i + 8));
            __m256  cf_5  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_5));
            __m256  diff5 = _mm256_sub_ps(q_1, cf_5);
            acc5 = _mm256_fmadd_ps(diff5, diff5, acc5);

            __m128i c16_6 = _mm_loadu_si128((const __m128i *)(const void *)(c6 + i + 8));
            __m256  cf_6  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_6));
            __m256  diff6 = _mm256_sub_ps(q_1, cf_6);
            acc6 = _mm256_fmadd_ps(diff6, diff6, acc6);

            __m128i c16_7 = _mm_loadu_si128((const __m128i *)(const void *)(c7 + i + 8));
            __m256  cf_7  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_7));
            __m256  diff7 = _mm256_sub_ps(q_1, cf_7);
            acc7 = _mm256_fmadd_ps(diff7, diff7, acc7);
        }

        /* Periodic cutoff checkpoint (every 32 dimensions) */
        if ((i & 31) == 16)
        {
            __m128 sums_lo = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
            __m128 sums_hi = eq16_reduce_4x256_ps(acc4, acc5, acc6, acc7);
            __m128 cmp_lo = _mm_cmpgt_ps(sums_lo, vcutoff);
            __m128 cmp_hi = _mm_cmpgt_ps(sums_hi, vcutoff);
            int m_lo = _mm_movemask_ps(cmp_lo);
            int m_hi = _mm_movemask_ps(cmp_hi);
            int dead_mask = m_lo | (m_hi << 4);

            if (dead_mask == 0xFF)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 6 && i + 16 < dim)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);

                for (int k = 0; k < 8; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx2(
                            q_scaled, cands[k], i + 16, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    __m128 sums_lo = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
    __m128 sums_hi = eq16_reduce_4x256_ps(acc4, acc5, acc6, acc7);
    _mm_storeu_ps(out_dists, sums_lo);
    _mm_storeu_ps(out_dists + 4, sums_hi);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
        out_dists[4] += diff4 * diff4;
        out_dists[5] += diff5 * diff5;
        out_dists[6] += diff6 * diff6;
        out_dists[7] += diff7 * diff7;
    } // for (; i < dim; i++)
}
#endif // x86 AVX2

#if GRIC_HAVE_AVX512_TARGET
/**
 * eq16_dist_asym_resume_cutoff_avx512() - Resume single candidate distance.
 * @q_scaled:  Normalized query float vector.
 * @cand_eq16: Quantized candidate int16 vector.
 * @start_dim: Starting dimension index (multiple of 32).
 * @dim:       Vector dimension.
 * @init_sum:  Accumulated distance from earlier dimensions.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
GRIC_TARGET_AVX512
static float eq16_dist_asym_resume_cutoff_avx512(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    start_dim,
    long                    dim,
    float                   init_sum,
    float                   cutoff)
{
    long i = start_dim;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    float total = init_sum;

    for (; i <= dim - 32; i += 32)
    {
        __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i));
        __m512i c32_0 = _mm512_cvtepi16_epi32(c16_0);
        __m512  cf_0  = _mm512_cvtepi32_ps(c32_0);
        __m512  q_0   = _mm512_loadu_ps(q_scaled + i);
        __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i + 16));
        __m512i c32_1 = _mm512_cvtepi16_epi32(c16_1);
        __m512  cf_1  = _mm512_cvtepi32_ps(c32_1);
        __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);
        __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        float partial = total + _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        if (partial > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i <= dim - 32; i += 32)

    total += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}

GRIC_TARGET_AVX512
static void eq16_dist_asym_cutoff_batch_1x4_avx512(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 32; i += 32)
    {
        /* First 16 dimensions */
        {
            __m512  q_0   = _mm512_loadu_ps(q_scaled + i);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_0, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_0, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_0, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        /* Second 16 dimensions */
        {
            __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i + 16));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_1, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i + 16));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i + 16));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_1, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i + 16));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_1, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        /* Periodic cutoff checkpoint (every 32 dimensions) */
        {
            __m128 sums = _mm_set_ps(
                _mm512_reduce_add_ps(acc3),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc0));
            __m128 cmp = _mm_cmpgt_ps(sums, vcutoff);
            int dead_mask = _mm_movemask_ps(cmp);
            if (dead_mask == 0xF)
            {
                _mm_storeu_ps(out_dists, sums);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            int thresh = (i == 0) ? 2 : 1;
            if (dead_count >= thresh && i + 32 < dim)
            {
                _mm_storeu_ps(out_dists, sums);
                for (int k = 0; k < 4; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx512(
                            q_scaled, cands[k], i + 32, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    __m128 sums = _mm_set_ps(
        _mm512_reduce_add_ps(acc3),
        _mm512_reduce_add_ps(acc2),
        _mm512_reduce_add_ps(acc1),
        _mm512_reduce_add_ps(acc0));
    _mm_storeu_ps(out_dists, sums);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
    } // for (; i < dim; i++)
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_avx512() - AVX512 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
GRIC_TARGET_AVX512
static void eq16_dist_asym_cutoff_batch_1x8_avx512(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 32; i += 32)
    {
        /* First 16 dimensions */
        {
            __m512 q_0 = _mm512_loadu_ps(q_scaled + i);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_0, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_0, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_0, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

            __m256i c16_4 = _mm256_loadu_si256((const __m256i *)(const void *)(c4 + i));
            __m512  cf_4  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_4));
            __m512  diff4 = _mm512_sub_ps(q_0, cf_4);
            acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);

            __m256i c16_5 = _mm256_loadu_si256((const __m256i *)(const void *)(c5 + i));
            __m512  cf_5  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_5));
            __m512  diff5 = _mm512_sub_ps(q_0, cf_5);
            acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);

            __m256i c16_6 = _mm256_loadu_si256((const __m256i *)(const void *)(c6 + i));
            __m512  cf_6  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_6));
            __m512  diff6 = _mm512_sub_ps(q_0, cf_6);
            acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);

            __m256i c16_7 = _mm256_loadu_si256((const __m256i *)(const void *)(c7 + i));
            __m512  cf_7  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_7));
            __m512  diff7 = _mm512_sub_ps(q_0, cf_7);
            acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);
        }

        /* Second 16 dimensions */
        {
            __m512 q_1 = _mm512_loadu_ps(q_scaled + i + 16);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i + 16));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_1, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i + 16));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i + 16));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_1, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i + 16));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_1, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

            __m256i c16_4 = _mm256_loadu_si256((const __m256i *)(const void *)(c4 + i + 16));
            __m512  cf_4  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_4));
            __m512  diff4 = _mm512_sub_ps(q_1, cf_4);
            acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);

            __m256i c16_5 = _mm256_loadu_si256((const __m256i *)(const void *)(c5 + i + 16));
            __m512  cf_5  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_5));
            __m512  diff5 = _mm512_sub_ps(q_1, cf_5);
            acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);

            __m256i c16_6 = _mm256_loadu_si256((const __m256i *)(const void *)(c6 + i + 16));
            __m512  cf_6  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_6));
            __m512  diff6 = _mm512_sub_ps(q_1, cf_6);
            acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);

            __m256i c16_7 = _mm256_loadu_si256((const __m256i *)(const void *)(c7 + i + 16));
            __m512  cf_7  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_7));
            __m512  diff7 = _mm512_sub_ps(q_1, cf_7);
            acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);
        }

        /* Periodic cutoff checkpoint (every 32 dimensions) */
        {
            __m128 sums_lo = _mm_set_ps(
                _mm512_reduce_add_ps(acc3),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc0));
            __m128 sums_hi = _mm_set_ps(
                _mm512_reduce_add_ps(acc7),
                _mm512_reduce_add_ps(acc6),
                _mm512_reduce_add_ps(acc5),
                _mm512_reduce_add_ps(acc4));
            __m128 cmp_lo = _mm_cmpgt_ps(sums_lo, vcutoff);
            __m128 cmp_hi = _mm_cmpgt_ps(sums_hi, vcutoff);
            int m_lo = _mm_movemask_ps(cmp_lo);
            int m_hi = _mm_movemask_ps(cmp_hi);
            int dead_mask = m_lo | (m_hi << 4);

            if (dead_mask == 0xFF)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            int thresh = (i == 0) ? 4 : ((i == 32) ? 3 : 2);
            if (dead_count >= thresh && i + 32 < dim)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);

                for (int k = 0; k < 8; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx512(
                            q_scaled, cands[k], i + 32, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    __m128 sums_lo = _mm_set_ps(
        _mm512_reduce_add_ps(acc3),
        _mm512_reduce_add_ps(acc2),
        _mm512_reduce_add_ps(acc1),
        _mm512_reduce_add_ps(acc0));
    __m128 sums_hi = _mm_set_ps(
        _mm512_reduce_add_ps(acc7),
        _mm512_reduce_add_ps(acc6),
        _mm512_reduce_add_ps(acc5),
        _mm512_reduce_add_ps(acc4));
    _mm_storeu_ps(out_dists, sums_lo);
    _mm_storeu_ps(out_dists + 4, sums_hi);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
        out_dists[4] += diff4 * diff4;
        out_dists[5] += diff5 * diff5;
        out_dists[6] += diff6 * diff6;
        out_dists[7] += diff7 * diff7;
    } // for (; i < dim; i++)
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_dist_asym_cutoff_batch_1x4() - 1x4 asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 4 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 4 float squared distances.
 */
void eq16_dist_asym_cutoff_batch_1x4(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    if (dim < 16)
    {
        eq16_dist_asym_cutoff_batch_1x4_scalar(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 32)
    {
        eq16_dist_asym_cutoff_batch_1x4_avx512(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        eq16_dist_asym_cutoff_batch_1x4_avx2(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

    eq16_dist_asym_cutoff_batch_1x4_scalar(q_scaled, cands, dim, cutoff, out_dists);
}

/**
 * eq16_dist_asym_cutoff_batch_1x8() - 1x8 asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
void eq16_dist_asym_cutoff_batch_1x8(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    if (dim < 16)
    {
        eq16_dist_asym_cutoff_batch_1x8_scalar(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 32)
    {
        eq16_dist_asym_cutoff_batch_1x8_avx512(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        eq16_dist_asym_cutoff_batch_1x8_avx2(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

    eq16_dist_asym_cutoff_batch_1x8_scalar(q_scaled, cands, dim, cutoff, out_dists);
}

/**
 * eq16_refine_candidates_adc() - Refine candidate clusters using batched 1x4 ADC computation.
 * @cur_adc:          Normalized query float vector [dim].
 * @mat_eq16:         Quantized anchor matrix [num_clusters * dim].
 * @active_clusters:  Array of surviving candidate cluster indices (compacted in-place).
 * @num_active:       Number of input candidate clusters.
 * @dim:              Vector dimension.
 * @cutoff:           ADC cutoff threshold on sum of squared differences.
 * @clmembflag:       Cluster membership flag array (set to 0 for pruned clusters).
 * @out_num_active:   Output pointer to number of surviving clusters.
 * @out_pruned_count: Output pointer to accumulated pruned cluster count (may be NULL).
 */
void eq16_refine_candidates_adc(
    const float   *restrict cur_adc,
    const int16_t *restrict mat_eq16,
    int           *restrict active_clusters,
    int                     num_active,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    int next_num_active = 0;
    long pruned = 0;
    int idx = 0;

    for (; idx <= num_active - 8; idx += 8)
    {
        const int16_t *cands[8];
        for (int k = 0; k < 8; k++)
        {
            int c = active_clusters[idx + k];
            cands[k] = mat_eq16 + (size_t)c * (size_t)dim;
        }

        float dsq[8];
        eq16_dist_asym_cutoff_batch_1x8(cur_adc, cands, dim, cutoff, dsq);

        for (int k = 0; k < 8; k++)
        {
            int c = active_clusters[idx + k];
            if (dsq[k] > cutoff)
            {
                clmembflag[c] = 0;
                pruned++;
            }
            else
            {
                active_clusters[next_num_active++] = c;
            }
        }
    } // for (; idx <= num_active - 8; idx += 8)

    for (; idx <= num_active - 4; idx += 4)
    {
        int c0 = active_clusters[idx];
        int c1 = active_clusters[idx + 1];
        int c2 = active_clusters[idx + 2];
        int c3 = active_clusters[idx + 3];

        const int16_t *cands[4];
        cands[0] = mat_eq16 + (size_t)c0 * (size_t)dim;
        cands[1] = mat_eq16 + (size_t)c1 * (size_t)dim;
        cands[2] = mat_eq16 + (size_t)c2 * (size_t)dim;
        cands[3] = mat_eq16 + (size_t)c3 * (size_t)dim;

        float dsq[4];
        eq16_dist_asym_cutoff_batch_1x4(cur_adc, cands, dim, cutoff, dsq);

        if (dsq[0] > cutoff)
        {
            clmembflag[c0] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c0;
        }

        if (dsq[1] > cutoff)
        {
            clmembflag[c1] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c1;
        }

        if (dsq[2] > cutoff)
        {
            clmembflag[c2] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c2;
        }

        if (dsq[3] > cutoff)
        {
            clmembflag[c3] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c3;
        }
    } // for (; idx <= num_active - 4; idx += 4)

    for (; idx < num_active; idx++)
    {
        int c = active_clusters[idx];
        const int16_t *cand_anchor = mat_eq16 + (size_t)c * (size_t)dim;
        float dsq = eq16_dist_asym_cutoff_f32(cur_adc, cand_anchor, dim, cutoff);

        if (dsq > cutoff)
        {
            clmembflag[c] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c;
        }
    } // for (; idx < num_active; idx++)

    *out_num_active = next_num_active;
    if (out_pruned_count != NULL)
    {
        *out_pruned_count += pruned;
    }
}

/**
 * eq16_compute_lower_bound() - Guaranteed metric lower bound between two EQ16 vectors.
 * @a:       First EQ16 vector.
 * @b:       Second EQ16 vector.
 * @params:  EQ16 parameter struct.
 * @epsilon: Relative relaxation factor (0.0 for exact metric bound).
 *
 * Return: Conservative lower bound on true Euclidean distance.
 */
double eq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const EQ16Params *restrict params,
    double                     epsilon)
{
    uint64_t ssd = eq16_dist_squared_i16(a, b, params->dim);
    double d_quant = sqrt((double)ssd) * ((double)params->scale * 0.5);
    double lb = d_quant - 2.0 * (double)params->err_radius;

    if (lb < 0.0)
    {
        lb = 0.0;
    }

    if (epsilon > 0.0)
    {
        lb /= (1.0 + epsilon);
    }

    return lb;
}

/**
 * eq16_dist_squared_batch_1x4_i16() - Compute SSD from 1 query against 4 anchors in SIMD.
 * @q:            Query EQ16 vector.
 * @anchors:      Array of 4 pointers to anchor vectors.
 * @out_sq_dists: Output array of 4 uint64_t SSD values.
 * @dim:          Vector dimension.
 */
void eq16_dist_squared_batch_1x4_i16(
    const int16_t *restrict        q,
    const int16_t *const *restrict anchors,
    uint64_t *restrict             out_sq_dists,
    long                           dim)
{
    long i = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2 && dim >= 16)
    {
        __m256i sum0_lo = _mm256_setzero_si256(), sum0_hi = _mm256_setzero_si256();
        __m256i sum1_lo = _mm256_setzero_si256(), sum1_hi = _mm256_setzero_si256();
        __m256i sum2_lo = _mm256_setzero_si256(), sum2_hi = _mm256_setzero_si256();
        __m256i sum3_lo = _mm256_setzero_si256(), sum3_hi = _mm256_setzero_si256();

        const int16_t *a0 = anchors[0];
        const int16_t *a1 = anchors[1];
        const int16_t *a2 = anchors[2];
        const int16_t *a3 = anchors[3];

        for (; i <= dim - 16; i += 16)
        {
            __m256i vq = _mm256_loadu_si256((const __m256i *)(const void *)(q + i));
            __m256i q_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vq));
            __m256i q_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vq, 1));

            /* Anchor 0 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a0 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum0_lo = _mm256_add_epi64(sum0_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum0_hi = _mm256_add_epi64(sum0_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 1 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a1 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum1_lo = _mm256_add_epi64(sum1_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum1_hi = _mm256_add_epi64(sum1_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 2 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a2 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum2_lo = _mm256_add_epi64(sum2_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum2_hi = _mm256_add_epi64(sum2_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 3 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a3 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum3_lo = _mm256_add_epi64(sum3_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum3_hi = _mm256_add_epi64(sum3_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }
        } // for (; i <= dim - 16; i += 16)

        __m256i s0 = _mm256_add_epi64(sum0_lo, sum0_hi);
        __m128i r0 = _mm_add_epi64(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1));
        out_sq_dists[0] = (uint64_t)_mm_cvtsi128_si64(r0) + (uint64_t)_mm_extract_epi64(r0, 1);

        __m256i s1 = _mm256_add_epi64(sum1_lo, sum1_hi);
        __m128i r1 = _mm_add_epi64(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1));
        out_sq_dists[1] = (uint64_t)_mm_cvtsi128_si64(r1) + (uint64_t)_mm_extract_epi64(r1, 1);

        __m256i s2 = _mm256_add_epi64(sum2_lo, sum2_hi);
        __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(s2), _mm256_extracti128_si256(s2, 1));
        out_sq_dists[2] = (uint64_t)_mm_cvtsi128_si64(r2) + (uint64_t)_mm_extract_epi64(r2, 1);

        __m256i s3 = _mm256_add_epi64(sum3_lo, sum3_hi);
        __m128i r3 = _mm_add_epi64(_mm256_castsi256_si128(s3), _mm256_extracti128_si256(s3, 1));
        out_sq_dists[3] = (uint64_t)_mm_cvtsi128_si64(r3) + (uint64_t)_mm_extract_epi64(r3, 1);
    }
    else
#endif
    {
        out_sq_dists[0] = 0;
        out_sq_dists[1] = 0;
        out_sq_dists[2] = 0;
        out_sq_dists[3] = 0;
    }

    for (; i < dim; i++)
    {
        int64_t qv = (int64_t)q[i];
        for (int k = 0; k < 4; k++)
        {
            int64_t diff = qv - (int64_t)anchors[k][i];
            out_sq_dists[k] += (uint64_t)(diff * diff);
        }
    } // for (; i < dim; i++)
}

/**
 * eq16_batch_filter_candidates() - Bulk filter candidates using EQ16 lower bounds.
 */

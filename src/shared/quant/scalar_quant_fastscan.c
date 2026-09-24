/**
 * @file scalar_quant_fastscan.c
 * @brief SIMD fastscan and cutoff squared distance kernels for SQ16.
 */

#include "scalar_quant.h"
#include "gric_simd.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

/**
 * @brief Compute sum of squared differences between two int16 vectors with early cutoff.
 */
uint64_t sq16_dist_squared_cutoff_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (dim == 3)
    {
        int32_t d0 = (int32_t)a[0] - (int32_t)b[0];
        uint64_t total = (uint64_t)(d0 * d0);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
        int32_t d1 = (int32_t)a[1] - (int32_t)b[1];
        total += (uint64_t)(d1 * d1);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
        int32_t d2 = (int32_t)a[2] - (int32_t)b[2];
        total += (uint64_t)(d2 * d2);
        return total;
    }
    if (dim == 2)
    {
        int32_t d0 = (int32_t)a[0] - (int32_t)b[0];
        uint64_t total = (uint64_t)(d0 * d0);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
        int32_t d1 = (int32_t)a[1] - (int32_t)b[1];
        total += (uint64_t)(d1 * d1);
        return total;
    }
    if (dim < 16)
    {
        uint64_t total = 0;
        for (long k = 0; k < dim; k++)
        {
            int32_t diff = (int32_t)a[k] - (int32_t)b[k];
            total += (uint64_t)(diff * diff);
            if (total > ssd_cutoff)
            {
                return ssd_cutoff + 1;
            }
        }
        return total;
    }

    uint64_t total = 0;
    long i = 0;

#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m512i sum_vec512_0 = _mm512_setzero_si512();
    __m512i sum_vec512_1 = _mm512_setzero_si512();

    for (; i <= dim - 32; i += 32)
    {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        __m512i diff = _mm512_sub_epi16(va, vb);
        __m512i prod = _mm512_madd_epi16(diff, diff);

        __m256i prod_lo = _mm512_castsi512_si256(prod);
        __m256i prod_hi = _mm512_extracti64x4_epi64(prod, 1);

        __m512i q0 = _mm512_cvtepi32_epi64(prod_lo);
        __m512i q1 = _mm512_cvtepi32_epi64(prod_hi);

        sum_vec512_0 = _mm512_add_epi64(sum_vec512_0, q0);
        sum_vec512_1 = _mm512_add_epi64(sum_vec512_1, q1);

        __m512i sum_tot = _mm512_add_epi64(sum_vec512_0, sum_vec512_1);
        if ((uint64_t)_mm512_reduce_add_epi64(sum_tot) > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i <= dim - 32; i += 32)

    __m512i sum_tot = _mm512_add_epi64(sum_vec512_0, sum_vec512_1);
    total += (uint64_t)_mm512_reduce_add_epi64(sum_tot);
#elif !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();

    for (; i <= dim - 16; i += 16)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));
        __m256i diff = _mm256_sub_epi16(va, vb);
        __m256i prod = _mm256_madd_epi16(diff, diff);

        __m256i plo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod));
        __m256i phi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1));

        sum_lo = _mm256_add_epi64(sum_lo, plo);
        sum_hi = _mm256_add_epi64(sum_hi, phi);

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
    total += (uint64_t)_mm_cvtsi128_si64(s128) +
             (uint64_t)_mm_extract_epi64(s128, 1);
    if (total > ssd_cutoff)
    {
        return ssd_cutoff + 1;
    }
#endif

    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        total += (uint64_t)(diff * diff);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i < dim; i++)

    return total;
}

/**
 * @brief Compute guaranteed metric lower bound between two 16-bit quantized vectors.
 */
double sq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const SQ16Params *restrict params,
    double                     epsilon);

/**
 * @brief Compute sum of squared differences between 1 query and 4 int16 anchors using AVX2.
 */
void sq16_dist_squared_batch_1x4_i16(
    const int16_t *restrict        q,
    const int16_t *const *restrict anchors,
    uint64_t *restrict             out_sq_dists,
    long                           dim);

/**
 * @brief Bulk filter cluster candidates using 16-bit scalar quantization lower bounds.
 */
int sq16_batch_filter_candidates(
    const int16_t *restrict        q_sq16,
    const int16_t *const *restrict anchor_ptrs,
    const int                     *candidate_indices,
    int                            num_candidates,
    double                         cutoff_dist,
    const SQ16Params              *params,
    int *restrict                  clmembflag);

/**
 * @brief Save quantized dataset buffer and parameters to a binary .sq16 file.
 */
int sq16_save_sidecar(
    const char       *filepath,
    const SQ16Params *params,
    const int16_t    *data,
    long              num_frames);


/**
 * sq16_fastscan_32x_3d_scalar() - Scalar fallback for 32 3D candidate evaluation.
 */
static inline uint32_t sq16_fastscan_32x_3d_scalar(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_x,
    const int16_t *restrict block_y,
    const int16_t *restrict block_z,
    uint64_t                ssd_cutoff)
{
    uint32_t mask = 0;
    int32_t qx = query_sq16[0];
    int32_t qy = query_sq16[1];
    int32_t qz = query_sq16[2];

    for (int i = 0; i < 32; i++)
    {
        int32_t dx = qx - block_x[i];
        int32_t dy = qy - block_y[i];
        int32_t dz = qz - block_z[i];
        uint64_t dist = (uint64_t)((int64_t)dx * dx + (int64_t)dy * dy + (int64_t)dz * dz);
        if (dist <= ssd_cutoff)
        {
            mask |= (1U << i);
        }
    } // for (int i = 0; i < 32; i++)

    return mask;
}

/**
 * sq16_fastscan_32x_generic_scalar() - Scalar fallback for 32 D-dim candidate evaluation.
 */
static inline uint32_t sq16_fastscan_32x_generic_scalar(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    uint32_t mask = 0;

    for (int i = 0; i < 32; i++)
    {
        uint64_t dist = 0;
        for (long d = 0; d < dim; d++)
        {
            int32_t diff = (int32_t)query_sq16[d] - (int32_t)block_coords[d * 32 + i];
            dist += (uint64_t)((int64_t)diff * diff);
            if (dist > ssd_cutoff)
            {
                break;
            }
        } // for (long d = 0; d < dim; d++)

        if (dist <= ssd_cutoff)
        {
            mask |= (1U << i);
        }
    } // for (int i = 0; i < 32; i++)

    return mask;
}

#if !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * sq16_fastscan_32x_3d_avx2() - AVX2 256-bit SIMD kernel for 32 3D candidates.
 */
static inline uint32_t sq16_fastscan_32x_3d_avx2(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_x,
    const int16_t *restrict block_y,
    const int16_t *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= SQ16_FASTSCAN_MAX_3D_SSD)
    {
        return 0xFFFFFFFFU;
    }

    uint32_t q_xy = (uint16_t)query_sq16[0] |
                    ((uint32_t)(uint16_t)query_sq16[1] << 16);
    __m256i v_qxy = _mm256_set1_epi32((int32_t)q_xy);
    __m256i qz = _mm256_set1_epi16(query_sq16[2]);
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)((uint32_t)ssd_cutoff ^ 0x80000000U));
    __m256i v_zero = _mm256_setzero_si256();

    // Sub-block 0: candidates 0..15
    __m256i cx0 = _mm256_loadu_si256((const __m256i *)block_x);
    __m256i cy0 = _mm256_loadu_si256((const __m256i *)block_y);
    __m256i cz0 = _mm256_loadu_si256((const __m256i *)block_z);

    __m256i p0_lo = _mm256_unpacklo_epi16(cx0, cy0);
    __m256i p0_hi = _mm256_unpackhi_epi16(cx0, cy0);
    __m256i d0_lo = _mm256_sub_epi16(v_qxy, p0_lo);
    __m256i d0_hi = _mm256_sub_epi16(v_qxy, p0_hi);
    __m256i sum0_lo = _mm256_madd_epi16(d0_lo, d0_lo);
    __m256i sum0_hi = _mm256_madd_epi16(d0_hi, d0_hi);

    __m256i dz0 = _mm256_sub_epi16(qz, cz0);
    __m256i dz0_lo = _mm256_unpacklo_epi16(dz0, v_zero);
    __m256i dz0_hi = _mm256_unpackhi_epi16(dz0, v_zero);
    sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_madd_epi16(dz0_lo, dz0_lo));
    sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_madd_epi16(dz0_hi, dz0_hi));

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t fail0 = (uint32_t)(m0_lo & 0x0F) |
                     ((uint32_t)(m0_hi & 0x0F) << 4) |
                     ((uint32_t)(m0_lo & 0xF0) << 4) |
                     ((uint32_t)(m0_hi & 0xF0) << 8);
    uint32_t pass0 = (~fail0) & 0xFFFFU;

    // Sub-block 1: candidates 16..31
    __m256i cx1 = _mm256_loadu_si256((const __m256i *)(block_x + 16));
    __m256i cy1 = _mm256_loadu_si256((const __m256i *)(block_y + 16));
    __m256i cz1 = _mm256_loadu_si256((const __m256i *)(block_z + 16));

    __m256i p1_lo = _mm256_unpacklo_epi16(cx1, cy1);
    __m256i p1_hi = _mm256_unpackhi_epi16(cx1, cy1);
    __m256i d1_lo = _mm256_sub_epi16(v_qxy, p1_lo);
    __m256i d1_hi = _mm256_sub_epi16(v_qxy, p1_hi);
    __m256i sum1_lo = _mm256_madd_epi16(d1_lo, d1_lo);
    __m256i sum1_hi = _mm256_madd_epi16(d1_hi, d1_hi);

    __m256i dz1 = _mm256_sub_epi16(qz, cz1);
    __m256i dz1_lo = _mm256_unpacklo_epi16(dz1, v_zero);
    __m256i dz1_hi = _mm256_unpackhi_epi16(dz1, v_zero);
    sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_madd_epi16(dz1_lo, dz1_lo));
    sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_madd_epi16(dz1_hi, dz1_hi));

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t fail1 = (uint32_t)(m1_lo & 0x0F) |
                     ((uint32_t)(m1_hi & 0x0F) << 4) |
                     ((uint32_t)(m1_lo & 0xF0) << 4) |
                     ((uint32_t)(m1_hi & 0xF0) << 8);
    uint32_t pass1 = (~fail1) & 0xFFFFU;

    return (pass1 << 16) | pass0;
}

/**
 * sq16_fastscan_32x_generic_avx2() - AVX2 256-bit SIMD kernel for 32 D-dim candidates.
 */
static inline uint32_t sq16_fastscan_32x_generic_avx2(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return sq16_fastscan_32x_generic_scalar(query_sq16, block_coords, dim, ssd_cutoff);
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m256i v_clamp = _mm256_set1_epi32((int32_t)(cut32 + 1));
    __m256i v_all_ones = _mm256_set1_epi32(-1);

    __m256i sum0_lo = _mm256_setzero_si256();
    __m256i sum0_hi = _mm256_setzero_si256();
    __m256i sum1_lo = _mm256_setzero_si256();
    __m256i sum1_hi = _mm256_setzero_si256();

    /*
     * Dimension index d is declared before loop to track leftover dimension
     * for odd dimensionality vectors.
     */
    long d = 0;
    for (; d < dim - 1; d += 2)
    {
        uint32_t q_pair = (uint16_t)query_sq16[d] |
                          ((uint32_t)(uint16_t)query_sq16[d + 1] << 16);
        __m256i v_qpair = _mm256_set1_epi32((int32_t)q_pair);

        const int16_t *cd0_ptr = block_coords + d * 32;
        const int16_t *cd1_ptr = block_coords + (d + 1) * 32;

        __m256i c0_d = _mm256_loadu_si256((const __m256i *)cd0_ptr);
        __m256i c0_d1 = _mm256_loadu_si256((const __m256i *)cd1_ptr);
        __m256i p0_lo = _mm256_unpacklo_epi16(c0_d, c0_d1);
        __m256i p0_hi = _mm256_unpackhi_epi16(c0_d, c0_d1);
        __m256i diff0_lo = _mm256_sub_epi16(v_qpair, p0_lo);
        __m256i diff0_hi = _mm256_sub_epi16(v_qpair, p0_hi);
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_madd_epi16(diff0_lo, diff0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_madd_epi16(diff0_hi, diff0_hi));

        __m256i c1_d = _mm256_loadu_si256((const __m256i *)(cd0_ptr + 16));
        __m256i c1_d1 = _mm256_loadu_si256((const __m256i *)(cd1_ptr + 16));
        __m256i p1_lo = _mm256_unpacklo_epi16(c1_d, c1_d1);
        __m256i p1_hi = _mm256_unpackhi_epi16(c1_d, c1_d1);
        __m256i diff1_lo = _mm256_sub_epi16(v_qpair, p1_lo);
        __m256i diff1_hi = _mm256_sub_epi16(v_qpair, p1_hi);
        sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_madd_epi16(diff1_lo, diff1_lo));
        sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_madd_epi16(diff1_hi, diff1_hi));

        if (d + 2 < dim)
        {
            sum0_lo = _mm256_min_epu32(sum0_lo, v_clamp);
            sum0_hi = _mm256_min_epu32(sum0_hi, v_clamp);
            sum1_lo = _mm256_min_epu32(sum1_lo, v_clamp);
            sum1_hi = _mm256_min_epu32(sum1_hi, v_clamp);

            if (((d + 2) & 15) == 0)
            {
                __m256i f0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
                __m256i f0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
                __m256i f1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
                __m256i f1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);

                __m256i all_fail = _mm256_and_si256(
                    _mm256_and_si256(f0_lo, f0_hi),
                    _mm256_and_si256(f1_lo, f1_hi)
                );
                if (_mm256_testc_si256(all_fail, v_all_ones))
                {
                    return 0;
                }
            } // if (((d + 2) & 15) == 0)
        } // if (d + 2 < dim)
    } // for (; d < dim - 1; d += 2)

    if (d < dim)
    {
        __m256i v_zero = _mm256_setzero_si256();
        __m256i qd = _mm256_set1_epi16(query_sq16[d]);
        const int16_t *cd_ptr = block_coords + d * 32;

        __m256i c0 = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m256i diff0 = _mm256_sub_epi16(qd, c0);
        __m256i diff0_lo = _mm256_unpacklo_epi16(diff0, v_zero);
        __m256i diff0_hi = _mm256_unpackhi_epi16(diff0, v_zero);
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_madd_epi16(diff0_lo, diff0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_madd_epi16(diff0_hi, diff0_hi));

        __m256i c1 = _mm256_loadu_si256((const __m256i *)(cd_ptr + 16));
        __m256i diff1 = _mm256_sub_epi16(qd, c1);
        __m256i diff1_lo = _mm256_unpacklo_epi16(diff1, v_zero);
        __m256i diff1_hi = _mm256_unpackhi_epi16(diff1, v_zero);
        sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_madd_epi16(diff1_lo, diff1_lo));
        sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_madd_epi16(diff1_hi, diff1_hi));
    } // if (d < dim)

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t fail0 = (uint32_t)(m0_lo & 0x0F) |
                     ((uint32_t)(m0_hi & 0x0F) << 4) |
                     ((uint32_t)(m0_lo & 0xF0) << 4) |
                     ((uint32_t)(m0_hi & 0xF0) << 8);
    uint32_t pass0 = (~fail0) & 0xFFFFU;

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t fail1 = (uint32_t)(m1_lo & 0x0F) |
                     ((uint32_t)(m1_hi & 0x0F) << 4) |
                     ((uint32_t)(m1_lo & 0xF0) << 4) |
                     ((uint32_t)(m1_hi & 0xF0) << 8);
    uint32_t pass1 = (~fail1) & 0xFFFFU;

    return (pass1 << 16) | pass0;
}
#endif // __AVX2__

#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * sq16_fastscan_32x_3d_avx512() - AVX-512 512-bit SIMD kernel for 32 3D candidates.
 */
static inline uint32_t sq16_fastscan_32x_3d_avx512(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_x,
    const int16_t *restrict block_y,
    const int16_t *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= SQ16_FASTSCAN_MAX_3D_SSD)
    {
        return 0xFFFFFFFFU;
    }

    __m512i qx = _mm512_set1_epi16(query_sq16[0]);
    __m512i qy = _mm512_set1_epi16(query_sq16[1]);
    __m512i qz = _mm512_set1_epi16(query_sq16[2]);
    __m512i v_cut_u = _mm512_set1_epi32((int32_t)(uint32_t)ssd_cutoff);

    __m512i cx = _mm512_loadu_si512((const void *)block_x);
    __m512i cy = _mm512_loadu_si512((const void *)block_y);
    __m512i cz = _mm512_loadu_si512((const void *)block_z);

    __m512i dx = _mm512_sub_epi16(qx, cx);
    __m512i dy = _mm512_sub_epi16(qy, cy);
    __m512i dz = _mm512_sub_epi16(qz, cz);

    __m512i dx_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(dx));
    __m512i dx_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(dx, 1));
    __m512i sum_lo = _mm512_mullo_epi32(dx_lo, dx_lo);
    __m512i sum_hi = _mm512_mullo_epi32(dx_hi, dx_hi);

    __m512i dy_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(dy));
    __m512i dy_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(dy, 1));
    sum_lo = _mm512_add_epi32(sum_lo, _mm512_mullo_epi32(dy_lo, dy_lo));
    sum_hi = _mm512_add_epi32(sum_hi, _mm512_mullo_epi32(dy_hi, dy_hi));

    __m512i dz_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(dz));
    __m512i dz_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(dz, 1));
    sum_lo = _mm512_add_epi32(sum_lo, _mm512_mullo_epi32(dz_lo, dz_lo));
    sum_hi = _mm512_add_epi32(sum_hi, _mm512_mullo_epi32(dz_hi, dz_hi));

    __mmask16 pass_lo = _mm512_cmple_epu32_mask(sum_lo, v_cut_u);
    __mmask16 pass_hi = _mm512_cmple_epu32_mask(sum_hi, v_cut_u);

    return ((uint32_t)pass_hi << 16) | (uint32_t)pass_lo;
}

/**
 * sq16_fastscan_32x_generic_avx512() - AVX-512 512-bit SIMD kernel for 32 D-dim candidates.
 */
static inline uint32_t sq16_fastscan_32x_generic_avx512(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return sq16_fastscan_32x_generic_scalar(query_sq16, block_coords, dim, ssd_cutoff);
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m512i v_cut_u = _mm512_set1_epi32((int32_t)cut32);
    __m512i v_clamp = _mm512_set1_epi32((int32_t)(cut32 + 1));

    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();

    for (long d = 0; d < dim; d++)
    {
        __m512i qd = _mm512_set1_epi16(query_sq16[d]);
        __m512i cd = _mm512_loadu_si512((const void *)(block_coords + d * 32));
        __m512i diff = _mm512_sub_epi16(qd, cd);

        __m512i d_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(diff));
        __m512i d_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff, 1));

        sum_lo = _mm512_add_epi32(sum_lo, _mm512_mullo_epi32(d_lo, d_lo));
        sum_hi = _mm512_add_epi32(sum_hi, _mm512_mullo_epi32(d_hi, d_hi));

        if ((d & 1) == 1 && d + 1 < dim)
        {
            sum_lo = _mm512_min_epu32(sum_lo, v_clamp);
            sum_hi = _mm512_min_epu32(sum_hi, v_clamp);

            if (((d + 1) & 15) == 0)
            {
                __mmask16 p_lo = _mm512_cmple_epu32_mask(sum_lo, v_cut_u);
                __mmask16 p_hi = _mm512_cmple_epu32_mask(sum_hi, v_cut_u);
                if ((p_lo | p_hi) == 0)
                {
                    return 0;
                }
            } // if (((d + 1) & 15) == 0)
        }
    } // for (long d = 0; d < dim; d++)

    __mmask16 pass_lo = _mm512_cmple_epu32_mask(sum_lo, v_cut_u);
    __mmask16 pass_hi = _mm512_cmple_epu32_mask(sum_hi, v_cut_u);

    return ((uint32_t)pass_hi << 16) | (uint32_t)pass_lo;
}
#endif // __AVX512F__

/**
 * sq16_fastscan_32x_3d() - Evaluate SQ16 squared distance for 32 3D candidates in SIMD.
 * @query_sq16: Pointer to query's 3 quantized coordinates.
 * @block_x:    Pointer to 32 transposed X coordinates.
 * @block_y:    Pointer to 32 transposed Y coordinates.
 * @block_z:    Pointer to 32 transposed Z coordinates.
 * @ssd_cutoff: Squared distance threshold for candidate survival.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i satisfies dist^2 <= ssd_cutoff.
 */
uint32_t sq16_fastscan_32x_3d(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_x,
    const int16_t *restrict block_y,
    const int16_t *restrict block_z,
    uint64_t                ssd_cutoff)
{
#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return sq16_fastscan_32x_3d_avx512(query_sq16, block_x, block_y, block_z, ssd_cutoff);
#elif !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return sq16_fastscan_32x_3d_avx2(query_sq16, block_x, block_y, block_z, ssd_cutoff);
#else
    return sq16_fastscan_32x_3d_scalar(query_sq16, block_x, block_y, block_z, ssd_cutoff);
#endif
}

/**
 * sq16_fastscan_32x() - Evaluate SQ16 squared distance for 32 D-dim candidates in SIMD.
 * @query_sq16:   Pointer to query's dim quantized coordinates.
 * @block_coords: Pointer to 32*dim transposed coordinates.
 * @dim:          Vector dimensionality.
 * @ssd_cutoff:   Squared distance threshold for candidate survival.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i satisfies dist^2 <= ssd_cutoff.
 */
uint32_t sq16_fastscan_32x(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (dim == 3)
    {
        return sq16_fastscan_32x_3d(
            query_sq16,
            block_coords,
            block_coords + 32,
            block_coords + 64,
            ssd_cutoff
        );
    }

#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return sq16_fastscan_32x_generic_avx512(query_sq16, block_coords, dim, ssd_cutoff);
#elif !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return sq16_fastscan_32x_generic_avx2(query_sq16, block_coords, dim, ssd_cutoff);
#else
    return sq16_fastscan_32x_generic_scalar(query_sq16, block_coords, dim, ssd_cutoff);
#endif
}


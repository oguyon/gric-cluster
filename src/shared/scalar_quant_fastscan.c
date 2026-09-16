/**
 * @file scalar_quant_fastscan.c
 * @brief SIMD fastscan and cutoff squared distance kernels for SQ16.
 */

#include "scalar_quant.h"
#include "gric_simd.h"
#include <stdint.h>
#include <stddef.h>

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

    __m256i qx = _mm256_set1_epi16(query_sq16[0]);
    __m256i qy = _mm256_set1_epi16(query_sq16[1]);
    __m256i qz = _mm256_set1_epi16(query_sq16[2]);
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)((uint32_t)ssd_cutoff ^ 0x80000000U));

    // Sub-block 0: candidates 0..15
    __m256i cx0 = _mm256_loadu_si256((const __m256i *)block_x);
    __m256i cy0 = _mm256_loadu_si256((const __m256i *)block_y);
    __m256i cz0 = _mm256_loadu_si256((const __m256i *)block_z);

    __m256i dx0 = _mm256_sub_epi16(qx, cx0);
    __m256i dy0 = _mm256_sub_epi16(qy, cy0);
    __m256i dz0 = _mm256_sub_epi16(qz, cz0);

    __m256i dx0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dx0));
    __m256i dx0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dx0, 1));
    __m256i sum0_lo = _mm256_mullo_epi32(dx0_lo, dx0_lo);
    __m256i sum0_hi = _mm256_mullo_epi32(dx0_hi, dx0_hi);

    __m256i dy0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dy0));
    __m256i dy0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dy0, 1));
    sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(dy0_lo, dy0_lo));
    sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(dy0_hi, dy0_hi));

    __m256i dz0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dz0));
    __m256i dz0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dz0, 1));
    sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(dz0_lo, dz0_lo));
    sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(dz0_hi, dz0_hi));

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t pass0 = (~(uint32_t)((m0_hi << 8) | m0_lo)) & 0xFFFFU;

    // Sub-block 1: candidates 16..31
    __m256i cx1 = _mm256_loadu_si256((const __m256i *)(block_x + 16));
    __m256i cy1 = _mm256_loadu_si256((const __m256i *)(block_y + 16));
    __m256i cz1 = _mm256_loadu_si256((const __m256i *)(block_z + 16));

    __m256i dx1 = _mm256_sub_epi16(qx, cx1);
    __m256i dy1 = _mm256_sub_epi16(qy, cy1);
    __m256i dz1 = _mm256_sub_epi16(qz, cz1);

    __m256i dx1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dx1));
    __m256i dx1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dx1, 1));
    __m256i sum1_lo = _mm256_mullo_epi32(dx1_lo, dx1_lo);
    __m256i sum1_hi = _mm256_mullo_epi32(dx1_hi, dx1_hi);

    __m256i dy1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dy1));
    __m256i dy1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dy1, 1));
    sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_mullo_epi32(dy1_lo, dy1_lo));
    sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_mullo_epi32(dy1_hi, dy1_hi));

    __m256i dz1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dz1));
    __m256i dz1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dz1, 1));
    sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_mullo_epi32(dz1_lo, dz1_lo));
    sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_mullo_epi32(dz1_hi, dz1_hi));

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t pass1 = (~(uint32_t)((m1_hi << 8) | m1_lo)) & 0xFFFFU;

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

    __m256i sum0_lo = _mm256_setzero_si256();
    __m256i sum0_hi = _mm256_setzero_si256();
    __m256i sum1_lo = _mm256_setzero_si256();
    __m256i sum1_hi = _mm256_setzero_si256();

    for (long d = 0; d < dim; d++)
    {
        __m256i qd = _mm256_set1_epi16(query_sq16[d]);
        const int16_t *cd_ptr = block_coords + d * 32;

        __m256i c0 = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m256i diff0 = _mm256_sub_epi16(qd, c0);
        __m256i d0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff0));
        __m256i d0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff0, 1));
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(d0_lo, d0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(d0_hi, d0_hi));

        __m256i c1 = _mm256_loadu_si256((const __m256i *)(cd_ptr + 16));
        __m256i diff1 = _mm256_sub_epi16(qd, c1);
        __m256i d1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff1));
        __m256i d1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff1, 1));
        sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_mullo_epi32(d1_lo, d1_lo));
        sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_mullo_epi32(d1_hi, d1_hi));

        if ((d & 1) == 1 && d + 1 < dim)
        {
            sum0_lo = _mm256_min_epu32(sum0_lo, v_clamp);
            sum0_hi = _mm256_min_epu32(sum0_hi, v_clamp);
            sum1_lo = _mm256_min_epu32(sum1_lo, v_clamp);
            sum1_hi = _mm256_min_epu32(sum1_hi, v_clamp);
        }
    } // for (long d = 0; d < dim; d++)

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t pass0 = (~(uint32_t)((m0_hi << 8) | m0_lo)) & 0xFFFFU;

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t pass1 = (~(uint32_t)((m1_hi << 8) | m1_lo)) & 0xFFFFU;

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

#if !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * sq16_reduce_add_epi32_avx2() - Horizontal add of 8 32-bit vector elements using AVX2.
 * @v: 256-bit SIMD vector holding 8 int32 lanes.
 *
 * Return: Sum of all 8 lanes as unsigned 32-bit integer.
 */
static inline uint32_t sq16_reduce_add_epi32_avx2(
    __m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(s);
}
#endif // __AVX2__

#if GRIC_HAVE_AVX512_TARGET
/**
 * sq16_filter_anchor_matrix_avx512() - AVX-512 kernel for anchor filtering.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
GRIC_TARGET_AVX512
static void sq16_filter_anchor_matrix_avx512(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    uint32_t thresh32 = (sq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)sq16_ssd_thresh;
    int num_active = 0;
    int pruned_count = 0;

    if (dim == 128)
    {
        __m512i q0 = _mm512_loadu_si512((const void *)(cur_sq16 + 0));
        __m512i q1 = _mm512_loadu_si512((const void *)(cur_sq16 + 32));
        __m512i q2 = _mm512_loadu_si512((const void *)(cur_sq16 + 64));
        __m512i q3 = _mm512_loadu_si512((const void *)(cur_sq16 + 96));

        int i = 0;
        for (; i + 3 < num_clusters; i += 4)
        {
            if (i + 12 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 8) * 128),
                             _MM_HINT_T0);
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 10) * 128),
                             _MM_HINT_T0);
            }

            const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * 128;
            const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * 128;
            const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * 128;
            const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * 128;

            __m512i a0 = _mm512_loadu_si512((const void *)(a0_ptr + 0));
            __m512i a1 = _mm512_loadu_si512((const void *)(a1_ptr + 0));
            __m512i a2 = _mm512_loadu_si512((const void *)(a2_ptr + 0));
            __m512i a3 = _mm512_loadu_si512((const void *)(a3_ptr + 0));

            __m512i d0 = _mm512_sub_epi16(q0, a0);
            __m512i d1 = _mm512_sub_epi16(q0, a1);
            __m512i d2 = _mm512_sub_epi16(q0, a2);
            __m512i d3 = _mm512_sub_epi16(q0, a3);

            __m512i acc0 = _mm512_madd_epi16(d0, d0);
            __m512i acc1 = _mm512_madd_epi16(d1, d1);
            __m512i acc2 = _mm512_madd_epi16(d2, d2);
            __m512i acc3 = _mm512_madd_epi16(d3, d3);

            uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(acc0);
            uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(acc1);
            uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(acc2);
            uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(acc3);

            if (s0 > thresh32 && s1 > thresh32 && s2 > thresh32 && s3 > thresh32)
            {
                clmembflag[i + 0] = 0;
                clmembflag[i + 1] = 0;
                clmembflag[i + 2] = 0;
                clmembflag[i + 3] = 0;
                pruned_count += 4;
                continue;
            }

            uint32_t sums[4] = {s0, s1, s2, s3};
            for (int k = 0; k < 4; k++)
            {
                if (sums[k] > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * 128;
                uint32_t total = sums[k];

                // Block 1: dims 32..63
                __m512i b1 = _mm512_loadu_si512((const void *)(ak_ptr + 32));
                __m512i db1 = _mm512_sub_epi16(q1, b1);
                total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db1, db1));
                if (total > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                // Block 2: dims 64..95
                __m512i b2 = _mm512_loadu_si512((const void *)(ak_ptr + 64));
                __m512i db2 = _mm512_sub_epi16(q2, b2);
                total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db2, db2));
                if (total > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                // Block 3: dims 96..127
                __m512i b3 = _mm512_loadu_si512((const void *)(ak_ptr + 96));
                __m512i db3 = _mm512_sub_epi16(q3, b3);
                total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db3, db3));
                if (total > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                clmembflag[i + k] = 1;
                active_clusters[num_active++] = i + k;
            }
        }

        // Remainder loop
        for (; i < num_clusters; i++)
        {
            const int16_t *a_ptr = anchor_matrix + (size_t)i * 128;

            __m512i a0 = _mm512_loadu_si512((const void *)(a_ptr + 0));
            __m512i d0 = _mm512_sub_epi16(q0, a0);
            __m512i acc = _mm512_madd_epi16(d0, d0);
            if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            __m512i a1 = _mm512_loadu_si512((const void *)(a_ptr + 32));
            __m512i d1 = _mm512_sub_epi16(q1, a1);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d1, d1));
            if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            __m512i a2 = _mm512_loadu_si512((const void *)(a_ptr + 64));
            __m512i d2 = _mm512_sub_epi16(q2, a2);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d2, d2));
            if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            __m512i a3 = _mm512_loadu_si512((const void *)(a_ptr + 96));
            __m512i d3 = _mm512_sub_epi16(q3, a3);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d3, d3));
            if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            clmembflag[i] = 1;
            active_clusters[num_active++] = i;
        }
    }
    else if (dim <= 64)
    {
        if (dim <= 32)
        {
            __m512i q0 = _mm512_loadu_si512((const void *)cur_sq16);
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;

                __m512i d0 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a0_ptr));
                __m512i d1 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a1_ptr));
                __m512i d2 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a2_ptr));
                __m512i d3 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a3_ptr));

                uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d0, d0));
                uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d1, d1));
                uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d2, d2));
                uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d3, d3));

                uint32_t sums[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4; k++)
                {
                    if (sums[k] > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                    }
                    else
                    {
                        clmembflag[i + k] = 1;
                        active_clusters[num_active++] = i + k;
                    }
                }
            }
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m512i d = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a_ptr));
                uint32_t s = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d, d));
                if (s > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                }
                else
                {
                    clmembflag[i] = 1;
                    active_clusters[num_active++] = i;
                }
            }
        }
        else // 32 < dim <= 64
        {
            __m512i q0 = _mm512_loadu_si512((const void *)(cur_sq16 + 0));
            __m512i q1 = _mm512_loadu_si512((const void *)(cur_sq16 + 32));
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;

                __m512i d0 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a0_ptr));
                __m512i d1 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a1_ptr));
                __m512i d2 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a2_ptr));
                __m512i d3 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)a3_ptr));

                uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d0, d0));
                uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d1, d1));
                uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d2, d2));
                uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d3, d3));

                if (s0 > thresh32 && s1 > thresh32 && s2 > thresh32 && s3 > thresh32)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t sums[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4; k++)
                {
                    if (sums[k] > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }
                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * dim;
                    __m512i db1 = _mm512_sub_epi16(q1,
                                                   _mm512_loadu_si512((const void *)(ak_ptr + 32)));
                    uint32_t tot = sums[k] +
                        (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db1, db1));
                    if (tot > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }
                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            }
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m512i d0 = _mm512_sub_epi16(q0, _mm512_loadu_si512((const void *)(a_ptr + 0)));
                __m512i acc = _mm512_madd_epi16(d0, d0);
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                __m512i d1 = _mm512_sub_epi16(q1, _mm512_loadu_si512((const void *)(a_ptr + 32)));
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d1, d1));
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        }
    }
    else
    {
        // Arbitrary dimension (including D > 128): 32-element dimension-block streaming
        for (int i = 0; i < num_clusters; i++)
        {
            if (i + 16 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 16) * dim),
                             _MM_HINT_T0);
            }
            const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
            uint32_t total = 0;
            int pruned = 0;

            for (long d = 0; d < dim; d += 32)
            {
                if (d + 32 <= dim)
                {
                    __m512i q = _mm512_loadu_si512((const void *)(cur_sq16 + d));
                    __m512i a = _mm512_loadu_si512((const void *)(a_ptr + d));
                    __m512i diff = _mm512_sub_epi16(q, a);
                    total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(diff, diff));
                }
                else
                {
                    for (long rem = d; rem < dim; rem++)
                    {
                        int32_t df = (int32_t)cur_sq16[rem] - (int32_t)a_ptr[rem];
                        total += (uint32_t)(df * df);
                    }
                }
                if (total > thresh32)
                {
                    pruned = 1;
                    break;
                }
            } // for (long d = 0; d < dim; d += 32)

            if (pruned)
            {
                clmembflag[i] = 0;
                pruned_count++;
            }
            else
            {
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        }
    }

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * sq16_filter_anchor_matrix_avx2() - AVX2 kernel for anchor filtering.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
GRIC_TARGET_AVX2
static void sq16_filter_anchor_matrix_avx2(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    uint32_t thresh32 = (sq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)sq16_ssd_thresh;
    int num_active = 0;
    int pruned_count = 0;
    __m128i v_bias = _mm_set1_epi32((int32_t)0x80000000U);
    __m128i v_cut  = _mm_set1_epi32((int32_t)(thresh32 ^ 0x80000000U));

    if (dim == 128)
    {
        __m256i q0 = _mm256_load_si256((const __m256i *)(cur_sq16 + 0));
        __m256i q1 = _mm256_load_si256((const __m256i *)(cur_sq16 + 16));
        __m256i q2 = _mm256_load_si256((const __m256i *)(cur_sq16 + 32));
        __m256i q3 = _mm256_load_si256((const __m256i *)(cur_sq16 + 48));
        __m256i q4 = _mm256_load_si256((const __m256i *)(cur_sq16 + 64));
        __m256i q5 = _mm256_load_si256((const __m256i *)(cur_sq16 + 80));
        __m256i q6 = _mm256_load_si256((const __m256i *)(cur_sq16 + 96));
        __m256i q7 = _mm256_load_si256((const __m256i *)(cur_sq16 + 112));
        int i = 0;

        for (; i + 3 < num_clusters; i += 4)
        {
            if (i + 12 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 8) * 128),
                             _MM_HINT_T0);
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 10) * 128),
                             _MM_HINT_T0);
            }

            const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * 128;
            const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * 128;
            const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * 128;
            const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * 128;

            // Chunk 0: Anchor 0
            __m256i a0_0 = _mm256_load_si256((const __m256i *)(a0_ptr + 0));
            __m256i a0_1 = _mm256_load_si256((const __m256i *)(a0_ptr + 16));
            __m256i d0_0 = _mm256_sub_epi16(q0, a0_0);
            __m256i d0_1 = _mm256_sub_epi16(q1, a0_1);
            __m256i acc0 = _mm256_add_epi32(_mm256_madd_epi16(d0_0, d0_0),
                                            _mm256_madd_epi16(d0_1, d0_1));

            // Chunk 0: Anchor 1
            __m256i a1_0 = _mm256_load_si256((const __m256i *)(a1_ptr + 0));
            __m256i a1_1 = _mm256_load_si256((const __m256i *)(a1_ptr + 16));
            __m256i d1_0 = _mm256_sub_epi16(q0, a1_0);
            __m256i d1_1 = _mm256_sub_epi16(q1, a1_1);
            __m256i acc1 = _mm256_add_epi32(_mm256_madd_epi16(d1_0, d1_0),
                                            _mm256_madd_epi16(d1_1, d1_1));

            __m256i h01 = _mm256_hadd_epi32(acc0, acc1);

            // Chunk 0: Anchor 2
            __m256i a2_0 = _mm256_load_si256((const __m256i *)(a2_ptr + 0));
            __m256i a2_1 = _mm256_load_si256((const __m256i *)(a2_ptr + 16));
            __m256i d2_0 = _mm256_sub_epi16(q0, a2_0);
            __m256i d2_1 = _mm256_sub_epi16(q1, a2_1);
            __m256i acc2 = _mm256_add_epi32(_mm256_madd_epi16(d2_0, d2_0),
                                            _mm256_madd_epi16(d2_1, d2_1));

            // Chunk 0: Anchor 3
            __m256i a3_0 = _mm256_load_si256((const __m256i *)(a3_ptr + 0));
            __m256i a3_1 = _mm256_load_si256((const __m256i *)(a3_ptr + 16));
            __m256i d3_0 = _mm256_sub_epi16(q0, a3_0);
            __m256i d3_1 = _mm256_sub_epi16(q1, a3_1);
            __m256i acc3 = _mm256_add_epi32(_mm256_madd_epi16(d3_0, d3_0),
                                            _mm256_madd_epi16(d3_1, d3_1));

            __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
            __m256i h_all = _mm256_hadd_epi32(h01, h23);
            __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                         _mm256_extracti128_si256(h_all, 1));

            __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
            __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
            int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

            if (mask == 0xF)
            {
                clmembflag[i + 0] = 0;
                clmembflag[i + 1] = 0;
                clmembflag[i + 2] = 0;
                clmembflag[i + 3] = 0;
                pruned_count += 4;
                continue;
            }

            uint32_t s4[4];
            _mm_storeu_si128((__m128i *)s4, sum4);

            for (int k = 0; k < 4; k++)
            {
                if ((mask & (1 << k)) != 0)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * 128;
                uint32_t total = s4[k];

                // Chunk 1: dims 32..63
                __m256i a2 = _mm256_load_si256((const __m256i *)(ak_ptr + 32));
                __m256i a3 = _mm256_load_si256((const __m256i *)(ak_ptr + 48));
                __m256i d2 = _mm256_sub_epi16(q2, a2);
                __m256i d3 = _mm256_sub_epi16(q3, a3);
                __m256i acc_c1 = _mm256_add_epi32(_mm256_madd_epi16(d2, d2),
                                                  _mm256_madd_epi16(d3, d3));
                total += sq16_reduce_add_epi32_avx2(acc_c1);
                if (total > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                // Chunk 2 & 3: dims 64..127
                __m256i a4 = _mm256_load_si256((const __m256i *)(ak_ptr + 64));
                __m256i a5 = _mm256_load_si256((const __m256i *)(ak_ptr + 80));
                __m256i d4 = _mm256_sub_epi16(q4, a4);
                __m256i d5 = _mm256_sub_epi16(q5, a5);
                __m256i acc_c23 = _mm256_add_epi32(_mm256_madd_epi16(d4, d4),
                                                   _mm256_madd_epi16(d5, d5));

                __m256i a6 = _mm256_load_si256((const __m256i *)(ak_ptr + 96));
                __m256i a7 = _mm256_load_si256((const __m256i *)(ak_ptr + 112));
                __m256i d6 = _mm256_sub_epi16(q6, a6);
                __m256i d7 = _mm256_sub_epi16(q7, a7);
                acc_c23 = _mm256_add_epi32(acc_c23, _mm256_madd_epi16(d6, d6));
                acc_c23 = _mm256_add_epi32(acc_c23, _mm256_madd_epi16(d7, d7));

                total += sq16_reduce_add_epi32_avx2(acc_c23);
                if (total > thresh32)
                {
                    clmembflag[i + k] = 0;
                    pruned_count++;
                    continue;
                }

                clmembflag[i + k] = 1;
                active_clusters[num_active++] = i + k;
            }
        }

        // Remainder loop
        for (; i < num_clusters; i++)
        {
            const int16_t *a_ptr = anchor_matrix + (size_t)i * 128;

            // Chunk 0: dims 0..31
            __m256i a0 = _mm256_load_si256((const __m256i *)(a_ptr + 0));
            __m256i a1 = _mm256_load_si256((const __m256i *)(a_ptr + 16));
            __m256i d0 = _mm256_sub_epi16(q0, a0);
            __m256i d1 = _mm256_sub_epi16(q1, a1);
            __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                           _mm256_madd_epi16(d1, d1));
            if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            // Chunk 1: dims 32..63
            __m256i a2 = _mm256_load_si256((const __m256i *)(a_ptr + 32));
            __m256i a3 = _mm256_load_si256((const __m256i *)(a_ptr + 48));
            __m256i d2 = _mm256_sub_epi16(q2, a2);
            __m256i d3 = _mm256_sub_epi16(q3, a3);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d2, d2));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d3, d3));
            if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            // Chunk 2 & 3: dims 64..127
            __m256i a4 = _mm256_load_si256((const __m256i *)(a_ptr + 64));
            __m256i a5 = _mm256_load_si256((const __m256i *)(a_ptr + 80));
            __m256i d4 = _mm256_sub_epi16(q4, a4);
            __m256i d5 = _mm256_sub_epi16(q5, a5);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d4, d4));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d5, d5));

            __m256i a6 = _mm256_load_si256((const __m256i *)(a_ptr + 96));
            __m256i a7 = _mm256_load_si256((const __m256i *)(a_ptr + 112));
            __m256i d6 = _mm256_sub_epi16(q6, a6);
            __m256i d7 = _mm256_sub_epi16(q7, a7);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d6, d6));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d7, d7));
            if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
            {
                clmembflag[i] = 0;
                pruned_count++;
                continue;
            }

            clmembflag[i] = 1;
            active_clusters[num_active++] = i;
        }
    }
    else if (dim <= 64)
    {
        if (dim <= 32)
        {
            __m256i q0 = (dim >= 16) ? _mm256_loadu_si256((const __m256i *)(cur_sq16 + 0))
                                     : _mm256_setzero_si256();
            __m256i q1 = (dim > 16)  ? _mm256_loadu_si256((const __m256i *)(cur_sq16 + 16))
                                     : _mm256_setzero_si256();
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;

                __m256i d0_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a0_ptr));
                __m256i acc0 = _mm256_madd_epi16(d0_0, d0_0);
                __m256i d1_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a1_ptr));
                __m256i acc1 = _mm256_madd_epi16(d1_0, d1_0);
                __m256i d2_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a2_ptr));
                __m256i acc2 = _mm256_madd_epi16(d2_0, d2_0);
                __m256i d3_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a3_ptr));
                __m256i acc3 = _mm256_madd_epi16(d3_0, d3_0);

                if (dim > 16)
                {
                    __m256i d0_1 = _mm256_sub_epi16(q1,
                        _mm256_loadu_si256((const __m256i *)(a0_ptr + 16)));
                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0_1, d0_1));
                    __m256i d1_1 = _mm256_sub_epi16(q1,
                        _mm256_loadu_si256((const __m256i *)(a1_ptr + 16)));
                    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1_1, d1_1));
                    __m256i d2_1 = _mm256_sub_epi16(q1,
                        _mm256_loadu_si256((const __m256i *)(a2_ptr + 16)));
                    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2_1, d2_1));
                    __m256i d3_1 = _mm256_sub_epi16(q1,
                        _mm256_loadu_si256((const __m256i *)(a3_ptr + 16)));
                    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3_1, d3_1));
                }

                __m256i h01 = _mm256_hadd_epi32(acc0, acc1);
                __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
                __m256i h_all = _mm256_hadd_epi32(h01, h23);
                __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                             _mm256_extracti128_si256(h_all, 1));

                __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
                __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

                for (int k = 0; k < 4; k++)
                {
                    if ((mask & (1 << k)) != 0)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                    }
                    else
                    {
                        clmembflag[i + k] = 1;
                        active_clusters[num_active++] = i + k;
                    }
                }
            }
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m256i d0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a_ptr));
                __m256i acc = _mm256_madd_epi16(d0, d0);
                if (dim > 16)
                {
                    __m256i da1 = _mm256_loadu_si256((const __m256i *)(a_ptr + 16));
                    __m256i d1 = _mm256_sub_epi16(q1, da1);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d1, d1));
                }
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                }
                else
                {
                    clmembflag[i] = 1;
                    active_clusters[num_active++] = i;
                }
            }
        }
        else // 32 < dim <= 64
        {
            __m256i q0 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 0));
            __m256i q1 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 16));
            __m256i q2 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 32));
            __m256i q3 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 48));

            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;

                __m256i a0_0 = _mm256_loadu_si256((const __m256i *)(a0_ptr + 0));
                __m256i a0_1 = _mm256_loadu_si256((const __m256i *)(a0_ptr + 16));
                __m256i d0_0 = _mm256_sub_epi16(q0, a0_0);
                __m256i d0_1 = _mm256_sub_epi16(q1, a0_1);
                __m256i acc0 = _mm256_add_epi32(_mm256_madd_epi16(d0_0, d0_0),
                                                _mm256_madd_epi16(d0_1, d0_1));

                __m256i a1_0 = _mm256_loadu_si256((const __m256i *)(a1_ptr + 0));
                __m256i a1_1 = _mm256_loadu_si256((const __m256i *)(a1_ptr + 16));
                __m256i d1_0 = _mm256_sub_epi16(q0, a1_0);
                __m256i d1_1 = _mm256_sub_epi16(q1, a1_1);
                __m256i acc1 = _mm256_add_epi32(_mm256_madd_epi16(d1_0, d1_0),
                                                _mm256_madd_epi16(d1_1, d1_1));

                __m256i h01 = _mm256_hadd_epi32(acc0, acc1);

                __m256i a2_0 = _mm256_loadu_si256((const __m256i *)(a2_ptr + 0));
                __m256i a2_1 = _mm256_loadu_si256((const __m256i *)(a2_ptr + 16));
                __m256i d2_0 = _mm256_sub_epi16(q0, a2_0);
                __m256i d2_1 = _mm256_sub_epi16(q1, a2_1);
                __m256i acc2 = _mm256_add_epi32(_mm256_madd_epi16(d2_0, d2_0),
                                                _mm256_madd_epi16(d2_1, d2_1));

                __m256i a3_0 = _mm256_loadu_si256((const __m256i *)(a3_ptr + 0));
                __m256i a3_1 = _mm256_loadu_si256((const __m256i *)(a3_ptr + 16));
                __m256i d3_0 = _mm256_sub_epi16(q0, a3_0);
                __m256i d3_1 = _mm256_sub_epi16(q1, a3_1);
                __m256i acc3 = _mm256_add_epi32(_mm256_madd_epi16(d3_0, d3_0),
                                                _mm256_madd_epi16(d3_1, d3_1));

                __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
                __m256i h_all = _mm256_hadd_epi32(h01, h23);
                __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                             _mm256_extracti128_si256(h_all, 1));

                __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
                __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

                if (mask == 0xF)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t s4[4];
                _mm_storeu_si128((__m128i *)s4, sum4);

                for (int k = 0; k < 4; k++)
                {
                    if ((mask & (1 << k)) != 0)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * dim;
                    uint32_t total = s4[k];

                    __m256i a2 = _mm256_loadu_si256((const __m256i *)(ak_ptr + 32));
                    __m256i a3 = _mm256_loadu_si256((const __m256i *)(ak_ptr + 48));
                    __m256i d2 = _mm256_sub_epi16(q2, a2);
                    __m256i d3 = _mm256_sub_epi16(q3, a3);
                    __m256i acc_c1 = _mm256_add_epi32(_mm256_madd_epi16(d2, d2),
                                                      _mm256_madd_epi16(d3, d3));
                    total += sq16_reduce_add_epi32_avx2(acc_c1);
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            }
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m256i a0 = _mm256_loadu_si256((const __m256i *)(a_ptr + 0));
                __m256i a1 = _mm256_loadu_si256((const __m256i *)(a_ptr + 16));
                __m256i d0 = _mm256_sub_epi16(q0, a0);
                __m256i d1 = _mm256_sub_epi16(q1, a1);
                __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                               _mm256_madd_epi16(d1, d1));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                __m256i a2 = _mm256_loadu_si256((const __m256i *)(a_ptr + 32));
                __m256i a3 = _mm256_loadu_si256((const __m256i *)(a_ptr + 48));
                __m256i d2 = _mm256_sub_epi16(q2, a2);
                __m256i d3 = _mm256_sub_epi16(q3, a3);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d2, d2));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d3, d3));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        }
    }
    else
    {
        // Arbitrary dimension (including D > 128): 32-element dimension-block streaming
        for (int i = 0; i < num_clusters; i++)
        {
            if (i + 16 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 16) * dim),
                             _MM_HINT_T0);
            }
            const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
            uint32_t total = 0;
            int pruned = 0;

            for (long d = 0; d < dim; d += 32)
            {
                if (d + 32 <= dim)
                {
                    __m256i q0 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d));
                    __m256i q1 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d + 16));
                    __m256i a0 = _mm256_loadu_si256((const __m256i *)(a_ptr + d));
                    __m256i a1 = _mm256_loadu_si256((const __m256i *)(a_ptr + d + 16));
                    __m256i d0 = _mm256_sub_epi16(q0, a0);
                    __m256i d1 = _mm256_sub_epi16(q1, a1);
                    __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                                   _mm256_madd_epi16(d1, d1));
                    total += sq16_reduce_add_epi32_avx2(acc);
                }
                else
                {
                    for (long rem = d; rem < dim; rem++)
                    {
                        int32_t df = (int32_t)cur_sq16[rem] - (int32_t)a_ptr[rem];
                        total += (uint32_t)(df * df);
                    }
                }
                if (total > thresh32)
                {
                    pruned = 1;
                    break;
                }
            }

            if (pruned)
            {
                clmembflag[i] = 0;
                pruned_count++;
            }
            else
            {
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        }
    }

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}
#endif // AVX2

/**
 * sq16_filter_anchor_matrix_scalar() - Scalar fallback for anchor filtering.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
static void sq16_filter_anchor_matrix_scalar(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    int num_active = 0;
    int pruned_count = 0;

    for (int i = 0; i < num_clusters; i++)
    {
        const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
        uint64_t total = 0;
        int pruned = 0;

        for (long d = 0; d < dim; d++)
        {
            int32_t diff = (int32_t)cur_sq16[d] - (int32_t)a_ptr[d];
            total += (uint64_t)(diff * diff);
            if (total > sq16_ssd_thresh)
            {
                pruned = 1;
                break;
            }
        }

        if (pruned)
        {
            clmembflag[i] = 0;
            pruned_count++;
        }
        else
        {
            clmembflag[i] = 1;
            active_clusters[num_active++] = i;
        }
    }

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}

/**
 * sq16_filter_anchor_matrix() - Bulk filter cluster anchors using SQ16 lower bounds.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
void sq16_filter_anchor_matrix(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    if (num_clusters <= 0 || cur_sq16 == NULL || anchor_matrix == NULL)
    {
        if (out_num_active)
        {
            *out_num_active = 0;
        }
        if (out_pruned_count)
        {
            *out_pruned_count = 0;
        }
        return;
    }

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        sq16_filter_anchor_matrix_avx512(cur_sq16, anchor_matrix, num_clusters, dim,
                                         sq16_ssd_thresh, clmembflag, active_clusters,
                                         out_num_active, out_pruned_count);
        return;
    }
#endif

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        sq16_filter_anchor_matrix_avx2(cur_sq16, anchor_matrix, num_clusters, dim,
                                       sq16_ssd_thresh, clmembflag, active_clusters,
                                       out_num_active, out_pruned_count);
        return;
    }
#endif

    sq16_filter_anchor_matrix_scalar(cur_sq16, anchor_matrix, num_clusters, dim,
                                     sq16_ssd_thresh, clmembflag, active_clusters,
                                     out_num_active, out_pruned_count);
}



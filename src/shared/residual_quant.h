#ifndef RESIDUAL_QUANT_H
#define RESIDUAL_QUANT_H

/**
 * @file residual_quant.h
 * @brief 8-bit Residual Vector Quantization (RQ8) with metric lower-bounding.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/** Magic identifier for .rq8 sidecar files */
#define RQ8_FILE_MAGIC "RQ8_0001"

/** Universal transposed SIMD block size (32 candidates, 32 bytes per dimension) */
#define RQ8_FASTSCAN_BLOCK_SIZE 32

/**
 * @brief Parameters defining 8-bit residual quantization within a cluster ball.
 */
typedef struct
{
    float rlim;       /**< Maximum cluster covering radius */
    float scale;      /**< Step delta: rlim / 127.0f */
    float inv_scale;  /**< Reciprocal step: 127.0f / rlim */
    float err_radius; /**< Max single-vector Euclidean error: sqrt(dim) * scale * 0.5f */
    long  dim;        /**< Vector dimension (element count per frame) */
} RQ8Params;

/**
 * @brief Detected CPU SIMD vector register width for FastScan.
 */
typedef enum
{
    RQ8_SIMD_SCALAR = 0,
    RQ8_SIMD_AVX2   = 1,
    RQ8_SIMD_AVX512 = 2
} RQ8SimdMode;

/**
 * @brief Initialize RQ8 parameters from cluster radius and dimension.
 */
void rq8_init_params(
    RQ8Params *params,
    float      rlim,
    long       dim);

/**
 * @brief Quantize a single-precision float residual (src - anchor) to int8_t [-127, 127].
 */
void rq8_quantize_residual_float(
    const float     *restrict src,
    const float     *restrict anchor,
    int8_t          *restrict dst,
    const RQ8Params *restrict params);

/**
 * @brief Quantize a double-precision residual (src - anchor) to int8_t [-127, 127].
 */
void rq8_quantize_residual_double(
    const double    *restrict src,
    const double    *restrict anchor,
    int8_t          *restrict dst,
    const RQ8Params *restrict params);

/**
 * @brief Quantize a single-precision query residual (query - anchor) to int16_t.
 */
void rq8_quantize_query_residual_float(
    const float     *restrict query,
    const float     *restrict anchor,
    int16_t         *restrict dst,
    const RQ8Params *restrict params);

/**
 * @brief Quantize a double-precision query residual (query - anchor) to int16_t.
 */
void rq8_quantize_query_residual_double(
    const double    *restrict query,
    const double    *restrict anchor,
    int16_t         *restrict dst,
    const RQ8Params *restrict params);

/**
 * @brief Compute guaranteed metric cutoff threshold in squared integer units.
 */
static inline uint64_t rq8_compute_cutoff_thresh(
    double           cur_tau,
    const RQ8Params *params,
    double           eps)
{
    if (params == NULL || params->scale <= 0.0f)
    {
        return UINT64_MAX;
    }

    double eff_tau = cur_tau * (1.0 + eps);
    double raw_thresh = (eff_tau + 2.0 * (double)params->err_radius) * (double)params->inv_scale;
    if (raw_thresh >= 4294967295.0)
    {
        return UINT64_MAX;
    }
    return (raw_thresh > 0.0) ? (uint64_t)(raw_thresh * raw_thresh) : 0;
}

/**
 * @brief Compute guaranteed metric lower bound between query and candidate residual.
 */
static inline double rq8_compute_lower_bound(
    uint64_t         ssd,
    const RQ8Params *params,
    double           eps)
{
    if (params == NULL || params->scale <= 0.0f)
    {
        return 0.0;
    }

    double d_quant = sqrt((double)ssd) * (double)params->scale;
    double d_lb = d_quant - 2.0 * (double)params->err_radius;
    if (d_lb < 0.0)
    {
        d_lb = 0.0;
    }

    if (eps > 0.0)
    {
        d_lb /= (1.0 + eps);
    }

    return d_lb;
}

/**
 * @brief Compute sum of squared differences between int16 query residual and int8 candidate.
 */
static inline uint64_t rq8_dist_squared_cutoff_i8(
    const int16_t *restrict q_res,
    const int8_t  *restrict cand_res,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (dim == 3)
    {
        int32_t d0 = (int32_t)q_res[0] - (int32_t)cand_res[0];
        uint64_t total = (uint64_t)(d0 * d0);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
        int32_t d1 = (int32_t)q_res[1] - (int32_t)cand_res[1];
        total += (uint64_t)(d1 * d1);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
        int32_t d2 = (int32_t)q_res[2] - (int32_t)cand_res[2];
        total += (uint64_t)(d2 * d2);
        return total;
    } // if (dim == 3)

    if (dim < 16)
    {
        uint64_t total = 0;
        for (long k = 0; k < dim; k++)
        {
            int32_t diff = (int32_t)q_res[k] - (int32_t)cand_res[k];
            total += (uint64_t)(diff * diff);
            if (total > ssd_cutoff)
            {
                return ssd_cutoff + 1;
            }
        } // for (long k = 0; k < dim; k++)
        return total;
    } // if (dim < 16)

    uint64_t total = 0;
    long i = 0;

#if defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m512i sum_vec512 = _mm512_setzero_si512();

    for (; i <= dim - 64; i += 64)
    {
        __m512i vm = _mm512_loadu_si512((const void *)(cand_res + i));
        __m512i vm_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vm));
        __m512i vm_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vm, 1));

        __m512i vq_lo = _mm512_loadu_si512((const void *)(q_res + i));
        __m512i vq_hi = _mm512_loadu_si512((const void *)(q_res + i + 32));

        __m512i diff_lo = _mm512_sub_epi16(vq_lo, vm_lo);
        __m512i diff_hi = _mm512_sub_epi16(vq_hi, vm_hi);

        __m512i prod_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        __m512i prod_hi = _mm512_madd_epi16(diff_hi, diff_hi);

        sum_vec512 = _mm512_add_epi32(sum_vec512, prod_lo);
        sum_vec512 = _mm512_add_epi32(sum_vec512, prod_hi);
    } // for (; i <= dim - 64; i += 64)

    total += (uint64_t)_mm512_reduce_add_epi32(sum_vec512);
    if (total > ssd_cutoff)
    {
        return ssd_cutoff + 1;
    }
#elif defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256i sum_vec256 = _mm256_setzero_si256();

    for (; i <= dim - 32; i += 32)
    {
        __m256i vm = _mm256_loadu_si256((const __m256i *)(const void *)(cand_res + i));
        __m256i vm_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vm));
        __m256i vm_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vm, 1));

        __m256i vq_lo = _mm256_loadu_si256((const __m256i *)(const void *)(q_res + i));
        __m256i vq_hi = _mm256_loadu_si256((const __m256i *)(const void *)(q_res + i + 16));

        __m256i diff_lo = _mm256_sub_epi16(vq_lo, vm_lo);
        __m256i diff_hi = _mm256_sub_epi16(vq_hi, vm_hi);

        __m256i prod_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        __m256i prod_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum_vec256 = _mm256_add_epi32(sum_vec256, prod_lo);
        sum_vec256 = _mm256_add_epi32(sum_vec256, prod_hi);
    } // for (; i <= dim - 32; i += 32)

    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum_vec256),
                                 _mm256_extracti128_si256(sum_vec256, 1));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
    total += (uint64_t)(uint32_t)_mm_cvtsi128_si32(s128);
    if (total > ssd_cutoff)
    {
        return ssd_cutoff + 1;
    }
#endif

    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)q_res[i] - (int32_t)cand_res[i];
        total += (uint64_t)(diff * diff);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i < dim; i++)

    return total;
}

/**
 * @brief Scalar fallback for 32 3D candidate evaluation.
 */
static inline uint32_t rq8_fastscan_32x_3d_scalar(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
    uint32_t mask = 0;
    int32_t qx = query_res[0];
    int32_t qy = query_res[1];
    int32_t qz = query_res[2];

    for (int i = 0; i < 32; i++)
    {
        int32_t dx = qx - (int32_t)block_x[i];
        int32_t dy = qy - (int32_t)block_y[i];
        int32_t dz = qz - (int32_t)block_z[i];
        uint64_t dist = (uint64_t)(dx * dx + dy * dy + dz * dz);
        if (dist <= ssd_cutoff)
        {
            mask |= (1U << i);
        }
    } // for (int i = 0; i < 32; i++)

    return mask;
}

/**
 * @brief Scalar fallback for 32 D-dim candidate evaluation in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x_generic_scalar(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    uint32_t mask = 0;

    for (int i = 0; i < 32; i++)
    {
        uint64_t dist = 0;
        for (long d = 0; d < dim; d++)
        {
            int32_t diff = (int32_t)query_res[d] - (int32_t)block_coords[d * 32 + i];
            dist += (uint64_t)(diff * diff);
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

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * @brief AVX2 256-bit SIMD kernel for 32 3D candidates in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x_3d_avx2(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m256i qx = _mm256_set1_epi16(query_res[0]);
    __m256i qy = _mm256_set1_epi16(query_res[1]);
    __m256i qz = _mm256_set1_epi16(query_res[2]);

    // Candidates 0..31: 32 bytes loaded per axis
    __m256i cx = _mm256_loadu_si256((const __m256i *)block_x);
    __m256i cy = _mm256_loadu_si256((const __m256i *)block_y);
    __m256i cz = _mm256_loadu_si256((const __m256i *)block_z);

    // Sub-block 0: candidates 0..15
    __m256i cx0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cx));
    __m256i cy0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cy));
    __m256i cz0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cz));

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
    __m256i cx1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(cx, 1));
    __m256i cy1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(cy, 1));
    __m256i cz1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(cz, 1));

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
 * @brief AVX2 256-bit SIMD kernel for 32 D-dim candidates in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x_generic_avx2(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
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
        __m256i qd = _mm256_set1_epi16(query_res[d]);
        const int8_t *cd_ptr = block_coords + d * 32;

        __m256i c32 = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m256i c0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(c32));
        __m256i c1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(c32, 1));

        __m256i diff0 = _mm256_sub_epi16(qd, c0);
        __m256i d0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff0));
        __m256i d0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff0, 1));
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(d0_lo, d0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(d0_hi, d0_hi));

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

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * @brief AVX-512 512-bit SIMD kernel for 32 3D candidates in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x_3d_avx512(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m512i v_bias = _mm512_set1_epi32((int32_t)0x80000000U);
    __m512i v_cut = _mm512_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m512i qx = _mm512_set1_epi16(query_res[0]);
    __m512i qy = _mm512_set1_epi16(query_res[1]);
    __m512i qz = _mm512_set1_epi16(query_res[2]);

    __m256i rx = _mm256_loadu_si256((const __m256i *)block_x);
    __m256i ry = _mm256_loadu_si256((const __m256i *)block_y);
    __m256i rz = _mm256_loadu_si256((const __m256i *)block_z);

    __m512i cx = _mm512_cvtepi8_epi16(rx);
    __m512i cy = _mm512_cvtepi8_epi16(ry);
    __m512i cz = _mm512_cvtepi8_epi16(rz);

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

    __mmask16 fail_lo = _mm512_cmpgt_epi32_mask(_mm512_xor_si512(sum_lo, v_bias), v_cut);
    __mmask16 fail_hi = _mm512_cmpgt_epi32_mask(_mm512_xor_si512(sum_hi, v_bias), v_cut);

    uint32_t fail32 = ((uint32_t)fail_hi << 16) | (uint32_t)fail_lo;
    return ~fail32;
}

/**
 * @brief AVX-512 512-bit SIMD kernel for 32 D-dim candidates in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x_generic_avx512(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff > (uint64_t)INT32_MAX)
    {
        return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m512i v_bias = _mm512_set1_epi32((int32_t)0x80000000U);
    __m512i v_cut = _mm512_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m512i v_clamp = _mm512_set1_epi32((int32_t)(cut32 + 1));

    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();

    for (long d = 0; d < dim; d++)
    {
        __m512i qd = _mm512_set1_epi16(query_res[d]);
        const int8_t *cd_ptr = block_coords + d * 32;

        __m256i r32 = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m512i c32 = _mm512_cvtepi8_epi16(r32);
        __m512i diff = _mm512_sub_epi16(qd, c32);

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

    __mmask16 fail_lo = _mm512_cmpgt_epi32_mask(_mm512_xor_si512(sum_lo, v_bias), v_cut);
    __mmask16 fail_hi = _mm512_cmpgt_epi32_mask(_mm512_xor_si512(sum_hi, v_bias), v_cut);

    uint32_t fail32 = ((uint32_t)fail_hi << 16) | (uint32_t)fail_lo;
    return ~fail32;
}
#endif // __AVX512F__ && __AVX512BW__

/**
 * @brief Dispatch FastScan for 32 3D candidates.
 */
static inline uint32_t rq8_fastscan_32x_3d(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return rq8_fastscan_32x_3d_avx512(query_res, block_x, block_y, block_z, ssd_cutoff);
#elif defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return rq8_fastscan_32x_3d_avx2(query_res, block_x, block_y, block_z, ssd_cutoff);
#else
    return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
#endif
}

/**
 * @brief Dispatch FastScan for 32 D-dim candidates in transposed layout.
 */
static inline uint32_t rq8_fastscan_32x(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (dim == 3)
    {
        return rq8_fastscan_32x_3d(
            query_res,
            block_coords,
            block_coords + 32,
            block_coords + 64,
            ssd_cutoff);
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return rq8_fastscan_32x_generic_avx512(query_res, block_coords, dim, ssd_cutoff);
#elif defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return rq8_fastscan_32x_generic_avx2(query_res, block_coords, dim, ssd_cutoff);
#else
    return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
#endif
}

/**
 * @brief Save quantized residual dataset buffer and parameters to a binary .rq8 file.
 */
int rq8_save_sidecar(
    const char      *filepath,
    const RQ8Params *params,
    const int8_t    *data,
    long             num_frames);

/**
 * @brief Load quantized residual dataset buffer and parameters from a binary .rq8 file.
 */
int rq8_load_sidecar(
    const char *filepath,
    RQ8Params  *params,
    int8_t    **data,
    long       *num_frames);

/**
 * @brief Query the active SIMD register mode available on the host CPU.
 */
RQ8SimdMode rq8_get_simd_mode(void);

/**
 * @brief Human-readable string for the active SIMD register mode.
 */
const char *rq8_get_simd_mode_str(void);

#endif // RESIDUAL_QUANT_H

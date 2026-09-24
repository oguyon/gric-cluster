#ifndef RESIDUAL_QUANT_H
#define RESIDUAL_QUANT_H

/**
 * @file residual_quant.h
 * @brief 8-bit Residual Vector Quantization (RQ8) with metric lower-bounding.
 */

#include "gric_simd.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

/** Magic identifier for .rq8 sidecar files */
#define RQ8_FILE_MAGIC "RQ8_0002"

#include "e8_lattice.h"

/** Universal transposed SIMD block size (32 candidates, 32 bytes per dimension) */
#define RQ8_FASTSCAN_BLOCK_SIZE 32

/**
 * @brief Lattice quantization geometry mode for RQ8.
 */
typedef enum
{
    RQ8_LATTICE_CUBIC = 0, /**< Standard Z^D scalar hypercube */
    RQ8_LATTICE_E8    = 1  /**< E8 block lattice quantization */
} RQ8LatticeMode;

/**
 * @brief Parameters defining 8-bit residual quantization within a cluster ball.
 */
typedef struct
{
    float          rlim;         /**< Maximum cluster covering radius */
    float          scale;        /**< Step delta: rlim / 127.0f */
    float          inv_scale;    /**< Reciprocal step: 127.0f / rlim */
    float          err_radius;   /**< Max Euclidean error: sqrt(dim) * scale * 0.5f */
    long           dim;          /**< Vector dimension (element count per frame) */
    RQ8LatticeMode lattice_mode; /**< Lattice quantization mode */
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
 * @brief Initialize RQ8 parameters with specific lattice geometry (Cubic or E8).
 */
void rq8_init_params_ex(
    RQ8Params      *params,
    float           rlim,
    long            dim,
    RQ8LatticeMode  lattice_mode);

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
int rq8_quantize_query_residual_float(
    const float     *restrict query,
    const float     *restrict anchor,
    int16_t         *restrict dst,
    const RQ8Params *restrict params);

/**
 * @brief Quantize a double-precision query residual (query - anchor) to int16_t.
 */
int rq8_quantize_query_residual_double(
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
 * @brief Compute guaranteed metric cutoff threshold in squared float units for ADC.
 */
static inline float rq8_compute_cutoff_thresh_adc(
    double           cur_tau,
    const RQ8Params *params,
    double           eps)
{
    if (params == NULL || params->scale <= 0.0f)
    {
        return 1e30f;
    }

    double eff_tau = cur_tau * (1.0 + eps);
    double raw_thresh = (eff_tau + 1.0 * (double)params->err_radius) * (double)params->inv_scale;
    return (raw_thresh > 0.0) ? (float)(raw_thresh * raw_thresh) : 0.0f;
}

/**
 * @brief Compute guaranteed metric lower bound between query and candidate residual in ADC.
 */
static inline double rq8_compute_lower_bound_adc(
    float            dist_sq,
    const RQ8Params *params,
    double           eps)
{
    if (params == NULL || params->scale <= 0.0f)
    {
        return 0.0;
    }

    double d_quant = sqrt((double)dist_sq) * (double)params->scale;
    double d_lb = d_quant - 1.0 * (double)params->err_radius;
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
 * rq8_prepare_query_residual_float() - Compute normalized float residual vector for ADC.
 * @query:  Pointer to query float vector [dim].
 * @anchor: Pointer to cluster anchor float vector [dim].
 * @dst:    Pointer to destination float vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
static inline void rq8_prepare_query_residual_float(
    const float     *restrict query,
    const float     *restrict anchor,
    float           *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    float inv_scale = params->inv_scale;
    for (long i = 0; i < dim; i++)
    {
        dst[i] = (query[i] - anchor[i]) * inv_scale;
    }
}

/**
 * rq8_prepare_query_residual_double() - Compute normalized float residual vector for ADC.
 * @query:  Pointer to query double vector [dim].
 * @anchor: Pointer to cluster anchor double vector [dim].
 * @dst:    Pointer to destination float vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
static inline void rq8_prepare_query_residual_double(
    const double    *restrict query,
    const double    *restrict anchor,
    float           *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    double inv_scale = (double)params->inv_scale;
    for (long i = 0; i < dim; i++)
    {
        dst[i] = (float)((query[i] - anchor[i]) * inv_scale);
    }
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
    uint64_t total = 0;
    long i = 0;

#if defined(__AVX2__) && !defined(__CUDACC__)
    if (dim >= 8)
    {
        __m256i acc64 = _mm256_setzero_si256();
        for (; i <= dim - 8; i += 8)
        {
            __m128i c8 = _mm_loadl_epi64((const __m128i *)&cand_res[i]);
            __m256i c256 = _mm256_cvtepi8_epi32(c8);
            __m128i q8 = _mm_loadu_si128((const __m128i *)&q_res[i]);
            __m256i q256 = _mm256_cvtepi16_epi32(q8);
            __m256i diff = _mm256_sub_epi32(q256, c256);
            __m256i p32 = _mm256_mullo_epi32(diff, diff);
            __m256i sq64_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p32));
            __m256i sq64_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(p32, 1));
            acc64 = _mm256_add_epi64(acc64, _mm256_add_epi64(sq64_lo, sq64_hi));

            if ((i & 31) == 24 || i == dim - 8)
            {
                uint64_t partial = (uint64_t)_mm256_extract_epi64(acc64, 0) +
                                   (uint64_t)_mm256_extract_epi64(acc64, 1) +
                                   (uint64_t)_mm256_extract_epi64(acc64, 2) +
                                   (uint64_t)_mm256_extract_epi64(acc64, 3);
                if (partial > ssd_cutoff)
                {
                    return ssd_cutoff + 1;
                }
            }
        }
        total = (uint64_t)_mm256_extract_epi64(acc64, 0) +
                (uint64_t)_mm256_extract_epi64(acc64, 1) +
                (uint64_t)_mm256_extract_epi64(acc64, 2) +
                (uint64_t)_mm256_extract_epi64(acc64, 3);
    }
#endif

    for (; i < dim; i++)
    {
        int64_t diff = (int64_t)q_res[i] - (int64_t)cand_res[i];
        total += (uint64_t)(diff * diff);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i < dim; i++)

    return total;
}

/**
 * @brief Compute squared difference between float query residual and int8 candidate with cutoff.
 */
static inline float rq8_dist_asym_cutoff_f32(
    const float  *restrict q_res,
    const int8_t *restrict cand_res,
    long                   dim,
    float                  cutoff_f)
{
    float total = 0.0f;
    long i = 0;

#if defined(__AVX2__) && !defined(__CUDACC__)
    if (dim >= 8)
    {
        __m256 acc = _mm256_setzero_ps();
        for (; i <= dim - 8; i += 8)
        {
            __m128i c8 = _mm_loadl_epi64((const __m128i *)(const void *)&cand_res[i]);
            __m256 c256 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c8));
            __m256 q256 = _mm256_loadu_ps(&q_res[i]);
            __m256 diff = _mm256_sub_ps(q256, c256);
            acc = _mm256_fmadd_ps(diff, diff, acc);

            if ((i & 31) == 24 || i == dim - 8)
            {
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 sum4 = _mm_add_ps(lo, hi);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                if (_mm_cvtss_f32(sum4) > cutoff_f)
                {
                    return cutoff_f + 1.0f;
                }
            }
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4);
        sum4 = _mm_hadd_ps(sum4, sum4);
        total = _mm_cvtss_f32(sum4);
    }
#endif

    for (; i < dim; i++)
    {
        float diff = q_res[i] - (float)cand_res[i];
        total += diff * diff;
        if (total > cutoff_f)
        {
            return cutoff_f + 1.0f;
        }
    }
    return total;
}

/**
 * @brief Batched 1x4 evaluation of query residual against 4 int8 candidates with early cutoff.
 */
static inline void rq8_dist_asym_cutoff_batch_1x4(
    const float        *restrict q_res,
    const int8_t *const *restrict cands,
    long                         dim,
    float                        cutoff_f,
    float                        out_dist_sq[4])
{
    out_dist_sq[0] = 0.0f;
    out_dist_sq[1] = 0.0f;
    out_dist_sq[2] = 0.0f;
    out_dist_sq[3] = 0.0f;

    long i = 0;

#if defined(__AVX2__) && !defined(__CUDACC__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i <= dim - 8; i += 8)
    {
        __m256 q = _mm256_loadu_ps(&q_res[i]);

        __m128i c0_8 = _mm_loadl_epi64((const __m128i *)(const void *)&cands[0][i]);
        __m256 c0_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c0_8));
        __m256 diff0 = _mm256_sub_ps(q, c0_ps);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m128i c1_8 = _mm_loadl_epi64((const __m128i *)(const void *)&cands[1][i]);
        __m256 c1_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c1_8));
        __m256 diff1 = _mm256_sub_ps(q, c1_ps);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        __m128i c2_8 = _mm_loadl_epi64((const __m128i *)(const void *)&cands[2][i]);
        __m256 c2_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c2_8));
        __m256 diff2 = _mm256_sub_ps(q, c2_ps);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        __m128i c3_8 = _mm_loadl_epi64((const __m128i *)(const void *)&cands[3][i]);
        __m256 c3_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c3_8));
        __m256 diff3 = _mm256_sub_ps(q, c3_ps);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        if ((i & 63) == 56 || i == dim - 8)
        {
            __m128 sum0 = _mm_add_ps(
                _mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1)
            );
            sum0 = _mm_hadd_ps(sum0, sum0);
            sum0 = _mm_hadd_ps(sum0, sum0);

            __m128 sum1 = _mm_add_ps(
                _mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1)
            );
            sum1 = _mm_hadd_ps(sum1, sum1);
            sum1 = _mm_hadd_ps(sum1, sum1);

            __m128 sum2 = _mm_add_ps(
                _mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1)
            );
            sum2 = _mm_hadd_ps(sum2, sum2);
            sum2 = _mm_hadd_ps(sum2, sum2);

            __m128 sum3 = _mm_add_ps(
                _mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1)
            );
            sum3 = _mm_hadd_ps(sum3, sum3);
            sum3 = _mm_hadd_ps(sum3, sum3);

            if (_mm_cvtss_f32(sum0) > cutoff_f && _mm_cvtss_f32(sum1) > cutoff_f &&
                _mm_cvtss_f32(sum2) > cutoff_f && _mm_cvtss_f32(sum3) > cutoff_f)
            {
                out_dist_sq[0] = cutoff_f + 1.0f;
                out_dist_sq[1] = cutoff_f + 1.0f;
                out_dist_sq[2] = cutoff_f + 1.0f;
                out_dist_sq[3] = cutoff_f + 1.0f;
                return;
            }
        }
    } // for (; i <= dim - 8; i += 8)

    __m128 s0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    s0 = _mm_hadd_ps(s0, s0);
    s0 = _mm_hadd_ps(s0, s0);
    out_dist_sq[0] = _mm_cvtss_f32(s0);

    __m128 s1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
    s1 = _mm_hadd_ps(s1, s1);
    s1 = _mm_hadd_ps(s1, s1);
    out_dist_sq[1] = _mm_cvtss_f32(s1);

    __m128 s2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
    s2 = _mm_hadd_ps(s2, s2);
    s2 = _mm_hadd_ps(s2, s2);
    out_dist_sq[2] = _mm_cvtss_f32(s2);

    __m128 s3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
    s3 = _mm_hadd_ps(s3, s3);
    s3 = _mm_hadd_ps(s3, s3);
    out_dist_sq[3] = _mm_cvtss_f32(s3);
#endif

    for (; i < dim; i++)
    {
        float q = q_res[i];
        float d0 = q - (float)cands[0][i];
        out_dist_sq[0] += d0 * d0;
        float d1 = q - (float)cands[1][i];
        out_dist_sq[1] += d1 * d1;
        float d2 = q - (float)cands[2][i];
        out_dist_sq[2] += d2 * d2;
        float d3 = q - (float)cands[3][i];
        out_dist_sq[3] += d3 * d3;
    }
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
        uint64_t dist = (uint64_t)((int64_t)dx * (int64_t)dx);
        dist += (uint64_t)((int64_t)dy * (int64_t)dy);
        dist += (uint64_t)((int64_t)dz * (int64_t)dz);
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
            int64_t diff = (int64_t)query_res[d] - (int64_t)block_coords[d * 32 + i];
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

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * @brief AVX2 256-bit SIMD kernel for 32 3D candidates in transposed layout.
 */
GRIC_TARGET_AVX2
static inline uint32_t rq8_fastscan_32x_3d_avx2(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= 0xFFFFFFFFULL)
    {
        return 0xFFFFFFFFU;
    }

    for (int k = 0; k < 3; k++)
    {
        if (query_res[k] > 32640 || query_res[k] < -32640)
        {
            return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
        }
    }

    __m256i raw_x = _mm256_loadu_si256((const __m256i *)(const void *)block_x);
    __m256i raw_y = _mm256_loadu_si256((const __m256i *)(const void *)block_y);
    __m256i raw_z = _mm256_loadu_si256((const __m256i *)(const void *)block_z);

    __m256i pair_xy_lo = _mm256_unpacklo_epi8(raw_x, raw_y);
    __m256i pair_xy_hi = _mm256_unpackhi_epi8(raw_x, raw_y);

    __m128i plo0 = _mm256_castsi256_si128(pair_xy_lo);
    __m128i plo1 = _mm256_extracti128_si256(pair_xy_lo, 1);
    __m128i phi0 = _mm256_castsi256_si128(pair_xy_hi);
    __m128i phi1 = _mm256_extracti128_si256(pair_xy_hi, 1);

    __m256i c_0_7 = _mm256_cvtepi8_epi16(plo0);
    __m256i c_16_23 = _mm256_cvtepi8_epi16(plo1);
    __m256i c_8_15 = _mm256_cvtepi8_epi16(phi0);
    __m256i c_24_31 = _mm256_cvtepi8_epi16(phi1);

    int16_t q0 = query_res[0];
    int16_t q1 = query_res[1];
    int32_t q_xy = (int32_t)((uint16_t)q0 | ((uint32_t)(uint16_t)q1 << 16));
    __m256i vq_xy = _mm256_set1_epi32(q_xy);

    __m256i dxy0 = _mm256_sub_epi16(vq_xy, c_0_7);
    __m256i dxy1 = _mm256_sub_epi16(vq_xy, c_8_15);
    __m256i dxy2 = _mm256_sub_epi16(vq_xy, c_16_23);
    __m256i dxy3 = _mm256_sub_epi16(vq_xy, c_24_31);

    __m256i acc0 = _mm256_madd_epi16(dxy0, dxy0);
    __m256i acc1 = _mm256_madd_epi16(dxy1, dxy1);
    __m256i acc2 = _mm256_madd_epi16(dxy2, dxy2);
    __m256i acc3 = _mm256_madd_epi16(dxy3, dxy3);

    __m128i bz0 = _mm256_castsi256_si128(raw_z);
    __m128i bz1 = _mm256_extracti128_si256(raw_z, 1);

    __m256i cz0 = _mm256_cvtepi8_epi32(bz0);
    __m256i cz1 = _mm256_cvtepi8_epi32(_mm_srli_si128(bz0, 8));
    __m256i cz2 = _mm256_cvtepi8_epi32(bz1);
    __m256i cz3 = _mm256_cvtepi8_epi32(_mm_srli_si128(bz1, 8));

    __m256i qz = _mm256_set1_epi32((int32_t)query_res[2]);
    __m256i dz0 = _mm256_sub_epi32(qz, cz0);
    __m256i dz1 = _mm256_sub_epi32(qz, cz1);
    __m256i dz2 = _mm256_sub_epi32(qz, cz2);
    __m256i dz3 = _mm256_sub_epi32(qz, cz3);

    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(dz0, dz0));
    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(dz1, dz1));
    acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(dz2, dz2));
    acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(dz3, dz3));

    __m256i vcut = _mm256_set1_epi32((int32_t)(uint32_t)ssd_cutoff);
    __m256i m0 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc0, vcut), acc0);
    __m256i m1 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc1, vcut), acc1);
    __m256i m2 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc2, vcut), acc2);
    __m256i m3 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc3, vcut), acc3);

    uint32_t mask0 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m0));
    uint32_t mask1 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m1));
    uint32_t mask2 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m2));
    uint32_t mask3 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m3));

    return mask0 | (mask1 << 8) | (mask2 << 16) | (mask3 << 24);
}

/**
 * @brief AVX2 256-bit SIMD kernel for 32 D-dim candidates in transposed layout.
 */
GRIC_TARGET_AVX2
static inline uint32_t rq8_fastscan_32x_generic_avx2(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= 0xFFFFFFFFULL)
    {
        return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
    }

    __m256i v_bound = _mm256_set1_epi16(32640);
    __m256i v_nbound = _mm256_set1_epi16(-32640);
    long d_check = 0;
    for (; d_check <= dim - 16; d_check += 16)
    {
        __m256i q = _mm256_loadu_si256((const __m256i *)(const void *)(query_res + d_check));
        __m256i gt = _mm256_cmpgt_epi16(q, v_bound);
        __m256i lt = _mm256_cmpgt_epi16(v_nbound, q);
        if (!_mm256_testz_si256(gt, gt) || !_mm256_testz_si256(lt, lt))
        {
            return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
        }
    } // for (; d_check <= dim - 16; d_check += 16)

    for (; d_check < dim; d_check++)
    {
        if (query_res[d_check] > 32640 || query_res[d_check] < -32640)
        {
            return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
        }
    } // for (; d_check < dim; d_check++)

    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    long d = 0;
    for (; d <= dim - 2; d += 2)
    {
        __m256i v_d0 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_coords + d * 32)
        );
        __m256i v_d1 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_coords + (d + 1) * 32)
        );

        __m256i pair_lo = _mm256_unpacklo_epi8(v_d0, v_d1);
        __m256i pair_hi = _mm256_unpackhi_epi8(v_d0, v_d1);

        __m128i plo0 = _mm256_castsi256_si128(pair_lo);
        __m128i plo1 = _mm256_extracti128_si256(pair_lo, 1);
        __m128i phi0 = _mm256_castsi256_si128(pair_hi);
        __m128i phi1 = _mm256_extracti128_si256(pair_hi, 1);

        __m256i c_0_7 = _mm256_cvtepi8_epi16(plo0);
        __m256i c_16_23 = _mm256_cvtepi8_epi16(plo1);
        __m256i c_8_15 = _mm256_cvtepi8_epi16(phi0);
        __m256i c_24_31 = _mm256_cvtepi8_epi16(phi1);

        int16_t q0 = query_res[d];
        int16_t q1 = query_res[d + 1];
        int32_t q_pair = (int32_t)((uint16_t)q0 | ((uint32_t)(uint16_t)q1 << 16));
        __m256i vq = _mm256_set1_epi32(q_pair);

        __m256i diff0 = _mm256_sub_epi16(vq, c_0_7);
        __m256i diff1 = _mm256_sub_epi16(vq, c_8_15);
        __m256i diff2 = _mm256_sub_epi16(vq, c_16_23);
        __m256i diff3 = _mm256_sub_epi16(vq, c_24_31);

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(diff0, diff0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(diff1, diff1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(diff2, diff2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(diff3, diff3));

        if (d == 30 || d == 62 || d == 126 || d == 254)
        {
            __m256i vcut = _mm256_set1_epi32((int32_t)(uint32_t)ssd_cutoff);
            __m256i m0 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc0, vcut), acc0);
            __m256i m1 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc1, vcut), acc1);
            __m256i m2 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc2, vcut), acc2);
            __m256i m3 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc3, vcut), acc3);

            __m256i any_pass = _mm256_or_si256(
                _mm256_or_si256(m0, m1),
                _mm256_or_si256(m2, m3)
            );
            if (_mm256_testz_si256(any_pass, any_pass))
            {
                return 0;
            }
        }
    } // for (; d <= dim - 2; d += 2)

    if (d < dim)
    {
        __m256i v_d0 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_coords + d * 32)
        );
        __m128i b0 = _mm256_castsi256_si128(v_d0);
        __m128i b1 = _mm256_extracti128_si256(v_d0, 1);

        __m256i c0 = _mm256_cvtepi8_epi32(b0);
        __m256i c1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8));
        __m256i c2 = _mm256_cvtepi8_epi32(b1);
        __m256i c3 = _mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8));

        __m256i vq = _mm256_set1_epi32((int32_t)query_res[d]);
        __m256i d0 = _mm256_sub_epi32(vq, c0);
        __m256i d1 = _mm256_sub_epi32(vq, c1);
        __m256i d2 = _mm256_sub_epi32(vq, c2);
        __m256i d3 = _mm256_sub_epi32(vq, c3);

        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(d0, d0));
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(d1, d1));
        acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(d2, d2));
        acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(d3, d3));
    } // if (d < dim)

    __m256i vcut = _mm256_set1_epi32((int32_t)(uint32_t)ssd_cutoff);
    __m256i m0 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc0, vcut), acc0);
    __m256i m1 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc1, vcut), acc1);
    __m256i m2 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc2, vcut), acc2);
    __m256i m3 = _mm256_cmpeq_epi32(_mm256_min_epu32(acc3, vcut), acc3);

    uint32_t mask0 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m0));
    uint32_t mask1 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m1));
    uint32_t mask2 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m2));
    uint32_t mask3 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(m3));

    return mask0 | (mask1 << 8) | (mask2 << 16) | (mask3 << 24);
}
#endif // __AVX2__

#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET

/**
 * @brief AVX-512 512-bit SIMD kernel for 32 3D candidates in transposed layout.
 */
GRIC_TARGET_AVX512
static inline uint32_t rq8_fastscan_32x_3d_avx512(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_x,
    const int8_t  *restrict block_y,
    const int8_t  *restrict block_z,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= 0xFFFFFFFFULL)
    {
        return 0xFFFFFFFFU;
    }

    __m512i qx = _mm512_set1_epi32((int32_t)query_res[0]);
    __m512i qy = _mm512_set1_epi32((int32_t)query_res[1]);
    __m512i qz = _mm512_set1_epi32((int32_t)query_res[2]);
    __m512i vcut = _mm512_set1_epi32((int32_t)(uint32_t)ssd_cutoff);

    /* Candidates 0..15 */
    __m512i x0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)block_x));
    __m512i y0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)block_y));
    __m512i z0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)block_z));
    __m512i dx0 = _mm512_sub_epi32(qx, x0);
    __m512i dy0 = _mm512_sub_epi32(qy, y0);
    __m512i dz0 = _mm512_sub_epi32(qz, z0);
    __m512i dsq0 = _mm512_add_epi32(
        _mm512_mullo_epi32(dx0, dx0),
        _mm512_add_epi32(_mm512_mullo_epi32(dy0, dy0), _mm512_mullo_epi32(dz0, dz0))
    );
    __mmask16 m0 = _mm512_cmple_epu32_mask(dsq0, vcut);

    /* Candidates 16..31 */
    __m128i bx1 = _mm_loadu_si128((const __m128i *)(const void *)(block_x + 16));
    __m128i by1 = _mm_loadu_si128((const __m128i *)(const void *)(block_y + 16));
    __m128i bz1 = _mm_loadu_si128((const __m128i *)(const void *)(block_z + 16));
    __m512i x1 = _mm512_cvtepi8_epi32(bx1);
    __m512i y1 = _mm512_cvtepi8_epi32(by1);
    __m512i z1 = _mm512_cvtepi8_epi32(bz1);
    __m512i dx1 = _mm512_sub_epi32(qx, x1);
    __m512i dy1 = _mm512_sub_epi32(qy, y1);
    __m512i dz1 = _mm512_sub_epi32(qz, z1);
    __m512i dsq1 = _mm512_add_epi32(
        _mm512_mullo_epi32(dx1, dx1),
        _mm512_add_epi32(_mm512_mullo_epi32(dy1, dy1), _mm512_mullo_epi32(dz1, dz1))
    );
    __mmask16 m1 = _mm512_cmple_epu32_mask(dsq1, vcut);

    return (uint32_t)m0 | ((uint32_t)m1 << 16);
}

/**
 * @brief AVX-512 512-bit SIMD kernel for 32 D-dim candidates in transposed layout.
 */
GRIC_TARGET_AVX512
static inline uint32_t rq8_fastscan_32x_generic_avx512(
    const int16_t *restrict query_res,
    const int8_t  *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    for (long d = 0; d < dim; d++)
    {
        __m128i b0 = _mm_loadu_si128((const __m128i *)(const void *)(block_coords + d * 32));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(const void *)(block_coords + d * 32 + 16));

        __m512i c0 = _mm512_cvtepi8_epi32(b0);
        __m512i c1 = _mm512_cvtepi8_epi32(b1);

        __m512i vq = _mm512_set1_epi32((int32_t)query_res[d]);
        __m512i diff0 = _mm512_sub_epi32(vq, c0);
        __m512i diff1 = _mm512_sub_epi32(vq, c1);

        __m512i p0 = _mm512_mullo_epi32(diff0, diff0);
        __m512i p1 = _mm512_mullo_epi32(diff1, diff1);

        acc0 = _mm512_add_epi64(acc0, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p0)));
        acc1 = _mm512_add_epi64(acc1,
                                _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(p0, 1)));
        acc2 = _mm512_add_epi64(acc2, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(p1)));
        acc3 = _mm512_add_epi64(acc3,
                                _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(p1, 1)));

        if (d == 31 || d == 63 || d == 127 || d == 255)
        {
            __m512i v_cut = _mm512_set1_epi64((int64_t)ssd_cutoff);
            __mmask8 m0 = _mm512_cmple_epu64_mask(acc0, v_cut);
            __mmask8 m1 = _mm512_cmple_epu64_mask(acc1, v_cut);
            __mmask8 m2 = _mm512_cmple_epu64_mask(acc2, v_cut);
            __mmask8 m3 = _mm512_cmple_epu64_mask(acc3, v_cut);
            if ((m0 | m1 | m2 | m3) == 0)
            {
                return 0;
            }
        }
    } // for (long d = 0; d < dim; d++)

    __m512i v_cut = _mm512_set1_epi64((int64_t)ssd_cutoff);
    __mmask8 m0 = _mm512_cmple_epu64_mask(acc0, v_cut);
    __mmask8 m1 = _mm512_cmple_epu64_mask(acc1, v_cut);
    __mmask8 m2 = _mm512_cmple_epu64_mask(acc2, v_cut);
    __mmask8 m3 = _mm512_cmple_epu64_mask(acc3, v_cut);

    return (uint32_t)m0 | ((uint32_t)m1 << 8) | ((uint32_t)m2 << 16) | ((uint32_t)m3 << 24);
}
#endif // GRIC_HAVE_AVX512_TARGET

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
#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return rq8_fastscan_32x_3d_avx512(query_res, block_x, block_y, block_z, ssd_cutoff);
    }
#endif
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return rq8_fastscan_32x_3d_avx2(query_res, block_x, block_y, block_z, ssd_cutoff);
    }
#endif
    return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
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

#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return rq8_fastscan_32x_generic_avx512(query_res, block_coords, dim, ssd_cutoff);
    }
#endif
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return rq8_fastscan_32x_generic_avx2(query_res, block_coords, dim, ssd_cutoff);
    }
#endif
    return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
}

/**
 * @brief Scalar fallback for 32-candidate RQ8 ADC FastScan.
 */
static inline uint32_t rq8_fastscan_32x_adc_scalar(
    const float  *restrict q_adc,
    const int8_t *restrict block_coords,
    long                  dim,
    float                 cutoff_f)
{
    uint32_t mask = 0;
    for (int i = 0; i < 32; i++)
    {
        float dist = 0.0f;
        for (long d = 0; d < dim; d++)
        {
            float diff = q_adc[d] - (float)block_coords[d * 32 + i];
            dist += diff * diff;
            if (dist > cutoff_f)
            {
                break;
            }
        }
        if (dist <= cutoff_f)
        {
            mask |= (1U << i);
        }
    }
    return mask;
}

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
GRIC_TARGET_AVX2
static inline uint32_t rq8_fastscan_32x_adc_avx2(
    const float  *restrict q_adc,
    const int8_t *restrict block_coords,
    long                  dim,
    float                 cutoff_f)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (long d = 0; d < dim; d++)
    {
        __m256 v_qd = _mm256_set1_ps(q_adc[d]);
        const int8_t *cd_ptr = block_coords + d * 32;

        __m256i raw32 = _mm256_loadu_si256((const __m256i *)(const void *)cd_ptr);
        __m128i b_lo = _mm256_castsi256_si128(raw32);
        __m128i b_hi = _mm256_extracti128_si256(raw32, 1);

        __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_lo));
        __m256 c1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b_lo, 8)));
        __m256 c2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b_hi));
        __m256 c3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b_hi, 8)));

        __m256 diff0 = _mm256_sub_ps(v_qd, c0);
        __m256 diff1 = _mm256_sub_ps(v_qd, c1);
        __m256 diff2 = _mm256_sub_ps(v_qd, c2);
        __m256 diff3 = _mm256_sub_ps(v_qd, c3);

        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        if (d == 31 || d == 63 || d == 127 || d == 255)
        {
            __m256 v_cut = _mm256_set1_ps(cutoff_f);
            __m256 cmp0 = _mm256_cmp_ps(acc0, v_cut, _CMP_LE_OQ);
            __m256 cmp1 = _mm256_cmp_ps(acc1, v_cut, _CMP_LE_OQ);
            __m256 cmp2 = _mm256_cmp_ps(acc2, v_cut, _CMP_LE_OQ);
            __m256 cmp3 = _mm256_cmp_ps(acc3, v_cut, _CMP_LE_OQ);
            __m256 any_le = _mm256_or_ps(
                _mm256_or_ps(cmp0, cmp1), _mm256_or_ps(cmp2, cmp3)
            );
            if (_mm256_movemask_ps(any_le) == 0)
            {
                return 0;
            }
        }
    } // for (long d = 0; d < dim; d++)

    __m256 v_cut = _mm256_set1_ps(cutoff_f);
    int m0 = _mm256_movemask_ps(_mm256_cmp_ps(acc0, v_cut, _CMP_LE_OQ));
    int m1 = _mm256_movemask_ps(_mm256_cmp_ps(acc1, v_cut, _CMP_LE_OQ));
    int m2 = _mm256_movemask_ps(_mm256_cmp_ps(acc2, v_cut, _CMP_LE_OQ));
    int m3 = _mm256_movemask_ps(_mm256_cmp_ps(acc3, v_cut, _CMP_LE_OQ));

    return (uint32_t)m0 | ((uint32_t)m1 << 8) |
           ((uint32_t)m2 << 16) | ((uint32_t)m3 << 24);
}
#endif // AVX2

/**
 * @brief Dispatch FastScan for 32 candidates using Asymmetric Distance Computation (ADC).
 */
static inline uint32_t rq8_fastscan_32x_adc(
    const float  *restrict query_res,
    const int8_t *restrict block_coords,
    long                  dim,
    float                 cutoff_f)
{
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return rq8_fastscan_32x_adc_avx2(query_res, block_coords, dim, cutoff_f);
    }
#endif
    return rq8_fastscan_32x_adc_scalar(query_res, block_coords, dim, cutoff_f);
}

/**
 * @brief Save quantized residual dataset buffer and parameters to a binary .rq8 file.
 */
int rq8_save_sidecar(
    const char      *filepath,
    const RQ8Params *params,
    const int8_t    *data,
    long             num_frames,
    uint64_t         fingerprint);

/**
 * @brief Load quantized residual dataset buffer and parameters from a binary .rq8 file.
 */
int rq8_load_sidecar(
    const char *filepath,
    RQ8Params  *params,
    int8_t    **data,
    long       *num_frames,
    uint64_t   *fingerprint);

/**
 * @brief Query the active SIMD register mode available on the host CPU.
 */
RQ8SimdMode rq8_get_simd_mode(void);

/**
 * @brief Human-readable string for the active SIMD register mode.
 */
const char *rq8_get_simd_mode_str(void);

#endif // RESIDUAL_QUANT_H

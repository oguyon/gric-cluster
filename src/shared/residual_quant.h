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

    __m256i raw_x = _mm256_loadu_si256((const __m256i *)(const void *)block_x);
    __m256i raw_y = _mm256_loadu_si256((const __m256i *)(const void *)block_y);
    __m256i raw_z = _mm256_loadu_si256((const __m256i *)(const void *)block_z);

    __m128i bx0 = _mm256_castsi256_si128(raw_x);
    __m128i bx1 = _mm256_extracti128_si256(raw_x, 1);
    __m128i by0 = _mm256_castsi256_si128(raw_y);
    __m128i by1 = _mm256_extracti128_si256(raw_y, 1);
    __m128i bz0 = _mm256_castsi256_si128(raw_z);
    __m128i bz1 = _mm256_extracti128_si256(raw_z, 1);

    __m256i qx = _mm256_set1_epi32((int32_t)query_res[0]);
    __m256i qy = _mm256_set1_epi32((int32_t)query_res[1]);
    __m256i qz = _mm256_set1_epi32((int32_t)query_res[2]);

    __m256i vcut = _mm256_set1_epi32((int32_t)(uint32_t)ssd_cutoff);
    uint32_t mask = 0;

    /* Chunk 0: Candidates 0..7 */
    {
        __m256i x = _mm256_cvtepi8_epi32(bx0);
        __m256i y = _mm256_cvtepi8_epi32(by0);
        __m256i z = _mm256_cvtepi8_epi32(bz0);
        __m256i dx = _mm256_sub_epi32(qx, x);
        __m256i dy = _mm256_sub_epi32(qy, y);
        __m256i dz = _mm256_sub_epi32(qz, z);
        __m256i dsq = _mm256_add_epi32(
            _mm256_mullo_epi32(dx, dx),
            _mm256_add_epi32(_mm256_mullo_epi32(dy, dy), _mm256_mullo_epi32(dz, dz))
        );
        __m256i min_val = _mm256_min_epu32(dsq, vcut);
        __m256i cmp = _mm256_cmpeq_epi32(min_val, dsq);
        mask |= (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(cmp));
    }

    /* Chunk 1: Candidates 8..15 */
    {
        __m256i x = _mm256_cvtepi8_epi32(_mm_srli_si128(bx0, 8));
        __m256i y = _mm256_cvtepi8_epi32(_mm_srli_si128(by0, 8));
        __m256i z = _mm256_cvtepi8_epi32(_mm_srli_si128(bz0, 8));
        __m256i dx = _mm256_sub_epi32(qx, x);
        __m256i dy = _mm256_sub_epi32(qy, y);
        __m256i dz = _mm256_sub_epi32(qz, z);
        __m256i dsq = _mm256_add_epi32(
            _mm256_mullo_epi32(dx, dx),
            _mm256_add_epi32(_mm256_mullo_epi32(dy, dy), _mm256_mullo_epi32(dz, dz))
        );
        __m256i min_val = _mm256_min_epu32(dsq, vcut);
        __m256i cmp = _mm256_cmpeq_epi32(min_val, dsq);
        mask |= ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) << 8);
    }

    /* Chunk 2: Candidates 16..23 */
    {
        __m256i x = _mm256_cvtepi8_epi32(bx1);
        __m256i y = _mm256_cvtepi8_epi32(by1);
        __m256i z = _mm256_cvtepi8_epi32(bz1);
        __m256i dx = _mm256_sub_epi32(qx, x);
        __m256i dy = _mm256_sub_epi32(qy, y);
        __m256i dz = _mm256_sub_epi32(qz, z);
        __m256i dsq = _mm256_add_epi32(
            _mm256_mullo_epi32(dx, dx),
            _mm256_add_epi32(_mm256_mullo_epi32(dy, dy), _mm256_mullo_epi32(dz, dz))
        );
        __m256i min_val = _mm256_min_epu32(dsq, vcut);
        __m256i cmp = _mm256_cmpeq_epi32(min_val, dsq);
        mask |= ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) << 16);
    }

    /* Chunk 3: Candidates 24..31 */
    {
        __m256i x = _mm256_cvtepi8_epi32(_mm_srli_si128(bx1, 8));
        __m256i y = _mm256_cvtepi8_epi32(_mm_srli_si128(by1, 8));
        __m256i z = _mm256_cvtepi8_epi32(_mm_srli_si128(bz1, 8));
        __m256i dx = _mm256_sub_epi32(qx, x);
        __m256i dy = _mm256_sub_epi32(qy, y);
        __m256i dz = _mm256_sub_epi32(qz, z);
        __m256i dsq = _mm256_add_epi32(
            _mm256_mullo_epi32(dx, dx),
            _mm256_add_epi32(_mm256_mullo_epi32(dy, dy), _mm256_mullo_epi32(dz, dz))
        );
        __m256i min_val = _mm256_min_epu32(dsq, vcut);
        __m256i cmp = _mm256_cmpeq_epi32(min_val, dsq);
        mask |= ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) << 24);
    }

    return mask;
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
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();

    for (long d = 0; d < dim; d++)
    {
        __m256i v32 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_coords + d * 32)
        );
        __m128i b_lo = _mm256_castsi256_si128(v32);
        __m128i b_hi = _mm256_extracti128_si256(v32, 1);

        __m256i c0 = _mm256_cvtepi8_epi32(b_lo);
        __m256i c1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b_lo, 8));
        __m256i c2 = _mm256_cvtepi8_epi32(b_hi);
        __m256i c3 = _mm256_cvtepi8_epi32(_mm_srli_si128(b_hi, 8));

        __m256i vq = _mm256_set1_epi32((int32_t)query_res[d]);
        __m256i d0 = _mm256_sub_epi32(vq, c0);
        __m256i d1 = _mm256_sub_epi32(vq, c1);
        __m256i d2 = _mm256_sub_epi32(vq, c2);
        __m256i d3 = _mm256_sub_epi32(vq, c3);

        __m256i p0 = _mm256_mullo_epi32(d0, d0);
        __m256i p1 = _mm256_mullo_epi32(d1, d1);
        __m256i p2 = _mm256_mullo_epi32(d2, d2);
        __m256i p3 = _mm256_mullo_epi32(d3, d3);

        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p0)));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(p0, 1)));
        acc2 = _mm256_add_epi64(acc2, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p1)));
        acc3 = _mm256_add_epi64(acc3, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(p1, 1)));
        acc4 = _mm256_add_epi64(acc4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p2)));
        acc5 = _mm256_add_epi64(acc5, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(p2, 1)));
        acc6 = _mm256_add_epi64(acc6, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p3)));
        acc7 = _mm256_add_epi64(acc7, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(p3, 1)));
    } // for (long d = 0; d < dim; d++)

    __m256i v_cut = _mm256_set1_epi64x((int64_t)ssd_cutoff);
    __m256i v_bias = _mm256_set1_epi64x((int64_t)0x8000000000000000ULL);
    __m256i cut_b = _mm256_xor_si256(v_cut, v_bias);

    uint32_t m0 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc0, v_bias), cut_b))) & 0x0F);
    uint32_t m1 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc1, v_bias), cut_b))) & 0x0F);
    uint32_t m2 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc2, v_bias), cut_b))) & 0x0F);
    uint32_t m3 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc3, v_bias), cut_b))) & 0x0F);
    uint32_t m4 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc4, v_bias), cut_b))) & 0x0F);
    uint32_t m5 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc5, v_bias), cut_b))) & 0x0F);
    uint32_t m6 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc6, v_bias), cut_b))) & 0x0F);
    uint32_t m7 = (uint32_t)(~_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_xor_si256(acc7, v_bias), cut_b))) & 0x0F);

    return m0 | (m1 << 4) | (m2 << 8) | (m3 << 12) |
           (m4 << 16) | (m5 << 20) | (m6 << 24) | (m7 << 28);
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

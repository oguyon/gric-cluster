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
#define RQ8_FILE_MAGIC "RQ8_0002"

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
    for (long i = 0; i < dim; i++)
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
    return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
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
    return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
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
    return rq8_fastscan_32x_3d_scalar(query_res, block_x, block_y, block_z, ssd_cutoff);
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
    return rq8_fastscan_32x_generic_scalar(query_res, block_coords, dim, ssd_cutoff);
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

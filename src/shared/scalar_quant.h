#ifndef SCALAR_QUANT_H
#define SCALAR_QUANT_H

/**
 * @file scalar_quant.h
 * @brief 8-bit Scalar Quantization (SQ8) with metric lower-bounding.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif
#endif

/** Magic identifier for .sq8 sidecar files */
#define SQ8_FILE_MAGIC "SQ8_0001"

/**
 * @brief Parameters defining uniform 8-bit scalar quantization.
 */
typedef struct
{
    float min_val;    /**< Minimum value mapped to 0 */
    float max_val;    /**< Maximum value mapped to 255 */
    float scale;      /**< Step delta: (max_val - min_val) / 255.0f */
    float inv_scale;  /**< Reciprocal step: 255.0f / (max_val - min_val) */
    float err_radius; /**< Max single-vector Euclidean error: sqrt(dim) * scale * 0.5f */
    long  dim;        /**< Vector dimension (element count per frame) */
} SQ8Params;

/**
 * @brief Initialize SQ8 parameters from known min and max values.
 */
void sq8_init_params(
    SQ8Params *params,
    float      min_val,
    float      max_val,
    long       dim);

/**
 * @brief Calibrate SQ8 parameters by scanning a float array.
 */
void sq8_calibrate_float(
    SQ8Params   *params,
    const float *data,
    long         num_elements,
    long         dim);

/**
 * @brief Calibrate SQ8 parameters by scanning a double array.
 */
void sq8_calibrate_double(
    SQ8Params    *params,
    const double *data,
    long          num_elements,
    long          dim);

/**
 * @brief Quantize a single-precision float frame to 8-bit unsigned integers.
 */
void sq8_quantize_float(
    const float     *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params);

/**
 * @brief Quantize a double-precision frame to 8-bit unsigned integers.
 */
void sq8_quantize_double(
    const double    *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params);

/**
 * @brief Compute sum of squared differences between two uint8 vectors using SIMD.
 */
uint64_t sq8_dist_squared_u8(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    long                    dim);

/**
 * @brief Compute dot product between two uint8 vectors using SIMD / AVX-VNNI.
 */
uint64_t sq8_dot_product_u8(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    long                    dim);

/**
 * @brief Compute guaranteed metric lower bound between two quantized vectors.
 */
double sq8_compute_lower_bound(
    const uint8_t   *restrict a,
    const uint8_t   *restrict b,
    const SQ8Params *restrict params,
    double                    epsilon);

/**
 * @brief Save quantized dataset buffer and parameters to a binary .sq8 file.
 */
int sq8_save_sidecar(
    const char      *filepath,
    const SQ8Params *params,
    const uint8_t   *data,
    long             num_frames);

/**
 * @brief Load quantized dataset buffer and parameters from a binary .sq8 file.
 */
int sq8_load_sidecar(
    const char *filepath,
    SQ8Params  *params,
    uint8_t   **data,
    long       *num_frames);

/** Magic identifier for .sq16 sidecar files */
#define SQ16_FILE_MAGIC "SQ16_0001"
#define SQ16_MAX_DIFF_U16 32767ULL
#define SQ16_FASTSCAN_MAX_3D_SSD (3ULL * SQ16_MAX_DIFF_U16 * SQ16_MAX_DIFF_U16)

/**
 * @brief Parameters defining uniform 16-bit scalar quantization into [0, 32767].
 */
typedef struct
{
    float min_val;    /**< Minimum value mapped to 0 */
    float max_val;    /**< Maximum value mapped to 32767 */
    float scale;      /**< Step delta: (max_val - min_val) / 32767.0f */
    float inv_scale;  /**< Reciprocal step: 32767.0f / (max_val - min_val) */
    float err_radius; /**< Max single-vector Euclidean error: sqrt(dim) * scale * 0.5f */
    long  dim;        /**< Vector dimension (element count per frame) */
} SQ16Params;

/**
 * @brief Initialize SQ16 parameters from known min and max values.
 */
void sq16_init_params(
    SQ16Params *params,
    float       min_val,
    float       max_val,
    long        dim);

/**
 * @brief Calibrate SQ16 parameters by scanning a float array.
 */
void sq16_calibrate_float(
    SQ16Params  *params,
    const float *data,
    long         num_elements,
    long         dim);

/**
 * @brief Calibrate SQ16 parameters by scanning a double array.
 */
void sq16_calibrate_double(
    SQ16Params   *params,
    const double *data,
    long          num_elements,
    long          dim);

/**
 * @brief Quantize a single-precision float frame to 16-bit signed integers in [0, 32767].
 */
void sq16_quantize_float(
    const float      *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params);

/**
 * @brief Quantize a double-precision frame to 16-bit signed integers in [0, 32767].
 */
void sq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params);

/**
 * @brief Compute sum of squared differences between two int16 vectors using SIMD.
 */
uint64_t sq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim);

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

/**
 * @brief Compute sum of squared differences between two int16 vectors with early cutoff.
 */
static inline uint64_t sq16_dist_squared_cutoff_i16(
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
 * @brief Load quantized dataset buffer and parameters from a binary .sq16 file.
 */
int sq16_load_sidecar(
    const char  *filepath,
    SQ16Params  *params,
    int16_t    **data,
    long        *num_frames);

/** Universal transposed SIMD block size (32 candidates, 64 bytes per dimension) */
#define SQ16_FASTSCAN_BLOCK_SIZE 32

/**
 * @brief Detected CPU SIMD vector register width for FastScan.
 */
typedef enum
{
    SQ16_SIMD_SCALAR = 0,
    SQ16_SIMD_AVX2   = 1,
    SQ16_SIMD_AVX512 = 2
} SQ16SimdMode;

/**
 * @brief Query the active SIMD register mode available on the host CPU.
 */
SQ16SimdMode sq16_get_simd_mode(void);

/**
 * @brief Human-readable string for the active SIMD register mode.
 */
const char *sq16_get_simd_mode_str(void);

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
static inline uint32_t sq16_fastscan_32x_3d(
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
static inline uint32_t sq16_fastscan_32x(
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

#endif // SCALAR_QUANT_H

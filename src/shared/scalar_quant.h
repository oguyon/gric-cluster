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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
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
    uint64_t total = 0;
    long i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
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
#elif defined(__AVX2__) && \
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

#endif // SCALAR_QUANT_H

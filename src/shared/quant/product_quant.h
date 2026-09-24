#ifndef PRODUCT_QUANT_H
#define PRODUCT_QUANT_H

/**
 * @file product_quant.h
 * @brief Product Quantization (PQ) and Asymmetric Distance Computation (ADC) FastScan.
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

#ifdef __cplusplus
extern "C" {
#endif

/** Magic identifier for .pq sidecar files */
#define PQ_FILE_MAGIC "PQ01_0001"

/** Universal transposed SIMD block size (32 candidates, 32 bytes per subquantizer) */
#define PQ_FASTSCAN_BLOCK_SIZE 32

/** Maximum subquantizers supported */
#define PQ_MAX_M 256

/**
 * @brief Codebook parameters and centroid storage for Product Quantization.
 */
typedef struct
{
    long   dim;           /**< Total vector dimension D */
    int    m;             /**< Number of subquantizers */
    int    d_sub;         /**< Subvector dimension: d_sub = dim / m */
    int    k_centroids;   /**< Centroids per subquantizer (16 for 4-bit, 256 for 8-bit) */
    int    is_4bit;       /**< 1 for 4-bit (FastScan), 0 for 8-bit */
    float *centroids;     /**< Centroid buffer: [m * k_centroids * d_sub] */
    float *sub_radius;    /**< Max quantization radius per subvector */
    float  total_radius;  /**< Max total Euclidean error across all subvectors */
} PQCodebook;

/**
 * @brief Parameters defining quantized query lookup tables (LUTs) in L1 cache.
 */
typedef struct
{
    int      m;           /**< Number of subquantizers */
    int      k_centroids; /**< Number of centroids (16 or 256) */
    float    scale;       /**< Multiplier mapping float dist^2 to uint8 LUT */
    float    inv_scale;   /**< Reciprocal mapping uint8 dist sum to float dist^2 */
    uint8_t *lut_u8;      /**< Packed query LUT: [m * 16] or [m * 256] */
} PQLookupTable;

/**
 * @brief Detected CPU SIMD vector register width for PQ FastScan.
 */
typedef enum
{
    PQ_SIMD_SCALAR = 0,
    PQ_SIMD_AVX2   = 1,
    PQ_SIMD_AVX512 = 2
} PQSimdMode;

/**
 * pq_get_simd_mode() - Query active SIMD mode available on host CPU.
 *
 * Return: Active PQSimdMode enum value.
 */
PQSimdMode pq_get_simd_mode(void);

/**
 * pq_get_simd_mode_str() - Human-readable string of host SIMD mode.
 *
 * Return: Static string name of SIMD mode.
 */
const char *pq_get_simd_mode_str(void);

/**
 * pq_codebook_alloc() - Allocate memory for a PQ codebook.
 * @dim:         Total frame dimension.
 * @m:           Number of subquantizers.
 * @k_centroids: Centroids per subquantizer (16 or 256).
 *
 * Return: Pointer to allocated PQCodebook, or NULL on error.
 */
PQCodebook *pq_codebook_alloc(
    long dim,
    int  m,
    int  k_centroids);

/**
 * pq_codebook_free() - Free PQ codebook structure and internal buffers.
 * @codebook: Pointer to codebook to free.
 */
void pq_codebook_free(
    PQCodebook *codebook);

/**
 * pq_train_codebook() - Train subvector centroids using k-means clustering.
 * @codebook:   Active codebook to populate.
 * @train_data: Array of training frames in contiguous float format.
 * @num_frames: Number of training frames.
 * @max_iters:  Maximum k-means iterations per subquantizer.
 *
 * Return: 0 on success, -1 on error.
 */
int pq_train_codebook(
    PQCodebook  *codebook,
    const float *train_data,
    long         num_frames,
    int          max_iters);

/**
 * pq_quantize_frame_float() - Quantize a float frame into subvector codes.
 * @src:      Input float frame of length dim.
 * @dst_code: Output array of m subquantizer centroid indices (uint8_t).
 * @codebook: Trained PQ codebook.
 */
void pq_quantize_frame_float(
    const float      *restrict src,
    uint8_t          *restrict dst_code,
    const PQCodebook *restrict codebook);

/**
 * pq_quantize_frame_double() - Quantize a double frame into subvector codes.
 * @src:      Input double frame of length dim.
 * @dst_code: Output array of m subquantizer centroid indices (uint8_t).
 * @codebook: Trained PQ codebook.
 */
void pq_quantize_frame_double(
    const double     *restrict src,
    uint8_t          *restrict dst_code,
    const PQCodebook *restrict codebook);

/**
 * pq_build_query_lut_float() - Build 8-bit quantized distance LUT for a query.
 * @query:    Query frame pixel array (float).
 * @codebook: Trained PQ codebook.
 * @lut:      Output PQLookupTable structure to populate.
 * @cur_tau:  Current search radius bound (for dynamic scaling).
 */
void pq_build_query_lut_float(
    const float      *restrict query,
    const PQCodebook *restrict codebook,
    PQLookupTable    *restrict lut,
    double                     cur_tau);

/**
 * pq_build_query_lut_double() - Build 8-bit quantized distance LUT for double query.
 * @query:    Query frame pixel array (double).
 * @codebook: Trained PQ codebook.
 * @lut:      Output PQLookupTable structure to populate.
 * @cur_tau:  Current search radius bound (for dynamic scaling).
 */
void pq_build_query_lut_double(
    const double     *restrict query,
    const PQCodebook *restrict codebook,
    PQLookupTable    *restrict lut,
    double                     cur_tau);

/**
 * pq_transpose_block_codes() - Transpose 32 candidate codes into FastScan layout.
 * @src_codes:   Array of 32 vector codes: [32 * m] (row-major).
 * @dst_block:   Output transposed block buffer: [m * 32] (column-major).
 * @m:           Number of subquantizers.
 * @num_vectors: Number of valid vectors in block (1..32).
 */
void pq_transpose_block_codes(
    const uint8_t *restrict src_codes,
    uint8_t       *restrict dst_block,
    int                     m,
    int                     num_vectors);

/**
 * pq_save_sidecar() - Save codebook and pre-quantized codes to binary .pq file.
 * @filepath:   Destination sidecar path.
 * @codebook:   Trained codebook.
 * @transposed: Transposed block codes buffer.
 * @num_frames: Total frames stored.
 *
 * Return: 0 on success, -1 on error.
 */
int pq_save_sidecar(
    const char       *filepath,
    const PQCodebook *codebook,
    const uint8_t    *transposed,
    long              num_frames);

/**
 * pq_load_sidecar() - Load codebook and pre-quantized codes from binary .pq file.
 * @filepath:       Source sidecar path.
 * @codebook_out:   Output pointer receiving loaded PQCodebook.
 * @transposed_out: Output pointer receiving allocated transposed buffer.
 * @num_frames_out: Output receiving frame count.
 *
 * Return: 0 on success, -1 on error.
 */
int pq_load_sidecar(
    const char   *filepath,
    PQCodebook  **codebook_out,
    uint8_t     **transposed_out,
    long         *num_frames_out);

/**
 * pq_fastscan_32x_scalar() - Scalar reference evaluation of 32 candidates.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 32].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i dist <= cutoff_u8.
 */
static inline uint32_t pq_fastscan_32x_scalar(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    uint32_t mask = 0;

    for (int i = 0; i < 32; i++)
    {
        uint32_t sum = 0;
        for (int s = 0; s < m; s++)
        {
            uint8_t code = block_codes[s * 32 + i] & 0x0F;
            sum += (uint32_t)query_lut[s * 16 + code];
        } // for (int s = 0; s < m; s++)

        if (sum <= (uint32_t)cutoff_u8)
        {
            mask |= (1U << i);
        }
    } // for (int i = 0; i < 32; i++)

    return mask;
}

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))

/**
 * pq_fastscan_32x_avx2() - AVX2 SIMD evaluation of 32 candidates using pshufb.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 32].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i dist <= cutoff_u8.
 */
GRIC_TARGET_AVX2
static inline uint32_t pq_fastscan_32x_avx2(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    __m256i acc = _mm256_setzero_si256();
    __m256i v_cutoff = _mm256_set1_epi8((int8_t)cutoff_u8);
    __m256i v_bias = _mm256_set1_epi8((int8_t)0x80);
    __m256i v_all_ones = _mm256_set1_epi8(-1);

    for (int s = 0; s < m; s++)
    {
        /* Load 16-byte LUT and broadcast to both 128-bit lanes */
        __m128i lut128 = _mm_loadu_si128((const __m128i *)(const void *)(query_lut + s * 16));
        __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

        /* Load 32 candidate codes (1 byte per candidate in [0..15]) */
        __m256i codes = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_codes + s * 32)
        );

        /* Single-instruction 32-way parallel table lookup */
        __m256i looked_up = _mm256_shuffle_epi8(lut256, codes);

        /* Saturating unsigned 8-bit accumulation */
        acc = _mm256_adds_epu8(acc, looked_up);

        if (((s + 1) & 15) == 0 && s + 1 < m)
        {
            __m256i acc_biased = _mm256_xor_si256(acc, v_bias);
            __m256i cut_biased = _mm256_xor_si256(v_cutoff, v_bias);
            __m256i fail = _mm256_cmpgt_epi8(acc_biased, cut_biased);
            if (_mm256_testc_si256(fail, v_all_ones))
            {
                return 0;
            }
        }
    } // for (int s = 0; s < m; s++)

    /*
     * Compare acc <= v_cutoff.
     * In AVX2, _mm256_cmpgt_epi8 does signed comparison.
     * Subtracting 0x80 (XOR with 0x80) maps uint8 to int8 monotonically.
     */
    __m256i acc_biased = _mm256_xor_si256(acc, v_bias);
    __m256i cut_biased = _mm256_xor_si256(v_cutoff, v_bias);
    __m256i fail = _mm256_cmpgt_epi8(acc_biased, cut_biased);

    int fail_mask = _mm256_movemask_epi8(fail);
    return (~(uint32_t)fail_mask);
}

/**
 * pq_fastscan_64x_avx2() - AVX2 SIMD evaluation of 64 candidates concurrently.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 64].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 64-bit bitmask where bit i is 1 if candidate i dist <= cutoff_u8.
 */
GRIC_TARGET_AVX2
static inline uint64_t pq_fastscan_64x_avx2(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i v_cutoff = _mm256_set1_epi8((int8_t)cutoff_u8);
    __m256i v_bias = _mm256_set1_epi8((int8_t)0x80);
    __m256i v_all_ones = _mm256_set1_epi8(-1);

    for (int s = 0; s < m; s++)
    {
        /* Broadcast 16-byte LUT once for all 64 candidates */
        __m128i lut128 = _mm_loadu_si128((const __m128i *)(const void *)(query_lut + s * 16));
        __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

        /* Load 64 candidate codes (32 in block 0, 32 in block 1) */
        const uint8_t *s_codes = block_codes + s * 64;
        __m256i codes0 = _mm256_loadu_si256((const __m256i *)(const void *)s_codes);
        __m256i codes1 = _mm256_loadu_si256((const __m256i *)(const void *)(s_codes + 32));

        /* Dual 32-way parallel table lookups */
        __m256i looked_up0 = _mm256_shuffle_epi8(lut256, codes0);
        __m256i looked_up1 = _mm256_shuffle_epi8(lut256, codes1);

        acc0 = _mm256_adds_epu8(acc0, looked_up0);
        acc1 = _mm256_adds_epu8(acc1, looked_up1);

        if (((s + 1) & 15) == 0 && s + 1 < m)
        {
            __m256i cut_biased = _mm256_xor_si256(v_cutoff, v_bias);
            __m256i fail0 = _mm256_cmpgt_epi8(_mm256_xor_si256(acc0, v_bias), cut_biased);
            __m256i fail1 = _mm256_cmpgt_epi8(_mm256_xor_si256(acc1, v_bias), cut_biased);
            __m256i all_fail = _mm256_and_si256(fail0, fail1);
            if (_mm256_testc_si256(all_fail, v_all_ones))
            {
                return 0;
            }
        }
    } // for (int s = 0; s < m; s++)

    __m256i cut_biased = _mm256_xor_si256(v_cutoff, v_bias);
    __m256i fail0 = _mm256_cmpgt_epi8(_mm256_xor_si256(acc0, v_bias), cut_biased);
    __m256i fail1 = _mm256_cmpgt_epi8(_mm256_xor_si256(acc1, v_bias), cut_biased);

    int mask0 = _mm256_movemask_epi8(fail0);
    int mask1 = _mm256_movemask_epi8(fail1);

    uint32_t pass0 = ~(uint32_t)mask0;
    uint32_t pass1 = ~(uint32_t)mask1;

    return (uint64_t)pass0 | ((uint64_t)pass1 << 32);
}
#endif // x86_64

#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET

/**
 * pq_fastscan_32x_avx512() - AVX-512 evaluation of 32 candidates using pshufb.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 32].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i dist <= cutoff_u8.
 */
GRIC_TARGET_AVX512
static inline uint32_t pq_fastscan_32x_avx512(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    __m256i acc = _mm256_setzero_si256();

    for (int s = 0; s < m; s++)
    {
        __m128i lut128 = _mm_loadu_si128((const __m128i *)(const void *)(query_lut + s * 16));
        __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
        __m256i codes = _mm256_loadu_si256(
            (const __m256i *)(const void *)(block_codes + s * 32)
        );
        __m256i looked_up = _mm256_shuffle_epi8(lut256, codes);
        acc = _mm256_adds_epu8(acc, looked_up);
    } // for (int s = 0; s < m; s++)

    __m256i v_cutoff = _mm256_set1_epi8((int8_t)cutoff_u8);
    __mmask32 pass = _mm256_cmple_epu8_mask(acc, v_cutoff);
    return (uint32_t)pass;
}

/**
 * pq_fastscan_64x_avx512() - AVX-512 evaluation of 64 candidates in 512-bit vector.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 64].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 64-bit bitmask where bit i is 1 if candidate i dist <= cutoff_u8.
 */
GRIC_TARGET_AVX512
static inline uint64_t pq_fastscan_64x_avx512(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    __m512i acc = _mm512_setzero_si512();

    for (int s = 0; s < m; s++)
    {
        __m128i lut128 = _mm_loadu_si128((const __m128i *)(const void *)(query_lut + s * 16));
        __m512i lut512 = _mm512_broadcast_i32x4(lut128);
        __m512i codes = _mm512_loadu_si512((const void *)(block_codes + s * 64));
        __m512i looked_up = _mm512_shuffle_epi8(lut512, codes);
        acc = _mm512_adds_epu8(acc, looked_up);
    } // for (int s = 0; s < m; s++)

    __m512i v_cutoff = _mm512_set1_epi8((int8_t)cutoff_u8);
    __mmask64 pass = _mm512_cmple_epu8_mask(acc, v_cutoff);
    return (uint64_t)pass;
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * pq_fastscan_32x() - Universal dispatcher for 32 candidate evaluations.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 32].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i passes cutoff.
 */
static inline uint32_t pq_fastscan_32x(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return pq_fastscan_32x_avx512(query_lut, block_codes, m, cutoff_u8);
    }
#endif
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return pq_fastscan_32x_avx2(query_lut, block_codes, m, cutoff_u8);
    }
#endif
    return pq_fastscan_32x_scalar(query_lut, block_codes, m, cutoff_u8);
}

/**
 * pq_fastscan_64x_scalar() - Scalar evaluation of 64 candidates.
 */
static inline uint64_t pq_fastscan_64x_scalar(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
    uint32_t lo = pq_fastscan_32x_scalar(query_lut, block_codes, m, cutoff_u8);
    uint32_t hi = pq_fastscan_32x_scalar(query_lut, block_codes + 32, m, cutoff_u8);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}

/**
 * pq_fastscan_64x() - Universal dispatcher for 64 candidate evaluations.
 * @query_lut:   Array of m 16-byte lookup tables: [m * 16].
 * @block_codes: Transposed candidate codes: [m * 64].
 * @m:           Number of subquantizers.
 * @cutoff_u8:   Cutoff distance threshold in uint8 accumulator units.
 *
 * Return: 64-bit bitmask where bit i is 1 if candidate i passes cutoff.
 */
static inline uint64_t pq_fastscan_64x(
    const uint8_t *restrict query_lut,
    const uint8_t *restrict block_codes,
    int                     m,
    uint8_t                 cutoff_u8)
{
#if !defined(__CUDACC__) && GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return pq_fastscan_64x_avx512(query_lut, block_codes, m, cutoff_u8);
    }
#endif
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return pq_fastscan_64x_avx2(query_lut, block_codes, m, cutoff_u8);
    }
#endif
    return pq_fastscan_64x_scalar(query_lut, block_codes, m, cutoff_u8);
}

#ifdef __cplusplus
}
#endif

#endif // PRODUCT_QUANT_H

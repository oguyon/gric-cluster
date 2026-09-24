#ifndef RABIT_QUANT_H
#define RABIT_QUANT_H

/**
 * @file rabit_quant.h
 * @brief Randomized Bit Quantization (RaBitQ) with metric lower-bounding.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif
#endif

/** Magic identifier for .rabitq sidecar files */
#define RABITQ_FILE_MAGIC "RBQ_0001"

/** Transposed SIMD block size (32 candidates per FastScan block) */
#define RABITQ_FASTSCAN_BLOCK_SIZE 32

/** Default deterministic random seed for orthogonal Walsh-Hadamard transform */
#define RABITQ_DEFAULT_SEED 0x9e3779b97f4a7c15ULL

/**
 * @brief Per-vector compact metadata stored alongside quantized bit codes.
 */
typedef struct
{
    float norm;        /**< Original L2 Euclidean norm ||x|| */
    float recon_scale; /**< Hypercube projection scale alpha_x */
    float err_norm;    /**< Residual error norm ||e_x|| = sqrt(||x||^2 - ||x_tilde||^2) */
} RaBitQMeta;

/**
 * @brief Parameters defining Randomized Bit Quantization.
 */
typedef struct
{
    long     dim;                 /**< Original vector dimension D */
    long     dim_pad;             /**< Power-of-2 padded dimension D_pad */
    int      bits;                /**< Bits per dimension: 1 or 2 */
    uint64_t seed;                /**< Deterministic random seed */
    float   *sign_flips;          /**< [dim_pad] array of +1.0f / -1.0f random signs */
    size_t   code_bytes_per_vec;  /**< Number of packed code bytes per vector */
} RaBitQParams;

/**
 * @brief Precomputed query lookup tables (LUT) in L1 cache for FastScan ADC.
 */
typedef struct
{
    long     dim_pad;     /**< Padded dimension */
    int      num_nibbles; /**< Number of 4-dim blocks: dim_pad / 4 */
    float    q_norm;      /**< Euclidean norm of query ||q|| */
    float    scale;       /**< Multiplier mapping float partial sums to int8 */
    float    inv_scale;   /**< Reciprocal step mapping int8 sum back to float */
    float    base_offset; /**< Precomputed sum of bias terms across all nibbles */
    int8_t  *lut_i8;      /**< Transposed query LUT: [num_nibbles * 16] */
} RaBitQLookupTable;

/**
 * @brief Detected CPU SIMD mode for RaBitQ FastScan.
 */
typedef enum
{
    RABITQ_SIMD_SCALAR = 0,
    RABITQ_SIMD_AVX2   = 1,
    RABITQ_SIMD_AVX512 = 2
} RaBitQSimdMode;

/**
 * @brief Query the active SIMD register mode available on the host CPU.
 */
RaBitQSimdMode rabitq_get_simd_mode(void);

/**
 * @brief Human-readable string for the active SIMD register mode.
 */
const char *rabitq_get_simd_mode_str(void);

/**
 * @brief Compute the smallest power of 2 greater than or equal to dim.
 */
long rabitq_compute_pad_dim(
    long dim);

/**
 * @brief Initialize RaBitQ parameters and random sign-flip vector.
 */
int rabitq_init_params(
    RaBitQParams *params,
    long          dim,
    int           bits,
    uint64_t      seed);

/**
 * @brief Free resources associated with RaBitQ parameters.
 */
void rabitq_free_params(
    RaBitQParams *params);

/**
 * @brief Compute in-place Fast Walsh-Hadamard Transform with random sign flips.
 */
void rabitq_rotate_vector_float(
    const float        *restrict src,
    float              *restrict dst,
    const RaBitQParams *restrict params);

/**
 * @brief Compute in-place Fast Walsh-Hadamard Transform for double precision input.
 */
void rabitq_rotate_vector_double(
    const double       *restrict src,
    float              *restrict dst,
    const RaBitQParams *restrict params);

/**
 * @brief Quantize a rotated float vector into packed bit codes and metadata.
 */
void rabitq_quantize_rotated_float(
    const float        *restrict rotated,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params);

/**
 * @brief Full end-to-end quantization of a raw float vector.
 */
void rabitq_quantize_vector_float(
    const float        *restrict src,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params);

/**
 * @brief Full end-to-end quantization of a raw double vector.
 */
void rabitq_quantize_vector_double(
    const double       *restrict src,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params);

/**
 * @brief Build precomputed query lookup table (LUT) for FastScan ADC.
 */
int rabitq_build_query_lut(
    const float        *restrict rotated_query,
    float                        q_norm,
    const RaBitQParams *restrict params,
    RaBitQLookupTable  *restrict out_lut);

/**
 * @brief Free allocated query lookup table buffers.
 */
void rabitq_free_query_lut(
    RaBitQLookupTable *lut);

/**
 * @brief Compute guaranteed metric lower bound Euclidean distance between query and candidate.
 */
double rabitq_compute_lower_bound(
    const float             *restrict rotated_query,
    float                             q_norm,
    const uint8_t           *restrict cand_codes,
    const RaBitQMeta        *restrict cand_meta,
    const RaBitQParams      *restrict params,
    double                            epsilon);

/**
 * @brief FastScan evaluation of 32 candidates against query LUT with early pruning bitmask.
 */
uint32_t rabitq_fastscan_32x(
    const RaBitQLookupTable *restrict q_lut,
    const uint8_t           *restrict block_codes,
    const RaBitQMeta        *restrict block_meta,
    double                            tau_cutoff,
    double                            epsilon);

/**
 * @brief Save quantized dataset buffer, metadata, and parameters to a binary .rabitq file.
 */
int rabitq_save_sidecar(
    const char         *filepath,
    const RaBitQParams *params,
    const RaBitQMeta   *meta_array,
    const uint8_t      *codes_array,
    long                num_frames);

/**
 * @brief Load quantized dataset buffer, metadata, and parameters from a binary .rabitq file.
 */
int rabitq_load_sidecar(
    const char    *filepath,
    RaBitQParams  *params,
    RaBitQMeta   **out_meta,
    uint8_t      **out_codes,
    long          *out_num_frames);

#ifdef __cplusplus
}
#endif

#endif // RABIT_QUANT_H

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
 * @brief Quantize float frame with spectral dimension permutation.
 */
void sq16_quantize_float_perm(
    const float      *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Quantize a double-precision frame to 16-bit signed integers in [0, 32767].
 */
void sq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params);

/**
 * @brief Quantize double frame with spectral dimension permutation.
 */
void sq16_quantize_double_perm(
    const double     *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Compute sum of squared differences between two int16 vectors using SIMD.
 */
uint64_t sq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim);


/**
 * @brief Compute sum of squared differences between two int16 vectors with early cutoff.
 */
uint64_t sq16_dist_squared_cutoff_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim,
    uint64_t                ssd_cutoff);

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
 * @brief Set coordinates for a cluster into Block-8 interleaved format across all dimensions.
 */
void sq16_set_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    int                     cl_idx,
    const int16_t *restrict anchor_coords,
    long                    dim);

/**
 * @brief Rebuild Block-8 interleaved buffer from row-major anchor matrix.
 */
void sq16_rebuild_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim);

/**
 * @brief Bulk filter contiguous cluster anchors using SQ16 lower bounds and SIMD.
 */
void sq16_filter_anchor_matrix(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count);

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
 * sq16_fastscan_32x_3d() - Evaluate SQ16 squared distance for 32 3D candidates in SIMD.
 * @query_sq16: Pointer to query's 3 quantized coordinates.
 * @block_x:    Pointer to 32 transposed x-coordinates.
 * @block_y:    Pointer to 32 transposed y-coordinates.
 * @block_z:    Pointer to 32 transposed z-coordinates.
 * @ssd_cutoff: Squared distance threshold for candidate survival.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i satisfies dist^2 <= ssd_cutoff.
 */
uint32_t sq16_fastscan_32x_3d(
    const int16_t *restrict query_sq16,
    const int16_t *restrict block_x,
    const int16_t *restrict block_y,
    const int16_t *restrict block_z,
    uint64_t                ssd_cutoff);

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
    uint64_t                ssd_cutoff);

#endif // SCALAR_QUANT_H

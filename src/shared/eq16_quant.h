#ifndef EQ16_QUANT_H
#define EQ16_QUANT_H

/**
 * @file eq16_quant.h
 * @brief 16-bit E8 Block Lattice Quantization (EQ16) with metric lower-bounding.
 *
 * Replaces cubic scalar quantization (SQ16 on Z^D) with optimal 8-dimensional
 * E8 root lattice quantization (EQ16 on E8^(D/8)). Achieves a 29.29% reduction in
 * covering radius and 50%+ more candidate pruning while maintaining identical
 * 16-bit integer SIMD memory layout and hardware throughput.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif
#endif

/** Magic identifier for .eq16 sidecar files */
#define EQ16_FILE_MAGIC "EQ16_0001"

/** Maximum coordinate difference representable in int16 */
#define EQ16_MAX_DIFF_I16 32767ULL

/** Block size for SIMD transposed FastScan */
#define EQ16_FASTSCAN_BLOCK_SIZE 32

/**
 * @brief Parameters defining uniform 16-bit E8 block lattice quantization.
 */
typedef struct
{
    float min_val;    /**< Minimum dynamic range value */
    float max_val;    /**< Maximum dynamic range value */
    float center;     /**< Dynamic range center offset: 0.5f * (min_val + max_val) */
    float scale;      /**< Lattice fundamental step Delta */
    float inv_scale;  /**< Reciprocal step 1.0f / Delta */
    float err_radius; /**< Metric covering radius: sqrt(k + 0.25*rem) * scale */
    long  dim;        /**< Vector dimension */
} EQ16Params;

/**
 * @brief Initialize EQ16 parameters from known min and max values.
 */
void eq16_init_params(
    EQ16Params *params,
    float       min_val,
    float       max_val,
    long        dim);

/**
 * @brief Calibrate EQ16 parameters by scanning a float array.
 */
void eq16_calibrate_float(
    EQ16Params  *params,
    const float *data,
    long         num_elements,
    long         dim);

/**
 * @brief Calibrate EQ16 parameters by scanning a double array.
 */
void eq16_calibrate_double(
    EQ16Params   *params,
    const double *data,
    long          num_elements,
    long          dim);

/**
 * @brief Quantize a float frame onto E8 lattice into doubled int16 coordinates.
 */
void eq16_quantize_float(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params);

/**
 * @brief Quantize a double frame onto E8 lattice into doubled int16 coordinates.
 */
void eq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params);

/**
 * @brief Prepare float query vector for Asymmetric Distance Computation (ADC).
 *
 * Normalizes query coordinates: dst[d] = (src[d] - center) * (2.0f * inv_scale).
 */
void eq16_prepare_query_adc_float(
    const float      *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params);

/**
 * @brief Prepare double query vector for Asymmetric Distance Computation (ADC).
 */
void eq16_prepare_query_adc_double(
    const double     *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params);

/**
 * @brief Quantize a float frame onto E8 lattice with dimension permutation.
 */
void eq16_quantize_float_perm(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Quantize a double frame onto E8 lattice with dimension permutation.
 */
void eq16_quantize_double_perm(
    const double     *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Prepare float query vector for ADC with dimension permutation.
 */
void eq16_prepare_query_adc_float_perm(
    const float      *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Prepare double query vector for ADC with dimension permutation.
 */
void eq16_prepare_query_adc_double_perm(
    const double     *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim);

/**
 * @brief Compute sum of squared differences between two int16 EQ16 vectors using SIMD.
 */
uint64_t eq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim);

/**
 * @brief Compute squared difference with early cutoff between two int16 EQ16 vectors.
 */
uint64_t eq16_dist_squared_cutoff_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim,
    uint64_t                ssd_cutoff);

/**
 * @brief Compute asymmetric squared distance with early cutoff for float query and int16 EQ16.
 *
 * Computes sum((q_scaled[d] - cand_eq16[d])^2) with early exit when sum > cutoff.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
float eq16_dist_asym_cutoff_f32(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff);

/**
 * @brief Compute asymmetric squared distance with cutoff for 1 query against 4 candidates.
 *
 * Evaluates sum((q_scaled[d] - cand[d])^2) concurrently across 4 candidates.
 */
void eq16_dist_asym_cutoff_batch_1x4(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists);

/**
 * @brief Refine surviving cluster candidates using batched 1x4 ADC distance computation.
 */
void eq16_refine_candidates_adc(
    const float   *restrict cur_adc,
    const int16_t *restrict mat_eq16,
    int           *restrict active_clusters,
    int                     num_active,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count);

/**
 * @brief Compute guaranteed metric lower bound between two EQ16 quantized vectors.
 */
double eq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const EQ16Params *restrict params,
    double                     epsilon);

/**
 * @brief Compute sum of squared differences between 1 query and 4 int16 anchors using AVX2.
 */
void eq16_dist_squared_batch_1x4_i16(
    const int16_t *restrict        q,
    const int16_t *const *restrict anchors,
    uint64_t *restrict             out_sq_dists,
    long                           dim);

/**
 * @brief Bulk filter cluster candidates using EQ16 lower bounds.
 */
int eq16_batch_filter_candidates(
    const int16_t *restrict        q_eq16,
    const int16_t *const *restrict anchor_ptrs,
    const int                     *candidate_indices,
    int                            num_candidates,
    double                         cutoff_dist,
    const EQ16Params              *params,
    int *restrict                  clmembflag);

/**
 * @brief Set coordinates into Block-8 interleaved format across all dimensions.
 */
void eq16_set_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    int                     cl_idx,
    const int16_t *restrict anchor_coords,
    long                    dim);

/**
 * @brief Rebuild Block-8 interleaved buffer from contiguous anchor matrix.
 */
void eq16_rebuild_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim);

/**
 * @brief Bulk filter contiguous cluster anchors using EQ16 lower bounds and SIMD.
 */
void eq16_filter_anchor_matrix(
    const int16_t *restrict cur_eq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                eq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count);

/**
 * @brief Set coordinates into Block-16 interleaved float format across all dimensions.
 */
void eq16_set_anchor_adc_interleaved(
    float         *restrict matrix_adc_interleaved,
    int                     cl_idx,
    const int16_t *restrict anchor_coords,
    long                    dim);

/**
 * @brief Rebuild Block-16 interleaved float buffer from contiguous anchor matrix.
 */
void eq16_rebuild_anchor_adc_interleaved(
    float         *restrict matrix_adc_interleaved,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim);

/**
 * @brief Bulk filter cluster anchors using Asymmetric Distance Computation (ADC).
 */
void eq16_filter_anchor_matrix_adc(
    const float   *restrict cur_eq16_adc,
    const int16_t *restrict anchor_matrix,
    const float   *restrict anchor_adc_interleaved,
    int                     num_clusters,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count);

/**
 * @brief Save quantized dataset buffer and parameters to a binary .eq16 file.
 */
int eq16_save_sidecar(
    const char       *filepath,
    const EQ16Params *params,
    const int16_t    *data,
    long              num_frames);

/**
 * @brief Load quantized dataset buffer and parameters from a binary .eq16 file.
 */
int eq16_load_sidecar(
    const char  *filepath,
    EQ16Params  *params,
    int16_t    **data,
    long        *num_frames);

/**
 * @brief Print statistics for EQ16 SIMD early exit checkpoints.
 */
void eq16_print_checkpoint_stats(void);

/**
 * @brief Evaluate EQ16 ADC distance for 32 candidates in SIMD against float query.
 *
 * Evaluates sum((q_adc[d] - block_coords[d*32 + lane])^2) across 32 candidates.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i distance <= cutoff_f.
 */
uint32_t eq16_fastscan_32x_adc(
    const float   *restrict q_adc,
    const int16_t *restrict block_coords,
    long                    dim,
    float                   cutoff_f);

/**
 * @brief Evaluate EQ16 SDC squared distance for 32 candidates in SIMD against int16 query.
 *
 * Evaluates sum((query_eq16[d] - block_coords[d*32 + lane])^2) across 32 candidates.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i SSD <= ssd_cutoff.
 */
uint32_t eq16_fastscan_32x_i16(
    const int16_t *restrict query_eq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff);

#ifdef __cplusplus
}
#endif

#endif // EQ16_QUANT_H

/**
 * @file knn_cache_layout.h
 * @brief In-memory layout reorganizations for k-NN: FastScan transposed blocks & IVF.
 *
 * Declares functions that restructure flat quantized code buffers into SIMD-friendly
 * blocked formats. This includes 32-vector transposed blocks for AVX2/AVX-512 FastScan
 * kernels (SQ16, EQ16, RQ8, PQ, RaBitQ) and inverted-file (IVF) index layouts for fast
 * cluster-oriented candidate lookup.
 */

#ifndef KNN_CACHE_LAYOUT_H
#define KNN_CACHE_LAYOUT_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_model_build_transposed_sq16() - Build cluster-local transposed SQ16 FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_sq16(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_transposed_eq16() - Build cluster-local transposed EQ16 FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_eq16(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_transposed_rq8() - Build cluster-local transposed RQ8 FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_rq8(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_transposed_pq() - Build cluster-local transposed PQ FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_pq(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_transposed_rabitq() - Build cluster-local transposed RaBitQ FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_rabitq(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_ivf_layout() - Reorganize dataset into contiguous per-cluster IVF layout.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_ivf_layout(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_eq16_cluster_layout() - Reorganize EQ16 vectors into cluster-contiguous order.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_eq16_cluster_layout(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_rq8_cluster_layout() - Reorganize RQ8 vectors into cluster-contiguous order.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_rq8_cluster_layout(
    KnnModel        *model,
    const KnnConfig *config);

#ifdef __cplusplus
}
#endif

#endif // KNN_CACHE_LAYOUT_H

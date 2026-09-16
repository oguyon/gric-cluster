/**
 * @file knn_cache.h
 * @brief Memory caching and scalar quantization sidecar generation/loading for k-NN.
 */

#ifndef KNN_CACHE_H
#define KNN_CACHE_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_model_build_or_load_sq8() - Build or load quantized SQ8 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_sq8(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_sq16() - Build or load quantized SQ16 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_sq16(
    KnnModel        *model,
    const KnnConfig *config);

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
 * knn_model_build_or_load_rq8() - Build or load quantized RQ8 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_rq8(
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
 * knn_model_build_or_load_pq() - Build or load PQ codebook and codes into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_pq(
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
 * knn_model_cache_dataset() - Preload dataset frames into resident RAM buffer if feasible.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_cache_dataset(
    KnnModel  *model,
    KnnConfig *config);

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

#ifdef __cplusplus
}
#endif

#endif // KNN_CACHE_H

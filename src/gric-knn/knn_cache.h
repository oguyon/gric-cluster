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
 * knn_model_cache_dataset() - Preload dataset frames into resident RAM buffer if feasible.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_cache_dataset(
    KnnModel  *model,
    KnnConfig *config);

#ifdef __cplusplus
}
#endif

#endif // KNN_CACHE_H

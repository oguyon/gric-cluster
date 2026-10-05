/**
 * @file knn_cache.h
 * @brief Memory caching and quantization sidecar generation/loading for k-NN.
 *
 * Declares public APIs for building, caching, and loading quantized dataset buffers
 * and sidecar files across multiple quantization schemes (SQ8, SQ16, EQ16, RQ8, PQ,
 * and RaBitQ). Also provides memory preloading for raw dataset frames.
 */

#ifndef KNN_CACHE_H
#define KNN_CACHE_H

#include "knn_defs.h"
#include "knn_cache_layout.h"

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
 * knn_model_build_or_load_eq16() - Build or load quantized EQ16 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_eq16(
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
 * knn_model_build_or_load_rabitq() - Build or load RaBitQ bit codes into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_rabitq(
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

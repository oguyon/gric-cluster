#ifndef KNN_LOADER_H
#define KNN_LOADER_H

/**
 * @file knn_loader.h
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 *
 * Declares functions that parse, load, and initialize resident cluster models from
 * Pass 1 clustering outputs (gric-cluster). Handles cluster assignments, anchor coordinates,
 * bounding radii, and quantization codebooks, as well as model cleanup routines.
 */

#include "knn_defs.h"

/**
 * knn_model_load() - Load Pass 1 clustering artifacts and anchors into memory
 * @cluster_dir:     Path to clustering directory (e.g., `<name>.clusterdat/`).
 * @input_data_path: Path to original dataset input file.
 * @model:           Pointer to KnnModel to initialize and populate.
 * @use_double:      1 for double-precision coordinates, 0 for single-precision float.
 *
 * Loads cluster membership, anchor vectors, radii, and k-NN graph into KnnModel.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_load(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model,
    int         use_double);

/**
 * knn_model_build_or_load_sq8() - Build or load quantized SQ8 dataset buffer into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_sq8(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_eq16() - Build or load quantized EQ16 dataset buffer into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_eq16(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_sq16() - Build or load quantized SQ16 dataset buffer into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_sq16(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_rq8() - Build or load quantized RQ8 dataset buffer into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_rq8(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_pq() - Build or load quantized PQ codebook and codes into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_pq(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_build_or_load_rabitq() - Build or load quantized RaBitQ codes into KnnModel
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_build_or_load_rabitq(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * knn_model_cache_dataset() - Preload dataset frames into resident RAM buffer
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to active KnnConfig.
 *
 * Preloads raw coordinate vectors from disk or mmap into memory for fast lookup.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_model_cache_dataset(
    KnnModel  *model,
    KnnConfig *config);

/**
 * knn_model_free() - Free all resident buffers and structures within a KnnModel
 * @model: Pointer to KnnModel to free.
 *
 * Releases anchors, membership tables, cluster graphs, and quantization buffers.
 */
void knn_model_free(
    KnnModel *model);

#endif // KNN_LOADER_H

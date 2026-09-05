#ifndef KNN_LOADER_H
#define KNN_LOADER_H

/**
 * @file knn_loader.h
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#include "knn_defs.h"

/**
 * @brief Load Pass 1 clustering artifacts and anchors into memory.
 * @param cluster_dir     Path to clustering directory (e.g., `<name>.clusterdat/`).
 * @param input_data_path Path to original dataset input file.
 * @param model           Pointer to KnnModel to initialize and populate.
 * @return 0 on success, -1 on failure.
 */
int knn_model_load(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model,
    int         use_double);

/**
 * @brief Build or load quantized SQ8 dataset buffer into KnnModel.
 * @param model  Pointer to initialized KnnModel.
 * @param config Pointer to KnnConfig.
 * @return 0 on success, -1 on failure.
 */
int knn_model_build_or_load_sq8(
    KnnModel        *model,
    const KnnConfig *config);

/**
 * @brief Free all resident buffers and structures within a KnnModel.
 * @param model Pointer to KnnModel to free.
 */
void knn_model_free(
    KnnModel *model);

#endif // KNN_LOADER_H

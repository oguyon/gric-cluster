#ifndef KNN_LOADER_H
#define KNN_LOADER_H

/**
 * @file knn_loader.h
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#include "knn_defs.h"

int knn_model_load(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model);

void knn_model_free(
    KnnModel *model);

#endif // KNN_LOADER_H

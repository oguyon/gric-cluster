#ifndef KNN_TREE_H
#define KNN_TREE_H

/**
 * @file knn_tree.h
 * @brief Cluster proximity graph builder for gric-knn.
 */

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_build_cluster_graph() - Build proximity graph on cluster anchors from DCC matrix.
 * @model: Pointer to resident KnnModel with populated dcc_matrix.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_build_cluster_graph(
    KnnModel *model);

/**
 * knn_free_cluster_graph() - Free cluster proximity graph allocations.
 * @model: Pointer to resident KnnModel.
 */
void knn_free_cluster_graph(
    KnnModel *model);

#ifdef __cplusplus
}
#endif

#endif // KNN_TREE_H

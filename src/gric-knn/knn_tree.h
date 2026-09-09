#ifndef KNN_TREE_H
#define KNN_TREE_H

/**
 * @file knn_tree.h
 * @brief Super-Cluster (Meta-Cluster) hierarchy builder for gric-knn.
 */

#include "knn_defs.h"

/**
 * knn_build_super_clusters() - Group M clusters into K super-clusters.
 * @model: Pointer to KnnModel with populated clusters, radii, and dcc_matrix.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_build_super_clusters(
    KnnModel *model);

/**
 * knn_free_super_clusters() - Free memory allocated for super-cluster structures.
 * @model: Pointer to KnnModel.
 */
void knn_free_super_clusters(
    KnnModel *model);

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

#endif // KNN_TREE_H

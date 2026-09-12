/**
 * @file knn_tree.c
 * @brief Cluster proximity graph builder for gric-knn.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_tree.h"
#include <stdlib.h>

/** Helper record for sorting cluster neighbors by DCC distance */
typedef struct
{
    int    cluster_id;
    double dist;
} DccNeighborPair;

/**
 * compare_dcc_pairs() - Compare two DccNeighborPair structs by ascending distance.
 * @a: Pointer to first pair.
 * @b: Pointer to second pair.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_dcc_pairs(
    const void *a,
    const void *b)
{
    const DccNeighborPair *pa = (const DccNeighborPair *)a;
    const DccNeighborPair *pb = (const DccNeighborPair *)b;
    if (pa->dist < pb->dist)
    {
        return -1;
    }
    if (pa->dist > pb->dist)
    {
        return 1;
    }
    return 0;
}

/**
 * knn_build_cluster_graph() - Build proximity graph on cluster anchors from DCC matrix.
 * @model: Pointer to resident KnnModel with populated dcc_matrix.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_build_cluster_graph(
    KnnModel *model)
{
    if (model == NULL || model->dcc_matrix == NULL || model->num_clusters <= 1)
    {
        return 0;
    }

    int M = model->num_clusters;
    int k_adj = 48;
    if (k_adj >= M)
    {
        k_adj = M - 1;
    }

    model->cluster_graph_k = k_adj;
    model->cluster_graph_adj = (int *)malloc((size_t)M * (size_t)k_adj * sizeof(int));
    if (model->cluster_graph_adj == NULL)
    {
        return -1;
    }

    DccNeighborPair *pairs = (DccNeighborPair *)malloc((size_t)M * sizeof(DccNeighborPair));
    if (pairs == NULL)
    {
        free(model->cluster_graph_adj);
        model->cluster_graph_adj = NULL;
        return -1;
    }

    for (int c = 0; c < M; c++)
    {
        int count = 0;
        for (int other = 0; other < M; other++)
        {
            if (other == c)
            {
                continue;
            }
            pairs[count].cluster_id = other;
            pairs[count].dist = model->dcc_matrix[(size_t)c * (size_t)M + (size_t)other];
            count++;
        }

        qsort(pairs, (size_t)count, sizeof(DccNeighborPair), compare_dcc_pairs);

        int *row_adj = model->cluster_graph_adj + (size_t)c * (size_t)k_adj;
        for (int i = 0; i < k_adj; i++)
        {
            row_adj[i] = pairs[i].cluster_id;
        }
    } // for (int c = 0; c < M; c++)

    free(pairs);
    return 0;
}

/**
 * knn_free_cluster_graph() - Free cluster proximity graph allocations.
 * @model: Pointer to resident KnnModel.
 */
void knn_free_cluster_graph(
    KnnModel *model)
{
    if (model == NULL)
    {
        return;
    }

    if (model->cluster_graph_adj != NULL)
    {
        free(model->cluster_graph_adj);
        model->cluster_graph_adj = NULL;
    }
    model->cluster_graph_k = 0;
}


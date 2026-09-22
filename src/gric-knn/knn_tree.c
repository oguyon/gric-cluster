/**
 * @file knn_tree.c
 * @brief Cluster proximity graph builder for gric-knn.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_tree.h"
#include "e8_lattice.h"

#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Helper record for sorting cluster neighbors by DCC distance */
typedef struct
{
    int    cluster_id;
    double dist;
} DccNeighborPair;

/**
 * max_heap_sift_down() - Sift down root element in a max-heap of DccNeighborPair.
 * @heap: Pointer to array of pairs.
 * @idx:  Starting index to sift down from.
 * @n:    Number of elements in the heap.
 */
static inline void max_heap_sift_down(
    DccNeighborPair *heap,
    int              idx,
    int              n)
{
    DccNeighborPair val = heap[idx];
    while (1)
    {
        int left = 2 * idx + 1;
        if (left >= n)
        {
            break;
        }
        int right = left + 1;
        int largest = (right < n && heap[right].dist > heap[left].dist) ? right : left;
        if (heap[largest].dist <= val.dist)
        {
            break;
        }
        heap[idx] = heap[largest];
        idx = largest;
    }
    heap[idx] = val;
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
    int k_adj = (model->use_e8_graph) ? E8_NUM_ROOTS : 48;
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

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        DccNeighborPair stack_heap[256];
        DccNeighborPair *heap = (k_adj <= 256)
            ? stack_heap
            : (DccNeighborPair *)malloc((size_t)k_adj * sizeof(DccNeighborPair));

        if (heap != NULL)
        {
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (int c = 0; c < M; c++)
            {
                int count = 0;
                for (int other = 0; other < M; other++)
                {
                    if (other == c)
                    {
                        continue;
                    }
                    double d = model->dcc_matrix[(size_t)c * (size_t)M + (size_t)other];
                    if (count < k_adj)
                    {
                        heap[count].cluster_id = other;
                        heap[count].dist = d;
                        count++;
                        if (count == k_adj)
                        {
                            for (int i = (k_adj - 2) / 2; i >= 0; i--)
                            {
                                max_heap_sift_down(heap, i, k_adj);
                            }
                        }
                    }
                    else if (d < heap[0].dist)
                    {
                        heap[0].cluster_id = other;
                        heap[0].dist = d;
                        max_heap_sift_down(heap, 0, k_adj);
                    }
                } // for (int other = 0; other < M; other++)

                /* Sort heap ascending via heapsort */
                for (int i = count - 1; i > 0; i--)
                {
                    DccNeighborPair tmp = heap[0];
                    heap[0] = heap[i];
                    heap[i] = tmp;
                    max_heap_sift_down(heap, 0, i);
                }

                int *row_adj = model->cluster_graph_adj + (size_t)c * (size_t)k_adj;
                for (int i = 0; i < count; i++)
                {
                    row_adj[i] = heap[i].cluster_id;
                }
            } // for (int c = 0; c < M; c++)

            if (heap != stack_heap)
            {
                free(heap);
            }
        }
    } // OpenMP parallel

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

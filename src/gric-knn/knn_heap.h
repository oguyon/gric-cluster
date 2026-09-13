#ifndef KNN_HEAP_H
#define KNN_HEAP_H

/**
 * @file knn_heap.h
 * @brief Bounded binary max-heap for k-nearest neighbors tracking.
 */

#include "knn_defs.h"

/**
 * @brief Allocate and initialize a bounded max-heap with capacity k.
 * @param heap Pointer to the KnnMaxHeap structure.
 * @param k    Capacity of the heap (number of nearest neighbors).
 * @return 0 on success, -1 on allocation failure.
 */
int knn_heap_init(
    KnnMaxHeap *heap,
    int         k);

/**
 * @brief Free heap resources and reset state.
 * @param heap Pointer to the KnnMaxHeap structure.
 */
void knn_heap_free(
    KnnMaxHeap *heap);

/**
 * @brief Reset element count to 0 for reuse in the next query.
 * @param heap Pointer to the KnnMaxHeap structure.
 */
void knn_heap_reset(
    KnnMaxHeap *heap);

/**
 * @brief Peek at the maximum distance currently stored at root of the heap.
 * @param heap Pointer to the KnnMaxHeap structure.
 * @return Maximum distance in heap if full, or 1e30 if not full or empty.
 */
static inline double knn_heap_peek_max_dist(
    const KnnMaxHeap *heap)
{
    if (heap == NULL || heap->count < heap->k)
    {
        return 1e30;
    }

    if (heap->capacity > 0)
    {
        return (double)heap->tau;
    }

    return heap->data[0].dist;
}

/**
 * @brief Get candidate frame index at rank idx.
 * @param heap Pointer to the KnnMaxHeap structure.
 * @param idx  Neighbor rank index.
 * @return Candidate frame index, or -1 on error.
 */
static inline int knn_heap_get_id(
    const KnnMaxHeap *heap,
    int               idx)
{
    if (heap == NULL || idx < 0 || idx >= heap->count)
    {
        return -1;
    }

    if (heap->capacity > 0)
    {
        return heap->simd_id[idx];
    }

    return heap->data[idx].frame_id;
}

/**
 * @brief Get candidate distance at rank idx.
 * @param heap Pointer to the KnnMaxHeap structure.
 * @param idx  Neighbor rank index.
 * @return Candidate distance, or 1e30 on error.
 */
static inline double knn_heap_get_dist(
    const KnnMaxHeap *heap,
    int               idx)
{
    if (heap == NULL || idx < 0 || idx >= heap->count)
    {
        return 1e30;
    }

    if (heap->capacity > 0)
    {
        return (double)heap->simd_dist[idx];
    }

    return heap->data[idx].dist;
}

/**
 * @brief Test whether a specific frame_id is already present in the heap.
 * @param heap     Pointer to the KnnMaxHeap structure.
 * @param frame_id Candidate frame index.
 * @return 1 if found, 0 otherwise.
 */
int knn_heap_contains(
    const KnnMaxHeap *heap,
    int               frame_id);

/**
 * @brief Push a new neighbor into the bounded max-heap.
 * @param heap     Pointer to the KnnMaxHeap structure.
 * @param frame_id Candidate frame index.
 * @param dist     Distance from query frame to candidate.
 */
void knn_heap_push(
    KnnMaxHeap *heap,
    int         frame_id,
    double      dist);

/**
 * @brief Extract heap elements into arrays sorted in ascending order by distance.
 * @param heap          Pointer to the KnnMaxHeap structure.
 * @param out_indices   Output array for sorted frame indices.
 * @param out_distances Output array for sorted distances.
 * @param k             Number of neighbors requested.
 */
void knn_heap_extract_sorted(
    KnnMaxHeap *heap,
    int        *out_indices,
    double     *out_distances,
    int         k);

#endif // KNN_HEAP_H

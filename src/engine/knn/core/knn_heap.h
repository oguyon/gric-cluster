#ifndef KNN_HEAP_H
#define KNN_HEAP_H

/**
 * @file knn_heap.h
 * @brief Bounded binary max-heap for k-nearest neighbors tracking.
 *
 * Declares data structures and operations for maintaining a thread-local top-k candidate
 * collection. Supports SIMD fixed-size sorted registers for small k (k <= 64) and dynamic
 * binary max-heaps for larger k, providing fast threshold inspection, insertion, and sorting.
 */

#include "knn_defs.h"

/**
 * knn_heap_init() - Allocate and initialize a bounded max-heap with capacity k
 * @heap: Pointer to KnnMaxHeap structure to initialize.
 * @k:    Capacity of the heap (number of nearest neighbors requested).
 *
 * Allocates internal heap storage or configures SIMD registers for top-k tracking.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
int knn_heap_init(
    KnnMaxHeap *heap,
    int         k);

/**
 * knn_heap_free() - Free heap resources and reset state
 * @heap: Pointer to KnnMaxHeap structure.
 *
 * Deallocates dynamic storage and resets heap counters to zero.
 */
void knn_heap_free(
    KnnMaxHeap *heap);

/**
 * knn_heap_reset() - Reset element count to 0 for reuse in the next query
 * @heap: Pointer to KnnMaxHeap structure.
 *
 * Clears count and resets threshold distance tau to infinity without freeing memory.
 */
void knn_heap_reset(
    KnnMaxHeap *heap);

/**
 * knn_heap_peek_max_dist() - Peek at the maximum distance currently stored in the heap
 * @heap: Pointer to KnnMaxHeap structure.
 *
 * Return: Maximum distance in heap if full, or 1e30 if not full or empty.
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
 * knn_heap_get_id() - Get candidate frame index at rank idx
 * @heap: Pointer to KnnMaxHeap structure.
 * @idx:  Neighbor rank index [0..count-1].
 *
 * Return: Candidate frame index, or -1 on error.
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
 * knn_heap_get_dist() - Get candidate distance at rank idx
 * @heap: Pointer to KnnMaxHeap structure.
 * @idx:  Neighbor rank index [0..count-1].
 *
 * Return: Candidate distance, or 1e30 on error.
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
 * knn_heap_contains() - Test whether a specific frame_id is already present in the heap
 * @heap:     Pointer to KnnMaxHeap structure.
 * @frame_id: Candidate frame index.
 *
 * Return: 1 if found in heap, 0 otherwise.
 */
int knn_heap_contains(
    const KnnMaxHeap *heap,
    int               frame_id);

/**
 * knn_heap_push() - Push a new neighbor into the bounded max-heap
 * @heap:     Pointer to KnnMaxHeap structure.
 * @frame_id: Candidate frame index.
 * @dist:     Distance from query frame to candidate.
 *
 * Inserts candidate if distance is strictly smaller than the kth-neighbor distance tau.
 */
void knn_heap_push(
    KnnMaxHeap *heap,
    int         frame_id,
    double      dist);

/**
 * knn_heap_extract_sorted() - Extract heap elements into arrays sorted in ascending order
 * @heap:          Pointer to KnnMaxHeap structure.
 * @out_indices:   Output array for sorted frame indices.
 * @out_distances: Output array for sorted distances.
 * @k:             Number of neighbors requested.
 *
 * Sorts and copies elements in ascending order of distance into output arrays.
 */
void knn_heap_extract_sorted(
    KnnMaxHeap *heap,
    int        *out_indices,
    double     *out_distances,
    int         k);

#endif // KNN_HEAP_H

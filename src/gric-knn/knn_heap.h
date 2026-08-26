#ifndef KNN_HEAP_H
#define KNN_HEAP_H

/**
 * @file knn_heap.h
 * @brief Bounded binary max-heap for k-nearest neighbors tracking.
 */

#include "knn_defs.h"

int knn_heap_init(
    KnnMaxHeap *heap,
    int         k);

void knn_heap_free(
    KnnMaxHeap *heap);

void knn_heap_reset(
    KnnMaxHeap *heap);

double knn_heap_peek_max_dist(
    const KnnMaxHeap *heap);

int knn_heap_contains(
    const KnnMaxHeap *heap,
    int               frame_id);

void knn_heap_push(
    KnnMaxHeap *heap,
    int         frame_id,
    double      dist);

void knn_heap_extract_sorted(
    KnnMaxHeap *heap,
    int        *out_indices,
    double     *out_distances,
    int         k);

#endif // KNN_HEAP_H

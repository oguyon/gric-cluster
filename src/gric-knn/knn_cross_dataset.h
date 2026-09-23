/**
 * @file knn_cross_dataset.h
 * @brief Cross-dataset k-NN search via basin expansion and graph routing.
 *
 * Declares the entry-point solver knn_search_cross_dataset_frame() for finding the
 * nearest neighbors of query frames (dataset C) within a reference dataset (dataset A).
 * Combines graph routing, cluster basin expansion, and member candidate evaluation.
 */

#ifndef KNN_CROSS_DATASET_H
#define KNN_CROSS_DATASET_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_search_cross_dataset_frame() - Search single query frame against target dataset model.
 * @query_id:      Frame index of query.
 * @query_data:    Raw pixel buffer of query frame.
 * @model:         Pointer to target KnnModel.
 * @config:        Pointer to active KnnConfig.
 * @cand_reader:   Reader context for candidate frames.
 * @cand_buffer:   Thread-local buffer for reading candidate frames.
 * @anchor_dists:  Thread-local buffer for query-to-anchor distances.
 * @scores_buffer: Thread-local buffer for cluster scores.
 * @heap:          Max-heap for current query.
 * @tracker:       Thread-local trajectory tracker.
 * @visited:       Frame visited tracker.
 * @telem:         Thread-local telemetry record.
 */
void knn_search_cross_dataset_frame(
    long                  query_id,
    const void *restrict  query_data,
    const KnnModel       *model,
    const KnnConfig      *config,
    KnnFrameReader       *cand_reader,
    void       *restrict  cand_buffer,
    double     *restrict  anchor_dists,
    ClusterScore *restrict scores_buffer,
    KnnMaxHeap           *heap,
    KnnTrajectoryTracker *tracker,
    KnnVisitedTracker    *visited,
    KnnTelemetry *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_CROSS_DATASET_H

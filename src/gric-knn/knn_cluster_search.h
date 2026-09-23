/**
 * @file knn_cluster_search.h
 * @brief Intra-dataset cluster candidate scoring and graph routing.
 *
 * Declares high-level search routines for single-dataset queries, including the primary
 * per-frame solver knn_search_single_frame(). Interfaces between cluster-level pruning
 * algorithms and frame-level member evaluation kernels.
 */

#ifndef KNN_CLUSTER_SEARCH_H
#define KNN_CLUSTER_SEARCH_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_search_single_frame() - Execute single query frame search against model clusters.
 * @query_id:      Frame index of query.
 * @query_data:    Raw pixel buffer of query frame.
 * @model:         Pointer to populated KnnModel.
 * @config:        Pointer to KnnConfig options.
 * @reader:        Reader context for streaming candidates.
 * @cand_buffer:   Thread-local buffer for reading candidate frames.
 * @scores_buffer: Thread-local buffer for sorting cluster candidate scores.
 * @graph_scratch: Thread-local scratch space for cluster graph routing.
 * @all_heaps:     Global array of per-query KnnMaxHeap heaps.
 * @bucket_locks:  Bucket locks array for thread synchronization (or NULL).
 * @visited:       Frame visited tracker.
 * @telem:         Thread-local telemetry record.
 */
void knn_search_single_frame(
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnClusterGraphScratch *graph_scratch,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnTelemetry  *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_CLUSTER_SEARCH_H

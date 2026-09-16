/**
 * @file knn_member_search.h
 * @brief Intra-cluster and candidate-cluster member evaluation, fastscan, and batching.
 */

#ifndef KNN_MEMBER_SEARCH_H
#define KNN_MEMBER_SEARCH_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"
#include "knn_reader.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * struct KnnCandidateBatch - Accumulator for batched distance evaluations.
 * @cand_ids: Array of up to 8 candidate frame indices.
 * @ptrs:     Array of up to 8 pointers to candidate pixel buffers.
 * @count:    Number of valid entries currently accumulated in the batch [0..8].
 */
typedef struct
{
    long        cand_ids[8];
    const void *ptrs[8];
    int         count;
} KnnCandidateBatch;

/**
 * knn_batch_init() - Zero-initialize a candidate evaluation batch.
 * @batch: Pointer to KnnCandidateBatch struct.
 */
void knn_batch_init(
    KnnCandidateBatch *batch);

/**
 * knn_batch_flush() - Compute distances for accumulated candidates and update heaps.
 * @batch:        Pointer to active KnnCandidateBatch.
 * @query_id:     Frame index of current query.
 * @query_data:   Raw pixel vector for query frame.
 * @model:        Active KnnModel.
 * @config:       Active KnnConfig.
 * @heap:         Per-query max-heap.
 * @all_heaps:    Global array of all frame heaps (for mutual updates).
 * @bucket_locks: Bucket locks array for thread synchronization (or NULL).
 * @telem:        Thread-local telemetry record.
 */
void knn_batch_flush(
    KnnCandidateBatch      *batch,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnTelemetry  *restrict telem);

/**
 * knn_eval_cluster_members() - Search all members of a cluster with metric and SIMD pruning.
 * @c:                 Index of target cluster to evaluate.
 * @d_anchor:          Computed distance from query frame to target cluster anchor.
 * @home_cluster_id:   Query's home cluster index (or -1 if unassigned).
 * @r_home:            Distance to query's home cluster anchor.
 * @anchor_is_sq16:    1 if d_anchor was computed via SQ16, 0 otherwise.
 * @query_id:          Index of query frame.
 * @query_data:        Raw pixel buffer for query frame.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            KnnFrameReader context for reading on-disk frames.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Global array of all frame heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @pivots:            Array of measured anchor pivots (or NULL).
 * @num_pivots:        Number of active anchor pivots.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_cluster_members(
    int                     c,
    double                  d_anchor,
    int                     home_cluster_id,
    double                  r_home,
    int                     anchor_is_sq16,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    const MeasuredPivot    *pivots,
    int                     num_pivots,
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_MEMBER_SEARCH_H

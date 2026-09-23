/**
 * @file knn_cross_eval.h
 * @brief Declarations for cross-dataset intra-cluster and inter-cluster candidate evaluation.
 *
 * Declares functions that evaluate individual frames belonging to candidate clusters
 * identified during cross-dataset routing. Includes routines for intra-cluster member
 * scoring and multi-cluster member evaluation with metric lower-bound pruning.
 */

#ifndef KNN_CROSS_EVAL_H
#define KNN_CROSS_EVAL_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"
#include "cluster_locator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_cross_eval_intra_cluster() - Evaluate frames inside best cluster and prune by radius.
 * @best_c:           Cluster ID containing best seed.
 * @best_seed_id:     Frame ID of best seed.
 * @min_d_anchor:     Distance to cluster anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Array of evaluated pivot frame IDs.
 * @seed_pivot_dists: Array of distances from query to seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Target KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
void knn_cross_eval_intra_cluster(
    int                    best_c,
    long                   best_seed_id,
    double                 min_d_anchor,
    int                    num_seed_pivots,
    const long            *seed_pivot_ids,
    const double          *seed_pivot_dists,
    const void *restrict   query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    void *restrict         cand_buffer,
    double *restrict       anchor_dists,
    KnnMaxHeap            *heap,
    KnnVisitedTracker     *visited,
    KnnTelemetry *restrict telem);

/**
 * knn_cross_eval_inter_clusters() - Evaluate surviving clusters in best-first priority order.
 * @loc_res:          Cluster locator result.
 * @best_c:           ID of best cluster already evaluated.
 * @min_d_anchor:     Distance to best cluster anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Array of evaluated pivot frame IDs.
 * @seed_pivot_dists: Array of distances from query to seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Target KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @scores_buffer:    Scratch buffer for candidate cluster sorting.
 * @active_mask:      Mask of surviving clusters.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
void knn_cross_eval_inter_clusters(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    double                      min_d_anchor,
    int                         num_seed_pivots,
    const long                 *seed_pivot_ids,
    const double               *seed_pivot_dists,
    const void *restrict        query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    void *restrict              cand_buffer,
    double *restrict            anchor_dists,
    ClusterScore *restrict      scores_buffer,
    uint8_t                    *active_mask,
    KnnMaxHeap                 *heap,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_CROSS_EVAL_H

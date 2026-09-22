/**
 * @file knn_cross_route.h
 * @brief Declarations for cross-dataset graph frontier routing and basin expansion.
 */

#ifndef KNN_CROSS_ROUTE_H
#define KNN_CROSS_ROUTE_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"
#include "cluster_locator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_cross_seed_frontier() - Populate search frontier with nearest evaluated anchor seeds.
 * @loc_res:        Cluster locator result with evaluated anchors.
 * @best_c:         Best anchor cluster ID.
 * @query_data:     Query frame pixel data.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @cand_reader:    Candidate frame reader.
 * @cand_buffer:    Candidate frame pixel buffer.
 * @heap:           Max-heap for current query.
 * @frontier:       Output frontier node array.
 * @frontier_count: Pointer to frontier count.
 * @best_seed_id:   Pointer to best seed frame ID.
 * @best_seed_dist: Pointer to best seed distance.
 * @visited:        Per-query frame visited tracker.
 * @telem:          Telemetry record.
 */
void knn_cross_seed_frontier(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    const void *restrict        query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    void *restrict              cand_buffer,
    KnnMaxHeap                 *heap,
    FrontierNode               *frontier,
    int                        *frontier_count,
    long                       *best_seed_id,
    double                     *best_seed_dist,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem);

/**
 * knn_greedy_route_to_basin() - Route greedily along proximity graph towards query basin.
 * @query_data:       Query frame pixel buffer.
 * @model:            Target dataset KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Reader context for candidate frames.
 * @cand_buffer:      Buffer for reading candidate frames.
 * @heap:             Max-heap tracking nearest neighbors.
 * @best_seed_id:     In/out pointer to current best frame ID.
 * @best_seed_dist:   In/out pointer to distance of best frame.
 * @last_parent_id:   In/out pointer to parent frame ID.
 * @last_parent_dist: In/out pointer to parent distance.
 * @seed_pivot_ids:   Array storing evaluated pivot frame IDs.
 * @seed_pivot_dists: Array of computed distances to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
void knn_greedy_route_to_basin(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                *last_parent_id,
    double              *last_parent_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem);

/**
 * knn_direct_basin_expansion() - Direct 1-hop and 2-hop graph expansion around seed.
 * @query_data:       Query frame pixel buffer.
 * @model:            Target dataset KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Reader context for candidate frames.
 * @cand_buffer:      Buffer for reading candidate frames.
 * @heap:             Max-heap tracking nearest neighbors.
 * @best_seed_id:     In/out pointer to current best frame ID.
 * @best_seed_dist:   In/out pointer to distance of best frame.
 * @parent_id:        Parent frame ID from greedy routing.
 * @parent_dist:      Parent distance from query.
 * @seed_pivot_ids:   Array storing evaluated pivot frame IDs.
 * @seed_pivot_dists: Array of computed distances to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
void knn_direct_basin_expansion(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                 parent_id,
    double               parent_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem);

/**
 * knn_cross_explore_graph_frontier() - Best-first multi-seed graph frontier exploration.
 * @query_data:       Query frame pixel data.
 * @model:            Target KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame buffer.
 * @heap:             Max-heap for current query.
 * @frontier:         Frontier node array.
 * @frontier_count:   Number of active frontier nodes.
 * @best_seed_id:     In/out pointer to best frame ID.
 * @best_seed_dist:   In/out pointer to best frame distance.
 * @seed_pivot_ids:   Array of seed pivot IDs.
 * @seed_pivot_dists: Array of distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Return: 1 if global containment criterion was satisfied, 0 otherwise.
 */
int knn_cross_explore_graph_frontier(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    FrontierNode        *frontier,
    int                  frontier_count,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_CROSS_ROUTE_H

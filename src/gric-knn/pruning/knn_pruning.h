/**
 * @file knn_pruning.h
 * @brief Metric lower bounding and distance pruning for k-NN search.
 *
 * Declares inline mathematical bounds, multi-pivot metric pruning rules, and early-cutoff
 * distance computation primitives. Includes triangular inequality lower and upper bound
 * evaluators, angular directional pruning, and quantized distance filters (SQ8, SQ16, EQ16).
 */

#ifndef KNN_PRUNING_H
#define KNN_PRUNING_H

#include "knn_engine_internal.h"
#include "framedistance.h"
#include "residual_quant.h"
#include "scalar_quant.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Inline Fast-Path Helpers & Binary Search Routines
 * ------------------------------------------------------------------------- */

/**
 * is_in_pivots() - Check if a cluster ID is already in measured pivots.
 * @cluster_id: Cluster index to check.
 * @pivots:     Array of measured pivots.
 * @num_pivots: Number of measured pivots.
 *
 * Return: 1 if cluster is in pivots, 0 otherwise.
 */
static inline int is_in_pivots(
    int                  cluster_id,
    const MeasuredPivot *pivots,
    int                  num_pivots)
{
    if (pivots == NULL)
    {
        return 0;
    }
    for (int p = 0; p < num_pivots; p++)
    {
        if (pivots[p].cluster_id == cluster_id)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * find_member_lower_bound() - Binary search for first member with r_anchor >= val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element >= val.
 */
static inline int find_member_lower_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor < val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * find_member_upper_bound() - Binary search for first member with r_anchor > val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element > val.
 */
static inline int find_member_upper_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor <= val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * compute_euclidean_distance() - Vectorized Euclidean distance between frames.
 * @da:        Pointer to first pixel array.
 * @db:        Pointer to second pixel array.
 * @size:      Number of elements in frame.
 * @is_double: 1 for double precision, 0 for float.
 *
 * Return: Euclidean L2 distance.
 */
static inline double compute_euclidean_distance(
    const void *restrict da,
    const void *restrict db,
    long                 size,
    int                  is_double)
{
    if (is_double)
    {
        return framedist_double((const double *)da, (const double *)db, size);
    }
    return framedist_float((const float *)da, (const float *)db, size);
}

/**
 * compute_euclidean_distance_cutoff() - Euclidean distance with early cutoff.
 * @da:        Pointer to first pixel array.
 * @db:        Pointer to second pixel array.
 * @size:      Number of elements in frame.
 * @is_double: 1 for double precision, 0 for float.
 * @cutoff_sq: Squared distance cutoff.
 *
 * Return: Euclidean L2 distance.
 */
static inline double compute_euclidean_distance_cutoff(
    const void *restrict da,
    const void *restrict db,
    long                 size,
    int                  is_double,
    double               cutoff_sq)
{
    if (is_double)
    {
        return framedist_squared_cutoff_double(
            (const double *)da, (const double *)db, size, cutoff_sq
        );
    }
    return framedist_squared_cutoff_float(
        (const float *)da, (const float *)db, size, cutoff_sq
    );
}

/**
 * is_member_pruned_by_sq16_cached() - Evaluate SQ16 using precomputed SSD cutoff.
 * @query_sq16:  Pointer to quantized query vector [dim].
 * @cand_id:     Index of candidate dataset frame.
 * @ssd_cutoff:  Precomputed SSD cutoff threshold.
 * @model:       Active KnnModel.
 * @telem:       Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_sq16_cached(
    const int16_t  *query_sq16,
    long            cand_id,
    uint64_t        ssd_cutoff,
    const KnnModel *model,
    KnnTelemetry   *telem)
{
    if (ssd_cutoff == UINT64_MAX)
    {
        return 0;
    }

    const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                               (size_t)cand_id * (size_t)model->frame_elements;
    telem->sq16_evaluations++;
    uint64_t ssd = sq16_dist_squared_cutoff_i16(
        query_sq16, cand_sq16, model->frame_elements, ssd_cutoff
    );

    if (ssd > ssd_cutoff)
    {
        telem->sq16_members_pruned++;
        return 1;
    }

    return 0;
}

/**
 * is_member_pruned_by_eq16_cached() - Evaluate EQ16 using precomputed SSD cutoff.
 * @query_eq16:  Pointer to quantized query vector [dim].
 * @cand_id:     Index of candidate dataset frame.
 * @ssd_cutoff:  Precomputed SSD cutoff threshold.
 * @model:       Active KnnModel.
 * @telem:       Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_eq16_cached(
    const int16_t  *query_eq16,
    const float    *query_eq16_adc,
    long            cand_id,
    uint64_t        ssd_cutoff,
    const KnnModel *model,
    KnnTelemetry   *telem)
{
    if (ssd_cutoff == UINT64_MAX || model->eq16_dataset_buffer == NULL)
    {
        return 0;
    }

    size_t pos = (size_t)cand_id;
    if (model->frame_to_cluster_pos != NULL)
    {
        pos = (size_t)model->frame_to_cluster_pos[cand_id];
    }

    if (query_eq16_adc != NULL)
    {
        const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                                   pos * (size_t)model->frame_elements;
        telem->eq16_evaluations++;
        float cutoff_f = (float)ssd_cutoff;
        float dist_sq = eq16_dist_asym_cutoff_f32(
            query_eq16_adc, cand_eq16, model->frame_elements, cutoff_f
        );

        if (dist_sq > cutoff_f)
        {
            telem->eq16_members_pruned++;
            return 1;
        }

        return 0;
    }

    if (query_eq16 == NULL)
    {
        return 0;
    }

    const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                               pos * (size_t)model->frame_elements;
    telem->eq16_evaluations++;
    uint64_t ssd = eq16_dist_squared_cutoff_i16(
        query_eq16, cand_eq16, model->frame_elements, ssd_cutoff
    );

    if (ssd > ssd_cutoff)
    {
        telem->eq16_members_pruned++;
        return 1;
    }

    return 0;
}

/**
 * is_member_pruned_by_rq8_cached() - Evaluate RQ8 using precomputed SSD cutoff.
 * @query_rq8:  Pointer to quantized query residual [dim].
 * @cand_id:    Index of candidate dataset frame.
 * @ssd_cutoff: Precomputed SSD cutoff threshold.
 * @model:      Active KnnModel.
 * @telem:      Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_rq8_cached(
    const int16_t  *query_rq8,
    long            cand_id,
    uint64_t        ssd_cutoff,
    const KnnModel *model,
    KnnTelemetry   *telem)
{
    if (ssd_cutoff == UINT64_MAX || query_rq8 == NULL || model->rq8_dataset_buffer == NULL)
    {
        return 0;
    }

    size_t pos = (size_t)cand_id;
    if (model->frame_to_cluster_pos != NULL)
    {
        pos = (size_t)model->frame_to_cluster_pos[cand_id];
    }

    const int8_t *cand_res = model->rq8_dataset_buffer +
                             pos * (size_t)model->frame_elements;
    telem->rq8_evaluations++;
    uint64_t ssd = rq8_dist_squared_cutoff_i8(
        query_rq8, cand_res, model->frame_elements, ssd_cutoff
    );

    if (ssd > ssd_cutoff)
    {
        telem->rq8_members_pruned++;
        return 1;
    }

    return 0;
}

/**
 * is_member_pruned_by_rq8_adc_cached() - Evaluate RQ8 ADC using precomputed float cutoff.
 * @query_rq8_adc: Pointer to normalized query residual [dim].
 * @cand_id:       Index of candidate dataset frame.
 * @cutoff_sq:     Precomputed squared float cutoff threshold.
 * @model:         Active KnnModel.
 * @telem:         Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_rq8_adc_cached(
    const float    *query_rq8_adc,
    long            cand_id,
    float           cutoff_sq,
    const KnnModel *model,
    KnnTelemetry   *telem)
{
    if (cutoff_sq >= 1e29f || query_rq8_adc == NULL || model->rq8_dataset_buffer == NULL)
    {
        return 0;
    }

    size_t pos = (size_t)cand_id;
    if (model->frame_to_cluster_pos != NULL)
    {
        pos = (size_t)model->frame_to_cluster_pos[cand_id];
    }

    const int8_t *cand_res = model->rq8_dataset_buffer +
                             pos * (size_t)model->frame_elements;
    telem->rq8_evaluations++;
    float dist_sq = rq8_dist_asym_cutoff_f32(
        query_rq8_adc, cand_res, model->frame_elements, cutoff_sq
    );

    if (dist_sq > cutoff_sq)
    {
        telem->rq8_members_pruned++;
        return 1;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Metric Pruning & Distance Predicate Prototypes (Defined in knn_pruning.c)
 * ------------------------------------------------------------------------- */

/**
 * compare_cluster_scores() - Sort cluster candidates by ascending lower bound.
 * @a: Pointer to first ClusterScore.
 * @b: Pointer to second ClusterScore.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
int compare_cluster_scores(
    const void *a,
    const void *b);

/**
 * is_cluster_pruned_by_pivots() - Check if all cluster members are pruned via pivots.
 * @c:           Candidate cluster index.
 * @cl_radius:   Radius of candidate cluster c.
 * @current_tau: Current search horizon.
 * @eps_factor:  1.0 + epsilon slack factor.
 * @model:       Active KnnModel.
 * @config:      Active KnnConfig.
 * @pivots:      Array of measured anchor pivots.
 * @num_pivots:  Number of active pivots.
 * @sq16_delta:  Quantization error allowance.
 *
 * Return: 1 if cluster is guaranteed to have no members within current_tau, 0 otherwise.
 */
int is_cluster_pruned_by_pivots(
    int                  c,
    double               cl_radius,
    double               current_tau,
    double               eps_factor,
    const KnnModel      *model,
    const KnnConfig     *config,
    const MeasuredPivot *pivots,
    int                  num_pivots,
    double               sq16_delta);

/**
 * is_member_pruned_by_sq8() - Check if member frame is pruned by 8-bit scalar quantization.
 * @query_sq8: Pointer to query frame's SQ8 quantized coordinates.
 * @cand_id:   Global index of candidate frame.
 * @cur_tau:   Current maximum distance threshold in heap.
 * @model:     Active KnnModel.
 * @config:    Active KnnConfig.
 * @telem:     Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_member_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * is_graph_pruned_by_sq8() - Check if graph neighbor is pruned by 8-bit scalar quantization.
 * @query_sq8: Pointer to query frame's SQ8 quantized coordinates.
 * @cand_id:   Global index of candidate neighbor frame.
 * @cur_tau:   Current maximum distance threshold in heap.
 * @model:     Active KnnModel.
 * @config:    Active KnnConfig.
 * @telem:     Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_graph_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * compute_sq16_cutoff_thresh() - Compute integer squared-distance cutoff for SQ16.
 * @cur_tau: Current search horizon distance.
 * @model:   Active KnnModel containing SQ16 scale factors.
 * @config:  Active KnnConfig.
 *
 * Return: Quantized integer distance cutoff threshold.
 */
uint64_t compute_sq16_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config);

/**
 * compute_eq16_cutoff_thresh() - Compute integer squared-distance cutoff for EQ16.
 * @cur_tau: Current search horizon distance.
 * @model:   Active KnnModel containing EQ16 scale factors.
 * @config:  Active KnnConfig.
 *
 * Return: Quantized integer distance cutoff threshold.
 */
uint64_t compute_eq16_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config);

/**
 * compute_rq8_cutoff_thresh() - Compute integer distance cutoff threshold for RQ8.
 * @cur_tau: Current search horizon distance.
 * @model:   Active KnnModel containing RQ8 parameters.
 * @config:  Active KnnConfig.
 *
 * Return: Quantized integer distance cutoff threshold.
 */
uint64_t compute_rq8_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config);

/**
 * compute_rq8_cutoff_thresh_cluster() - Compute cluster-level integer cutoff for RQ8.
 * @cur_tau: Current search horizon distance.
 * @params:  RQ8 parameters for candidate cluster.
 * @config:  Active KnnConfig.
 *
 * Return: Cluster-specific quantized integer cutoff threshold.
 */
uint64_t compute_rq8_cutoff_thresh_cluster(
    double           cur_tau,
    const RQ8Params *params,
    const KnnConfig *config);

/**
 * compute_rq8_cutoff_thresh_adc_cluster() - Compute cluster float cutoff for RQ8 ADC.
 * @cur_tau: Current search horizon distance.
 * @params:  RQ8 parameters for candidate cluster.
 * @config:  Active KnnConfig.
 *
 * Return: Cluster-specific float distance cutoff threshold for ADC.
 */
float compute_rq8_cutoff_thresh_adc_cluster(
    double           cur_tau,
    const RQ8Params *params,
    const KnnConfig *config);

/**
 * is_member_pruned_by_rq8() - Check if member frame is pruned by 8-bit residual quantization.
 * @query_rq8:     Query frame's RQ8 quantized residual codes.
 * @query_rq8_adc: Query frame's RQ8 ADC float representation (or NULL).
 * @cand_id:       Global index of candidate frame.
 * @cur_tau:       Current maximum distance threshold in heap.
 * @model:         Active KnnModel.
 * @config:        Active KnnConfig.
 * @telem:         Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_member_pruned_by_rq8(
    const int16_t   *query_rq8,
    const float     *query_rq8_adc,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * is_member_pruned_by_sq16() - Check if member is pruned by 16-bit scalar quantization.
 * @query_sq16: Query frame's SQ16 quantized coordinates.
 * @cand_id:    Global index of candidate frame.
 * @cur_tau:    Current maximum distance threshold in heap.
 * @model:      Active KnnModel.
 * @config:     Active KnnConfig.
 * @telem:      Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_member_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * is_member_pruned_by_eq16() - Check if member is pruned by E8 lattice quantization.
 * @query_eq16:     Query frame's EQ16 quantized coordinates.
 * @query_eq16_adc: Query frame's EQ16 ADC float coordinates (or NULL).
 * @cand_id:        Global index of candidate frame.
 * @cur_tau:        Current maximum distance threshold in heap.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @telem:          Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_member_pruned_by_eq16(
    const int16_t   *query_eq16,
    const float     *query_eq16_adc,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * is_graph_pruned_by_sq16() - Check if graph neighbor is pruned by 16-bit scalar quantization.
 * @query_sq16: Query frame's SQ16 quantized coordinates.
 * @cand_id:    Global index of candidate frame.
 * @cur_tau:    Current maximum distance threshold in heap.
 * @model:      Active KnnModel.
 * @config:     Active KnnConfig.
 * @telem:      Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_graph_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * is_graph_pruned_by_eq16() - Check if graph neighbor is pruned by E8 lattice quantization.
 * @query_eq16:     Query frame's EQ16 quantized coordinates.
 * @query_eq16_adc: Query frame's EQ16 ADC float coordinates (or NULL).
 * @cand_id:        Global index of candidate frame.
 * @cur_tau:        Current maximum distance threshold in heap.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @telem:          Thread-local telemetry record.
 *
 * Return: 1 if candidate distance is guaranteed >= cur_tau, 0 otherwise.
 */
int is_graph_pruned_by_eq16(
    const int16_t   *query_eq16,
    const float     *query_eq16_adc,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

/**
 * check_temporal_separation() - Verify temporal separation constraint between two frames.
 * @query_id:     Global index of query frame.
 * @candidate_id: Global index of candidate frame.
 * @config:       Active KnnConfig specifying min_frame_separation and max_frame_separation.
 *
 * Return: 1 if separation constraint is satisfied, 0 if candidate must be pruned.
 */
int check_temporal_separation(
    long             query_id,
    long             candidate_id,
    const KnnConfig *config);

/**
 * is_candidate_angular_pruned() - Check if candidate is pruned via angular law-of-cosines bound.
 * @d0:      Distance from query to first reference anchor.
 * @ri:      Radial distance from first anchor to query.
 * @rj:      Radial distance from second anchor to candidate.
 * @di:      Distance from candidate to first anchor.
 * @d_A_ij:  Inter-anchor distance between reference anchors.
 * @tau_eff: Effective search horizon distance threshold.
 * @rlim:    Maximum allowable neighbor distance cutoff.
 *
 * Return: 1 if angular lower bound exceeds horizon, 0 otherwise.
 */
int is_candidate_angular_pruned(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij,
    double tau_eff,
    double rlim);

/**
 * compute_angular_lower_bound_sq() - Compute squared angular Euclidean distance lower bound.
 * @d0:     Distance from query to first reference anchor.
 * @ri:     Radial distance from first anchor to query.
 * @rj:     Radial distance from second anchor to candidate.
 * @di:     Distance from candidate to first anchor.
 * @d_A_ij: Inter-anchor distance between reference anchors.
 *
 * Return: Squared Euclidean distance lower bound.
 */
double compute_angular_lower_bound_sq(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij);

/**
 * is_member_pruned_by_pointwise_pivots() - Prune candidate via measured pointwise seed pivots.
 * @cand_id:          Candidate frame index.
 * @d_anchor:         Distance from query to candidate's cluster anchor.
 * @r_cand:           Radial distance from cluster anchor to candidate.
 * @dcc_best:         Distance between query's nearest cluster and candidate's cluster.
 * @min_d_anchor:     Distance to query's nearest anchor.
 * @num_seed_pivots:  Number of active measured seed pivots.
 * @seed_pivot_ids:   Array of seed frame indices.
 * @seed_pivot_dists: Array of query-to-seed frame distances.
 * @model:            Active KnnModel.
 * @tau_eff:          Effective search horizon distance.
 * @rlim_cutoff:      Optional maximum distance cutoff.
 * @telem:            Thread-local telemetry record.
 *
 * Return: 1 if candidate is guaranteed out of range, 0 otherwise.
 */
int is_member_pruned_by_pointwise_pivots(
    long            cand_id,
    double          d_anchor,
    double          r_cand,
    double          dcc_best,
    double          min_d_anchor,
    int             num_seed_pivots,
    const long     *seed_pivot_ids,
    const double   *seed_pivot_dists,
    const KnnModel *model,
    double          tau_eff,
    double          rlim_cutoff,
    KnnTelemetry   *telem);

/**
 * record_neighbor_and_reciprocal() - Insert neighbor into heap and update reciprocals.
 * @query_id:     Global query frame index.
 * @cand_id:      Global candidate frame index.
 * @dist:         Exact computed Euclidean distance between query and candidate.
 * @config:       Active KnnConfig specifying reciprocal mode.
 * @model:        Active KnnModel.
 * @heap:         Query frame's max-heap.
 * @all_heaps:    Global array of heaps for all frames (for reciprocal updates).
 * @bucket_locks: Array of OpenMP bucket mutexes synchronizing updates (or NULL).
 */
void record_neighbor_and_reciprocal(
    long             query_id,
    long             cand_id,
    double           dist,
    const KnnConfig *config,
    const KnnModel  *model,
    KnnMaxHeap      *heap,
    KnnMaxHeap      *all_heaps
#ifdef _OPENMP
    , omp_lock_t    *bucket_locks
#endif
);

/**
 * cluster_pq_push() - Push cluster node into a min-priority queue ordered by distance.
 * @pq:         Priority queue array.
 * @size:       In/out pointer to current queue size.
 * @cluster_id: Cluster index to insert.
 * @dist:       Distance metric to associate with cluster.
 */
void cluster_pq_push(
    ClusterPqNode *pq,
    int           *size,
    int            cluster_id,
    double         dist);

/**
 * cluster_pq_pop() - Pop minimum distance node from cluster min-priority queue.
 * @pq:   Priority queue array.
 * @size: In/out pointer to current queue size.
 *
 * Return: ClusterPqNode with the minimum distance.
 */
ClusterPqNode cluster_pq_pop(
    ClusterPqNode *pq,
    int           *size);

/**
 * knn_compute_anchor_distance() - Compute distance to cluster anchor with caching and SQ16.
 * @query_data:      Raw pixel buffer of query frame.
 * @c:               Target cluster index.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @visited:         Per-query visited tracker containing precomputed anchor filter results.
 * @anchor_is_sq16:  Output flag set to 1 if distance was derived from SQ16/EQ16, 0 if exact.
 * @telem:           Thread-local telemetry record.
 *
 * Return: Estimated or exact Euclidean distance from query to cluster anchor.
 */
double knn_compute_anchor_distance(
    const void *restrict query_data,
    int                  c,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnVisitedTracker   *visited,
    int                 *anchor_is_sq16,
    KnnTelemetry        *telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_PRUNING_H

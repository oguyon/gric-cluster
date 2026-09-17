/**
 * @file knn_pruning.h
 * @brief Metric lower bounding and distance pruning for k-NN search.
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

    const int8_t *cand_res = model->rq8_dataset_buffer +
                             (size_t)cand_id * (size_t)model->frame_elements;
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

/* -------------------------------------------------------------------------
 * Metric Pruning & Distance Predicate Prototypes (Defined in knn_pruning.c)
 * ------------------------------------------------------------------------- */

int compare_cluster_scores(
    const void *a,
    const void *b);

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

int is_member_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

int is_graph_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

uint64_t compute_sq16_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config);

uint64_t compute_rq8_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config);

uint64_t compute_rq8_cutoff_thresh_cluster(
    double           cur_tau,
    const RQ8Params *params,
    const KnnConfig *config);

int is_member_pruned_by_rq8(
    const int16_t   *query_rq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

int is_member_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

int is_graph_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem);

int check_temporal_separation(
    long             query_id,
    long             candidate_id,
    const KnnConfig *config);

int is_candidate_angular_pruned(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij,
    double tau_eff,
    double rlim);

double compute_angular_lower_bound_sq(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij);

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

void cluster_pq_push(
    ClusterPqNode *pq,
    int           *size,
    int            cluster_id,
    double         dist);

ClusterPqNode cluster_pq_pop(
    ClusterPqNode *pq,
    int           *size);

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

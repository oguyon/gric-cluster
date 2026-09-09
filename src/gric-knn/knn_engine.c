/**
 * @file knn_engine.c
 * @brief High-performance metric-pruned k-NN solver engine.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_engine.h"
#include "knn_heap.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "cluster_locator.h"
#include "framedistance.h"
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/** Cluster candidate record for sorting by ascending lower bound and center proximity */
typedef struct
{
    int    id;
    double lb;
    double dcc;
} ClusterScore;

#define MAX_MEASURED_PIVOTS 8
#define GRAPH_FRONTIER_MAX 256
#define KNN_NUM_BUCKET_LOCKS 4096
#define KNN_BUCKET_LOCK_MASK (KNN_NUM_BUCKET_LOCKS - 1)

/** Measured anchor pivot record for Multi-Anchor Pivot Bounding (AESA) */
typedef struct
{
    int    cluster_id;
    double d_anchor;
} MeasuredPivot;

/** Graph frontier node for approximate dynamic search */
typedef struct
{
    long    frame_id;
    double  dist;
    long    parent_id;
    double  parent_dist;
    uint8_t expanded;
} FrontierNode;

/**
 * compare_cluster_scores() - Sort cluster candidates by ascending lower bound.
 * @a: Pointer to first ClusterScore.
 * @b: Pointer to second ClusterScore.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_cluster_scores(
    const void *a,
    const void *b)
{
    const ClusterScore *ca = (const ClusterScore *)a;
    const ClusterScore *cb = (const ClusterScore *)b;
    if (ca->lb < cb->lb)
    {
        return -1;
    }
    if (ca->lb > cb->lb)
    {
        return 1;
    }
    if (ca->dcc < cb->dcc)
    {
        return -1;
    }
    if (ca->dcc > cb->dcc)
    {
        return 1;
    }
    return 0;
}

/** Quantized member candidate record for query-centric evaluation ordering */
typedef struct
{
    long     cand_id;
    uint64_t ssd;
} Sq16MemberCandidate;

/**
 * compare_sq16_candidates() - Sort SQ16 candidates by ascending quantized SSD.
 * @a: Pointer to first Sq16MemberCandidate.
 * @b: Pointer to second Sq16MemberCandidate.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_sq16_candidates(
    const void *a,
    const void *b)
{
    const Sq16MemberCandidate *ca = (const Sq16MemberCandidate *)a;
    const Sq16MemberCandidate *cb = (const Sq16MemberCandidate *)b;
    if (ca->ssd < cb->ssd)
    {
        return -1;
    }
    if (ca->ssd > cb->ssd)
    {
        return 1;
    }
    return 0;
}

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
 * is_member_pruned_by_sq8() - Evaluate SQ8 metric lower bound against current search radius.
 * @query_sq8: Pointer to quantized query vector [dim].
 * @cand_id:   Index of candidate dataset frame.
 * @cur_tau:   Current distance to k-th nearest neighbor (or cutoff radius).
 * @model:     Active KnnModel.
 * @config:    Active KnnConfig.
 * @telem:     Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem)
{
    if (!config->use_sq8 || model->sq8_dataset_buffer == NULL || query_sq8 == NULL)
    {
        return 0;
    }

    const uint8_t *cand_sq8 = model->sq8_dataset_buffer +
                              (size_t)cand_id * (size_t)model->frame_elements;
    double eps = config->sq8_approx ? config->epsilon : 0.0;
    telem->sq8_evaluations++;
    double d_lb = sq8_compute_lower_bound(query_sq8, cand_sq8, &model->sq8_params, eps);

    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < cur_tau)
    {
        cur_tau = config->rlim_cutoff;
    }

    if (d_lb > cur_tau)
    {
        telem->sq8_members_pruned++;
        return 1;
    }

    return 0;
}

/**
 * is_graph_pruned_by_sq8() - Evaluate SQ8 metric lower bound for graph candidate pruning.
 * @query_sq8: Pointer to quantized query vector [dim].
 * @cand_id:   Index of candidate dataset frame.
 * @cur_tau:   Current distance threshold (e.g. heap max or routing threshold).
 * @model:     Active KnnModel.
 * @config:    Active KnnConfig.
 * @telem:     Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_graph_pruned_by_sq8(
    const uint8_t   *query_sq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem)
{
    if (!config->use_sq8 || model->sq8_dataset_buffer == NULL || query_sq8 == NULL)
    {
        return 0;
    }

    const uint8_t *cand_sq8 = model->sq8_dataset_buffer +
                              (size_t)cand_id * (size_t)model->frame_elements;
    double eps = config->sq8_approx ? config->epsilon : 0.0;
    telem->sq8_evaluations++;
    double d_lb = sq8_compute_lower_bound(query_sq8, cand_sq8, &model->sq8_params, eps);

    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < cur_tau)
    {
        cur_tau = config->rlim_cutoff;
    }

    if (d_lb > cur_tau)
    {
        telem->sq8_graph_pruned++;
        return 1;
    }

    return 0;
}

/**
 * compute_sq16_cutoff_thresh() - Precompute squared integer cutoff threshold for SQ16.
 * @cur_tau: Current search radius.
 * @model:   Active KnnModel.
 * @config:  Active KnnConfig.
 *
 * Return: Cutoff SSD threshold, or UINT64_MAX if bounds cannot prune.
 */
static inline uint64_t compute_sq16_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config)
{
    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < cur_tau)
    {
        cur_tau = config->rlim_cutoff;
    }

    double eps = config->sq16_approx ? config->epsilon : 0.0;
    double raw_thresh = (cur_tau * (1.0 + eps) +
                        2.0 * (double)model->sq16_params.err_radius) *
                        (double)model->sq16_params.inv_scale;
    if (raw_thresh >= 4294967295.0)
    {
        return UINT64_MAX;
    }

    return (raw_thresh > 0.0) ? (uint64_t)(raw_thresh * raw_thresh) : 0;
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
 * is_member_pruned_by_sq16() - Evaluate SQ16 metric lower bound against current search radius.
 * @query_sq16: Pointer to quantized query vector [dim].
 * @cand_id:    Index of candidate dataset frame.
 * @cur_tau:    Current distance to k-th nearest neighbor (or cutoff radius).
 * @model:      Active KnnModel.
 * @config:     Active KnnConfig.
 * @telem:      Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem)
{
    if (!config->use_sq16 || model->sq16_dataset_buffer == NULL || query_sq16 == NULL)
    {
        return 0;
    }

    uint64_t ssd_cutoff = compute_sq16_cutoff_thresh(cur_tau, model, config);
    return is_member_pruned_by_sq16_cached(query_sq16, cand_id, ssd_cutoff, model, telem);
}

/**
 * is_graph_pruned_by_sq16() - Evaluate SQ16 metric lower bound for graph candidate pruning.
 * @query_sq16: Pointer to quantized query vector [dim].
 * @cand_id:    Index of candidate dataset frame.
 * @cur_tau:    Current distance threshold (e.g. heap max or routing threshold).
 * @model:      Active KnnModel.
 * @config:     Active KnnConfig.
 * @telem:      Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_graph_pruned_by_sq16(
    const int16_t   *query_sq16,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem)
{
    if (!config->use_sq16 || model->sq16_dataset_buffer == NULL || query_sq16 == NULL)
    {
        return 0;
    }

    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < cur_tau)
    {
        cur_tau = config->rlim_cutoff;
    }

    double eps = config->sq16_approx ? config->epsilon : 0.0;
    double raw_thresh = (cur_tau * (1.0 + eps) + 2.0 * (double)model->sq16_params.err_radius) *
                        (double)model->sq16_params.inv_scale;
    if (raw_thresh >= 4294967295.0)
    {
        return 0;
    }
    uint64_t ssd_cutoff = (raw_thresh > 0.0) ? (uint64_t)(raw_thresh * raw_thresh) : 0;

    const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                               (size_t)cand_id * (size_t)model->frame_elements;
    telem->sq16_evaluations++;
    uint64_t ssd = sq16_dist_squared_cutoff_i16(
        query_sq16, cand_sq16, model->frame_elements, ssd_cutoff
    );

    if (ssd > ssd_cutoff)
    {
        telem->sq16_graph_pruned++;
        return 1;
    }

    return 0;
}

/**
 * check_temporal_separation() - Verify if candidate satisfies temporal criteria.
 * @query_id:     Frame ID of query.
 * @candidate_id: Frame ID of candidate.
 * @config:       Active KnnConfig.
 *
 * Return: 1 if candidate is temporally valid, 0 if excluded.
 */
static inline int check_temporal_separation(
    long             query_id,
    long             candidate_id,
    const KnnConfig *config)
{
    if (config->query_data_path != NULL)
    {
        return 1; // In cross-dataset mode, queries have no temporal continuity with candidates
    }

    if (query_id == candidate_id)
    {
        return 0; // Self-match exclusion
    }

    if (config->past_only && candidate_id > query_id)
    {
        return 0;
    }

    if (config->future_only && candidate_id < query_id)
    {
        return 0;
    }

    long diff = query_id - candidate_id;
    if (diff < 0)
    {
        diff = -diff;
    }

    if (diff < (long)config->min_temporal_sep)
    {
        return 0;
    }

    return 1;
}

/**
 * is_candidate_angular_pruned() - Prune candidate neighbor using angular geometry.
 * @d0:        Distance from query to central node: d(b, a0).
 * @ri:        Distance from central node to evaluated neighbor: d(a0, ai).
 * @rj:        Distance from central node to un-evaluated neighbor: d(a0, aj).
 * @di:        Evaluated distance from query to neighbor: d(b, ai).
 * @d_A_ij:    Precomputed mutual distance between ai and aj: d(ai, aj).
 * @tau_eff:   Effective search threshold radius (tau / eps_factor).
 * @rlim:      Cutoff radius cutoff (if > 0).
 *
 * Return: 1 if candidate is pruned, 0 if candidate remains viable.
 */
static inline int is_candidate_angular_pruned(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij,
    double tau_eff,
    double rlim)
{
    // Fast path: 1D metric pivot bound check
    double lb_met = fabs(di - d_A_ij);
    if (lb_met >= tau_eff || (rlim > 0.0 && lb_met >= rlim))
    {
        return 1;
    }

    if (d0 <= 1e-12 || ri <= 1e-12 || rj <= 1e-12)
    {
        return 0;
    }

    double cos_th_i = (d0 * d0 + ri * ri - di * di) / (2.0 * d0 * ri);
    if (cos_th_i > 1.0)
    {
        cos_th_i = 1.0;
    }
    else if (cos_th_i < -1.0)
    {
        cos_th_i = -1.0;
    }

    double cos_phi = (ri * ri + rj * rj - d_A_ij * d_A_ij) / (2.0 * ri * rj);
    if (cos_phi > 1.0)
    {
        cos_phi = 1.0;
    }
    else if (cos_phi < -1.0)
    {
        cos_phi = -1.0;
    }

    double sin_th_i = sqrt(fmax(0.0, 1.0 - cos_th_i * cos_th_i));
    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

    double cos_th_j_max = cos_th_i * cos_phi + sin_th_i * sin_phi;
    if (cos_th_j_max > 1.0)
    {
        cos_th_j_max = 1.0;
    }

    double lb2 = d0 * d0 + rj * rj - 2.0 * d0 * rj * cos_th_j_max;
    double tau_target = (rlim > 0.0 && rlim < tau_eff) ? rlim : tau_eff;

    if (lb2 >= tau_target * tau_target)
    {
        return 1;
    }

    return 0;
}

/**
 * compute_angular_lower_bound_sq() - Compute squared lower bound from angular geometry.
 * @d0:     Distance from query to central node: d(b, a0).
 * @ri:     Distance from central node to evaluated neighbor: d(a0, ai).
 * @rj:     Distance from central node to un-evaluated neighbor: d(a0, aj).
 * @di:     Evaluated distance from query to neighbor: d(b, ai).
 * @d_A_ij: Precomputed mutual distance between ai and aj: d(ai, aj).
 *
 * Return: Maximum squared lower bound from metric pivot and angular cosine bounds.
 */
static inline double compute_angular_lower_bound_sq(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij)
{
    double diff = fabs(di - d_A_ij);
    double lb2_met = diff * diff;

    if (d0 <= 1e-12 || ri <= 1e-12 || rj <= 1e-12)
    {
        return lb2_met;
    }

    double cos_th_i = (d0 * d0 + ri * ri - di * di) / (2.0 * d0 * ri);
    if (cos_th_i > 1.0)
    {
        cos_th_i = 1.0;
    }
    else if (cos_th_i < -1.0)
    {
        cos_th_i = -1.0;
    }

    double cos_phi = (ri * ri + rj * rj - d_A_ij * d_A_ij) / (2.0 * ri * rj);
    if (cos_phi > 1.0)
    {
        cos_phi = 1.0;
    }
    else if (cos_phi < -1.0)
    {
        cos_phi = -1.0;
    }

    double sin_th_i = sqrt(fmax(0.0, 1.0 - cos_th_i * cos_th_i));
    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

    double cos_th_j_max = cos_th_i * cos_phi + sin_th_i * sin_phi;
    if (cos_th_j_max > 1.0)
    {
        cos_th_j_max = 1.0;
    }

    double lb2_ang = d0 * d0 + rj * rj - 2.0 * d0 * rj * cos_th_j_max;
    return fmax(lb2_met, lb2_ang);
}

/**
 * is_member_pruned_by_pointwise_pivots() - Multi-Pivot Pointwise Member Bounding.
 * @cand_id:          Candidate frame index.
 * @d_anchor:         Distance from query to candidate cluster anchor.
 * @r_cand:           Distance from anchor to member candidate.
 * @dcc_best:         Distance from closest anchor to candidate cluster anchor.
 * @min_d_anchor:     Distance from query to closest anchor.
 * @num_seed_pivots:  Number of evaluated graph seed pivots [0..8].
 * @seed_pivot_ids:   Array of evaluated graph seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seeds.
 * @model:            KnnModel pointer.
 * @tau_eff:          Effective search radius bound.
 * @rlim_cutoff:      Radius cutoff (if any).
 * @telem:            Telemetry structure to record pruning counters.
 *
 * Return: 1 if candidate is pruned, 0 if candidate must be evaluated.
 */
static inline int is_member_pruned_by_pointwise_pivots(
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
    KnnTelemetry   *telem)
{
    // Pivot 1: Own cluster anchor
    double lb1 = fabs(d_anchor - r_cand);
    if (lb1 >= tau_eff || (rlim_cutoff > 0.0 && lb1 >= rlim_cutoff))
    {
        telem->level3_annular_pruned++;
        return 1;
    }

    // Pivot 2: Closest cluster anchor (if distinct from own anchor)
    if (dcc_best > 0.0)
    {
        double lb2_1 = dcc_best - r_cand - min_d_anchor;
        double lb2_2 = min_d_anchor - dcc_best - r_cand;
        double lb2 = (lb2_1 > lb2_2) ? lb2_1 : lb2_2;
        if (lb2 >= tau_eff || (rlim_cutoff > 0.0 && lb2 >= rlim_cutoff))
        {
            telem->multi_pivot_pruned++;
            return 1;
        }
    }

    // Pivots 3..H: Evaluated Graph Seed Hops
    if (num_seed_pivots > 0 && model->has_knn_graph && model->graph_distances != NULL)
    {
        int graph_k = model->graph_k;
        for (int p = 0; p < num_seed_pivots; p++)
        {
            long   s_id = seed_pivot_ids[p];
            double s_dist = seed_pivot_dists[p];

            double r_gball =
                (double)model->graph_distances[s_id * (long)graph_k + (graph_k - 1)];
            double lb_gball = r_gball - s_dist;

            const uint32_t *snbrs = &model->graph_indices[s_id * (long)graph_k];
            const float    *sdists = &model->graph_distances[s_id * (long)graph_k];

            double edge_d = -1.0;
            for (int kn = 0; kn < graph_k; kn++)
            {
                if ((long)snbrs[kn] == cand_id)
                {
                    edge_d = (double)sdists[kn];
                    break;
                }
            }

            if (edge_d >= 0.0)
            {
                double lb_edge = fabs(s_dist - edge_d);
                if (lb_edge >= tau_eff || (rlim_cutoff > 0.0 && lb_edge >= rlim_cutoff))
                {
                    telem->graph_edges_pruned++;
                    return 1;
                }
            }
            else if (lb_gball >= tau_eff || (rlim_cutoff > 0.0 && lb_gball >= rlim_cutoff))
            {
                telem->graph_edges_pruned++;
                return 1;
            }
        } // for (int p = 0; ...)
    }

    return 0;
}

/**
 * record_neighbor_and_reciprocal() - Insert candidate into heap and update reciprocal neighbor.
 * @query_id:     Query frame index.
 * @cand_id:      Candidate frame index.
 * @dist:         Calculated distance.
 * @config:       Active KnnConfig.
 * @model:        Active KnnModel.
 * @heap:         Heap for query frame.
 * @all_heaps:    All heaps array.
 * @bucket_locks: OpenMP bucket locks (if multithreaded).
 */
static inline void record_neighbor_and_reciprocal(
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
)
{
    if (config->rlim_cutoff > 0.0 && dist > config->rlim_cutoff)
    {
        return;
    }

#ifdef _OPENMP
    if (bucket_locks != NULL)
    {
        omp_set_lock(&bucket_locks[query_id & KNN_BUCKET_LOCK_MASK]);
        knn_heap_push(heap, (int)cand_id, dist);
        omp_unset_lock(&bucket_locks[query_id & KNN_BUCKET_LOCK_MASK]);
    }
    else
    {
        knn_heap_push(heap, (int)cand_id, dist);
    }
#else
    knn_heap_push(heap, (int)cand_id, dist);
#endif

    if (config->use_reciprocal && cand_id > query_id &&
        cand_id < model->total_dataset_frames && all_heaps != NULL)
    {
        KnnMaxHeap *target_heap = &all_heaps[cand_id];
        if (target_heap->count < target_heap->k || dist < target_heap->data[0].dist)
        {
#ifdef _OPENMP
            if (bucket_locks != NULL)
            {
                omp_set_lock(&bucket_locks[cand_id & KNN_BUCKET_LOCK_MASK]);
                knn_heap_push(target_heap, (int)query_id, dist);
                omp_unset_lock(&bucket_locks[cand_id & KNN_BUCKET_LOCK_MASK]);
            }
            else
            {
                knn_heap_push(target_heap, (int)query_id, dist);
            }
#else
            knn_heap_push(target_heap, (int)query_id, dist);
#endif
        }
    }
}

/**
 * knn_search_intra_cluster() - Search members of the query's home cluster.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance from query to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_search_intra_cluster(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    double                 r_home,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    const KnnCluster *home_cl = &model->clusters[home_cluster_id];
    telem->total_candidates_considered += (uint64_t)home_cl->num_members;
    double eps_factor = 1.0 + config->epsilon;

    int start_m = 0;
    int end_m = home_cl->num_members;
    double tau_init = knn_heap_peek_max_dist(heap);
    if (tau_init < 1e20)
    {
        float r_min = (float)fmax(0.0, r_home - tau_init / eps_factor);
        float r_max = (float)(r_home + tau_init / eps_factor);
        start_m = find_member_lower_bound(home_cl->members, home_cl->num_members, r_min);
        end_m = find_member_upper_bound(home_cl->members, home_cl->num_members, r_max);
        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(home_cl->num_members - end_m);
    }

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;
    long frame_elem = model->frame_elements;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    int sq16_active = (config->use_sq16 && model->sq16_dataset_buffer != NULL &&
                       visited->query_sq16 != NULL);

    for (int m = start_m; m < end_m; m++)
    {
        long cand_id = (long)home_cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double r_cand = (double)home_cl->members[m].r_anchor;
        double lb_annular = fabs(r_home - r_cand);

        if (lb_annular >= current_tau / eps_factor)
        {
            telem->level3_annular_pruned++;
            continue;
        }

        if (sq16_active)
        {
            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }
            if (is_member_pruned_by_sq16_cached(
                    visited->query_sq16, cand_id, cached_ssd_cutoff, model, telem
                ))
            {
                continue;
            }
        }
        else if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                          model, config, telem))
        {
            continue;
        }

        if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                    model, config, telem))
        {
            continue;
        }

        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }

        if (!config->use_batch_dist)
        {
            const void *cand_ptr = NULL;
            if (reader->memory_data != NULL)
            {
                cand_ptr = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_ptr = cand_buffer;
            }
            if (cand_ptr != NULL)
            {
                telem->framedist_calls++;
                double c_tau = (config->rlim_cutoff > 0.0 && config->rlim_cutoff < current_tau)
                               ? config->rlim_cutoff
                               : current_tau;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                );
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            }
            continue;
        }

        const void *dest = NULL;
        if (reader->memory_data != NULL)
        {
            dest = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
        }
        else
        {
            void *buf_dest = (char *)cand_buffer + (size_t)batch_count * frame_bytes;
            if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
            {
                dest = buf_dest;
            }
        }
        if (dest != NULL)
        {
            batch_cand_ids[batch_count] = cand_id;
            batch_ptrs[batch_count] = dest;
            batch_count++;
            if (batch_count == 4)
            {
                double dists[4];
                if (model->is_double)
                {
                    framedist_batch_1x4_double(
                        (const double *)query_data,
                        (const double *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                else
                {
                    framedist_batch_1x4_float(
                        (const float *)query_data,
                        (const float *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                telem->framedist_calls += 4;
                for (int b = 0; b < 4; b++)
                {
                    record_neighbor_and_reciprocal(
                        query_id, batch_cand_ids[b], dists[b],
                        config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
                batch_count = 0;
            }
        }
    } // for (int m = start_m; ...)

    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
}

/**
 * knn_warm_start_nearest_cluster() - Search nearest neighboring cluster if heap is not full.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @pivots:          Output pivot array.
 * @num_pivots:      Output pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 *
 * Return: Cluster ID of nearest cluster used for warm start, or -1.
 */
static int knn_warm_start_nearest_cluster(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    if (home_cluster_id < 0 || home_cluster_id >= model->num_clusters)
    {
        return -1;
    }

    // Fast Graph Warm-Start (if precomputed k-NN graph is resident)
    if (model->has_knn_graph && model->graph_indices != NULL &&
        query_id >= 0 && query_id < model->total_dataset_frames)
    {
        int g_k = model->graph_k;
        const uint32_t *nb_indices = &model->graph_indices[(size_t)query_id * (size_t)g_k];
        const float    *nb_dists = (model->graph_distances != NULL) ?
            &model->graph_distances[(size_t)query_id * (size_t)g_k] : NULL;

        for (int j = 0; j < g_k; j++)
        {
            long nb = (long)nb_indices[j];
            if (nb >= 0 && nb < model->total_dataset_frames && nb != query_id)
            {
                if (!knn_visited_is_visited(visited, nb))
                {
                    if (check_temporal_separation(query_id, nb, config))
                    {
                        double d = (nb_dists != NULL) ? (double)nb_dists[j] : -1.0;
                        if (d < 0.0)
                        {
                            const void *cand_data = NULL;
                            if (reader->memory_data != NULL)
                            {
                                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                                cand_data = (const char *)reader->memory_data +
                                            (size_t)nb * (size_t)model->frame_elements * el_sz;
                            }
                            else if (knn_reader_read_frame(reader, nb, cand_buffer) == 0)
                            {
                                cand_data = cand_buffer;
                            }
                            if (cand_data != NULL)
                            {
                                telem->framedist_calls++;
                                d = compute_euclidean_distance(
                                    query_data, cand_data, model->frame_elements, model->is_double
                                );
                            }
                        }
                        if (d >= 0.0)
                        {
                            knn_visited_check_and_mark(visited, nb);
                            knn_heap_push(heap, (int)nb, d);
                        }
                    }
                }
            }
        }
        if (heap->count >= heap->k)
        {
            return -1;
        }
    }

    if (heap->count >= heap->k)
    {
        return -1;
    }

    // Multi-Cluster Nearest Warm-Start until heap saturates
    int M = model->num_clusters;
    int visited_warm[8];
    int num_warm = 0;
    int first_warm_c = -1;
    double eps_factor = 1.0 + config->epsilon;

    while (heap->count < heap->k && num_warm < 8)
    {
        int best_c = -1;
        double min_dcc = 1e19;

        for (int c = 0; c < M; c++)
        {
            if (c == home_cluster_id || model->clusters[c].num_members == 0)
            {
                continue;
            }
            int already_w = 0;
            for (int w = 0; w < num_warm; w++)
            {
                if (visited_warm[w] == c)
                {
                    already_w = 1;
                    break;
                }
            }
            if (already_w)
            {
                continue;
            }

            double dcc = model->dcc_matrix[home_cluster_id * M + c];
            if (dcc > 0.0 && dcc < min_dcc)
            {
                min_dcc = dcc;
                best_c = c;
            }
        }

        if (best_c < 0)
        {
            break;
        }
        visited_warm[num_warm++] = best_c;
        if (first_warm_c < 0)
        {
            first_warm_c = best_c;
        }

        const KnnCluster *warm_cl = &model->clusters[best_c];
        telem->total_candidates_considered += (uint64_t)warm_cl->num_members;
        telem->framedist_calls++;

        double d_anchor = compute_euclidean_distance(
            query_data, warm_cl->anchor_data, model->frame_elements, model->is_double
        );

        for (int m = 0; m < warm_cl->num_members; m++)
        {
            long cand_id = (long)warm_cl->members[m].frame_id;

            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }
            double r_cand = (double)warm_cl->members[m].r_anchor;
            double lb1 = fabs(d_anchor - r_cand);
            double current_tau = knn_heap_peek_max_dist(heap);
            if (lb1 >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }
            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
            {
                continue;
            }
            if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
            {
                telem->reciprocal_reused++;
                continue;
            }
            const void *cand_data = NULL;
            if (reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)reader->memory_data +
                            (size_t)cand_id * (size_t)model->frame_elements * el_sz;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, model->frame_elements, model->is_double
                );
                record_neighbor_and_reciprocal(
                    query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        } // for (int m = 0; ...)

        if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS)
        {
            pivots[*num_pivots].cluster_id = best_c;
            pivots[*num_pivots].d_anchor = d_anchor;
            (*num_pivots)++;
        }
    } // while (heap->count < heap->k && num_warm < 8)

    return first_warm_c;
}

/**
 * knn_score_candidate_clusters() - Filter super-clusters and score candidate clusters.
 * @home_cluster_id: Home cluster index.
 * @pivots:          Array of measured anchor pivots.
 * @num_pivots:      Number of measured anchor pivots.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @heap:            Max-heap for current query.
 * @scores_buffer:   Output array for scored candidate clusters.
 * @telem:           Telemetry record.
 *
 * Return: Number of candidate clusters populated in scores_buffer.
 */
static int knn_score_candidate_clusters(
    int                     home_cluster_id,
    const MeasuredPivot    *pivots,
    int                     num_pivots,
    double                  r_home,
    const KnnModel         *model,
    const KnnConfig        *config,
    const KnnMaxHeap       *heap,
    ClusterScore *restrict  scores_buffer,
    KnnTelemetry *restrict  telem)
{
    int M = model->num_clusters;
    int num_cand_clusters = 0;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);

    if (model->num_super_clusters > 1 && home_cluster_id >= 0 &&
        model->cluster_super_map != NULL)
    {
        int home_super_id = model->cluster_super_map[home_cluster_id];
        double r_home_super = model->super_clusters[home_super_id].radius;
        int K = model->num_super_clusters;

        for (int s = 0; s < K; s++)
        {
            if (s != home_super_id)
            {
                double dss = model->dss_matrix[home_super_id * K + s];
                double r_s = model->super_clusters[s].radius;
                double lb_super = dss - r_home_super - r_s;
                if (lb_super < 0.0)
                {
                    lb_super = 0.0;
                }

                // Level 0: Super-Cluster Pruning
                if (lb_super >= current_tau / eps_factor)
                {
                    telem->level0_super_clusters_pruned++;
                    telem->level1_clusters_pruned +=
                        (uint64_t)model->super_clusters[s].num_clusters;
                    continue; // Skip all child clusters in super-cluster s
                }
            }

            // Populate child clusters of surviving super-clusters
            const KnnSuperCluster *sc = &model->super_clusters[s];
            for (int ci = 0; ci < sc->num_clusters; ci++)
            {
                int q = sc->cluster_ids[ci];
                if (q == home_cluster_id || is_in_pivots(q, pivots, num_pivots) ||
                    model->clusters[q].num_members == 0)
                {
                    continue;
                }

                double dcc = model->dcc_matrix[home_cluster_id * M + q];
                double r_q = model->clusters[q].radius;
                double lb = dcc - r_home - r_q;
                if (lb < 0.0)
                {
                    lb = 0.0;
                }

                // Level 1: Cluster-level DCC bound
                if (current_tau < 1e18 && lb >= current_tau / eps_factor)
                {
                    telem->level1_clusters_pruned++;
                    continue;
                }

                // Multi-Point TE4 Triangulation against measured pivots
                if (num_pivots >= 2)
                {
                    int    c1 = pivots[0].cluster_id;
                    int    c2 = pivots[1].cluster_id;
                    double d1 = pivots[0].d_anchor;
                    double d2 = pivots[1].d_anchor;
                    double d12 = model->dcc_matrix[c1 * M + c2];
                    double d1q = model->dcc_matrix[c1 * M + q];
                    double d2q = model->dcc_matrix[c2 * M + q];

                    double min_d = calc_min_dist_4pt(d1, d2, d12, d1q, d2q);
                    if (min_d - r_q - 1e-5 >= current_tau / eps_factor)
                    {
                        telem->level1_clusters_pruned++;
                        continue;
                    }
                    if (min_d - r_q > lb)
                    {
                        lb = min_d - r_q;
                    }
                }

                scores_buffer[num_cand_clusters].id = q;
                scores_buffer[num_cand_clusters].lb = lb;
                scores_buffer[num_cand_clusters].dcc = dcc;
                num_cand_clusters++;
            } // for (int ci = 0; ...)
        } // for (int s = 0; ...)
    }
    else
    {
        for (int q = 0; q < M; q++)
        {
            if (q == home_cluster_id || is_in_pivots(q, pivots, num_pivots) ||
                model->clusters[q].num_members == 0)
            {
                continue;
            }

            double dcc = model->dcc_matrix[home_cluster_id * M + q];
            double r_q = model->clusters[q].radius;
            double lb = dcc - r_home - r_q;
            if (lb < 0.0)
            {
                lb = 0.0;
            }

            // Level 1: Cluster-level DCC bound
            if (current_tau < 1e18 && lb >= current_tau / eps_factor)
            {
                telem->level1_clusters_pruned++;
                continue;
            }

            // Multi-Point TE4 Triangulation against measured pivots
            if (num_pivots >= 2)
            {
                int    c1 = pivots[0].cluster_id;
                int    c2 = pivots[1].cluster_id;
                double d1 = pivots[0].d_anchor;
                double d2 = pivots[1].d_anchor;
                double d12 = model->dcc_matrix[c1 * M + c2];
                double d1q = model->dcc_matrix[c1 * M + q];
                double d2q = model->dcc_matrix[c2 * M + q];

                double min_d = calc_min_dist_4pt(d1, d2, d12, d1q, d2q);
                if (min_d - r_q - 1e-5 >= current_tau / eps_factor)
                {
                    telem->level1_clusters_pruned++;
                    continue;
                }
                if (min_d - r_q > lb)
                {
                    lb = min_d - r_q;
                }
            }

            scores_buffer[num_cand_clusters].id = q;
            scores_buffer[num_cand_clusters].lb = lb;
            scores_buffer[num_cand_clusters].dcc = dcc;
            num_cand_clusters++;
        } // for (int q = 0; ...)
    }

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    return num_cand_clusters;
}

/** Priority queue node for cluster graph routing */
typedef struct
{
    int    cluster_id;
    double dist;
} ClusterPqNode;

/** Thread-local scratch buffers for cluster graph routing */
typedef struct
{
    ClusterPqNode *pq;
    uint32_t      *enqueued_tags;
    uint32_t       enqueued_epoch;
    double        *anchor_dists;
    uint8_t       *anchor_is_sq16;
} KnnClusterGraphScratch;

/**
 * cluster_pq_push() - Push cluster node into min-heap.
 * @pq:         Min-heap buffer.
 * @size:       Pointer to heap size.
 * @cluster_id: Cluster index.
 * @dist:       Cluster anchor distance.
 */
static inline void cluster_pq_push(
    ClusterPqNode *pq,
    int           *size,
    int            cluster_id,
    double         dist)
{
    int i = (*size)++;
    while (i > 0)
    {
        int p = (i - 1) / 2;
        if (pq[p].dist <= dist)
        {
            break;
        }
        pq[i] = pq[p];
        i = p;
    }
    pq[i].cluster_id = cluster_id;
    pq[i].dist = dist;
}

/**
 * cluster_pq_pop() - Pop min-distance cluster from min-heap.
 * @pq:   Min-heap buffer.
 * @size: Pointer to heap size.
 *
 * Return: Popped ClusterPqNode with minimum distance.
 */
static inline ClusterPqNode cluster_pq_pop(
    ClusterPqNode *pq,
    int           *size)
{
    ClusterPqNode top = pq[0];
    (*size)--;
    if (*size > 0)
    {
        ClusterPqNode last = pq[*size];
        int i = 0;
        while (2 * i + 1 < *size)
        {
            int left = 2 * i + 1;
            int right = left + 1;
            int smallest = left;
            if (right < *size && pq[right].dist < pq[left].dist)
            {
                smallest = right;
            }
            if (last.dist <= pq[smallest].dist)
            {
                break;
            }
            pq[i] = pq[smallest];
            i = smallest;
        }
        pq[i] = last;
    }
    return top;
}

/**
 * knn_compute_anchor_distance() - Compute distance from query to cluster anchor.
 * @query_data:     Query frame pixel data.
 * @c:              Cluster index.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @visited:        Per-query frame visited tracker.
 * @anchor_is_sq16: Output flag set to 1 if SQ16 was used.
 * @telem:          Active KnnTelemetry.
 *
 * Return: Euclidean distance between query and anchor.
 */
static inline double knn_compute_anchor_distance(
    const void *restrict query_data,
    int                  c,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnVisitedTracker   *visited,
    int                 *anchor_is_sq16,
    KnnTelemetry        *telem)
{
    long frame_elem = model->frame_elements;
    if (config->use_sq16 && model->anchor_sq16_buffer != NULL && visited->query_sq16 != NULL)
    {
        const int16_t *anchor_sq16 = model->anchor_sq16_buffer +
                                     (size_t)c * (size_t)frame_elem;
        telem->sq16_evaluations++;
        uint64_t ssd = sq16_dist_squared_cutoff_i16(
            visited->query_sq16, anchor_sq16, frame_elem, UINT64_MAX
        );
        *anchor_is_sq16 = 1;
        return sqrt((double)ssd) * (double)model->sq16_params.scale;
    }

    *anchor_is_sq16 = 0;
    if (config->use_sq8 && model->anchor_sq8_buffer != NULL && visited->query_sq8 != NULL)
    {
        const uint8_t *anchor_sq8 = model->anchor_sq8_buffer +
                                    (size_t)c * (size_t)frame_elem;
        telem->sq8_evaluations++;
        uint64_t ssd = sq8_dist_squared_u8(visited->query_sq8, anchor_sq8, frame_elem);
        return sqrt((double)ssd) * (double)model->sq8_params.scale;
    }

    telem->framedist_calls++;
    return compute_euclidean_distance(
        query_data, model->clusters[c].anchor_data, frame_elem, model->is_double
    );
}

/**
 * knn_eval_candidate_cluster_members() - Evaluate members of a single candidate cluster.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @q:               Candidate cluster index.
 * @d_anchor:        Query-to-anchor distance for cluster q.
 * @anchor_is_sq16:  1 if d_anchor was computed using SQ16.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all heaps.
 * @bucket_locks:    OpenMP bucket locks.
 * @pivots:          Pivot array.
 * @num_pivots:      Pointer to pivot count.
 * @visited:         Per-query frame visited tracker.
 * @batch_count:     Pointer to current batch count.
 * @batch_cand_ids:  Batch candidate frame IDs buffer.
 * @batch_ptrs:      Batch candidate pointers buffer.
 * @telem:           Telemetry record.
 */
static void knn_eval_candidate_cluster_members(
    long                   query_id,
    const void *restrict   query_data,
    int                    q,
    double                 d_anchor,
    int                    anchor_is_sq16,
    int                    home_cluster_id,
    double                 r_home,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    int                   *batch_count,
    long                   batch_cand_ids[4],
    const void            *batch_ptrs[4],
    KnnTelemetry  *restrict telem)
{
    const KnnCluster *cl = &model->clusters[q];
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));
    double current_tau = knn_heap_peek_max_dist(heap);

    // Level 3: Radially-Sorted Annular Window Slicing & Multi-Pivot Filter
    telem->total_candidates_considered += (uint64_t)cl->num_members;

    double tau_eff = current_tau / eps_factor;
    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
    {
        tau_eff = config->rlim_cutoff;
    }

    double sq16_delta = anchor_is_sq16 ? 2.0 * (double)model->sq16_params.err_radius : 0.0;
    float r_min = (float)fmax(0.0, d_anchor - tau_eff - sq16_delta);
    float r_max = (float)(d_anchor + tau_eff + sq16_delta);

    int start_m = find_member_lower_bound(cl->members, cl->num_members, r_min);
    int end_m = find_member_upper_bound(cl->members, cl->num_members, r_max);

    telem->level3_annular_pruned +=
        (uint64_t)start_m + (uint64_t)(cl->num_members - end_m);

    int count_m = end_m - start_m;
    if (count_m <= 0)
    {
        return;
    }

    double dcc_home = (home_cluster_id >= 0 && home_cluster_id < M) ?
        model->dcc_matrix[(size_t)home_cluster_id * (size_t)M + (size_t)q] : 0.0;
    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    int sq16_active = (config->use_sq16 && model->sq16_dataset_buffer != NULL &&
                       visited->query_sq16 != NULL);

    if (sq16_active && count_m <= 256)
    {
        Sq16MemberCandidate cand_list[256];
        int num_cands = 0;

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
        }

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;

            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }

            double r_cand = (double)cl->members[m].r_anchor;

            // Primary pivot lower bound: anchor A_q
            double lb1 = fabs(d_anchor - r_cand) - sq16_delta;
            if (lb1 < 0.0)
            {
                lb1 = 0.0;
            }
            if (lb1 >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Secondary pivot lower bound: home anchor A_home
            double lb_home = dcc_home - r_cand - r_home;
            if (lb_home >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            if (config->rlim_cutoff > 0.0 &&
                (lb1 >= config->rlim_cutoff || lb_home >= config->rlim_cutoff))
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Multi-Anchor Pivot Bounding (AESA / LAESA Indexing)
            if (config->use_multi_pivot && *num_pivots > 0)
            {
                int pruned_by_pivot = 0;
                for (int p = 0; p < *num_pivots; p++)
                {
                    int p_cl = pivots[p].cluster_id;
                    if (p_cl == q || p_cl == home_cluster_id)
                    {
                        continue;
                    }
                    double d_qp = pivots[p].d_anchor;
                    double dcc_pq = model->dcc_matrix[(size_t)p_cl * (size_t)M + (size_t)q];
                    if (dcc_pq > 0.0)
                    {
                        double lb_p1 = dcc_pq - d_qp - r_cand;
                        double lb_p2 = d_qp - (dcc_pq + r_cand);
                        double max_lb_p = (lb_p1 > lb_p2) ? lb_p1 : lb_p2;
                        if (max_lb_p >= current_tau / eps_factor ||
                            (config->rlim_cutoff > 0.0 && max_lb_p >= config->rlim_cutoff))
                        {
                            pruned_by_pivot = 1;
                            break;
                        }
                    }
                } // for (int p = 0; ...)

                if (pruned_by_pivot)
                {
                    telem->level3_annular_pruned++;
                    continue;
                }
            }

            const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                                       (size_t)cand_id * (size_t)frame_elem;
            telem->sq16_evaluations++;
            uint64_t ssd = sq16_dist_squared_cutoff_i16(
                visited->query_sq16, cand_sq16, frame_elem, cached_ssd_cutoff
            );

            if (ssd > cached_ssd_cutoff)
            {
                telem->sq16_members_pruned++;
                continue;
            }

            cand_list[num_cands].cand_id = cand_id;
            cand_list[num_cands].ssd = ssd;
            num_cands++;
        } // for (int m = start_m; ...)

        if (num_cands > 1)
        {
            qsort(cand_list, (size_t)num_cands, sizeof(Sq16MemberCandidate),
                  compare_sq16_candidates);
        }

        for (int i = 0; i < num_cands; i++)
        {
            long cand_id = cand_list[i].cand_id;

            current_tau = knn_heap_peek_max_dist(heap);
            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }

            if (cand_list[i].ssd > cached_ssd_cutoff)
            {
                telem->sq16_members_pruned += (uint64_t)(num_cands - i);
                break;
            }

            if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
            {
                telem->reciprocal_reused++;
                continue;
            }

            const void *cand_ptr = NULL;
            if (reader->memory_data != NULL)
            {
                cand_ptr = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_ptr = cand_buffer;
            }

            if (cand_ptr != NULL)
            {
                telem->framedist_calls++;
                double c_tau = (config->rlim_cutoff > 0.0 && config->rlim_cutoff < current_tau)
                               ? config->rlim_cutoff
                               : current_tau;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                );
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            }
        } // for (int i = 0; ...)

        return;
    } // if (sq16_active && count_m <= 256)

    for (int m = start_m; m < end_m; m++)
    {
        long cand_id = (long)cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }

        current_tau = knn_heap_peek_max_dist(heap);
        double r_cand = (double)cl->members[m].r_anchor;

        // Primary pivot lower bound: anchor A_q
        double lb1 = fabs(d_anchor - r_cand) - sq16_delta;
        if (lb1 < 0.0)
        {
            lb1 = 0.0;
        }
        if (lb1 >= current_tau / eps_factor)
        {
            telem->level3_annular_pruned++;
            continue;
        }

        // Secondary pivot lower bound: home anchor A_home
        double lb_home = dcc_home - r_cand - r_home;
        if (lb_home >= current_tau / eps_factor)
        {
            telem->level3_annular_pruned++;
            continue;
        }

        // Multi-Anchor Pivot Bounding (AESA / LAESA Indexing)
        if (config->use_multi_pivot && *num_pivots > 0)
        {
            int pruned_by_pivot = 0;
            for (int p = 0; p < *num_pivots; p++)
            {
                int p_cl = pivots[p].cluster_id;
                if (p_cl == q || p_cl == home_cluster_id)
                {
                    continue;
                }
                double d_qp = pivots[p].d_anchor;
                double dcc_pq = model->dcc_matrix[(size_t)p_cl * (size_t)M + (size_t)q];
                if (dcc_pq > 0.0)
                {
                    double lb_p1 = dcc_pq - d_qp - r_cand;
                    double lb_p2 = d_qp - (dcc_pq + r_cand);
                    double max_lb_p = (lb_p1 > lb_p2) ? lb_p1 : lb_p2;
                    if (max_lb_p >= current_tau / eps_factor ||
                        (config->rlim_cutoff > 0.0 && max_lb_p >= config->rlim_cutoff))
                    {
                        pruned_by_pivot = 1;
                        break;
                    }
                }
            } // for (int p = 0; ...)

            if (pruned_by_pivot)
            {
                telem->level3_annular_pruned++;
                continue;
            }
        }

        if (config->rlim_cutoff > 0.0 &&
            (lb1 >= config->rlim_cutoff || lb_home >= config->rlim_cutoff))
        {
            telem->level3_annular_pruned++;
            continue;
        }

        if (sq16_active)
        {
            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }
            if (is_member_pruned_by_sq16_cached(
                    visited->query_sq16, cand_id, cached_ssd_cutoff, model, telem
                ))
            {
                continue;
            }
        }
        else if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                          model, config, telem))
        {
            continue;
        }

        if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                    model, config, telem))
        {
            continue;
        }

        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }

        // Level 4: Exact Distance Evaluation
        if (!config->use_batch_dist)
        {
            const void *cand_ptr = NULL;
            if (reader->memory_data != NULL)
            {
                cand_ptr = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_ptr = cand_buffer;
            }
            if (cand_ptr != NULL)
            {
                telem->framedist_calls++;
                double c_tau = (config->rlim_cutoff > 0.0 && config->rlim_cutoff < current_tau)
                               ? config->rlim_cutoff
                               : current_tau;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                );
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            }
            continue;
        }

        const void *dest = NULL;
        if (reader->memory_data != NULL)
        {
            dest = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
        }
        else
        {
            void *buf_dest = (char *)cand_buffer + (size_t)(*batch_count) * frame_bytes;
            if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
            {
                dest = buf_dest;
            }
        }
        if (dest != NULL)
        {
            batch_cand_ids[*batch_count] = cand_id;
            batch_ptrs[*batch_count] = dest;
            (*batch_count)++;
            if (*batch_count == 4)
            {
                double dists[4];
                if (model->is_double)
                {
                    framedist_batch_1x4_double(
                        (const double *)query_data,
                        (const double *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                else
                {
                    framedist_batch_1x4_float(
                        (const float *)query_data,
                        (const float *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                telem->framedist_calls += 4;
                for (int b = 0; b < 4; b++)
                {
                    record_neighbor_and_reciprocal(
                        query_id, batch_cand_ids[b], dists[b],
                        config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
                *batch_count = 0;
            }
        }
    } // for (int m = start_m; ...)
}

/**
 * knn_search_inter_clusters() - Evaluate inter-cluster candidate members.
 * @query_id:          Index of query frame.
 * @query_data:        Query frame pixel data.
 * @home_cluster_id:   Home cluster index.
 * @r_home:            Distance to home anchor.
 * @num_cand_clusters: Number of candidate clusters.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            KnnFrameReader context.
 * @cand_buffer:       Candidate pixel buffer.
 * @scores_buffer:     Candidate cluster records.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Array of all heaps.
 * @bucket_locks:      OpenMP bucket locks.
 * @pivots:            Pivot array.
 * @num_pivots:        Pointer to pivot count.
 * @visited:           Per-query frame visited tracker.
 * @telem:             Telemetry record.
 */
static void knn_search_inter_clusters(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    double                 r_home,
    int                    num_cand_clusters,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;

    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        double current_tau = knn_heap_peek_max_dist(heap);

        // Level 1: Cluster-level DCC bound
        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // Multi-Point TE4 Triangulation against measured pivots
        const KnnCluster *cl = &model->clusters[q];
        if (num_pivots != NULL && *num_pivots >= 2)
        {
            int pruned_by_te4 = 0;
            int n_p = (*num_pivots > 8) ? 8 : *num_pivots;
            for (int p1 = 0; p1 < n_p - 1; p1++)
            {
                int    c1 = pivots[p1].cluster_id;
                double d1 = pivots[p1].d_anchor;
                for (int p2 = p1 + 1; p2 < n_p; p2++)
                {
                    int    c2 = pivots[p2].cluster_id;
                    double d2 = pivots[p2].d_anchor;
                    double d12 = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)c2];
                    double d1q = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)q];
                    double d2q = model->dcc_matrix[(size_t)c2 * (size_t)M + (size_t)q];

                    double min_d = calc_min_dist_4pt(d1, d2, d12, d1q, d2q);
                    if (min_d - cl->radius - 1e-5 >= current_tau / eps_factor)
                    {
                        pruned_by_te4 = 1;
                        break;
                    }
                    if (min_d - cl->radius > lb_cluster)
                    {
                        lb_cluster = min_d - cl->radius;
                        scores_buffer[idx].lb = lb_cluster;
                    }
                }
                if (pruned_by_te4)
                {
                    break;
                }
            }
            if (pruned_by_te4)
            {
                telem->level1_clusters_pruned++;
                continue;
            }
        }

        // Level 2: Query-to-Anchor evaluation
        int    anchor_is_sq16 = 0;
        double d_anchor = knn_compute_anchor_distance(
            query_data, q, model, config, visited, &anchor_is_sq16, telem
        );

        if (!anchor_is_sq16)
        {
            // Dynamic Bound Tightening (Multi-Pivot)
            if (config->use_multi_pivot)
            {
                for (int j = idx + 1; j < num_cand_clusters; j++)
                {
                    int other_q = scores_buffer[j].id;
                    double dcc_pivot = model->dcc_matrix[(size_t)q * (size_t)M + (size_t)other_q];
                    if (dcc_pivot > 0.0)
                    {
                        double r_other = model->clusters[other_q].radius;
                        double lb_pivot = dcc_pivot - d_anchor - r_other;
                        if (lb_pivot > scores_buffer[j].lb)
                        {
                            scores_buffer[j].lb = lb_pivot;
                        }
                    }
                } // for (int j = idx + 1; ...)
            }

            // Record measured anchor pivot for Multi-Anchor Pivot Bounding (AESA)
            if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS)
            {
                int already_present = 0;
                for (int p = 0; p < *num_pivots; p++)
                {
                    if (pivots[p].cluster_id == q)
                    {
                        already_present = 1;
                        break;
                    }
                }
                if (!already_present)
                {
                    pivots[*num_pivots].cluster_id = q;
                    pivots[*num_pivots].d_anchor = d_anchor;
                    (*num_pivots)++;
                }
            }
        }

        double sq_err = anchor_is_sq16 ? (double)model->sq16_params.err_radius : 0.0;
        double lb_anchor = d_anchor - cl->radius - sq_err;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
            continue;
        }

        // Level 3 & 4: Evaluate candidate cluster members
        knn_eval_candidate_cluster_members(
            query_id, query_data, q, d_anchor, anchor_is_sq16,
            home_cluster_id, r_home, model, config, reader, cand_buffer,
            heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, num_pivots, visited,
            &batch_count, batch_cand_ids, batch_ptrs, telem
        );
    } // for (int idx = 0; idx < num_cand_clusters; idx++)

    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
}

/**
 * knn_search_cluster_graph() - Execute best-first graph routing over cluster proximity graph.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @scratch:         Thread-local cluster graph scratch buffers.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all heaps.
 * @bucket_locks:    OpenMP bucket locks.
 * @pivots:          Pivot array.
 * @num_pivots:      Pointer to pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_search_cluster_graph(
    long                    query_id,
    const void *restrict    query_data,
    int                     home_cluster_id,
    double                  r_home,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnClusterGraphScratch *scratch,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    MeasuredPivot          *pivots,
    int                    *num_pivots,
    KnnVisitedTracker      *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    int k_adj = model->cluster_graph_k;
    double eps_factor = 1.0 + config->epsilon;
    double rlim = (config->rlim_cutoff > 0.0) ? config->rlim_cutoff : model->model_rlim;
    if (rlim <= 0.0)
    {
        rlim = 1e9;
    }

    scratch->enqueued_epoch++;
    if (scratch->enqueued_epoch == 0)
    {
        memset(scratch->enqueued_tags, 0, (size_t)M * sizeof(uint32_t));
        scratch->enqueued_epoch = 1;
    }
    uint32_t epoch = scratch->enqueued_epoch;

    int pq_size = 0;
    ClusterPqNode *pq = scratch->pq;

    /* Mark home cluster as enqueued so it is not re-evaluated */
    scratch->enqueued_tags[home_cluster_id] = epoch;
    scratch->anchor_dists[home_cluster_id] = r_home;
    scratch->anchor_is_sq16[home_cluster_id] = 0;

    /* Mark any clusters already evaluated in warm-start / pivots */
    if (num_pivots != NULL)
    {
        for (int p = 0; p < *num_pivots; p++)
        {
            int p_cl = pivots[p].cluster_id;
            if (p_cl >= 0 && p_cl < M)
            {
                scratch->enqueued_tags[p_cl] = epoch;
                scratch->anchor_dists[p_cl] = pivots[p].d_anchor;
                scratch->anchor_is_sq16[p_cl] = 0;
            }
        }
    }

    /* Seed priority queue with graph neighbors of home cluster and pivots */
    int n_seed_pivots = (num_pivots != NULL) ? *num_pivots : 1;
    for (int p = 0; p < n_seed_pivots; p++)
    {
        int p_cl = (num_pivots != NULL) ? pivots[p].cluster_id : home_cluster_id;
        if (p_cl < 0 || p_cl >= M)
        {
            continue;
        }
        const int *adj = model->cluster_graph_adj + (size_t)p_cl * (size_t)k_adj;
        for (int i = 0; i < k_adj; i++)
        {
            int nbr = adj[i];
            if (nbr < 0 || nbr >= M || scratch->enqueued_tags[nbr] == epoch)
            {
                continue;
            }
            scratch->enqueued_tags[nbr] = epoch;

            int is_sq16 = 0;
            double d_nbr = knn_compute_anchor_distance(
                query_data, nbr, model, config, visited, &is_sq16, telem
            );
            scratch->anchor_dists[nbr] = d_nbr;
            scratch->anchor_is_sq16[nbr] = (uint8_t)is_sq16;
            cluster_pq_push(pq, &pq_size, nbr, d_nbr);
        }
    }

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;

    int ef_limit = (config->ef_cluster > 0) ? config->ef_cluster : 60;
    if (ef_limit > M)
    {
        ef_limit = M;
    }
    int clusters_evaluated = 0;

    while (pq_size > 0 && clusters_evaluated < ef_limit)
    {
        ClusterPqNode top = cluster_pq_pop(pq, &pq_size);
        int c = top.cluster_id;
        double d_anchor = top.dist;
        int anchor_is_sq16 = (int)scratch->anchor_is_sq16[c];

        double current_tau = knn_heap_peek_max_dist(heap);
        const KnnCluster *cl = &model->clusters[c];

        double sq_err = anchor_is_sq16 ? (double)model->sq16_params.err_radius : 0.0;
        double lb_anchor = d_anchor - cl->radius - sq_err;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        /* Early termination: if heap is full and anchor distance is far beyond tau + rlim */
        if (heap->count >= heap->k && clusters_evaluated >= 20 &&
            d_anchor > current_tau / eps_factor + rlim)
        {
            break;
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
        }
        else
        {
            clusters_evaluated++;
            telem->clusters_graph_evaluated++;

            /* Record measured anchor pivot if useful */
            if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS && !anchor_is_sq16)
            {
                int already_present = 0;
                for (int p = 0; p < *num_pivots; p++)
                {
                    if (pivots[p].cluster_id == c)
                    {
                        already_present = 1;
                        break;
                    }
                }
                if (!already_present)
                {
                    pivots[*num_pivots].cluster_id = c;
                    pivots[*num_pivots].d_anchor = d_anchor;
                    (*num_pivots)++;
                }
            }

            /* Evaluate members of cluster c */
            knn_eval_candidate_cluster_members(
                query_id, query_data, c, d_anchor, anchor_is_sq16,
                home_cluster_id, r_home, model, config, reader,
                cand_buffer, heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                pivots, num_pivots, visited,
                &batch_count, batch_cand_ids, batch_ptrs, telem
            );
        }

        /* Expand neighbors into priority queue if within search horizon */
        if (d_anchor <= current_tau / eps_factor + rlim)
        {
            const int *adj = model->cluster_graph_adj + (size_t)c * (size_t)k_adj;
            for (int i = 0; i < k_adj; i++)
            {
                int nbr = adj[i];
                if (nbr < 0 || nbr >= M || scratch->enqueued_tags[nbr] == epoch)
                {
                    continue;
                }
                scratch->enqueued_tags[nbr] = epoch;

                int is_sq16 = 0;
                double d_nbr = knn_compute_anchor_distance(
                    query_data, nbr, model, config, visited, &is_sq16, telem
                );
                scratch->anchor_dists[nbr] = d_nbr;
                scratch->anchor_is_sq16[nbr] = (uint8_t)is_sq16;
                cluster_pq_push(pq, &pq_size, nbr, d_nbr);
            }
        }
    } // while (pq_size > 0 && ...)

    /* Flush any pending batched distance computations */
    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count, dists, frame_elem
            );
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count, dists, frame_elem
            );
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
}

/**
 * knn_search_single_frame() - Execute multi-level metric pruned search for one query frame.
 * @query_id:       Index of query frame.
 * @query_data:     Pixel buffer of query frame.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @reader:         Thread-local KnnFrameReader.
 * @cand_buffer:    Scratch buffer for candidate frame pixels.
 * @scores_buffer:  Scratch buffer for cluster sorting.
 * @graph_scratch:  Scratch buffers for cluster graph search.
 * @all_heaps:      Array of KnnMaxHeap structures for all frames.
 * @bucket_locks:   Array of OpenMP bucket locks (if OpenMP enabled).
 * @visited:        Per-query frame visited tracker.
 * @telem:          Thread-local KnnTelemetry.
 */
static void knn_search_single_frame(
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
    KnnTelemetry  *restrict telem)
{
    KnnMaxHeap *heap = &all_heaps[query_id];
    int home_cluster_id = model->frame_cluster_map[query_id];
    double r_home = (double)model->frame_r_anchor[query_id];
    int M = model->num_clusters;

    MeasuredPivot pivots[MAX_MEASURED_PIVOTS];
    int           num_pivots = 0;
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        pivots[num_pivots].cluster_id = home_cluster_id;
        pivots[num_pivots].d_anchor = r_home;
        num_pivots++;
    }

    // Step 1: Intra-Cluster Search (Home cluster c_p)
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        knn_search_intra_cluster(
            query_id, query_data, home_cluster_id, r_home, model, config,
            reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, telem
        );
    }

    // Fast tau-Contraction: Warm-start tau by searching closest neighbor cluster
    knn_warm_start_nearest_cluster(
        query_id, query_data, home_cluster_id, model, config, reader,
        cand_buffer, heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        pivots, &num_pivots, visited, telem
    );

    if (config->approx_mode)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        return;
    }

    // Step 2: Search remaining clusters (graph-guided routing or linear inter-cluster scan)
    if (config->use_cluster_graph && model->cluster_graph_adj != NULL &&
        model->cluster_graph_k > 0 && home_cluster_id >= 0 && home_cluster_id < M)
    {
        knn_search_cluster_graph(
            query_id, query_data, home_cluster_id, r_home, model, config,
            reader, cand_buffer, graph_scratch, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }
    else
    {
        int num_cand_clusters = knn_score_candidate_clusters(
            home_cluster_id, pivots, num_pivots, r_home, model, config,
            heap, scores_buffer, telem
        );

        knn_search_inter_clusters(
            query_id, query_data, home_cluster_id, r_home, num_cand_clusters,
            model, config, reader, cand_buffer, scores_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }
}

/**
 * knn_cross_seed_frontier() - Populate dynamic search frontier with nearest evaluated anchor seeds.
 * @loc_res:         Cluster locator result with evaluated anchors.
 * @best_c:          Best anchor cluster ID.
 * @query_data:      Query frame pixel data.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @cand_reader:     Candidate frame reader.
 * @cand_buffer:     Candidate frame pixel buffer.
 * @heap:            Max-heap for current query.
 * @frontier:        Output frontier node array.
 * @frontier_count:  Pointer to frontier count.
 * @best_seed_id:    Pointer to best seed frame ID.
 * @best_seed_dist:  Pointer to best seed distance.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_cross_seed_frontier(
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
    KnnTelemetry *restrict      telem)
{
    int sorted_eval_clusters[32];
    double sorted_eval_dists[32];
    int num_eval = loc_res->num_evaluated_anchors;
    if (num_eval > 32)
    {
        num_eval = 32;
    }
    for (int e = 0; e < num_eval; e++)
    {
        sorted_eval_clusters[e] = loc_res->evaluated_clusters[e];
        sorted_eval_dists[e] = loc_res->evaluated_dists[e];
    }
    for (int i = 0; i < num_eval - 1; i++)
    {
        for (int j = i + 1; j < num_eval; j++)
        {
            if (sorted_eval_dists[j] < sorted_eval_dists[i])
            {
                double tmp_d = sorted_eval_dists[i];
                sorted_eval_dists[i] = sorted_eval_dists[j];
                sorted_eval_dists[j] = tmp_d;
                int tmp_c = sorted_eval_clusters[i];
                sorted_eval_clusters[i] = sorted_eval_clusters[j];
                sorted_eval_clusters[j] = tmp_c;
            }
        }
    }

    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int max_frontier_seeds = (config->approx_mode) ? 2 : 8;
    for (int e = 0; e < num_eval && *frontier_count < max_frontier_seeds; e++)
    {
        int anc_c = sorted_eval_clusters[e];
        if (anc_c >= 0 && anc_c < M && anc_c != best_c)
        {
            const KnnCluster *anc_cl = &model->clusters[anc_c];
            if (anc_cl->num_members > 0)
            {
                for (int mi = 0; mi < anc_cl->num_members && mi < 2; mi++)
                {
                    long s_frame = (long)anc_cl->members[mi].frame_id;
                    if (knn_visited_check_and_mark(visited, s_frame))
                    {
                        continue;
                    }
                    if (!knn_heap_contains(heap, (int)s_frame))
                    {
                        double tau = knn_heap_peek_max_dist(heap);
                        if (best_seed_dist != NULL && *best_seed_dist > tau)
                        {
                            tau = *best_seed_dist;
                        }
                        if (is_graph_pruned_by_sq16(visited->query_sq16, s_frame, tau,
                                                    model, config, telem) ||
                            is_graph_pruned_by_sq8(visited->query_sq8, s_frame, tau,
                                                   model, config, telem))
                        {
                            continue;
                        }

                        if (knn_reader_read_frame(cand_reader, s_frame, cand_buffer) == 0)
                        {
                            telem->framedist_calls++;
                            telem->graph_seeds_evaluated++;
                            double d = compute_euclidean_distance(
                                query_data, cand_buffer, frame_elem, model->is_double
                            );
                            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                            {
                                knn_heap_push(heap, (int)s_frame, d);
                            }
                            if (d < *best_seed_dist)
                            {
                                *best_seed_id = s_frame;
                                *best_seed_dist = d;
                            }
                            if (frontier_count != NULL && *frontier_count < GRAPH_FRONTIER_MAX)
                            {
                                frontier[*frontier_count].frame_id = s_frame;
                                frontier[*frontier_count].dist = d;
                                frontier[*frontier_count].parent_id = -1;
                                frontier[*frontier_count].parent_dist = -1.0;
                                frontier[*frontier_count].expanded = 0;
                                (*frontier_count)++;
                            }
                        }
                    }
                } // for (int mi = 0; ...)
            }
        }
    } // for (int e = 0; ...)
}

/**
 * knn_greedy_route_to_basin() - Fast Phase 1 greedy 1-NN routing along graph.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_greedy_route_to_basin(
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
    KnnTelemetry *restrict telem)
{
    long curr_u = *best_seed_id;
    double curr_d = *best_seed_dist;
    long parent_u = (last_parent_id != NULL) ? *last_parent_id : -1;
    double parent_d = (last_parent_dist != NULL) ? *last_parent_dist : -1.0;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    int max_hops = 16;
    int check_k = (graph_k < 16) ? graph_k : 16;

    for (int hop = 0; hop < max_hops; hop++)
    {
        if (curr_d <= 1e-9)
        {
            break;
        }

        const uint32_t *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];
        int improved = 0;
        long next_u = curr_u;
        double next_d = curr_d;

        int cand_order[16];
        float cand_alignment[16];
        for (int i = 0; i < check_k; i++)
        {
            cand_order[i] = i;
            cand_alignment[i] = 0.0f;
        }

        // Inherit direction from parent hop to prioritize forward descent
        if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
            parent_u >= 0 && parent_d > 1e-9 && curr_d > 1e-9)
        {
            int p_idx = -1;
            for (int k = 0; k < graph_k; k++)
            {
                if ((long)neighbors[k] == parent_u)
                {
                    p_idx = k;
                    break;
                }
            }

            if (p_idx >= 0)
            {
                double d_edge_p = (double)n_dists[p_idx];
                if (d_edge_p > 1e-9)
                {
                    double cos_alpha = (curr_d * curr_d + d_edge_p * d_edge_p -
                                        parent_d * parent_d) /
                                       (2.0 * curr_d * d_edge_p);
                    cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                    double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                    for (int m = 0; m < check_k; m++)
                    {
                        if (m == p_idx)
                        {
                            cand_alignment[m] = -1.0f;
                            continue;
                        }

                        double r_m = (double)n_dists[m];
                        float d_mut = knn_get_mutual_dist(model, curr_u, p_idx, m);
                        if (d_mut > 0.0f && r_m > 1e-9)
                        {
                            double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                              (double)d_mut * d_mut) /
                                             (2.0 * d_edge_p * r_m);
                            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                            double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                            cand_alignment[m] = (float)(cos_alpha * cos_phi +
                                                        sin_alpha * sin_phi);
                        }
                    }

                    for (int i = 0; i < check_k - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < check_k; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                }
            }
        }

        for (int step = 0; step < check_k; step++)
        {
            int k_idx = cand_order[step];
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            if (knn_visited_is_visited(visited, nb_id))
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_d - d_edge);
            if (lb_edge >= curr_d)
            {
                telem->graph_edges_pruned++;
                continue;
            }

            double routing_tau = next_d;
            double tau_heap = knn_heap_peek_max_dist(heap);
            if (tau_heap > routing_tau)
            {
                routing_tau = tau_heap;
            }
            if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, routing_tau,
                                        model, config, telem) ||
                is_graph_pruned_by_sq8(visited->query_sq8, nb_id, routing_tau,
                                       model, config, telem))
            {
                continue;
            }

            knn_visited_mark(visited, nb_id);

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_dist = d;
                    *best_seed_id = nb_id;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (d < next_d)
                {
                    next_u = nb_id;
                    next_d = d;
                    improved = 1;
                    break;
                }
            }
        } // for (int step = 0; ...)

        if (!improved)
        {
            break;
        }

        parent_u = curr_u;
        parent_d = curr_d;
        curr_u = next_u;
        curr_d = next_d;
    } // for (int hop = 0; ...)

    *best_seed_id = curr_u;
    *best_seed_dist = curr_d;
    if (last_parent_id != NULL)
    {
        *last_parent_id = parent_u;
    }
    if (last_parent_dist != NULL)
    {
        *last_parent_dist = parent_d;
    }
}

/**
 * knn_direct_basin_expansion() - Direct Option B1 neighbor expansion around 1-NN basin.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID (u*).
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_direct_basin_expansion(
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
    KnnTelemetry *restrict telem)
{
    long center_u = *best_seed_id;
    double center_d = *best_seed_dist;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    const uint32_t *neighbors = &model->graph_indices[center_u * (long)graph_k];
    const float    *n_dists = &model->graph_distances[center_u * (long)graph_k];

    float local_lb2[64];
    int cand_order[64];
    float cand_alignment[64];
    int eval_lim = (graph_k < 64) ? graph_k : 64;

    for (int i = 0; i < eval_lim; i++)
    {
        cand_order[i] = i;
        cand_alignment[i] = 0.0f;
        local_lb2[i] = 0.0f;
    }

    int first_eval_done = 0;

    // Option 1: Inter-Node Direction Inheritance from greedy routing hop
    if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
        parent_id >= 0 && parent_dist > 1e-9 && center_d > 1e-9)
    {
        int p_idx = -1;
        for (int k = 0; k < graph_k; k++)
        {
            if ((long)neighbors[k] == parent_id)
            {
                p_idx = k;
                break;
            }
        }

        if (p_idx >= 0)
        {
            double d_edge_p = (double)n_dists[p_idx];
            if (d_edge_p > 1e-9)
            {
                double cos_alpha = (center_d * center_d + d_edge_p * d_edge_p -
                                    parent_dist * parent_dist) /
                                   (2.0 * center_d * d_edge_p);
                cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                for (int m = 0; m < eval_lim; m++)
                {
                    if (m == p_idx)
                    {
                        cand_alignment[m] = -1.0f;
                        continue;
                    }

                    double r_m = (double)n_dists[m];
                    float d_mut = knn_get_mutual_dist(model, center_u, p_idx, m);
                    if (d_mut > 0.0f && r_m > 1e-9)
                    {
                        double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                          (double)d_mut * d_mut) /
                                         (2.0 * d_edge_p * r_m);
                        cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                        double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                        double cos_th = cos_alpha * cos_phi + sin_alpha * sin_phi;
                        cand_alignment[m] = (float)fmin(1.0, cos_th);

                        double lb2_ang = center_d * center_d + r_m * r_m -
                                         2.0 * center_d * r_m * fmin(1.0, cos_th);
                        double lb_tri = fabs(parent_dist - (double)d_mut);
                        double lb2_tri = lb_tri * lb_tri;
                        local_lb2[m] = (float)fmax(lb2_ang, lb2_tri);
                    }
                }

                for (int i = 0; i < eval_lim - 1; i++)
                {
                    int best_p = i;
                    float max_a = cand_alignment[cand_order[i]];
                    for (int j = i + 1; j < eval_lim; j++)
                    {
                        if (cand_alignment[cand_order[j]] > max_a)
                        {
                            max_a = cand_alignment[cand_order[j]];
                            best_p = j;
                        }
                    }
                    if (best_p != i)
                    {
                        int tmp = cand_order[i];
                        cand_order[i] = cand_order[best_p];
                        cand_order[best_p] = tmp;
                    }
                }
                first_eval_done = 1;
            }
        }
    }

    long improving_node = -1;
    int non_improving_streak = 0;
    const int max_streak = 8;

    for (int step = 0; step < eval_lim; step++)
    {
        int k_idx = cand_order[step];
        long nb_id = (long)neighbors[k_idx];
        if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == center_u)
        {
            continue;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_eff = current_tau / eps_factor;
        double tau_eff2 = tau_eff * tau_eff;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
        }

        if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
            (double)local_lb2[k_idx] >= tau_eff2)
        {
            telem->angular_pruned++;
            if (heap->count >= heap->k)
            {
                non_improving_streak++;
                if (non_improving_streak >= max_streak)
                {
                    break;
                }
            }
            continue;
        }

        if (knn_visited_check_and_mark(visited, nb_id))
        {
            continue;
        }

        double d_edge = (double)n_dists[k_idx];
        double lb_edge = fabs(center_d - d_edge);
        if (heap->count >= heap->k &&
            (lb_edge >= current_tau / eps_factor ||
             (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
        {
            telem->graph_edges_pruned++;
            non_improving_streak++;
            if (non_improving_streak >= max_streak)
            {
                break;
            }
            continue;
        }

        if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
                                    model, config, telem) ||
            is_graph_pruned_by_sq8(visited->query_sq8, nb_id, current_tau,
                                   model, config, telem))
        {
            if (heap->count >= heap->k)
            {
                non_improving_streak++;
                if (non_improving_streak >= max_streak)
                {
                    break;
                }
            }
            continue;
        }

        if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
        {
            telem->framedist_calls++;
            telem->graph_seeds_evaluated++;
            double d = compute_euclidean_distance(
                query_data, cand_buffer, frame_elem, model->is_double
            );
            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
            {
                knn_heap_push(heap, (int)nb_id, d);
            }
            if (heap->count >= heap->k)
            {
                if (d < current_tau)
                {
                    non_improving_streak = 0;
                }
                else
                {
                    non_improving_streak++;
                    if (non_improving_streak >= max_streak)
                    {
                        break;
                    }
                }
            }
            if (d < *best_seed_dist)
            {
                *best_seed_dist = d;
                *best_seed_id = nb_id;
                improving_node = nb_id;
            }
            if (*num_seed_pivots < 8)
            {
                seed_pivot_ids[*num_seed_pivots] = nb_id;
                seed_pivot_dists[*num_seed_pivots] = d;
                (*num_seed_pivots)++;
            }

            if (model->graph_mutual_dists != NULL)
            {
                for (int rem = step + 1; rem < eval_lim; rem++)
                {
                    int m_idx = cand_order[rem];
                    if (m_idx < 64)
                    {
                        double r_m = (double)n_dists[m_idx];
                        float d_mut = knn_get_mutual_dist(
                            model, center_u, k_idx, m_idx
                        );
                        if (d_mut > 0.0f)
                        {
                            // Opportunity 3: Mutual distance triangle lower bound
                            double lb_tri = fabs(d - (double)d_mut);
                            double lb2_tri = lb_tri * lb_tri;
                            if (lb2_tri > (double)local_lb2[m_idx])
                            {
                                local_lb2[m_idx] = (float)lb2_tri;
                            }

                            if (config->use_angular_bound)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    center_d, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }
                            }

                            if (!first_eval_done && config->use_angular_bound &&
                                center_d > 1e-9 && d_edge > 1e-9 && r_m > 1e-9)
                            {
                                double cos_th = (center_d * center_d +
                                                 d_edge * d_edge - d * d) /
                                                (2.0 * center_d * d_edge);
                                cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                  (double)d_mut * d_mut) /
                                                 (2.0 * d_edge * r_m);
                                cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                sin_th * sin_phi);
                            }
                        }
                    }
                } // for (int rem = step + 1; ...)

                if (!first_eval_done)
                {
                    first_eval_done = 1;
                    for (int i = step + 1; i < eval_lim - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < eval_lim; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                }
            } // if (config->use_angular_bound ...)
        }
    } // for (int step = 0; ...)

    // Phase 2b: Also check top neighbors of the closest discovered nodes in heap
    int top_seeds[3];
    int num_top_seeds = 0;
    if (improving_node >= 0)
    {
        top_seeds[num_top_seeds++] = (int)improving_node;
    }

    for (int i = 0; i < heap->count && num_top_seeds < 3; i++)
    {
        int cand = heap->data[i].frame_id;
        if (cand != (int)center_u && (num_top_seeds == 0 || cand != top_seeds[0]))
        {
            top_seeds[num_top_seeds++] = cand;
        }
    }

    for (int ts = 0; ts < num_top_seeds; ts++)
    {
        long s_node = (long)top_seeds[ts];
        if (s_node >= 0 && s_node < model->total_dataset_frames)
        {
            double s_dist = -1.0;
            for (int h = 0; h < heap->count; h++)
            {
                if (heap->data[h].frame_id == (int)s_node)
                {
                    s_dist = heap->data[h].dist;
                    break;
                }
            }

            const uint32_t *s_nb = &model->graph_indices[s_node * (long)graph_k];
            const float    *s_dst = &model->graph_distances[s_node * (long)graph_k];
            int check_k = (graph_k < 12) ? graph_k : 12;

            int cand_order_s[12];
            float cand_alignment_s[12];
            float local_lb2_s[12];
            for (int i = 0; i < check_k; i++)
            {
                cand_order_s[i] = i;
                cand_alignment_s[i] = 0.0f;
                local_lb2_s[i] = 0.0f;
            }

            if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
                center_u >= 0 && center_d > 1e-9 && s_dist > 1e-9)
            {
                int p_idx = -1;
                for (int k = 0; k < graph_k; k++)
                {
                    if ((long)s_nb[k] == center_u)
                    {
                        p_idx = k;
                        break;
                    }
                }

                if (p_idx >= 0)
                {
                    double d_edge_p = (double)s_dst[p_idx];
                    if (d_edge_p > 1e-9)
                    {
                        double cos_alpha = (s_dist * s_dist + d_edge_p * d_edge_p -
                                            center_d * center_d) /
                                           (2.0 * s_dist * d_edge_p);
                        cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                        double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                        for (int m = 0; m < check_k; m++)
                        {
                            if (m == p_idx)
                            {
                                cand_alignment_s[m] = -1.0f;
                                continue;
                            }

                            double r_m = (double)s_dst[m];
                            float d_mut = knn_get_mutual_dist(model, s_node, p_idx, m);
                            if (d_mut > 0.0f && r_m > 1e-9)
                            {
                                double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                                  (double)d_mut * d_mut) /
                                                 (2.0 * d_edge_p * r_m);
                                cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                double cos_th = cos_alpha * cos_phi + sin_alpha * sin_phi;
                                cand_alignment_s[m] = (float)fmin(1.0, cos_th);

                                double lb2_ang = s_dist * s_dist + r_m * r_m -
                                                 2.0 * s_dist * r_m * fmin(1.0, cos_th);
                                double lb_tri = fabs(center_d - (double)d_mut);
                                double lb2_tri = lb_tri * lb_tri;
                                local_lb2_s[m] = (float)fmax(lb2_ang, lb2_tri);
                            }
                        }

                        for (int i = 0; i < check_k - 1; i++)
                        {
                            int best_p = i;
                            float max_a = cand_alignment_s[cand_order_s[i]];
                            for (int j = i + 1; j < check_k; j++)
                            {
                                if (cand_alignment_s[cand_order_s[j]] > max_a)
                                {
                                    max_a = cand_alignment_s[cand_order_s[j]];
                                    best_p = j;
                                }
                            }
                            if (best_p != i)
                            {
                                int tmp = cand_order_s[i];
                                cand_order_s[i] = cand_order_s[best_p];
                                cand_order_s[best_p] = tmp;
                            }
                        }
                    }
                }
            }

            for (int step = 0; step < check_k; step++)
            {
                int i = cand_order_s[step];
                long nb2 = (long)s_nb[i];
                if (nb2 < 0 || nb2 >= model->total_dataset_frames)
                {
                    continue;
                }

                double current_tau = knn_heap_peek_max_dist(heap);
                double tau_eff = current_tau / eps_factor;
                double tau_eff2 = tau_eff * tau_eff;
                if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
                {
                    tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
                }

                if ((heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
                    (double)local_lb2_s[i] >= tau_eff2)
                {
                    telem->angular_pruned++;
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb2))
                {
                    continue;
                }

                double d_edge = (double)s_dst[i];
                if (heap->count >= heap->k && d_edge >= tau_eff)
                {
                    telem->graph_edges_pruned++;
                    continue;
                }

                if (is_graph_pruned_by_sq16(visited->query_sq16, nb2, current_tau,
                                            model, config, telem) ||
                    is_graph_pruned_by_sq8(visited->query_sq8, nb2, current_tau,
                                           model, config, telem))
                {
                    continue;
                }

                if (knn_reader_read_frame(cand_reader, nb2, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    telem->graph_seeds_evaluated++;
                    double d2 = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d2 <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)nb2, d2);
                    }
                    if (d2 < *best_seed_dist)
                    {
                        *best_seed_dist = d2;
                        *best_seed_id = nb2;
                    }
                }
            } // for (int step = 0; ...)
        }
    } // for (int ts = 0; ...)
}

/**
 * knn_cross_explore_graph_frontier() - Explore KNN graph frontier for cross-dataset query.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @heap:             Max-heap for current query.
 * @frontier:         Frontier node array.
 * @frontier_count:   Frontier node count.
 * @best_seed_id:     Pointer to best seed ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed pivot IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Return: 1 if global containment criterion was satisfied, 0 otherwise.
 */
static int knn_cross_explore_graph_frontier(
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
    KnnTelemetry *restrict telem)
{
    int graph_k = model->graph_k;
    int target_k = heap->k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int max_expansions = (config->approx_mode) ? 4 : 16;
    int max_eval_k = (config->approx_mode) ? ((graph_k < 12) ? graph_k : 12) : graph_k;
    int expansions_done = 0;
    int stagnation_count = 0;
    double prev_tau = 1e20;

    while ((expansions_done < max_expansions || heap->count < target_k) &&
           expansions_done < GRAPH_FRONTIER_MAX)
    {
        int best_idx = -1;
        double min_f_dist = 1e20;
        for (int fi = 0; fi < frontier_count; fi++)
        {
            if (!frontier[fi].expanded && frontier[fi].dist < min_f_dist)
            {
                min_f_dist = frontier[fi].dist;
                best_idx = fi;
            }
        }

        if (best_idx < 0)
        {
            break;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        if (heap->count >= target_k && min_f_dist > current_tau / eps_factor)
        {
            break;
        }

        if (config->approx_mode && heap->count >= target_k)
        {
            if (current_tau >= prev_tau * 0.9999)
            {
                stagnation_count++;
                if (stagnation_count >= 3)
                {
                    break;
                }
            }
            else
            {
                stagnation_count = 0;
                prev_tau = current_tau;
            }
        }

        frontier[best_idx].expanded = 1;
        expansions_done++;

        long curr_u = frontier[best_idx].frame_id;
        double curr_u_dist = frontier[best_idx].dist;
        long parent_id = frontier[best_idx].parent_id;
        double parent_dist = frontier[best_idx].parent_dist;
        if (curr_u_dist < *best_seed_dist)
        {
            *best_seed_id = curr_u;
            *best_seed_dist = curr_u_dist;
        }
        uint32_t const *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];

        float local_lb2[64];
        int cand_order[64];
        float cand_alignment[64];
        int eval_lim = (max_eval_k < 64) ? max_eval_k : 64;
        for (int i = 0; i < eval_lim; i++)
        {
            cand_order[i] = i;
            cand_alignment[i] = 0.0f;
            local_lb2[i] = 0.0f;
        }

        int first_eval_done = 0;

        // Option 1: Inter-Node Direction Inheritance from parent hop
        if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
            parent_id >= 0 && parent_dist > 1e-9 && curr_u_dist > 1e-9)
        {
            int p_idx = -1;
            for (int k = 0; k < graph_k; k++)
            {
                if ((long)neighbors[k] == parent_id)
                {
                    p_idx = k;
                    break;
                }
            }

            if (p_idx >= 0)
            {
                double d_edge_p = (double)n_dists[p_idx];
                if (d_edge_p > 1e-9)
                {
                    double cos_alpha = (curr_u_dist * curr_u_dist +
                                        d_edge_p * d_edge_p -
                                        parent_dist * parent_dist) /
                                       (2.0 * curr_u_dist * d_edge_p);
                    cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                    double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                    for (int m = 0; m < eval_lim; m++)
                    {
                        if (m == p_idx)
                        {
                            cand_alignment[m] = -1.0f;
                            continue;
                        }

                        double r_m = (double)n_dists[m];
                        float d_mut = knn_get_mutual_dist(
                            model, curr_u, p_idx, m
                        );
                        if (d_mut > 0.0f && r_m > 1e-9)
                        {
                            double cos_phi = (d_edge_p * d_edge_p +
                                              r_m * r_m -
                                              (double)d_mut * d_mut) /
                                             (2.0 * d_edge_p * r_m);
                            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                            double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                            double cos_th = cos_alpha * cos_phi +
                                            sin_alpha * sin_phi;
                            cand_alignment[m] = (float)fmin(1.0, cos_th);

                            double lb2_ang = curr_u_dist * curr_u_dist +
                                             r_m * r_m -
                                             2.0 * curr_u_dist * r_m *
                                             fmin(1.0, cos_th);
                            double lb_tri = fabs(parent_dist - (double)d_mut);
                            double lb2_tri = lb_tri * lb_tri;
                            local_lb2[m] = (float)fmax(lb2_ang, lb2_tri);
                        }
                    }

                    for (int i = 0; i < eval_lim - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < eval_lim; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                    first_eval_done = 1;
                }
            }
        }

        for (int step = 0; step < eval_lim; step++)
        {
            int k_idx = cand_order[step];
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double tau_eff2 = tau_eff * tau_eff;
            if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
            {
                tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
            }

            if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
                (double)local_lb2[k_idx] >= tau_eff2)
            {
                telem->angular_pruned++;
                continue;
            }

            if (knn_visited_check_and_mark(visited, nb_id))
            {
                continue;
            }

            int already_in_frontier = 0;
            for (int fi = 0; fi < frontier_count; fi++)
            {
                if (frontier[fi].frame_id == nb_id)
                {
                    already_in_frontier = 1;
                    break;
                }
            }
            if (already_in_frontier)
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_u_dist - d_edge);
            current_tau = knn_heap_peek_max_dist(heap);

            if (heap->count >= heap->k &&
                (lb_edge >= current_tau / eps_factor ||
                 (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            double sq_tau = current_tau;
            if (best_seed_dist != NULL && *best_seed_dist > sq_tau)
            {
                sq_tau = *best_seed_dist;
            }
            if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, sq_tau,
                                        model, config, telem) ||
                is_graph_pruned_by_sq8(visited->query_sq8, nb_id, sq_tau,
                                       model, config, telem))
            {
                continue;
            }

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_id = nb_id;
                    *best_seed_dist = d;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                {
                    for (int rem = step + 1; rem < eval_lim; rem++)
                    {
                        int m_idx = cand_order[rem];
                        if (m_idx < 64)
                        {
                            double r_m = (double)n_dists[m_idx];
                            float d_mut = knn_get_mutual_dist(
                                model, curr_u, k_idx, m_idx
                            );
                            if (d_mut > 0.0f)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    curr_u_dist, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }

                                if (!first_eval_done && curr_u_dist > 1e-9 &&
                                    d_edge > 1e-9 && r_m > 1e-9)
                                {
                                    double cos_th = (curr_u_dist * curr_u_dist +
                                                     d_edge * d_edge - d * d) /
                                                    (2.0 * curr_u_dist * d_edge);
                                    cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                    double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                    double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                      (double)d_mut * d_mut) /
                                                     (2.0 * d_edge * r_m);
                                    cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                    cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                    sin_th * sin_phi);
                                }
                            }
                        }
                    } // for (int rem = step + 1; ...)

                    if (!first_eval_done)
                    {
                        first_eval_done = 1;
                        for (int i = step + 1; i < eval_lim - 1; i++)
                        {
                            int best_p = i;
                            float max_a = cand_alignment[cand_order[i]];
                            for (int j = i + 1; j < eval_lim; j++)
                            {
                                if (cand_alignment[cand_order[j]] > max_a)
                                {
                                    max_a = cand_alignment[cand_order[j]];
                                    best_p = j;
                                }
                            }
                            if (best_p != i)
                            {
                                int tmp = cand_order[i];
                                cand_order[i] = cand_order[best_p];
                                cand_order[best_p] = tmp;
                            }
                        }
                    }
                }

                current_tau = knn_heap_peek_max_dist(heap);
                if ((d < current_tau / eps_factor || heap->count < heap->k) &&
                    frontier_count < GRAPH_FRONTIER_MAX)
                {
                    frontier[frontier_count].frame_id = nb_id;
                    frontier[frontier_count].dist = d;
                    frontier[frontier_count].parent_id = curr_u;
                    frontier[frontier_count].parent_dist = curr_u_dist;
                    frontier[frontier_count].expanded = 0;
                    frontier_count++;
                }
            }
        } // for (int step = 0; ...)
    } // while (expansions_done ...)

    // Global Containment Criterion via Local k-Ball Radius R_k(best_seed_id)
    if (*best_seed_id >= 0 && heap->count >= heap->k)
    {
        double current_tau = knn_heap_peek_max_dist(heap);
        double r_graph_ball =
            (double)model->graph_distances[*best_seed_id * (long)graph_k + (graph_k - 1)];

        if (r_graph_ball > 0.0 && (*best_seed_dist + current_tau / eps_factor) <= r_graph_ball)
        {
            const uint32_t *neighbors = &model->graph_indices[*best_seed_id * (long)graph_k];
            const float *n_dists = &model->graph_distances[*best_seed_id * (long)graph_k];

            float local_lb2_gc[64];
            int eval_gc_lim = (graph_k < 64) ? graph_k : 64;
            for (int i = 0; i < eval_gc_lim; i++)
            {
                local_lb2_gc[i] = 0.0f;
            }

            for (int k_idx = 0; k_idx < graph_k; k_idx++)
            {
                long nb_id = (long)neighbors[k_idx];
                if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == *best_seed_id)
                {
                    continue;
                }

                current_tau = knn_heap_peek_max_dist(heap);
                double tau_eff = current_tau / eps_factor;
                double tau_eff2 = tau_eff * tau_eff;
                if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
                {
                    tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
                }

                if (k_idx < 64 && (double)local_lb2_gc[k_idx] >= tau_eff2)
                {
                    telem->angular_pruned++;
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb_id))
                {
                    continue;
                }
                if (knn_heap_contains(heap, (int)nb_id))
                {
                    continue;
                }

                double d_edge = (double)n_dists[k_idx];
                double lb_edge = fabs(*best_seed_dist - d_edge);
                current_tau = knn_heap_peek_max_dist(heap);

                if (lb_edge >= current_tau / eps_factor ||
                    (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff))
                {
                    telem->graph_edges_pruned++;
                    continue;
                }

                if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
                                            model, config, telem) ||
                    is_graph_pruned_by_sq8(visited->query_sq8, nb_id, current_tau,
                                           model, config, telem))
                {
                    continue;
                }

                if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    telem->graph_seeds_evaluated++;
                    double d = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)nb_id, d);
                    }

                    if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                    {
                        for (int m_idx = k_idx + 1; m_idx < graph_k; m_idx++)
                        {
                            if (m_idx < 64)
                            {
                                double r_m = (double)n_dists[m_idx];
                                float d_mut = knn_get_mutual_dist(
                                    model, *best_seed_id, k_idx, m_idx
                                );
                                if (d_mut > 0.0f)
                                {
                                    double lb2_m = compute_angular_lower_bound_sq(
                                        *best_seed_dist, d_edge, r_m, d, (double)d_mut
                                    );
                                    if (lb2_m > (double)local_lb2_gc[m_idx])
                                    {
                                        local_lb2_gc[m_idx] = (float)lb2_m;
                                    }
                                }
                            }
                        } // for (int m_idx = ...)
                    }
                }
            } // for (int k_idx = 0; ...)

            telem->level0_super_clusters_pruned += (uint64_t)model->num_super_clusters;
            telem->level1_clusters_pruned += (uint64_t)model->num_clusters;
            telem->global_containment_hits++;
            return 1;
        } // if contained
    } // Global Containment

    return 0;
}

/**
 * knn_cross_eval_intra_cluster() - Search target cluster containing best seed for cross dataset.
 * @best_c:           Initial best anchor cluster.
 * @best_seed_id:     Best seed frame ID.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Computed distances to seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_cross_eval_intra_cluster(
    int                    best_c,
    long                   best_seed_id,
    double                 min_d_anchor,
    int                    num_seed_pivots,
    const long            *seed_pivot_ids,
    const double          *seed_pivot_dists,
    const void *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    void *restrict         cand_buffer,
    double        *restrict anchor_dists,
    KnnMaxHeap            *heap,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int target_c = best_c;
    if (best_seed_id >= 0 && model->frame_cluster_map != NULL)
    {
        int mapped_c = (int)model->frame_cluster_map[best_seed_id];
        if (mapped_c >= 0 && mapped_c < M)
        {
            target_c = mapped_c;
        }
    }

    if (target_c >= 0 && target_c < M)
    {
        const KnnCluster *target_cl = &model->clusters[target_c];
        telem->total_candidates_considered += (uint64_t)target_cl->num_members;

        double d_anchor_target = anchor_dists[target_c];
        if (d_anchor_target < 0.0)
        {
            telem->framedist_calls++;
            d_anchor_target = compute_euclidean_distance(
                query_data, target_cl->anchor_data, frame_elem, model->is_double
            );
            anchor_dists[target_c] = d_anchor_target;
        }

        int start_m = 0;
        int end_m = target_cl->num_members;
        double tau_init = knn_heap_peek_max_dist(heap);
        if (tau_init < 1e20)
        {
            float r_min = (float)fmax(0.0, d_anchor_target - tau_init / eps_factor);
            float r_max = (float)(d_anchor_target + tau_init / eps_factor);
            start_m = find_member_lower_bound(
                target_cl->members, target_cl->num_members, r_min
            );
            end_m = find_member_upper_bound(
                target_cl->members, target_cl->num_members, r_max
            );
            telem->level3_annular_pruned +=
                (uint64_t)start_m + (uint64_t)(target_cl->num_members - end_m);
        }

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)target_cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                continue;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double r_cand = (double)target_cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor_target + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor_target + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor_target, r_cand, 0.0, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
            {
                continue;
            }

            const void *cand_data = NULL;
            if (cand_reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)cand_reader->memory_data +
                            (size_t)cand_id * (size_t)frame_elem * el_sz;
            }
            else if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // if (target_c >= 0 ...)
}

/**
 * knn_cross_eval_inter_clusters() - Evaluate inter-cluster candidate members for cross dataset.
 * @loc_res:          Cluster locator results.
 * @best_c:           Best anchor cluster index.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Distances to evaluated seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
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
static void knn_cross_eval_inter_clusters(
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
    KnnTelemetry *restrict      telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);
    int num_cand_clusters = 0;

    for (int q = 0; q < M; q++)
    {
        if (q == best_c || model->clusters[q].num_members == 0)
        {
            continue;
        }

        if (config->rlim_cutoff > 0.0 && active_mask != NULL && !active_mask[q])
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // 3P test against all evaluated anchor pivots using tightened tau_k
        int pruned_3p = 0;
        double r_q = model->cluster_radii[q];
        for (int e = 0; e < loc_res->num_evaluated_anchors; e++)
        {
            int p = loc_res->evaluated_clusters[e];
            double d_p = loc_res->evaluated_dists[e];
            double dcc = model->dcc_matrix[p * M + q];
            double lb1 = dcc - r_q - d_p;
            double lb2 = d_p - dcc - r_q;
            double lb = (lb1 > lb2) ? lb1 : lb2;

            if (lb >= current_tau / eps_factor ||
                (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
            {
                active_mask[q] = 0;
                pruned_3p = 1;
                break;
            }
        }

        if (pruned_3p)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (anchor_dists[q] < 0.0)
        {
            telem->framedist_calls++;
            anchor_dists[q] = compute_euclidean_distance(
                query_data, model->clusters[q].anchor_data, frame_elem, model->is_double
            );
        }

        double d_a = anchor_dists[q];
        double lb = d_a - r_q;
        if (lb < 0.0)
        {
            lb = 0.0;
        }

        if (lb >= current_tau / eps_factor ||
            (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = d_a;
        num_cand_clusters++;
    } // for (int q = 0; ...)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    // Inter-Cluster Search on surviving candidate clusters
    int max_cands = (config->approx_mode) ? 4 : num_cand_clusters;
    for (int idx = 0; idx < num_cand_clusters && idx < max_cands; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        current_tau = knn_heap_peek_max_dist(heap);

        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (config->rlim_cutoff > 0.0 && lb_cluster >= config->rlim_cutoff)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        const KnnCluster *cl = &model->clusters[q];
        double d_anchor = anchor_dists[q];
        telem->total_candidates_considered += (uint64_t)cl->num_members;

        double tau_eff = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff = config->rlim_cutoff;
        }

        float r_min = (float)fmax(0.0, d_anchor - tau_eff);
        float r_max = (float)(d_anchor + tau_eff);

        int start_m = find_member_lower_bound(cl->members, cl->num_members, r_min);
        int end_m = find_member_upper_bound(cl->members, cl->num_members, r_max);

        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(cl->num_members - end_m);

        double dcc_best = model->dcc_matrix[best_c * M + q];

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            tau_eff = current_tau / eps_factor;
            double r_cand = (double)cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor, r_cand, dcc_best, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
            {
                continue;
            }

            const void *cand_data = NULL;
            if (cand_reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)cand_reader->memory_data +
                            (size_t)cand_id * (size_t)frame_elem * el_sz;
            }
            else if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // for (int idx = 0; ...)
}

/**
 * knn_update_trajectory_tracker() - Update thread-local trajectory tracker from heap.
 * @tracker:        Pointer to KnnTrajectoryTracker.
 * @query_id:       Current query index.
 * @best_c:         Best cluster ID.
 * @best_seed_id:   Best seed node frame ID.
 * @best_seed_dist: Distance to best seed node.
 * @heap:           Current query top-k heap.
 */
static inline void knn_update_trajectory_tracker(
    KnnTrajectoryTracker *tracker,
    long                  query_id,
    int                   best_c,
    long                  best_seed_id,
    double                best_seed_dist,
    const KnnMaxHeap     *heap)
{
    if (tracker == NULL)
    {
        return;
    }

    tracker->prev_query_id = query_id;
    tracker->prev_cluster_id = best_c;
    tracker->prev_best_seed = best_seed_id;
    tracker->prev_best_dist = best_seed_dist;

    int copy_k = (heap->count < 64) ? heap->count : 64;
    tracker->num_cached = copy_k;
    for (int j = 0; j < copy_k; j++)
    {
        tracker->cached_ids[j] = heap->data[j].frame_id;
        tracker->cached_dists[j] = heap->data[j].dist;
    }
}

/**
 * knn_search_cross_dataset_frame() - Metric-pruned k-NN search for external query frame.
 * @query_id:      Index of query frame.
 * @query_data:    Pixel buffer of query frame.
 * @model:         Active KnnModel.
 * @config:        Active KnnConfig.
 * @cand_reader:   Thread-local candidate frame reader.
 * @cand_buffer:   Scratch buffer for candidate frame pixels.
 * @anchor_dists:  Scratch buffer for query-to-anchor distances [num_clusters].
 * @scores_buffer: Scratch buffer for cluster sorting [num_clusters].
 * @heap:          KnnMaxHeap structure for this query frame.
 * @tracker:       Thread-local trajectory tracker for sequential continuity.
 * @visited:       Per-query frame visited tracker.
 * @telem:         Thread-local KnnTelemetry.
 */
static void knn_search_cross_dataset_frame(
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
    KnnTelemetry *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int best_c = -1;
    double min_d_anchor = 1e20;
    long best_seed_id = -1;
    double best_seed_dist = 1e20;
    int num_seed_pivots = 0;
    long seed_pivot_ids[8];
    double seed_pivot_dists[8];

    for (int c = 0; c < M; c++)
    {
        anchor_dists[c] = -1.0;
    }

    double effective_rlim = 0.0;
    if (config->rlim_cutoff > 0.0)
    {
        effective_rlim = config->rlim_cutoff;
    }
    else if (config->refuse_unclustered)
    {
        effective_rlim = model->model_rlim;
    }

    uint8_t *active_mask = (uint8_t *)alloca((size_t)M);
    int prev_c = (tracker != NULL) ? tracker->prev_cluster_id : -1;
    ClusterLocatorConfig loc_cfg;
    memset(&loc_cfg, 0, sizeof(loc_cfg));
    loc_cfg.max_targets = (config->approx_mode) ? 24 : 32;
    loc_cfg.strict_rlim = (effective_rlim > 0.0) ? config->refuse_unclustered : 0;
    loc_cfg.te4_mode = 1;
    loc_cfg.te5_mode = 1;
    loc_cfg.rlim = effective_rlim;
    loc_cfg.tau_max = knn_heap_peek_max_dist(heap);
    loc_cfg.epsilon = config->epsilon;
    loc_cfg.prev_cluster_id = prev_c;
    loc_cfg.is_double = model->is_double;
    loc_cfg.query_sq16 = visited->query_sq16;
    loc_cfg.query_sq8 = visited->query_sq8;
    loc_cfg.sq16_params =
        (config->use_sq16 && model->anchor_sq16_buffer != NULL) ?
            &model->sq16_params : NULL;
    loc_cfg.sq8_params =
        (config->use_sq8 && model->anchor_sq8_buffer != NULL) ?
            &model->sq8_params : NULL;
    loc_cfg.anchors_sq16_buf = model->anchor_sq16_buffer;
    loc_cfg.anchors_sq8_buf = model->anchor_sq8_buffer;

    ClusterLocatorResult loc_res;
    loc_res.active_cluster_mask = active_mask;

    int loc_ret = cluster_locate_sample(
        query_data, frame_elem, M, model->anchor_ptrs,
        model->cluster_radii, model->dcc_matrix, &loc_cfg, &loc_res
    );

    telem->framedist_calls += (uint64_t)loc_res.num_evaluated_anchors;
    for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
    {
        anchor_dists[loc_res.evaluated_clusters[e]] = loc_res.evaluated_dists[e];
    }

    if (loc_ret == CLUSTER_LOCATE_REJECTED ||
        (loc_cfg.strict_rlim && loc_res.best_cluster_id < 0))
    {
        telem->out_of_cluster_rejected++;
        return;
    }

    best_c = loc_res.best_cluster_id;
    min_d_anchor = loc_res.best_anchor_dist;
    if (best_c < 0)
    {
        best_c = 0;
    }

    // Push evaluated anchor frames into heap and seed_pivot_ids
    for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
    {
        int c = loc_res.evaluated_clusters[e];
        double d_anc = loc_res.evaluated_dists[e];
        if (c >= 0 && c < M && model->clusters[c].num_members > 0)
        {
            if (model->clusters[c].members[0].r_anchor < 1e-4f)
            {
                long anc_frame = (long)model->clusters[c].members[0].frame_id;
                knn_visited_check_and_mark(visited, anc_frame);
                if (config->rlim_cutoff <= 0.0 || d_anc <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)anc_frame, d_anc);
                }
                if (num_seed_pivots < 8)
                {
                    seed_pivot_ids[num_seed_pivots] = anc_frame;
                    seed_pivot_dists[num_seed_pivots] = d_anc;
                    num_seed_pivots++;
                }
                if (c == best_c)
                {
                    best_seed_id = anc_frame;
                    best_seed_dist = d_anc;
                }
            }
        }
    } // for (int e = 0; ...)

    // Select additional seed in best_c near d_anchor_best
    if (best_c >= 0 && best_c < M)
    {
        const KnnCluster *best_cl = &model->clusters[best_c];
        double d_anchor_best = (anchor_dists[best_c] >= 0.0) ?
            anchor_dists[best_c] : min_d_anchor;

        int mid = find_member_lower_bound(
            best_cl->members, best_cl->num_members, (float)d_anchor_best
        );
        if (mid >= best_cl->num_members)
        {
            mid = best_cl->num_members - 1;
        }
        if (mid > 0)
        {
            long seed_0 = (long)best_cl->members[mid].frame_id;
            if (!knn_visited_check_and_mark(visited, seed_0))
            {
                if (knn_reader_read_frame(cand_reader, seed_0, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    double d0 = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d0 <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)seed_0, d0);
                    }
                    if (d0 < best_seed_dist)
                    {
                        best_seed_dist = d0;
                        best_seed_id = seed_0;
                    }
                    if (num_seed_pivots < 8)
                    {
                        seed_pivot_ids[num_seed_pivots] = seed_0;
                        seed_pivot_dists[num_seed_pivots] = d0;
                        num_seed_pivots++;
                    }
                }
            }
        }
    } // if (best_c >= 0 ...)

    if (best_seed_id < 0 && best_c >= 0 && best_c < M)
    {
        const KnnCluster *best_cl = &model->clusters[best_c];
        if (best_cl->num_members > 0)
        {
            long seed_0 = (long)best_cl->members[0].frame_id;
            knn_visited_check_and_mark(visited, seed_0);
            if (knn_reader_read_frame(cand_reader, seed_0, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d0 = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d0 <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)seed_0, d0);
                }
                best_seed_id = seed_0;
                best_seed_dist = d0;
                if (num_seed_pivots < 8)
                {
                    seed_pivot_ids[num_seed_pivots] = seed_0;
                    seed_pivot_dists[num_seed_pivots] = d0;
                    num_seed_pivots++;
                }
            }
        }
    }

    // Option 2: Sequential Trajectory Seed Warm-Start
    if (config->use_trajectory && tracker != NULL &&
        tracker->prev_query_id == query_id - 1 && tracker->prev_best_seed >= 0)
    {
        long prev_seed = tracker->prev_best_seed;
        if (!knn_visited_check_and_mark(visited, prev_seed))
        {
            if (knn_reader_read_frame(cand_reader, prev_seed, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d_prev = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d_prev <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)prev_seed, d_prev);
                }
                if (d_prev < best_seed_dist)
                {
                    best_seed_dist = d_prev;
                    best_seed_id = prev_seed;
                    if (num_seed_pivots < 8)
                    {
                        seed_pivot_ids[num_seed_pivots] = prev_seed;
                        seed_pivot_dists[num_seed_pivots] = d_prev;
                        num_seed_pivots++;
                    }
                    telem->trajectory_warmstarts++;
                }
                if (prev_seed >= 0 && prev_seed < model->total_dataset_frames)
                {
                    int prev_seed_c = model->frame_cluster_map[prev_seed];
                    if (prev_seed_c >= 0 && prev_seed_c < M && d_prev < min_d_anchor)
                    {
                        best_c = prev_seed_c;
                        min_d_anchor = d_prev;
                    }
                }
            }
        }
    }

    // Graph Search: Dynamic Frontier Search across candidate seeds
    if (model->has_knn_graph && best_seed_id >= 0 && model->graph_indices != NULL &&
        model->graph_distances != NULL)
    {
        // Phase 1 & 2 (Approx mode): Fast Greedy 1-NN Routing + Direct Basin Expansion
        if (config->approx_mode)
        {
            long last_p_id = -1;
            double last_p_dist = -1.0;
            knn_greedy_route_to_basin(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, &last_p_id, &last_p_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

            knn_direct_basin_expansion(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, last_p_id, last_p_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

        }

        // Opportunity 1: In approx mode, if basin expansion has saturated the top-k heap,
        // verify whether any other evaluated candidate cluster could overlap the search ball.
        int skip_frontier = 0;
        if (config->approx_mode && heap->count >= heap->k)
        {
            double current_tau = knn_heap_peek_max_dist(heap);
            double eps_factor = 1.0 + config->epsilon;
            double tau_eff = current_tau / eps_factor;
            double min_other_lb = 1e30;
            for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
            {
                int c = loc_res.evaluated_clusters[e];
                if (c != best_c && c >= 0 && c < M)
                {
                    double d_anc = loc_res.evaluated_dists[e];
                    double r_c = model->cluster_radii[c];
                    double lb = d_anc - r_c;
                    if (lb < min_other_lb)
                    {
                        min_other_lb = lb;
                    }
                }
            }
            if (tau_eff <= min_other_lb)
            {
                skip_frontier = 1;
            }
        }

        if (!skip_frontier)
        {
            FrontierNode frontier[GRAPH_FRONTIER_MAX];
            int frontier_count = 0;

            frontier[0].frame_id = best_seed_id;
            frontier[0].dist = best_seed_dist;
            frontier[0].parent_id = -1;
            frontier[0].parent_dist = -1.0;
            frontier[0].expanded = 0;
            frontier_count = 1;

            knn_cross_seed_frontier(
                &loc_res, best_c, query_data, model, config, cand_reader,
                cand_buffer, heap, frontier, &frontier_count,
                &best_seed_id, &best_seed_dist, visited, telem
            );

            // Phase 2: Local Basin Top-k Collection with Angular Prioritization
            int contained = knn_cross_explore_graph_frontier(
                query_data, model, config, cand_reader, cand_buffer, heap,
                frontier, frontier_count, &best_seed_id, &best_seed_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

            if (contained)
            {
                knn_update_trajectory_tracker(
                    tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
                );
                return;
            }
        }
    }

    // Intra-Cluster Member Search in target cluster (containing best_seed_id)
    knn_cross_eval_intra_cluster(
        best_c, best_seed_id, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists, heap, visited, telem
    );

    for (int j = 0; j < heap->count; j++)
    {
        if (heap->data[j].dist < best_seed_dist)
        {
            best_seed_dist = heap->data[j].dist;
            best_seed_id = heap->data[j].frame_id;
        }
    }

    if (best_seed_id >= 0 && best_seed_id < model->total_dataset_frames)
    {
        int seed_c = model->frame_cluster_map[best_seed_id];
        if (seed_c >= 0 && seed_c < M)
        {
            best_c = seed_c;
        }
    }

    if (config->approx_mode && heap->count >= heap->k && best_seed_dist < 1e-5)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        knn_update_trajectory_tracker(
            tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
        );
        return;
    }

    // Inter-Cluster Search on surviving candidate clusters
    knn_cross_eval_inter_clusters(
        &loc_res, best_c, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists,
        scores_buffer, active_mask, heap, visited, telem
    );

    for (int j = 0; j < heap->count; j++)
    {
        if (heap->data[j].dist < best_seed_dist)
        {
            best_seed_dist = heap->data[j].dist;
            best_seed_id = heap->data[j].frame_id;
        }
    }

    if (best_seed_id >= 0 && best_seed_id < model->total_dataset_frames)
    {
        int seed_c = model->frame_cluster_map[best_seed_id];
        if (seed_c >= 0 && seed_c < M)
        {
            best_c = seed_c;
        }
    }

    knn_update_trajectory_tracker(
        tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
    );
}

/**
 * knn_run_search() - Multi-threaded driver executing k-NN search across all frames.
 * @config:    Active KnnConfig.
 * @model:     Active KnnModel.
 * @results:   Output KnnResults structure to populate.
 * @telemetry: Output aggregated KnnTelemetry structure.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry)
{
    if (config == NULL || model == NULL || results == NULL || telemetry == NULL)
    {
        return -1;
    }

    memset(telemetry, 0, sizeof(KnnTelemetry));

    int is_cross_dataset = (config->query_data_path != NULL) ? 1 : 0;
    long N_query = model->total_dataset_frames;
    long N_cand = model->total_dataset_frames;
    long q_w = model->frame_width;
    long q_h = model->frame_height;

    if (is_cross_dataset)
    {
        if (knn_reader_inspect(config->query_data_path, &N_query, &q_w, &q_h) != 0 ||
            N_query <= 0)
        {
            fprintf(stderr, "Error: Could not inspect query dataset '%s'\n",
                    config->query_data_path);
            return -1;
        }

        if (q_w * q_h != model->frame_elements)
        {
            fprintf(stderr,
                    "Error: Query frame dim (%ld) does not match model (%ld)\n",
                    q_w * q_h, model->frame_elements);
            return -1;
        }
    }

    int k = config->k;

    results->num_queries = N_query;
    results->indices = (int *)malloc((size_t)N_query * (size_t)k * sizeof(int));
    results->distances = (double *)malloc((size_t)N_query * (size_t)k * sizeof(double));

    if (results->indices == NULL || results->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for results buffer\n");
        return -1;
    }

    KnnMaxHeap *all_heaps = (KnnMaxHeap *)malloc((size_t)N_query * sizeof(KnnMaxHeap));
    if (all_heaps == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for heaps array\n");
        knn_results_free(results);
        return -1;
    }

    int search_k = (config->ef_search > 0) ? config->ef_search :
                   (config->approx_mode ? 2 * k : k);
    if (search_k < k)
    {
        search_k = k;
    }

    for (long i = 0; i < N_query; i++)
    {
        if (knn_heap_init(&all_heaps[i], search_k) != 0)
        {
            fprintf(stderr, "Error: Failed to init heap %ld\n", i);
            for (long j = 0; j < i; j++)
            {
                knn_heap_free(&all_heaps[j]);
            }
            free(all_heaps);
            knn_results_free(results);
            return -1;
        }
    } // for (long i = 0; ...)

#ifdef _OPENMP
    omp_lock_t bucket_locks[KNN_NUM_BUCKET_LOCKS];
    if (!is_cross_dataset)
    {
        for (int b = 0; b < KNN_NUM_BUCKET_LOCKS; b++)
        {
            omp_init_lock(&bucket_locks[b]);
        }
    }
#endif

    KnnFrameReader master_cand_reader;
    KnnFrameReader master_query_reader;

    if (config->memory_data != NULL)
    {
        if (knn_reader_open_memory(&master_cand_reader, config->memory_data, N_cand,
                                   model->frame_elements, model->is_double) != 0)
        {
            knn_results_free(results);
            return -1;
        }
    }
    else if (knn_reader_open(&master_cand_reader, config->input_data_path, N_cand,
                             model->frame_width, model->frame_height, model->is_double) != 0)
    {
        knn_results_free(results);
        return -1;
    }

    if (is_cross_dataset)
    {
        if (knn_reader_open(&master_query_reader, config->query_data_path, N_query,
                            q_w, q_h, model->is_double) != 0)
        {
            knn_reader_close(&master_cand_reader);
            knn_results_free(results);
            return -1;
        }
    }

    int nthreads = config->nthreads;
#ifdef _OPENMP
    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
    }
    else
    {
        nthreads = omp_get_max_threads();
    }
#else
    nthreads = 1;
#endif

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long progress_step = N_query / 100;
    if (progress_step < 1)
    {
        progress_step = 1;
    }

    uint64_t global_telem_calls = 0;
    uint64_t global_telem_l0 = 0;
    uint64_t global_telem_l1 = 0;
    uint64_t global_telem_l2 = 0;
    uint64_t global_telem_l3 = 0;
    uint64_t global_telem_temp = 0;
    uint64_t global_telem_recip = 0;
    uint64_t global_telem_graph_seeds = 0;
    uint64_t global_telem_graph_edges = 0;
    uint64_t global_telem_multi_pivot = 0;
    uint64_t global_telem_angular = 0;
    uint64_t global_telem_containment = 0;
    uint64_t global_telem_cand = 0;
    uint64_t global_telem_traj = 0;
    uint64_t global_telem_rejected = 0;
    uint64_t global_telem_sq8_evals = 0;
    uint64_t global_telem_sq8_pruned = 0;
    uint64_t global_telem_sq8_graph_pruned = 0;
    uint64_t global_telem_sq16_evals = 0;
    uint64_t global_telem_sq16_pruned = 0;
    uint64_t global_telem_sq16_graph_pruned = 0;
    uint64_t global_telem_graph_clusters = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:global_telem_calls, global_telem_l0, global_telem_l1, \
                                 global_telem_l2, global_telem_l3, global_telem_temp,   \
                                 global_telem_recip, global_telem_graph_seeds,          \
                                 global_telem_graph_edges, global_telem_multi_pivot,    \
                                 global_telem_angular, global_telem_containment,        \
                                 global_telem_cand, global_telem_traj,                  \
                                 global_telem_rejected,                                 \
                                 global_telem_sq8_evals, global_telem_sq8_pruned,      \
                                 global_telem_sq8_graph_pruned,                         \
                                 global_telem_sq16_evals, global_telem_sq16_pruned,    \
                                 global_telem_sq16_graph_pruned,                        \
                                 global_telem_graph_clusters)
#endif
    {
        KnnFrameReader thread_cand_reader;
        KnnFrameReader thread_query_reader;

        knn_reader_clone_thread(&master_cand_reader, &thread_cand_reader);
        if (is_cross_dataset)
        {
            knn_reader_clone_thread(&master_query_reader, &thread_query_reader);
        }

        size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
        void *query_buffer =
            malloc((size_t)model->frame_elements * elem_size);
        void *cand_buffer =
            malloc((size_t)4 * model->frame_elements * elem_size);
        uint8_t *query_sq8 =
            config->use_sq8 ? (uint8_t *)malloc((size_t)model->frame_elements) : NULL;
        int16_t *query_sq16 =
            config->use_sq16 ? (int16_t *)malloc((size_t)model->frame_elements *
                                                 sizeof(int16_t)) : NULL;
        double *anchor_dists =
            (double *)malloc((size_t)model->num_clusters * sizeof(double));
        ClusterScore *scores_buf =
            (ClusterScore *)malloc((size_t)model->num_clusters * sizeof(ClusterScore));

        KnnClusterGraphScratch graph_scratch;
        memset(&graph_scratch, 0, sizeof(KnnClusterGraphScratch));
        if (config->use_cluster_graph && model->cluster_graph_adj != NULL)
        {
            graph_scratch.pq = (ClusterPqNode *)malloc(
                (size_t)model->num_clusters * sizeof(ClusterPqNode)
            );
            graph_scratch.enqueued_tags = (uint32_t *)calloc(
                (size_t)model->num_clusters, sizeof(uint32_t)
            );
            graph_scratch.enqueued_epoch = 1;
            graph_scratch.anchor_dists = anchor_dists;
            graph_scratch.anchor_is_sq16 = (uint8_t *)malloc(
                (size_t)model->num_clusters * sizeof(uint8_t)
            );
        }

        KnnTelemetry thread_telem;
        memset(&thread_telem, 0, sizeof(KnnTelemetry));
        KnnTrajectoryTracker thread_tracker;
        memset(&thread_tracker, 0, sizeof(KnnTrajectoryTracker));
        thread_tracker.prev_query_id = -999;
        thread_tracker.prev_cluster_id = -1;
        thread_tracker.prev_best_seed = -1;
        thread_tracker.prev_best_dist = 1e20;

        KnnVisitedTracker visited;
        visited.tags = (uint32_t *)calloc((size_t)N_cand, sizeof(uint32_t));
        visited.epoch = 1;
        visited.query_sq8 = query_sq8;
        visited.query_sq16 = query_sq16;

#ifdef _OPENMP
#pragma omp for schedule(guided, 16)
#endif
        for (long i = 0; i < N_query; i++)
        {
            visited.epoch++;
            if (visited.epoch == 0)
            {
                if (visited.tags != NULL)
                {
                    memset(visited.tags, 0, (size_t)N_cand * sizeof(uint32_t));
                }
                visited.epoch = 1;
            }

            KnnFrameReader *active_qreader = is_cross_dataset ? &thread_query_reader :
                                                                &thread_cand_reader;

            if (knn_reader_read_frame(active_qreader, i, query_buffer) == 0)
            {
                if (config->use_sq8 && query_sq8 != NULL)
                {
                    if (model->is_double)
                    {
                        sq8_quantize_double(
                            (const double *)query_buffer, query_sq8, &model->sq8_params);
                    }
                    else
                    {
                        sq8_quantize_float(
                            (const float *)query_buffer, query_sq8, &model->sq8_params);
                    }
                }

                if (config->use_sq16 && query_sq16 != NULL)
                {
                    if (model->is_double)
                    {
                        sq16_quantize_double(
                            (const double *)query_buffer, query_sq16, &model->sq16_params);
                    }
                    else
                    {
                        sq16_quantize_float(
                            (const float *)query_buffer, query_sq16, &model->sq16_params);
                    }
                }

                if (is_cross_dataset)
                {
                    knn_search_cross_dataset_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, anchor_dists, scores_buf, &all_heaps[i],
                        &thread_tracker, &visited, &thread_telem);
                }
                else
                {
                    knn_search_single_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, scores_buf, &graph_scratch, all_heaps,
#ifdef _OPENMP
                        bucket_locks,
#endif
                        &visited, &thread_telem);
                }
                thread_telem.total_queries++;
            }

            if (config->progress_mode && i % progress_step == 0)
            {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0)
#endif
                {
                    double pct = 100.0 * (double)i / (double)N_query;
                    int bar_offset = 40 - (int)(pct * 0.4);
                    if (bar_offset < 0)
                    {
                        bar_offset = 0;
                    }
                    if (bar_offset > 40)
                    {
                        bar_offset = 40;
                    }
                    const char *bar = "========================================";
                    printf("\rSearching k-NN: [%-40s] %5.1f%% (%ld / %ld frames)",
                           &bar[bar_offset], pct, i, N_query);
                    fflush(stdout);
                }
            }
        } // for (long i = 0; ...)

        global_telem_calls += thread_telem.framedist_calls;
        global_telem_l0 += thread_telem.level0_super_clusters_pruned;
        global_telem_l1 += thread_telem.level1_clusters_pruned;
        global_telem_l2 += thread_telem.level2_anchors_pruned;
        global_telem_l3 += thread_telem.level3_annular_pruned;
        global_telem_temp += thread_telem.temporal_pruned;
        global_telem_recip += thread_telem.reciprocal_reused;
        global_telem_graph_seeds += thread_telem.graph_seeds_evaluated;
        global_telem_graph_edges += thread_telem.graph_edges_pruned;
        global_telem_multi_pivot += thread_telem.multi_pivot_pruned;
        global_telem_angular += thread_telem.angular_pruned;
        global_telem_containment += thread_telem.global_containment_hits;
        global_telem_cand += thread_telem.total_candidates_considered;
        global_telem_traj += thread_telem.trajectory_warmstarts;
        global_telem_rejected += thread_telem.out_of_cluster_rejected;
        global_telem_sq8_evals += thread_telem.sq8_evaluations;
        global_telem_sq8_pruned += thread_telem.sq8_members_pruned;
        global_telem_sq8_graph_pruned += thread_telem.sq8_graph_pruned;
        global_telem_sq16_evals += thread_telem.sq16_evaluations;
        global_telem_sq16_pruned += thread_telem.sq16_members_pruned;
        global_telem_sq16_graph_pruned += thread_telem.sq16_graph_pruned;
        global_telem_graph_clusters += thread_telem.clusters_graph_evaluated;

        if (visited.tags != NULL)
        {
            free(visited.tags);
        }

        if (query_sq8 != NULL)
        {
            free(query_sq8);
        }

        if (query_sq16 != NULL)
        {
            free(query_sq16);
        }

        if (graph_scratch.pq != NULL)
        {
            free(graph_scratch.pq);
        }
        if (graph_scratch.enqueued_tags != NULL)
        {
            free(graph_scratch.enqueued_tags);
        }
        if (graph_scratch.anchor_is_sq16 != NULL)
        {
            free(graph_scratch.anchor_is_sq16);
        }

        free(scores_buf);
        free(anchor_dists);
        free(cand_buffer);
        free(query_buffer);

        if (is_cross_dataset)
        {
            knn_reader_close_thread(&thread_query_reader);
        }
        knn_reader_close_thread(&thread_cand_reader);
    } // OpenMP parallel block

    // Extract sorted results in parallel from all heaps
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < N_query; i++)
    {
        knn_heap_extract_sorted(&all_heaps[i], &results->indices[i * k],
                                &results->distances[i * k], k);
        knn_heap_free(&all_heaps[i]);
    }
    free(all_heaps);

#ifdef _OPENMP
    if (!is_cross_dataset)
    {
        for (int b = 0; b < KNN_NUM_BUCKET_LOCKS; b++)
        {
            omp_destroy_lock(&bucket_locks[b]);
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    if (is_cross_dataset)
    {
        knn_reader_close(&master_query_reader);
    }
    knn_reader_close(&master_cand_reader);

    if (config->progress_mode)
    {
        printf("\rSearching k-NN: [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N_query, N_query);
        fflush(stdout);
    }

    telemetry->total_queries = (uint64_t)N_query;
    telemetry->framedist_calls = global_telem_calls;
    telemetry->level0_super_clusters_pruned = global_telem_l0;
    telemetry->level1_clusters_pruned = global_telem_l1;
    telemetry->level2_anchors_pruned = global_telem_l2;
    telemetry->level3_annular_pruned = global_telem_l3;
    telemetry->temporal_pruned = global_telem_temp;
    telemetry->reciprocal_reused = global_telem_recip;
    telemetry->graph_seeds_evaluated = global_telem_graph_seeds;
    telemetry->graph_edges_pruned = global_telem_graph_edges;
    telemetry->multi_pivot_pruned = global_telem_multi_pivot;
    telemetry->angular_pruned = global_telem_angular;
    telemetry->global_containment_hits = global_telem_containment;
    telemetry->trajectory_warmstarts = global_telem_traj;
    telemetry->out_of_cluster_rejected = global_telem_rejected;
    telemetry->sq8_evaluations = global_telem_sq8_evals;
    telemetry->sq8_members_pruned = global_telem_sq8_pruned;
    telemetry->sq8_graph_pruned = global_telem_sq8_graph_pruned;
    telemetry->sq16_evaluations = global_telem_sq16_evals;
    telemetry->sq16_members_pruned = global_telem_sq16_pruned;
    telemetry->sq16_graph_pruned = global_telem_sq16_graph_pruned;
    telemetry->clusters_graph_evaluated = global_telem_graph_clusters;
    telemetry->total_candidates_considered = global_telem_cand;
    telemetry->time_search_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return 0;
}

/**
 * knn_results_free() - Clean up KnnResults arrays.
 * @results: Pointer to KnnResults.
 */
void knn_results_free(
    KnnResults *results)
{
    if (results == NULL)
    {
        return;
    }

    if (results->indices != NULL)
    {
        free(results->indices);
        results->indices = NULL;
    }

    if (results->distances != NULL)
    {
        free(results->distances);
        results->distances = NULL;
    }
}


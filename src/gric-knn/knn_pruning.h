/**
 * @file knn_pruning.h
 * @brief Metric lower bounding and distance pruning for k-NN search.
 */

#ifndef KNN_PRUNING_H
#define KNN_PRUNING_H

#include "knn_engine_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

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
static inline int is_cluster_pruned_by_pivots(
    int                  c,
    double               cl_radius,
    double               current_tau,
    double               eps_factor,
    const KnnModel      *model,
    const KnnConfig     *config,
    const MeasuredPivot *pivots,
    int                  num_pivots,
    double               sq16_delta)
{
    if (!config->use_multi_pivot || pivots == NULL || num_pivots <= 0)
    {
        return 0;
    }

    size_t M = (size_t)model->num_clusters;
    for (int p = 0; p < num_pivots; p++)
    {
        int p_cl = pivots[p].cluster_id;
        if (p_cl == c)
        {
            continue;
        }

        double dcc_pc = model->dcc_matrix[(size_t)p_cl * M + (size_t)c];
        if (dcc_pc > 0.0)
        {
            double d_qp = pivots[p].d_anchor;
            double lb_p = fabs(d_qp - dcc_pc) - cl_radius;
            if (lb_p - sq16_delta >= current_tau / eps_factor ||
                (config->rlim_cutoff > 0.0 &&
                 lb_p - sq16_delta >= config->rlim_cutoff))
            {
                return 1;
            }
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
 * compute_rq8_cutoff_thresh() - Compute RQ8 squared distance cutoff threshold.
 * @cur_tau: Current search radius (heap max dist or rlim_cutoff).
 * @model:   Active KnnModel.
 * @config:  Active KnnConfig.
 *
 * Return: Threshold on integer sum-of-squared differences, or UINT64_MAX if disabled.
 */
static inline uint64_t compute_rq8_cutoff_thresh(
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config)
{
    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < cur_tau)
    {
        cur_tau = config->rlim_cutoff;
    }

    double eps = config->rq8_approx ? config->epsilon : 0.0;
    return rq8_compute_cutoff_thresh(cur_tau, &model->rq8_params, eps);
}

/**
 * is_member_pruned_by_rq8_cached() - Evaluate RQ8 using precomputed SSD cutoff.
 * @query_rq8:   Pointer to quantized int16 query residual [dim].
 * @cand_id:     Index of candidate dataset frame.
 * @ssd_cutoff:  Precomputed SSD cutoff threshold.
 * @model:       Active KnnModel.
 * @telem:       Active KnnTelemetry.
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

/**
 * is_member_pruned_by_rq8() - Evaluate RQ8 metric lower bound against current search radius.
 * @query_rq8: Pointer to quantized int16 query residual [dim].
 * @cand_id:   Index of candidate dataset frame.
 * @cur_tau:   Current distance to k-th nearest neighbor (or cutoff radius).
 * @model:     Active KnnModel.
 * @config:    Active KnnConfig.
 * @telem:     Active KnnTelemetry.
 *
 * Return: 1 if pruned, 0 if candidate must be evaluated in full precision.
 */
static inline int is_member_pruned_by_rq8(
    const int16_t   *query_rq8,
    long             cand_id,
    double           cur_tau,
    const KnnModel  *model,
    const KnnConfig *config,
    KnnTelemetry    *telem)
{
    if (!config->use_rq8 || model->rq8_dataset_buffer == NULL || query_rq8 == NULL)
    {
        return 0;
    }

    uint64_t ssd_cutoff = compute_rq8_cutoff_thresh(cur_tau, model, config);
    return is_member_pruned_by_rq8_cached(query_rq8, cand_id, ssd_cutoff, model, telem);
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
        return 1;
    }

    if (query_id == candidate_id)
    {
        return 0;
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
 * @d0:      Distance from query to central node: d(b, a0).
 * @ri:      Distance from central node to evaluated neighbor: d(a0, ai).
 * @rj:      Distance from central node to un-evaluated neighbor: d(a0, aj).
 * @di:      Evaluated distance from query to neighbor: d(b, ai).
 * @d_A_ij:  Precomputed mutual distance between ai and aj: d(ai, aj).
 * @tau_eff: Effective search threshold radius (tau / eps_factor).
 * @rlim:    Cutoff radius cutoff (if > 0).
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
    double lb1 = fabs(d_anchor - r_cand);
    if (lb1 >= tau_eff || (rlim_cutoff > 0.0 && lb1 >= rlim_cutoff))
    {
        telem->level3_annular_pruned++;
        return 1;
    }

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

    if (heap->count < heap->k || (float)dist < heap->tau)
    {
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
    }

    if (config->use_reciprocal && cand_id > query_id &&
        cand_id < model->total_dataset_frames && all_heaps != NULL)
    {
        KnnMaxHeap *target_heap = &all_heaps[cand_id];
        if (target_heap->count < target_heap->k || dist < knn_heap_peek_max_dist(target_heap))
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
            int right = 2 * i + 2;
            int best = left;
            if (right < *size && pq[right].dist < pq[left].dist)
            {
                best = right;
            }
            if (last.dist <= pq[best].dist)
            {
                break;
            }
            pq[i] = pq[best];
            i = best;
        }
        pq[i] = last;
    }
    return top;
}

/**
 * knn_compute_anchor_distance() - Compute distance between query frame and cluster anchor.
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

    const void *anchor_data = (model->anchor_ptrs != NULL) ?
                              model->anchor_ptrs[c] : model->clusters[c].anchor_data;
    telem->framedist_calls++;
    return compute_euclidean_distance(
        query_data, anchor_data, frame_elem, model->is_double
    );
}

#ifdef __cplusplus
}
#endif

#endif // KNN_PRUNING_H

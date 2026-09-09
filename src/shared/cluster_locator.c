/**
 * @file cluster_locator.c
 * @brief Standalone shared cluster localization and 3P metric bounding engine.
 */

#include "cluster_locator.h"
#include <alloca.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * calc_min_dist_4pt() - Computes minimum distance using a 4-point configuration.
 * @d14: Distance between point 1 and 4.
 * @d24: Distance between point 2 and 4.
 * @d12: Distance between point 1 and 2.
 * @d13: Distance between point 1 and 3.
 * @d23: Distance between point 2 and 3.
 *
 * Return: Min distance between point 3 and 4 in reconstructed 2D space.
 */
double calc_min_dist_4pt(
    double d14,
    double d24,
    double d12,
    double d13,
    double d23)
{
    if (d12 < 1e-9)
    {
        return fabs(d14 - d13);
    }

    double x3 = (d13 * d13 + d12 * d12 - d23 * d23) / (2.0 * d12);
    double y3_sq = d13 * d13 - x3 * x3;
    double y3 = (y3_sq > 0.0) ? sqrt(y3_sq) : 0.0;

    double x4 = (d14 * d14 + d12 * d12 - d24 * d24) / (2.0 * d12);
    double y4_sq = d14 * d14 - x4 * x4;
    double y4 = (y4_sq > 0.0) ? sqrt(y4_sq) : 0.0;

    return sqrt((x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4));
}

/**
 * calc_min_dist_5pt() - Computes minimum distance using a 5-point configuration.
 * @d_f_c1:  Distance from frame F to C1.
 * @d_f_c2:  Distance from frame F to C2.
 * @d_f_c3:  Distance from frame F to C3.
 * @d_t_c1:  Distance from target T to C1.
 * @d_t_c2:  Distance from target T to C2.
 * @d_t_c3:  Distance from target T to C3.
 * @d_c1_c2: Distance between C1 and C2.
 * @d_c1_c3: Distance between C1 and C3.
 * @d_c2_c3: Distance between C2 and C3.
 *
 * Return: Computed distance between F and T in reconstructed 3D space.
 */
double calc_min_dist_5pt(
    double d_f_c1,
    double d_f_c2,
    double d_f_c3,
    double d_t_c1,
    double d_t_c2,
    double d_t_c3,
    double d_c1_c2,
    double d_c1_c3,
    double d_c2_c3)
{
    if (d_c1_c2 < 1e-9)
    {
        return 0.0;
    }

    double x3 = (d_c1_c3 * d_c1_c3 + d_c1_c2 * d_c1_c2 - d_c2_c3 * d_c2_c3) / (2.0 * d_c1_c2);
    double y3_sq = d_c1_c3 * d_c1_c3 - x3 * x3;
    if (y3_sq < 1e-9)
    {
        return 0.0;
    }
    double y3 = sqrt(y3_sq);

    double xF = (d_f_c1 * d_f_c1 + d_c1_c2 * d_c1_c2 - d_f_c2 * d_f_c2) / (2.0 * d_c1_c2);
    double yF =
        (d_f_c1 * d_f_c1 + d_c1_c3 * d_c1_c3 - d_f_c3 * d_f_c3 - 2.0 * xF * x3) / (2.0 * y3);
    double zF_sq = d_f_c1 * d_f_c1 - xF * xF - yF * yF;
    double zF = (zF_sq > 0.0) ? sqrt(zF_sq) : 0.0;

    double xT = (d_t_c1 * d_t_c1 + d_c1_c2 * d_c1_c2 - d_t_c2 * d_t_c2) / (2.0 * d_c1_c2);
    double yT =
        (d_t_c1 * d_t_c1 + d_c1_c3 * d_c1_c3 - d_t_c3 * d_t_c3 - 2.0 * xT * x3) / (2.0 * y3);
    double zT_sq = d_t_c1 * d_t_c1 - xT * xT - yT * yT;
    double zT = (zT_sq > 0.0) ? sqrt(zT_sq) : 0.0;

    return sqrt((xF - xT) * (xF - xT) + (yF - yT) * (yF - yT) + (zF - zT) * (zF - zT));
}

/**
 * compute_vector_distance() - Computes Euclidean distance between two vectors.
 * @a: Pointer to vector a.
 * @b: Pointer to vector b.
 * @n: Number of elements.
 *
 * Return: Euclidean distance.
 */
static inline double compute_vector_distance(
    const void *restrict a,
    const void *restrict b,
    long                 n,
    int                  is_double)
{
    if (is_double)
    {
        const double *restrict da = (const double *)a;
        const double *restrict db = (const double *)b;
        double sum = 0.0;
        for (long i = 0; i < n; i++)
        {
            double diff = da[i] - db[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    else
    {
        const float *restrict fa = (const float *)a;
        const float *restrict fb = (const float *)b;
        float sum = 0.0f;
        for (long i = 0; i < n; i++)
        {
            float diff = fa[i] - fb[i];
            sum += diff * diff;
        }
        return (double)sqrtf(sum);
    }
}

/**
 * select_best_first_target() - Selects next target minimizing lower bound to query.
 * @num_clusters:        Total cluster count.
 * @active_mask:         Active cluster flags.
 * @evaluated_clusters:  Array of already evaluated cluster IDs.
 * @evaluated_dists:     Measured distances to evaluated anchors.
 * @num_eval:            Number of evaluated anchors.
 * @dcc_matrix:          Dense M x M inter-cluster distance matrix.
 *
 * Return: Selected cluster index, or -1 if none available.
 */
static int select_best_first_target(
    int            num_clusters,
    const uint8_t *active_mask,
    const int     *evaluated_clusters,
    const double  *evaluated_dists,
    int            num_eval,
    const double  *dcc_matrix)
{
    int    best_target = -1;
    double min_max_lb = 1e30;

    for (int c = 0; c < num_clusters; c++)
    {
        if (active_mask[c] == 0)
        {
            continue;
        }

        double max_lb = 0.0;
        for (int e = 0; e < num_eval; e++)
        {
            int    eval_c = evaluated_clusters[e];
            double d_p = evaluated_dists[e];
            double dcc = dcc_matrix[eval_c * num_clusters + c];
            double lb = fabs(d_p - dcc);
            if (lb > max_lb)
            {
                max_lb = lb;
            }
        }

        if (max_lb < min_max_lb)
        {
            min_max_lb = max_lb;
            best_target = c;
        }
    }

    return best_target;
}

/**
 * get_anchor_sq16() - Resolve pointer to SQ16 anchor vector.
 * @config:         Active locator configuration.
 * @c:              Cluster index.
 * @frame_elements: Element count per vector.
 *
 * Return: Pointer to SQ16 anchor buffer, or NULL if unavailable.
 */
static inline const int16_t *get_anchor_sq16(
    const ClusterLocatorConfig *config,
    int                         c,
    long                        frame_elements)
{
    if (config->anchors_sq16_buf != NULL)
    {
        return config->anchors_sq16_buf + (size_t)c * (size_t)frame_elements;
    }
    if (config->anchors_sq16_ptrs != NULL)
    {
        return config->anchors_sq16_ptrs[c];
    }
    return NULL;
}

/**
 * get_anchor_sq8() - Resolve pointer to SQ8 anchor vector.
 * @config:         Active locator configuration.
 * @c:              Cluster index.
 * @frame_elements: Element count per vector.
 *
 * Return: Pointer to SQ8 anchor buffer, or NULL if unavailable.
 */
static inline const uint8_t *get_anchor_sq8(
    const ClusterLocatorConfig *config,
    int                         c,
    long                        frame_elements)
{
    if (config->anchors_sq8_buf != NULL)
    {
        return config->anchors_sq8_buf + (size_t)c * (size_t)frame_elements;
    }
    if (config->anchors_sq8_ptrs != NULL)
    {
        return config->anchors_sq8_ptrs[c];
    }
    return NULL;
}

/**
 * cluster_locate_sample() - Locates the matching/closest cluster for sample q.
 * @query_data:       Pointer to sample/query frame vector.
 * @frame_elements:   Number of elements per vector.
 * @num_clusters:     Total clusters M in model.
 * @cluster_anchors:  Array of pointers to anchor vectors [M].
 * @cluster_radii:    Array of cluster radii [M].
 * @dcc_matrix:       Dense M x M inter-cluster distance matrix.
 * @config:           Tuning and runtime configuration.
 * @result:           Output structure to populate.
 *
 * Return: CLUSTER_LOCATE_SUCCESS (0), CLUSTER_LOCATE_REJECTED (1), or error (-1).
 */
int cluster_locate_sample(
    const void                 *query_data,
    long                        frame_elements,
    int                         num_clusters,
    const void *const          *cluster_anchors,
    const double               *cluster_radii,
    const double               *dcc_matrix,
    const ClusterLocatorConfig *config,
    ClusterLocatorResult       *result)
{
    if (query_data == NULL || cluster_anchors == NULL || cluster_radii == NULL ||
        dcc_matrix == NULL || config == NULL || result == NULL || num_clusters <= 0)
    {
        return CLUSTER_LOCATE_ERROR;
    }

    int max_eval = config->max_targets > 0 ? config->max_targets : 8;
    if (max_eval > 32)
    {
        max_eval = 32;
    }

    double eps_factor = 1.0 + (config->epsilon > 0.0 ? config->epsilon : 0.0);
    double tau_eff = (config->tau_max > 0.0) ? (config->tau_max / eps_factor) : 1e20;
    if (config->rlim > 0.0 && config->rlim < tau_eff)
    {
        tau_eff = config->rlim;
    }

    result->best_cluster_id = -1;
    result->best_anchor_dist = 1e20;
    result->num_evaluated_anchors = 0;

    if (result->active_cluster_mask != NULL)
    {
        memset(result->active_cluster_mask, 1, (size_t)num_clusters);
    }
    else
    {
        return CLUSTER_LOCATE_ERROR;
    }

    uint64_t sq16_ssd_thresh = UINT64_MAX;
    int sq16_active = 0;
    if (config->query_sq16 != NULL && config->sq16_params != NULL &&
        (config->anchors_sq16_buf != NULL || config->anchors_sq16_ptrs != NULL))
    {
        double raw_thresh = (tau_eff + 2.0 * (double)config->sq16_params->err_radius) *
                            (double)config->sq16_params->inv_scale;
        if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
        {
            sq16_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
            sq16_active = 1;
        }
    }

    uint64_t sq8_ssd_thresh = UINT64_MAX;
    int sq8_active = 0;
    if (!sq16_active && config->query_sq8 != NULL && config->sq8_params != NULL &&
        (config->anchors_sq8_buf != NULL || config->anchors_sq8_ptrs != NULL))
    {
        double raw_thresh = (tau_eff + 2.0 * (double)config->sq8_params->err_radius) *
                            (double)config->sq8_params->inv_scale;
        if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
        {
            sq8_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
            sq8_active = 1;
        }
    }

    // Step 0: Trajectory Warm-Starting (Attempt 0)
    if (config->prev_cluster_id >= 0 && config->prev_cluster_id < num_clusters)
    {
        int p_id = config->prev_cluster_id;
        int skip_prev = 0;

        if (sq16_active)
        {
            const int16_t *a_sq16 = get_anchor_sq16(config, p_id, frame_elements);
            if (a_sq16 != NULL)
            {
                uint64_t ssd = sq16_dist_squared_cutoff_i16(
                    config->query_sq16, a_sq16, frame_elements, sq16_ssd_thresh);
                if (ssd > sq16_ssd_thresh)
                {
                    result->active_cluster_mask[p_id] = 0;
                    skip_prev = 1;
                }
            }
        }
        else if (sq8_active)
        {
            const uint8_t *a_sq8 = get_anchor_sq8(config, p_id, frame_elements);
            if (a_sq8 != NULL)
            {
                uint64_t ssd = sq8_dist_squared_u8(config->query_sq8, a_sq8, frame_elements);
                if (ssd > sq8_ssd_thresh)
                {
                    result->active_cluster_mask[p_id] = 0;
                    skip_prev = 1;
                }
            }
        }

        if (!skip_prev)
        {
            double d_prev = compute_vector_distance(
                query_data, cluster_anchors[p_id], frame_elements, config->is_double);

            result->evaluated_clusters[0] = p_id;
            result->evaluated_dists[0] = d_prev;
            result->num_evaluated_anchors = 1;
            result->best_cluster_id = p_id;
            result->best_anchor_dist = d_prev;
            result->active_cluster_mask[p_id] = 0;

            if (d_prev < tau_eff)
            {
                tau_eff = d_prev;
                if (sq16_active)
                {
                    double raw_thresh = (tau_eff +
                                         2.0 * (double)config->sq16_params->err_radius) *
                                        (double)config->sq16_params->inv_scale;
                    if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
                    {
                        sq16_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
                    }
                }
                else if (sq8_active)
                {
                    double raw_thresh = (tau_eff +
                                         2.0 * (double)config->sq8_params->err_radius) *
                                        (double)config->sq8_params->inv_scale;
                    if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
                    {
                        sq8_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
                    }
                }
            }

            if (config->rlim > 0.0 && d_prev <= config->rlim)
            {
                return CLUSTER_LOCATE_SUCCESS;
            }

            // Apply 3P metric pruning against prev_cluster
            for (int c = 0; c < num_clusters; c++)
            {
                if (c == p_id || result->active_cluster_mask[c] == 0)
                {
                    continue;
                }
                double dcc = dcc_matrix[p_id * num_clusters + c];
                double r_c = cluster_radii[c];
                double lb1 = dcc - r_c - d_prev;
                double lb2 = d_prev - dcc - r_c;
                double lb = (lb1 > lb2) ? lb1 : lb2;

                if (lb >= tau_eff)
                {
                    result->active_cluster_mask[c] = 0;
                }
            }
        }
    }

    // Fast initial screening pass with SQ16/SQ8 when strict_rlim is enabled
    int best_quant_candidate = -1;
    if (config->strict_rlim && config->rlim > 0.0)
    {
        if (sq16_active)
        {
            uint64_t min_sq_ssd = UINT64_MAX;
            int      active_left = 0;
            for (int c = 0; c < num_clusters; c++)
            {
                if (result->active_cluster_mask[c] == 0)
                {
                    continue;
                }
                const int16_t *a_sq16 = get_anchor_sq16(config, c, frame_elements);
                if (a_sq16 != NULL)
                {
                    uint64_t ssd = sq16_dist_squared_cutoff_i16(
                        config->query_sq16, a_sq16, frame_elements, sq16_ssd_thresh);
                    if (ssd > sq16_ssd_thresh)
                    {
                        result->active_cluster_mask[c] = 0;
                    }
                    else
                    {
                        active_left++;
                        if (ssd < min_sq_ssd)
                        {
                            min_sq_ssd = ssd;
                            best_quant_candidate = c;
                        }
                    }
                }
            }
            if (active_left == 0)
            {
                result->best_cluster_id = -1;
                return CLUSTER_LOCATE_REJECTED;
            }
        }
        else if (sq8_active)
        {
            uint64_t min_sq_ssd = UINT64_MAX;
            int      active_left = 0;
            for (int c = 0; c < num_clusters; c++)
            {
                if (result->active_cluster_mask[c] == 0)
                {
                    continue;
                }
                const uint8_t *a_sq8 = get_anchor_sq8(config, c, frame_elements);
                if (a_sq8 != NULL)
                {
                    uint64_t ssd = sq8_dist_squared_u8(config->query_sq8, a_sq8, frame_elements);
                    if (ssd > sq8_ssd_thresh)
                    {
                        result->active_cluster_mask[c] = 0;
                    }
                    else
                    {
                        active_left++;
                        if (ssd < min_sq_ssd)
                        {
                            min_sq_ssd = ssd;
                            best_quant_candidate = c;
                        }
                    }
                }
            }
            if (active_left == 0)
            {
                result->best_cluster_id = -1;
                return CLUSTER_LOCATE_REJECTED;
            }
        }
    }

    // Step 1: Iterative Target Selection & Metric Bounding Loop
    while (result->num_evaluated_anchors < max_eval)
    {
        // Count active clusters
        int active_count = 0;
        for (int c = 0; c < num_clusters; c++)
        {
            if (result->active_cluster_mask[c])
            {
                active_count++;
            }
        }

        if (active_count == 0)
        {
            break; // All clusters pruned
        }

        // Select next measurement target
        int next_target = -1;
        if (result->num_evaluated_anchors == 0)
        {
            if (best_quant_candidate >= 0 && result->active_cluster_mask[best_quant_candidate])
            {
                next_target = best_quant_candidate;
            }
            else
            {
                for (int c = 0; c < num_clusters; c++)
                {
                    if (result->active_cluster_mask[c])
                    {
                        next_target = c;
                        break;
                    }
                }
            }
        }
        else
        {
            next_target = select_best_first_target(
                num_clusters, result->active_cluster_mask, result->evaluated_clusters,
                result->evaluated_dists, result->num_evaluated_anchors, dcc_matrix);
        }

        if (next_target < 0)
        {
            break;
        }

        // Measure distance to selected target
        double d_target = compute_vector_distance(
            query_data, cluster_anchors[next_target], frame_elements, config->is_double);

        int idx = result->num_evaluated_anchors;
        result->evaluated_clusters[idx] = next_target;
        result->evaluated_dists[idx] = d_target;
        result->num_evaluated_anchors++;
        result->active_cluster_mask[next_target] = 0;

        if (d_target < result->best_anchor_dist)
        {
            result->best_anchor_dist = d_target;
            result->best_cluster_id = next_target;
        }

        if (d_target < tau_eff)
        {
            tau_eff = d_target;
            if (sq16_active)
            {
                double raw_thresh = (tau_eff +
                                     2.0 * (double)config->sq16_params->err_radius) *
                                    (double)config->sq16_params->inv_scale;
                if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
                {
                    sq16_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
                }
            }
            else if (sq8_active)
            {
                double raw_thresh = (tau_eff +
                                     2.0 * (double)config->sq8_params->err_radius) *
                                    (double)config->sq8_params->inv_scale;
                if (raw_thresh > 0.0 && raw_thresh < 4294967295.0)
                {
                    sq8_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
                }
            }
        }

        // Satisficing exit: matched cluster within rlim
        if ((config->rlim > 0.0 && d_target <= config->rlim) || d_target < 1e-6)
        {
            return CLUSTER_LOCATE_SUCCESS;
        }

        // 3P, 4P, and 5P Metric Pruning against measured target
        for (int c = 0; c < num_clusters; c++)
        {
            if (result->active_cluster_mask[c] == 0 || c == next_target)
            {
                continue;
            }

            double dcc = dcc_matrix[next_target * num_clusters + c];
            double r_c = cluster_radii[c];
            double lb1 = dcc - r_c - d_target;
            double lb2 = d_target - dcc - r_c;
            double lb = (lb1 > lb2) ? lb1 : lb2;

            if (lb >= tau_eff)
            {
                result->active_cluster_mask[c] = 0;
                continue;
            }

            // 4P (TE4) 2D Triangulation Metric Pruning
            if (config->te4_mode && result->num_evaluated_anchors >= 2)
            {
                for (int p = 0; p < result->num_evaluated_anchors - 1; p++)
                {
                    int    p_prev = result->evaluated_clusters[p];
                    double d_p_prev = result->evaluated_dists[p];
                    double d_p_t = dcc_matrix[next_target * num_clusters + p_prev];
                    double d_t_c = dcc;
                    double d_p_c = dcc_matrix[p_prev * num_clusters + c];

                    double min_d = calc_min_dist_4pt(d_target, d_p_prev, d_p_t, d_t_c, d_p_c);
                    if (min_d - r_c >= tau_eff)
                    {
                        result->active_cluster_mask[c] = 0;
                        break;
                    }
                }
            }

            // 5P (TE5) 3D Triangulation Metric Pruning
            if (config->te5_mode && result->active_cluster_mask[c] &&
                result->num_evaluated_anchors >= 3)
            {
                int eval_n = result->num_evaluated_anchors;
                for (int p1 = 0; p1 < eval_n - 2; p1++)
                {
                    for (int p2 = p1 + 1; p2 < eval_n - 1; p2++)
                    {
                        int    c1 = result->evaluated_clusters[p1];
                        double d_f_c1 = result->evaluated_dists[p1];
                        int    c2 = result->evaluated_clusters[p2];
                        double d_f_c2 = result->evaluated_dists[p2];
                        int    c3 = next_target;
                        double d_f_c3 = d_target;

                        double d_t_c1 = dcc_matrix[c * num_clusters + c1];
                        double d_t_c2 = dcc_matrix[c * num_clusters + c2];
                        double d_t_c3 = dcc;

                        double d_c1_c2 = dcc_matrix[c1 * num_clusters + c2];
                        double d_c1_c3 = dcc_matrix[c1 * num_clusters + c3];
                        double d_c2_c3 = dcc_matrix[c2 * num_clusters + c3];

                        double min_d5 = calc_min_dist_5pt(
                            d_f_c1, d_f_c2, d_f_c3,
                            d_t_c1, d_t_c2, d_t_c3,
                            d_c1_c2, d_c1_c3, d_c2_c3);

                        if (min_d5 - r_c >= tau_eff)
                        {
                            result->active_cluster_mask[c] = 0;
                            break;
                        }
                    }
                    if (result->active_cluster_mask[c] == 0)
                    {
                        break;
                    }
                }
            }
        } // for (int c = 0; ...)
    } // while (result->num_evaluated_anchors < max_eval)

    if (config->strict_rlim && config->rlim > 0.0)
    {
        if (result->best_cluster_id < 0 || result->best_anchor_dist > config->rlim)
        {
            result->best_cluster_id = -1;
            return CLUSTER_LOCATE_REJECTED;
        }
    }

    return CLUSTER_LOCATE_SUCCESS;
}

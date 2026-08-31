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
 * compute_vector_distance() - Computes Euclidean distance between two vectors.
 * @a: Pointer to vector a.
 * @b: Pointer to vector b.
 * @n: Number of elements.
 *
 * Return: Euclidean distance.
 */
static inline double compute_vector_distance(
    const double *restrict a,
    const double *restrict b,
    long                   n)
{
    double sum = 0.0;
    for (long i = 0; i < n; i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * select_greedy_max_spread_target() - Selects next target maximizing distance spread.
 * @num_clusters: Total cluster count.
 * @active_mask: Active cluster flags.
 * @evaluated_clusters: Array of already evaluated cluster IDs.
 * @num_eval: Number of evaluated anchors.
 * @dcc_matrix: Dense M x M inter-cluster distance matrix.
 *
 * Return: Selected cluster index, or -1 if none available.
 */
static int select_greedy_max_spread_target(
    int            num_clusters,
    const uint8_t *active_mask,
    const int     *evaluated_clusters,
    int            num_eval,
    const double  *dcc_matrix)
{
    int    best_target = -1;
    double max_min_dcc = -1.0;

    for (int c = 0; c < num_clusters; c++)
    {
        if (active_mask[c] == 0)
        {
            continue;
        }

        // Compute minimum distance to any already evaluated anchor
        double min_dist_to_eval = 1e20;
        for (int e = 0; e < num_eval; e++)
        {
            int    eval_c = evaluated_clusters[e];
            double dcc = dcc_matrix[eval_c * num_clusters + c];
            if (dcc < min_dist_to_eval)
            {
                min_dist_to_eval = dcc;
            }
        }

        if (min_dist_to_eval > max_min_dcc)
        {
            max_min_dcc = min_dist_to_eval;
            best_target = c;
        }
    }

    return best_target;
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
 * Return: 0 on success, -1 on error.
 */
int cluster_locate_sample(
    const double               *query_data,
    long                        frame_elements,
    int                         num_clusters,
    const double *const        *cluster_anchors,
    const double               *cluster_radii,
    const double               *dcc_matrix,
    const ClusterLocatorConfig *config,
    ClusterLocatorResult       *result)
{
    if (query_data == NULL || cluster_anchors == NULL || cluster_radii == NULL ||
        dcc_matrix == NULL || config == NULL || result == NULL || num_clusters <= 0)
    {
        return -1;
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
        return -1;
    }

    // Step 0: Trajectory Warm-Starting (Attempt 0)
    if (config->prev_cluster_id >= 0 && config->prev_cluster_id < num_clusters)
    {
        int    p_id = config->prev_cluster_id;
        double d_prev =
            compute_vector_distance(query_data, cluster_anchors[p_id], frame_elements);

        result->evaluated_clusters[0] = p_id;
        result->evaluated_dists[0] = d_prev;
        result->num_evaluated_anchors = 1;
        result->best_cluster_id = p_id;
        result->best_anchor_dist = d_prev;

        if (d_prev < tau_eff)
        {
            tau_eff = d_prev;
        }

        // Apply 3P metric pruning against prev_cluster
        for (int c = 0; c < num_clusters; c++)
        {
            if (c == p_id)
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

        if (config->rlim > 0.0 && d_prev <= config->rlim)
        {
            return 0; // Matched within rlim on attempt 0!
        }
    }

    // Step 1: Iterative Target Selection & 3P Bounding Loop
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

        if (active_count <= 1)
        {
            break; // Resolved to single or zero candidate
        }

        // Select next measurement target
        int next_target = -1;
        if (result->num_evaluated_anchors == 0)
        {
            // Initial anchor: Cluster 0 or medoid
            next_target = 0;
        }
        else
        {
            next_target = select_greedy_max_spread_target(
                num_clusters, result->active_cluster_mask, result->evaluated_clusters,
                result->num_evaluated_anchors, dcc_matrix);
        }

        if (next_target < 0)
        {
            break;
        }

        // Measure distance to selected target
        double d_target = compute_vector_distance(query_data, cluster_anchors[next_target],
                                                 frame_elements);

        int idx = result->num_evaluated_anchors;
        result->evaluated_clusters[idx] = next_target;
        result->evaluated_dists[idx] = d_target;
        result->num_evaluated_anchors++;

        if (d_target < result->best_anchor_dist)
        {
            result->best_anchor_dist = d_target;
            result->best_cluster_id = next_target;
        }

        if (d_target < tau_eff)
        {
            tau_eff = d_target;
        }

        // 3P Metric Pruning against measured target
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
            }
        }

        if (config->rlim > 0.0 && result->best_anchor_dist <= config->rlim)
        {
            break; // Resolved within rlim
        }
    }

    return 0;
}

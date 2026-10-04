/**
 * @file cluster_bounds.c
 * @brief Implementation of distance bound propagation using the triangle inequality.
 */

#include "cluster_bounds.h"
#include "cluster_math.h"
#include "cluster_core.h"
#include <math.h>
#include <stdlib.h>

/**
 * update_dcc_bounds() - Update dcc_min/dcc_max and propagate bounds to other clusters.
 * @state: Running state of the clustering execution.
 * @config: Config parameters of the clustering execution.
 * @i: First cluster index.
 * @j: Second cluster index.
 * @d_exact: Exactly measured distance between i and j.
 *
 * Sets exact bounds for pair (i, j) and uses the triangle inequality to propagate
 * upper and lower bounds for all other active clusters in O(K) time.
 */
void update_dcc_bounds(
    ClusterState  *state,
    ClusterConfig *config,
    int            i,
    int            j,
    double         d_exact)
{
    (void)config;

    if (!dcc_is_measured(state, i, j))
    {
        state->telemetry.dcc_entries_populated++;
    }

    dcc_set_pair(state, i, j, d_exact);

    for (int k = 0; k < state->num_clusters; k++)
    {
        if (k == i || k == j)
        {
            continue;
        }

        double max_jk = dcc_get_max(state, j, k);
        double max_ik = dcc_get_max(state, i, k);
        double min_jk = dcc_get_min(state, j, k);
        double min_ik = dcc_get_min(state, i, k);

        // Upper Bound Refinement for (i, k) using (j, k)
        if (max_jk < 1e18)
        {
            double new_max = d_exact + max_jk;
            if (new_max < max_ik)
            {
                dcc_set_max_pair(state, i, k, new_max);
                max_ik = new_max;
            }
        }

        // Upper Bound Refinement for (j, k) using (i, k)
        if (max_ik < 1e18)
        {
            double new_max = d_exact + max_ik;
            if (new_max < max_jk)
            {
                dcc_set_max_pair(state, j, k, new_max);
                max_jk = new_max;
            }
        }

        // Lower Bound Refinement for (i, k) using (j, k)
        if (max_jk < 1e18)
        {
            double l1 = d_exact - max_jk;
            if (l1 > 0.0 && l1 > min_ik)
            {
                dcc_set_min_pair(state, i, k, l1);
                min_ik = l1;
            }
        }
        double l2 = min_jk - d_exact;
        if (l2 > 0.0 && l2 > min_ik)
        {
            dcc_set_min_pair(state, i, k, l2);
            min_ik = l2;
        }

        // Lower Bound Refinement for (j, k) using (i, k)
        if (max_ik < 1e18)
        {
            double l3 = d_exact - max_ik;
            if (l3 > 0.0 && l3 > min_jk)
            {
                dcc_set_min_pair(state, j, k, l3);
                min_jk = l3;
            }
        }
        double l4 = min_ik - d_exact;
        if (l4 > 0.0 && l4 > min_jk)
        {
            dcc_set_min_pair(state, j, k, l4);
            min_jk = l4;
        }
    }
}

/**
 * refine_sparse_bounds() - Select and measure the closest unmeasured active cluster pairs.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 *
 * Scans all active cluster pairs, gathers unmeasured ones, sorts them by dcc_min
 * in ascending order, and computes/propagates exact distances for the top E pairs.
 */
void refine_sparse_bounds(
    ClusterConfig *config,
    ClusterState  *state)
{
    int E = config->optim.sparse_dcc_extra_evals;
    int K = state->num_clusters;
    if (E <= 0 || K <= 1)
    {
        return;
    }

    uint64_t total_pairs = (uint64_t)K * (K - 1) / 2;
    if (state->telemetry.dcc_entries_populated >= total_pairs)
    {
        return;
    }

    int Q = state->scratch.refine_queue_capacity;
    if (Q <= 0)
    {
        Q = 1024;
    }

    /* Rebuild the queue if the number of clusters has changed (topology changed)
     * or if we do not have enough elements left to satisfy E requests. */
    if (state->num_clusters != state->scratch.refine_queue_last_num_clusters ||
        state->scratch.refine_queue_idx + E > state->scratch.refine_queue_size)
    {
        Candidate best_pairs[Q];
        for (int k = 0; k < Q; k++)
        {
            best_pairs[k].id = -1;
            best_pairs[k].p = -1e30;
        }

        for (int r = 1; r < state->num_clusters; r++)
        {
            const char   *measured_row = dcc_row_measured(state, r);
            const double *dcc_min_row = dcc_row_dist(state, r);
            for (int c = 0; c < r; c++)
            {
                int is_meas = (measured_row != NULL) ? (int)measured_row[c]
                                                     : (dcc_min_row[c] >= 0.0);
                if (!is_meas)
                {
                    double dcc_val = dcc_min_row[c];
                    double score = -dcc_val;

                    if (score > best_pairs[Q - 1].p)
                    {
                        int k = Q - 2;
                        while (k >= 0 && score > best_pairs[k].p)
                        {
                            best_pairs[k + 1] = best_pairs[k];
                            k--;
                        }
                        best_pairs[k + 1].id = (c << 16) | r;
                        best_pairs[k + 1].p = score;
                    }
                }
            }
        }

        /* Copy back to scratch queue */
        int count = 0;
        for (int k = 0; k < Q; k++)
        {
            if (best_pairs[k].id != -1)
            {
                state->scratch.refine_queue[count] = best_pairs[k];
                count++;
            }
        }
        state->scratch.refine_queue_size = count;
        state->scratch.refine_queue_idx = 0;
        state->scratch.refine_queue_last_num_clusters = state->num_clusters;
    }

    /* Pop the next E candidates from the queue */
    int found = 0;
    for (int idx = 0; idx < E; idx++)
    {
        int q_idx = state->scratch.refine_queue_idx + idx;
        if (q_idx < state->scratch.refine_queue_size)
        {
            found++;
        }
    }

    if (found <= 0)
    {
        return;
    }

    double distances[E];
    for (int idx = 0; idx < found; idx++)
    {
        int q_idx = state->scratch.refine_queue_idx + idx;
        int i = state->scratch.refine_queue[q_idx].id >> 16;
        int j = state->scratch.refine_queue[q_idx].id & 0xFFFF;

        distances[idx] = get_dist(
            &state->clusters[i].anchor,
            &state->clusters[j].anchor,
            -1,
            -1.0,
            -1.0,
            config,
            state);
    }

    for (int idx = 0; idx < found; idx++)
    {
        int q_idx = state->scratch.refine_queue_idx + idx;
        int i = state->scratch.refine_queue[q_idx].id >> 16;
        int j = state->scratch.refine_queue[q_idx].id & 0xFFFF;
        update_dcc_bounds(state, config, i, j, distances[idx]);
    }

    state->scratch.refine_queue_idx += found;
}

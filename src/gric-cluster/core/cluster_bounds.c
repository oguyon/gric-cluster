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
    int N = config->algo.maxnbclust;

    state->scratch.dcc_min[i * N + j] = d_exact;
    state->scratch.dcc_min[j * N + i] = d_exact;
    state->scratch.dcc_max[i * N + j] = d_exact;
    state->scratch.dcc_max[j * N + i] = d_exact;
    state->scratch.dcc_measured[i * N + j] = 1;
    state->scratch.dcc_measured[j * N + i] = 1;

    for (int k = 0; k < state->num_clusters; k++)
    {
        if (k == i || k == j)
        {
            continue;
        }

        // Upper Bound Refinement for (i, k) using (j, k)
        if (state->scratch.dcc_max[j * N + k] < 1e18)
        {
            double new_max = d_exact + state->scratch.dcc_max[j * N + k];
            if (new_max < state->scratch.dcc_max[i * N + k])
            {
                state->scratch.dcc_max[i * N + k] = new_max;
                state->scratch.dcc_max[k * N + i] = new_max;
            }
        }

        // Upper Bound Refinement for (j, k) using (i, k)
        if (state->scratch.dcc_max[i * N + k] < 1e18)
        {
            double new_max = d_exact + state->scratch.dcc_max[i * N + k];
            if (new_max < state->scratch.dcc_max[j * N + k])
            {
                state->scratch.dcc_max[j * N + k] = new_max;
                state->scratch.dcc_max[k * N + j] = new_max;
            }
        }

        // Lower Bound Refinement for (i, k) using (j, k)
        if (state->scratch.dcc_max[j * N + k] < 1e18)
        {
            double l1 = d_exact - state->scratch.dcc_max[j * N + k];
            if (l1 > 0.0 && l1 > state->scratch.dcc_min[i * N + k])
            {
                state->scratch.dcc_min[i * N + k] = l1;
                state->scratch.dcc_min[k * N + i] = l1;
            }
        }
        double l2 = state->scratch.dcc_min[j * N + k] - d_exact;
        if (l2 > 0.0 && l2 > state->scratch.dcc_min[i * N + k])
        {
            state->scratch.dcc_min[i * N + k] = l2;
            state->scratch.dcc_min[k * N + i] = l2;
        }

        // Lower Bound Refinement for (j, k) using (i, k)
        if (state->scratch.dcc_max[i * N + k] < 1e18)
        {
            double l3 = d_exact - state->scratch.dcc_max[i * N + k];
            if (l3 > 0.0 && l3 > state->scratch.dcc_min[j * N + k])
            {
                state->scratch.dcc_min[j * N + k] = l3;
                state->scratch.dcc_min[k * N + j] = l3;
            }
        }
        double l4 = state->scratch.dcc_min[i * N + k] - d_exact;
        if (l4 > 0.0 && l4 > state->scratch.dcc_min[j * N + k])
        {
            state->scratch.dcc_min[j * N + k] = l4;
            state->scratch.dcc_min[k * N + j] = l4;
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
    int N = config->algo.maxnbclust;
    int E = config->optim.sparse_dcc_extra_evals;
    if (E <= 0 || state->num_clusters <= 1)
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

        #pragma omp parallel
        {
            Candidate local_best[Q];
            for (int k = 0; k < Q; k++)
            {
                local_best[k].id = -1;
                local_best[k].p = -1e30;
            }

            #pragma omp for nowait
            for (int i = 0; i < state->num_clusters; i++)
            {
                char *measured_row = &state->scratch.dcc_measured[i * N];
                double *dcc_min_row = &state->scratch.dcc_min[i * N];
                for (int j = i + 1; j < state->num_clusters; j++)
                {
                    if (!measured_row[j])
                    {
                        double dcc_val = dcc_min_row[j];
                        double score = -dcc_val;

                        if (score > local_best[Q - 1].p)
                        {
                            int k = Q - 2;
                            while (k >= 0 && score > local_best[k].p)
                            {
                                local_best[k + 1] = local_best[k];
                                k--;
                            }
                            local_best[k + 1].id = (i << 16) | j;
                            local_best[k + 1].p = score;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int idx = 0; idx < Q; idx++)
                {
                    if (local_best[idx].id == -1)
                    {
                        continue;
                    }
                    double score = local_best[idx].p;
                    if (score > best_pairs[Q - 1].p)
                    {
                        int k = Q - 2;
                        while (k >= 0 && score > best_pairs[k].p)
                        {
                            best_pairs[k + 1] = best_pairs[k];
                            k--;
                        }
                        best_pairs[k + 1] = local_best[idx];
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
    #pragma omp parallel for if(found >= 2)
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

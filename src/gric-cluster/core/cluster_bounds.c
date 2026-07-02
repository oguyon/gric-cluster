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

    int max_pairs = state->num_clusters * (state->num_clusters - 1) / 2;
    Candidate *unmeasured_pairs = (Candidate *)malloc(max_pairs * sizeof(Candidate));
    if (!unmeasured_pairs)
    {
        return;
    }

    int pair_count = 0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        for (int j = i + 1; j < state->num_clusters; j++)
        {
            if (!state->scratch.dcc_measured[i * N + j])
            {
                unmeasured_pairs[pair_count].id = (i << 16) | j;
                unmeasured_pairs[pair_count].p = -state->scratch.dcc_min[i * N + j];
                pair_count++;
            }
        }
    }

    if (pair_count > 0)
    {
        qsort(unmeasured_pairs, pair_count, sizeof(Candidate), compare_candidates);

        int limit = (E < pair_count) ? E : pair_count;
        for (int idx = 0; idx < limit; idx++)
        {
            int i = unmeasured_pairs[idx].id >> 16;
            int j = unmeasured_pairs[idx].id & 0xFFFF;

            double d = get_dist(&state->clusters[i].anchor,
                                &state->clusters[j].anchor, -1, -1.0, -1.0,
                                config, state);
            update_dcc_bounds(state, config, i, j, d);
        }
    }

    free(unmeasured_pairs);
}

/**
 * @file cluster_bounds.c
 * @brief Implementation of distance bound propagation using the triangle inequality.
 */

#include "cluster_bounds.h"
#include "cluster_math.h"
#include "cluster_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * update_dcc_bounds() - Update dcc_min/dcc_max and propagate bounds to other clusters.
 * @state:   Running state of the clustering execution.
 * @config:  Config parameters of the clustering execution.
 * @i:       First cluster index.
 * @j:       Second cluster index.
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

    int num_clusters = state->num_clusters;
    for (int k = 0; k < num_clusters; k++)
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
    } // for (int k = 0; k < num_clusters; k++)
}

/**
 * refine_queue_insert() - Insert an unmeasured pair into sorted refine queue.
 * @queue:    Array of candidate items.
 * @size:     Pointer to current number of items in queue.
 * @capacity: Maximum capacity of the queue.
 * @id:       Encoded pair identifier ((c << 16) | r).
 * @score:    Candidate priority score (-dcc_val).
 *
 * Maintains the candidate queue sorted descending by score (ascending by bound distance).
 * Drops the worst candidate if the queue is full and the new score is better.
 */
static inline void refine_queue_insert(
    Candidate *queue,
    int       *size,
    int        capacity,
    int        id,
    double     score)
{
    int count = *size;
    if (count == capacity && score <= queue[capacity - 1].p)
    {
        return;
    }

    int low = 0;
    int high = count;
    while (low < high)
    {
        int mid = low + (high - low) / 2;
        if (queue[mid].p < score)
        {
            high = mid;
        }
        else
        {
            low = mid + 1;
        }
    }
    int pos = low;

    if (count < capacity)
    {
        if (count > pos)
        {
            memmove(&queue[pos + 1], &queue[pos], (size_t)(count - pos) * sizeof(Candidate));
        }
        queue[pos].id = id;
        queue[pos].p = score;
        (*size)++;
    }
    else
    {
        if (capacity - 1 > pos)
        {
            memmove(&queue[pos + 1], &queue[pos],
                    (size_t)(capacity - 1 - pos) * sizeof(Candidate));
        }
        queue[pos].id = id;
        queue[pos].p = score;
    }
}

/**
 * refine_sparse_bounds() - Select and measure the closest unmeasured active cluster pairs.
 * @config: Config parameters of the clustering execution.
 * @state:  Running state of the clustering execution.
 *
 * Incrementally updates the priority queue with unmeasured pairs from new cluster rows,
 * and computes exact distances and triangle inequality updates for the top E pairs.
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

    /* Compact existing remaining candidates to the front of the queue */
    int rem = state->scratch.refine_queue_size - state->scratch.refine_queue_idx;
    if (rem > 0 && state->scratch.refine_queue_idx > 0)
    {
        memmove(&state->scratch.refine_queue[0],
                &state->scratch.refine_queue[state->scratch.refine_queue_idx],
                (size_t)rem * sizeof(Candidate));
    }
    else if (rem < 0)
    {
        rem = 0;
    }
    state->scratch.refine_queue_idx = 0;
    state->scratch.refine_queue_size = rem;

    /* Incrementally scan only newly created cluster rows */
    int last_K = state->scratch.refine_queue_last_num_clusters;
    if (last_K < K)
    {
        int start_r = (last_K > 1) ? last_K : 1;
        for (int r = start_r; r < K; r++)
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
                    refine_queue_insert(state->scratch.refine_queue,
                                        &state->scratch.refine_queue_size,
                                        Q,
                                        (c << 16) | r,
                                        -dcc_val);
                }
            }
        }
        state->scratch.refine_queue_last_num_clusters = K;
    }

    /* If queue is depleted below E candidates, perform full scan to repopulate */
    if (state->scratch.refine_queue_size < E)
    {
        state->scratch.refine_queue_size = 0;
        state->scratch.refine_queue_idx = 0;
        for (int r = 1; r < K; r++)
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
                    refine_queue_insert(state->scratch.refine_queue,
                                        &state->scratch.refine_queue_size,
                                        Q,
                                        (c << 16) | r,
                                        -dcc_val);
                }
            }
        }
        state->scratch.refine_queue_last_num_clusters = K;
    }

    /* Pop and measure up to E candidates from the queue */
    int found = 0;
    while (state->scratch.refine_queue_idx < state->scratch.refine_queue_size && found < E)
    {
        int q_idx = state->scratch.refine_queue_idx++;
        int i = state->scratch.refine_queue[q_idx].id >> 16;
        int j = state->scratch.refine_queue[q_idx].id & 0xFFFF;

        if (dcc_is_measured(state, i, j))
        {
            continue;
        }

        double d = get_dist(
            &state->clusters[i].anchor,
            &state->clusters[j].anchor,
            -1,
            -1.0,
            -1.0,
            config,
            state);
        update_dcc_bounds(state, config, i, j, d);
        found++;
    }
}

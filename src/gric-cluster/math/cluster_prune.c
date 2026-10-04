/**
 * @file cluster_prune.c
 * @brief Geometric and trajectory pruning optimization.
 *
 * Implements trajectory prediction routines and the 5-point triangle inequality bounds checking
 * to skip distance evaluations against distant clusters.
 *
 * Main Functions:
 * - get_prediction_candidates: Predicts future cluster matches based on geometric trajectory.
 * - prune_candidates_te5: Filters out candidates using 5-point triangle inequality.
 */
#include "cluster_prune.h"
#include "cluster_core.h"
#include "cluster_dcc.h"
#include "cluster_math.h"
#include "cluster_locator.h"
#include "gric_simd.h"
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * get_prediction_candidates() - Produce a ranked list of
 *     predicted next-cluster candidates.
 * @state:          Current clustering state (assignments,
 *                  telemetry).
 * @config:         Clustering configuration (pred_len,
 *                  pred_h, pred_n).
 * @candidates:     Output array of cluster indices, sorted
 *                  by descending match count.
 * @max_candidates: Maximum number of candidates to return.
 *
 * Uses transition history and trajectory matching: searches
 * recent assignment history for subsequences matching the
 * last @config->optim.pred_len assignments, then tallies
 * which cluster followed each match.  The top candidates
 * are returned sorted by frequency.
 *
 * Return: Number of candidates written (0 if none found).
 */
int get_prediction_candidates(
    ClusterState  *state,
    ClusterConfig *config,
    int           *candidates,
    int            max_candidates)
{
    long total = state->telemetry.total_frames_processed;
    int len = config->optim.pred_len;
    int h = config->optim.pred_h;

    if (total < len)
    {
        return 0;
    }

    long search_limit = total - len;
    long search_start = (total > h) ? total - h : 0;
    if (search_start > search_limit)
    {
        search_start = search_limit;
    }

    int *pattern = &state->assignments[total - len];

    /* Thread-local pre-allocated buffers to avoid heap allocations inside the loop */
    static __thread int *counts = NULL;
    static __thread Candidate *cand_list = NULL;
    static __thread int *touched = NULL;
    static __thread int allocated_size = 0;

    int maxcl = config->algo.maxnbclust;
    if (counts == NULL || allocated_size < maxcl)
    {
        free(counts);
        free(cand_list);
        free(touched);
        counts = (int *)calloc((size_t)maxcl, sizeof(int));
        cand_list = (Candidate *)malloc((size_t)maxcl * sizeof(Candidate));
        touched = (int *)malloc((size_t)maxcl * sizeof(int));
        allocated_size = maxcl;
    }

    if (counts == NULL || cand_list == NULL || touched == NULL)
    {
        return 0;
    }

    int num_touched = 0;
    int pat0 = pattern[0];
    int pat1 = (len > 1) ? pattern[1] : 0;
    int pat2 = (len > 2) ? pattern[2] : 0;

    for (long i = search_start; i < search_limit; i++)
    {
        if (state->assignments[i] == pat0)
        {
            int match = 1;
            if (len == 2)
            {
                match = (state->assignments[i + 1] == pat1);
            }
            else if (len == 3)
            {
                match = (state->assignments[i + 1] == pat1 &&
                         state->assignments[i + 2] == pat2);
            }
            else
            {
                for (int k = 1; k < len; k++)
                {
                    if (state->assignments[i + k] != pattern[k])
                    {
                        match = 0;
                        break;
                    }
                }
            }

            if (match)
            {
                int next_cluster = state->assignments[i + len];
                if (next_cluster >= 0 && next_cluster < state->num_clusters)
                {
                    if (counts[next_cluster] == 0)
                    {
                        touched[num_touched++] = next_cluster;
                    }
                    counts[next_cluster]++;
                }
            }
        }
    }

    if (num_touched == 0)
    {
        return 0;
    }

    /* Sort touched cluster indices in ascending order to preserve canonical iteration order */
    for (int k = 1; k < num_touched; k++)
    {
        int key = touched[k];
        int j = k - 1;
        while (j >= 0 && touched[j] > key)
        {
            touched[j + 1] = touched[j];
            j--;
        }
        touched[j + 1] = key;
    }

    for (int k = 0; k < num_touched; k++)
    {
        int cl_idx = touched[k];
        cand_list[k].id = cl_idx;
        cand_list[k].p = (double)counts[cl_idx];
        counts[cl_idx] = 0;
    }

    qsort(cand_list, (size_t)num_touched, sizeof(Candidate), compare_candidates);

    int n_out = (num_touched < max_candidates) ? num_touched : max_candidates;
    for (int cand_idx = 0; cand_idx < n_out; cand_idx++)
    {
        candidates[cand_idx] = cand_list[cand_idx].id;
    }

    return n_out;
}

/**
 * prune_candidates_te5() - Prune candidates using the
 *     5-point triangle inequality.
 * @config:       Clustering configuration (rlim, te5_mode,
 *                maxnbclust, sparse_dcc_mode).
 * @state:        Clustering state (clusters, scratch,
 *                telemetry).
 * @temp_indices: Array of cluster indices measured so far.
 * @temp_dists:   Array of frame-to-cluster distances
 *                corresponding to @temp_indices.
 * @temp_count:   Number of entries in @temp_indices /
 *                @temp_dists.
 *
 * For every triplet of already-measured reference clusters
 * (c1, c2, c3), computes a lower bound on the distance
 * from the current frame to each remaining candidate
 * cluster using the 5-point triangle inequality.  If the
 * bound exceeds rlim the candidate is pruned (clmembflag
 * set to 0).
 *
 * Requires at least 3 measured clusters (temp_count >= 3).
 */
void prune_candidates_te5(
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count)
{
    if (!config->optim.te5_mode || temp_count < 3)
        return;

    int c3 = temp_indices[temp_count - 1]; // Current cluster (newest anchor)
    double d_f_c3 = temp_dists[temp_count - 1];

    int max_te5_pairs = 16;
    int pair_count = 0;
    long total_pruned_te5 = 0;

    for (int p = 0; p < temp_count - 2 && pair_count < max_te5_pairs; p++)
    {
        for (int q = p + 1; q < temp_count - 1 && pair_count < max_te5_pairs; q++)
        {
            if (state->scratch.num_active_clusters <= 0)
            {
                break;
            }
            pair_count++;
            int c1 = temp_indices[p];
            double d_f_c1 = temp_dists[p];
            int c2 = temp_indices[q];
            double d_f_c2 = temp_dists[q];

            // Get inter-cluster distances (lazy load)
            double d_c1_c2 = 0.0;
            double d_c1_c3 = 0.0;
            double d_c2_c3 = 0.0;

            if (config->optim.sparse_dcc_mode)
            {
                if (!dcc_is_measured(state, c1, c2) ||
                    !dcc_is_measured(state, c1, c3) ||
                    !dcc_is_measured(state, c2, c3))
                {
                    continue;
                }
                d_c1_c2 = dcc_get_dist(state, c1, c2);
                d_c1_c3 = dcc_get_dist(state, c1, c3);
                d_c2_c3 = dcc_get_dist(state, c2, c3);
            }
            else
            {
                d_c1_c2 = dcc_get_dist(state, c1, c2);
                if (d_c1_c2 < 0.0)
                {
                    d_c1_c2 = get_dist(&state->clusters[c1].anchor,
                                       &state->clusters[c2].anchor,
                                       -1, -1.0, -1.0, config, state);
                    dcc_set_pair(state, c1, c2, d_c1_c2);
                }

                d_c1_c3 = dcc_get_dist(state, c1, c3);
                if (d_c1_c3 < 0.0)
                {
                    d_c1_c3 = get_dist(&state->clusters[c1].anchor,
                                       &state->clusters[c3].anchor,
                                       -1, -1.0, -1.0, config, state);
                    dcc_set_pair(state, c1, c3, d_c1_c3);
                }

                d_c2_c3 = dcc_get_dist(state, c2, c3);
                if (d_c2_c3 < 0.0)
                {
                    d_c2_c3 = get_dist(&state->clusters[c2].anchor,
                                       &state->clusters[c3].anchor,
                                       -1, -1.0, -1.0, config, state);
                    dcc_set_pair(state, c2, c3, d_c2_c3);
                }
            }

            TE5Ref te5_ref;
            calc_te5_ref_init(
                &te5_ref, d_f_c1, d_f_c2, d_f_c3, d_c1_c2, d_c1_c3, d_c2_c3
            );

            int active_cnt = state->scratch.num_active_clusters;
            int *act = state->scratch.active_clusters;
            int idx = 0;

            while (idx < active_cnt)
            {
                int kk = act[idx];
                if (kk == c1 || kk == c2 || kk == c3)
                {
                    idx++;
                    continue;
                }

                double d_k_c1 = 0.0;
                double d_k_c2 = 0.0;
                double d_k_c3 = 0.0;

                if (config->optim.sparse_dcc_mode)
                {
                    if (!dcc_is_measured(state, c1, kk) ||
                        !dcc_is_measured(state, c2, kk) ||
                        !dcc_is_measured(state, c3, kk))
                    {
                        idx++;
                        continue;
                    }
                    d_k_c1 = dcc_get_dist(state, c1, kk);
                    d_k_c2 = dcc_get_dist(state, c2, kk);
                    d_k_c3 = dcc_get_dist(state, c3, kk);
                }
                else
                {
                    d_k_c1 = dcc_get_dist(state, c1, kk);
                    if (d_k_c1 < 0.0)
                    {
                        d_k_c1 = get_dist(
                            &state->clusters[kk].anchor,
                            &state->clusters[c1].anchor, -1, -1.0, -1.0,
                            config, state
                        );
                        dcc_set_pair(state, kk, c1, d_k_c1);
                    }

                    d_k_c2 = dcc_get_dist(state, c2, kk);
                    if (d_k_c2 < 0.0)
                    {
                        d_k_c2 = get_dist(
                            &state->clusters[kk].anchor,
                            &state->clusters[c2].anchor, -1, -1.0, -1.0,
                            config, state
                        );
                        dcc_set_pair(state, kk, c2, d_k_c2);
                    }

                    d_k_c3 = dcc_get_dist(state, c3, kk);
                    if (d_k_c3 < 0.0)
                    {
                        d_k_c3 = get_dist(
                            &state->clusters[kk].anchor,
                            &state->clusters[c3].anchor, -1, -1.0, -1.0,
                            config, state
                        );
                        dcc_set_pair(state, kk, c3, d_k_c3);
                    }
                }

                double min_d = calc_min_dist_5pt_ref(
                    &te5_ref, d_k_c1, d_k_c2, d_k_c3
                );
                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[kk] = 0;
                    state->scratch.entropy_p_current[kk] = 0.0;
                    total_pruned_te5++;
                    active_cnt--;
                    act[idx] = act[active_cnt];
                }
                else
                {
                    idx++;
                }
            } // while (idx < active_cnt)

            state->scratch.num_active_clusters = active_cnt;
        }
    }
    state->telemetry.clusters_pruned += total_pruned_te5;
}

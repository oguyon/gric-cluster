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
#include "cluster_math.h"
#include <stdlib.h>
#include <string.h>

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
        return 0;

    long search_limit = total - len;
    long search_start = (total > h) ? total - h : 0;
    if (search_start > search_limit)
        search_start = search_limit;

    int *pattern = &state->assignments[total - len];

    int *counts = (int *)calloc(state->num_clusters, sizeof(int));
    if (!counts)
        return 0;

    for (long i = search_start; i < search_limit; i++)
    {
        if (state->assignments[i] == pattern[0])
        {
            if (memcmp(&state->assignments[i], pattern, len * sizeof(int)) == 0)
            {
                int next_cluster = state->assignments[i + len];
                if (next_cluster >= 0 && next_cluster < state->num_clusters)
                {
                    counts[next_cluster]++;
                }
            }
        }
    }

    int count_non_zero = 0;
    for (int cl_idx = 0; cl_idx < state->num_clusters; cl_idx++)
        if (counts[cl_idx] > 0)
            count_non_zero++;

    if (count_non_zero == 0)
    {
        free(counts);
        return 0;
    }

    Candidate *cand_list = (Candidate *)malloc(count_non_zero * sizeof(Candidate));
    int idx = 0;
    for (int cl_idx = 0; cl_idx < state->num_clusters; cl_idx++)
    {
        if (counts[cl_idx] > 0)
        {
            cand_list[idx].id = cl_idx;
            cand_list[idx].p = (double)counts[cl_idx];
            idx++;
        }
    }

    qsort(cand_list, count_non_zero, sizeof(Candidate), compare_candidates);

    int n_out = (count_non_zero < max_candidates) ? count_non_zero : max_candidates;
    for (int cand_idx = 0; cand_idx < n_out; cand_idx++)
    {
        candidates[cand_idx] = cand_list[cand_idx].id;
    }

    free(cand_list);
    free(counts);
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

    for (int p = 0; p < temp_count - 2; p++)
    {
        for (int q = p + 1; q < temp_count - 1; q++)
        {
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
                if (!state->scratch.dcc_measured[c1 * config->algo.maxnbclust + c2] ||
                    !state->scratch.dcc_measured[c1 * config->algo.maxnbclust + c3] ||
                    !state->scratch.dcc_measured[c2 * config->algo.maxnbclust + c3])
                {
                    continue;
                }
                d_c1_c2 = state->scratch.dcc_min[c1 * config->algo.maxnbclust + c2];
                d_c1_c3 = state->scratch.dcc_min[c1 * config->algo.maxnbclust + c3];
                d_c2_c3 = state->scratch.dcc_min[c2 * config->algo.maxnbclust + c3];
            }
            else
            {
                d_c1_c2 = state->scratch.dcc_min[c1 * config->algo.maxnbclust + c2];
                if (d_c1_c2 < 0.0)
                {
                    d_c1_c2 = get_dist(&state->clusters[c1].anchor, &state->clusters[c2].anchor, -1,
                                       -1.0, -1.0, config, state);
                    state->scratch.dcc_min[c1 * config->algo.maxnbclust + c2] = d_c1_c2;
                    state->scratch.dcc_min[c2 * config->algo.maxnbclust + c1] = d_c1_c2;
                    state->scratch.dcc_max[c1 * config->algo.maxnbclust + c2] = d_c1_c2;
                    state->scratch.dcc_max[c2 * config->algo.maxnbclust + c1] = d_c1_c2;
                    state->scratch.dcc_measured[c1 * config->algo.maxnbclust + c2] = 1;
                    state->scratch.dcc_measured[c2 * config->algo.maxnbclust + c1] = 1;
                }

                d_c1_c3 = state->scratch.dcc_min[c1 * config->algo.maxnbclust + c3];
                if (d_c1_c3 < 0.0)
                {
                    d_c1_c3 = get_dist(&state->clusters[c1].anchor, &state->clusters[c3].anchor, -1,
                                       -1.0, -1.0, config, state);
                    state->scratch.dcc_min[c1 * config->algo.maxnbclust + c3] = d_c1_c3;
                    state->scratch.dcc_min[c3 * config->algo.maxnbclust + c1] = d_c1_c3;
                    state->scratch.dcc_max[c1 * config->algo.maxnbclust + c3] = d_c1_c3;
                    state->scratch.dcc_max[c3 * config->algo.maxnbclust + c1] = d_c1_c3;
                    state->scratch.dcc_measured[c1 * config->algo.maxnbclust + c3] = 1;
                    state->scratch.dcc_measured[c3 * config->algo.maxnbclust + c1] = 1;
                }

                d_c2_c3 = state->scratch.dcc_min[c2 * config->algo.maxnbclust + c3];
                if (d_c2_c3 < 0.0)
                {
                    d_c2_c3 = get_dist(&state->clusters[c2].anchor, &state->clusters[c3].anchor, -1,
                                       -1.0, -1.0, config, state);
                    state->scratch.dcc_min[c2 * config->algo.maxnbclust + c3] = d_c2_c3;
                    state->scratch.dcc_min[c3 * config->algo.maxnbclust + c2] = d_c2_c3;
                    state->scratch.dcc_max[c2 * config->algo.maxnbclust + c3] = d_c2_c3;
                    state->scratch.dcc_max[c3 * config->algo.maxnbclust + c2] = d_c2_c3;
                    state->scratch.dcc_measured[c2 * config->algo.maxnbclust + c3] = 1;
                    state->scratch.dcc_measured[c3 * config->algo.maxnbclust + c2] = 1;
                }
            }

            long local_pruned_te5 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned_te5) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
            for (int cl_idx = 0; cl_idx < state->num_clusters; cl_idx++)
            {
                if (!state->scratch.clmembflag[cl_idx])
                    continue;
                if (cl_idx == c1 || cl_idx == c2 || cl_idx == c3)
                    continue;

                double d_k_c1 = 0.0;
                double d_k_c2 = 0.0;
                double d_k_c3 = 0.0;

                if (config->optim.sparse_dcc_mode)
                {
                    if (!state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c1] ||
                        !state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c2] ||
                        !state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c3])
                    {
                        continue;
                    }
                    d_k_c1 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c1];
                    d_k_c2 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c2];
                    d_k_c3 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c3];
                }
                else
                {
                    d_k_c1 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c1];
                    if (d_k_c1 < 0.0)
                    {
#ifdef _OPENMP
#pragma omp critical(dcc_cache)
#endif
                        {
                            d_k_c1 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c1];
                            if (d_k_c1 < 0.0)
                            {
                                d_k_c1 = get_dist(
                                    &state->clusters[cl_idx].anchor,
                                    &state->clusters[c1].anchor, -1, -1.0, -1.0,
                                    config, state);
                                state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c1] = d_k_c1;
                                state->scratch.dcc_min[c1 * config->algo.maxnbclust + cl_idx] = d_k_c1;
                                state->scratch.dcc_max[cl_idx * config->algo.maxnbclust + c1] = d_k_c1;
                                state->scratch.dcc_max[c1 * config->algo.maxnbclust + cl_idx] = d_k_c1;
                                state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c1] = 1;
                                state->scratch.dcc_measured[c1 * config->algo.maxnbclust + cl_idx] = 1;
                            }
                        }
                    }

                    d_k_c2 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c2];
                    if (d_k_c2 < 0.0)
                    {
#ifdef _OPENMP
#pragma omp critical(dcc_cache)
#endif
                        {
                            d_k_c2 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c2];
                            if (d_k_c2 < 0.0)
                            {
                                d_k_c2 = get_dist(
                                    &state->clusters[cl_idx].anchor,
                                    &state->clusters[c2].anchor, -1, -1.0, -1.0,
                                    config, state);
                                state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c2] = d_k_c2;
                                state->scratch.dcc_min[c2 * config->algo.maxnbclust + cl_idx] = d_k_c2;
                                state->scratch.dcc_max[cl_idx * config->algo.maxnbclust + c2] = d_k_c2;
                                state->scratch.dcc_max[c2 * config->algo.maxnbclust + cl_idx] = d_k_c2;
                                state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c2] = 1;
                                state->scratch.dcc_measured[c2 * config->algo.maxnbclust + cl_idx] = 1;
                            }
                        }
                    }

                    d_k_c3 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c3];
                    if (d_k_c3 < 0.0)
                    {
#ifdef _OPENMP
#pragma omp critical(dcc_cache)
#endif
                        {
                            d_k_c3 = state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c3];
                            if (d_k_c3 < 0.0)
                            {
                                d_k_c3 = get_dist(
                                    &state->clusters[cl_idx].anchor,
                                    &state->clusters[c3].anchor, -1, -1.0, -1.0,
                                    config, state);
                                state->scratch.dcc_min[cl_idx * config->algo.maxnbclust + c3] = d_k_c3;
                                state->scratch.dcc_min[c3 * config->algo.maxnbclust + cl_idx] = d_k_c3;
                                state->scratch.dcc_max[cl_idx * config->algo.maxnbclust + c3] = d_k_c3;
                                state->scratch.dcc_max[c3 * config->algo.maxnbclust + cl_idx] = d_k_c3;
                                state->scratch.dcc_measured[cl_idx * config->algo.maxnbclust + c3] = 1;
                                state->scratch.dcc_measured[c3 * config->algo.maxnbclust + cl_idx] = 1;
                            }
                        }
                    }
                }

                double min_d = calc_min_dist_5pt(d_f_c1, d_f_c2, d_f_c3, d_k_c1, d_k_c2, d_k_c3,
                                                 d_c1_c2, d_c1_c3, d_c2_c3);

                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[cl_idx] = 0;
                    local_pruned_te5++;
                }
            }
            state->telemetry.clusters_pruned += local_pruned_te5;
        }
    }
}

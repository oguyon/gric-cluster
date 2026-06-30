#include "cluster_prune.h"
#include "cluster_core.h"
#include "cluster_math.h"
#include <stdlib.h>
#include <string.h>

int get_prediction_candidates(ClusterState *state, ClusterConfig *config, int *candidates, int max_candidates) {
    long total = state->total_frames_processed;
    int len = config->pred_len;
    int h = config->pred_h;

    if (total < len) return 0;

    long search_limit = total - len;
    long search_start = (total > h) ? total - h : 0;
    if (search_start > search_limit) search_start = search_limit;

    int *pattern = &state->assignments[total - len];

    int *counts = (int *)calloc(state->num_clusters, sizeof(int));
    if (!counts) return 0;

    for (long i = search_start; i < search_limit; i++) {
        if (state->assignments[i] == pattern[0]) {
            if (memcmp(&state->assignments[i], pattern, len * sizeof(int)) == 0) {
                int next_cluster = state->assignments[i + len];
                if (next_cluster >= 0 && next_cluster < state->num_clusters) {
                    counts[next_cluster]++;
                }
            }
        }
    }

    int count_non_zero = 0;
    for(int i=0; i<state->num_clusters; i++) if (counts[i] > 0) count_non_zero++;

    if (count_non_zero == 0) {
        free(counts);
        return 0;
    }

    Candidate *cand_list = (Candidate *)malloc(count_non_zero * sizeof(Candidate));
    int idx = 0;
    for(int i=0; i<state->num_clusters; i++) {
        if (counts[i] > 0) {
            cand_list[idx].id = i;
            cand_list[idx].p = (double)counts[i];
            idx++;
        }
    }

    qsort(cand_list, count_non_zero, sizeof(Candidate), compare_candidates);

    int n_out = (count_non_zero < max_candidates) ? count_non_zero : max_candidates;
    for(int i=0; i<n_out; i++) {
        candidates[i] = cand_list[i].id;
    }

    free(cand_list);
    free(counts);
    return n_out;
}

void prune_candidates_te5(ClusterConfig *config, ClusterState *state, int *temp_indices, double *temp_dists, int temp_count) {
    if (!config->te5_mode || temp_count < 3) return;

    int c3 = temp_indices[temp_count - 1]; // Current cluster (newest anchor)
    double d_f_c3 = temp_dists[temp_count - 1];

    for (int p = 0; p < temp_count - 2; p++) {
        for (int q = p + 1; q < temp_count - 1; q++) {
            int c1 = temp_indices[p];
            double d_f_c1 = temp_dists[p];
            int c2 = temp_indices[q];
            double d_f_c2 = temp_dists[q];

            // Get inter-cluster distances (lazy load)
            double d_c1_c2 = state->dccarray[c1 * config->maxnbclust + c2];
            if (d_c1_c2 < 0) {
                 d_c1_c2 = get_dist(&state->clusters[c1].anchor, &state->clusters[c2].anchor, -1, -1.0, -1.0, config, state);
                 state->dccarray[c1 * config->maxnbclust + c2] = d_c1_c2;
                 state->dccarray[c2 * config->maxnbclust + c1] = d_c1_c2;
            }

            double d_c1_c3 = state->dccarray[c1 * config->maxnbclust + c3];
            if (d_c1_c3 < 0) {
                    d_c1_c3 = get_dist(&state->clusters[c1].anchor, &state->clusters[c3].anchor, -1, -1.0, -1.0, config, state);
                    state->dccarray[c1 * config->maxnbclust + c3] = d_c1_c3;
                    state->dccarray[c3 * config->maxnbclust + c1] = d_c1_c3;
            }

            double d_c2_c3 = state->dccarray[c2 * config->maxnbclust + c3];
            if (d_c2_c3 < 0) {
                    d_c2_c3 = get_dist(&state->clusters[c2].anchor, &state->clusters[c3].anchor, -1, -1.0, -1.0, config, state);
                    state->dccarray[c2 * config->maxnbclust + c3] = d_c2_c3;
                    state->dccarray[c3 * config->maxnbclust + c2] = d_c2_c3;
            }

            long local_pruned_te5 = 0;
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:local_pruned_te5)
            #endif
            for (int k = 0; k < state->num_clusters; k++) {
                if (!state->clmembflag[k]) continue;
                if (k == c1 || k == c2 || k == c3) continue;

                // Lazy load k distances (thread-safe cache fill)
                double d_k_c1 = state->dccarray[k * config->maxnbclust + c1];
                if (d_k_c1 < 0) {
                    #ifdef _OPENMP
                    #pragma omp critical(dcc_cache)
                    #endif
                    {
                        d_k_c1 = state->dccarray[k * config->maxnbclust + c1];
                        if (d_k_c1 < 0) {
                            d_k_c1 = get_dist(&state->clusters[k].anchor, &state->clusters[c1].anchor, -1, -1.0, -1.0, config, state);
                            state->dccarray[k * config->maxnbclust + c1] = d_k_c1;
                            state->dccarray[c1 * config->maxnbclust + k] = d_k_c1;
                        }
                    }
                }

                double d_k_c2 = state->dccarray[k * config->maxnbclust + c2];
                if (d_k_c2 < 0) {
                    #ifdef _OPENMP
                    #pragma omp critical(dcc_cache)
                    #endif
                    {
                        d_k_c2 = state->dccarray[k * config->maxnbclust + c2];
                        if (d_k_c2 < 0) {
                            d_k_c2 = get_dist(&state->clusters[k].anchor, &state->clusters[c2].anchor, -1, -1.0, -1.0, config, state);
                            state->dccarray[k * config->maxnbclust + c2] = d_k_c2;
                            state->dccarray[c2 * config->maxnbclust + k] = d_k_c2;
                        }
                    }
                }

                double d_k_c3 = state->dccarray[k * config->maxnbclust + c3];
                if (d_k_c3 < 0) {
                    #ifdef _OPENMP
                    #pragma omp critical(dcc_cache)
                    #endif
                    {
                        d_k_c3 = state->dccarray[k * config->maxnbclust + c3];
                        if (d_k_c3 < 0) {
                            d_k_c3 = get_dist(&state->clusters[k].anchor, &state->clusters[c3].anchor, -1, -1.0, -1.0, config, state);
                            state->dccarray[k * config->maxnbclust + c3] = d_k_c3;
                            state->dccarray[c3 * config->maxnbclust + k] = d_k_c3;
                        }
                    }
                }

                double min_d = calc_min_dist_5pt(d_f_c1, d_f_c2, d_f_c3,
                                                 d_k_c1, d_k_c2, d_k_c3,
                                                 d_c1_c2, d_c1_c3, d_c2_c3);

                if (min_d > config->rlim) {
                    state->clmembflag[k] = 0;
                    local_pruned_te5++;
                }
            }
            state->clusters_pruned += local_pruned_te5;
        }
    }
}

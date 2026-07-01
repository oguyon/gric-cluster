/**
 * @file cluster_step_helpers.c
 * @brief Implementations of helper procedures for processing a single image frame.
 *
 * Implements specific calculations (priors, sorting, geometric updates, merges)
 * used during single frame ingestion steps.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_step_helpers.h"
#include "cluster_core.h"
#include "cluster_math.h"
#include "cluster_mgmt.h"
#include "cluster_prune.h"
#include "frameread.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANSI_COLOR_RED    "\x1b[31m"
#define ANSI_COLOR_GREEN  "\x1b[32m"
#define ANSI_COLOR_BLUE   "\x1b[34m"
#define ANSI_COLOR_RESET  "\x1b[0m"
#define ANSI_BG_GREEN     "\x1b[42m"
#define ANSI_COLOR_BLACK  "\x1b[30m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"

#define OMP_MIN_CLUSTERS 256

static void evaluate_and_pruning_candidate(
    int             cj,
    Frame          *current_frame,
    ClusterConfig  *config,
    ClusterState   *state,
    int            *temp_indices,
    double         *temp_dists,
    int            *temp_count,
    int            *assigned_cluster,
    int            *found,
    double         *out_dfc,
    int             is_prediction)
{
    if (*temp_count < state->telemetry.max_steps_recorded && state->num_clusters > 0)
    {
        int pruned_cnt = 0;
        for (int pc = 0; pc < state->num_clusters; pc++)
        {
            if (state->scratch.clmembflag[pc] == 0)
            {
                pruned_cnt++;
            }
        }
        state->telemetry.pruned_fraction_sum[*temp_count] +=
            (double)pruned_cnt / state->num_clusters;
        state->telemetry.step_counts[*temp_count]++;
    }

    double dfc = get_dist(current_frame, &state->clusters[cj].anchor,
                          state->clusters[cj].id, state->clusters[cj].prob,
                          state->scratch.current_gprobs[cj], config, state);

    if (out_dfc)
    {
        *out_dfc = dfc;
    }

    if (*temp_count < config->algo.maxnbclust)
    {
        temp_indices[*temp_count] = cj;
        temp_dists[*temp_count] = dfc;
        (*temp_count)++;
    }

    add_visitor(&state->cluster_visitors[cj], state->telemetry.total_frames_processed);

    if (dfc < config->algo.rlim)
    {
        *assigned_cluster = cj;
        state->clusters[cj].prob += config->algo.deltaprob;
        *found = 1;
        if (config->output.verbose_level >= 2)
        {
            printf(ANSI_COLOR_GREEN "  [VV] Frame %ld assigned to Cluster %d%s\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, *assigned_cluster,
                   is_prediction ? " (Prediction)" : "");
        }
        return;
    }

    long local_pruned = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
    for (int cl = 0; cl < state->num_clusters; cl++)
    {
        if (state->scratch.clmembflag[cl] == 0)
        {
            continue;
        }

        double dcc = state->scratch.dccarray[cj * config->algo.maxnbclust + cl];
        if (dcc < 0)
        {
            dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[cl].anchor, -1,
                           -1.0, -1.0, config, state);
            state->scratch.dccarray[cj * config->algo.maxnbclust + cl] = dcc;
            state->scratch.dccarray[cl * config->algo.maxnbclust + cj] = dcc;
        }

        if (dcc - dfc > config->algo.rlim)
        {
            state->scratch.clmembflag[cl] = 0;
            local_pruned++;
        }
        else if (dfc - dcc > config->algo.rlim)
        {
            state->scratch.clmembflag[cl] = 0;
            local_pruned++;
        }
    }
    state->telemetry.clusters_pruned += local_pruned;

    if (config->optim.te4_mode && *temp_count > 1)
    {
        for (int p = 0; p < *temp_count - 1; p++)
        {
            int    cprev = temp_indices[p];
            double d_m_cprev = temp_dists[p];
            double d_ci_cprev = state->scratch.dccarray[cj * config->algo.maxnbclust + cprev];

            if (d_ci_cprev < 0)
            {
                d_ci_cprev = get_dist(&state->clusters[cj].anchor, &state->clusters[cprev].anchor,
                                      -1, -1.0, -1.0, config, state);
                state->scratch.dccarray[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                state->scratch.dccarray[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
            }

            long local_pruned_te4 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned_te4) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
            for (int k = 0; k < state->num_clusters; k++)
            {
                if (!state->scratch.clmembflag[k])
                {
                    continue;
                }
                if (k == cj || k == cprev)
                {
                    continue;
                }

                double d_ci_ck = state->scratch.dccarray[cj * config->algo.maxnbclust + k];
                if (d_ci_ck < 0)
                {
                    d_ci_ck = get_dist(&state->clusters[cj].anchor, &state->clusters[k].anchor,
                                       -1, -1.0, -1.0, config, state);
                    state->scratch.dccarray[cj * config->algo.maxnbclust + k] = d_ci_ck;
                    state->scratch.dccarray[k * config->algo.maxnbclust + cj] = d_ci_ck;
                }

                double d_cprev_ck = state->scratch.dccarray[cprev * config->algo.maxnbclust + k];
                if (d_cprev_ck < 0)
                {
                    d_cprev_ck = get_dist(
                        &state->clusters[cprev].anchor, &state->clusters[k].anchor,
                        -1, -1.0, -1.0, config, state);
                    state->scratch.dccarray[cprev * config->algo.maxnbclust + k] = d_cprev_ck;
                    state->scratch.dccarray[k * config->algo.maxnbclust + cprev] = d_cprev_ck;
                }

                double min_d = calc_min_dist_4pt(dfc, d_m_cprev, d_ci_cprev, d_ci_ck, d_cprev_ck);
                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[k] = 0;
                    local_pruned_te4++;
                }
            }
            state->telemetry.clusters_pruned += local_pruned_te4;
        }
    }

    if (config->optim.te5_mode)
    {
        prune_candidates_te5(config, state, temp_indices, temp_dists, *temp_count);
    }

    state->scratch.clmembflag[cj] = 0;
}

void initialize_initial_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *assigned_cluster)
{
    state->clusters[0].anchor = *current_frame;
    state->clusters[0].id = 0;
    state->clusters[0].prob = 1.0;
    state->num_clusters = 1;
    state->scratch.dccarray[0] = 0.0;

    add_visitor(&state->cluster_visitors[0], state->telemetry.total_frames_processed);
    *assigned_cluster = 0;

    if (config->output.verbose_level >= 2)
    {
        printf(ANSI_COLOR_ORANGE
               "  [VV] Frame %5ld created initial Cluster    0\n" ANSI_COLOR_RESET,
               state->telemetry.total_frames_processed);
    }
}

void compute_priors_and_mixing(
    ClusterConfig *config,
    ClusterState  *state,
    int            prev_assigned_cluster,
    Candidate     *sorting_candidates)
{
    double sum_prob = 0.0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        sum_prob += state->clusters[i].prob;
    }
    if (sum_prob > 0)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].prob /= sum_prob;
        }
    }

    for (int i = 0; i < state->num_clusters; i++)
    {
        state->scratch.current_gprobs[i] = 1.0;
    }
    for (int i = 0; i < state->num_clusters; i++)
    {
        state->scratch.clmembflag[i] = 1;
    }

    double trans_prob_sum = 0.0;
    if (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            trans_prob_sum += (double)state->transition_matrix[
                prev_assigned_cluster * config->algo.maxnbclust + i];
        }
    }

    for (int i = 0; i < state->num_clusters; i++)
    {
        double prior = state->clusters[i].prob;
        double tp = 0.0;
        if (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1 &&
            trans_prob_sum > 0.0)
        {
            tp = (double)state->transition_matrix[
                prev_assigned_cluster * config->algo.maxnbclust + i] / trans_prob_sum;
            state->scratch.mixed_probs[i] =
                (1.0 - config->algo.tm_mixing_coeff) * prior +
                config->algo.tm_mixing_coeff * tp;
        }
        else
        {
            state->scratch.mixed_probs[i] = prior;
        }
    }

    if (!config->optim.gprob_mode)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            sorting_candidates[i].id = i;
            sorting_candidates[i].p = state->scratch.mixed_probs[i];
        }
        qsort(sorting_candidates, state->num_clusters, sizeof(Candidate),
              compare_candidates);
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->scratch.probsortedclindex[i] = sorting_candidates[i].id;
        }
    }
}

int evaluate_prediction_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *temp_indices,
    double         *temp_dists,
    int           *temp_count,
    int           *assigned_cluster,
    int           *found)
{
    if (config->optim.pred_mode &&
        state->telemetry.total_frames_processed >= config->optim.pred_len)
    {
        int *pred_candidates = (int *)malloc(config->optim.pred_n * sizeof(int));
        if (pred_candidates)
        {
            int num_preds =
                get_prediction_candidates(state, config, pred_candidates,
                                           config->optim.pred_n);

            for (int p = 0; p < num_preds; p++)
            {
                int cj = pred_candidates[p];
                if (cj >= state->num_clusters || !state->scratch.clmembflag[cj])
                {
                    continue;
                }

                evaluate_and_pruning_candidate(cj, current_frame, config, state,
                                               temp_indices, temp_dists, temp_count,
                                               assigned_cluster, found, NULL, 1);
                if (*found)
                {
                    break;
                }
            }
            free(pred_candidates);
        }
    }
    return *found;
}

int evaluate_standard_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *temp_indices,
    double         *temp_dists,
    int           *temp_count,
    int           *assigned_cluster,
    int           *found,
    double        *out_dfc,
    int           *k_search,
    Candidate     *verbose_candidates)
{
    while (!(*found))
    {
        if (config->output.verbose_level >= 2 && verbose_candidates)
        {
            int vcount = 0;
            for (int i = 0; i < state->num_clusters; i++)
            {
                if (state->scratch.clmembflag[i])
                {
                    double p = state->scratch.mixed_probs[i];
                    if (config->optim.gprob_mode)
                    {
                        p *= state->scratch.current_gprobs[i];
                    }
                    verbose_candidates[vcount].id = i;
                    verbose_candidates[vcount].p = p;
                    vcount++;
                }
            }

            if (vcount > 0)
            {
                qsort(verbose_candidates, vcount, sizeof(Candidate), compare_candidates);
                printf("  [VV] Cluster ranking:");
                for (int i = 0; i < vcount; i++)
                {
                    printf(" [%4d %12.5e]", verbose_candidates[i].id,
                           verbose_candidates[i].p);
                    if (i < vcount - 1)
                    {
                        printf(" >");
                    }
                }
                printf("\n");
            }
        }

        int cj = -1;

        if (!config->optim.gprob_mode)
        {
            while (*k_search < state->num_clusters &&
                   state->scratch.clmembflag[state->scratch.probsortedclindex[*k_search]] == 0)
            {
                (*k_search)++;
            }
            if (*k_search >= state->num_clusters)
            {
                break;
            }
            cj = state->scratch.probsortedclindex[*k_search];
            (*k_search)++;
        }
        else
        {
            double max_p = -1.0;
            cj = -1;
            for (int i = 0; i < state->num_clusters; i++)
            {
                if (state->scratch.clmembflag[i])
                {
                    double p = state->scratch.mixed_probs[i] *
                               state->scratch.current_gprobs[i];
                    if (p > max_p)
                    {
                        max_p = p;
                        cj = i;
                    }
                }
            }
            if (cj == -1)
            {
                break;
            }
        }

        evaluate_and_pruning_candidate(cj, current_frame, config, state,
                                       temp_indices, temp_dists, temp_count,
                                       assigned_cluster, found, out_dfc, 0);
        if (*found)
        {
            break;
        }

        int active_cluster_count = 0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i])
            {
                active_cluster_count++;
            }
        }

        if ((config->optim.gprob_mode ||
             (config->output.distall_mode && state->distall_out) ||
             config->output.verbose_level >= 2) &&
            active_cluster_count > 1)
        {
            update_geometric_probabilities(config, state, cj, *out_dfc);
        }
    }
    return *found;
}

void update_geometric_probabilities(
    ClusterConfig *config,
    ClusterState  *state,
    int            cj,
    double         dfc)
{
    int match_count = state->cluster_visitors[cj].count;
    if (match_count > 0)
    {
        match_count--;
    }

    if (config->output.verbose_level >= 2)
    {
        printf("  [VV] Distance > rlim. Found %d matches in distinfo for Cluster "
               "%4d (Frame %5d).\n",
               match_count, cj, state->clusters[cj].anchor.id);
    }

    int start_idx = 0;
    if (state->cluster_visitors[cj].count > config->optim.max_gprob_visitors)
    {
        start_idx = state->cluster_visitors[cj].count -
                    config->optim.max_gprob_visitors;
    }
    for (int i = start_idx; i < state->cluster_visitors[cj].count; i++)
    {
        int k_idx = state->cluster_visitors[cj].frames[i];
        if (k_idx == state->telemetry.total_frames_processed)
        {
            continue;
        }

        int target_cl = state->frame_infos[k_idx].assignment;
        if (target_cl < 0 || target_cl >= state->num_clusters)
        {
            continue;
        }
        int is_active = state->scratch.clmembflag[target_cl];

        if (config->output.verbose_level >= 2)
        {
            if (is_active)
            {
                printf(ANSI_BG_GREEN ANSI_COLOR_BLACK
                       "  [VV]   Frame %5d also had distance measurement to "
                       "Cluster %4d (Anchor Frame %5d). Frame %5d cluster "
                       "membership is %4d. " ANSI_COLOR_RESET "\n",
                       k_idx, cj, state->clusters[cj].anchor.id, k_idx, target_cl);
            }
            else
            {
                printf("  [VV]   Frame %5d also had distance measurement to "
                       "Cluster %4d (Anchor Frame %5d). Frame %5d cluster "
                       "membership is %4d.\n",
                       k_idx, cj, state->clusters[cj].anchor.id, k_idx, target_cl);
            }
        }

        if (!is_active)
        {
            continue;
        }

        double dist_k = -1.0;
        for (int d_idx = 0; d_idx < state->frame_infos[k_idx].num_dists; d_idx++)
        {
            if (state->frame_infos[k_idx].cluster_indices[d_idx] == cj)
            {
                dist_k = state->frame_infos[k_idx].distances[d_idx];
                break;
            }
        }

        if (dist_k >= 0)
        {
            double dr = fabs(dfc - dist_k) / config->algo.rlim;
            double val = fmatch(dr, config->optim.fmatch_a, config->optim.fmatch_b);

            if (config->output.verbose_level >= 2)
            {
                printf("    dist %5ld-%-5d = %12.5e  dist %5d-%-5d = %12.5e, "
                       "fmatch=%12.5e, updating GProb(Cluster %4d) from %12.5e to "
                       "%12.5e\n",
                       state->telemetry.total_frames_processed,
                       state->clusters[cj].anchor.id,
                       dfc, k_idx, state->clusters[cj].anchor.id, dist_k, val,
                       target_cl, state->scratch.current_gprobs[target_cl],
                       state->scratch.current_gprobs[target_cl] * val);
            }

            state->scratch.current_gprobs[target_cl] *= val;
        }
    }
}

int handle_new_cluster_creation(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *prev_assigned_cluster,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count)
{
    if (state->num_clusters < config->algo.maxnbclust)
    {
        int assigned_cluster = state->num_clusters;
        state->clusters[state->num_clusters].anchor = *current_frame;
        state->clusters[state->num_clusters].id = state->num_clusters;
        state->clusters[state->num_clusters].prob = 1.0;

        for (int i = 0; i < state->num_clusters; i++)
        {
            double d =
                get_dist(&state->clusters[state->num_clusters].anchor,
                         &state->clusters[i].anchor, -1, -1.0, -1.0, config, state);
            state->scratch.dccarray[
                state->num_clusters * config->algo.maxnbclust + i] = d;
            state->scratch.dccarray[
                i * config->algo.maxnbclust + state->num_clusters] = d;
        }
        state->scratch.dccarray[
            state->num_clusters * config->algo.maxnbclust + state->num_clusters] = 0.0;

        if (config->output.verbose_level >= 2)
        {
            printf(ANSI_COLOR_GREEN
                   "  [VV] Frame %5ld assigned to Cluster %4d\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, assigned_cluster);
            printf(ANSI_COLOR_ORANGE
                   "  [VV] Frame %5ld created new Cluster %4d\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, state->num_clusters);
        }

        add_visitor(&state->cluster_visitors[state->num_clusters],
                    state->telemetry.total_frames_processed);

        if (*temp_count < config->algo.maxnbclust)
        {
            temp_indices[*temp_count] = state->num_clusters;
            temp_dists[*temp_count] = 0.0;
            (*temp_count)++;
        }

        state->num_clusters++;
        free(current_frame);
        return assigned_cluster;
    }

    if (config->algo.maxcl_strategy == MAXCL_STOP)
    {
        printf(ANSI_COLOR_ORANGE "Max clusters limit reached.\n" ANSI_COLOR_RESET);
        printf("Frames clustered: %ld\n", state->telemetry.total_frames_processed);
        free_frame(current_frame);
        return -2;
    }
    else if (config->algo.maxcl_strategy == MAXCL_DISCARD)
    {
        int scan_limit = (int)(state->num_clusters * config->algo.discard_fraction);
        if (scan_limit < 1)
        {
            scan_limit = state->num_clusters;
        }

        int min_idx = -1;
        int min_count = -1;

        for (int i = 0; i < scan_limit; i++)
        {
            int count = state->cluster_visitors[i].count;
            if (min_idx == -1 || count < min_count)
            {
                min_count = count;
                min_idx = i;
            }
        }

        if (min_idx != -1)
        {
            remove_cluster(state, config, min_idx, -1);
            if (*prev_assigned_cluster == min_idx)
            {
                *prev_assigned_cluster = -1;
            }
            else if (*prev_assigned_cluster > min_idx)
            {
                (*prev_assigned_cluster)--;
            }
            int assigned_cluster = state->num_clusters;
            state->clusters[state->num_clusters].anchor = *current_frame;
            state->clusters[state->num_clusters].id = state->num_clusters;
            state->clusters[state->num_clusters].prob = 1.0;

            for (int i = 0; i < state->num_clusters; i++)
            {
                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                    config, state);
                state->scratch.dccarray[
                    state->num_clusters * config->algo.maxnbclust + i] = d;
                state->scratch.dccarray[
                    i * config->algo.maxnbclust + state->num_clusters] = d;
            }
            state->scratch.dccarray[state->num_clusters * config->algo.maxnbclust +
                            state->num_clusters] = 0.0;

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }

            state->num_clusters++;
            free(current_frame);
            return assigned_cluster;
        }

        free_frame(current_frame);
        return -2;
    }
    else if (config->algo.maxcl_strategy == MAXCL_MERGE)
    {
        int    best_i = -1, best_j = -1;
        double min_d = -1.0;

        for (int i = 0; i < state->num_clusters; i++)
        {
            for (int j = i + 1; j < state->num_clusters; j++)
            {
                double d = state->scratch.dccarray[i * config->algo.maxnbclust + j];
                if (d >= 0 && (min_d < 0 || d < min_d))
                {
                    min_d = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i != -1)
        {
            int count_i = state->cluster_visitors[best_i].count;
            int count_j = state->cluster_visitors[best_j].count;
            int target = (count_i >= count_j) ? best_i : best_j;
            int remove = (count_i >= count_j) ? best_j : best_i;

            if (config->output.verbose_level >= 1)
            {
                printf("Merging cluster %d into %d (dist %.4f)\n", remove, target,
                       min_d);
            }

            remove_cluster(state, config, remove, target);
            if (*prev_assigned_cluster == remove)
            {
                if (target > remove)
                {
                    *prev_assigned_cluster = target - 1;
                }
                else
                {
                    *prev_assigned_cluster = target;
                }
            }
            else if (*prev_assigned_cluster > remove)
            {
                (*prev_assigned_cluster)--;
            }

            int assigned_cluster = state->num_clusters;
            state->clusters[state->num_clusters].anchor = *current_frame;
            state->clusters[state->num_clusters].id = state->num_clusters;
            state->clusters[state->num_clusters].prob = 1.0;

            for (int i = 0; i < state->num_clusters; i++)
            {
                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                    config, state);
                state->scratch.dccarray[
                    state->num_clusters * config->algo.maxnbclust + i] = d;
                state->scratch.dccarray[
                    i * config->algo.maxnbclust + state->num_clusters] = d;
            }
            state->scratch.dccarray[state->num_clusters * config->algo.maxnbclust +
                            state->num_clusters] = 0.0;

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }

            state->num_clusters++;
            free(current_frame);
            return assigned_cluster;
        }

        free_frame(current_frame);
        return -2;
    }

    free_frame(current_frame);
    return -2;
}

void record_step_assignment(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int            assigned_cluster,
    int           *prev_assigned_cluster,
    FILE          *ascii_out,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count,
    long           start_pruned_val)
{
    if (state->telemetry.total_frames_processed > 0 && *prev_assigned_cluster != -1 &&
        assigned_cluster != -1)
    {
        state->transition_matrix[*prev_assigned_cluster * config->algo.maxnbclust +
                                 assigned_cluster]++;
    }
    *prev_assigned_cluster = assigned_cluster;

    state->assignments[state->telemetry.total_frames_processed] = assigned_cluster;
    if (ascii_out)
    {
        if (config->input.stream_input_mode)
        {
            fprintf(ascii_out, "%ld %d %lu %ld.%09ld\n",
                    state->telemetry.total_frames_processed,
                    assigned_cluster, current_frame->cnt0, current_frame->atime.tv_sec,
                    current_frame->atime.tv_nsec);
        }
        else
        {
            fprintf(ascii_out, "%ld %d\n",
                    state->telemetry.total_frames_processed, assigned_cluster);
        }
    }

    state->frame_infos[state->telemetry.total_frames_processed].assignment = assigned_cluster;
    state->frame_infos[state->telemetry.total_frames_processed].num_dists = temp_count;
    if (temp_count > 0)
    {
        state->frame_infos[state->telemetry.total_frames_processed].cluster_indices =
            (int *)malloc(temp_count * sizeof(int));
        state->frame_infos[state->telemetry.total_frames_processed].distances =
            (double *)malloc(temp_count * sizeof(double));
        if (state->frame_infos[state->telemetry.total_frames_processed].cluster_indices &&
            state->frame_infos[state->telemetry.total_frames_processed].distances)
        {
            memcpy(state->frame_infos[state->telemetry.total_frames_processed].cluster_indices,
                   temp_indices, temp_count * sizeof(int));
            memcpy(state->frame_infos[state->telemetry.total_frames_processed].distances,
                   temp_dists, temp_count * sizeof(double));
        }
    }
    else
    {
        state->frame_infos[state->telemetry.total_frames_processed].cluster_indices = NULL;
        state->frame_infos[state->telemetry.total_frames_processed].distances = NULL;
    }

    state->telemetry.total_frames_processed++;

    if (state->telemetry.dist_counts && temp_count <= config->algo.maxnbclust)
    {
        state->telemetry.dist_counts[temp_count]++;
        state->telemetry.pruned_counts_by_dist[temp_count] +=
            (state->telemetry.clusters_pruned - start_pruned_val);
    }
}

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cluster_core.h"
#include "framedistance.h"
#include "frameread.h"

#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_BG_GREEN "\x1b[42m"
#define ANSI_COLOR_BLACK "\x1b[30m"
#define ANSI_COLOR_RESET "\x1b[0m"

int compare_candidates(const void *a, const void *b)
{
    Candidate *ca = (Candidate *)a;
    Candidate *cb = (Candidate *)b;
    if (ca->p < cb->p)
        return 1;
    if (ca->p > cb->p)
        return -1;
    return 0;
}

int compare_doubles(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

double get_dist(Frame *a, Frame *b, int cluster_idx, double cluster_prob, double current_gprob,
                ClusterConfig *config, ClusterState *state)
{
#ifdef _OPENMP
#pragma omp atomic
#endif
    state->framedist_calls++;
    double d = framedist(a, b);
    if (config->distall_mode && state->distall_out)
    {
        double ratio = (config->rlim > 0.0) ? d / config->rlim : -1.0;
        fprintf(state->distall_out, "%-8d %-8d %-12.6f %-12.6f %-8d %-12.6f %-12.6f\n", a->id,
                b->id, d, ratio, cluster_idx, cluster_prob, current_gprob);
    }
    if (config->verbose_level >= 2 && cluster_idx >= 0)
    {
        printf(ANSI_COLOR_BLUE
               "  [VV] Computed distance: Frame %5d to Cluster %4d = %12.5e\n" ANSI_COLOR_RESET,
               a->id, cluster_idx, d);
    }
    return d;
}

void run_scandist(ClusterConfig *config, char *out_dir)
{
    long nframes = get_num_frames();
    if (nframes < 2)
    {
        printf("Not enough frames to calculate distances.\n");
        return;
    }

    long process_limit = (nframes > config->maxnbfr) ? config->maxnbfr : nframes;
    double *distances = (double *)malloc((process_limit - 1) * sizeof(double));
    if (!distances)
    {
        perror("Memory allocation failed");
        return;
    }

    Frame *prev = getframe();
    if (!prev)
    {
        free(distances);
        return;
    }

    FILE *scan_out = NULL;
    char scan_path[1024];
    if (out_dir)
    {
        snprintf(scan_path, sizeof(scan_path), "%s/dist-scan.txt", out_dir);
        scan_out = fopen(scan_path, "w");
        if (scan_out)
        {
            fprintf(scan_out, "# Frame1 Frame2 Distance\n");
        }
    }

    printf("Scanning distances\n");

    long count = 0;
    // Loop from 1 to process_limit-1
    for (long i = 1; i < process_limit; i++)
    {
        Frame *curr = getframe();
        if (!curr)
            break;

        double d = framedist(prev, curr);
        distances[count++] = d;

        if (scan_out)
        {
            fprintf(scan_out, "%d %d %.6f\n", prev->id, curr->id, d);
        }

        if (config->progress_mode && (i % 10 == 0 || i == process_limit - 1))
        {
            printf("\rScanning frame %ld / %ld", i, process_limit);
            fflush(stdout);
        }

        free_frame(prev);
        prev = curr;
    }
    free_frame(prev);
    if (scan_out)
        fclose(scan_out);
    if (config->progress_mode)
        printf("\n");

    if (count > 0)
    {
        qsort(distances, count, sizeof(double), compare_doubles);

        double min_val = distances[0];
        double max_val = distances[count - 1];
        double median_val;
        double p20_val;
        double p80_val;

        if (count % 2 == 1)
        {
            median_val = distances[count / 2];
        }
        else
        {
            median_val = (distances[count / 2 - 1] + distances[count / 2]) / 2.0;
        }

        double p20_idx = (count - 1) * 0.2;
        int p20_i = (int)p20_idx;
        double p20_f = p20_idx - p20_i;
        if (p20_i + 1 < count)
            p20_val = distances[p20_i] * (1.0 - p20_f) + distances[p20_i + 1] * p20_f;
        else
            p20_val = distances[p20_i];

        double p80_idx = (count - 1) * 0.8;
        int p80_i = (int)p80_idx;
        double p80_f = p80_idx - p80_i;
        if (p80_i + 1 < count)
            p80_val = distances[p80_i] * (1.0 - p80_f) + distances[p80_i + 1] * p80_f;
        else
            p80_val = distances[p80_i];

        if (config->scandist_mode)
        {
            printf("Distance statistics (%ld intervals):\n", count);
            printf("%-10s %.6f\n", "Min:", min_val);
            printf("%-10s %.6f\n", "20%:", p20_val);
            printf("%-10s %.6f\n", "Median:", median_val);
            printf("%-10s %.6f\n", "80%:", p80_val);
            printf("%-10s %.6f\n", "Max:", max_val);
        }
        else if (config->auto_rlim_mode)
        {
            config->rlim = config->auto_rlim_factor * median_val;
            printf("Auto-rlim: Median distance = %.6f, Multiplier = %.6f -> rlim = %.6f\n",
                   median_val, config->auto_rlim_factor, config->rlim);
        }
    }
    else
    {
        printf("No distances calculated.\n");
    }

    free(distances);
}

void run_clustering(ClusterConfig *config, ClusterState *state)
{
#ifdef _OPENMP
    if (config->ncpu > 1)
    {
        omp_set_num_threads(config->ncpu);
    }
#endif

    long actual_frames = get_num_frames();
    if (actual_frames > config->maxnbfr)
        actual_frames = config->maxnbfr;

    // Allocate assignments array
    state->assignments = (int *)malloc(actual_frames * sizeof(int));
    state->frame_infos = (FrameInfo *)calloc(actual_frames, sizeof(FrameInfo));

    state->max_steps_recorded = config->maxnbclust;
    state->pruned_fraction_sum = (double *)calloc(state->max_steps_recorded, sizeof(double));
    state->step_counts = (long *)calloc(state->max_steps_recorded, sizeof(long));
    state->transition_matrix =
        (long *)calloc(config->maxnbclust * config->maxnbclust, sizeof(long));
    state->mixed_probs = (double *)calloc(config->maxnbclust, sizeof(double));

    state->dist_counts = (long *)calloc(config->maxnbclust + 1, sizeof(long));
    state->pruned_counts_by_dist = (long *)calloc(config->maxnbclust + 1, sizeof(long));

    int *temp_indices = (int *)malloc(config->maxnbclust * sizeof(int));
    double *temp_dists = (double *)malloc(config->maxnbclust * sizeof(double));
    if (!temp_indices || !temp_dists)
    {
        perror("Memory allocation failed for temp buffers");
        return;
    }

    Candidate *verbose_candidates = NULL;
    if (config->verbose_level >= 2)
    {
        verbose_candidates = (Candidate *)malloc(config->maxnbclust * sizeof(Candidate));
    }

    // For sorting candidates when transition matrix is used
    Candidate *sorting_candidates = (Candidate *)malloc(config->maxnbclust * sizeof(Candidate));

    FILE *ascii_out = NULL;
    if (config->output_membership)
    {
        char out_path[1024];
        if (config->user_outdir)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt", config->user_outdir);
        }
        else
        {
            snprintf(out_path, sizeof(out_path), "frame_membership.txt");
        }

        ascii_out = fopen(out_path, "w");
        if (!ascii_out)
        {
            perror("Failed to open frame_membership.txt");
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    long prev_missed_frames = 0;
    int prev_assigned_cluster = -1;
    Frame *current_frame;
    while ((current_frame = getframe()) != NULL)
    {
        if (stop_requested)
        {
            printf(ANSI_COLOR_ORANGE
                   "\nStopping clustering on user request (CTRL+C).\n" ANSI_COLOR_RESET);
            free_frame(current_frame);
            break;
        }

        if (state->total_frames_processed >= config->maxnbfr)
        {
            free_frame(current_frame);
            break;
        }

        if (config->verbose_level >= 2)
        {
            printf("\n  [VV] Processing Frame %5ld (Clusters: %4d)\n",
                   state->total_frames_processed, state->num_clusters);
        }

        int assigned_cluster = -1;
        int temp_count = 0;
        long start_pruned_val = state->clusters_pruned;

        if (state->num_clusters == 0)
        {
            // Step 0
            state->clusters[0].anchor = *current_frame;
            state->clusters[0].id = 0;
            state->clusters[0].prob = 1.0;
            state->num_clusters = 1;
            assigned_cluster = 0;
            state->dccarray[0] = 0.0;
            free(current_frame); // Struct only, data transferred

            add_visitor(&state->cluster_visitors[0], state->total_frames_processed);

            temp_indices[0] = 0;
            temp_dists[0] = 0.0;
            temp_count = 1;

            if (config->verbose_level >= 2)
            {
                printf(ANSI_COLOR_ORANGE
                       "  [VV] Frame %5ld created initial Cluster    0\n" ANSI_COLOR_RESET,
                       state->total_frames_processed);
            }
        }
        else
        {
            // Step 1
            double sum_prob = 0.0;
            for (int i = 0; i < state->num_clusters; i++)
                sum_prob += state->clusters[i].prob;
            if (sum_prob > 0)
            {
                for (int i = 0; i < state->num_clusters; i++)
                    state->clusters[i].prob /= sum_prob;
            }

            for (int i = 0; i < state->num_clusters; i++)
                state->current_gprobs[i] = 1.0;
            for (int i = 0; i < state->num_clusters; i++)
                state->clmembflag[i] = 1;

            // Calculate mixed probabilities
            double trans_prob_sum = 0.0;
            if (config->tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    trans_prob_sum +=
                        (double)state
                            ->transition_matrix[prev_assigned_cluster * config->maxnbclust + i];
                }
            }

            for (int i = 0; i < state->num_clusters; i++)
            {
                double prior = state->clusters[i].prob;
                double tp = 0.0;
                if (config->tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1 &&
                    trans_prob_sum > 0.0)
                {
                    tp = (double)state
                             ->transition_matrix[prev_assigned_cluster * config->maxnbclust + i] /
                         trans_prob_sum;
                    state->mixed_probs[i] =
                        (1.0 - config->tm_mixing_coeff) * prior + config->tm_mixing_coeff * tp;
                }
                else
                {
                    state->mixed_probs[i] = prior;
                }
            }

            if (!config->gprob_mode)
            {
                // Sort based on mixed_probs
                for (int i = 0; i < state->num_clusters; i++)
                {
                    sorting_candidates[i].id = i;
                    sorting_candidates[i].p = state->mixed_probs[i];
                }
                qsort(sorting_candidates, state->num_clusters, sizeof(Candidate),
                      compare_candidates);
                for (int i = 0; i < state->num_clusters; i++)
                {
                    state->probsortedclindex[i] = sorting_candidates[i].id;
                }
            }

            int k = 0;
            int found = 0;

            if (config->pred_mode && state->total_frames_processed >= config->pred_len)
            {
                int *pred_candidates = (int *)malloc(config->pred_n * sizeof(int));
                if (pred_candidates)
                {
                    int num_preds =
                        get_prediction_candidates(state, config, pred_candidates, config->pred_n);

                    for (int p = 0; p < num_preds; p++)
                    {
                        int cj = pred_candidates[p];
                        if (cj >= state->num_clusters || !state->clmembflag[cj])
                            continue;

                        if (temp_count < state->max_steps_recorded && state->num_clusters > 0)
                        {
                            int pruned_cnt = 0;
                            for (int pc = 0; pc < state->num_clusters; pc++)
                                if (state->clmembflag[pc] == 0)
                                    pruned_cnt++;
                            state->pruned_fraction_sum[temp_count] +=
                                (double)pruned_cnt / state->num_clusters;
                            state->step_counts[temp_count]++;
                        }

                        double dfc = get_dist(current_frame, &state->clusters[cj].anchor,
                                              state->clusters[cj].id, state->clusters[cj].prob,
                                              state->current_gprobs[cj], config, state);

                        if (temp_count < config->maxnbclust)
                        {
                            temp_indices[temp_count] = cj;
                            temp_dists[temp_count] = dfc;
                            temp_count++;
                        }

                        add_visitor(&state->cluster_visitors[cj], state->total_frames_processed);

                        if (dfc < config->rlim)
                        {
                            assigned_cluster = cj;
                            state->clusters[cj].prob += config->deltaprob;
                            found = 1;
                            if (config->verbose_level >= 2)
                            {
                                printf(ANSI_COLOR_GREEN "  [VV] Frame %ld assigned to Cluster %d "
                                                        "(Prediction)\n" ANSI_COLOR_RESET,
                                       state->total_frames_processed, assigned_cluster);
                            }
                            break;
                        }

                        long local_pruned = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
                        for (int cl = 0; cl < state->num_clusters; cl++)
                        {
                            if (state->clmembflag[cl] == 0)
                                continue;

                            double dcc = state->dccarray[cj * config->maxnbclust + cl];
                            if (dcc < 0)
                            {
                                dcc = get_dist(&state->clusters[cj].anchor,
                                               &state->clusters[cl].anchor, -1, -1.0, -1.0, config,
                                               state);
                                state->dccarray[cj * config->maxnbclust + cl] = dcc;
                                state->dccarray[cl * config->maxnbclust + cj] = dcc;
                            }

                            if (dcc - dfc > config->rlim)
                            {
                                state->clmembflag[cl] = 0;
                                local_pruned++;
                            }
                            else if (dfc - dcc > config->rlim)
                            {
                                state->clmembflag[cl] = 0;
                                local_pruned++;
                            }
                        }
                        state->clusters_pruned += local_pruned;

                        // TE4 Pruning
                        if (config->te4_mode && temp_count > 1)
                        {
                            for (int p = 0; p < temp_count - 1; p++)
                            {
                                int cprev = temp_indices[p];
                                double d_m_cprev = temp_dists[p];
                                double d_ci_cprev =
                                    state->dccarray[cj * config->maxnbclust + cprev];

                                if (d_ci_cprev < 0)
                                {
                                    d_ci_cprev = get_dist(&state->clusters[cj].anchor,
                                                          &state->clusters[cprev].anchor, -1, -1.0,
                                                          -1.0, config, state);
                                    state->dccarray[cj * config->maxnbclust + cprev] = d_ci_cprev;
                                    state->dccarray[cprev * config->maxnbclust + cj] = d_ci_cprev;
                                }

                                long local_pruned_te4 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned_te4) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
                                for (int k = 0; k < state->num_clusters; k++)
                                {
                                    if (!state->clmembflag[k])
                                        continue;
                                    if (k == cj || k == cprev)
                                        continue;

                                    double d_ci_ck = state->dccarray[cj * config->maxnbclust + k];
                                    if (d_ci_ck < 0)
                                    {
                                        d_ci_ck = get_dist(&state->clusters[cj].anchor,
                                                           &state->clusters[k].anchor, -1, -1.0,
                                                           -1.0, config, state);
                                        state->dccarray[cj * config->maxnbclust + k] = d_ci_ck;
                                        state->dccarray[k * config->maxnbclust + cj] = d_ci_ck;
                                    }

                                    double d_cprev_ck =
                                        state->dccarray[cprev * config->maxnbclust + k];
                                    if (d_cprev_ck < 0)
                                    {
                                        d_cprev_ck = get_dist(&state->clusters[cprev].anchor,
                                                              &state->clusters[k].anchor, -1, -1.0,
                                                              -1.0, config, state);
                                        state->dccarray[cprev * config->maxnbclust + k] =
                                            d_cprev_ck;
                                        state->dccarray[k * config->maxnbclust + cprev] =
                                            d_cprev_ck;
                                    }

                                    double min_d = calc_min_dist_4pt(dfc, d_m_cprev, d_ci_cprev,
                                                                     d_ci_ck, d_cprev_ck);
                                    if (min_d > config->rlim)
                                    {
                                        state->clmembflag[k] = 0;
                                        local_pruned_te4++;
                                    }
                                }
                                state->clusters_pruned += local_pruned_te4;
                            }
                        }

                        // TE5 Pruning
                        if (config->te5_mode)
                        {
                            prune_candidates_te5(config, state, temp_indices, temp_dists,
                                                 temp_count);
                        }

                        state->clmembflag[cj] = 0;
                    }
                    free(pred_candidates);
                }
            }

            while (!found)
            {
                if (config->verbose_level >= 2 && verbose_candidates)
                {
                    int vcount = 0;
                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        if (state->clmembflag[i])
                        {
                            double p = state->mixed_probs[i];
                            if (config->gprob_mode)
                            {
                                p *= state->current_gprobs[i];
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
                                printf(" >");
                        }
                        printf("\n");
                    }
                }

                int cj = -1;

                if (!config->gprob_mode)
                {
                    while (k < state->num_clusters &&
                           state->clmembflag[state->probsortedclindex[k]] == 0)
                        k++;
                    if (k >= state->num_clusters)
                        break;
                    cj = state->probsortedclindex[k];
                    k++;
                }
                else
                {
                    double max_p = -1.0;
                    cj = -1;
                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        if (state->clmembflag[i])
                        {
                            double p = state->mixed_probs[i] * state->current_gprobs[i];
                            if (p > max_p)
                            {
                                max_p = p;
                                cj = i;
                            }
                        }
                    }
                    if (cj == -1)
                        break;
                }

                // Track pruning stats
                if (temp_count < state->max_steps_recorded && state->num_clusters > 0)
                {
                    int pruned_cnt = 0;
                    for (int pc = 0; pc < state->num_clusters; pc++)
                    {
                        if (state->clmembflag[pc] == 0)
                            pruned_cnt++;
                    }
                    state->pruned_fraction_sum[temp_count] +=
                        (double)pruned_cnt / state->num_clusters;
                    state->step_counts[temp_count]++;
                }

                double dfc =
                    get_dist(current_frame, &state->clusters[cj].anchor, state->clusters[cj].id,
                             state->clusters[cj].prob, state->current_gprobs[cj], config, state);

                if (temp_count < config->maxnbclust)
                {
                    temp_indices[temp_count] = cj;
                    temp_dists[temp_count] = dfc;
                    temp_count++;
                }

                add_visitor(&state->cluster_visitors[cj], state->total_frames_processed);

                if (dfc < config->rlim)
                {
                    assigned_cluster = cj;
                    state->clusters[cj].prob += config->deltaprob;
                    found = 1;
                    if (config->verbose_level >= 2)
                    {
                        printf(ANSI_COLOR_GREEN
                               "  [VV] Frame %ld assigned to Cluster %d\n" ANSI_COLOR_RESET,
                               state->total_frames_processed, assigned_cluster);
                    }
                    break;
                }

                long local_pruned = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
                for (int cl = 0; cl < state->num_clusters; cl++)
                {
                    if (state->clmembflag[cl] == 0)
                        continue;

                    double dcc = state->dccarray[cj * config->maxnbclust + cl];
                    if (dcc < 0)
                    {
                        dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[cl].anchor, -1,
                                       -1.0, -1.0, config, state);
                        state->dccarray[cj * config->maxnbclust + cl] = dcc;
                        state->dccarray[cl * config->maxnbclust + cj] = dcc;
                    }

                    if (dcc - dfc > config->rlim)
                    {
                        state->clmembflag[cl] = 0;
                        local_pruned++;
                    }
                    else if (dfc - dcc > config->rlim)
                    {
                        state->clmembflag[cl] = 0;
                        local_pruned++;
                    }
                }
                state->clusters_pruned += local_pruned;

                // TE4 Pruning
                if (config->te4_mode && temp_count > 1)
                {
                    for (int p = 0; p < temp_count - 1; p++)
                    {
                        int cprev = temp_indices[p];
                        double d_m_cprev = temp_dists[p];
                        double d_ci_cprev = state->dccarray[cj * config->maxnbclust + cprev];

                        if (d_ci_cprev < 0)
                        {
                            d_ci_cprev = get_dist(&state->clusters[cj].anchor,
                                                  &state->clusters[cprev].anchor, -1, -1.0, -1.0,
                                                  config, state);
                            state->dccarray[cj * config->maxnbclust + cprev] = d_ci_cprev;
                            state->dccarray[cprev * config->maxnbclust + cj] = d_ci_cprev;
                        }

                        long local_pruned_te4 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned_te4) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
                        for (int k = 0; k < state->num_clusters; k++)
                        {
                            if (!state->clmembflag[k])
                                continue;
                            if (k == cj || k == cprev)
                                continue;

                            double d_ci_ck = state->dccarray[cj * config->maxnbclust + k];
                            if (d_ci_ck < 0)
                            {
                                d_ci_ck = get_dist(&state->clusters[cj].anchor,
                                                   &state->clusters[k].anchor, -1, -1.0, -1.0,
                                                   config, state);
                                state->dccarray[cj * config->maxnbclust + k] = d_ci_ck;
                                state->dccarray[k * config->maxnbclust + cj] = d_ci_ck;
                            }

                            double d_cprev_ck = state->dccarray[cprev * config->maxnbclust + k];
                            if (d_cprev_ck < 0)
                            {
                                d_cprev_ck = get_dist(&state->clusters[cprev].anchor,
                                                      &state->clusters[k].anchor, -1, -1.0, -1.0,
                                                      config, state);
                                state->dccarray[cprev * config->maxnbclust + k] = d_cprev_ck;
                                state->dccarray[k * config->maxnbclust + cprev] = d_cprev_ck;
                            }

                            double min_d =
                                calc_min_dist_4pt(dfc, d_m_cprev, d_ci_cprev, d_ci_ck, d_cprev_ck);
                            if (min_d > config->rlim)
                            {
                                state->clmembflag[k] = 0;
                                local_pruned_te4++;
                            }
                        }
                        state->clusters_pruned += local_pruned_te4;
                    }
                }

                // TE5 Pruning
                if (config->te5_mode)
                {
                    prune_candidates_te5(config, state, temp_indices, temp_dists, temp_count);
                }

                if (state->clmembflag[cj])
                {
                    state->clmembflag[cj] = 0;
                }

                int active_cluster_count = 0;
                for (int i = 0; i < state->num_clusters; i++)
                {
                    if (state->clmembflag[i])
                        active_cluster_count++;
                }

                if ((config->gprob_mode || (config->distall_mode && state->distall_out) ||
                     config->verbose_level >= 2) &&
                    active_cluster_count > 1)
                {
                    int match_count = state->cluster_visitors[cj].count;
                    if (match_count > 0)
                        match_count--;

                    if (config->verbose_level >= 2)
                    {
                        printf("  [VV] Distance > rlim. Found %d matches in distinfo for Cluster "
                               "%4d (Frame %5d).\n",
                               match_count, cj, state->clusters[cj].anchor.id);
                    }

                    int start_idx = 0;
                    if (state->cluster_visitors[cj].count > config->max_gprob_visitors)
                    {
                        start_idx = state->cluster_visitors[cj].count - config->max_gprob_visitors;
                    }
                    for (int i = start_idx; i < state->cluster_visitors[cj].count; i++)
                    {
                        int k_idx = state->cluster_visitors[cj].frames[i];
                        if (k_idx == state->total_frames_processed)
                            continue;

                        int target_cl = state->frame_infos[k_idx].assignment;
                        if (target_cl < 0 || target_cl >= state->num_clusters)
                            continue; // Skip discarded/invalid
                        int is_active = state->clmembflag[target_cl];

                        if (config->verbose_level >= 2)
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
                            continue;

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
                            double dr = fabs(dfc - dist_k) / config->rlim;
                            double val = fmatch(dr, config->fmatch_a, config->fmatch_b);

                            if (config->verbose_level >= 2)
                            {
                                printf("    dist %5ld-%-5d = %12.5e  dist %5d-%-5d = %12.5e, "
                                       "fmatch=%12.5e, updating GProb(Cluster %4d) from %12.5e to "
                                       "%12.5e\n",
                                       state->total_frames_processed, state->clusters[cj].anchor.id,
                                       dfc, k_idx, state->clusters[cj].anchor.id, dist_k, val,
                                       target_cl, state->current_gprobs[target_cl],
                                       state->current_gprobs[target_cl] * val);
                            }

                            state->current_gprobs[target_cl] *= val;
                        }
                    }
                }
            }

            if (!found)
            {
                if (state->num_clusters < config->maxnbclust)
                {
                    assigned_cluster = state->num_clusters;
                    state->clusters[state->num_clusters].anchor = *current_frame;
                    state->clusters[state->num_clusters].id = state->num_clusters;
                    state->clusters[state->num_clusters].prob = 1.0;

                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        double d =
                            get_dist(&state->clusters[state->num_clusters].anchor,
                                     &state->clusters[i].anchor, -1, -1.0, -1.0, config, state);
                        state->dccarray[state->num_clusters * config->maxnbclust + i] = d;
                        state->dccarray[i * config->maxnbclust + state->num_clusters] = d;
                    }
                    state
                        ->dccarray[state->num_clusters * config->maxnbclust + state->num_clusters] =
                        0.0;

                    if (config->verbose_level >= 2)
                    {
                        printf(ANSI_COLOR_GREEN
                               "  [VV] Frame %5ld assigned to Cluster %4d\n" ANSI_COLOR_RESET,
                               state->total_frames_processed, assigned_cluster);
                        printf(ANSI_COLOR_ORANGE
                               "  [VV] Frame %5ld created new Cluster %4d\n" ANSI_COLOR_RESET,
                               state->total_frames_processed, state->num_clusters);
                    }

                    add_visitor(&state->cluster_visitors[state->num_clusters],
                                state->total_frames_processed);

                    if (temp_count < config->maxnbclust)
                    {
                        temp_indices[temp_count] = state->num_clusters;
                        temp_dists[temp_count] = 0.0;
                        temp_count++;
                    }

                    state->num_clusters++;
                    free(current_frame);
                }
                else
                {
                    // Max clusters reached - apply strategy
                    if (config->maxcl_strategy == MAXCL_STOP)
                    {
                        printf(ANSI_COLOR_ORANGE "Max clusters limit reached.\n" ANSI_COLOR_RESET);
                        printf("Frames clustered: %ld\n", state->total_frames_processed);
                        free_frame(current_frame);
                        break;
                    }
                    else if (config->maxcl_strategy == MAXCL_DISCARD)
                    {
                        // Find oldest/smallest cluster to discard
                        int scan_limit = (int)(state->num_clusters * config->discard_fraction);
                        if (scan_limit < 1)
                            scan_limit = state->num_clusters;

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
                            if (prev_assigned_cluster == min_idx)
                                prev_assigned_cluster = -1;
                            else if (prev_assigned_cluster > min_idx)
                                prev_assigned_cluster--;
                            assigned_cluster = state->num_clusters;
                            state->clusters[state->num_clusters].anchor = *current_frame;
                            state->clusters[state->num_clusters].id = state->num_clusters;
                            state->clusters[state->num_clusters].prob = 1.0;

                            for (int i = 0; i < state->num_clusters; i++)
                            {
                                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                                    config, state);
                                state->dccarray[state->num_clusters * config->maxnbclust + i] = d;
                                state->dccarray[i * config->maxnbclust + state->num_clusters] = d;
                            }
                            state->dccarray[state->num_clusters * config->maxnbclust +
                                            state->num_clusters] = 0.0;

                            add_visitor(&state->cluster_visitors[state->num_clusters],
                                        state->total_frames_processed);

                            if (temp_count < config->maxnbclust)
                            {
                                temp_indices[temp_count] = state->num_clusters;
                                temp_dists[temp_count] = 0.0;
                                temp_count++;
                            }

                            state->num_clusters++;
                            free(current_frame);
                        }
                        else
                        {
                            free_frame(current_frame);
                            break;
                        }
                    }
                    else if (config->maxcl_strategy == MAXCL_MERGE)
                    {
                        // Find closest pair
                        int best_i = -1, best_j = -1;
                        double min_d = -1.0;

                        for (int i = 0; i < state->num_clusters; i++)
                        {
                            for (int j = i + 1; j < state->num_clusters; j++)
                            {
                                double d = state->dccarray[i * config->maxnbclust + j];
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
                            // Merge smaller into larger
                            int count_i = state->cluster_visitors[best_i].count;
                            int count_j = state->cluster_visitors[best_j].count;
                            int target = (count_i >= count_j) ? best_i : best_j;
                            int remove = (count_i >= count_j) ? best_j : best_i;

                            if (config->verbose_level >= 1)
                            {
                                printf("Merging cluster %d into %d (dist %.4f)\n", remove, target,
                                       min_d);
                            }

                            remove_cluster(state, config, remove, target);
                            if (prev_assigned_cluster == remove)
                            {
                                if (target > remove)
                                    prev_assigned_cluster = target - 1;
                                else
                                    prev_assigned_cluster = target;
                            }
                            else if (prev_assigned_cluster > remove)
                            {
                                prev_assigned_cluster--;
                            }

                            // Now create new cluster for current frame
                            assigned_cluster = state->num_clusters;
                            state->clusters[state->num_clusters].anchor = *current_frame;
                            state->clusters[state->num_clusters].id = state->num_clusters;
                            state->clusters[state->num_clusters].prob = 1.0;

                            for (int i = 0; i < state->num_clusters; i++)
                            {
                                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                                    config, state);
                                state->dccarray[state->num_clusters * config->maxnbclust + i] = d;
                                state->dccarray[i * config->maxnbclust + state->num_clusters] = d;
                            }
                            state->dccarray[state->num_clusters * config->maxnbclust +
                                            state->num_clusters] = 0.0;

                            add_visitor(&state->cluster_visitors[state->num_clusters],
                                        state->total_frames_processed);

                            if (temp_count < config->maxnbclust)
                            {
                                temp_indices[temp_count] = state->num_clusters;
                                temp_dists[temp_count] = 0.0;
                                temp_count++;
                            }

                            state->num_clusters++;
                            free(current_frame);
                        }
                        else
                        {
                            free_frame(current_frame);
                            break;
                        }
                    }
                }
            }
            else
            {
                free_frame(current_frame);
            }
        }

        // Update transition matrix
        if (state->total_frames_processed > 0 && prev_assigned_cluster != -1 &&
            assigned_cluster != -1)
        {
            state->transition_matrix[prev_assigned_cluster * config->maxnbclust +
                                     assigned_cluster]++;
        }
        prev_assigned_cluster = assigned_cluster;

        state->assignments[state->total_frames_processed] = assigned_cluster;
        if (ascii_out)
        {
            if (config->stream_input_mode)
            {
                fprintf(ascii_out, "%ld %d %lu %ld.%09ld\n", state->total_frames_processed,
                        assigned_cluster, current_frame->cnt0, current_frame->atime.tv_sec,
                        current_frame->atime.tv_nsec);
            }
            else
            {
                fprintf(ascii_out, "%ld %d\n", state->total_frames_processed, assigned_cluster);
            }
        }

        state->frame_infos[state->total_frames_processed].assignment = assigned_cluster;
        state->frame_infos[state->total_frames_processed].num_dists = temp_count;
        if (temp_count > 0)
        {
            state->frame_infos[state->total_frames_processed].cluster_indices =
                (int *)malloc(temp_count * sizeof(int));
            state->frame_infos[state->total_frames_processed].distances =
                (double *)malloc(temp_count * sizeof(double));
            if (state->frame_infos[state->total_frames_processed].cluster_indices &&
                state->frame_infos[state->total_frames_processed].distances)
            {
                memcpy(state->frame_infos[state->total_frames_processed].cluster_indices,
                       temp_indices, temp_count * sizeof(int));
                memcpy(state->frame_infos[state->total_frames_processed].distances, temp_dists,
                       temp_count * sizeof(double));
            }
        }
        else
        {
            state->frame_infos[state->total_frames_processed].cluster_indices = NULL;
            state->frame_infos[state->total_frames_processed].distances = NULL;
        }

        state->total_frames_processed++;

        if (state->dist_counts && temp_count <= config->maxnbclust)
        {
            state->dist_counts[temp_count]++;
            state->pruned_counts_by_dist[temp_count] += (state->clusters_pruned - start_pruned_val);
        }

        if (config->progress_mode && (state->total_frames_processed % 10 == 0 ||
                                      state->total_frames_processed == actual_frames))
        {
            state->total_missed_frames = get_missed_frames();
            double avg_dists = (state->total_frames_processed > 0)
                                   ? (double)state->framedist_calls / state->total_frames_processed
                                   : 0.0;

            printf("\rProcessing frame %ld / %ld (Clusters: %d, Dists: %ld, Avg Dists/Frame: %.3f, "
                   "Pruned: %ld, ",
                   state->total_frames_processed, actual_frames, state->num_clusters,
                   state->framedist_calls, avg_dists, state->clusters_pruned);

            if (state->total_missed_frames > prev_missed_frames)
            {
                printf("\x1b[1;37;41mMissed: %ld\x1b[0m", state->total_missed_frames);
            }
            else
            {
                printf("Missed: %ld", state->total_missed_frames);
            }

            if (config->stream_input_mode)
            {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double total_elapsed =
                    (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
                double wait_time = get_stream_wait_time();
                double wait_frac = (total_elapsed > 0) ? (wait_time / total_elapsed * 100.0) : 0.0;
                printf(", Wait: %.1f%%", wait_frac);
            }

            if (is_3d_stream_mode())
            {
                printf(", Slice: %ld/%ld, Lag: %ld", get_stream_read_slice(),
                       get_stream_write_slice(), get_stream_lag());
            }

            printf(")");
            fflush(stdout);

            prev_missed_frames = state->total_missed_frames;
        }
    }

    if (config->progress_mode)
        printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms =
        (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    if (state->num_clusters < config->maxnbclust && !stop_requested)
    {
        printf(ANSI_COLOR_GREEN "All frames clustered.\n" ANSI_COLOR_RESET);
    }

    printf("Analysis complete.\n");
    printf("Total clusters: %d\n", state->num_clusters);
    printf("Processing time: %.3f ms\n", elapsed_ms);
    printf("Framedist calls: %ld\n", state->framedist_calls);

    if (ascii_out)
        fclose(ascii_out);

    if (state->dist_counts)
    {
        printf("Samples resolved per distance count:\n");
        for (int k = 0; k <= config->maxnbclust; k++)
        {
            if (state->dist_counts[k] > 0)
            {
                printf("  Count %4d: %8ld samples, %12ld samples pruned away\n", k,
                       state->dist_counts[k], state->pruned_counts_by_dist[k]);
            }
        }
    }

    free(temp_indices);
    free(temp_dists);
    if (verbose_candidates)
        free(verbose_candidates);
    if (sorting_candidates)
        free(sorting_candidates);
}

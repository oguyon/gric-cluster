/**
 * @file cluster_core.c
 * @brief High-level orchestration of the clustering loop.
 *
 * Implements the core read-frame and assign loop of run_clustering,
 * calling down to sub-modules for single-step calculations.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_core.h"
#include "cluster_step.h"
#include "framedistance.h"
#include "frameread.h"
#include "cluster_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ANSI_COLOR_RED   "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_BLUE  "\x1b[34m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"
#define ANSI_BG_GREEN    "\x1b[42m"
#define ANSI_COLOR_BLACK "\x1b[30m"

/**
 * get_dist() - High-level distance evaluation between a frame and a cluster anchor.
 * @a:            Pointer to the first Frame.
 * @b:            Pointer to the second Frame (cluster anchor).
 * @cluster_idx:  Index of the cluster.
 * @cluster_prob: Prior predictive probability of matching the cluster.
 * @current_gprob: Geometric consistency probability.
 * @config:       Pointer to the active ClusterConfig.
 * @state:        Pointer to the active ClusterState.
 *
 * Wraps the raw `framedist` call, records statistics, writes to the distance log
 * if configured, and prints verbose traces if requested.
 *
 * Return: The Euclidean distance between frames.
 */
double get_dist(
    Frame         *a,
    Frame         *b,
    int            cluster_idx,
    double         cluster_prob,
    double         current_gprob,
    ClusterConfig *config,
    ClusterState  *state)
{
#ifdef _OPENMP
#pragma omp atomic
#endif
    state->telemetry.framedist_calls++;
    if (cluster_idx >= 0)
    {
#ifdef _OPENMP
#pragma omp atomic
#endif
        state->telemetry.framedist_calls_sample++;
    }
    else
    {
#ifdef _OPENMP
#pragma omp atomic
#endif
        state->telemetry.framedist_calls_intercluster++;
    }
    double d = framedist(a, b);

    if (config->output.distall_mode && state->distall_out)
    {
        double ratio = (config->algo.rlim > 0.0) ? d / config->algo.rlim : -1.0;
        fprintf(state->distall_out, "%-8d %-8d %-12.6f %-12.6f %-8d %-12.6f %-12.6f\n",
                a->id, b->id, d, ratio, cluster_idx, cluster_prob, current_gprob);
    }

    if (config->output.verbose_level >= 2 && cluster_idx >= 0)
    {
        printf(ANSI_COLOR_BLUE
               "  [VV] Computed distance: Frame %5d to Cluster %4d = %12.5e\n" ANSI_COLOR_RESET,
               a->id, cluster_idx, d);
    }

    return d;
}

/**
 * run_clustering() - Main entry point to perform the clustering algorithm.
 * @config: Pointer to the active ClusterConfig.
 * @state:  Pointer to the active ClusterState.
 *
 * Reads frames sequentially from the configured source, initializes the first
 * cluster, and assigns frames to matching clusters, handling new cluster creation
 * and eviction policies.
 */
void run_clustering(
    ClusterConfig *config,
    ClusterState  *state)
{
#ifdef _OPENMP
    if (config->optim.ncpu > 1)
    {
        omp_set_num_threads(config->optim.ncpu);
    }
#endif

    long actual_frames = get_num_frames();
    if (actual_frames > config->input.maxnbfr)
    {
        actual_frames = config->input.maxnbfr;
    }

    state->assignments = (int *)malloc(actual_frames * sizeof(int));
    state->frame_infos = (FrameInfo *)calloc(actual_frames, sizeof(FrameInfo));

    // Allocate telemetry and scratch tracking matrices
    {
        state->telemetry.max_steps_recorded = config->algo.maxnbclust;
        state->telemetry.pruned_fraction_sum =
            (double *)calloc(state->telemetry.max_steps_recorded, sizeof(double));
        state->telemetry.step_counts =
            (long *)calloc(state->telemetry.max_steps_recorded, sizeof(long));

        state->transition_matrix =
            (long *)calloc(config->algo.maxnbclust * config->algo.maxnbclust, sizeof(long));
        state->scratch.mixed_probs = (double *)calloc(config->algo.maxnbclust, sizeof(double));

        state->telemetry.dist_counts =
            (long *)calloc(config->algo.maxnbclust + 1, sizeof(long));
        state->telemetry.pruned_counts_by_dist =
            (long *)calloc(config->algo.maxnbclust + 1, sizeof(long));
        state->telemetry.cluster_query_counts =
            (long *)calloc(config->algo.maxnbclust, sizeof(long));
    } // Allocate telemetry and scratch tracking matrices

    int       *temp_indices = NULL;
    double    *temp_dists = NULL;
    Candidate *verbose_candidates = NULL;
    Candidate *sorting_candidates = NULL;

    // Allocate reusable query and candidate buffers
    {
        temp_indices = (int *)malloc(config->algo.maxnbclust * sizeof(int));
        temp_dists = (double *)malloc(config->algo.maxnbclust * sizeof(double));

        if (!temp_indices || !temp_dists)
        {
            perror("Memory allocation failed for temp buffers");
            return;
        }

        if (config->output.verbose_level >= 2)
        {
            verbose_candidates = (Candidate *)malloc(config->algo.maxnbclust * sizeof(Candidate));
        }

        sorting_candidates =
            (Candidate *)malloc(config->algo.maxnbclust * sizeof(Candidate));
    } // Allocate reusable query and candidate buffers

    FILE *ascii_out = NULL;
    if (config->output.output_membership)
    {
        char out_path[1024];
        if (config->output.user_outdir)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt",
                     config->output.user_outdir);
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

    int  prev_assigned_cluster = -1;
    long prev_missed_frames = 0;

    printf("Clustering sequence\n");

    // Main clustering loop: reads and assigns each frame sequentially
    for (long i = 0; i < actual_frames; i++)
    {
        // Stop execution if SIGINT interrupt signal was received
        if (stop_requested)
        {
            break;
        }

        // Fetch the next frame from the configured input source (FITS, MP4, or Stream)
        struct timespec io_start, io_end;
        clock_gettime(CLOCK_MONOTONIC, &io_start);
        Frame *current_frame = getframe();
        clock_gettime(CLOCK_MONOTONIC, &io_end);
        state->telemetry.time_io_ms += (io_end.tv_sec - io_start.tv_sec) * 1000.0 +
                                       (io_end.tv_nsec - io_start.tv_nsec) / 1000000.0;
        if (!current_frame)
        {
            break;
        }

        long start_pruned_val = state->telemetry.clusters_pruned;

        // Perform assignment logic: match to existing clusters, prune, or create a new cluster
        int res = cluster_frame(config, state, current_frame, &prev_assigned_cluster,
                                ascii_out, temp_indices, temp_dists, sorting_candidates,
                                verbose_candidates);
        // Exit loop if the max cluster count was reached and the strategy is to stop
        if (res == -2)
        {
            break;
        }

        if (state->shm_ptr != NULL)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - start.tv_sec) * 1000.0 +
                             (now.tv_nsec - start.tv_nsec) / 1000000.0;
            gric_shm_update(state, GRIC_STATUS_RUNNING, elapsed);
        }

        // Periodically print progress, telemetry stats, and streaming frame rates (fps)
        if (config->output.progress_mode &&
            (state->telemetry.total_frames_processed % 10 == 0 ||
             state->telemetry.total_frames_processed == actual_frames))
        {
            state->telemetry.total_missed_frames = get_missed_frames();
            double avg_dists = (state->telemetry.total_frames_processed > 0)
                                   ? (double)state->telemetry.framedist_calls /
                                     state->telemetry.total_frames_processed
                                   : 0.0;

            printf("\rProcessing frame %ld / %ld (Clusters: %d, Dists: %ld, Avg Dists/Frame: %.3f, "
                   "Pruned: %ld, ",
                   state->telemetry.total_frames_processed, actual_frames, state->num_clusters,
                   state->telemetry.framedist_calls, avg_dists, state->telemetry.clusters_pruned);

            if (state->telemetry.total_missed_frames > prev_missed_frames)
            {
                printf("\x1b[1;37;41mMissed: %ld\x1b[0m", state->telemetry.total_missed_frames);
            }
            else
            {
                printf("Missed: %ld", state->telemetry.total_missed_frames);
            }

            if (config->input.stream_input_mode)
            {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double rate = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
                printf(", fps: %.1f", (rate > 0.0) ?
                       state->telemetry.total_frames_processed / rate : 0.0);
            }

            printf(")");
            fflush(stdout);

            prev_missed_frames = state->telemetry.total_missed_frames;
        }
    }

    if (config->output.progress_mode)
    {
        printf("\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms =
        (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    if (state->num_clusters < config->algo.maxnbclust && !stop_requested)
    {
        printf(ANSI_COLOR_GREEN "All frames clustered.\n" ANSI_COLOR_RESET);
    }

    printf("Analysis complete.\n");
    printf("Total clusters: %d\n", state->num_clusters);
    printf("Processing time: %.3f ms\n", elapsed_ms);
    printf("Framedist calls: %ld (sample-to-cluster: %ld, inter-cluster: %ld)\n",
           state->telemetry.framedist_calls,
           state->telemetry.framedist_calls_sample,
           state->telemetry.framedist_calls_intercluster);

    double total_steps_ms = state->telemetry.time_step_1 +
                            state->telemetry.time_step_2 +
                            state->telemetry.time_step_3a +
                            state->telemetry.time_step_3b +
                            state->telemetry.time_step_3c +
                            state->telemetry.time_step_4 +
                            state->telemetry.time_step_5 +
                            state->telemetry.time_step_refine;

    if (total_steps_ms > 0.0)
    {
        printf("\nDetailed Step Timing Breakdown:\n");
        printf("  Step 1 (Base case):      %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_1,
               100.0 * state->telemetry.time_step_1 / total_steps_ms);
        printf("  Step 2 (Prediction):     %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_2,
               100.0 * state->telemetry.time_step_2 / total_steps_ms);
        printf("  Step 3a (Priors/Prune):  %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_3a,
               100.0 * state->telemetry.time_step_3a / total_steps_ms);
        printf("  Step 3b (Select Target): %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_3b,
               100.0 * state->telemetry.time_step_3b / total_steps_ms);
        if (state->telemetry.time_step_3b > 0.0)
        {
            printf("    - Score & Sort:        %9.3f ms (%5.1f%% of 3b)\n",
                   state->telemetry.time_step_3b_score,
                   100.0 * state->telemetry.time_step_3b_score / state->telemetry.time_step_3b);
            printf("    - Filter & Select:     %9.3f ms (%5.1f%% of 3b)\n",
                   state->telemetry.time_step_3b_filter,
                   100.0 * state->telemetry.time_step_3b_filter / state->telemetry.time_step_3b);
            printf("    - Entropy Eval:        %9.3f ms (%5.1f%% of 3b)\n",
                   state->telemetry.time_step_3b_eval,
                   100.0 * state->telemetry.time_step_3b_eval / state->telemetry.time_step_3b);
        }
        printf("  Step 3c (Measure Dist):  %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_3c,
               100.0 * state->telemetry.time_step_3c / total_steps_ms);
        printf("  Step 4 (Create/Evict):   %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_4,
               100.0 * state->telemetry.time_step_4 / total_steps_ms);
        printf("  Step 5 (Serialization):  %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_5,
               100.0 * state->telemetry.time_step_5 / total_steps_ms);
        printf("  DCC Bounds Refinement:   %9.3f ms (%5.1f%%)\n",
               state->telemetry.time_step_refine,
               100.0 * state->telemetry.time_step_refine / total_steps_ms);
        printf("  -------------------------------------------\n");
        printf("  Total Timed Steps:       %9.3f ms (100.0%%)\n\n", total_steps_ms);
    }

    /* Feature 4: Entropy-guided diagnostics */
    if (config->optim.entropy_mode)
    {
        uint64_t total_ecalls =
            state->telemetry.entropy_frames_gated
            + state->telemetry.entropy_frames_evaluated;
        printf("Entropy Diagnostics:\n");
        if (state->telemetry.total_frames_processed > 0
            && total_ecalls > 0)
        {
            double avg_init =
                state->telemetry.entropy_sum_initial
                / (double)state->telemetry
                      .total_frames_processed;
            printf("  Avg initial entropy:   "
                   "%6.2f bits  (~%4.1f effective"
                   " candidates)\n",
                   avg_init, pow(2.0, avg_init));
            printf("  Max initial entropy:   "
                   "%6.2f bits  (~%4.1f effective"
                   " candidates)\n",
                   state->telemetry
                       .entropy_max_initial,
                   pow(2.0, state->telemetry
                                .entropy_max_initial));

            double gate_ratio =
                (double)state->telemetry
                    .entropy_frames_gated
                / (double)total_ecalls;
            printf("  Entropy gate ratio:    "
                   "%5.1f%%  (%lu gated, %lu"
                   " evaluated)\n",
                   100.0 * gate_ratio,
                   (unsigned long)state->telemetry
                       .entropy_frames_gated,
                   (unsigned long)state->telemetry
                       .entropy_frames_evaluated);
            if (config->optim.entropy_fast_mode)
            {
                printf("  Surrogate mode:        "
                       "popcount-only (Shannon"
                       " eval skipped)\n");
            }
            if (avg_init > 5.0)
            {
                printf("  NOTE: High initial entropy"
                       " suggests many overlapping"
                       " clusters.\n"
                       "        Consider reducing"
                       " rlim.\n");
            }
            else if (gate_ratio > 0.95)
            {
                printf("  NOTE: Gate ratio > 95%%"
                       " — greedy mode may be"
                       " sufficient.\n");
            }
        }
        else
        {
            printf("  No entropy evaluations"
                   " recorded.\n");
        }
        printf("\n");
    }

    if (ascii_out)
    {
        fclose(ascii_out);
    }

    if (state->telemetry.dist_counts)
    {
        printf("Samples resolved per distance count:\n");
        for (int k = 0; k <= config->algo.maxnbclust; k++)
        {
            if (state->telemetry.dist_counts[k] > 0)
            {
                printf("  Count %4d: %8ld samples, %12ld samples pruned away\n", k,
                       state->telemetry.dist_counts[k], state->telemetry.pruned_counts_by_dist[k]);
            }
        }
    }

    free(sorting_candidates);
    if (verbose_candidates)
    {
        free(verbose_candidates);
    }
    free(temp_indices);
    free(temp_dists);
}

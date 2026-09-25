/**
 * @file cluster_core.c
 * @brief High-level orchestration of the clustering loop.
 *
 * Implements the core read-frame and assign loop of run_clustering,
 * calling down to sub-modules for single-step calculations.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_core.h"
#include "cluster_core_multitile.h"
#include "cluster_reassign.h"
#include "cluster_step.h"
#include "framedistance.h"
#include "frameread.h"
#include "cluster_shm.h"
#include "tile_map.h"
#include "tile_state.h"
#include "eq16_quant.h"
#include "frame_info_arena.h"
#ifdef USE_CUDA
#include "cluster_cuda.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

volatile sig_atomic_t stop_requested = 0;

#define ANSI_COLOR_RED   "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_BLUE  "\x1b[34m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"
#define ANSI_BG_GREEN    "\x1b[42m"
#define ANSI_COLOR_BLACK "\x1b[30m"

/**
 * get_dist_cutoff() - High-level distance evaluation with early-exit cutoff.
 * @a:             Pointer to the first Frame.
 * @b:             Pointer to the second Frame (cluster anchor).
 * @cluster_idx:   Index of the cluster.
 * @cluster_prob:  Prior predictive probability of matching the cluster.
 * @current_gprob: Geometric consistency probability.
 * @cutoff_sq:     Squared distance threshold for early exit (<= 0.0 disables cutoff).
 * @config:        Pointer to the active ClusterConfig.
 * @state:         Pointer to the active ClusterState.
 *
 * Purpose & Context ("What is this used for?"):
 * Wraps framedist_cutoff(), records statistics, writes to the distance log
 * if configured, and prints verbose traces if requested.
 *
 * Return: Euclidean distance if <= sqrt(cutoff_sq), or value > sqrt(cutoff_sq) on cutoff.
 */
double get_dist_cutoff(
    Frame         *a,
    Frame         *b,
    int            cluster_idx,
    double         cluster_prob,
    double         current_gprob,
    double         cutoff_sq,
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

    if (config->output.distall_mode)
    {
        cutoff_sq = 0.0;
    }

    double d = framedist_cutoff(a, b, cutoff_sq);

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
 * get_dist() - High-level distance evaluation between a frame and a cluster anchor.
 * @a:             Pointer to the first Frame.
 * @b:             Pointer to the second Frame (cluster anchor).
 * @cluster_idx:   Index of the cluster.
 * @cluster_prob:  Prior predictive probability of matching the cluster.
 * @current_gprob: Geometric consistency probability.
 * @config:        Pointer to the active ClusterConfig.
 * @state:         Pointer to the active ClusterState.
 *
 * Purpose & Context ("What is this used for?"):
 * Wraps get_dist_cutoff() with cutoff disabled (cutoff_sq = 0.0).
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
    return get_dist_cutoff(
        a,
        b,
        cluster_idx,
        cluster_prob,
        current_gprob,
        0.0,
        config,
        state);
}

/**
 * cluster_core_dispatch_multitile() - Evaluate and dispatch multi-tile clustering if needed.
 * @config: Active clustering configuration.
 *
 * Purpose & Context ("What is this used for?"):
 * Inspects tile grid geometry (tile_grid_x, tile_grid_y) or custom tile map FITS file.
 * If multiple tiles are detected, creates the TileMap and MultiTileState, applies tile
 * configuration overrides, executes run_clustering_multitile(), and frees resources.
 *
 * Return: 1 if multi-tile clustering was handled and completed, 0 for single-tile execution.
 */
static int cluster_core_dispatch_multitile(
    ClusterConfig *config)
{
    int num_tiles = 1;
    if (config->input.tile_grid_x > 0 && config->input.tile_grid_y > 0)
    {
        num_tiles = config->input.tile_grid_x * config->input.tile_grid_y;
    }
    else if (config->input.tile_map_file != NULL)
    {
        num_tiles = 2; /* actual count from FITS */
    }

    if (num_tiles <= 1)
    {
        return 0;
    }

    long w = get_frame_width();
    long h = get_frame_height();

    TileMap *tm = NULL;
    if (config->input.tile_map_file != NULL)
    {
        tm = tilemap_load_fits(config->input.tile_map_file, w, h);
    }
    else
    {
        tm = tilemap_create_grid(w, h, config->input.tile_grid_x, config->input.tile_grid_y);
    }
    if (tm == NULL)
    {
        fprintf(stderr, "ERROR: tile map creation failed\n");
        return 1;
    }

    printf("Multi-tile mode: %d tiles (%ldx%ld image)\n", tm->num_tiles, w, h);

    MultiTileState *mts = multitile_init(config, tm, config->input.maxnbfr);
    if (mts == NULL)
    {
        fprintf(stderr, "ERROR: multitile_init failed\n");
        tilemap_free(tm);
        return 1;
    }

    /* Load per-tile config overrides */
    if (config->input.tile_config_file)
    {
        multitile_load_tile_config(mts, config->input.tile_config_file);
    }

    run_clustering_multitile(config, mts);

    multitile_free(mts);
    tilemap_free(tm);
    return 1;
}

/**
 * cluster_core_resolve_quantization() - Determine default quantization scheme.
 * @config: Active clustering configuration.
 *
 * Purpose & Context ("What is this used for?"):
 * Automatically selects the optimal vector quantization mode based on frame dimension:
 * - Dimensions divisible by 8 and >= 8 default to E8 lattice quantization (EQ16 with ADC).
 * - Dimensions >= 32 default to 16-bit scalar quantization (SQ16).
 * - Smaller dimensions default to 8-bit scalar quantization (SQ8).
 * Normalizes negative sentinel flags (-1) to concrete boolean values (0 or 1).
 */
static void cluster_core_resolve_quantization(
    ClusterConfig *config)
{
    if (config->optim.use_sq8 < 0 && config->optim.use_sq16 < 0 && config->optim.use_eq16 < 0)
    {
        long dim = get_frame_width() * get_frame_height();
        if (dim >= 8 && (dim % 8 == 0))
        {
            config->optim.use_eq16 = 1;
            config->optim.use_sq16 = 0;
            config->optim.use_sq8 = 0;
        }
        else if (dim >= 32)
        {
            config->optim.use_sq16 = 1;
            config->optim.use_eq16 = 0;
            config->optim.use_sq8 = 0;
        }
        else
        {
            config->optim.use_sq8 = 1;
            config->optim.use_sq16 = 0;
            config->optim.use_eq16 = 0;
        }
    }
    else
    {
        if (config->optim.use_sq8 < 0)
        {
            config->optim.use_sq8 = 0;
        }
        if (config->optim.use_sq16 < 0)
        {
            config->optim.use_sq16 = 0;
        }
        if (config->optim.use_eq16 < 0)
        {
            config->optim.use_eq16 = 0;
        }
    }

    if (config->optim.use_eq16 && config->optim.use_eq16_adc < 0)
    {
        config->optim.use_eq16_adc = 1;
    }
}

/**
 * cluster_core_allocate_state() - Pre-allocate state tables and tracking matrices.
 * @config:        Active clustering configuration.
 * @state:         Clustering state receiving allocated pointers.
 * @actual_frames: Maximum number of frames to allocate storage for.
 *
 * Purpose & Context ("What is this used for?"):
 * Pre-allocates frame assignment arrays, distance logs, frame metadata structs, telemetry
 * accumulators, transition frequency matrix, and SQ16 memoization hash tables.
 *
 * Return: 0 on success, -1 on allocation failure.
 */
static int cluster_core_allocate_state(
    ClusterConfig *config,
    ClusterState  *state,
    long           actual_frames)
{
    state->assignments = (int *)malloc(actual_frames * sizeof(int));
    state->assignment_dists = (double *)malloc(actual_frames * sizeof(double));
    state->frame_infos = (FrameInfo *)calloc(actual_frames, sizeof(FrameInfo));
    frame_info_arena_init(&state->frame_info_arena, FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE);

    if (!state->assignments || !state->assignment_dists || !state->frame_infos)
    {
        return -1;
    }

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

    if (config->optim.use_sq16 && config->optim.use_memo)
    {
        quant_memo_init(&state->scratch.memo_table,
                        QUANT_MEMO_DEFAULT_CAPACITY,
                        config->optim.sq16_params.err_radius,
                        config->algo.rlim);
        state->telemetry.memo_cache_capacity = state->scratch.memo_table.capacity;
    }

    return 0;
}

/**
 * cluster_core_setup_output_files() - Open text membership and evaluation log files.
 * @config:    Clustering configuration specifying output directories and modes.
 * @state:     Clustering state receiving evals_out handle.
 * @ascii_out: Output pointer to receive opened frame_membership.txt FILE handle.
 *
 * Purpose & Context ("What is this used for?"):
 * Opens frame_membership.txt and frame_evals.txt with 64KB stream buffers for high-speed
 * sequential writing unless text output is suppressed by command-line options.
 */
static void cluster_core_setup_output_files(
    ClusterConfig *config,
    ClusterState  *state,
    FILE         **ascii_out)
{
    *ascii_out = NULL;
    if (config->output.output_membership && !config->output.no_txt)
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

        *ascii_out = fopen(out_path, "w");
        if (!*ascii_out)
        {
            perror("Failed to open frame_membership.txt");
        }
        else
        {
            setvbuf(*ascii_out, NULL, _IOFBF, 65536);
        }
    }

    state->evals_out = NULL;
    if (config->output.output_evals && !config->output.no_txt)
    {
        char out_path[1024];
        if (config->output.user_outdir)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_evals.txt",
                     config->output.user_outdir);
        }
        else
        {
            snprintf(out_path, sizeof(out_path), "frame_evals.txt");
        }

        state->evals_out = fopen(out_path, "w");
        if (!state->evals_out)
        {
            perror("Failed to open frame_evals.txt");
        }
        else
        {
            setvbuf(state->evals_out, NULL, _IOFBF, 65536);
        }
    }
}

/**
 * cluster_core_run_pass1_loop() - Sequential frame ingestion and cluster assignment loop.
 * @config:             Active clustering configuration.
 * @state:              Mutable clustering state.
 * @actual_frames:      Total number of frames to process.
 * @ascii_out:          Open membership log file handle.
 * @temp_indices:       Pre-allocated candidate index buffer.
 * @temp_dists:         Pre-allocated distance evaluation buffer.
 * @sorting_candidates: Pre-allocated candidate sorting array.
 * @verbose_candidates: Pre-allocated verbose candidate array.
 * @start_time:         Monotonic clock start time for FPS and rate calculations.
 *
 * Purpose & Context ("What is this used for?"):
 * The main computational driver of single-tile clustering. Sequentially reads incoming frames
 * from FITS/video/stream inputs, invokes cluster_frame() across Steps 1-5, monitors stop
 * signals, updates shared memory telemetry, and prints terminal progress indicators.
 */
static void cluster_core_run_pass1_loop(
    ClusterConfig   *config,
    ClusterState    *state,
    long             actual_frames,
    FILE            *ascii_out,
    int             *temp_indices,
    double          *temp_dists,
    Candidate       *sorting_candidates,
    Candidate       *verbose_candidates,
    struct timespec  start_time)
{
    int  prev_assigned_cluster = -1;
    long prev_missed_frames = 0;

    for (long i = 0; i < actual_frames; i++)
    {
        if (stop_requested)
        {
            break;
        }

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

        int res = cluster_frame(config, state, current_frame, &prev_assigned_cluster,
                                ascii_out, temp_indices, temp_dists, sorting_candidates,
                                verbose_candidates);
        if (res == -2)
        {
            break;
        }

        if (state->shm_ptr != NULL)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - start_time.tv_sec) * 1000.0 +
                             (now.tv_nsec - start_time.tv_nsec) / 1000000.0;
            gric_shm_update(state, GRIC_STATUS_RUNNING, elapsed);
        }

        if (config->output.progress_mode &&
            (state->telemetry.total_frames_processed % 10 == 0 ||
             state->telemetry.total_frames_processed == actual_frames))
        {
            state->telemetry.total_missed_frames = get_missed_frames();
            double avg_dists = (state->telemetry.total_frames_processed > 0)
                                   ? (double)state->telemetry.framedist_calls /
                                     state->telemetry.total_frames_processed
                                   : 0.0;

            printf("\rProcessing frame %ld / %ld (Clusters: %d, Dists: %ld, "
                   "Avg Dists/Frame: %.3f, Pruned: %ld, ",
                   state->telemetry.total_frames_processed, actual_frames,
                   state->num_clusters, state->telemetry.framedist_calls,
                   avg_dists, state->telemetry.clusters_pruned);

            if (state->telemetry.total_missed_frames > prev_missed_frames)
            {
                printf("\x1b[1;37;41mMissed: %ld\x1b[0m",
                       state->telemetry.total_missed_frames);
            }
            else
            {
                printf("Missed: %ld", state->telemetry.total_missed_frames);
            }

            if (config->input.stream_input_mode)
            {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double rate = (now.tv_sec - start_time.tv_sec) +
                              (now.tv_nsec - start_time.tv_nsec) / 1e9;
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
}

/**
 * cluster_core_print_diagnostics() - Output detailed step profiling and telemetry reports.
 * @config:     Active clustering configuration.
 * @state:      Clustering state holding profiling and telemetry counters.
 * @elapsed_ms: Total elapsed clustering wall-clock time in milliseconds.
 *
 * Purpose & Context ("What is this used for?"):
 * Summarizes end-of-run diagnostics to stdout, including total clusters discovered, distance
 * evaluation counts, microsecond timing breakdown across pipeline steps 1 through 5, entropy
 * gating statistics, temporal prediction hit rates, and quantization pruning performance.
 */
static void cluster_core_print_diagnostics(
    ClusterConfig *config,
    ClusterState  *state,
    double         elapsed_ms)
{
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
                            state->telemetry.time_step_refine +
                            state->telemetry.time_pass2;

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
        if (state->telemetry.time_step_3a > 0.0)
        {
            printf("    - Priors & Mixing:     %9.3f ms (%5.1f%% of 3a)\n",
                   state->telemetry.time_step_3a_priors,
                   100.0 * state->telemetry.time_step_3a_priors /
                   state->telemetry.time_step_3a);
            printf("    - SQ Lower-Bound:      %9.3f ms (%5.1f%% of 3a)\n",
                   state->telemetry.time_step_3a_sq_filter,
                   100.0 * state->telemetry.time_step_3a_sq_filter /
                   state->telemetry.time_step_3a);
            if (state->telemetry.time_step_3a_sq_tier1 > 0.0 ||
                state->telemetry.time_step_3a_sq_tier2 > 0.0)
            {
                printf("        * Tier 1 (SDC Coarse): %7.3f ms (%5.1f%% of SQ)\n",
                       state->telemetry.time_step_3a_sq_tier1,
                       100.0 * state->telemetry.time_step_3a_sq_tier1 /
                       state->telemetry.time_step_3a_sq_filter);
                printf("        * Tier 2 (ADC Refine): %7.3f ms (%5.1f%% of SQ)\n",
                       state->telemetry.time_step_3a_sq_tier2,
                       100.0 * state->telemetry.time_step_3a_sq_tier2 /
                       state->telemetry.time_step_3a_sq_filter);
            }
            printf("    - Subsequent Prune:    %9.3f ms (%5.1f%% of 3a)\n",
                   state->telemetry.time_step_3a_subsequent,
                   100.0 * state->telemetry.time_step_3a_subsequent /
                   state->telemetry.time_step_3a);
        }
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
        if (state->telemetry.time_pass2 > 0.0)
        {
            printf("  Pass 2 (Reassignment):   %9.3f ms (%5.1f%%)\n",
                   state->telemetry.time_pass2,
                   100.0 * state->telemetry.time_pass2 / total_steps_ms);
        }
        printf("  -------------------------------------------\n");
        printf("  Total Timed Steps:       %9.3f ms (100.0%%)\n\n", total_steps_ms);
    }

    if (config->optim.entropy_mode)
    {
        uint64_t total_ecalls =
            state->telemetry.entropy_frames_gated + state->telemetry.entropy_frames_evaluated;
        printf("Entropy Diagnostics:\n");
        if (state->telemetry.total_frames_processed > 0 && total_ecalls > 0)
        {
            double avg_init =
                state->telemetry.entropy_sum_initial /
                (double)state->telemetry.total_frames_processed;
            printf("  Avg initial entropy:   %6.2f bits  (~%4.1f effective candidates)\n",
                   avg_init, pow(2.0, avg_init));
            printf("  Max initial entropy:   %6.2f bits  (~%4.1f effective candidates)\n",
                   state->telemetry.entropy_max_initial,
                   pow(2.0, state->telemetry.entropy_max_initial));

            double gate_ratio =
                (double)state->telemetry.entropy_frames_gated / (double)total_ecalls;
            printf("  Entropy gate ratio:    %5.1f%%  (%lu gated, %lu evaluated)\n",
                   100.0 * gate_ratio,
                   (unsigned long)state->telemetry.entropy_frames_gated,
                   (unsigned long)state->telemetry.entropy_frames_evaluated);
            if (config->optim.entropy_fast_mode)
            {
                printf("  Surrogate mode:        popcount-only (Shannon eval skipped)\n");
            }
            if (avg_init > 5.0)
            {
                printf("  NOTE: High initial entropy suggests many overlapping clusters.\n"
                       "        Consider reducing rlim.\n");
            }
            else if (gate_ratio > 0.95)
            {
                printf("  NOTE: Gate ratio > 95%% — greedy mode may be sufficient.\n");
            }
        }
        else
        {
            printf("  No entropy evaluations recorded.\n");
        }
        printf("\n");
    }

    if (config->optim.pred_mode && state->telemetry.pred_attempts > 0)
    {
        uint64_t att = state->telemetry.pred_attempts;
        uint64_t hits = state->telemetry.pred_hits;
        uint64_t same = state->telemetry.pred_same_as_last;
        double hit_pct = 100.0 * (double)hits / (double)att;
        double same_pct = 100.0 * (double)same / (double)att;

        printf("Prediction Diagnostics:\n");
        printf("  Attempts:       %8lu\n", (unsigned long)att);
        printf("  Hits (1st ok):  %8lu  (%5.1f%%)\n", (unsigned long)hits, hit_pct);
        printf("  Misses:         %8lu  (%5.1f%%)\n", (unsigned long)(att - hits),
               100.0 - hit_pct);
        printf("  Same as last:   %8lu  (%5.1f%%)\n\n", (unsigned long)same, same_pct);
    }

    if (config->optim.use_eq16)
    {
        printf("E8 Lattice Quantization (EQ16) Diagnostics:\n");
        printf("  Mode:           %s\n",
               config->optim.use_eq16_adc ? "Asymmetric Distance Computation (ADC)"
                                          : "Symmetric Distance Computation (SDC)");
        printf("  Slack Margin:   %s\n",
               config->optim.use_eq16_adc ? "1.0 * R_c (tight lower bound)"
                                          : "2.0 * R_c (symmetric bound)");
        printf("  EQ16 Evaluated: %8lu\n", (unsigned long)state->telemetry.eq16_evals);
        printf("  EQ16 Pruned:    %8lu\n", (unsigned long)state->telemetry.eq16_pruned);
        eq16_print_checkpoint_stats();
        printf("\n");
    }
    else if (config->optim.use_sq16)
    {
        printf("Scalar Quantization (SQ16) Diagnostics:\n");
        printf("  SQ16 Evaluated: %8lu\n", (unsigned long)state->telemetry.sq16_evals);
        printf("  SQ16 Pruned:    %8lu\n", (unsigned long)state->telemetry.sq16_pruned);
        if (config->optim.use_memo)
        {
            double hit_pct = state->telemetry.memo_lookups > 0
                ? (100.0 * (double)state->telemetry.memo_hits /
                   (double)state->telemetry.memo_lookups)
                : 0.0;
            double mb = (double)(state->telemetry.memo_cache_capacity *
                                 sizeof(QuantizedMemoEntry)) / (1024.0 * 1024.0);
            printf("  Memo Cache:     %8lu / %lu entries (%.2f MB)\n",
                   (unsigned long)state->telemetry.memo_cache_entries,
                   (unsigned long)state->telemetry.memo_cache_capacity,
                   mb);
            printf("  Cell Hits:      %8lu / %lu lookups (%5.1f%%)\n",
                   (unsigned long)state->telemetry.memo_hits,
                   (unsigned long)state->telemetry.memo_lookups,
                   hit_pct);
        }
        printf("\n");
    }
    else if (config->optim.use_sq8)
    {
        printf("Scalar Quantization (SQ8) Diagnostics:\n");
        printf("  SQ8 Evaluated:  %8lu\n", (unsigned long)state->telemetry.sq8_evals);
        printf("  SQ8 Pruned:     %8lu\n\n", (unsigned long)state->telemetry.sq8_pruned);
    }

    print_clustering_metrics(state, -1);
    printf("\n");

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
}

/**
 * cluster_core_cleanup() - Release temporary scratch buffers and quantization matrices.
 * @config:             Active clustering configuration.
 * @state:              Clustering state holding matrices to free.
 * @ascii_out:          File handle to close if open.
 * @temp_indices:       Scratch index buffer to free.
 * @temp_dists:         Scratch distance buffer to free.
 * @sorting_candidates: Candidate buffer to free.
 * @verbose_candidates: Verbose candidate buffer to free.
 *
 * Purpose & Context ("What is this used for?"):
 * Flushes and closes open text files, frees temporary SIMD quantization structures and
 * candidate scratch buffers, ensuring no memory leaks occur at clustering completion.
 */
static void cluster_core_cleanup(
    ClusterConfig *config,
    ClusterState  *state,
    FILE          *ascii_out,
    int           *temp_indices,
    double        *temp_dists,
    Candidate     *sorting_candidates,
    Candidate     *verbose_candidates)
{
    if (ascii_out)
    {
        fclose(ascii_out);
    }
    if (state->evals_out)
    {
        fclose(state->evals_out);
        state->evals_out = NULL;
    }

    if (state->current_frame_sq8)
    {
        free(state->current_frame_sq8);
        state->current_frame_sq8 = NULL;
    }
    if (state->current_frame_sq16)
    {
        free(state->current_frame_sq16);
        state->current_frame_sq16 = NULL;
    }
    if (state->current_frame_eq16)
    {
        free(state->current_frame_eq16);
        state->current_frame_eq16 = NULL;
    }
    if (state->current_frame_eq16_adc)
    {
        free(state->current_frame_eq16_adc);
        state->current_frame_eq16_adc = NULL;
    }
    if (config->optim.use_sq16 && config->optim.use_memo)
    {
        quant_memo_free(&state->scratch.memo_table);
    }
    if (state->anchor_matrix_sq8)
    {
        free(state->anchor_matrix_sq8);
        state->anchor_matrix_sq8 = NULL;
    }
    else if (state->clusters)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->clusters[i].anchor_sq8)
            {
                free(state->clusters[i].anchor_sq8);
                state->clusters[i].anchor_sq8 = NULL;
            }
        }
    }

    if (state->anchor_matrix_eq16)
    {
        free(state->anchor_matrix_eq16);
        state->anchor_matrix_eq16 = NULL;
    }
    else if (state->clusters)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->clusters[i].anchor_eq16)
            {
                free(state->clusters[i].anchor_eq16);
                state->clusters[i].anchor_eq16 = NULL;
            }
        }
    }

    if (state->anchor_matrix_sq16)
    {
        free(state->anchor_matrix_sq16);
        state->anchor_matrix_sq16 = NULL;
    }
    else if (state->clusters)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->clusters[i].anchor_sq16)
            {
                free(state->clusters[i].anchor_sq16);
                state->clusters[i].anchor_sq16 = NULL;
            }
        }
    }

    if (state->anchor_matrix_sq16_interleaved)
    {
        free(state->anchor_matrix_sq16_interleaved);
        state->anchor_matrix_sq16_interleaved = NULL;
    }
    if (state->anchor_matrix_eq16_interleaved)
    {
        free(state->anchor_matrix_eq16_interleaved);
        state->anchor_matrix_eq16_interleaved = NULL;
    }
    if (state->anchor_matrix_adc_interleaved)
    {
        free(state->anchor_matrix_adc_interleaved);
        state->anchor_matrix_adc_interleaved = NULL;
    }
    if (state->anchor_matrix_float)
    {
        free(state->anchor_matrix_float);
        state->anchor_matrix_float = NULL;
    }
    if (state->anchor_norms_float)
    {
        free(state->anchor_norms_float);
        state->anchor_norms_float = NULL;
    }

    free(sorting_candidates);
    if (verbose_candidates)
    {
        free(verbose_candidates);
    }
    free(temp_indices);
    free(temp_dists);
}

/**
 * run_clustering() - Main entry point to perform the clustering algorithm.
 * @config: Pointer to the active ClusterConfig.
 * @state:  Pointer to the active ClusterState.
 *
 * Purpose & Context ("What is this used for?"):
 * High-level orchestrator for single-tile and multi-tile clustering. Configures threading,
 * resolves vector quantization defaults, allocates tracking state, invokes pass 1 frame
 * ingestion, executes optional pass 2 nearest-anchor reassignment, prints telemetry,
 * and releases resources.
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

    if (cluster_core_dispatch_multitile(config))
    {
        return;
    }

    cluster_core_resolve_quantization(config);

    long actual_frames = get_num_frames();
    if (actual_frames > config->input.maxnbfr)
    {
        actual_frames = config->input.maxnbfr;
    }

    if (cluster_core_allocate_state(config, state, actual_frames) != 0)
    {
        fprintf(stderr, "ERROR: cluster_core_allocate_state failed\n");
        return;
    }

    int       *temp_indices = (int *)malloc(config->algo.maxnbclust * sizeof(int));
    double    *temp_dists = (double *)malloc(config->algo.maxnbclust * sizeof(double));
    Candidate *sorting_candidates =
        (Candidate *)malloc(config->algo.maxnbclust * sizeof(Candidate));
    Candidate *verbose_candidates = NULL;

    if (!temp_indices || !temp_dists || !sorting_candidates)
    {
        perror("Memory allocation failed for temp buffers");
        return;
    }

    if (config->output.verbose_level >= 2)
    {
        verbose_candidates =
            (Candidate *)malloc(config->algo.maxnbclust * sizeof(Candidate));
    }

    FILE *ascii_out = NULL;
    cluster_core_setup_output_files(config, state, &ascii_out);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    printf("Clustering sequence\n");

#ifdef USE_CUDA
    int gpu_pass1_executed = 0;
    if (config->optim.use_gpu_pass1)
    {
        if (cluster_cuda_is_available())
        {
            if (ascii_out != NULL)
            {
                fclose(ascii_out);
                ascii_out = NULL;
            }
            if (cluster_cuda_run_pass1_bruteforce(config, state) == 0)
            {
                gpu_pass1_executed = 1;
            }
            else
            {
                fprintf(stderr,
                        "Warning: GPU Pass 1 failed, falling back to CPU.\n");
                if (config->output.output_membership && !config->output.no_txt)
                {
                    char out_path[1024];
                    if (config->output.user_outdir != NULL)
                    {
                        snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt",
                                 config->output.user_outdir);
                    }
                    else
                    {
                        snprintf(out_path, sizeof(out_path), "frame_membership.txt");
                    }
                    ascii_out = fopen(out_path, "w");
                    if (ascii_out != NULL)
                    {
                        setvbuf(ascii_out, NULL, _IOFBF, 65536);
                    }
                }
            }
        }
        else
        {
            fprintf(stderr,
                    "Warning: GPU Pass 1 requested but CUDA not available. "
                    "Falling back to CPU.\n");
        }
    }

    if (!gpu_pass1_executed)
#endif
    {
        cluster_core_run_pass1_loop(config, state, actual_frames, ascii_out,
                                    temp_indices, temp_dists, sorting_candidates,
                                    verbose_candidates, start);
    }

    if (state->scratch.cluster_probs != NULL)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].prob = state->scratch.cluster_probs[i];
        }
    }

    if (ascii_out)
    {
        fclose(ascii_out);
        ascii_out = NULL;
    }

    if (config->algo.pass2_nearest_mode && !stop_requested)
    {
        run_second_pass_clustering(config, state);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms =
        (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    cluster_core_print_diagnostics(config, state, elapsed_ms);

    cluster_core_cleanup(config, state, ascii_out, temp_indices, temp_dists,
                         sorting_candidates, verbose_candidates);
}

/**
 * print_clustering_metrics() - Computes and prints quality metrics of clustering.
 * @state:   Pointer to the active ClusterState.
 * @tile_id: ID of the tile (-1 for single-tile).
 *
 * Purpose & Context ("What is this used for?"):
 * Evaluates quality-of-fit and dispersion metrics for the clustering output, including
 * global root-mean-square (RMS) assignment distance, per-cluster RMS radius, cluster size
 * distribution (min, max, mean), and Shannon entropy of cluster probabilities. Used by
 * gric-cluster at completion of single-tile clustering and by run_clustering_multitile
 * for each tile.
 */
void print_clustering_metrics(
    const ClusterState *state,
    int                 tile_id)
{
    int K = state->num_clusters;
    if (K <= 0)
    {
        return;
    }

    long total_frames = state->telemetry.total_frames_processed;
    if (total_frames <= 0)
    {
        return;
    }

    long *counts = (long *)calloc((size_t)K, sizeof(long));
    double *sum_sq_dist = (double *)calloc((size_t)K, sizeof(double));
    double *min_dist = (double *)malloc((size_t)K * sizeof(double));
    double *max_dist = (double *)malloc((size_t)K * sizeof(double));

    if (!counts || !sum_sq_dist || !min_dist || !max_dist)
    {
        if (counts) free(counts);
        if (sum_sq_dist) free(sum_sq_dist);
        if (min_dist) free(min_dist);
        if (max_dist) free(max_dist);
        return;
    }

    for (int k = 0; k < K; k++)
    {
        min_dist[k] = 1e19;
        max_dist[k] = -1.0;
    }

    long assigned_count = 0;
    for (long t = 0; t < total_frames; t++)
    {
        int assigned_cl = state->frame_infos[t].assignment;
        if (assigned_cl < 0 || assigned_cl >= K)
        {
            continue;
        }

        double d = 0.0;
        if (state->frame_infos[t].cluster_indices && state->frame_infos[t].distances)
        {
            for (int i = 0; i < state->frame_infos[t].num_dists; i++)
            {
                if (state->frame_infos[t].cluster_indices[i] == assigned_cl)
                {
                    d = state->frame_infos[t].distances[i];
                    break;
                }
            }
        }

        counts[assigned_cl]++;
        sum_sq_dist[assigned_cl] += d * d;
        if (d < min_dist[assigned_cl])
        {
            min_dist[assigned_cl] = d;
        }
        if (d > max_dist[assigned_cl])
        {
            max_dist[assigned_cl] = d;
        }
        assigned_count++;
    }

    if (assigned_count <= 0)
    {
        free(counts);
        free(sum_sq_dist);
        free(min_dist);
        free(max_dist);
        return;
    }

    double total_sum_sq = 0.0;
    long min_size = -1;
    long max_size = 0;
    double entropy = 0.0;
    int active_clusters = 0;
    double sum_rms = 0.0;

    for (int k = 0; k < K; k++)
    {
        if (counts[k] > 0)
        {
            active_clusters++;
            total_sum_sq += sum_sq_dist[k];
            if (min_size == -1 || counts[k] < min_size)
            {
                min_size = counts[k];
            }
            if (counts[k] > max_size)
            {
                max_size = counts[k];
            }

            double p = (double)counts[k] / (double)assigned_count;
            entropy -= p * log2(p);

            double rms = sqrt(sum_sq_dist[k] / (double)counts[k]);
            sum_rms += rms;
        }
    }

    double global_rms = sqrt(total_sum_sq / (double)assigned_count);
    double avg_cluster_rms = (active_clusters > 0) ? (sum_rms / (double)active_clusters) : 0.0;
    double avg_size = (double)assigned_count / (double)active_clusters;

    double global_max_dist = 0.0;
    for (int k = 0; k < K; k++)
    {
        if (counts[k] > 0 && max_dist[k] > global_max_dist)
        {
            global_max_dist = max_dist[k];
        }
    }

    if (tile_id >= 0)
    {
        printf("  Tile %3d Clustering Metrics:\n", tile_id);
    }
    else
    {
        printf("Clustering Metrics:\n");
    }
    printf("    Clusters:            %d (%d active)\n", K, active_clusters);
    printf("    Assigned Frames:     %ld / %ld\n", assigned_count, total_frames);
    printf("    RMS Distance:        %.4f  (Avg cluster RMS: %.4f, Max: %.4f)\n",
           global_rms, avg_cluster_rms, global_max_dist);
    printf("    Cluster Sizes:       Min=%ld, Max=%ld, Mean=%.1f\n",
           min_size, max_size, avg_size);
    printf("    Cluster Entropy:     %.4f bits\n", entropy);

    free(counts);
    free(sum_sq_dist);
    free(min_dist);
    free(max_dist);
}

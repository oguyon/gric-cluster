/**
 * @file main.c
 * @brief Entry point for the gric-cluster application.
 *
 * Parses initial CLI arguments, configures signal handlers, loads inputs, and invokes
 * the main clustering runner.
 *
 * Main Functions:
 * - main: High-level orchestrator of the clustering pipeline.
 */
#include "cluster_core.h"
#include "cluster_defs.h"
#include "cluster_cli.h"
#include "cluster_help.h"
#include "cluster_io.h"
#include "cluster_scandist.h"
#include "config_utils.h"
#include "cluster_shm.h"
#include "frameread.h"
#include "gric_profile.h"
#include "cli_colors.h"
#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

volatile sig_atomic_t stop_requested = 0;

void handle_sigint(int sig)
{
    (void)sig;
    stop_requested = 1;
}

void print_args_on_error(int argc, char *argv[])
{
    fprintf(stderr, "\nProgram arguments:\n");
    for (int arg_idx = 0; arg_idx < argc; arg_idx++)
    {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", arg_idx, argv[arg_idx]);
    }
    fprintf(stderr, "\n");
}

/**
 * cleanup_stale_output_files - Remove stale completion files from previous runs
 * @dir: Directory path to clean
 *
 * Deletes any existing result and log files before a new clustering session
 * begins to prevent aborted runs from leaving behind mismatched results.
 */
static void cleanup_stale_output_files(
    const char *dir)
{
    if (dir == NULL)
    {
        return;
    }

    static const char *const files[] = {
        "anchors.bin",
        "anchors.txt",
        "anchors.fits",
        "cluster_counts.bin",
        "cluster_counts.txt",
        "cluster_radii.bin",
        "cluster_radii.txt",
        "frame_membership.bin",
        "frame_membership.txt",
        "frame_evals.txt",
        "dcc.bin",
        "dcc.txt",
        "cluster_run.log"
    };

    char path[1024];
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++)
    {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
        unlink(path);
    }
}


int main(int argc, char *argv[])
{
    init_colors_io();
    init_colors_help();
    struct timespec prog_start;
    clock_gettime(CLOCK_REALTIME, &prog_start);

    ClusterConfig config;
    GricProfile dataset_prof;
    memset(&dataset_prof, 0, sizeof(GricProfile));
    char *cmdline = NULL;
    int parse_rc = cluster_cli_parse(argc, argv, &config, &dataset_prof, &cmdline);
    if (parse_rc != 0)
    {
        return (parse_rc == 2) ? 0 : 1;
    }

    set_frameread_precision(config.algo.use_double);
    if (init_frameread(config.input.fits_filename,
                       config.input.stream_input_mode,
                       config.input.cnt2sync_mode,
                       config.input.filelist_mode) != 0)
    {
        if (cmdline)
            free(cmdline);
        print_args_on_error(argc, argv);
        return 1;
    }

    // Determine output directory
    char *out_dir = NULL;
    int out_dir_alloc = 0; // Flag to track if out_dir was malloced locally
    if (config.output.user_outdir)
    {
        out_dir = strdup(config.output.user_outdir);
        out_dir_alloc = 1;
    }
    else
    {
        out_dir = create_output_dir_name(config.input.fits_filename);
        out_dir_alloc = 1;
    }

    if (!out_dir)
    {
        perror("Memory allocation failed for output directory name");
        if (cmdline)
            free(cmdline);
        return 1;
    }

    struct stat st = {0};
    if (stat(out_dir, &st) == -1)
    {
        if (mkdir(out_dir, 0777) != 0)
        {
            perror("Failed to create output directory");
            free(out_dir);
            if (cmdline)
                free(cmdline);
            return 1;
        }
    }

    if (!config.output.user_outdir)
    {
        config.output.user_outdir = out_dir;
    }
    else
    {
        free(out_dir);
        out_dir = NULL;
    }

    if (config.output.user_outdir)
    {
        cleanup_stale_output_files(config.output.user_outdir);
    }

    ClusterState state;
    memset(&state, 0, sizeof(ClusterState));

    if (dataset_prof.perm_dim != NULL)
    {
        state.perm_dim = dataset_prof.perm_dim;
        state.residual_tail = dataset_prof.residual_tail;
        dataset_prof.perm_dim = NULL;
        dataset_prof.residual_tail = NULL;
    }

    if (config.output.distall_mode)
    {
        char out_path[1024];
        if (config.output.user_outdir)
            snprintf(out_path, sizeof(out_path), "%s/distall.txt", config.output.user_outdir);
        else
        {
            char *tmp = create_output_dir_name(config.input.fits_filename);
            snprintf(out_path, sizeof(out_path), "%s/distall.txt", tmp);
            free(tmp);
        }
        state.distall_out = fopen(out_path, "w");
        if (!state.distall_out)
        {
            perror("Failed to open distall.txt in output directory");
            if (cmdline)
                free(cmdline);
            return 1;
        }
        // ... (header printing)
    }

    if (!config.input.scandist_mode)
    {
        signal(SIGINT, handle_sigint);
        printf("CTRL+C to stop clustering and write results\n");
    }

    if (config.input.scandist_mode || config.algo.auto_rlim_mode)
    {
        run_scandist(&config, config.output.user_outdir);
        if (config.input.scandist_mode)
        {
            if (state.distall_out)
                fclose(state.distall_out);
            close_frameread();
            if (config.output.user_outdir && out_dir_alloc)
                free(config.output.user_outdir);
            if (cmdline)
                free(cmdline);
            return 0;
        }
        reset_frameread();
    }

    // Allocate State
    size_t max_clusters = (size_t)config.algo.maxnbclust;
    size_t cluster_pairs = max_clusters * max_clusters;
    /* Number of 64-bit words in the consistency bitmask per cluster pair */
    int words = (config.algo.maxnbclust + 63) / 64;
    size_t consistency_words = cluster_pairs * (size_t)words;

    state.clusters = (Cluster *)malloc(max_clusters * sizeof(Cluster));
    state.scratch.dcc_min = (double *)malloc(cluster_pairs * sizeof(double));
    state.scratch.dcc_max = (double *)malloc(cluster_pairs * sizeof(double));
    state.scratch.dcc_measured = (char *)malloc(cluster_pairs * sizeof(char));

    if (config.optim.sparse_dcc_mode)
    {
        for (size_t r = 0; r < max_clusters; r++)
        {
            for (size_t c = 0; c < max_clusters; c++)
            {
                size_t idx = r * max_clusters + c;
                if (r == c)
                {
                    state.scratch.dcc_min[idx] = 0.0;
                    state.scratch.dcc_max[idx] = 0.0;
                    state.scratch.dcc_measured[idx] = 1;
                }
                else
                {
                    state.scratch.dcc_min[idx] = 0.0;
                    state.scratch.dcc_max[idx] = 1e19;
                    state.scratch.dcc_measured[idx] = 0;
                }
            }
        }
    }
    else
    {
        for (size_t ii = 0; ii < cluster_pairs; ii++)
        {
            state.scratch.dcc_min[ii] = -1.0;
            state.scratch.dcc_max[ii] = -1.0;
            state.scratch.dcc_measured[ii] = 0;
        }
    }

    state.scratch.current_gprobs = (double *)malloc(max_clusters * sizeof(double));
    state.cluster_visitors = (VisitorList *)calloc(max_clusters, sizeof(VisitorList));
    state.scratch.probsortedclindex = (int *)malloc(max_clusters * sizeof(int));
    state.scratch.clmembflag = (int *)malloc(max_clusters * sizeof(int));
    state.scratch.consistency_mask =
        (config.optim.gprob_mode || config.optim.entropy_mode)
            ? (uint64_t *)calloc(consistency_words, sizeof(uint64_t))
            : NULL;
    state.scratch.entropy_p_current = (double *)malloc(max_clusters * sizeof(double));
    state.scratch.entropy_candidates = (Candidate *)malloc(max_clusters * sizeof(Candidate));
    state.scratch.entropy_prob_scores = (TargetScore *)malloc(max_clusters * sizeof(TargetScore));
    state.scratch.entropy_prune_scores = (TargetScore *)malloc(max_clusters * sizeof(TargetScore));
    state.scratch.entropy_active_indices = (int *)malloc(max_clusters * sizeof(int));
    state.scratch.entropy_plog2p = (double *)malloc(max_clusters * sizeof(double));
    state.scratch.entropy_visited = (uint8_t *)malloc(max_clusters * sizeof(uint8_t));
    /* Scratch buffers for sparse DCC bound refinement scheduling */
    state.scratch.refine_queue = (Candidate *)malloc(1024 * sizeof(Candidate));
    state.scratch.refine_queue_size = 0;
    state.scratch.refine_queue_idx = 0;
    state.scratch.refine_queue_capacity = 1024;
    state.scratch.refine_queue_last_num_clusters = 0;
    state.scratch.tuple_pred_candidates = (int *)malloc(max_clusters * sizeof(int));
    state.scratch.tuple_pred_count = 0;
    size_t pred_cap = config.optim.pred_n > 0 ? (size_t)config.optim.pred_n : 16;
    if (pred_cap < max_clusters)
    {
        pred_cap = max_clusters;
    }
    state.scratch.pred_candidates = (int *)malloc(pred_cap * sizeof(int));
    state.scratch.local_candidates = (int *)malloc(pred_cap * sizeof(int));
    state.scratch.sq16_cand_indices = (int *)malloc(max_clusters * sizeof(int));
    state.scratch.sq16_anchor_ptrs =
        (const int16_t **)malloc(max_clusters * sizeof(const int16_t *));
    state.scratch.d_min_scratch = (double *)malloc(max_clusters * sizeof(double));
    state.scratch.d_max_scratch = (double *)malloc(max_clusters * sizeof(double));

    // Run Clustering
    if (gric_shm_init(&config, &state) != 0)
    {
        fprintf(stderr, "Warning: Failed to initialize shared memory status tracking.\n");
    }

    struct timespec clust_start, clust_end;
    clock_gettime(CLOCK_MONOTONIC, &clust_start);
    run_clustering(&config, &state);
    clock_gettime(CLOCK_MONOTONIC, &clust_end);
    double clust_ms = (clust_end.tv_sec - clust_start.tv_sec) * 1000.0 +
                      (clust_end.tv_nsec - clust_start.tv_nsec) / 1000000.0;

    int final_status = stop_requested ? GRIC_STATUS_ABORTED : GRIC_STATUS_SUCCESS;
    gric_shm_update(&state, final_status, clust_ms);

    if (state.distall_out)
        fclose(state.distall_out);

    // Write Results
    struct timespec out_start, out_end;
    clock_gettime(CLOCK_MONOTONIC, &out_start);
    write_results(&config, &state);
    clock_gettime(CLOCK_MONOTONIC, &out_end);
    double out_ms = (out_end.tv_sec - out_start.tv_sec) * 1000.0 +
                    (out_end.tv_nsec - out_start.tv_nsec) / 1000000.0;

    struct rusage usage;
    long max_rss = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        max_rss = usage.ru_maxrss;
    }

    write_run_log(&config, &state, cmdline, prog_start, clust_ms, out_ms, max_rss);
    if (cmdline)
        free(cmdline);

    // Cleanup
    for (int cl_idx = 0; cl_idx < state.num_clusters; cl_idx++)
    {
        if (state.clusters[cl_idx].anchor.data)
            free(state.clusters[cl_idx].anchor.data);
    }
    free(state.clusters);

    for (long frame_idx = 0; frame_idx < state.telemetry.total_frames_processed; frame_idx++)
    {
        if (state.frame_infos[frame_idx].cluster_indices)
            free(state.frame_infos[frame_idx].cluster_indices);
        if (state.frame_infos[frame_idx].distances)
            free(state.frame_infos[frame_idx].distances);
    }
    free(state.frame_infos);

    for (int cl_idx = 0; cl_idx < config.algo.maxnbclust; cl_idx++)
    {
        if (state.cluster_visitors[cl_idx].frames)
            free(state.cluster_visitors[cl_idx].frames);
    }
    free(state.cluster_visitors);
    free(state.scratch.current_gprobs);

    free(state.scratch.dcc_min);
    free(state.scratch.dcc_max);
    free(state.scratch.dcc_measured);
    free(state.scratch.probsortedclindex);
    free(state.scratch.clmembflag);
    if (state.scratch.consistency_mask)
        free(state.scratch.consistency_mask);
    free(state.scratch.entropy_p_current);
    free(state.scratch.entropy_candidates);
    free(state.scratch.entropy_prob_scores);
    free(state.scratch.entropy_prune_scores);
    free(state.scratch.entropy_active_indices);
    free(state.scratch.entropy_plog2p);
    free(state.scratch.entropy_visited);
    free(state.scratch.refine_queue);
    free(state.scratch.tuple_pred_candidates);
    if (state.scratch.pred_candidates)
    {
        free(state.scratch.pred_candidates);
    }
    if (state.scratch.local_candidates)
    {
        free(state.scratch.local_candidates);
    }
    free(state.scratch.sq16_cand_indices);
    free((void *)state.scratch.sq16_anchor_ptrs);
    free(state.scratch.d_min_scratch);
    free(state.scratch.d_max_scratch);
    free(state.assignments);

    if (state.telemetry.pruned_fraction_sum)
        free(state.telemetry.pruned_fraction_sum);
    if (state.telemetry.step_counts)
        free(state.telemetry.step_counts);
    if (state.transition_matrix)
        free(state.transition_matrix);
    if (state.scratch.mixed_probs)
        free(state.scratch.mixed_probs);
    if (state.telemetry.dist_counts)
        free(state.telemetry.dist_counts);
    if (state.telemetry.pruned_counts_by_dist)
        free(state.telemetry.pruned_counts_by_dist);
    if (state.telemetry.cluster_query_counts)
        free(state.telemetry.cluster_query_counts);

    if (config.output.user_outdir && out_dir_alloc)
        free(config.output.user_outdir);

    gric_shm_cleanup(&state);
    if (config.output.shm_filename)
        free(config.output.shm_filename);

    if (state.perm_dim != NULL)
        free(state.perm_dim);
    if (state.residual_tail != NULL)
        free(state.residual_tail);
    gric_profile_free(&dataset_prof);

    close_frameread();

    return 0;
}
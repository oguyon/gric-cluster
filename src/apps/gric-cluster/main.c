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
#include "cluster_dcc.h"
#include "cluster_cli.h"
#include "cluster_help.h"
#include "cluster_io.h"
#include "cluster_scandist.h"
#include "config_utils.h"
#include "cluster_shm.h"
#include "frameread.h"
#include "frame_info_arena.h"
#include "gric_mem.h"
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
 * cleanup_stale_output_files() - Remove stale completion files from previous runs.
 * @dir: Directory path to clean.
 *
 * Deletes existing result and log files before a new clustering session begins to prevent
 * aborted runs from leaving behind mismatched results.
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
    for (size_t ii = 0; ii < sizeof(files) / sizeof(files[0]); ii++)
    {
        snprintf(path, sizeof(path), "%s/%s", dir, files[ii]);
        unlink(path);
    }
}

/**
 * cluster_setup_output_dir() - Prepare and create the output directory.
 * @config:        Clustering configuration structure.
 * @out_dir_alloc: Output flag set to 1 if output dir was allocated locally.
 *
 * Return: 0 on success, non-zero on error.
 */
static int cluster_setup_output_dir(
    ClusterConfig *config,
    int           *out_dir_alloc)
{
    char *out_dir = NULL;
    *out_dir_alloc = 0;

    if (config->output.user_outdir != NULL)
    {
        out_dir = strdup(config->output.user_outdir);
        *out_dir_alloc = 1;
    }
    else
    {
        out_dir = create_output_dir_name(config->input.fits_filename);
        *out_dir_alloc = 1;
    }

    if (out_dir == NULL)
    {
        perror("Memory allocation failed for output directory name");
        return -1;
    }

    struct stat st = {0};
    if (stat(out_dir, &st) == -1)
    {
        if (mkdir(out_dir, 0777) != 0)
        {
            perror("Failed to create output directory");
            free(out_dir);
            return -1;
        }
    }

    if (config->output.user_outdir == NULL)
    {
        config->output.user_outdir = out_dir;
    }
    else
    {
        free(out_dir);
    }

    if (config->output.user_outdir != NULL)
    {
        cleanup_stale_output_files(config->output.user_outdir);
    }

    return 0;
}

/**
 * cluster_state_alloc_scratch() - Allocate scratch memory buffers for state.
 * @state:        Clustering runtime state.
 * @config:       Clustering configuration.
 * @max_clusters: Maximum number of clusters.
 */
static void cluster_state_alloc_scratch(
    ClusterState        *state,
    const ClusterConfig *config,
    size_t               max_clusters)
{
    size_t cluster_pairs = max_clusters * max_clusters;
    int words = (config->algo.maxnbclust + 63) / 64;
    size_t consistency_words = cluster_pairs * (size_t)words;

    state->scratch.current_gprobs = (double *)malloc(max_clusters * sizeof(double));
    state->scratch.probsortedclindex = (int *)malloc(max_clusters * sizeof(int));

    if (posix_memalign((void **)&state->scratch.cluster_probs, 64,
                       max_clusters * sizeof(double)) != 0)
    {
        state->scratch.cluster_probs = NULL;
    }
    else
    {
        memset(state->scratch.cluster_probs, 0, max_clusters * sizeof(double));
    }

    if (config->optim.pred_mode)
    {
        if (posix_memalign((void **)&state->scratch.pred_probs, 64,
                           max_clusters * sizeof(double)) != 0)
        {
            state->scratch.pred_probs = NULL;
        }
        else
        {
            memset(state->scratch.pred_probs, 0, max_clusters * sizeof(double));
        }
    }
    else
    {
        state->scratch.pred_probs = NULL;
    }

    state->scratch.clmembflag = (int *)malloc(max_clusters * sizeof(int));
    state->scratch.active_clusters = (int *)malloc(max_clusters * sizeof(int));
    state->scratch.num_active_clusters = 0;
    state->scratch.consistency_mask =
        (config->optim.gprob_mode || config->optim.entropy_mode)
            ? (uint64_t *)calloc(consistency_words, sizeof(uint64_t))
            : NULL;

    state->scratch.entropy_p_current = (double *)malloc(max_clusters * sizeof(double));
    state->scratch.entropy_candidates = (Candidate *)malloc(max_clusters * sizeof(Candidate));
    state->scratch.entropy_prob_scores = (TargetScore *)malloc(max_clusters * sizeof(TargetScore));
    state->scratch.entropy_prune_scores =
        (TargetScore *)malloc(max_clusters * sizeof(TargetScore));
    state->scratch.entropy_active_indices = (int *)malloc(max_clusters * sizeof(int));
    state->scratch.entropy_plog2p = (double *)malloc(max_clusters * sizeof(double));
    state->scratch.entropy_visited = (uint8_t *)malloc(max_clusters * sizeof(uint8_t));

    state->scratch.refine_queue = (Candidate *)malloc(1024 * sizeof(Candidate));
    state->scratch.refine_queue_size = 0;
    state->scratch.refine_queue_idx = 0;
    state->scratch.refine_queue_capacity = 1024;
    state->scratch.refine_queue_last_num_clusters = 0;
    state->scratch.tuple_pred_candidates = (int *)malloc(max_clusters * sizeof(int));
    state->scratch.tuple_pred_count = 0;

    size_t pred_cap = config->optim.pred_n > 0 ? (size_t)config->optim.pred_n : 16;
    if (pred_cap < max_clusters)
    {
        pred_cap = max_clusters;
    }
    state->scratch.pred_candidates = (int *)malloc(pred_cap * sizeof(int));
    state->scratch.local_candidates = (int *)malloc(pred_cap * sizeof(int));
    state->scratch.sq16_cand_indices = (int *)malloc(max_clusters * sizeof(int));
    state->scratch.sq16_anchor_ptrs =
        (const int16_t **)malloc(max_clusters * sizeof(const int16_t *));
    state->scratch.d_min_scratch = (double *)malloc(max_clusters * sizeof(double));
    state->scratch.d_max_scratch = (double *)malloc(max_clusters * sizeof(double));
}

/**
 * cluster_state_alloc() - Allocate cluster arrays and matrices for runtime state.
 * @state:        Clustering runtime state.
 * @config:       Clustering configuration.
 * @dataset_prof: Dataset profile containing optional permutation arrays.
 *
 * Return: 0 on success, non-zero on error.
 */
static int cluster_state_alloc(
    ClusterState        *state,
    const ClusterConfig *config,
    GricProfile         *dataset_prof)
{
    memset(state, 0, sizeof(ClusterState));

    if (config->optim.sq16_calibrated)
    {
        state->sq16_calibrated = 1;
    }
    if (config->optim.sq8_calibrated)
    {
        state->sq8_calibrated = 1;
    }

    if (dataset_prof->perm_dim != NULL)
    {
        state->perm_dim = dataset_prof->perm_dim;
        state->residual_tail = dataset_prof->residual_tail;
        dataset_prof->perm_dim = NULL;
        dataset_prof->residual_tail = NULL;
    }

    if (config->output.distall_mode)
    {
        char out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s/distall.txt", config->output.user_outdir);
        state->distall_out = fopen(out_path, "w");
        if (state->distall_out == NULL)
        {
            perror("Failed to open distall.txt in output directory");
            return -1;
        }
    }

    size_t max_clusters = (size_t)config->algo.maxnbclust;
    state->clusters = (Cluster *)malloc(max_clusters * sizeof(Cluster));
    int use_sq16 = config->optim.use_sq16 || config->output.dcc_sq16_output;
    dcc_init_matrix(state, max_clusters, config->optim.sparse_dcc_mode, use_sq16);
    state->scratch.dcc_sq16_scale = 16384.0 / config->algo.rlim;
    state->cluster_visitors = (VisitorList *)calloc(max_clusters, sizeof(VisitorList));

    cluster_state_alloc_scratch(state, config, max_clusters);
    return 0;
}

/**
 * cluster_execute() - Execute clustering and record execution time.
 * @state:    Clustering runtime state.
 * @config:   Clustering configuration.
 * @clust_ms: Pointer to store elapsed clustering milliseconds.
 */
static void cluster_execute(
    ClusterState  *state,
    ClusterConfig *config,
    double        *clust_ms)
{
    if (gric_shm_init(config, state) != 0)
    {
        fprintf(stderr, "Warning: Failed to initialize shared memory status tracking.\n");
    }

    struct timespec clust_start, clust_end;
    clock_gettime(CLOCK_MONOTONIC, &clust_start);
    run_clustering(config, state);
    clock_gettime(CLOCK_MONOTONIC, &clust_end);

    *clust_ms = (clust_end.tv_sec - clust_start.tv_sec) * 1000.0 +
                (clust_end.tv_nsec - clust_start.tv_nsec) / 1000000.0;

    int final_status = stop_requested ? GRIC_STATUS_ABORTED : GRIC_STATUS_SUCCESS;
    gric_shm_update(state, final_status, *clust_ms);

    if (state->distall_out != NULL)
    {
        fclose(state->distall_out);
        state->distall_out = NULL;
    }
}

/**
 * cluster_save_and_log() - Save clustering results and write execution log.
 * @state:      Clustering runtime state.
 * @config:     Clustering configuration.
 * @cmdline:    Command line string for provenance.
 * @prog_start: Timestamp when process started.
 * @clust_ms:   Elapsed clustering milliseconds.
 */
static void cluster_save_and_log(
    ClusterState    *state,
    ClusterConfig   *config,
    const char      *cmdline,
    struct timespec  prog_start,
    double           clust_ms)
{
    struct timespec out_start, out_end;
    clock_gettime(CLOCK_MONOTONIC, &out_start);
    write_results(config, state);
    clock_gettime(CLOCK_MONOTONIC, &out_end);

    double out_ms = (out_end.tv_sec - out_start.tv_sec) * 1000.0 +
                    (out_end.tv_nsec - out_start.tv_nsec) / 1000000.0;

    struct rusage usage;
    long max_rss = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        max_rss = usage.ru_maxrss;
    }

    write_run_log(config, state, cmdline, prog_start, clust_ms, out_ms, max_rss);
}

/**
 * cluster_state_free() - Release all allocated cluster state buffers.
 * @state:         Clustering runtime state.
 * @config:        Clustering configuration.
 * @dataset_prof:  Dataset profile structure.
 * @out_dir_alloc: Flag indicating if output dir needs freeing.
 */
static void cluster_state_free(
    ClusterState  *state,
    ClusterConfig *config,
    GricProfile   *dataset_prof,
    int            out_dir_alloc)
{
    for (int cl_idx = 0; cl_idx < state->num_clusters; cl_idx++)
    {
        if (state->clusters[cl_idx].anchor.data != NULL)
        {
            free(state->clusters[cl_idx].anchor.data);
        }
    }
    free(state->clusters);

    frame_info_arena_destroy(&state->frame_info_arena);
    free(state->frame_infos);

    for (int cl_idx = 0; cl_idx < config->algo.maxnbclust; cl_idx++)
    {
        if (state->cluster_visitors[cl_idx].records != NULL)
        {
            free(state->cluster_visitors[cl_idx].records);
        }
    }
    free(state->cluster_visitors);
    free(state->scratch.current_gprobs);

    dcc_free_matrix(state);
    free(state->scratch.probsortedclindex);
    if (state->scratch.cluster_probs != NULL)
    {
        free(state->scratch.cluster_probs);
    }
    if (state->scratch.pred_probs != NULL)
    {
        free(state->scratch.pred_probs);
    }
    free(state->scratch.clmembflag);
    free(state->scratch.active_clusters);
    if (state->scratch.consistency_mask != NULL)
    {
        free(state->scratch.consistency_mask);
    }
    free(state->scratch.entropy_p_current);
    free(state->scratch.entropy_candidates);
    free(state->scratch.entropy_prob_scores);
    free(state->scratch.entropy_prune_scores);
    free(state->scratch.entropy_active_indices);
    free(state->scratch.entropy_plog2p);
    free(state->scratch.entropy_visited);
    free(state->scratch.refine_queue);
    free(state->scratch.tuple_pred_candidates);
    if (state->scratch.pred_candidates != NULL)
    {
        free(state->scratch.pred_candidates);
    }
    if (state->scratch.local_candidates != NULL)
    {
        free(state->scratch.local_candidates);
    }
    free(state->scratch.sq16_cand_indices);
    free((void *)state->scratch.sq16_anchor_ptrs);
    free(state->scratch.d_min_scratch);
    free(state->scratch.d_max_scratch);
    free(state->assignments);
    if (state->assignment_dists != NULL)
    {
        free(state->assignment_dists);
    }

    if (state->telemetry.pruned_fraction_sum != NULL)
    {
        free(state->telemetry.pruned_fraction_sum);
    }
    if (state->telemetry.step_counts != NULL)
    {
        free(state->telemetry.step_counts);
    }
    if (state->transition_matrix != NULL)
    {
        free(state->transition_matrix);
    }
    if (state->scratch.mixed_probs != NULL)
    {
        free(state->scratch.mixed_probs);
    }
    if (state->telemetry.dist_counts != NULL)
    {
        free(state->telemetry.dist_counts);
    }
    if (state->telemetry.pruned_counts_by_dist != NULL)
    {
        free(state->telemetry.pruned_counts_by_dist);
    }
    if (state->telemetry.cluster_query_counts != NULL)
    {
        free(state->telemetry.cluster_query_counts);
    }

    if (config->output.user_outdir != NULL && out_dir_alloc)
    {
        free(config->output.user_outdir);
    }

    gric_shm_cleanup(state);
    if (config->output.shm_filename != NULL)
    {
        free(config->output.shm_filename);
    }

    if (state->perm_dim != NULL)
    {
        free(state->perm_dim);
    }
    if (state->residual_tail != NULL)
    {
        free(state->residual_tail);
    }
    gric_profile_free(dataset_prof);
    close_frameread();
}

/**
 * cluster_run_scandist_if_requested() - Execute distance calibration scan if requested.
 * @config:        Clustering configuration.
 * @out_dir_alloc: Flag indicating if output dir was allocated.
 * @cmdline:       Command line string.
 * @completed:     Output flag set to 1 if scandist-only mode finished early.
 *
 * Return: Process return code (0 for normal exit).
 */
static int cluster_run_scandist_if_requested(
    ClusterConfig *config,
    int            out_dir_alloc,
    char          *cmdline,
    int           *completed)
{
    *completed = 0;

    if (!config->input.scandist_mode)
    {
        signal(SIGINT, handle_sigint);
        printf("CTRL+C to stop clustering and write results\n");
    }

    if (config->input.scandist_mode || config->algo.auto_rlim_mode)
    {
        run_scandist(config, config->output.user_outdir);
        if (config->input.scandist_mode)
        {
            close_frameread();
            if (config->output.user_outdir != NULL && out_dir_alloc)
            {
                free(config->output.user_outdir);
            }
            if (cmdline != NULL)
            {
                free(cmdline);
            }
            *completed = 1;
            return 0;
        }
        reset_frameread();
    }

    return 0;
}

/**
 * cluster_run_session() - Orchestrate output dir, allocation, clustering, and saving.
 * @config:        Clustering configuration.
 * @dataset_prof:  Dataset profile structure.
 * @cmdline:       Command line string.
 * @prog_start:    Start timestamp.
 *
 * Return: Process exit code.
 */
static int cluster_run_session(
    ClusterConfig   *config,
    GricProfile     *dataset_prof,
    char            *cmdline,
    struct timespec  prog_start)
{
    int out_dir_alloc = 0;
    if (cluster_setup_output_dir(config, &out_dir_alloc) != 0)
    {
        return 1;
    }

    int scandist_completed = 0;
    int rc = cluster_run_scandist_if_requested(config, out_dir_alloc, cmdline,
                                               &scandist_completed);
    if (scandist_completed)
    {
        return rc;
    }

    ClusterState state;
    if (cluster_state_alloc(&state, config, dataset_prof) != 0)
    {
        return 1;
    }

    double clust_ms = 0.0;
    cluster_execute(&state, config, &clust_ms);
    cluster_save_and_log(&state, config, cmdline, prog_start, clust_ms);
    cluster_state_free(&state, config, dataset_prof, out_dir_alloc);

    return 0;
}

int main(int argc, char *argv[])
{
    cli_colors_init();
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
        if (cmdline != NULL)
        {
            free(cmdline);
        }
        print_args_on_error(argc, argv);
        return 1;
    }

    int rc = cluster_run_session(&config, &dataset_prof, cmdline, prog_start);
    if (cmdline != NULL)
    {
        free(cmdline);
    }
    return rc;
}
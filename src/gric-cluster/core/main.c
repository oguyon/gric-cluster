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
#include "cluster_help.h"
#include "cluster_io.h"
#include "cluster_scandist.h"
#include "config_utils.h"
#include "frameread.h"
#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

volatile sig_atomic_t stop_requested = 0;

void handle_sigint(int sig)
{
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

int main(int argc, char *argv[])
{
    init_colors_io();
    init_colors_help();
    struct timespec prog_start;
    clock_gettime(CLOCK_REALTIME, &prog_start);

    char *cmdline = (char *)malloc(8192);
    if (cmdline)
    {
        size_t cmdline_pos = 0;
        size_t cmdline_remaining = 8192;
        for (int arg_idx = 0; arg_idx < argc; arg_idx++)
        {
            int written = snprintf(cmdline + cmdline_pos, cmdline_remaining,
                                   "%s%s", argv[arg_idx],
                                   (arg_idx < argc - 1) ? " " : "");
            if (written < 0 || (size_t)written >= cmdline_remaining)
            {
                fprintf(stderr, "Warning: command line string truncated\n");
                break;
            }
            cmdline_pos += (size_t)written;
            cmdline_remaining -= (size_t)written;
        }
    }

    // Check for help option early
    for (int arg_idx = 1; arg_idx < argc; arg_idx++)
    {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                print_help_keyword(argv[arg_idx + 1]);
            }
            else
            {
                print_help(argv[0]);
            }
            if (cmdline)
                free(cmdline);
            return 0;
        }
    }

    if (argc < 2)
    {
        print_usage(argv[0]);
        if (cmdline)
            free(cmdline);
        return 1;
    }

    ClusterConfig config;
    memset(&config, 0, sizeof(ClusterConfig));
    // Set defaults
    config.algo.deltaprob = 0.01;
    config.algo.maxnbclust = 1000;
    config.optim.ncpu = 1;
    config.input.maxnbfr = 100000;
    config.optim.fmatch_a = 2.0;
    config.optim.fmatch_b = 0.5;
    config.optim.max_gprob_visitors = 1000;
    config.output.progress_mode = 1;
    config.optim.pred_len = 10;
    config.optim.pred_h = 1000;
    config.optim.pred_n = 2;
    config.algo.maxcl_strategy = MAXCL_STOP;
    config.algo.discard_fraction = 0.5;
    config.optim.entropy_max_targets = 15;
    config.optim.entropy_min_prob = 0.001;

    // Output defaults (disabled by default, except membership and dcc)
    config.output.output_dcc = 1;
    config.output.output_tm = 0;
    config.output.output_anchors = 0;
    config.output.output_counts = 0;
    config.output.output_membership = 1;
    config.output.output_discarded = 0;
    config.output.output_clustered = 0;
    config.output.output_clusters = 0;

    int arg_idx = 1;
    int rlim_set = 0;
    const char *confw_filename = NULL;

    while (arg_idx < argc)
    {
        char *key = argv[arg_idx];
        char *val = (arg_idx + 1 < argc) ? argv[arg_idx + 1] : NULL;

        // Check for -conf and -confw explicitly or handle via apply_option?
        // apply_option handles config-specific logic mostly, but -conf is meta.

        if (strcmp(key, "-conf") == 0)
        {
            if (!val)
            {
                fprintf(stderr, "Error: -conf requires a filename\n");
                if (cmdline)
                    free(cmdline);
                return 1;
            }
            if (read_config_file(val, &config) != 0)
            {
                fprintf(stderr, "Error: Could not read config file %s\n", val);
                if (cmdline)
                    free(cmdline);
                return 1;
            }
            // If config loaded rlim, we can consider it set?
            // But we don't know for sure if it was set in config.
            // However, our smart positional logic below handles this.
            arg_idx += 2;
            continue;
        }

        if (strcmp(key, "-confw") == 0)
        {
            if (!val)
            {
                fprintf(stderr, "Error: -confw requires a filename\n");
                if (cmdline)
                    free(cmdline);
                return 1;
            }
            confw_filename = val;
            arg_idx += 2;
            continue;
        }

        int res = apply_option(&config, key, val);
        if (res >= 0)
        {
            arg_idx += (1 + res);
            // If explicit -rlim was used, mark it
            if (strcmp(key, "-rlim") == 0 || strcmp(key, "rlim") == 0)
                rlim_set = 1;
            continue;
        }

        // Positional or Unknown
        if (key[0] == '-' && !isdigit(key[1]))
        {
            fprintf(stderr, "Error: Unknown option: %s\n", key);
            print_usage(argv[0]);
            if (cmdline)
                free(cmdline);
            print_args_on_error(argc, argv);
            return 1;
        }

        // Positional logic
        if (!config.input.scandist_mode && !rlim_set)
        {
            // Try parse as rlim
            char *endptr;
            double v = strtod(key, &endptr);
            if (*endptr == '\0')
            {
                config.algo.rlim = v;
                rlim_set = 1;
            }
            else if (key[0] == 'a' && isdigit(key[1]))
            {
                config.algo.auto_rlim_factor = atof(key + 1);
                config.algo.auto_rlim_mode = 1;
                rlim_set = 1;
            }
            else
            {
                // Not a number, assume filename
                if (config.input.fits_filename != NULL)
                {
                    fprintf(stderr,
                            "Error: Too many arguments or multiple input files specified (already "
                            "have '%s', found '%s')\n",
                            config.input.fits_filename, key);
                    if (cmdline)
                        free(cmdline);
                    print_args_on_error(argc, argv);
                    return 1;
                }
                config.input.fits_filename = key;
            }
        }
        else
        {
            if (config.input.fits_filename != NULL)
            {
                fprintf(stderr,
                        "Error: Too many arguments or multiple input files specified (already have "
                        "'%s', found '%s')\n",
                        config.input.fits_filename, key);
                if (cmdline)
                    free(cmdline);
                print_args_on_error(argc, argv);
                return 1;
            }
            config.input.fits_filename = key;
        }
        arg_idx++;
    }

    if (confw_filename)
    {
        if (write_config_file(confw_filename, &config) != 0)
        {
            fprintf(stderr, "Error: Could not write config file %s\n", confw_filename);
            if (cmdline)
                free(cmdline);
            return 1;
        }
        printf("Configuration written to %s\n", confw_filename);
    }

    if (!config.input.fits_filename)
    {
        fprintf(stderr, "Error: Missing input file or stream name.\n");
        if (!config.input.scandist_mode)
            print_usage(argv[0]);
        if (cmdline)
            free(cmdline);
        print_args_on_error(argc, argv);
        return 1;
    }

    if (init_frameread(config.input.fits_filename, config.input.stream_input_mode, config.input.cnt2sync_mode,
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

    ClusterState state;
    memset(&state, 0, sizeof(ClusterState));

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
    state.clusters = (Cluster *)malloc(config.algo.maxnbclust * sizeof(Cluster));
    state.scratch.dccarray = (double *)malloc(config.algo.maxnbclust * config.algo.maxnbclust * sizeof(double));
    for (int ii = 0; ii < config.algo.maxnbclust * config.algo.maxnbclust; ii++)
        state.scratch.dccarray[ii] = -1.0;

    state.scratch.current_gprobs = (double *)malloc(config.algo.maxnbclust * sizeof(double));
    state.cluster_visitors = (VisitorList *)calloc(config.algo.maxnbclust, sizeof(VisitorList));
    state.scratch.probsortedclindex = (int *)malloc(config.algo.maxnbclust * sizeof(int));
    state.scratch.clmembflag = (int *)malloc(config.algo.maxnbclust * sizeof(int));
    int words = (config.algo.maxnbclust + 63) / 64;
    state.scratch.consistency_mask = (uint64_t *)calloc(config.algo.maxnbclust * config.algo.maxnbclust * words, sizeof(uint64_t));
    state.scratch.entropy_p_current = (double *)malloc(config.algo.maxnbclust * sizeof(double));
    state.scratch.entropy_candidates = (Candidate *)malloc(config.algo.maxnbclust * sizeof(Candidate));

    // Run Clustering
    struct timespec clust_start, clust_end;
    clock_gettime(CLOCK_MONOTONIC, &clust_start);
    run_clustering(&config, &state);
    clock_gettime(CLOCK_MONOTONIC, &clust_end);
    double clust_ms = (clust_end.tv_sec - clust_start.tv_sec) * 1000.0 +
                      (clust_end.tv_nsec - clust_start.tv_nsec) / 1000000.0;

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

    free(state.scratch.dccarray);
    free(state.scratch.probsortedclindex);
    free(state.scratch.clmembflag);
    free(state.scratch.consistency_mask);
    free(state.scratch.entropy_p_current);
    free(state.scratch.entropy_candidates);
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

    close_frameread();

    return 0;
}
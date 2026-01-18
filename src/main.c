#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <ctype.h>
#include "cluster_defs.h"
#include "cluster_core.h"
#include "cluster_io.h"
#include "frameread.h"
#include "config_utils.h"

volatile sig_atomic_t stop_requested = 0;

void handle_sigint(int sig) {
    stop_requested = 1;
}

void print_args_on_error(int argc, char *argv[]) {
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++) {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
}

int main(int argc, char *argv[]) {
    struct timespec prog_start;
    clock_gettime(CLOCK_REALTIME, &prog_start);

    char *cmdline = (char *)malloc(8192);
    if (cmdline) {
        cmdline[0] = '\0';
        for (int i = 0; i < argc; i++) {
            strcat(cmdline, argv[i]);
            if (i < argc - 1) strcat(cmdline, " ");
        }
    }

    // Check for help option early
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if (i + 1 < argc) {
                print_help_keyword(argv[i+1]);
            } else {
                print_help(argv[0]);
            }
            if (cmdline) free(cmdline);
            return 0;
        }
    }

    if (argc < 2) {
        print_usage(argv[0]);
        if (cmdline) free(cmdline);
        return 1;
    }

    ClusterConfig config;
    memset(&config, 0, sizeof(ClusterConfig));
    // Set defaults
    config.deltaprob = 0.01;
    config.maxnbclust = 1000;
    config.ncpu = 1;
    config.maxnbfr = 100000;
    config.fmatch_a = 2.0;
    config.fmatch_b = 0.5;
    config.max_gprob_visitors = 1000;
    config.progress_mode = 1;
    config.pred_len = 10;
    config.pred_h = 1000;
    config.pred_n = 2;
    config.maxcl_strategy = MAXCL_STOP;
    config.discard_fraction = 0.5;

    // Output defaults (disabled by default, except membership and dcc)
    config.output_dcc = 1;
    config.output_tm = 0;
    config.output_anchors = 0;
    config.output_counts = 0;
    config.output_membership = 1;
    config.output_discarded = 0;
    config.output_clustered = 0;
    config.output_clusters = 0;

    int arg_idx = 1;
    int rlim_set = 0;
    const char *confw_filename = NULL;

    while (arg_idx < argc) {
        char *key = argv[arg_idx];
        char *val = (arg_idx + 1 < argc) ? argv[arg_idx + 1] : NULL;

        // Check for -conf and -confw explicitly or handle via apply_option?
        // apply_option handles config-specific logic mostly, but -conf is meta.
        
        if (strcmp(key, "-conf") == 0) {
            if (!val) {
                fprintf(stderr, "Error: -conf requires a filename\n");
                if (cmdline) free(cmdline);
                return 1;
            }
            if (read_config_file(val, &config) != 0) {
                fprintf(stderr, "Error: Could not read config file %s\n", val);
                if (cmdline) free(cmdline);
                return 1;
            }
            // If config loaded rlim, we can consider it set?
            // But we don't know for sure if it was set in config.
            // However, our smart positional logic below handles this.
            arg_idx += 2;
            continue;
        }

        if (strcmp(key, "-confw") == 0) {
            if (!val) {
                fprintf(stderr, "Error: -confw requires a filename\n");
                if (cmdline) free(cmdline);
                return 1;
            }
            confw_filename = val;
            arg_idx += 2;
            continue;
        }

        int res = apply_option(&config, key, val);
        if (res >= 0) {
            arg_idx += (1 + res);
            // If explicit -rlim was used, mark it
            if (strcmp(key, "-rlim") == 0 || strcmp(key, "rlim") == 0) rlim_set = 1;
            continue;
        }

        // Positional or Unknown
        if (key[0] == '-' && !isdigit(key[1])) {
             fprintf(stderr, "Error: Unknown option: %s\n", key);
             print_usage(argv[0]);
             if (cmdline) free(cmdline);
             print_args_on_error(argc, argv);
             return 1;
        }

        // Positional logic
        if (!config.scandist_mode && !rlim_set) {
            // Try parse as rlim
            char *endptr;
            double v = strtod(key, &endptr);
            if (*endptr == '\0') {
                config.rlim = v;
                rlim_set = 1;
            } else if (key[0] == 'a' && isdigit(key[1])) {
                config.auto_rlim_factor = atof(key+1);
                config.auto_rlim_mode = 1;
                rlim_set = 1;
            } else {
                // Not a number, assume filename
                if (config.fits_filename != NULL) {
                    fprintf(stderr, "Error: Too many arguments or multiple input files specified (already have '%s', found '%s')\n", config.fits_filename, key);
                    if (cmdline) free(cmdline);
                    print_args_on_error(argc, argv);
                    return 1;
                }
                config.fits_filename = key;
            }
        } else {
            if (config.fits_filename != NULL) {
                fprintf(stderr, "Error: Too many arguments or multiple input files specified (already have '%s', found '%s')\n", config.fits_filename, key);
                if (cmdline) free(cmdline);
                print_args_on_error(argc, argv);
                return 1;
            }
            config.fits_filename = key;
        }
        arg_idx++;
    }

    if (confw_filename) {
        if (write_config_file(confw_filename, &config) != 0) {
             fprintf(stderr, "Error: Could not write config file %s\n", confw_filename);
             if (cmdline) free(cmdline);
             return 1;
        }
        printf("Configuration written to %s\n", confw_filename);
    }

    if (!config.fits_filename) {
        fprintf(stderr, "Error: Missing input file or stream name.\n");
        if (!config.scandist_mode) print_usage(argv[0]);
        if (cmdline) free(cmdline);
        print_args_on_error(argc, argv);
        return 1;
    }

    if (init_frameread(config.fits_filename, config.stream_input_mode, config.cnt2sync_mode) != 0) {
        if (cmdline) free(cmdline);
        print_args_on_error(argc, argv);
        return 1;
    }

    // Determine output directory
    char *out_dir = NULL;
    int out_dir_alloc = 0; // Flag to track if out_dir was malloced locally
    if (config.user_outdir) {
        out_dir = strdup(config.user_outdir);
        out_dir_alloc = 1;
    } else {
        out_dir = create_output_dir_name(config.fits_filename);
        out_dir_alloc = 1;
    }

    if (!out_dir) {
        perror("Memory allocation failed for output directory name");
        if (cmdline) free(cmdline);
        return 1;
    }

    struct stat st = {0};
    if (stat(out_dir, &st) == -1) {
        if (mkdir(out_dir, 0777) != 0) {
            perror("Failed to create output directory");
            free(out_dir);
            if (cmdline) free(cmdline);
            return 1;
        }
    }

    if (!config.user_outdir) {
        config.user_outdir = out_dir;
    } else {
        free(out_dir);
        out_dir = NULL;
        out_dir_alloc = 0;
    }

    ClusterState state;
    memset(&state, 0, sizeof(ClusterState));

    if (config.distall_mode) {
        char out_path[1024];
        if (config.user_outdir)
            snprintf(out_path, sizeof(out_path), "%s/distall.txt", config.user_outdir);
        else {
             char *tmp = create_output_dir_name(config.fits_filename);
             snprintf(out_path, sizeof(out_path), "%s/distall.txt", tmp);
             free(tmp);
        }
        state.distall_out = fopen(out_path, "w");
        if (!state.distall_out) {
            perror("Failed to open distall.txt in output directory");
            if (cmdline) free(cmdline);
            return 1;
        }
        // ... (header printing)
    }

    if (!config.scandist_mode) {
        signal(SIGINT, handle_sigint);
        printf("CTRL+C to stop clustering and write results\n");
    }

    if (config.scandist_mode || config.auto_rlim_mode) {
        run_scandist(&config, config.user_outdir);
        if (config.scandist_mode) {
             if (state.distall_out) fclose(state.distall_out);
             close_frameread();
             if (config.user_outdir && out_dir_alloc) free(config.user_outdir);
             if (cmdline) free(cmdline);
             return 0;
        }
        reset_frameread();
    }

    // Allocate State
    state.clusters = (Cluster *)malloc(config.maxnbclust * sizeof(Cluster));
    state.dccarray = (double *)malloc(config.maxnbclust * config.maxnbclust * sizeof(double));
    for (int i = 0; i < config.maxnbclust * config.maxnbclust; i++) state.dccarray[i] = -1.0;

    state.current_gprobs = (double *)malloc(config.maxnbclust * sizeof(double));
    state.cluster_visitors = (VisitorList *)calloc(config.maxnbclust, sizeof(VisitorList));
    state.probsortedclindex = (int *)malloc(config.maxnbclust * sizeof(int));
    state.clmembflag = (int *)malloc(config.maxnbclust * sizeof(int));

    // Run Clustering
    struct timespec clust_start, clust_end;
    clock_gettime(CLOCK_MONOTONIC, &clust_start);
    run_clustering(&config, &state);
    clock_gettime(CLOCK_MONOTONIC, &clust_end);
    double clust_ms = (clust_end.tv_sec - clust_start.tv_sec) * 1000.0 + (clust_end.tv_nsec - clust_start.tv_nsec) / 1000000.0;

    if (state.distall_out) fclose(state.distall_out);

    // Write Results
    struct timespec out_start, out_end;
    clock_gettime(CLOCK_MONOTONIC, &out_start);
    write_results(&config, &state);
    clock_gettime(CLOCK_MONOTONIC, &out_end);
    double out_ms = (out_end.tv_sec - out_start.tv_sec) * 1000.0 + (out_end.tv_nsec - out_start.tv_nsec) / 1000000.0;

    struct rusage usage;
    long max_rss = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        max_rss = usage.ru_maxrss;
    }

    write_run_log(&config, &state, cmdline, prog_start, clust_ms, out_ms, max_rss);
    if (cmdline) free(cmdline);

    // Cleanup
    for (int i = 0; i < state.num_clusters; i++) {
        if (state.clusters[i].anchor.data) free(state.clusters[i].anchor.data);
    }
    free(state.clusters);

    for (long i = 0; i < state.total_frames_processed; i++) {
        if (state.frame_infos[i].cluster_indices) free(state.frame_infos[i].cluster_indices);
        if (state.frame_infos[i].distances) free(state.frame_infos[i].distances);
    }
    free(state.frame_infos);

    for (int i = 0; i < config.maxnbclust; i++) {
        if (state.cluster_visitors[i].frames) free(state.cluster_visitors[i].frames);
    }
    free(state.cluster_visitors);
    free(state.current_gprobs);

    free(state.dccarray);
    free(state.probsortedclindex);
    free(state.clmembflag);
    free(state.assignments);

    if (state.pruned_fraction_sum) free(state.pruned_fraction_sum);
    if (state.step_counts) free(state.step_counts);
    if (state.transition_matrix) free(state.transition_matrix);
    if (state.mixed_probs) free(state.mixed_probs);
    if (state.dist_counts) free(state.dist_counts);
    if (state.pruned_counts_by_dist) free(state.pruned_counts_by_dist);

    if (config.user_outdir && out_dir_alloc) free(config.user_outdir);

    close_frameread();

    return 0;
}
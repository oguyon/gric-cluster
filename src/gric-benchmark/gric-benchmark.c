/**
 * @file gric-benchmark.c
 * @brief Main runner for the GRIC suite auto-benchmarking utility.
 *
 * Orchestrates clustering benchmarks by running tests against different parameters
 * and configurations, tracking execution timings and result metrics.
 *
 * Main Functions:
 * - init_config: Initializes benchmark configurations.
 * - main: Entry point of the benchmarking runner.
 */
#include "benchmark.h"
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/**
 * @brief Per-test result for the final summary table.
 */
typedef struct
{
    char   pattern[64];
    char   algo[32];
    char   time_ms[64];
    double dist_total;
    double dist_sample;
    double dist_inter;
    double avg_dist;
    int    clusters;
    char   mem_kb[64];
    int    nsamples;
} TestResult;

/**
 * @brief Initialize configuration with default values.
 *
 * @param config Pointer to the BenchmarkConfig struct.
 */
static void init_config(
    BenchmarkConfig *config)
{
    config->nsamples = 20000;
    strcpy(config->rlim, "0.10");
    config->rlim_set = 0;
    config->maxcl = 1000;
    config->maxim = 100000;
    config->maxim_set = 0;
    strcpy(config->type, "txt");
    config->reuse_mp4 = 0;
    config->pattern_count = 0;
    config->extra_options_count = 0;
    config->build_first = 0;
    config->use_entropy = 0;
} // init_config

int main(
    int   argc,
    char *argv[])
{
    BenchmarkConfig config;
    init_config(&config);
    init_colors();

    static struct option long_options[] =
    {
        {"help",     no_argument,       0, 'h'},
        {"nsamples", required_argument, 0, 'n'},
        {"rlim",     required_argument, 0, 'r'},
        {"pattern",  required_argument, 0, 'p'},
        {"file",     required_argument, 0, 'f'},
        {"type",     required_argument, 0, 't'},
        {"options",  required_argument, 0, 'o'},
        {"build",    no_argument,       0, 'b'},
        {"mp4r",     no_argument,       0, 1001},
        {"maxcl",    required_argument, 0, 1002},
        {"maxim",    required_argument, 0, 1003},
        {"entropy",  no_argument,       0, 1004},
        {0, 0, 0, 0}
    };

    char *test_list_file = NULL;
    int opt;
    int option_index = 0;
    while ((opt = getopt_long_only(argc, argv, "hn:r:p:f:t:o:b",
                                   long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_help(argv[0]);
                return 0;
            case 'n':
                config.nsamples = atoi(optarg);
                break;
            case 'r':
                strncpy(config.rlim, optarg, sizeof(config.rlim) - 1);
                config.rlim[sizeof(config.rlim) - 1] = '\0';
                config.rlim_set = 1;
                break;
            case 'p':
                if (config.pattern_count < MAX_PATTERNS)
                {
                    config.patterns[config.pattern_count++] = strdup(optarg);
                }
                break;
            case 'f':
                test_list_file = optarg;
                break;
            case 't':
                strncpy(config.type, optarg, sizeof(config.type) - 1);
                config.type[sizeof(config.type) - 1] = '\0';
                break;
            case 'o':
                if (config.extra_options_count < MAX_OPTIONS)
                {
                    /* Merge option flag and its value if passed separately */
                    if (optarg[0] == '-' && optind < argc && argv[optind][0] != '-')
                    {
                        size_t len = strlen(optarg) + 2 + strlen(argv[optind]);
                        char *merged = malloc(len);

                        if (merged != NULL)
                        {
                            snprintf(merged, len, "%s %s", optarg, argv[optind]);
                            config.extra_options[config.extra_options_count++] = merged;
                            optind++;
                        }
                        else
                        {
                            config.extra_options[config.extra_options_count++] = optarg;
                            if (config.extra_options_count < MAX_OPTIONS)
                            {
                                config.extra_options[config.extra_options_count++] = argv[optind];
                                optind++;
                            }
                        }
                    }
                    else
                    {
                        config.extra_options[config.extra_options_count++] = optarg;
                    }
                }
                break;
            case 'b':
                config.build_first = 1;
                break;
            case 1001: /* -mp4r */
                config.reuse_mp4 = 1;
                break;
            case 1002: /* -maxcl */
                config.maxcl = atoi(optarg);
                break;
            case 1003: /* -maxim */
                config.maxim = atoi(optarg);
                config.maxim_set = 1;
                break;
            case 1004: /* -entropy */
                config.use_entropy = 1;
                break;
            default:
                fprintf(stderr, "Error: Unknown option\n");
                print_help(argv[0]);
                return 1;
        }
    }

    /* Process and filter extra options to override native configuration */
    {
        for (int ii = 0; ii < config.extra_options_count; )
        {
            int processed = 0;

            if (strcmp(config.extra_options[ii], "-maxcl") == 0)
            {
                if (ii + 1 < config.extra_options_count && config.extra_options[ii + 1][0] != '-')
                {
                    config.maxcl = atoi(config.extra_options[ii + 1]);
                    /* Free if dynamically allocated */
                    {
                        int is_argv_1 = 0;
                        int is_argv_2 = 0;

                        for (int jj = 0; jj < argc; jj++)
                        {
                            if (config.extra_options[ii] == argv[jj])
                            {
                                is_argv_1 = 1;
                            }
                            if (config.extra_options[ii + 1] == argv[jj])
                            {
                                is_argv_2 = 1;
                            }
                        }

                        if (!is_argv_1)
                        {
                            free(config.extra_options[ii]);
                        }
                        if (!is_argv_2)
                        {
                            free(config.extra_options[ii + 1]);
                        }
                    }

                    /* Shift elements */
                    for (int jj = ii; jj < config.extra_options_count - 2; jj++)
                    {
                        config.extra_options[jj] = config.extra_options[jj + 2];
                    }
                    config.extra_options_count -= 2;
                    processed = 1;
                }
            }
            else if (strncmp(config.extra_options[ii], "-maxcl ", 7) == 0)
            {
                config.maxcl = atoi(config.extra_options[ii] + 7);
                /* Free if dynamically allocated */
                {
                    int is_argv = 0;

                    for (int jj = 0; jj < argc; jj++)
                    {
                        if (config.extra_options[ii] == argv[jj])
                        {
                            is_argv = 1;
                            break;
                        }
                    }

                    if (!is_argv)
                    {
                        free(config.extra_options[ii]);
                    }
                }

                /* Shift elements */
                for (int jj = ii; jj < config.extra_options_count - 1; jj++)
                {
                    config.extra_options[jj] = config.extra_options[jj + 1];
                }
                config.extra_options_count -= 1;
                processed = 1;
            }
            else if (strcmp(config.extra_options[ii], "-maxim") == 0)
            {
                if (ii + 1 < config.extra_options_count && config.extra_options[ii + 1][0] != '-')
                {
                    config.maxim = atoi(config.extra_options[ii + 1]);
                    config.maxim_set = 1;
                    /* Free if dynamically allocated */
                    {
                        int is_argv_1 = 0;
                        int is_argv_2 = 0;

                        for (int jj = 0; jj < argc; jj++)
                        {
                            if (config.extra_options[ii] == argv[jj])
                            {
                                is_argv_1 = 1;
                            }
                            if (config.extra_options[ii + 1] == argv[jj])
                            {
                                is_argv_2 = 1;
                            }
                        }

                        if (!is_argv_1)
                        {
                            free(config.extra_options[ii]);
                        }
                        if (!is_argv_2)
                        {
                            free(config.extra_options[ii + 1]);
                        }
                    }

                    /* Shift elements */
                    for (int jj = ii; jj < config.extra_options_count - 2; jj++)
                    {
                        config.extra_options[jj] = config.extra_options[jj + 2];
                    }
                    config.extra_options_count -= 2;
                    processed = 1;
                }
            }
            else if (strncmp(config.extra_options[ii], "-maxim ", 7) == 0)
            {
                config.maxim = atoi(config.extra_options[ii] + 7);
                config.maxim_set = 1;
                /* Free if dynamically allocated */
                {
                    int is_argv = 0;

                    for (int jj = 0; jj < argc; jj++)
                    {
                        if (config.extra_options[ii] == argv[jj])
                        {
                            is_argv = 1;
                            break;
                        }
                    }

                    if (!is_argv)
                    {
                        free(config.extra_options[ii]);
                    }
                }

                /* Shift elements */
                for (int jj = ii; jj < config.extra_options_count - 1; jj++)
                {
                    config.extra_options[jj] = config.extra_options[jj + 1];
                }
                config.extra_options_count -= 1;
                processed = 1;
            }

            if (!processed)
            {
                ii++;
            }
        }
    }

    /* Validate type */
    if (strcmp(config.type, "txt") != 0 &&
        strcmp(config.type, "mp4") != 0 &&
        strcmp(config.type, "stream") != 0)
    {
        fprintf(stderr, "Error: Invalid type '%s'. Use 'txt', 'mp4', or 'stream'.\n", config.type);
        return 1;
    }

    /* Check if current directory ends with "/benchmarks" */
    char cwd[1024];
    int in_benchmarks_dir = 0;
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        size_t len = strlen(cwd);
        if (len >= 11 && strcmp(cwd + len - 11, "/benchmarks") == 0)
        {
            in_benchmarks_dir = 1;
        }
    }

    /* Default patterns if none selected */
    if (config.pattern_count == 0)
    {
        int loaded = -1;
        if (test_list_file != NULL)
        {
            loaded = load_test_file(test_list_file, config.patterns,
                                    &config.pattern_count, MAX_PATTERNS);
            if (loaded != 0)
            {
                fprintf(stderr, "Error: Could not load test file '%s'\n", test_list_file);
                return 1;
            }
        }
        else
        {
            /* Determine path to default tests file */
            char default_file[1024];
            if (in_benchmarks_dir)
            {
                strcpy(default_file, "default_tests.txt");
            }
            else
            {
                strcpy(default_file, "benchmarks/default_tests.txt");
            }

            loaded = load_test_file(default_file, config.patterns,
                                    &config.pattern_count, MAX_PATTERNS);
            if (loaded != 0)
            {
                printf("Warning: Default test file '%s' not found. "
                       "Falling back to built-in patterns.\n", default_file);
                /* Fallback to hardcoded list */
                static const char *fallback_patterns[] =
                {
                    "2Dspiral",
                    "2Dcircle-shuffle",
                    "2Dspiral-shuffle",
                    "2Drand",
                    "3Drand",
                    "2DcircleP10n",
                    "3Dspiral",
                    "3Dstar",
                    "3Dconcentric",
                    "5Dtree",
                    "3Dconcentric_dense"
                };
                for (int ii = 0; ii < 11; ii++)
                {
                    config.patterns[config.pattern_count++] = strdup(fallback_patterns[ii]);
                }
            }
        }
    }

    /* Resolve binary directory paths relative to argv[0] */
    char bin_dir[1024] = "";
    char *last_slash = strrchr(argv[0], '/');
    if (last_slash != NULL)
    {
        size_t len = last_slash - argv[0] + 1;
        if (len < sizeof(bin_dir))
        {
            strncpy(bin_dir, argv[0], len);
            bin_dir[len] = '\0';
        }
    }
    else
    {
        strcpy(bin_dir, "../build/");
    }

    /* Build project if requested */
    if (config.build_first)
    {
        if (rebuild_project(bin_dir) != 0)
        {
            return 1;
        }
    }

    char read_prefix[256] = "";
    char write_prefix[256] = "";
    if (in_benchmarks_dir)
    {
        strcpy(read_prefix, "");
        strcpy(write_prefix, "../benchmarks-out/");
        mkdir("../benchmarks-out", 0755);
    }
    else
    {
        strcpy(read_prefix, "benchmarks/");
        strcpy(write_prefix, "benchmarks-out/");
        mkdir("benchmarks", 0755);
        mkdir("benchmarks-out", 0755);
    }

    char mkseq_path[1024];
    char rnuc_path[1024];
    char clplot_path[1024];
    char txt2mp4_path[1024];

    snprintf(mkseq_path, sizeof(mkseq_path), "%sgric-mktxtseq", bin_dir);
    snprintf(rnuc_path, sizeof(rnuc_path), "%sgric-cluster", bin_dir);
    snprintf(clplot_path, sizeof(clplot_path), "%sgric-plot", bin_dir);
    snprintf(txt2mp4_path, sizeof(txt2mp4_path), "%sgric-ascii-spot-2-video", bin_dir);

    /* Verify that required binaries exist */
    if (access(mkseq_path, X_OK) != 0 || access(rnuc_path, X_OK) != 0)
    {
        fprintf(stderr, "Error: Required binaries not found or not executable.\n");
        fprintf(stderr, "  %s\n  %s\n", mkseq_path, rnuc_path);
        fprintf(stderr, "Please run with --build flag or compile using CMake first.\n");
        return 1;
    }

    /* Create target output directories */
    char log_dir[512];
    snprintf(log_dir, sizeof(log_dir), "%sbenchmark_out", write_prefix);
    mkdir(log_dir, 0755);

    char cluster_out_dir_parent[512];
    snprintf(cluster_out_dir_parent, sizeof(cluster_out_dir_parent), "%sclusteroutdir", write_prefix);
    mkdir(cluster_out_dir_parent, 0755);

    /* Sync MAXIM with nsamples if not explicitly configured */
    if (!config.maxim_set)
    {
        config.maxim = config.nsamples;
    }

    char nbsample_str[32];
    char maxcl_str[32];
    char maxim_str[32];
    snprintf(nbsample_str, sizeof(nbsample_str), "%d", config.nsamples);
    snprintf(maxcl_str, sizeof(maxcl_str), "%d", config.maxcl);
    snprintf(maxim_str, sizeof(maxim_str), "%d", config.maxim);

    /* Initialize summary file header if missing */
    char summary_path[512];
    snprintf(summary_path, sizeof(summary_path), "%sbenchmark_summary.md", write_prefix);

    FILE *sum_fp = fopen(summary_path, "r");
    if (sum_fp == NULL)
    {
        sum_fp = fopen(summary_path, "w");
        if (sum_fp != NULL)
        {
            fprintf(sum_fp,
                    "| Pattern | Type | Algo | Time (ms) | "
                    "Dist Calls | Clusters | Memory (KB) |\n");
            fprintf(sum_fp, "|---|---|---|---|---|---|---|\n");
            fclose(sum_fp);
        }
    }
    else
    {
        fclose(sum_fp);
    }

    /* Allocate result storage for the summary table */
    TestResult *results = calloc(
        config.pattern_count, sizeof(TestResult));
    int result_count = 0;

    /* Run selected benchmarks */
    for (int ii = 0; ii < config.pattern_count; ii++)
    {
        const char *pattern = config.patterns[ii];
        int is_entropy = config.use_entropy;
        for (int jj = 0; jj < config.extra_options_count; jj++)
        {
            if (config.extra_options[jj] != NULL && strstr(config.extra_options[jj], "-entropy") != NULL)
            {
                is_entropy = 1;
                break;
            }
        }
        printf("========================================================\n");
        printf("Benchmark: Pattern=%s Type=%s Algo=%s\n",
               pattern, config.type, is_entropy ? "gric-entropy" : "gric-greedy");

        /* 1. Generate text data */
        char txt_file[512];
        snprintf(txt_file, sizeof(txt_file), "%s%s.txt", read_prefix, pattern);

        int skip_gen = 0;
        if (config.reuse_mp4 && access(txt_file, F_OK) == 0)
        {
            printf("Re-using existing data file: %s\n", txt_file);
            skip_gen = 1;
        }

        if (!skip_gen)
        {
            printf("Generating data for pattern: %s\n", pattern);
            char *gen_args[16];
            int gen_argc = 0;
            gen_args[gen_argc++] = mkseq_path;
            gen_args[gen_argc++] = nbsample_str;
            gen_args[gen_argc++] = txt_file;

            if (strcmp(pattern, "2Dspiral") == 0)
            {
                gen_args[gen_argc++] = "2Dspiral";
            }
            else if (strcmp(pattern, "2Dcircle-shuffle") == 0)
            {
                gen_args[gen_argc++] = "2Dcircle";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "2Dspiral-shuffle") == 0)
            {
                gen_args[gen_argc++] = "2Dspiral";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "2Drand") == 0)
            {
                gen_args[gen_argc++] = "2Drand";
            }
            else if (strcmp(pattern, "3Drand") == 0)
            {
                gen_args[gen_argc++] = "3Drand";
            }
            else if (strcmp(pattern, "2DcircleP10n") == 0)
            {
                gen_args[gen_argc++] = "2Dcircle10";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.04";
            }
            else if (strcmp(pattern, "3Dspiral") == 0)
            {
                gen_args[gen_argc++] = "3Dspiral";
            }
            else if (strcmp(pattern, "3Dstar") == 0)
            {
                gen_args[gen_argc++] = "3Dstar30";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.02";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "3Dconcentric") == 0)
            {
                gen_args[gen_argc++] = "3Dconcentric5";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.02";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "5Dtree") == 0)
            {
                gen_args[gen_argc++] = "5Dtree";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.02";
                gen_args[gen_argc++] = "-shuffle";
            }
            else if (strcmp(pattern, "3Dconcentric_dense") == 0)
            {
                gen_args[gen_argc++] = "3Dconcentric_dense10";
                gen_args[gen_argc++] = "-noise";
                gen_args[gen_argc++] = "0.05";
                gen_args[gen_argc++] = "-shuffle";
            }
            else
            {
                fprintf(stderr, "Error: Unknown pattern '%s'\n", pattern);
                continue;
            }
            gen_args[gen_argc] = NULL;

            int gen_status = run_command_redirect(mkseq_path, gen_args, "/dev/null");
            if (gen_status != 0)
            {
                fprintf(stderr, "Error: Data generation failed (exit code %d)\n", gen_status);
                continue;
            }
        }

        /* 2. Prepare Input File (TXT or MP4 conversion) */
        char input_file[512];
        if (strcmp(config.type, "txt") == 0)
        {
            strcpy(input_file, txt_file);
        }
        else if (strcmp(config.type, "mp4") == 0)
        {
            snprintf(input_file, sizeof(input_file), "%s%s.mp4", read_prefix, pattern);
            int skip_vid = 0;
            if (config.reuse_mp4 && access(input_file, F_OK) == 0)
            {
                printf("Re-using existing video file: %s\n", input_file);
                skip_vid = 1;
            }

            if (!skip_vid)
            {
                printf("Converting %s to %s...\n", txt_file, input_file);
                char *vid_args[] =
                {
                    txt2mp4_path,
                    "64",        /* VID_SIZE */
                    "0.1",       /* VID_ALPHA */
                    txt_file,
                    input_file,
                    "0.0",       /* VID_NOISE */
                    nbsample_str,
                    NULL
                };
                int vid_status = run_command_redirect(txt2mp4_path, vid_args, "/dev/null");
                if (vid_status != 0)
                {
                    fprintf(stderr, "Error: Video conversion failed (exit code %d)\n", vid_status);
                    continue;
                }
            }
        }
        else if (strcmp(config.type, "stream") == 0)
        {
            strcpy(input_file, pattern);
        }

        /* 3. Determine Radius Limit */
        char cur_rlim[32];
        if (strcmp(config.type, "mp4") == 0 || strcmp(config.type, "stream") == 0)
        {
            if (!config.rlim_set)
            {
                /* Default rlim for video/stream is 1000.0 */
                strcpy(cur_rlim, "1000.0");
            }
            else
            {
                strcpy(cur_rlim, config.rlim);
            }
        }
        else
        {
            if (!config.rlim_set && strcmp(pattern, "3Dconcentric_dense") == 0)
            {
                strcpy(cur_rlim, "0.40");
            }
            else if (!config.rlim_set && strcmp(pattern, "3Dspiral") == 0)
            {
                strcpy(cur_rlim, "0.02");
            }
            else if (!config.rlim_set && (strcmp(pattern, "3Drand") == 0 ||
                                          strcmp(pattern, "3Dconcentric") == 0 ||
                                          strcmp(pattern, "5Dtree") == 0))
            {
                strcpy(cur_rlim, "0.20");
            }
            else
            {
                strcpy(cur_rlim, config.rlim);
            }
        }

        /* 4. Construct and Run gric-cluster Command */
        char log_file[512];
        snprintf(log_file, sizeof(log_file),
                 "%sbenchmark_out/%s_%s_gric.log",
                 write_prefix, pattern, config.type);

        char out_dir[512];
        snprintf(out_dir, sizeof(out_dir), "%s%s.cluster.out", write_prefix, pattern);

        printf("Running gric-cluster on %s (rlim=%s)...\n", input_file, cur_rlim);

        char *cluster_args[256];
        int cluster_argc = 0;

        cluster_args[cluster_argc++] = "/usr/bin/time";
        cluster_args[cluster_argc++] = "-v";
        cluster_args[cluster_argc++] = rnuc_path;
        cluster_args[cluster_argc++] = cur_rlim;
        cluster_args[cluster_argc++] = "-maxcl";
        cluster_args[cluster_argc++] = maxcl_str;
        cluster_args[cluster_argc++] = "-maxim";
        cluster_args[cluster_argc++] = maxim_str;
        cluster_args[cluster_argc++] = "-outdir";
        cluster_args[cluster_argc++] = out_dir;
        cluster_args[cluster_argc++] = "-clustered";
        if (config.use_entropy)
        {
            cluster_args[cluster_argc++] = "-entropy";
        }

        int first_extra_arg_idx = cluster_argc;
        for (int jj = 0; jj < config.extra_options_count; jj++)
        {
            /* Reserve 3 slots for -stream, input_file, and the NULL terminator */
            split_args(config.extra_options[jj], cluster_args, &cluster_argc, 256 - 3);
        }

        if (strcmp(config.type, "stream") == 0)
        {
            cluster_args[cluster_argc++] = "-stream";
        }

        cluster_args[cluster_argc++] = input_file;
        cluster_args[cluster_argc] = NULL;

        int run_status = run_command_redirect("/usr/bin/time", cluster_args, log_file);
        if (run_status != 0)
        {
            fprintf(stderr, "Warning: gric-cluster exited with status %d\n", run_status);
        }

        /* Clean up split arguments memory */
        for (int jj = first_extra_arg_idx; jj < cluster_argc; jj++)
        {
            /* Only free if it's not one of static args or input_file */
            if (cluster_args[jj] != NULL &&
                strcmp(cluster_args[jj], "-stream") != 0 &&
                cluster_args[jj] != input_file)
            {
                free(cluster_args[jj]);
            }
        }

        /* 5. Optional: Plot result for txt inputs */
        if (strcmp(config.type, "txt") == 0 && access(clplot_path, X_OK) == 0)
        {
            char cluster_log[1024];
            snprintf(cluster_log, sizeof(cluster_log), "%s/cluster_run.log", out_dir);
            char *plot_args[] = {clplot_path, input_file, cluster_log, NULL};
            run_command_redirect(clplot_path, plot_args, "/dev/null");
        }

        /* 6. Extract Metrics and Log to Summary */
        char m_time[64], m_dists[64], m_dists_sample[64], m_dists_inter[64], m_clusters[64], m_mem[64];
        parse_metrics(log_file, m_time, m_dists, m_dists_sample, m_dists_inter, m_clusters, m_mem);

        printf("Result: Time=%sms, Clusters=%s, Mem=%sKB\n",
               m_time, m_clusters, m_mem);
        
        double total_dists = atof(m_dists);
        double sample_dists = atof(m_dists_sample);
        double inter_dists = atof(m_dists_inter);

        double avg_dists = (config.nsamples > 0) ? (total_dists / config.nsamples) : 0.0;
        double avg_sample_dists = (config.nsamples > 0) ? (sample_dists / config.nsamples) : 0.0;
        double avg_inter_dists = (config.nsamples > 0) ? (inter_dists / config.nsamples) : 0.0;

        if (strcmp(m_dists_sample, "N/A") != 0)
        {
            printf("%sDistances (sum): %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists, avg_dists, ANSI_COLOR_RESET);
            printf("%s  -> Sample-to-cluster: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists_sample, avg_sample_dists, ANSI_COLOR_RESET);
            printf("%s  -> Cluster-to-cluster: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists_inter, avg_inter_dists, ANSI_COLOR_RESET);
        }
        else
        {
            printf("%sDistances: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists, avg_dists, ANSI_COLOR_RESET);
        }

        sum_fp = fopen(summary_path, "a");
        if (sum_fp != NULL)
        {
            char dist_str[256];
            if (strcmp(m_dists_sample, "N/A") != 0)
            {
                snprintf(dist_str, sizeof(dist_str),
                         "%s (S:%s, C:%s)",
                         m_dists, m_dists_sample,
                         m_dists_inter);
            }
            else
            {
                snprintf(dist_str, sizeof(dist_str),
                         "%s", m_dists);
            }
            fprintf(sum_fp,
                    "| %s | %s | %s | %s | %s | %s | %s |\n",
                    pattern, config.type,
                    is_entropy ? "gric-entropy" : "gric-greedy",
                    m_time, dist_str, m_clusters, m_mem);
            fclose(sum_fp);
        }

        /* Store result for the final stdout summary table */
        if (results != NULL &&
            result_count < config.pattern_count)
        {
            TestResult *r = &results[result_count];
            snprintf(r->pattern, sizeof(r->pattern),
                     "%s", pattern);
            snprintf(r->algo, sizeof(r->algo), "%s",
                     is_entropy ? "entropy" : "greedy");
            snprintf(r->time_ms, sizeof(r->time_ms),
                     "%s", m_time);
            r->dist_total = total_dists;
            r->dist_sample = sample_dists;
            r->dist_inter = inter_dists;
            r->avg_dist = avg_dists;
            r->clusters = atoi(m_clusters);
            snprintf(r->mem_kb, sizeof(r->mem_kb),
                     "%s", m_mem);
            r->nsamples = config.nsamples;
            result_count++;
        }
    }

    /* Clean up allocated patterns memory */
    for (int ii = 0; ii < config.pattern_count; ii++)
    {
        free(config.patterns[ii]);
    }

    /* Clean up allocated extra options */
    for (int ii = 0; ii < config.extra_options_count; ii++)
    {
        int is_argv = 0;

        for (int jj = 0; jj < argc; jj++)
        {
            if (config.extra_options[ii] == argv[jj])
            {
                is_argv = 1;
                break;
            }
        }

        if (!is_argv)
        {
            free(config.extra_options[ii]);
        }
    }

    /*
     * Print the stdout summary table.
     *
     * One row per test, with key metrics in fixed-width
     * columns for easy visual comparison.
     */
    printf("\n");
    printf("========================================"
           "========================================"
           "================================\n");
    printf("%sSUMMARY%s\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("========================================"
           "========================================"
           "================================\n");

    if (results != NULL && result_count > 0)
    {
        /* Header */
        printf("%s%-22s %-8s %10s %10s %8s "
               "%8s %6s %10s%s\n",
               ANSI_BOLD,
               "Pattern", "Algo", "Time(ms)",
               "DistTot", "d/frm",
               "dS/frm", "Clust", "Mem(KB)",
               ANSI_COLOR_RESET);
        printf("---------------------- -------- "
               "---------- ---------- -------- "
               "-------- ------ ----------\n");

        /* One row per test */
        for (int ii = 0; ii < result_count; ii++)
        {
            TestResult *r = &results[ii];
            printf("%-22s %-8s %10s %10.0f %8.2f "
                   "%8.2f %6d %10s\n",
                   r->pattern,
                   r->algo,
                   r->time_ms,
                   r->dist_total,
                   r->avg_dist,
                   (r->nsamples > 0)
                       ? (r->dist_sample / r->nsamples)
                       : 0.0,
                   r->clusters,
                   r->mem_kb);
        }

        printf("========================================"
               "========================================"
               "================================\n");
    }

    free(results);

    printf("Benchmarks complete. "
           "Summary also appended to %s\n",
           summary_path);

    return 0;
} // main

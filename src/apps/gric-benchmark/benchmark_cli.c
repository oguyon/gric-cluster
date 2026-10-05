/**
 * @file benchmark_cli.c
 * @brief Command-line argument parsing and path configuration for gric-benchmark.
 */

#include "benchmark.h"
#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

void init_config(
    BenchmarkConfig *config)
{
    config->nsamples = 20000;
    strcpy(config->rlim, "0.10");
    config->rlim_set = 0;
    config->maxcl = 2500;
    config->maxim = 100000;
    config->maxim_set = 0;
    strcpy(config->type, "txt");
    config->reuse_mp4 = 0;
    config->pattern_count = 0;
    config->extra_options_count = 0;
    config->build_first = 0;
    config->use_entropy = 0;
} // init_config

static void filter_single_flag(
    BenchmarkConfig *config,
    const char      *flag_name,
    int             *target_int_val,
    int             *target_set_flag,
    int              argc,
    char            *argv[])
{
    size_t flag_len = strlen(flag_name);

    for (int ii = 0; ii < config->extra_options_count; )
    {
        int processed = 0;

        if (strcmp(config->extra_options[ii], flag_name) == 0)
        {
            if (ii + 1 < config->extra_options_count &&
                config->extra_options[ii + 1][0] != '-')
            {
                *target_int_val = atoi(config->extra_options[ii + 1]);
                if (target_set_flag != NULL)
                {
                    *target_set_flag = 1;
                }

                int is_argv_1 = 0;
                int is_argv_2 = 0;
                for (int jj = 0; jj < argc; jj++)
                {
                    if (config->extra_options[ii] == argv[jj])
                    {
                        is_argv_1 = 1;
                    }
                    if (config->extra_options[ii + 1] == argv[jj])
                    {
                        is_argv_2 = 1;
                    }
                }

                if (!is_argv_1)
                {
                    free(config->extra_options[ii]);
                }
                if (!is_argv_2)
                {
                    free(config->extra_options[ii + 1]);
                }

                for (int jj = ii; jj < config->extra_options_count - 2; jj++)
                {
                    config->extra_options[jj] = config->extra_options[jj + 2];
                }
                config->extra_options_count -= 2;
                processed = 1;
            }
        }
        else if (strncmp(config->extra_options[ii], flag_name, flag_len) == 0 &&
                 config->extra_options[ii][flag_len] == ' ')
        {
            *target_int_val = atoi(config->extra_options[ii] + flag_len + 1);
            if (target_set_flag != NULL)
            {
                *target_set_flag = 1;
            }

            int is_argv = 0;
            for (int jj = 0; jj < argc; jj++)
            {
                if (config->extra_options[ii] == argv[jj])
                {
                    is_argv = 1;
                    break;
                }
            }

            if (!is_argv)
            {
                free(config->extra_options[ii]);
            }

            for (int jj = ii; jj < config->extra_options_count - 1; jj++)
            {
                config->extra_options[jj] = config->extra_options[jj + 1];
            }
            config->extra_options_count -= 1;
            processed = 1;
        }

        if (!processed)
        {
            ii++;
        }
    }
}

int parse_benchmark_cli(
    int              argc,
    char            *argv[],
    BenchmarkConfig *config,
    char           **out_test_list_file)
{
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

    *out_test_list_file = NULL;
    int opt;
    int option_index = 0;
    while ((opt = getopt_long_only(argc, argv, "hn:r:p:f:t:o:b",
                                   long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_help(argv[0]);
                return 1;
            case 'n':
                config->nsamples = atoi(optarg);
                break;
            case 'r':
                strncpy(config->rlim, optarg, sizeof(config->rlim) - 1);
                config->rlim[sizeof(config->rlim) - 1] = '\0';
                config->rlim_set = 1;
                break;
            case 'p':
                if (config->pattern_count < MAX_PATTERNS)
                {
                    config->patterns[config->pattern_count++] = strdup(optarg);
                }
                break;
            case 'f':
                *out_test_list_file = optarg;
                break;
            case 't':
                strncpy(config->type, optarg, sizeof(config->type) - 1);
                config->type[sizeof(config->type) - 1] = '\0';
                break;
            case 'o':
                if (config->extra_options_count < MAX_OPTIONS)
                {
                    config->extra_options[config->extra_options_count++] = strdup(optarg);
                }
                break;
            case 'b':
                config->build_first = 1;
                break;
            case 1001:
                config->reuse_mp4 = 1;
                break;
            case 1002:
                config->maxcl = atoi(optarg);
                break;
            case 1003:
                config->maxim = atoi(optarg);
                config->maxim_set = 1;
                break;
            case 1004:
                config->use_entropy = 1;
                break;
            default:
                fprintf(stderr, "Error: Unknown option\n");
                print_help(argv[0]);
                return -1;
        }
    }

    /* Process positional arguments as extra options */
    while (optind < argc)
    {
        if (config->extra_options_count < MAX_OPTIONS)
        {
            config->extra_options[config->extra_options_count++] = argv[optind];
        }
        optind++;
    }

    /* Factorized option filtering */
    filter_single_flag(config, "-maxcl", &config->maxcl, NULL, argc, argv);
    filter_single_flag(config, "-maxim", &config->maxim, &config->maxim_set, argc, argv);

    /* Validate type */
    if (strcmp(config->type, "txt") != 0 &&
        strcmp(config->type, "mp4") != 0 &&
        strcmp(config->type, "stream") != 0 &&
        strcmp(config->type, "fits") != 0)
    {
        fprintf(stderr,
                "Error: Invalid type '%s'. Use 'txt', 'mp4', 'stream', or 'fits'.\n",
                config->type);
        return -1;
    }

    if (!config->maxim_set)
    {
        config->maxim = config->nsamples;
    }

    return 0;
} // parse_benchmark_cli

int resolve_benchmark_paths(
    const char     *argv0,
    BenchmarkPaths *paths)
{
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

    if (in_benchmarks_dir)
    {
        strcpy(paths->read_prefix, "");
        strcpy(paths->write_prefix, "../benchmarks-out/");
        mkdir("../benchmarks-out", 0755);
    }
    else
    {
        strcpy(paths->read_prefix, "benchmarks/");
        strcpy(paths->write_prefix, "benchmarks-out/");
        mkdir("benchmarks", 0755);
        mkdir("benchmarks-out", 0755);
    }

    paths->bin_dir[0] = '\0';
    const char *last_slash = strrchr(argv0, '/');
    if (last_slash != NULL)
    {
        size_t len = (size_t)(last_slash - argv0 + 1);
        if (len < sizeof(paths->bin_dir))
        {
            strncpy(paths->bin_dir, argv0, len);
            paths->bin_dir[len] = '\0';
        }
    }
    else
    {
        strcpy(paths->bin_dir, "../build/");
    }

    snprintf(paths->mkseq_path, sizeof(paths->mkseq_path), "%sgric-mktxtseq", paths->bin_dir);
    snprintf(paths->rnuc_path, sizeof(paths->rnuc_path), "%sgric-cluster", paths->bin_dir);
    snprintf(paths->clplot_path, sizeof(paths->clplot_path), "%sgric-plot", paths->bin_dir);
    snprintf(paths->txt2mp4_path, sizeof(paths->txt2mp4_path), "%sgric-ascii-spot-2-video",
             paths->bin_dir);
    snprintf(paths->genballs_path, sizeof(paths->genballs_path), "%sgric-gen-balls",
             paths->bin_dir);
    snprintf(paths->summary_path, sizeof(paths->summary_path), "%sbenchmark_summary.md",
             paths->write_prefix);

    /* Verify that required binaries exist */
    if (access(paths->mkseq_path, X_OK) != 0 || access(paths->rnuc_path, X_OK) != 0)
    {
        fprintf(stderr, "Error: Required binaries not found or not executable.\n");
        fprintf(stderr, "  %s\n  %s\n", paths->mkseq_path, paths->rnuc_path);
        fprintf(stderr, "Please run with --build flag or compile using CMake first.\n");
        return -1;
    }

    char log_dir[512];
    snprintf(log_dir, sizeof(log_dir), "%sbenchmark_out", paths->write_prefix);
    mkdir(log_dir, 0755);

    char cluster_out_dir_parent[512];
    snprintf(cluster_out_dir_parent, sizeof(cluster_out_dir_parent),
             "%sclusteroutdir", paths->write_prefix);
    mkdir(cluster_out_dir_parent, 0755);

    return 0;
} // resolve_benchmark_paths

void cleanup_benchmark_config(
    BenchmarkConfig *config,
    int              argc,
    char            *argv[])
{
    for (int ii = 0; ii < config->pattern_count; ii++)
    {
        free(config->patterns[ii]);
    }

    for (int ii = 0; ii < config->extra_options_count; ii++)
    {
        int is_argv = 0;
        for (int jj = 0; jj < argc; jj++)
        {
            if (config->extra_options[ii] == argv[jj])
            {
                is_argv = 1;
                break;
            }
        }

        if (!is_argv)
        {
            free(config->extra_options[ii]);
        }
    }
} // cleanup_benchmark_config

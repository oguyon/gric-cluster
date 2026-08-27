/**
 * @file gric-benchmark.c
 * @brief Main entry point for the GRIC suite auto-benchmarking utility.
 */

#include "benchmark.h"
#include <string.h>

int main(
    int   argc,
    char *argv[])
{
    BenchmarkConfig config;
    init_config(&config);
    cli_colors_init();

    char *test_list_file = NULL;
    int parse_status = parse_benchmark_cli(argc, argv, &config, &test_list_file);
    if (parse_status != 0)
    {
        return (parse_status > 0) ? 0 : 1;
    }

    BenchmarkPaths paths;
    if (resolve_benchmark_paths(argv[0], &paths) != 0)
    {
        return 1;
    }

    if (config.build_first)
    {
        if (rebuild_project(paths.bin_dir) != 0)
        {
            return 1;
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
            char default_file[1024];
            snprintf(default_file, sizeof(default_file), "%sdefault_tests.txt", paths.read_prefix);

            loaded = load_test_file(default_file, config.patterns,
                                    &config.pattern_count, MAX_PATTERNS);
            if (loaded != 0)
            {
                printf("Warning: Default test file '%s' not found. "
                       "Falling back to built-in patterns.\n", default_file);
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
                    "3Dconcentric_dense",
                    "balls_single",
                    "balls_coll"
                };
                for (int ii = 0; ii < 13; ii++)
                {
                    config.patterns[config.pattern_count++] = strdup(fallback_patterns[ii]);
                }
            }
        }
    }

    init_summary_file(paths.summary_path);

    TestResult *results = NULL;
    int result_count = 0;

    run_benchmark_suite(&config, &paths, &results, &result_count);

    print_summary_table(results, result_count);

    if (results != NULL)
    {
        free(results);
    }

    cleanup_benchmark_config(&config, argc, argv);

    printf("Benchmarks complete. Summary also appended to %s\n", paths.summary_path);

    return 0;
} // main

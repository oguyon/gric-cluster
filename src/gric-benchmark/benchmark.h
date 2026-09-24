/**
 * @file benchmark.h
 * @brief Benchmark runner structure definitions and function declarations.
 */

#ifndef GRIC_BENCHMARK_H
#define GRIC_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>
#include "cli_colors.h"

#define MAX_PATTERNS 32
#define MAX_OPTIONS  64

typedef struct
{
    int   nsamples;
    char  rlim[32];
    int   rlim_set;
    int   maxcl;
    int   maxim;
    int   maxim_set;
    char  type[32];
    int   reuse_mp4;
    char *patterns[MAX_PATTERNS];
    int   pattern_count;
    char *extra_options[MAX_OPTIONS];
    int   extra_options_count;
    int   build_first;
    int   use_entropy;
} BenchmarkConfig;

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

typedef struct
{
    char read_prefix[256];
    char write_prefix[256];
    char bin_dir[512];
    char mkseq_path[1024];
    char rnuc_path[1024];
    char clplot_path[1024];
    char txt2mp4_path[1024];
    char genballs_path[1024];
    char summary_path[512];
} BenchmarkPaths;

/**
 * init_config() - Initialize benchmark configuration with default values.
 * @config: Target configuration structure to initialize.
 */
void init_config(
    BenchmarkConfig *config);

/**
 * parse_benchmark_cli() - Parse command line arguments for benchmark runner.
 * @argc:               Argument count.
 * @argv:               Argument vector.
 * @config:             Output benchmark configuration.
 * @out_test_list_file: Output pointer to test list file path if specified.
 *
 * Return: 0 on success, -1 on invalid option, 1 if help requested.
 */
int parse_benchmark_cli(
    int              argc,
    char            *argv[],
    BenchmarkConfig *config,
    char           **out_test_list_file);

/**
 * resolve_benchmark_paths() - Locate companion executables relative to argv[0].
 * @argv0: Path to currently executing binary.
 * @paths: Output structure for resolved executable paths.
 *
 * Return: 0 on success, -1 if key executables cannot be located.
 */
int resolve_benchmark_paths(
    const char      *argv0,
    BenchmarkPaths  *paths);

/**
 * cleanup_benchmark_config() - Release memory allocated during CLI parsing.
 * @config: Configuration structure to clean.
 * @argc:   Argument count.
 * @argv:   Argument vector.
 */
void cleanup_benchmark_config(
    BenchmarkConfig *config,
    int              argc,
    char            *argv[]);

/**
 * run_benchmark_suite() - Execute all configured test patterns and capture metrics.
 * @config:            Benchmark test parameters.
 * @paths:             Resolved tool binary paths.
 * @out_results:       Output pointer to dynamically allocated results array.
 * @out_result_count:  Output count of completed test results.
 *
 * Return: 0 on success, -1 on execution failure.
 */
int run_benchmark_suite(
    const BenchmarkConfig *config,
    const BenchmarkPaths  *paths,
    TestResult           **out_results,
    int                   *out_result_count);

/**
 * init_summary_file() - Create or truncate benchmark summary output file.
 * @summary_path: Filesystem path to summary output file.
 */
void init_summary_file(
    const char *summary_path);

/**
 * append_summary_row() - Append a formatted result row to summary log.
 * @summary_path: Path to summary log file.
 * @pattern:      Test dataset pattern name.
 * @type:         Data element type (float / double).
 * @algo:         Clustering algorithm name.
 * @nsamples:     Number of processed samples.
 * @time_ms:      Elapsed time string.
 * @dists_str:    Number of distance evaluations string.
 * @clusters_str: Cluster count string.
 * @mem_kb:       Peak memory usage string.
 */
void append_summary_row(
    const char *summary_path,
    const char *pattern,
    const char *type,
    const char *algo,
    int         nsamples,
    const char *time_ms,
    const char *dists_str,
    const char *clusters_str,
    const char *mem_kb);

/**
 * print_summary_table() - Print summary table of all benchmark results to stdout.
 * @results:      Array of completed test results.
 * @result_count: Number of results in array.
 */
void print_summary_table(
    const TestResult *results,
    int               result_count);

/**
 * print_help() - Display command-line usage information for gric-benchmark.
 * @progname: Name of executable program.
 */
void print_help(
    const char *progname);

/**
 * run_command_redirect() - Fork and exec a child process redirecting stdout/stderr.
 * @path:     Executable path.
 * @argv:     Argument array (NULL terminated).
 * @log_path: File path to receive redirected output.
 *
 * Return: Process exit code, or -1 on fork/exec failure.
 */
int run_command_redirect(
    const char  *path,
    char *const  argv[],
    const char  *log_path);

/**
 * rebuild_project() - Trigger CMake build to ensure binaries are up to date.
 * @bin_dir: Directory containing built executables.
 *
 * Return: 0 on build success, non-zero on failure.
 */
int rebuild_project(
    const char *bin_dir);

/**
 * split_args() - Tokenize command-line argument string into argv array.
 * @str:      Input argument string.
 * @argv:     Output array of argument pointers.
 * @argc:     Output pointer for token count.
 * @max_args: Maximum number of tokens permitted.
 */
void split_args(
    const char  *str,
    char        *argv[],
    int         *argc,
    int          max_args);

/**
 * parse_metrics() - Extract runtime, distance counts, and memory from log output.
 * @log_path:                Path to execution log file.
 * @out_time:                Output buffer for wall clock time string.
 * @out_dists:               Output buffer for total distance evaluations string.
 * @out_dists_sample:        Output buffer for sample distance count string.
 * @out_dists_intercluster:  Output buffer for intercluster distance count string.
 * @out_clusters:            Output buffer for created cluster count string.
 * @out_mem:                 Output buffer for peak resident memory string.
 */
void parse_metrics(
    const char  *log_path,
    char        *out_time,
    char        *out_dists,
    char        *out_dists_sample,
    char        *out_dists_intercluster,
    char        *out_clusters,
    char        *out_mem);

/**
 * load_test_file() - Load list of benchmark test patterns from file.
 * @filepath:      Path to file containing pattern list.
 * @patterns:      Output array of pattern strings.
 * @pattern_count: Output count of loaded patterns.
 * @max_patterns:  Maximum capacity of patterns array.
 *
 * Return: 0 on success, -1 on file read error.
 */
int load_test_file(
    const char  *filepath,
    char        *patterns[],
    int         *pattern_count,
    int          max_patterns);

#endif /* GRIC_BENCHMARK_H */

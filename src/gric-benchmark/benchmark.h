/**
 * @file benchmark.h
 * @brief Benchmark runner structure definitions and function declarations.
 */

#ifndef GRIC_BENCHMARK_H
#define GRIC_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>
#include "shared/cli_colors.h"

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

/* Benchmark CLI & Config */
void init_config(
    BenchmarkConfig *config);

int parse_benchmark_cli(
    int              argc,
    char            *argv[],
    BenchmarkConfig *config,
    char           **out_test_list_file);

int resolve_benchmark_paths(
    const char      *argv0,
    BenchmarkPaths  *paths);

void cleanup_benchmark_config(
    BenchmarkConfig *config,
    int              argc,
    char            *argv[]);

/* Benchmark Runner */
int run_benchmark_suite(
    const BenchmarkConfig *config,
    const BenchmarkPaths  *paths,
    TestResult           **out_results,
    int                   *out_result_count);

/* Benchmark Reporting */
void init_summary_file(
    const char *summary_path);

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

void print_summary_table(
    const TestResult *results,
    int               result_count);

/* Utilities */
void print_help(
    const char *progname);

int run_command_redirect(
    const char  *path,
    char *const  argv[],
    const char  *log_path);

int rebuild_project(
    const char *bin_dir);

void split_args(
    const char  *str,
    char        *argv[],
    int         *argc,
    int          max_args);

void parse_metrics(
    const char  *log_path,
    char        *out_time,
    char        *out_dists,
    char        *out_dists_sample,
    char        *out_dists_intercluster,
    char        *out_clusters,
    char        *out_mem);

int load_test_file(
    const char  *filepath,
    char        *patterns[],
    int         *pattern_count,
    int          max_patterns);

#endif /* GRIC_BENCHMARK_H */

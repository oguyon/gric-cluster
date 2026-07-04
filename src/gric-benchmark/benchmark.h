#ifndef GRIC_BENCHMARK_H
#define GRIC_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_PATTERNS 32
#define MAX_OPTIONS  64

#include "shared/cli_colors.h"

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

/* Utilities Prototypes */



/* Print help/usage screen. */
void print_help(
    const char *progname);

/* Run a process redirecting output to log file. */
int run_command_redirect(
    const char  *path,
    char *const  argv[],
    const char  *log_path);

/* Build/Rebuild the project. */
int rebuild_project(
    const char *bin_dir);

/* Split space-separated args into argv array. */
void split_args(
    const char  *str,
    char        *argv[],
    int         *argc,
    int          max_args);

/* Extract performance metrics from a run log. */
void parse_metrics(
    const char  *log_path,
    char        *out_time,
    char        *out_dists,
    char        *out_dists_sample,
    char        *out_dists_intercluster,
    char        *out_clusters,
    char        *out_mem);

/* Test List Management Prototypes */

/* Load test patterns list from external file. */
int load_test_file(
    const char  *filepath,
    char        *patterns[],
    int         *pattern_count,
    int          max_patterns);

#endif /* GRIC_BENCHMARK_H */

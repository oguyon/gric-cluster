#ifndef CLUSTER_IO_H
#define CLUSTER_IO_H

#include "cluster_defs.h"

#include <time.h>

char* create_output_dir_name(const char* input_file);
// Print usage information
void print_usage(char *progname);

// Print detailed help for a specific keyword
void print_help_keyword(const char *keyword);

// Print detailed help information
void print_help(char *progname);

// Write results to disk
void write_results(ClusterConfig *config, ClusterState *state);
void write_run_log(ClusterConfig *config, ClusterState *state, const char *cmdline, struct timespec start_ts, double clust_ms, double out_ms, long max_rss);

#endif // CLUSTER_IO_H

#ifndef CLUSTER_IO_H
#define CLUSTER_IO_H

#include "cluster_defs.h"

#include <time.h>

/**
 * create_output_dir_name() - Generate timestamped output directory name for run.
 * @input_file: Source input filename.
 *
 * Return: Allocated string containing directory path (must be freed by caller).
 */
char *create_output_dir_name(const char *input_file);

/**
 * safe_mkdir() - Create a directory if it does not already exist.
 * @path: Directory filesystem path.
 *
 * Return: 0 on success or if already exists, -1 on failure.
 */
int safe_mkdir(const char *path);

/**
 * init_colors_io() - Initialize terminal color formatting for IO diagnostics.
 */
void init_colors_io(void);

/**
 * write_results() - Save cluster anchors, DCC matrix, and assignments to disk.
 * @config: Clustering configuration parameters.
 * @state:  Clustering dynamic state containing anchors and telemetry.
 */
void write_results(
    ClusterConfig *config,
    ClusterState  *state);

/**
 * write_run_log() - Record execution performance log and run summary.
 * @config:   Clustering configuration.
 * @state:    Clustering state.
 * @cmdline:  Raw invocation command-line string.
 * @start_ts: Run start timespec.
 * @clust_ms: Clustering wall-clock time in milliseconds.
 * @out_ms:   Output writing wall-clock time in milliseconds.
 * @max_rss:  Peak resident set size in kilobytes.
 */
void write_run_log(
    ClusterConfig  *config,
    ClusterState   *state,
    const char     *cmdline,
    struct timespec start_ts,
    double          clust_ms,
    double          out_ms,
    long            max_rss);

#endif // CLUSTER_IO_H

/**
 * @file analysis_state.h
 * @brief AnalysisState structure and function prototypes for gric-cluster-analysis.
 */

#ifndef ANALYSIS_STATE_H
#define ANALYSIS_STATE_H

#include <stdio.h>
#include <stdlib.h>
#include "cli_colors.h"

#define MAX_HISTOGRAM_LIMIT 10000

typedef struct
{
    char   cmdline[2048];
    char   start_time[128];
    double time_clustering_ms;
    double time_output_ms;
    double rlim;
    double dprob;
    int    maxcl;
    long   maxim;
    int    num_clusters;
    long   num_frames;
    long   num_dists;
    long   num_dists_sample;
    long   num_dists_intercluster;
    long   num_pruned;
    long   max_rss;

    /* Histograms */
    long  *dist_hist;
    long  *pruned_hist;
    long  *query_hist;
    int    max_hist_val;

    /* Membership data */
    int   *assignments;
    long   assignments_count;

    /* Derived statistics */
    long  *cluster_sizes;
    double shannon_entropy;
    double normalized_entropy;

    /* Temporal Lifetimes */
    long  *birth_frames;
    long  *death_frames;

    /* Transition Matrix */
    long  *transition_matrix;

    /* DCC Matrix */
    double *dcc_matrix;
    int    *dcc_measured;
} AnalysisState;

/**
 * init_state() - Initialize an analysis state structure to zero/defaults.
 * @state: Analysis state pointer.
 */
void init_state(
    AnalysisState *state);

/**
 * free_state() - Free dynamically allocated buffers in analysis state.
 * @state: Analysis state pointer.
 */
void free_state(
    AnalysisState *state);

/**
 * parse_log_file() - Read clustering run log file and extract run parameters.
 * @filename: Log file path.
 * @state:    Target analysis state structure.
 *
 * Return: 0 on success, -1 on error.
 */
int parse_log_file(
    const char    *filename,
    AnalysisState *state);

/**
 * parse_membership_file() - Read frame-to-cluster assignment file.
 * @filename: Membership file path.
 * @state:    Target analysis state structure.
 *
 * Return: 0 on success, -1 on error.
 */
int parse_membership_file(
    const char    *filename,
    AnalysisState *state);

/**
 * parse_dcc_file() - Read distance-between-cluster-centers matrix file.
 * @filename: DCC matrix file path.
 * @state:    Target analysis state structure.
 *
 * Return: 0 on success, -1 on error.
 */
int parse_dcc_file(
    const char    *filename,
    AnalysisState *state);

/**
 * compute_derived_stats() - Compute cluster size variance, entropy, and distributions.
 * @state: Initialized analysis state structure.
 */
void compute_derived_stats(
    AnalysisState *state);

/**
 * compute_euclidean_distance() - Compute Euclidean distance between two vectors.
 * @p1:  First vector coordinates.
 * @p2:  Second vector coordinates.
 * @dim: Number of dimensions.
 *
 * Return: Euclidean distance.
 */
double compute_euclidean_distance(
    const double *restrict p1,
    const double *restrict p2,
    int                    dim);

/**
 * analyze_spatial_spread() - Calculate member distances from assigned cluster centers.
 * @points_file:  Input dataset frames file path.
 * @anchors_file: Cluster anchor frames file path.
 * @state:        Active analysis state.
 *
 * Return: 0 on success, -1 on error.
 */
int analyze_spatial_spread(
    const char    *points_file,
    const char    *anchors_file,
    AnalysisState *state);

/**
 * print_ascii_histogram() - Render a text histogram to standard output.
 * @hist:        Histogram bin counts array.
 * @max_val:     Number of bins.
 * @total_count: Total number of samples.
 */
void print_ascii_histogram(
    const long *hist,
    int         max_val,
    long        total_count);

/**
 * write_text_report() - Output human-readable analysis summary report.
 * @out:           Output file stream.
 * @state:         Completed analysis state.
 * @color_enabled: Non-zero to include ANSI terminal color styling.
 */
void write_text_report(
    FILE          *out,
    AnalysisState *state,
    int            color_enabled);

/**
 * write_json_report() - Output machine-readable JSON analysis report.
 * @out:   Output file stream.
 * @state: Completed analysis state.
 */
void write_json_report(
    FILE          *out,
    AnalysisState *state);

#endif /* ANALYSIS_STATE_H */

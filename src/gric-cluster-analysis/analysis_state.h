/**
 * @file analysis_state.h
 * @brief AnalysisState structure and function prototypes for gric-cluster-analysis.
 */

#ifndef ANALYSIS_STATE_H
#define ANALYSIS_STATE_H

#include <stdio.h>
#include <stdlib.h>
#include "shared/cli_colors.h"

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

/* State Lifecycle */
void init_state(
    AnalysisState *state);

void free_state(
    AnalysisState *state);

/* Parsers */
int parse_log_file(
    const char    *filename,
    AnalysisState *state);

int parse_membership_file(
    const char    *filename,
    AnalysisState *state);

int parse_dcc_file(
    const char    *filename,
    AnalysisState *state);

/* Statistical Computations */
void compute_derived_stats(
    AnalysisState *state);

double compute_euclidean_distance(
    const double *restrict p1,
    const double *restrict p2,
    int                    dim);

int analyze_spatial_spread(
    const char    *points_file,
    const char    *anchors_file,
    AnalysisState *state);

/* Reporting & Formatting */
void print_ascii_histogram(
    const long *hist,
    int         max_val,
    long        total_count);

void write_text_report(
    FILE          *out,
    AnalysisState *state,
    int            color_enabled);

void write_json_report(
    FILE          *out,
    AnalysisState *state);

#endif /* ANALYSIS_STATE_H */

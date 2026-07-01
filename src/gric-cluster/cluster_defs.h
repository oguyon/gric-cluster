#ifndef CLUSTER_DEFS_H
#define CLUSTER_DEFS_H

#include "common.h"
#include <signal.h>
#include <stdio.h>

// Max Cluster Strategy Enum
typedef enum
{
    MAXCL_STOP = 0,
    MAXCL_DISCARD = 1,
    MAXCL_MERGE = 2
} MaxClustStrategy;

// Core clustering parameters
typedef struct
{
    double           rlim;
    int              auto_rlim_mode;
    double           auto_rlim_factor;
    double           deltaprob;
    int              maxnbclust;
    double           tm_mixing_coeff;
    MaxClustStrategy maxcl_strategy;
    double           discard_fraction;
} ConfigAlgorithm;

// Input configuration
typedef struct
{
    long  maxnbfr;
    char *fits_filename;
    int   scandist_mode;
    int   filelist_mode;
    int   stream_input_mode;
    int   cnt2sync_mode;
} ConfigInput;

// Optimization and acceleration
typedef struct
{
    int    ncpu;
    int    gprob_mode;
    double fmatch_a;
    double fmatch_b;
    int    max_gprob_visitors;
    int    pred_mode;
    int    pred_len;
    int    pred_h;
    int    pred_n;
    int    te4_mode;
    int    te5_mode;
} ConfigOptim;

// Output configuration
typedef struct
{
    char *user_outdir;
    int   progress_mode;
    int   average_mode;
    int   distall_mode;
    int   verbose_level;
    int   fitsout_mode;
    int   pngout_mode;
    int   output_dcc;
    int   output_tm;
    int   output_anchors;
    int   output_counts;
    int   output_membership;
    int   output_discarded;
    int   output_clustered;
    int   output_clusters;
} ConfigOutput;

// Configuration structure
typedef struct
{
    ConfigAlgorithm algo;
    ConfigInput     input;
    ConfigOptim     optim;
    ConfigOutput    output;
} ClusterConfig;

// VisitorList structure
typedef struct
{
    int *frames;
    int count;
    int capacity;
} VisitorList;

// Telemetry structure
typedef struct
{
    long    framedist_calls;
    long    clusters_pruned;
    long    total_frames_processed;
    long    total_missed_frames;
    double *pruned_fraction_sum;
    long   *step_counts;
    int     max_steps_recorded;
    long   *dist_counts;
    long   *pruned_counts_by_dist;
} ClusterTelemetry;

// Scratch/Calculation structure
typedef struct
{
    double *current_gprobs;     /**< Geometric probabilities computed during search */
    double *dccarray;           /**< Pairwise inter-cluster distances cache */
    int    *probsortedclindex;  /**< Cluster indices sorted by descending prior probability */
    int    *clmembflag;         /**< Flag indicating if a cluster is still an active candidate */
    double *mixed_probs;        /**< Prior predictive probabilities (frequency * sequence) */
} ClusterScratch;

// State structure
typedef struct
{
    Cluster          *clusters;
    VisitorList      *cluster_visitors;
    int              *assignments;
    FrameInfo        *frame_infos;
    int               num_clusters;
    FILE             *distall_out;
    long             *transition_matrix;
    ClusterTelemetry  telemetry;
    ClusterScratch    scratch;
} ClusterState;

// Candidate structure for sorting
typedef struct
{
    int id;
    double p;
} Candidate;

#define OMP_MIN_CLUSTERS 256

#endif // CLUSTER_DEFS_H

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
    int    entropy_mode;
    int    entropy_max_targets;
    double entropy_min_prob;
    int    sparse_dcc_mode;        /**< 1 to enable bounded sparse distance matrix, 0 for dense */
    int    sparse_dcc_extra_evals; /**< Number of extra inter-cluster measurements (E) per new cluster */
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
    char *shm_filename;
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
    long    framedist_calls_sample;
    long    framedist_calls_intercluster;
    long    clusters_pruned;
    long    total_frames_processed;
    long    total_missed_frames;
    double *pruned_fraction_sum;
    long   *step_counts;
    int     max_steps_recorded;
    long   *dist_counts;
    long   *pruned_counts_by_dist;
    long   *cluster_query_counts;
    double  last_assignment_dist;
    uint64_t num_new_clusters;
    uint64_t last_frame_dists;
    uint64_t last_frame_dfc;
    uint64_t last_frame_dcc;
    double  time_io_ms;
    double  time_step_1;
    double  time_step_2;
    double  time_step_3a;
    double  time_step_3b;
    double  time_step_3b_score;
    double  time_step_3b_filter;
    double  time_step_3b_eval;
    double  time_step_3c;
    double  time_step_4;
    double  time_step_5;
    double  time_step_refine;
} ClusterTelemetry;

// Candidate structure for sorting
typedef struct
{
    int id;
    double p;
} Candidate;

typedef struct
{
    int    id;
    double score;
} TargetScore;

// Scratch/Calculation structure
typedef struct
{
    double *current_gprobs;     /**< Geometric probabilities computed during search */
    double *dcc_min;            /**< Pairwise inter-cluster minimum distance bounds */
    double *dcc_max;            /**< Pairwise inter-cluster maximum distance bounds */
    char   *dcc_measured;       /**< 1 if exactly measured, 0 if estimated/unmeasured */
    int    *probsortedclindex;  /**< Cluster indices sorted by descending prior probability */
    int    *clmembflag;         /**< Flag indicating if a cluster is still an active candidate */
    double *mixed_probs;        /**< Prior predictive probabilities (frequency * sequence) */
    uint64_t *consistency_mask; /**< Precomputed 3D geometric consistency bitmask */
    double *entropy_p_current;  /**< Pre-allocated scratch buffer for entropy search probabilities */
    Candidate *entropy_candidates; /**< Pre-allocated scratch buffer for sorting candidates */
    TargetScore *entropy_prob_scores;  /**< Pre-allocated scratch buffer for target scores */
    TargetScore *entropy_prune_scores; /**< Pre-allocated scratch buffer for prune scores */
    int         *entropy_active_indices; /**< Pre-allocated active indices array */
    double      *entropy_plog2p;       /**< Pre-allocated plog2p probabilities array */
    uint8_t     *entropy_visited;      /**< Pre-allocated visited boolean array */
    Candidate   *refine_queue;                  /**< Pre-allocated queue of closest unmeasured pairs */
    int          refine_queue_size;             /**< Number of active items in the queue */
    int          refine_queue_idx;              /**< Current index in the queue */
    int          refine_queue_capacity;         /**< Capacity of the queue */
    int          refine_queue_last_num_clusters;/**< Number of clusters during last queue rebuild */
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
    void             *shm_ptr;
} ClusterState;

#define OMP_MIN_CLUSTERS 256

#endif // CLUSTER_DEFS_H

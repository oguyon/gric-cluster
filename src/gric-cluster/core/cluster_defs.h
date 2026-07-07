#ifndef CLUSTER_DEFS_H
#define CLUSTER_DEFS_H

/**
 * @file cluster_defs.h
 * @brief Core data structures and configuration types
 *        for the GRIC clustering engine.
 */

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

/** Core clustering parameters. */
typedef struct
{
    double           rlim;             /**< Distance threshold for assignment */
    int              auto_rlim_mode;   /**< 1 to auto-compute rlim from scandist */
    double           auto_rlim_factor; /**< Multiplier for auto-rlim */
    double           deltaprob;        /**< Min probability to remain candidate */
    int              maxnbclust;       /**< Maximum number of clusters */
    double           tm_mixing_coeff;  /**< Transition-matrix mixing weight */
    MaxClustStrategy maxcl_strategy;   /**< Strategy when maxnbclust reached */
    double           discard_fraction; /**< Fraction of clusters to discard */
} ConfigAlgorithm;

/** Input configuration. */
typedef struct
{
    long  maxnbfr;           /**< Max frames to process */
    char *fits_filename;     /**< Path to input FITS or stream */
    int   scandist_mode;     /**< 1 = scandist-only */
    int   filelist_mode;     /**< 1 = input is file list */
    int   stream_input_mode; /**< 1 = shared-memory stream */
    int   cnt2sync_mode;     /**< 1 = cnt2 semaphore sync */
    int   tile_grid_x;       /**< Tile grid columns (0=tilemap) */
    int   tile_grid_y;       /**< Tile grid rows (0=tilemap) */
    char *tile_map_file;     /**< Path to integer FITS tile map */
    char *tile_config_file;  /**< Per-tile ASCII config file */
    int   retrieval_window;  /**< Tuple retrieval lookback */
} ConfigInput;

/** Optimization and acceleration parameters. */
typedef struct
{
    int    ncpu;                    /**< Number of OpenMP threads */
    int    gprob_mode;              /**< 1 to enable geometric probability */
    double fmatch_a;                /**< gprob exponential decay parameter a */
    double fmatch_b;                /**< gprob exponential decay parameter b */
    int    max_gprob_visitors;      /**< Max visitor entries per cluster */
    int    pred_mode;               /**< 1 to enable trajectory prediction */
    int    pred_len;                /**< Pattern length for prediction matching */
    int    pred_h;                  /**< History horizon for pattern search */
    int    pred_n;                  /**< Max prediction candidates returned */
    int    te4_mode;                /**< 1 to enable 4-point triangle ineq. */
    int    te5_mode;                /**< 1 to enable 5-point triangle ineq. */
    int    entropy_mode;            /**< 1 to enable entropy-guided search */
    int    entropy_max_targets;     /**< Max targets evaluated per entropy step */
    double entropy_min_prob;        /**< Min prior prob to enter entropy eval */
    double entropy_gate_bits;       /**< Entropy gating threshold (bits) */
    double entropy_first_gate_bits; /**< Gate threshold at meas depth 0 (bits) */
    int    entropy_fast_mode;       /**< 1 to use popcount-only surrogate */
    int    sparse_dcc_mode;         /**< 1 to enable bounded sparse DCC, 0 for dense */
    int    sparse_dcc_extra_evals;  /**< Extra inter-cluster measurements per new cluster */
    int    soft_bayesian_mode;      /**< 1 to enable soft Bayesian updates */
    double soft_bayesian_sigma_coeff; /**< Coefficient multiplying rlim for sigma */
    int    disable_pass2;           /**< 1 to disable Pass 2 fusion (tuple prediction) */
} ConfigOptim;

/** Output configuration. */
typedef struct
{
    char *user_outdir;       /**< User-specified output directory path */
    int   progress_mode;     /**< 1 to print progress bar to stdout */
    int   average_mode;      /**< 1 to write per-cluster average frames */
    int   distall_mode;      /**< 1 to write all distances to distall.txt */
    int   verbose_level;     /**< 0 = quiet, 1 = verbose, 2 = very verbose */
    int   fitsout_mode;      /**< 1 to write results as FITS files */
    int   pngout_mode;       /**< 1 to write results as PNG files */
    int   output_dcc;        /**< 1 to write inter-cluster distance matrix */
    int   output_tm;         /**< 1 to write transition matrix */
    int   output_anchors;    /**< 1 to write cluster anchor frames */
    int   output_counts;     /**< 1 to write cluster member counts */
    int   output_membership; /**< 1 to write frame membership log */
    int   output_discarded;  /**< 1 to write discarded-frame list */
    int   output_clustered;  /**< 1 to write clustered-frame cube */
    int   output_clusters;   /**< 1 to write per-cluster frame lists */
    char *shm_filename;      /**< Shared-memory status image name */
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
    double  time_io_ms;            /**< Cumulative I/O time (ms) */
    double  time_step_1;           /**< Step 1: initial cluster setup (ms) */
    double  time_step_2;           /**< Step 2: prediction candidate retrieval (ms) */
    double  time_step_3a;          /**< Step 3a: prior mixing and pruning (ms) */
    double  time_step_3b;          /**< Step 3b: target selection (ms) */
    double  time_step_3b_score;    /**< Step 3b sub: scoring phase (ms) */
    double  time_step_3b_filter;   /**< Step 3b sub: filtering phase (ms) */
    double  time_step_3b_eval;     /**< Step 3b sub: evaluation phase (ms) */
    double  time_step_3c;          /**< Step 3c: distance measurement (ms) */
    double  time_step_4;           /**< Step 4: new cluster creation (ms) */
    double  time_step_5;           /**< Step 5: telemetry and serialization (ms) */
    double   time_step_refine;     /**< Sparse DCC bound refinement (ms) */
    double   entropy_sum_initial;      /**< Accumulated H at meas_idx==0 */
    double   entropy_max_initial;      /**< Maximum H at meas_idx==0 */
    double   entropy_last_initial;     /**< H at meas_idx==0 for last frame */
    uint64_t entropy_frames_gated;     /**< Calls where gate returned greedy */
    uint64_t entropy_frames_evaluated; /**< Calls where full eval ran */
} ClusterTelemetry;

// Candidate structure for sorting
typedef struct
{
    int id;
    double p;
} Candidate;

/** Cluster target with associated score for entropy-guided ranking. */
typedef struct
{
    int    id;    /**< Cluster index */
    double score; /**< Computed score (information gain or prune value) */
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
    int         *tuple_pred_candidates;         /**< Pre-populated candidates from joint prediction */
    int          tuple_pred_count;              /**< Number of candidates pre-populated */
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

/** Minimum cluster count for OpenMP parallelization of pruning loops. */
#define OMP_MIN_CLUSTERS 256

#endif // CLUSTER_DEFS_H

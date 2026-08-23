#ifndef KNN_DEFS_H
#define KNN_DEFS_H

/**
 * @file knn_defs.h
 * @brief Core data structures and configuration types for the gric-knn engine.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** Output format selection */
typedef enum
{
    KNN_FORMAT_AUTO = 0,
    KNN_FORMAT_FITS = 1,
    KNN_FORMAT_TXT  = 2
} KnnOutputFormat;

/** Compact metadata for a single frame member */
typedef struct
{
    uint32_t frame_id;
    float    r_anchor; /**< Precomputed Euclidean distance to its cluster anchor */
} MemberMeta;

/** Cluster index structure resident in RAM during Pass 2 */
typedef struct
{
    int         cluster_id;
    double     *anchor_data; /**< Cluster anchor frame pixel vector */
    double      radius;      /**< Max Euclidean distance from anchor to any member */
    int         num_members;
    int         capacity;
    MemberMeta *members;     /**< Array of member metadata records */
} KnnCluster;

/** Single nearest neighbor record */
typedef struct
{
    int    frame_id;
    double dist;
} KnnNeighbor;

/** Bounded Max-Heap for tracking top-k nearest neighbors */
typedef struct
{
    int          k;
    int          count;
    KnnNeighbor *data;
} KnnMaxHeap;

/** Runtime configuration parameters for gric-knn */
typedef struct
{
    char           *input_data_path;
    char           *cluster_dir;
    char           *output_path;
    int             k;
    int             min_temporal_sep; /**< Minimum frame index separation |i - j| >= dtmin */
    int             past_only;        /**< 1 to search only in past frames j < i */
    int             future_only;      /**< 1 to search only in future frames j > i */
    double          epsilon;          /**< Slack factor for (1+eps)-ANN pruning */
    double          rlim_cutoff;      /**< Optional max distance cutoff */
    const double   *memory_data;      /**< Optional in-memory dataset buffer [N * elements] */
    int             nthreads;         /**< Number of OpenMP worker threads */
    KnnOutputFormat output_format;    /**< Output format choice */
    int             progress_mode;    /**< 1 to show live progress bar */
    int             verbose_level;    /**< 0 = quiet, 1 = normal, 2 = verbose */
} KnnConfig;

/** Telemetry statistics for performance diagnostics */
typedef struct
{
    uint64_t total_queries;
    uint64_t total_candidates_considered;
    uint64_t level1_clusters_pruned;
    uint64_t level2_anchors_pruned;
    uint64_t level3_annular_pruned;
    uint64_t temporal_pruned;
    uint64_t framedist_calls;
    double   time_load_ms;
    double   time_search_ms;
    double   time_write_ms;
} KnnTelemetry;

/** Pass 1 Cluster Model resident in RAM */
typedef struct
{
    long        frame_width;
    long        frame_height;
    long        frame_elements;
    int         num_clusters;
    long        total_dataset_frames;
    KnnCluster *clusters;
    double     *dcc_matrix;          /**< Dense M x M inter-cluster distance matrix */
    int        *frame_cluster_map;   /**< Cluster ID for each frame index [0..N-1] */
    float      *frame_r_anchor;      /**< Distance to anchor for each frame [0..N-1] */
    int         is_fits_input;       /**< 1 if input dataset is FITS, 0 if ASCII */
} KnnModel;

/** Per-query result structure containing top-k neighbors */
typedef struct
{
    int    *indices;   /**< Size N x k */
    double *distances; /**< Size N x k */
} KnnResults;

#endif // KNN_DEFS_H

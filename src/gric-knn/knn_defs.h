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
    void       *anchor_data; /**< Cluster anchor frame pixel vector */
    double      radius;      /**< Max Euclidean distance from anchor to any member */
    int         num_members;
    int         capacity;
    MemberMeta *members;     /**< Array of member metadata records */
} KnnCluster;

/** Super-Cluster (Meta-Cluster) grouping multiple child clusters */
typedef struct
{
    int     super_id;          /**< Super-cluster index */
    int     medoid_cluster_id; /**< Index of central cluster acting as super-anchor */
    double  radius;            /**< Maximum distance from super-anchor to any child member */
    int     num_clusters;      /**< Number of constituent child clusters */
    int    *cluster_ids;       /**< Array of constituent child cluster indices */
} KnnSuperCluster;

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
    double          rlim_cutoff;       /**< Optional max distance cutoff */
    const void     *memory_data;       /**< Optional in-memory dataset buffer [N * elements] */
    char           *query_data_path;   /**< Optional path to external query dataset */
    long            query_num_frames;  /**< Frame count of external query dataset */
    int             nthreads;          /**< Number of OpenMP worker threads */
    KnnOutputFormat output_format;     /**< Output format choice */
    int             progress_mode;     /**< 1 to show live progress bar */
    int             verbose_level;     /**< 0 = quiet, 1 = normal, 2 = verbose */
    int             use_multi_pivot;   /**< 1 to enable Multi-Anchor Pivot Bounding (AESA) */
    int             use_reciprocal;    /**< 1 to enable Symmetric Distance Reciprocal Push */
    int             use_angular_bound; /**< 1 to enable Angular Cosine Directional Bounding */
    int             use_trajectory;    /**< 1 to enable Trajectory Momentum (smooth trajectories) */
    int             approx_mode;       /**< 1 to enable Fast Approximate Graph Search */
    int             ef_search;         /**< Search candidate pool size (default: 2*k in approx) */
    int             use_double;        /**< 1 to run computations in double precision */
} KnnConfig;

/** Telemetry statistics for performance diagnostics */
typedef struct
{
    uint64_t total_queries;
    uint64_t total_candidates_considered;
    uint64_t level0_super_clusters_pruned;
    uint64_t level1_clusters_pruned;
    uint64_t level2_anchors_pruned;
    uint64_t level3_annular_pruned;
    uint64_t temporal_pruned;
    uint64_t reciprocal_reused;
    uint64_t graph_seeds_evaluated;
    uint64_t graph_edges_pruned;
    uint64_t multi_pivot_pruned;
    uint64_t angular_pruned;
    uint64_t global_containment_hits;
    uint64_t framedist_calls;
    uint64_t trajectory_warmstarts;
    double   time_load_ms;
    double   time_search_ms;
    double   time_write_ms;
} KnnTelemetry;

/** Pass 1 Cluster Model resident in RAM */
typedef struct
{
    long             frame_width;
    long             frame_height;
    long             frame_elements;
    int              num_clusters;
    long             total_dataset_frames;
    int              is_double;           /**< 1 if anchors & queries in double precision */
    KnnCluster      *clusters;
    double          *dcc_matrix;          /**< Dense M x M inter-cluster distance matrix */
    int             *frame_cluster_map;   /**< Cluster ID for each frame index [0..N-1] */
    float           *frame_r_anchor;      /**< Distance to anchor for each frame [0..N-1] */
    int              num_super_clusters;  /**< Number of super-clusters K */
    KnnSuperCluster *super_clusters;      /**< Array of super-clusters [0..K-1] */
    double          *dss_matrix;          /**< Dense K x K inter-super-cluster distance matrix */
    int             *cluster_super_map;   /**< Super-cluster index for each cluster [0..M-1] */
    int              is_fits_input;       /**< 1 if input dataset is FITS, 0 if ASCII */
    int              has_knn_graph;       /**< 1 if precomputed k-NN graph is available */
    int              graph_k;             /**< Number of neighbors per node in graph */
    uint32_t        *graph_indices;       /**< [N x graph_k] neighbor node indices in A */
    float           *graph_distances;     /**< [N x graph_k] precomputed neighbor distances */
    float           *graph_mutual_dists;  /**< [N x (graph_k * (graph_k - 1) / 2)] mutual dists */
    const void     **anchor_ptrs;         /**< [M] array of anchor pointers */
    double          *cluster_radii;       /**< [M] array of cluster radii */
} KnnModel;

/** Per-query result structure containing top-k neighbors */
typedef struct
{
    long    num_queries; /**< Total query count */
    int    *indices;     /**< Size N x k */
    double *distances;   /**< Size N x k */
} KnnResults;

/**
 * struct KnnVisitedTracker - Per-query frame deduplication tracker.
 * @tags:  Array of query epochs indexed by candidate frame ID [N_cand].
 * @epoch: Monotonically increasing query epoch counter.
 */
typedef struct
{
    uint32_t *tags;
    uint32_t  epoch;
} KnnVisitedTracker;

/**
 * knn_visited_check_and_mark() - Check if frame was visited in current query and mark it.
 * @tracker:  Pointer to KnnVisitedTracker.
 * @frame_id: Frame index to check and mark.
 *
 * Return: 1 if already visited in this query, 0 if not visited yet (and now marked).
 */
static inline int knn_visited_check_and_mark(
    KnnVisitedTracker *tracker,
    long               frame_id)
{
    if (tracker == NULL || tracker->tags == NULL)
    {
        return 0;
    }

    if (tracker->tags[frame_id] == tracker->epoch)
    {
        return 1;
    }

    tracker->tags[frame_id] = tracker->epoch;
    return 0;
}

/**
 * struct KnnTrajectoryTracker - Thread-local sequential query trajectory state.
 * @prev_query_id:   Index of the preceding query frame evaluated by this thread.
 * @prev_cluster_id: Most recent matching cluster ID.
 * @prev_best_seed:  Most recent closest reference frame ID (1-NN match).
 * @prev_best_dist:  Distance to @prev_best_seed.
 * @num_cached:      Number of cached neighbor candidates from preceding query.
 * @cached_ids:      Candidate frame IDs from preceding query top-k.
 * @cached_dists:    Candidate distances from preceding query top-k.
 */
typedef struct
{
    long   prev_query_id;
    int    prev_cluster_id;
    long   prev_best_seed;
    double prev_best_dist;
    int    num_cached;
    int    cached_ids[64];
    double cached_dists[64];
} KnnTrajectoryTracker;

/**
 * knn_get_mutual_dist() - Retrieve precomputed mutual distance between neighbors i and j.
 * @model: Active KnnModel.
 * @u:     Central node frame index in A.
 * @i:     First neighbor local index in [0..graph_k-1].
 * @j:     Second neighbor local index in [0..graph_k-1].
 *
 * Return: Mutual Euclidean distance d_A(i, j) or 0.0f/error.
 */
static inline float knn_get_mutual_dist(
    const KnnModel *model,
    long            u,
    int             i,
    int             j)
{
    if (model == NULL || model->graph_mutual_dists == NULL || i == j)
    {
        return 0.0f;
    }

    int k = model->graph_k;
    if (i >= k || j >= k || u < 0 || u >= model->total_dataset_frames)
    {
        return -1.0f;
    }

    if (i > j)
    {
        int tmp = i;
        i = j;
        j = tmp;
    }

    long m_per_node = ((long)k * (long)(k - 1)) / 2;
    long pair_idx = (long)i * (long)k - ((long)i * (long)(i + 1)) / 2 + (long)(j - i - 1);
    long offset = u * m_per_node + pair_idx;

    return model->graph_mutual_dists[offset];
}

#endif // KNN_DEFS_H

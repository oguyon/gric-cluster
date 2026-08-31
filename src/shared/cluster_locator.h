/**
 * @file cluster_locator.h
 * @brief Standalone shared cluster localization and 3P metric bounding engine.
 */

#ifndef CLUSTER_LOCATOR_H
#define CLUSTER_LOCATOR_H

#include <stdbool.h>
#include <stdint.h>

/** Configuration flags matching gric-cluster options */
typedef struct
{
    int    max_targets;     /**< Max anchor distance evaluations (default: 8) */
    double rlim;            /**< Matching radius cutoff (0 = disabled) */
    double tau_max;         /**< Current k-NN upper bound distance */
    double epsilon;         /**< Slack factor (1 + eps) for approximate bounds */
    int    entropy_mode;    /**< 1 = Shannon entropy, 0 = greedy / gprob */
    int    entropy_fast;    /**< 1 = fast entropy early exit */
    int    gprob_mode;      /**< 1 = geometric probability weighting */
    int    te4_mode;        /**< 1 = 4-point geometric bounding */
    int    prev_cluster_id; /**< Preceding query cluster ID for trajectory warm-starting */
} ClusterLocatorConfig;

/** Results of the coarse cluster localization */
typedef struct
{
    int      best_cluster_id;        /**< Closest cluster index */
    double   best_anchor_dist;       /**< Distance to best cluster anchor */
    int      num_evaluated_anchors;  /**< Number of anchor distances computed */
    int      evaluated_clusters[32]; /**< Array of evaluated cluster indices */
    double   evaluated_dists[32];    /**< Array of computed anchor distances */
    uint8_t *active_cluster_mask;    /**< Array: 1 = surviving, 0 = pruned */
} ClusterLocatorResult;

/**
 * cluster_locate_sample() - Locates the matching/closest cluster for sample q.
 * @query_data:       Pointer to sample/query frame vector.
 * @frame_elements:   Number of elements per vector.
 * @num_clusters:     Total clusters M in model.
 * @cluster_anchors:  Array of pointers to anchor vectors [M].
 * @cluster_radii:    Array of cluster radii [M].
 * @dcc_matrix:       Dense M x M inter-cluster distance matrix.
 * @config:           Tuning and runtime configuration.
 * @result:           Output structure to populate.
 *
 * Return: 0 on success, -1 on error.
 */
int cluster_locate_sample(
    const double               *query_data,
    long                        frame_elements,
    int                         num_clusters,
    const double *const        *cluster_anchors,
    const double               *cluster_radii,
    const double               *dcc_matrix,
    const ClusterLocatorConfig *config,
    ClusterLocatorResult       *result);

#endif // CLUSTER_LOCATOR_H

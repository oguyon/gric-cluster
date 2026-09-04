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
    int    te5_mode;        /**< 1 = 5-point geometric bounding */
    int    prev_cluster_id; /**< Preceding query cluster ID for trajectory warm-starting */
    int    is_double;       /**< 1 = double precision, 0 = float */
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
 * calc_min_dist_4pt() - Computes minimum distance using a 4-point configuration.
 * @d14: Distance between point 1 and 4.
 * @d24: Distance between point 2 and 4.
 * @d12: Distance between point 1 and 2.
 * @d13: Distance between point 1 and 3.
 * @d23: Distance between point 2 and 3.
 *
 * Return: Min distance between point 3 and 4 in reconstructed 2D space.
 */
double calc_min_dist_4pt(
    double d14,
    double d24,
    double d12,
    double d13,
    double d23);

/**
 * calc_min_dist_5pt() - Computes minimum distance using a 5-point configuration.
 * @d_f_c1:  Distance from frame F to C1.
 * @d_f_c2:  Distance from frame F to C2.
 * @d_f_c3:  Distance from frame F to C3.
 * @d_t_c1:  Distance from target T to C1.
 * @d_t_c2:  Distance from target T to C2.
 * @d_t_c3:  Distance from target T to C3.
 * @d_c1_c2: Distance between C1 and C2.
 * @d_c1_c3: Distance between C1 and C3.
 * @d_c2_c3: Distance between C2 and C3.
 *
 * Return: Computed distance between F and T in reconstructed 3D space.
 */
double calc_min_dist_5pt(
    double d_f_c1,
    double d_f_c2,
    double d_f_c3,
    double d_t_c1,
    double d_t_c2,
    double d_t_c3,
    double d_c1_c2,
    double d_c1_c3,
    double d_c2_c3);

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
    const void                 *query_data,
    long                        frame_elements,
    int                         num_clusters,
    const void *const          *cluster_anchors,
    const double               *cluster_radii,
    const double               *dcc_matrix,
    const ClusterLocatorConfig *config,
    ClusterLocatorResult       *result);

#endif // CLUSTER_LOCATOR_H

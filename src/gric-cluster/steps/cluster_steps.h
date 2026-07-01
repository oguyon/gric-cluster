/**
 * @file cluster_steps.h
 * @brief Prototype definitions for single frame clustering steps.
 */

#ifndef CLUSTER_STEPS_H
#define CLUSTER_STEPS_H

#include "cluster_defs.h"
#include <stdio.h>

/**
 * @brief Initialize the very first cluster using the first ingested frame.
 */
void initialize_initial_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *assigned_cluster);

/**
 * @brief Compute predictive prior probabilities mixing frequency and sequence transitions.
 */
void compute_priors_and_mixing(
    ClusterConfig *config,
    ClusterState  *state,
    int            prev_assigned_cluster,
    Candidate     *sorting_candidates);

/**
 * @brief Select the next cluster candidate to target for distance evaluation.
 */
int select_next_measurement_target(
    ClusterConfig *config,
    ClusterState  *state,
    int           *k_search,
    const int     *pred_candidates,
    int            num_preds,
    int           *current_pred_idx);

/**
 * @brief Recompute the consistency bitmask for all cluster pairs.
 */
void recompute_consistency_mask(
    ClusterConfig *config,
    ClusterState  *state);

/**
 * @brief Update the consistency bitmask when a new cluster is added.
 */
void update_consistency_mask_for_new_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    int            new_cl);

/**
 * @brief Measure the distance from the current frame to a target cluster anchor.
 */
double measure_distance_to_cluster(
    int            cj,
    Frame         *current_frame,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int            is_prediction);

/**
 * @brief Prune candidate search space and update probabilities based on measured distance.
 */
void update_probabilities_and_pruning(
    int            cj,
    double         dfc,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count);

/**
 * @brief Update geometric match probabilities using matching visitor history.
 */
void update_geometric_probabilities(
    ClusterConfig *config,
    ClusterState  *state,
    int            cj,
    double         dfc);

/**
 * @brief Handle new cluster creation, capacity checking, and eviction heuristics.
 */
int handle_new_cluster_creation(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *prev_assigned_cluster,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count);

/**
 * @brief Record frame assignment, transition stats, telemetry, and serialize logs.
 */
void record_step_assignment(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int            assigned_cluster,
    int           *prev_assigned_cluster,
    FILE          *ascii_out,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count,
    long           start_pruned_val);

#endif // CLUSTER_STEPS_H

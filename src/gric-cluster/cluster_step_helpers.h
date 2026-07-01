/**
 * @file cluster_step_helpers.h
 * @brief Helper procedures for processing a single image frame.
 *
 * Provides prototype definitions for single frame clustering steps,
 * including sorting, predictions, standard matching, geometry probability
 * calculations, and cluster eviction handling.
 */

#ifndef CLUSTER_STEP_HELPERS_H
#define CLUSTER_STEP_HELPERS_H

#include "cluster_defs.h"
#include <stdio.h>

void initialize_initial_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *assigned_cluster);

void compute_priors_and_mixing(
    ClusterConfig *config,
    ClusterState  *state,
    int            prev_assigned_cluster,
    Candidate     *sorting_candidates);

int select_next_measurement_target(
    ClusterConfig *config,
    ClusterState  *state,
    int           *k_search,
    const int     *pred_candidates,
    int            num_preds,
    int           *current_pred_idx);

double measure_distance_to_cluster(
    int            cj,
    Frame         *current_frame,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int            is_prediction);

void update_probabilities_and_pruning(
    int            cj,
    double         dfc,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count);

void update_geometric_probabilities(
    ClusterConfig *config,
    ClusterState  *state,
    int            cj,
    double         dfc);

int handle_new_cluster_creation(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *prev_assigned_cluster,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count);

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

#endif // CLUSTER_STEP_HELPERS_H

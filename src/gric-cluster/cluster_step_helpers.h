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

int evaluate_prediction_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int           *assigned_cluster,
    int           *found);

int evaluate_standard_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int           *assigned_cluster,
    int           *found,
    double        *out_dfc,
    int           *k_search,
    Candidate     *verbose_candidates);

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

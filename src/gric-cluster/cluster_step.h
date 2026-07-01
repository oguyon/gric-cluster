/**
 * @file cluster_step.h
 * @brief Declarations for processing a single image frame.
 *
 * Provides interface for executing a complete clustering assignment
 * step for a single frame, including sorting, matching, pruning,
 * and eviction strategy resolution.
 */

#ifndef CLUSTER_STEP_H
#define CLUSTER_STEP_H

#include "cluster_defs.h"
#include <stdio.h>

int cluster_frame(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *prev_assigned_cluster,
    FILE          *ascii_out,
    int           *temp_indices,
    double        *temp_dists,
    Candidate     *sorting_candidates,
    Candidate     *verbose_candidates);

#endif // CLUSTER_STEP_H

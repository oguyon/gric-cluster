/**
 * @file cluster_step.c
 * @brief High-level orchestration of a single image frame assignment step.
 *
 * Implements a sequential driver invoking procedurized steps to cluster
 * a single frame: setup, mixing priors, prediction matching, standard matching,
 * and eviction strategy resolution.
 *
 * Distance measurements (calls to get_dist()) are performed at:
 * - Step 3 (Trajectory prediction): distance of current frame to predicted cluster anchors
 *   is measured to test for a match.
 * - Step 4 (Standard search): distance of current frame to candidates is measured in sequence
 *   of mixed probability. If the candidate does not match, distances between cluster anchors
 *   are computed (and cached in state->scratch.dccarray) to prune other candidate clusters
 *   via triangle inequalities.
 * - Step 5 (New cluster creation): pairwise distances between the new cluster anchor and all
 *   existing cluster anchors are measured and cached to maintain the dccarray matrix.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_step.h"
#include "cluster_step_helpers.h"
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
    Candidate     *verbose_candidates)
{
    int  assigned_cluster = -1;
    long start_pruned_val = state->telemetry.clusters_pruned;
    int  temp_count = 0;

    // Step 1: Base case setup.
    // If no clusters exist yet, the very first ingested frame serves as the anchor frame
    // for Cluster 0, initializing our clustering space.
    // Output: Sets state->num_clusters to 1, sets state->clusters[0], and assigns
    // assigned_cluster = 0.
    if (state->num_clusters == 0)
    {
        initialize_initial_cluster(config, state, current_frame, &assigned_cluster);
        temp_indices[0] = 0;
        temp_dists[0] = 0.0;
        temp_count = 1;
    }
    else
    {
        int found = 0;
        int k_search = 0;
        double dfc = 0.0;

        // Step 2: Probability normalization and mixing.
        // Cluster priors are normalized so they sum to 1. If sequence mixing is enabled,
        // we blend the normalized priors with transition matrix statistics based on the
        // previous frame's assignment, helping exploit temporal cluster transition sequences.
        // Output: Updates normalized state->clusters[i].prob, state->scratch.mixed_probs,
        // and state->scratch.probsortedclindex.
        compute_priors_and_mixing(config, state, *prev_assigned_cluster, sorting_candidates);

        // Step 3: Trajectory prediction shortcut.
        // Evaluates a set of predicted candidates derived from temporal transitions first.
        // If a candidate matches within the threshold 'rlim', we assign immediately and skip
        // the remaining search steps.
        // Distance evaluation: Measures distance to predicted cluster anchors to test for a match.
        // Output: Sets found = 1 and assigned_cluster to matched ID if found; updates
        // state->telemetry.clusters_pruned.
        evaluate_prediction_candidates(config, state, current_frame, temp_indices, temp_dists,
                                       &temp_count, &assigned_cluster, &found);

        // Step 4: Standard candidate evaluation and pruning.
        // If prediction matching failed, we search the cluster database ordered by descending
        // mixed probability. We compute distances and leverage Multi-Point Triangle Inequality
        // heuristics (TE4/TE5) to prune distant clusters, avoiding redundant calculations.
        // Distance evaluation: Measures distance to candidate cluster anchors. On mismatch,
        // measures/caches pairwise cluster-to-cluster distances to prune remaining candidates.
        // Output: Sets found = 1 and assigned_cluster to matched ID if found; updates
        // state->telemetry.clusters_pruned and state->scratch.current_gprobs.
        if (!found)
        {
            evaluate_standard_candidates(config, state, current_frame, temp_indices, temp_dists,
                                         &temp_count, &assigned_cluster, &found, &dfc,
                                         &k_search, verbose_candidates);
        }

        // Step 5: Handling of new cluster creation and cache limits.
        // If no existing cluster matches within 'rlim', we must create a new cluster.
        // If the max cluster capacity 'maxnbclust' is reached, we execute the configured
        // eviction strategy (Stop, Discard the oldest/smallest, or Merge the closest pair).
        // Distance evaluation: Measures and caches pairwise distances between the new
        // cluster anchor and all existing cluster anchors to populate the dccarray cache.
        // Output: Returns assigned_cluster for new cluster (or -2 to stop); increments
        // state->num_clusters, updates state->clusters, and updates prev_assigned_cluster.
        if (!found)
        {
            assigned_cluster = handle_new_cluster_creation(config, state, current_frame,
                                                           prev_assigned_cluster, temp_indices,
                                                           temp_dists, &temp_count);
            if (assigned_cluster == -2)
            {
                return -2; // Propagate stop signal
            }
        }
    }

    // Step 6: Telemetry and file serialization.
    // Record final assignment outcomes, update distance statistics, update the transition matrix,
    // and write the results to the frame membership log files if configured.
    // Output: Updates state->assignments, state->transition_matrix, ascii_out,
    // state->frame_infos, state->telemetry.total_frames_processed, and telemetry counts.
    if (assigned_cluster >= 0)
    {
        record_step_assignment(config, state, current_frame, assigned_cluster,
                               prev_assigned_cluster, ascii_out, temp_indices,
                               temp_dists, temp_count, start_pruned_val);
    }

    return assigned_cluster;
}

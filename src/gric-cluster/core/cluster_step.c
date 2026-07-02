/**
 * @file cluster_step.c
 * @brief High-level orchestration of a single image frame assignment step.
 *
 * Implements a sequential driver invoking procedurized steps to cluster
 * a single frame: setup, mixing priors, prediction matching, standard matching,
 * and eviction strategy resolution.
 *
 * Distance measurements (calls to get_dist()) are performed at:
 * - Step 3c (Distance measurement during search loop):
 *   - For predicted cluster anchors (if targeting a prediction candidate).
 *   - For standard search candidates (measured in sequence of mixed probability). If a candidate
 *     does not match, distances between cluster anchors are computed (and cached in
 *     state->scratch.dccarray) to prune other candidate clusters via triangle inequalities.
 * - Step 4 (New cluster creation): Pairwise distances between the new cluster anchor and all
 *   existing cluster anchors are measured and cached to maintain the dccarray matrix.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_step.h"
#include "cluster_steps.h"
#include "cluster_math.h"
#include "cluster_prune.h"
#include "cluster_bounds.h"
#include <stdio.h>
#include <stdlib.h>

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
    // Output: Sets state->num_clusters to 1, sets state->clusters[0], assigns
    // assigned_cluster = 0, and updates temp_indices, temp_dists, and temp_count.
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
        int first_iter = 1;
        int last_cj = -1;

        int *pred_candidates = NULL;
        int num_preds = 0;
        int current_pred_idx = 0;

        // Step 2: Retrieve prediction candidates.
        // Retrieves prediction candidates at the very start of processing the frame if
        // prediction mode is active.
        if (config->optim.pred_mode &&
            state->telemetry.total_frames_processed >= config->optim.pred_len)
        {
            pred_candidates = (int *)malloc(config->optim.pred_n * sizeof(int));
            if (pred_candidates)
            {
                num_preds = get_prediction_candidates(state, config, pred_candidates,
                                                      config->optim.pred_n);
            }
        }

        // Step 3: Iterative search loop (Prediction & Standard search).
        while (!found)
        {
            // Step 3a: Compute/update probabilities and candidate pruning.
            // On first iteration, computes the base mixed prior probabilities.
            // On subsequent iterations, prunes inconsistent candidate clusters and updates
            // geometric probabilities using the last measured target and distance.
            if (first_iter)
            {
                compute_priors_and_mixing(config, state, *prev_assigned_cluster, sorting_candidates);
                first_iter = 0;
            }
            else
            {
                update_probabilities_and_pruning(last_cj, dfc, config, state, temp_indices,
                                                 temp_dists, temp_count);
            }

            if (config->output.verbose_level >= 2 && verbose_candidates)
            {
                int vcount = 0;
                for (int i = 0; i < state->num_clusters; i++)
                {
                    if (state->scratch.clmembflag[i])
                    {
                        double p = state->scratch.mixed_probs[i];
                        if (config->optim.gprob_mode)
                        {
                            p *= state->scratch.current_gprobs[i];
                        }
                        verbose_candidates[vcount].id = i;
                        verbose_candidates[vcount].p = p;
                        vcount++;
                    }
                }

                if (vcount > 0)
                {
                    qsort(verbose_candidates, vcount, sizeof(Candidate), compare_candidates);
                    printf("  [VV] Cluster ranking:");
                    for (int i = 0; i < vcount; i++)
                    {
                        printf(" [%4d %12.5e]", verbose_candidates[i].id,
                               verbose_candidates[i].p);
                        if (i < vcount - 1)
                        {
                            printf(" >");
                        }
                    }
                    printf("\n");
                }
            }

            // Step 3b: Select next measurement target.
            // Output: Returns the cluster index cj of the next target, or -1 if all
            // candidates are pruned/exhausted.
            int cj = select_next_measurement_target(config, state, &k_search,
                                                    pred_candidates, num_preds,
                                                    &current_pred_idx);
            if (cj == -1)
            {
                break;
            }

            // Check if we are measuring a prediction candidate
            int is_prediction = 0;
            if (pred_candidates)
            {
                for (int p = 0; p < num_preds; p++)
                {
                    if (pred_candidates[p] == cj)
                    {
                        is_prediction = 1;
                        break;
                    }
                }
            }

            // Step 3c: Measure distance to target.
            // Output: Returns computed distance dfc; updates temp_indices/temp_dists and
            // increments temp_count.
            dfc = measure_distance_to_cluster(cj, current_frame, config, state,
                                              temp_indices, temp_dists, &temp_count,
                                              is_prediction);

            // Step 3d: Check if solved.
            // Output: If dfc < rlim, resolves assignment and exits loop. Otherwise, records
            // last_cj/dfc for Step 3a update.
            if (dfc < config->algo.rlim)
            {
                assigned_cluster = cj;
                found = 1;
                break;
            }
            else
            {
#ifdef _OPENMP
#pragma omp atomic
#endif
                state->telemetry.cluster_query_counts[cj]++;
            }

            last_cj = cj;
        }

        if (pred_candidates)
        {
            free(pred_candidates);
        }

        // Step 4: Handling of new cluster creation and cache limits.
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

    // Step 5: Telemetry and file serialization.
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

    if (config->optim.sparse_dcc_mode && config->optim.sparse_dcc_extra_evals > 0)
    {
        refine_sparse_bounds(config, state);
    }

    return assigned_cluster;
}

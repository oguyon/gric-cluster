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
 *     does not match, distances between cluster anchors are computed to tighten DCC
 *     bounds (dcc_min/dcc_max, tracked by dcc_measured) and prune other candidate
 *     clusters via triangle inequalities.
 * - Step 4 (New cluster creation): Pairwise distances between the new cluster anchor and all
 *   existing cluster anchors are measured and cached to maintain DCC bounds.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_step.h"
#include "cluster_steps.h"
#include "cluster_math.h"
#include "cluster_prune.h"
#include "cluster_bounds.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/**
 * cluster_frame() - Process one frame through the full clustering
 *                   pipeline (Steps 1-5).
 * @config:              Clustering configuration (algorithm, optim, I/O).
 * @state:               Mutable clustering state (clusters, telemetry,
 *                       scratch buffers).
 * @current_frame:       Pixel data of the frame to assign.
 * @prev_assigned_cluster: In/out pointer to the previously assigned
 *                       cluster index; updated on new assignment.
 * @ascii_out:           Open file handle for membership text log
 *                       (may be NULL).
 * @temp_indices:        Scratch array recording cluster indices
 *                       measured this frame.
 * @temp_dists:          Scratch array recording distances measured
 *                       this frame.
 * @sorting_candidates:  Scratch array for candidate sorting.
 * @verbose_candidates:  Scratch array for verbose-mode ranking
 *                       (may be NULL).
 *
 * Executes the sequential steps: base-case setup, prediction
 * retrieval, iterative search with pruning and measurement,
 * new-cluster creation / eviction, and telemetry recording.
 *
 * Return: Assigned cluster index (>= 0), or -2 to signal stop.
 */
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
    long start_dist_calls = state->telemetry.framedist_calls;
    long start_dfc_calls = state->telemetry.framedist_calls_sample;
    long start_dcc_calls = state->telemetry.framedist_calls_intercluster;
    int  temp_count = 0;

    // Step 1: Base case setup.
    // If no clusters exist yet, the very first ingested frame serves as the anchor frame
    // for Cluster 0, initializing our clustering space.
    // Output: Sets state->num_clusters to 1, sets state->clusters[0], assigns
    // assigned_cluster = 0, and updates temp_indices, temp_dists, and temp_count.
    if (state->num_clusters == 0)
    {
        struct timespec step_start, step_end;
        clock_gettime(CLOCK_MONOTONIC, &step_start);
        initialize_initial_cluster(config, state, current_frame, &assigned_cluster);
        clock_gettime(CLOCK_MONOTONIC, &step_end);
        state->telemetry.time_step_1 += (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                                        (step_end.tv_nsec - step_start.tv_nsec) / 1000000.0;
        state->telemetry.last_assignment_dist = 0.0;
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
        int meas_idx = 0;  /* measurement depth within this frame */

        int *pred_candidates = NULL;
        int num_preds = 0;
        int current_pred_idx = 0;

        // Step 2: Retrieve prediction candidates.
        // Retrieves prediction candidates at the very start of processing the frame if
        // prediction mode is active.
        struct timespec s2_start, s2_end;
        clock_gettime(CLOCK_MONOTONIC, &s2_start);
        if (config->optim.pred_mode &&
            state->telemetry.total_frames_processed >= config->optim.pred_len)
        {
            int *local_candidates = (int *)malloc((size_t)config->optim.pred_n * sizeof(int));
            int num_local = 0;
            if (local_candidates != NULL)
            {
                num_local = get_prediction_candidates(state, config, local_candidates,
                                                      config->optim.pred_n);
            }

            pred_candidates = (int *)malloc((size_t)config->optim.pred_n * sizeof(int));
            if (pred_candidates != NULL)
            {
                if (num_local == 1)
                {
                    /* Unambiguous local prediction: prioritize it first */
                    pred_candidates[num_preds++] = local_candidates[0];

                    /* Append joint predictions as fallback */
                    if (state->scratch.tuple_pred_count > 0)
                    {
                        for (int j = 0; j < state->scratch.tuple_pred_count &&
                             num_preds < config->optim.pred_n; j++)
                        {
                            int jc = state->scratch.tuple_pred_candidates[j];
                            if (jc != local_candidates[0])
                            {
                                pred_candidates[num_preds++] = jc;
                            }
                        }
                    }
                }
                else
                {
                    /* Ambiguous or no local match: prioritize joint predictions to resolve it */
                    if (state->scratch.tuple_pred_count > 0)
                    {
                        int n_out = (state->scratch.tuple_pred_count < config->optim.pred_n) ?
                                    state->scratch.tuple_pred_count : config->optim.pred_n;
                        for (int i = 0; i < n_out; i++)
                        {
                            pred_candidates[num_preds++] = state->scratch.tuple_pred_candidates[i];
                        }
                    }
                    if (num_preds < config->optim.pred_n && num_local > 0)
                    {
                        for (int i = 0; i < num_local && num_preds < config->optim.pred_n; i++)
                        {
                            int lc = local_candidates[i];
                            int dup = 0;
                            for (int k = 0; k < num_preds; k++)
                            {
                                if (pred_candidates[k] == lc)
                                {
                                    dup = 1;
                                    break;
                                }
                            }
                            if (!dup)
                            {
                                pred_candidates[num_preds++] = lc;
                            }
                        }
                    }
                }
            }
            if (local_candidates != NULL)
            {
                free(local_candidates);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &s2_end);
        state->telemetry.time_step_2 += (s2_end.tv_sec - s2_start.tv_sec) * 1000.0 +
                                        (s2_end.tv_nsec - s2_start.tv_nsec) / 1000000.0;

        // Step 3: Iterative search loop (Prediction & Standard search).
        while (!found)
        {
            // Step 3a: Compute/update probabilities and candidate pruning.
            // On first iteration, computes the base mixed prior probabilities.
            // On subsequent iterations, prunes inconsistent candidate clusters and updates
            // geometric probabilities using the last measured target and distance.
            if (first_iter)
            {
                struct timespec step_start, step_end;
                clock_gettime(CLOCK_MONOTONIC, &step_start);
                compute_priors_and_mixing(config, state, *prev_assigned_cluster, sorting_candidates);
                clock_gettime(CLOCK_MONOTONIC, &step_end);
                state->telemetry.time_step_3a += (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                                                 (step_end.tv_nsec - step_start.tv_nsec) / 1000000.0;
                first_iter = 0;
            }
            else
            {
                struct timespec step_start, step_end;
                clock_gettime(CLOCK_MONOTONIC, &step_start);
                update_probabilities_and_pruning(last_cj, dfc, config, state, temp_indices,
                                                 temp_dists, temp_count);
                clock_gettime(CLOCK_MONOTONIC, &step_end);
                state->telemetry.time_step_3a += (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                                                 (step_end.tv_nsec - step_start.tv_nsec) / 1000000.0;
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
            struct timespec s3b_start, s3b_end;
            clock_gettime(CLOCK_MONOTONIC, &s3b_start);
            int cj = select_next_measurement_target(config, state, &k_search,
                                                    pred_candidates, num_preds,
                                                    &current_pred_idx,
                                                    meas_idx);
            meas_idx++;
            clock_gettime(CLOCK_MONOTONIC, &s3b_end);
            state->telemetry.time_step_3b += (s3b_end.tv_sec - s3b_start.tv_sec) * 1000.0 +
                                             (s3b_end.tv_nsec - s3b_start.tv_nsec) / 1000000.0;
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
            struct timespec s3c_start, s3c_end;
            clock_gettime(CLOCK_MONOTONIC, &s3c_start);
            dfc = measure_distance_to_cluster(cj, current_frame, config, state,
                                              temp_indices, temp_dists, &temp_count,
                                              is_prediction);
            clock_gettime(CLOCK_MONOTONIC, &s3c_end);
            state->telemetry.time_step_3c += (s3c_end.tv_sec - s3c_start.tv_sec) * 1000.0 +
                                             (s3c_end.tv_nsec - s3c_start.tv_nsec) / 1000000.0;

            // Step 3d: Check if solved.
            // Output: If dfc < rlim, resolves assignment and exits loop. Otherwise, records
            // last_cj/dfc for Step 3a update.
            if (dfc < config->algo.rlim)
            {
                assigned_cluster = cj;
                state->telemetry.last_assignment_dist = dfc;
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
        // cluster anchor and all existing cluster anchors to populate DCC bounds.
        // Output: Returns assigned_cluster for new cluster (or -2 to stop); increments
        // state->num_clusters, updates state->clusters, and updates prev_assigned_cluster.
        if (!found)
        {
            struct timespec s4_start, s4_end;
            clock_gettime(CLOCK_MONOTONIC, &s4_start);
            assigned_cluster = handle_new_cluster_creation(config, state, current_frame,
                                                           prev_assigned_cluster, temp_indices,
                                                           temp_dists, &temp_count);
            clock_gettime(CLOCK_MONOTONIC, &s4_end);
            state->telemetry.time_step_4 += (s4_end.tv_sec - s4_start.tv_sec) * 1000.0 +
                                            (s4_end.tv_nsec - s4_start.tv_nsec) / 1000000.0;
            if (assigned_cluster == -2)
            {
                return -2; // Propagate stop signal
            }
            state->telemetry.last_assignment_dist = 0.0;
        }
    }

    // Step 5: Telemetry and file serialization.
    // Record final assignment outcomes, update distance statistics, update the transition matrix,
    // and write the results to the frame membership log files if configured.
    // Output: Updates state->assignments, state->transition_matrix, ascii_out,
    // state->frame_infos, state->telemetry.total_frames_processed, and telemetry counts.
    if (assigned_cluster >= 0)
    {
        struct timespec s5_start, s5_end;
        clock_gettime(CLOCK_MONOTONIC, &s5_start);
        record_step_assignment(config, state, current_frame, assigned_cluster,
                               prev_assigned_cluster, ascii_out, temp_indices,
                               temp_dists, temp_count, start_pruned_val);
        clock_gettime(CLOCK_MONOTONIC, &s5_end);
        state->telemetry.time_step_5 += (s5_end.tv_sec - s5_start.tv_sec) * 1000.0 +
                                        (s5_end.tv_nsec - s5_start.tv_nsec) / 1000000.0;
    }

    if (config->optim.sparse_dcc_mode && config->optim.sparse_dcc_extra_evals > 0)
    {
        struct timespec sr_start, sr_end;
        clock_gettime(CLOCK_MONOTONIC, &sr_start);
        refine_sparse_bounds(config, state);
        clock_gettime(CLOCK_MONOTONIC, &sr_end);
        state->telemetry.time_step_refine += (sr_end.tv_sec - sr_start.tv_sec) * 1000.0 +
                                             (sr_end.tv_nsec - sr_start.tv_nsec) / 1000000.0;
    }

    state->telemetry.last_frame_dists = state->telemetry.framedist_calls - start_dist_calls;
    state->telemetry.last_frame_dfc = state->telemetry.framedist_calls_sample - start_dfc_calls;
    state->telemetry.last_frame_dcc = state->telemetry.framedist_calls_intercluster - start_dcc_calls;

    return assigned_cluster;
}

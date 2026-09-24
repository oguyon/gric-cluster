/**
 * @file cluster_step.c
 * @brief High-level orchestration of a single image frame assignment step.
 *
 * Implements a sequential driver invoking procedurized steps to cluster
 * a single frame:
 * - Step 1: Base-case initialization (first cluster anchor frame).
 * - Step 2: Prediction candidate retrieval from temporal trajectory history.
 * - Step 3 Fast-Path: Quantized memoization cache lookup and direct candidate test.
 * - Step 3: Iterative search loop:
 *   - Step 3a: Prior mixing, geometric probability updates, and quant pruning.
 *   - Step 3b: Measurement target selection (maximum information gain).
 *   - Step 3c: Exact full-dimensional distance measurement to target anchor.
 *   - Step 3d: Threshold check against radius limit (rlim).
 * - Step 4: New cluster creation and capacity management (eviction strategies).
 * - Step 5: Telemetry recording, transition matrix updates, and file serialization.
 */

#define _POSIX_C_SOURCE 200809L

#include "cluster_step.h"
#include "cluster_step_prep.h"
#include "cluster_quant_filter.h"
#include "cluster_steps.h"
#include "cluster_math.h"
#include "cluster_bounds.h"
#include "quant_memo.h"
#include "gric_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


/**
 * cluster_step_check_memo_cache() - Check quantized memoization cache for direct match.
 * @config:            Active clustering configuration.
 * @state:             Mutable clustering state.
 * @current_frame:     Incoming frame to match.
 * @out_memo_h:        Output hash key of current frame.
 * @out_memo_h_aux:    Output auxiliary hash key of current frame.
 * @temp_indices:      Scratch array for cluster evaluation indices.
 * @temp_dists:        Scratch array for cluster evaluation distances.
 * @temp_count:        In/out pointer to number of evaluated distances.
 * @last_cj:           Output last measured cluster ID if verification failed.
 * @dfc:               Output distance to last measured cluster.
 * @need_prune_update: Output flag indicating subsequent pruning is needed.
 * @assigned_cluster:  Output cluster ID if memoization hit succeeded.
 *
 * Purpose & Context ("What is this used for?"):
 * Fast-path Step 3 optimization. Hashes the quantized frame representation (EQ16 or SQ16)
 * and probes the memoization cache table. If a previous identical or near-identical frame
 * was mapped, verifies the cluster distance within rlim and assigns immediately.
 *
 * Return: 1 if cache hit matched within radius limit, 0 otherwise.
 */
static int cluster_step_check_memo_cache(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    uint64_t      *out_memo_h,
    uint64_t      *out_memo_h_aux,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int           *last_cj,
    double        *dfc,
    int           *need_prune_update,
    int           *assigned_cluster)
{
    if (!(((config->optim.use_eq16 && state->current_frame_eq16 != NULL) ||
           (config->optim.use_sq16 && state->current_frame_sq16 != NULL)) &&
          config->optim.use_memo))
    {
        return 0;
    }

    const int16_t *memo_buf = (config->optim.use_eq16 && state->current_frame_eq16 != NULL)
                              ? state->current_frame_eq16
                              : state->current_frame_sq16;
    long fdim = (config->optim.use_eq16 && state->current_frame_eq16 != NULL)
                ? config->optim.eq16_params.dim
                : config->optim.sq16_params.dim;

    uint64_t memo_h = gric_hash_i16(memo_buf, fdim);
    uint64_t memo_h_aux = gric_hash_bytes(memo_buf,
                                         (size_t)fdim * sizeof(int16_t),
                                         GRIC_HASH_SEED ^ 0x517cc1b727220a95ULL);
    *out_memo_h = memo_h;
    *out_memo_h_aux = memo_h_aux;

    int   memo_cid = -1;
    float memo_adist = 0.0f;
    int   memo_needs_verify = 0;

    if (quant_memo_lookup(&state->scratch.memo_table,
                          memo_h, memo_h_aux,
                          &memo_cid, &memo_adist,
                          &memo_needs_verify))
    {
        if (memo_cid >= 0 && memo_cid < state->num_clusters)
        {
            int valid = 1;
            if (memo_needs_verify)
            {
                double d_check = measure_distance_to_cluster(
                    memo_cid, current_frame, config, state,
                    temp_indices, temp_dists, temp_count, 0);
                if (d_check >= config->algo.rlim)
                {
                    valid = 0;
                    *last_cj = memo_cid;
                    *dfc = d_check;
                    *need_prune_update = 1;
                }
                else
                {
                    memo_adist = (float)d_check;
                }
            }

            if (valid)
            {
                *assigned_cluster = memo_cid;
                state->telemetry.last_assignment_dist = (double)memo_adist;
                state->telemetry.memo_hits = state->scratch.memo_table.hit_count;
                state->telemetry.memo_lookups = state->scratch.memo_table.lookup_count;
                state->telemetry.memo_cache_entries = state->scratch.memo_table.entry_count;
                return 1;
            }
        }
    }

    state->telemetry.memo_hits = state->scratch.memo_table.hit_count;
    state->telemetry.memo_lookups = state->scratch.memo_table.lookup_count;
    state->telemetry.memo_cache_entries = state->scratch.memo_table.entry_count;
    return 0;
}

/**
 * cluster_step_check_predictions() - Directly test predicted cluster candidates.
 * @config:            Active clustering configuration.
 * @state:             Mutable clustering state.
 * @current_frame:     Incoming frame to match.
 * @pred_candidates:   Array of candidate cluster indices from prediction.
 * @num_preds:         Number of prediction candidates.
 * @current_pred_idx:  In/out pointer to index of next candidate in list.
 * @eq16_adc_cutoff:   Quantization pruning cutoff threshold for EQ16 ADC.
 * @eq16_ssd_thresh:   Quantization pruning threshold for EQ16 SSD.
 * @sq16_ssd_thresh:   Quantization pruning threshold for SQ16 SSD.
 * @temp_indices:      Scratch array for cluster evaluation indices.
 * @temp_dists:        Scratch array for cluster evaluation distances.
 * @temp_count:        In/out pointer to number of evaluated distances.
 * @meas_idx:          In/out measurement sequence counter.
 * @last_cj:           Output last measured cluster ID.
 * @dfc:               Output distance to last measured cluster.
 * @need_prune_update: Output flag indicating subsequent pruning is needed.
 * @assigned_cluster:  Output cluster ID if a candidate matched within rlim.
 *
 * Purpose & Context ("What is this used for?"):
 * Fast-path Step 3 verification. Tests predicted candidate clusters in order of likelihood
 * before initiating the full iterative search. If any prediction matches within rlim,
 * assignment completes without evaluating other clusters.
 *
 * Return: 1 if a prediction matched within radius limit, 0 otherwise.
 */
static int cluster_step_check_predictions(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    const int     *pred_candidates,
    int            num_preds,
    int           *current_pred_idx,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int           *meas_idx,
    int           *last_cj,
    double        *dfc,
    int           *need_prune_update,
    int           *assigned_cluster)
{
    while (*current_pred_idx < num_preds)
    {
        int cj = pred_candidates[(*current_pred_idx)++];
        if (cj < 0 || cj >= state->num_clusters)
        {
            continue;
        }

        if (cluster_candidate_is_pruned_by_quant(cj, config, state,
                                                 eq16_adc_cutoff,
                                                 eq16_ssd_thresh,
                                                 sq16_ssd_thresh))
        {
            continue;
        }

        struct timespec s3c_start, s3c_end;
        clock_gettime(CLOCK_MONOTONIC, &s3c_start);
        *dfc = measure_distance_to_cluster(cj, current_frame, config, state,
                                          temp_indices, temp_dists, temp_count, 1);
        (*meas_idx)++;
        clock_gettime(CLOCK_MONOTONIC, &s3c_end);
        state->telemetry.time_step_3c +=
            (s3c_end.tv_sec - s3c_start.tv_sec) * 1000.0 +
            (s3c_end.tv_nsec - s3c_start.tv_nsec) / 1000000.0;

        if (*dfc < config->algo.rlim)
        {
            *assigned_cluster = cj;
            state->telemetry.last_assignment_dist = *dfc;
            return 1;
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        state->telemetry.cluster_query_counts[cj]++;
        *last_cj = cj;
        *need_prune_update = 1;
    }

    return 0;
}

/**
 * cluster_step_search_loop() - Iteratively prune candidates and select next measurement.
 * @config:                Active clustering configuration.
 * @state:                 Mutable clustering state.
 * @current_frame:         Incoming frame to match.
 * @prev_assigned_cluster: Previously assigned cluster index.
 * @pred_candidates:       Array of candidate cluster indices from prediction.
 * @num_preds:             Number of prediction candidates.
 * @current_pred_idx:      In/out pointer to index of next candidate in list.
 * @eq16_adc_cutoff:       Quantization pruning cutoff threshold for EQ16 ADC.
 * @eq16_ssd_thresh:       Quantization pruning threshold for EQ16 SSD.
 * @sq16_ssd_thresh:       Quantization pruning threshold for SQ16 SSD.
 * @temp_indices:          Scratch array for cluster evaluation indices.
 * @temp_dists:            Scratch array for cluster evaluation distances.
 * @temp_count:            In/out pointer to number of evaluated distances.
 * @sorting_candidates:    Scratch buffer for candidate sorting.
 * @verbose_candidates:    Scratch buffer for verbose diagnostic printing.
 * @meas_idx:              Current measurement sequence counter.
 * @last_cj:               Last measured cluster ID.
 * @dfc:                   Distance to last measured cluster.
 * @need_prune_update:     Flag indicating subsequent pruning is needed.
 * @assigned_cluster:      Output cluster ID if assignment is found.
 *
 * Purpose & Context ("What is this used for?"):
 * Step 3 iterative candidate reduction and target evaluation loop. Integrates Bayesian priors,
 * transition matrix mixing, geometric bounds pruning, and entropy target selection to evaluate
 * clusters until a match within rlim is found or all candidates are eliminated.
 *
 * Return: 1 if frame matched an existing cluster, 0 if all candidates were eliminated.
 */
static int cluster_step_search_loop(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int            prev_assigned_cluster,
    int           *pred_candidates,
    int            num_preds,
    int           *current_pred_idx,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    Candidate     *sorting_candidates,
    Candidate     *verbose_candidates,
    int            meas_idx,
    int            last_cj,
    double         dfc,
    int            need_prune_update,
    int           *assigned_cluster)
{
    int first_iter = 1;
    int k_search = 0;

    while (1)
    {
        if (first_iter)
        {
            struct timespec step_start, step_end;
            clock_gettime(CLOCK_MONOTONIC, &step_start);

            int tm_active = (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1);
            int fast_eq16 = (config->optim.use_eq16 &&
                             (state->current_frame_eq16 != NULL ||
                              state->current_frame_eq16_adc != NULL) &&
                             state->anchor_matrix_eq16 != NULL &&
                             config->optim.pred_mode != 2 &&
                             !tm_active &&
                             config->optim.gprob_mode &&
                             !state->trace);
            int fast_sq16 = (config->optim.use_sq16 &&
                             state->current_frame_sq16 != NULL &&
                             state->anchor_matrix_sq16 != NULL &&
                             config->optim.pred_mode != 2 &&
                             !tm_active &&
                             config->optim.gprob_mode &&
                             !state->trace);

            if (need_prune_update && last_cj >= 0)
            {
                int num_cl = state->num_clusters;
                if (state->scratch.cluster_probs != NULL)
                {
                    cluster_normalize_probs(state->scratch.cluster_probs, num_cl);
                }
                const double *probs = state->scratch.cluster_probs;
                state->scratch.num_active_clusters = num_cl;
                for (int i = 0; i < num_cl; i++)
                {
                    state->scratch.clmembflag[i] = 1;
                    state->scratch.active_clusters[i] = i;
                    double p = probs ? probs[i] : state->clusters[i].prob;
                    state->scratch.current_gprobs[i] = 1.0;
                    state->scratch.mixed_probs[i] = p;
                    state->scratch.entropy_p_current[i] = p;
                }

                struct timespec t_pr_s, t_pr_e;
                clock_gettime(CLOCK_MONOTONIC, &t_pr_s);
                if (last_cj < num_cl)
                {
                    state->scratch.clmembflag[last_cj] = 0;
                }
                update_probabilities_and_pruning(
                    last_cj, dfc, config, state, temp_indices,
                    temp_dists, *temp_count);
                clock_gettime(CLOCK_MONOTONIC, &t_pr_e);
                double el_pr = (t_pr_e.tv_sec - t_pr_s.tv_sec) * 1000.0 +
                               (t_pr_e.tv_nsec - t_pr_s.tv_nsec) / 1000000.0;
                state->telemetry.time_step_3a_subsequent += el_pr;
                need_prune_update = 0;

                cluster_quant_filter_subsequent(config, state, eq16_adc_cutoff,
                                                eq16_ssd_thresh, sq16_ssd_thresh);
            }
            else
            {
                cluster_quant_filter_initial(config, state, sorting_candidates,
                                             prev_assigned_cluster, fast_eq16, fast_sq16,
                                             eq16_adc_cutoff, eq16_ssd_thresh,
                                             sq16_ssd_thresh, step_start);
            }

            clock_gettime(CLOCK_MONOTONIC, &step_end);
            state->telemetry.time_step_3a +=
                (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                (step_end.tv_nsec - step_start.tv_nsec) / 1000000.0;
            first_iter = 0;

            if (need_prune_update && last_cj >= 0)
            {
                if (last_cj < state->num_clusters)
                {
                    state->scratch.clmembflag[last_cj] = 0;
                }
                update_probabilities_and_pruning(last_cj, dfc, config, state, temp_indices,
                                                 temp_dists, *temp_count);
                need_prune_update = 0;
            }
        }
        else if (need_prune_update && last_cj >= 0)
        {
            struct timespec step_start, step_end;
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            update_probabilities_and_pruning(last_cj, dfc, config, state, temp_indices,
                                             temp_dists, *temp_count);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            double elapsed = (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                             (step_end.tv_nsec - step_start.tv_nsec) / 1000000.0;
            state->telemetry.time_step_3a += elapsed;
            state->telemetry.time_step_3a_subsequent += elapsed;
            need_prune_update = 0;
        }

        if (state->cross_tile_hook != NULL)
        {
            state->cross_tile_hook(state, state->cross_tile_ctx);
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

        struct timespec s3b_start, s3b_end;
        clock_gettime(CLOCK_MONOTONIC, &s3b_start);
        int cj = select_next_measurement_target(config, state, &k_search,
                                                pred_candidates, num_preds,
                                                current_pred_idx,
                                                meas_idx);
        meas_idx++;
        clock_gettime(CLOCK_MONOTONIC, &s3b_end);
        state->telemetry.time_step_3b += (s3b_end.tv_sec - s3b_start.tv_sec) * 1000.0 +
                                         (s3b_end.tv_nsec - s3b_start.tv_nsec) / 1000000.0;
        if (cj == -1)
        {
            break;
        }

        if (cluster_candidate_is_pruned_by_quant(cj, config, state,
                                                 eq16_adc_cutoff,
                                                 eq16_ssd_thresh,
                                                 sq16_ssd_thresh))
        {
            continue;
        }

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

        struct timespec s3c_start, s3c_end;
        clock_gettime(CLOCK_MONOTONIC, &s3c_start);
        dfc = measure_distance_to_cluster(cj, current_frame, config, state,
                                          temp_indices, temp_dists, temp_count,
                                          is_prediction);
        clock_gettime(CLOCK_MONOTONIC, &s3c_end);
        state->telemetry.time_step_3c += (s3c_end.tv_sec - s3c_start.tv_sec) * 1000.0 +
                                         (s3c_end.tv_nsec - s3c_start.tv_nsec) / 1000000.0;

        if (dfc < config->algo.rlim)
        {
            *assigned_cluster = cj;
            state->telemetry.last_assignment_dist = dfc;
            return 1;
        }

#ifdef _OPENMP
#pragma omp atomic
#endif
        state->telemetry.cluster_query_counts[cj]++;

        last_cj = cj;
        need_prune_update = 1;
    }

    return 0;
}

/**
 * cluster_step_finalize_frame() - Record telemetry, update memo table, and refine bounds.
 * @config:                Active clustering configuration.
 * @state:                 Mutable clustering state.
 * @current_frame:         Frame assigned or created this step.
 * @assigned_cluster:      Index of assigned cluster.
 * @prev_assigned_cluster: In/out pointer to previously assigned cluster index.
 * @ascii_out:             Text membership log handle.
 * @temp_indices:          Array of cluster indices measured this step.
 * @temp_dists:            Array of cluster distances measured this step.
 * @temp_count:            Number of measured distances this step.
 * @start_pruned_val:      Telemetry count of pruned clusters before this step.
 * @memo_lookup_done:      Non-zero if current frame was hashed for memoization.
 * @memo_h:                Primary hash key of quantized frame.
 * @memo_h_aux:            Auxiliary hash key of quantized frame.
 * @start_dist_calls:      Telemetry count of total distance calls before this step.
 * @start_dfc_calls:       Telemetry count of sample distance calls before this step.
 * @start_dcc_calls:       Telemetry count of inter-cluster calls before this step.
 *
 * Purpose & Context ("What is this used for?"):
 * Step 5 telemetry and serialization driver. Writes membership assignment logs, registers
 * newly resolved frame-to-cluster mappings into the SQ16/EQ16 memoization table, updates
 * pairwise inter-cluster bounds in sparse DCC mode, and computes per-step telemetry deltas.
 */
static void cluster_step_finalize_frame(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int            assigned_cluster,
    int           *prev_assigned_cluster,
    FILE          *ascii_out,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count,
    long           start_pruned_val,
    int            memo_lookup_done,
    uint64_t       memo_h,
    uint64_t       memo_h_aux,
    long           start_dist_calls,
    long           start_dfc_calls,
    long           start_dcc_calls)
{
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

        if (memo_lookup_done)
        {
            quant_memo_insert(&state->scratch.memo_table,
                              memo_h, memo_h_aux,
                              assigned_cluster,
                              (float)state->telemetry.last_assignment_dist,
                              (uint32_t)state->telemetry.total_frames_processed);
            state->telemetry.memo_cache_entries = state->scratch.memo_table.entry_count;
        }
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
    state->telemetry.last_frame_dcc =
        state->telemetry.framedist_calls_intercluster - start_dcc_calls;

    int K = state->num_clusters;
    state->telemetry.dcc_pairs_total = (K > 1) ? ((uint64_t)K * (K - 1) / 2) : 0;
}

/**
 * cluster_frame() - Process one frame through the full clustering pipeline (Steps 1-5).
 * @config:                Clustering configuration (algorithm, optim, I/O).
 * @state:                 Mutable clustering state (clusters, telemetry, scratch buffers).
 * @current_frame:         Pixel data of the frame to assign.
 * @prev_assigned_cluster: In/out pointer to previously assigned cluster index.
 * @ascii_out:             Open file handle for membership text log (may be NULL).
 * @temp_indices:          Scratch array recording cluster indices measured this frame.
 * @temp_dists:            Scratch array recording distances measured this frame.
 * @sorting_candidates:    Scratch array for candidate sorting.
 * @verbose_candidates:    Scratch array for verbose-mode ranking (may be NULL).
 *
 * Purpose & Context ("What is this used for?"):
 * Single-frame assignment engine orchestrating Steps 1-5: base case creation, prediction
 * candidate retrieval, memoization / direct fast-path testing, iterative pruning search loop,
 * capacity handling, and step telemetry serialization.
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
    int      assigned_cluster = -1;
    long     start_pruned_val = state->telemetry.clusters_pruned;
    long     start_dist_calls = state->telemetry.framedist_calls;
    long     start_dfc_calls = state->telemetry.framedist_calls_sample;
    long     start_dcc_calls = state->telemetry.framedist_calls_intercluster;
    int      temp_count = 0;
    uint64_t memo_h = 0;
    uint64_t memo_h_aux = 0;
    int      memo_lookup_done = 0;

    prepare_frame_quantization(config, state, current_frame);

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
        int    found = 0;
        double dfc = 0.0;
        int    last_cj = -1;
        int    need_prune_update = 0;
        int    meas_idx = 0;

        int *pred_candidates = state->scratch.pred_candidates;
        int  current_pred_idx = 0;
        int  first_pred = -1;

        struct timespec s2_start, s2_end;
        clock_gettime(CLOCK_MONOTONIC, &s2_start);
        int num_preds = retrieve_prediction_candidates(config, state, pred_candidates);
        clock_gettime(CLOCK_MONOTONIC, &s2_end);
        state->telemetry.time_step_2 += (s2_end.tv_sec - s2_start.tv_sec) * 1000.0 +
                                        (s2_end.tv_nsec - s2_start.tv_nsec) / 1000000.0;

        if (num_preds > 0 && pred_candidates != NULL)
        {
            first_pred = pred_candidates[0];
            state->telemetry.pred_attempts++;
            if (first_pred == *prev_assigned_cluster)
            {
                state->telemetry.pred_same_as_last++;
            }
        }

        float    eq16_adc_cutoff = 0.0f;
        uint64_t eq16_ssd_thresh = 0;
        uint64_t sq16_ssd_thresh = 0;
        cluster_compute_quant_thresholds(config, &eq16_adc_cutoff,
                                         &eq16_ssd_thresh, &sq16_ssd_thresh);

        if (((config->optim.use_eq16 && state->current_frame_eq16 != NULL) ||
             (config->optim.use_sq16 && state->current_frame_sq16 != NULL)) &&
            config->optim.use_memo)
        {
            memo_lookup_done = 1;
            found = cluster_step_check_memo_cache(config, state, current_frame,
                                                  &memo_h, &memo_h_aux,
                                                  temp_indices, temp_dists, &temp_count,
                                                  &last_cj, &dfc, &need_prune_update,
                                                  &assigned_cluster);
        }

        if (!found && num_preds > 0 && pred_candidates != NULL)
        {
            found = cluster_step_check_predictions(config, state, current_frame,
                                                   pred_candidates, num_preds,
                                                   &current_pred_idx,
                                                   eq16_adc_cutoff,
                                                   eq16_ssd_thresh,
                                                   sq16_ssd_thresh,
                                                   temp_indices, temp_dists, &temp_count,
                                                   &meas_idx, &last_cj, &dfc,
                                                   &need_prune_update,
                                                   &assigned_cluster);
        }

        if (!found)
        {
            found = cluster_step_search_loop(config, state, current_frame,
                                             *prev_assigned_cluster,
                                             pred_candidates, num_preds,
                                             &current_pred_idx,
                                             eq16_adc_cutoff,
                                             eq16_ssd_thresh,
                                             sq16_ssd_thresh,
                                             temp_indices, temp_dists, &temp_count,
                                             sorting_candidates, verbose_candidates,
                                             meas_idx, last_cj, dfc, need_prune_update,
                                             &assigned_cluster);
        }

        if (first_pred >= 0 && assigned_cluster == first_pred)
        {
            state->telemetry.pred_hits++;
        }

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
                return -2;
            }
            state->telemetry.last_assignment_dist = 0.0;
        }
    }

    cluster_step_finalize_frame(config, state, current_frame, assigned_cluster,
                                prev_assigned_cluster, ascii_out, temp_indices,
                                temp_dists, temp_count, start_pruned_val,
                                memo_lookup_done, memo_h, memo_h_aux,
                                start_dist_calls, start_dfc_calls, start_dcc_calls);

    return assigned_cluster;
}

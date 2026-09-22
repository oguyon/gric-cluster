/**
 * @file cluster_quant_filter.c
 * @brief Multi-tier quantization lower-bound candidate screening and pruning.
 *
 * Implements candidate filtering using coarse quantization (EQ16/SQ16/SQ8) lower bounds:
 * - Integer SIMD screening (SDC) with 2.0x slack factor.
 * - Exact Asymmetric Distance Computation (ADC) float refinement with 1.0x slack factor.
 * - Pruned candidates are marked in clmembflag and active_clusters is compacted.
 */

#define _POSIX_C_SOURCE 200809L

#include "cluster_quant_filter.h"
#include "cluster_steps.h"
#include "scalar_quant.h"
#include "eq16_quant.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * cluster_compute_quant_thresholds() - Computes effective cutoff thresholds for quantization.
 * @config:              Pointer to clustering configuration.
 * @out_eq16_adc_cutoff: Output pointer for EQ16 Asymmetric Distance (ADC) cutoff.
 * @out_eq16_ssd_thresh: Output pointer for EQ16 Symmetric SSD cutoff.
 * @out_sq16_ssd_thresh: Output pointer for SQ16 Symmetric SSD cutoff.
 */
void cluster_compute_quant_thresholds(
    const ClusterConfig *config,
    float               *out_eq16_adc_cutoff,
    uint64_t            *out_eq16_ssd_thresh,
    uint64_t            *out_sq16_ssd_thresh)
{
    float eq16_adc_cutoff = 0.0f;
    uint64_t eq16_ssd_thresh = 0;
    uint64_t sq16_ssd_thresh = 0;

    if (config->optim.use_sq16)
    {
        double raw_thresh = (config->algo.rlim +
                             2.0 * (double)config->optim.sq16_params.err_radius) /
                            (double)config->optim.sq16_params.scale;
        if (raw_thresh > 0.0)
        {
            sq16_ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
        }
    }

    if (config->optim.use_eq16)
    {
        double raw_thresh_sdc = (config->algo.rlim +
                                 2.0 * (double)config->optim.eq16_params.err_radius) /
                                ((double)config->optim.eq16_params.scale * 0.5);
        if (raw_thresh_sdc > 0.0)
        {
            eq16_ssd_thresh = (uint64_t)(raw_thresh_sdc * raw_thresh_sdc);
        }

        double raw_thresh_adc = (config->algo.rlim +
                                 1.0 * (double)config->optim.eq16_params.err_radius) /
                                ((double)config->optim.eq16_params.scale * 0.5);
        if (raw_thresh_adc > 0.0)
        {
            eq16_adc_cutoff = (float)(raw_thresh_adc * raw_thresh_adc);
        }
    }

    if (out_eq16_adc_cutoff != NULL)
    {
        *out_eq16_adc_cutoff = eq16_adc_cutoff;
    }
    if (out_eq16_ssd_thresh != NULL)
    {
        *out_eq16_ssd_thresh = eq16_ssd_thresh;
    }
    if (out_sq16_ssd_thresh != NULL)
    {
        *out_sq16_ssd_thresh = sq16_ssd_thresh;
    }
}

/**
 * cluster_candidate_is_pruned_by_quant() - Fast metric lower-bound test for a candidate.
 * @cj:              Cluster index to test.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state.
 * @eq16_adc_cutoff: Precomputed EQ16 ADC cutoff threshold.
 * @eq16_ssd_thresh: Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh: Precomputed SQ16 SSD threshold.
 *
 * Checks whether cluster @cj can be provably dismissed using precomputed quantization
 * lower bounds without requiring an exact full-dimensional distance calculation.
 *
 * Return: 1 if candidate is pruned by metric lower bound, 0 if it survives.
 */
int cluster_candidate_is_pruned_by_quant(
    int            cj,
    ClusterConfig *config,
    ClusterState  *state,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh)
{
    if (config->optim.use_eq16 &&
        state->clusters[cj].anchor_eq16 != NULL &&
        ((config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL) ||
         state->current_frame_eq16 != NULL))
    {
        state->telemetry.eq16_evals++;
        int is_pruned = 0;
        if (config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL)
        {
            float dsq = eq16_dist_asym_cutoff_f32(
                state->current_frame_eq16_adc,
                state->clusters[cj].anchor_eq16,
                config->optim.eq16_params.dim,
                eq16_adc_cutoff
            );
            is_pruned = (dsq > eq16_adc_cutoff);
        }
        else
        {
            uint64_t ssd = eq16_dist_squared_cutoff_i16(
                state->current_frame_eq16,
                state->clusters[cj].anchor_eq16,
                config->optim.eq16_params.dim,
                eq16_ssd_thresh
            );
            is_pruned = (ssd > eq16_ssd_thresh);
        }

        if (is_pruned)
        {
            state->telemetry.eq16_pruned++;
            state->telemetry.clusters_pruned++;
            state->scratch.clmembflag[cj] = 0;
            return 1;
        }
    }
    else if (config->optim.use_sq16 &&
             state->clusters[cj].anchor_sq16 != NULL &&
             state->current_frame_sq16 != NULL)
    {
        state->telemetry.sq16_evals++;
        uint64_t ssd = sq16_dist_squared_cutoff_i16(
            state->current_frame_sq16,
            state->clusters[cj].anchor_sq16,
            config->optim.sq16_params.dim,
            sq16_ssd_thresh
        );
        if (ssd > sq16_ssd_thresh)
        {
            state->telemetry.sq16_pruned++;
            state->telemetry.clusters_pruned++;
            state->scratch.clmembflag[cj] = 0;
            return 1;
        }
    }
    else if (config->optim.use_sq8 &&
             state->clusters[cj].anchor_sq8 != NULL &&
             state->current_frame_sq8 != NULL)
    {
        state->telemetry.sq8_evals++;
        double d_lb = sq8_compute_lower_bound(
            state->current_frame_sq8,
            state->clusters[cj].anchor_sq8,
            &config->optim.sq8_params,
            0.0
        );
        if (d_lb > config->algo.rlim)
        {
            state->telemetry.sq8_pruned++;
            state->telemetry.clusters_pruned++;
            state->scratch.clmembflag[cj] = 0;
            return 1;
        }
    }
    return 0;
}

/**
 * cluster_quant_filter_initial() - Initial matrix-level screening across all clusters.
 * @config:                Pointer to clustering configuration.
 * @state:                 Pointer to clustering state.
 * @sorting_candidates:    Scratch candidate array.
 * @prev_assigned_cluster: Index of previous cluster assignment (-1 if none).
 * @fast_eq16:             Flag indicating fast EQ16 path is enabled.
 * @fast_sq16:             Flag indicating fast SQ16 path is enabled.
 * @eq16_adc_cutoff:       Precomputed EQ16 ADC cutoff.
 * @eq16_ssd_thresh:       Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh:       Precomputed SQ16 SSD threshold.
 * @step_start:            Timestamp when Step 3a began.
 */
void cluster_quant_filter_initial(
    ClusterConfig  *config,
    ClusterState   *state,
    Candidate      *sorting_candidates,
    int             prev_assigned_cluster,
    int             fast_eq16,
    int             fast_sq16,
    float           eq16_adc_cutoff,
    uint64_t        eq16_ssd_thresh,
    uint64_t        sq16_ssd_thresh,
    struct timespec step_start)
{
    struct timespec t_mid1, t_mid2;

    if (fast_eq16)
    {
        if (state->scratch.cluster_probs != NULL)
        {
            cluster_normalize_probs(state->scratch.cluster_probs, state->num_clusters);
        }
        else
        {
            compute_priors_and_mixing(config, state, prev_assigned_cluster, sorting_candidates);
        }
        clock_gettime(CLOCK_MONOTONIC, &t_mid1);
        state->telemetry.time_step_3a_priors +=
            (t_mid1.tv_sec - step_start.tv_sec) * 1000.0 +
            (t_mid1.tv_nsec - step_start.tv_nsec) / 1000000.0;

        int num_active = 0;
        long pruned_count = 0;
        long dim = config->optim.eq16_params.dim;
        if (config->optim.use_eq16_adc &&
            state->current_frame_eq16_adc != NULL &&
            state->current_frame_eq16 != NULL)
        {
            /* Tier 1: Integer SIMD coarse screening with SDC (2.0 slack) */
            struct timespec t_t1_s, t_t1_e;
            clock_gettime(CLOCK_MONOTONIC, &t_t1_s);
            eq16_filter_anchor_matrix(
                state->current_frame_eq16,
                state->anchor_matrix_eq16,
                state->anchor_matrix_eq16_interleaved,
                state->num_clusters,
                dim,
                eq16_ssd_thresh,
                state->scratch.clmembflag,
                state->scratch.active_clusters,
                &num_active,
                &pruned_count
            );
            clock_gettime(CLOCK_MONOTONIC, &t_t1_e);
            state->telemetry.time_step_3a_sq_tier1 +=
                (t_t1_e.tv_sec - t_t1_s.tv_sec) * 1000.0 +
                (t_t1_e.tv_nsec - t_t1_s.tv_nsec) / 1000000.0;

            /* Tier 2: Exact ADC float refinement on surviving candidates (1.0 slack) */
            struct timespec t_t2_s, t_t2_e;
            clock_gettime(CLOCK_MONOTONIC, &t_t2_s);
            const float *cur_adc = state->current_frame_eq16_adc;
            const int16_t *mat_eq16 = state->anchor_matrix_eq16;
            eq16_refine_candidates_adc(
                cur_adc,
                mat_eq16,
                state->scratch.active_clusters,
                num_active,
                dim,
                eq16_adc_cutoff,
                state->scratch.clmembflag,
                &num_active,
                &pruned_count
            );
            clock_gettime(CLOCK_MONOTONIC, &t_t2_e);
            state->telemetry.time_step_3a_sq_tier2 +=
                (t_t2_e.tv_sec - t_t2_s.tv_sec) * 1000.0 +
                (t_t2_e.tv_nsec - t_t2_s.tv_nsec) / 1000000.0;
        }
        else if (config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL)
        {
            eq16_filter_anchor_matrix_adc(
                state->current_frame_eq16_adc,
                state->anchor_matrix_eq16,
                state->anchor_matrix_adc_interleaved,
                state->num_clusters,
                dim,
                eq16_adc_cutoff,
                state->scratch.clmembflag,
                state->scratch.active_clusters,
                &num_active,
                &pruned_count
            );
        }
        else
        {
            eq16_filter_anchor_matrix(
                state->current_frame_eq16,
                state->anchor_matrix_eq16,
                state->anchor_matrix_eq16_interleaved,
                state->num_clusters,
                dim,
                eq16_ssd_thresh,
                state->scratch.clmembflag,
                state->scratch.active_clusters,
                &num_active,
                &pruned_count
            );
        }
        state->scratch.num_active_clusters = num_active;
        state->telemetry.eq16_evals += state->num_clusters;
        state->telemetry.eq16_pruned += (uint64_t)pruned_count;
        state->telemetry.clusters_pruned += (uint64_t)pruned_count;

        clock_gettime(CLOCK_MONOTONIC, &t_mid2);
        state->telemetry.time_step_3a_sq_filter +=
            (t_mid2.tv_sec - t_mid1.tv_sec) * 1000.0 +
            (t_mid2.tv_nsec - t_mid1.tv_nsec) / 1000000.0;

        const double *probs = state->scratch.cluster_probs;
        for (int idx = 0; idx < num_active; idx++)
        {
            int c = state->scratch.active_clusters[idx];
            double p = probs ? probs[c] : state->clusters[c].prob;
            state->scratch.current_gprobs[c] = 1.0;
            state->scratch.mixed_probs[c] = p;
            state->scratch.entropy_p_current[c] = p;
        }
    }
    else if (fast_sq16)
    {
        if (state->scratch.cluster_probs != NULL)
        {
            cluster_normalize_probs(state->scratch.cluster_probs, state->num_clusters);
        }
        else
        {
            compute_priors_and_mixing(config, state, prev_assigned_cluster, sorting_candidates);
        }
        clock_gettime(CLOCK_MONOTONIC, &t_mid1);
        state->telemetry.time_step_3a_priors +=
            (t_mid1.tv_sec - step_start.tv_sec) * 1000.0 +
            (t_mid1.tv_nsec - step_start.tv_nsec) / 1000000.0;

        int num_active = 0;
        int pruned_count = 0;
        long dim = config->optim.sq16_params.dim;
        sq16_filter_anchor_matrix(
            state->current_frame_sq16,
            state->anchor_matrix_sq16,
            state->anchor_matrix_sq16_interleaved,
            state->num_clusters,
            dim,
            sq16_ssd_thresh,
            state->scratch.clmembflag,
            state->scratch.active_clusters,
            &num_active,
            &pruned_count
        );
        state->scratch.num_active_clusters = num_active;
        state->telemetry.sq16_evals += state->num_clusters;
        state->telemetry.sq16_pruned += pruned_count;
        state->telemetry.clusters_pruned += pruned_count;

        clock_gettime(CLOCK_MONOTONIC, &t_mid2);
        state->telemetry.time_step_3a_sq_filter +=
            (t_mid2.tv_sec - t_mid1.tv_sec) * 1000.0 +
            (t_mid2.tv_nsec - t_mid1.tv_nsec) / 1000000.0;

        const double *probs = state->scratch.cluster_probs;
        for (int idx = 0; idx < num_active; idx++)
        {
            int c = state->scratch.active_clusters[idx];
            double p = probs ? probs[c] : state->clusters[c].prob;
            state->scratch.current_gprobs[c] = 1.0;
            state->scratch.mixed_probs[c] = p;
            state->scratch.entropy_p_current[c] = p;
        }
    }
    else
    {
        compute_priors_and_mixing(config, state, prev_assigned_cluster, sorting_candidates);
        clock_gettime(CLOCK_MONOTONIC, &t_mid1);
        state->telemetry.time_step_3a_priors +=
            (t_mid1.tv_sec - step_start.tv_sec) * 1000.0 +
            (t_mid1.tv_nsec - step_start.tv_nsec) / 1000000.0;

        if (config->optim.use_eq16 &&
            ((config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL) ||
             state->current_frame_eq16 != NULL))
        {
            long dim = config->optim.eq16_params.dim;
            int use_adc = (config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL);
            const float *cur_adc = state->current_frame_eq16_adc;
            const int16_t *cur_eq16 = state->current_frame_eq16;
            const int16_t *mat_eq16 = state->anchor_matrix_eq16;
            int num_cl = state->num_clusters;

            if (mat_eq16 != NULL)
            {
                int num_active = 0;
                long pruned_count = 0;
                if (use_adc && cur_eq16 != NULL)
                {
                    /* Tier 1: Integer SIMD coarse screening with SDC (2.0 slack) */
                    eq16_filter_anchor_matrix(
                        cur_eq16,
                        mat_eq16,
                        state->anchor_matrix_eq16_interleaved,
                        num_cl,
                        dim,
                        eq16_ssd_thresh,
                        state->scratch.clmembflag,
                        state->scratch.active_clusters,
                        &num_active,
                        &pruned_count
                    );

                    /* Tier 2: ADC float refinement on surviving candidates */
                    eq16_refine_candidates_adc(
                        cur_adc,
                        mat_eq16,
                        state->scratch.active_clusters,
                        num_active,
                        dim,
                        eq16_adc_cutoff,
                        state->scratch.clmembflag,
                        &num_active,
                        &pruned_count
                    );
                }
                else if (use_adc)
                {
                    eq16_filter_anchor_matrix_adc(
                        cur_adc,
                        mat_eq16,
                        state->anchor_matrix_adc_interleaved,
                        num_cl,
                        dim,
                        eq16_adc_cutoff,
                        state->scratch.clmembflag,
                        state->scratch.active_clusters,
                        &num_active,
                        &pruned_count
                    );
                }
                else
                {
                    eq16_filter_anchor_matrix(
                        cur_eq16,
                        mat_eq16,
                        state->anchor_matrix_eq16_interleaved,
                        num_cl,
                        dim,
                        eq16_ssd_thresh,
                        state->scratch.clmembflag,
                        state->scratch.active_clusters,
                        &num_active,
                        &pruned_count
                    );
                }
                state->scratch.num_active_clusters = num_active;
                state->telemetry.eq16_evals += num_cl;
                state->telemetry.eq16_pruned += (uint64_t)pruned_count;
                state->telemetry.clusters_pruned += (uint64_t)pruned_count;
            }
            else
            {
                for (int i = 0; i < num_cl; i++)
                {
                    if (state->scratch.clmembflag[i])
                    {
                        const int16_t *a_ptr = state->clusters[i].anchor_eq16;
                        if (a_ptr != NULL)
                        {
                            state->telemetry.eq16_evals++;
                            int is_pruned = 0;
                            if (use_adc)
                            {
                                float dsq = eq16_dist_asym_cutoff_f32(
                                    cur_adc, a_ptr, dim, eq16_adc_cutoff
                                );
                                is_pruned = (dsq > eq16_adc_cutoff);
                            }
                            else
                            {
                                uint64_t ssd = eq16_dist_squared_cutoff_i16(
                                    cur_eq16, a_ptr, dim, eq16_ssd_thresh
                                );
                                is_pruned = (ssd > eq16_ssd_thresh);
                            }
                            if (is_pruned)
                            {
                                state->scratch.clmembflag[i] = 0;
                                state->telemetry.eq16_pruned++;
                                state->telemetry.clusters_pruned++;
                            }
                        }
                    }
                }

                int compact_idx = 0;
                for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
                {
                    int c = state->scratch.active_clusters[idx];
                    if (state->scratch.clmembflag[c])
                    {
                        state->scratch.active_clusters[compact_idx++] = c;
                    }
                }
                state->scratch.num_active_clusters = compact_idx;
            }
        }
        else if (config->optim.use_sq16 && state->current_frame_sq16 != NULL)
        {
            long dim = config->optim.sq16_params.dim;
            const int16_t *cur_sq16 = state->current_frame_sq16;
            const int16_t *mat_sq16 = state->anchor_matrix_sq16;
            int num_cl = state->num_clusters;

            if (mat_sq16 != NULL)
            {
                int num_active = 0;
                int pruned_count = 0;
                sq16_filter_anchor_matrix(
                    cur_sq16,
                    mat_sq16,
                    state->anchor_matrix_sq16_interleaved,
                    num_cl,
                    dim,
                    sq16_ssd_thresh,
                    state->scratch.clmembflag,
                    state->scratch.active_clusters,
                    &num_active,
                    &pruned_count
                );
                state->scratch.num_active_clusters = num_active;
                state->telemetry.sq16_evals += num_cl;
                state->telemetry.sq16_pruned += pruned_count;
                state->telemetry.clusters_pruned += pruned_count;
            }
            else
            {
                for (int i = 0; i < num_cl; i++)
                {
                    if (state->scratch.clmembflag[i])
                    {
                        const int16_t *a_ptr = state->clusters[i].anchor_sq16;
                        if (a_ptr != NULL)
                        {
                            state->telemetry.sq16_evals++;
                            uint64_t ssd = sq16_dist_squared_cutoff_i16(
                                cur_sq16, a_ptr, dim, sq16_ssd_thresh
                            );
                            if (ssd > sq16_ssd_thresh)
                            {
                                state->scratch.clmembflag[i] = 0;
                                state->telemetry.sq16_pruned++;
                                state->telemetry.clusters_pruned++;
                            }
                        }
                    }
                }

                int compact_idx = 0;
                for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
                {
                    int c = state->scratch.active_clusters[idx];
                    if (state->scratch.clmembflag[c])
                    {
                        state->scratch.active_clusters[compact_idx++] = c;
                    }
                }
                state->scratch.num_active_clusters = compact_idx;
            }
        }
        else if (config->optim.use_sq8 && state->current_frame_sq8 != NULL)
        {
            long dim = config->optim.sq8_params.dim;
            const uint8_t *cur_sq8 = state->current_frame_sq8;
            const uint8_t *mat_sq8 = state->anchor_matrix_sq8;
            int num_cl = state->num_clusters;

            if (mat_sq8 != NULL)
            {
                int num_active = 0;
                for (int i = 0; i < num_cl; i++)
                {
                    if (state->scratch.clmembflag[i])
                    {
                        const uint8_t *a_ptr = mat_sq8 + (size_t)i * (size_t)dim;
                        double d_lb = sq8_compute_lower_bound(
                            cur_sq8, a_ptr, &config->optim.sq8_params, 0.0
                        );
                        if (d_lb > config->algo.rlim)
                        {
                            state->scratch.clmembflag[i] = 0;
                            state->telemetry.sq8_pruned++;
                            state->telemetry.clusters_pruned++;
                        }
                        else
                        {
                            state->scratch.active_clusters[num_active++] = i;
                        }
                    }
                }
                state->scratch.num_active_clusters = num_active;
                state->telemetry.sq8_evals += num_cl;
            }
            else
            {
                for (int i = 0; i < num_cl; i++)
                {
                    if (state->scratch.clmembflag[i])
                    {
                        const uint8_t *a_ptr = state->clusters[i].anchor_sq8;
                        if (a_ptr != NULL)
                        {
                            state->telemetry.sq8_evals++;
                            double d_lb = sq8_compute_lower_bound(
                                cur_sq8, a_ptr, &config->optim.sq8_params, 0.0
                            );
                            if (d_lb > config->algo.rlim)
                            {
                                state->scratch.clmembflag[i] = 0;
                                state->telemetry.sq8_pruned++;
                                state->telemetry.clusters_pruned++;
                            }
                        }
                    }
                }

                int compact_idx = 0;
                for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
                {
                    int c = state->scratch.active_clusters[idx];
                    if (state->scratch.clmembflag[c])
                    {
                        state->scratch.active_clusters[compact_idx++] = c;
                    }
                }
                state->scratch.num_active_clusters = compact_idx;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_mid2);
        state->telemetry.time_step_3a_sq_filter +=
            (t_mid2.tv_sec - t_mid1.tv_sec) * 1000.0 +
            (t_mid2.tv_nsec - t_mid1.tv_nsec) / 1000000.0;
    }
}

/**
 * cluster_quant_filter_subsequent() - Filters surviving candidate clusters.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state.
 * @eq16_adc_cutoff: Precomputed EQ16 ADC cutoff.
 * @eq16_ssd_thresh: Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh: Precomputed SQ16 SSD threshold.
 *
 * Filters ONLY the surviving candidates in state->scratch.active_clusters using
 * EQ16/SQ16/SQ8 lower bounds, compacts the active array, and re-normalizes entropy_p_current.
 */
void cluster_quant_filter_subsequent(
    ClusterConfig *config,
    ClusterState  *state,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh)
{
    struct timespec t_sq_s, t_sq_e;
    clock_gettime(CLOCK_MONOTONIC, &t_sq_s);

    if (config->optim.use_eq16 &&
        ((config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL) ||
         state->current_frame_eq16 != NULL))
    {
        long dim = config->optim.eq16_params.dim;
        int use_adc = (config->optim.use_eq16_adc && state->current_frame_eq16_adc != NULL);
        const float *cur_adc = state->current_frame_eq16_adc;
        const int16_t *cur_eq16 = state->current_frame_eq16;
        const int16_t *mat_eq16 = state->anchor_matrix_eq16;
        int act_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int compact_cnt = 0;
        int local_pruned = 0;

        state->telemetry.eq16_evals += act_cnt;

        for (int idx = 0; idx < act_cnt; idx++)
        {
            int c = act[idx];
            const int16_t *a_ptr = mat_eq16
                ? (mat_eq16 + (size_t)c * (size_t)dim)
                : state->clusters[c].anchor_eq16;
            if (a_ptr != NULL)
            {
                int is_pruned = 0;
                if (use_adc)
                {
                    float dsq = eq16_dist_asym_cutoff_f32(
                        cur_adc, a_ptr, dim, eq16_adc_cutoff
                    );
                    is_pruned = (dsq > eq16_adc_cutoff);
                }
                else
                {
                    uint64_t ssd = eq16_dist_squared_cutoff_i16(
                        cur_eq16, a_ptr, dim, eq16_ssd_thresh
                    );
                    is_pruned = (ssd > eq16_ssd_thresh);
                }

                if (is_pruned)
                {
                    state->scratch.clmembflag[c] = 0;
                    state->scratch.entropy_p_current[c] = 0.0;
                    local_pruned++;
                    continue;
                }
            }
            act[compact_cnt++] = c;
        }
        state->scratch.num_active_clusters = compact_cnt;
        state->telemetry.eq16_pruned += local_pruned;
        state->telemetry.clusters_pruned += local_pruned;

        if (local_pruned > 0)
        {
            double sum_p = 0.0;
            for (int idx = 0; idx < compact_cnt; idx++)
            {
                sum_p += state->scratch.entropy_p_current[act[idx]];
            }
            if (sum_p > 0.0)
            {
                double inv_sum_p = 1.0 / sum_p;
                for (int idx = 0; idx < compact_cnt; idx++)
                {
                    state->scratch.entropy_p_current[act[idx]] *= inv_sum_p;
                }
            }
        }
    }
    else if (config->optim.use_sq16 && state->current_frame_sq16 != NULL)
    {
        long dim = config->optim.sq16_params.dim;
        const int16_t *cur_sq16 = state->current_frame_sq16;
        const int16_t *mat_sq16 = state->anchor_matrix_sq16;
        int act_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int compact_cnt = 0;
        int local_pruned = 0;

        state->telemetry.sq16_evals += act_cnt;

        for (int idx = 0; idx < act_cnt; idx++)
        {
            int c = act[idx];
            const int16_t *a_ptr = mat_sq16
                ? (mat_sq16 + (size_t)c * (size_t)dim)
                : state->clusters[c].anchor_sq16;
            if (a_ptr != NULL)
            {
                uint64_t ssd = sq16_dist_squared_cutoff_i16(
                    cur_sq16, a_ptr, dim, sq16_ssd_thresh
                );
                if (ssd > sq16_ssd_thresh)
                {
                    state->scratch.clmembflag[c] = 0;
                    state->scratch.entropy_p_current[c] = 0.0;
                    local_pruned++;
                    continue;
                }
            }
            act[compact_cnt++] = c;
        }
        state->scratch.num_active_clusters = compact_cnt;
        state->telemetry.sq16_pruned += local_pruned;
        state->telemetry.clusters_pruned += local_pruned;

        if (local_pruned > 0)
        {
            double sum_p = 0.0;
            for (int idx = 0; idx < compact_cnt; idx++)
            {
                sum_p += state->scratch.entropy_p_current[act[idx]];
            }
            if (sum_p > 0.0)
            {
                double inv_sum_p = 1.0 / sum_p;
                for (int idx = 0; idx < compact_cnt; idx++)
                {
                    state->scratch.entropy_p_current[act[idx]] *= inv_sum_p;
                }
            }
        }
    }
    else if (config->optim.use_sq8 && state->current_frame_sq8 != NULL)
    {
        long dim = config->optim.sq8_params.dim;
        const uint8_t *cur_sq8 = state->current_frame_sq8;
        const uint8_t *mat_sq8 = state->anchor_matrix_sq8;
        int act_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int compact_cnt = 0;
        int local_pruned = 0;

        state->telemetry.sq8_evals += act_cnt;

        for (int idx = 0; idx < act_cnt; idx++)
        {
            int c = act[idx];
            const uint8_t *a_ptr = mat_sq8
                ? (mat_sq8 + (size_t)c * (size_t)dim)
                : state->clusters[c].anchor_sq8;
            if (a_ptr != NULL)
            {
                double d_lb = sq8_compute_lower_bound(
                    cur_sq8, a_ptr, &config->optim.sq8_params, 0.0
                );
                if (d_lb > config->algo.rlim)
                {
                    state->scratch.clmembflag[c] = 0;
                    state->scratch.entropy_p_current[c] = 0.0;
                    local_pruned++;
                    continue;
                }
            }
            act[compact_cnt++] = c;
        }
        state->scratch.num_active_clusters = compact_cnt;
        state->telemetry.sq8_pruned += local_pruned;
        state->telemetry.clusters_pruned += local_pruned;

        if (local_pruned > 0)
        {
            double sum_p = 0.0;
            for (int idx = 0; idx < compact_cnt; idx++)
            {
                sum_p += state->scratch.entropy_p_current[act[idx]];
            }
            if (sum_p > 0.0)
            {
                double inv_sum_p = 1.0 / sum_p;
                for (int idx = 0; idx < compact_cnt; idx++)
                {
                    state->scratch.entropy_p_current[act[idx]] *= inv_sum_p;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_sq_e);
    state->telemetry.time_step_3a_sq_filter +=
        (t_sq_e.tv_sec - t_sq_s.tv_sec) * 1000.0 +
        (t_sq_e.tv_nsec - t_sq_s.tv_nsec) / 1000000.0;
}

/**
 * @file handle_new_cluster_creation.c
 * @brief New cluster creation and inter-cluster
 *        distance initialization.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include "cluster_core.h"
#include "frameread.h"
#include "cluster_bounds.h"
#include "cluster_gemm_dist.h"
#include "framedistance.h"
#include "gric_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cluster_trace.h"

#define ANSI_COLOR_GREEN  "\x1b[32m"
#define ANSI_COLOR_RESET  "\x1b[0m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"

/**
 * init_new_cluster_distances - Initialize distance entries for a newly created cluster.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @new_cl: Index of the newly created cluster.
 * @temp_indices: Array of cluster indices measured during the current frame search.
 * @temp_dists: Corresponding measured distances for each temp_indices entry.
 * @temp_count: Number of valid entries in temp_indices / temp_dists.
 *
 * Uses exact distance computations for clusters that were visited
 * during the search (recorded in temp_indices/temp_dists), and
 * triangle-inequality bound propagation for the remaining
 * unvisited clusters.
 */
static void init_new_cluster_distances(
    ClusterConfig *config,
    ClusterState  *state,
    int            new_cl,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count)
{
    int N = config->algo.maxnbclust;

    if (config->optim.sparse_dcc_mode)
    {
        // 1. Initialize bounds to [0, infinity] and measured flag to 0
        for (int r = 0; r < new_cl; r++)
        {
            state->scratch.dcc_min[new_cl * N + r] = 0.0;
            state->scratch.dcc_min[r * N + new_cl] = 0.0;
            state->scratch.dcc_max[new_cl * N + r] = 1e19;
            state->scratch.dcc_max[r * N + new_cl] = 1e19;
            state->scratch.dcc_measured[new_cl * N + r] = 0;
            state->scratch.dcc_measured[r * N + new_cl] = 0;
            if (state->scratch.dcc_sq16 != NULL)
            {
                state->scratch.dcc_sq16[new_cl * N + r] = DCC_SQ16_UNMEASURED;
                state->scratch.dcc_sq16[r * N + new_cl] = DCC_SQ16_UNMEASURED;
            }
        }

        state->scratch.dcc_min[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_max[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_measured[new_cl * N + new_cl] = 1;
        if (state->scratch.dcc_sq16 != NULL)
        {
            state->scratch.dcc_sq16[new_cl * N + new_cl] = 0;
        }

        // 2. Populate exact distances from the search loop
        char is_temp_index[new_cl];
        memset(is_temp_index, 0, new_cl * sizeof(char));

        for (int idx = 0; idx < temp_count; idx++)
        {
            int j = temp_indices[idx];
            if (j >= 0 && j < new_cl)
            {
                double d = temp_dists[idx];
                set_dcc_pair(state, N, new_cl, j, d);
                is_temp_index[j] = 1;
            }
        }

        // 3. Propagate bounds to unvisited clusters
        double *dcc_max_rows[temp_count];
        double *dcc_min_rows[temp_count];
        double  d_new_j_arr[temp_count];
        int     valid_count = 0;

        for (int idx = 0; idx < temp_count; idx++)
        {
            int j = temp_indices[idx];
            if (j >= 0 && j < new_cl)
            {
                dcc_max_rows[valid_count] = &state->scratch.dcc_max[j * N];
                dcc_min_rows[valid_count] = &state->scratch.dcc_min[j * N];
                d_new_j_arr[valid_count] = temp_dists[idx];
                valid_count++;
            }
        }

        #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
        for (int k = 0; k < new_cl; k++)
        {
            if (is_temp_index[k])
            {
                continue;
            }

            double *max_new_row = &state->scratch.dcc_max[new_cl * N];
            double *min_new_row = &state->scratch.dcc_min[new_cl * N];

            for (int idx = 0; idx < valid_count; idx++)
            {
                double  d_new_j = d_new_j_arr[idx];
                double *max_j_row = dcc_max_rows[idx];
                double *min_j_row = dcc_min_rows[idx];

                if (max_j_row[k] < 1e18)
                {
                    double new_max = d_new_j + max_j_row[k];
                    if (new_max < max_new_row[k])
                    {
                        max_new_row[k] = new_max;
                        state->scratch.dcc_max[k * N + new_cl] = new_max;
                    }
                }

                if (max_j_row[k] < 1e18)
                {
                    double l1 = d_new_j - max_j_row[k];
                    if (l1 > min_new_row[k])
                    {
                        min_new_row[k] = l1;
                        state->scratch.dcc_min[k * N + new_cl] = l1;
                    }
                }
                if (min_j_row[k] - d_new_j > min_new_row[k])
                {
                    min_new_row[k] = min_j_row[k] - d_new_j;
                    state->scratch.dcc_min[k * N + new_cl] = min_new_row[k];
                }
            }
        }
        state->telemetry.dcc_entries_populated += (uint64_t)valid_count;
    }
    else if (!config->output.distall_mode && config->optim.use_batch_dist &&
             !state->clusters[new_cl].anchor.is_double &&
             state->anchor_matrix_float != NULL)
    {
        long frame_elem = (long)state->clusters[new_cl].anchor.width *
                          (long)state->clusters[new_cl].anchor.height;
        const float *q = (const float *)state->clusters[new_cl].anchor.data;
        const float *anchors_mat = state->anchor_matrix_float;

        double   *dcc_min_row = &state->scratch.dcc_min[(size_t)new_cl * N];
        double   *dcc_max_row = &state->scratch.dcc_max[(size_t)new_cl * N];
        char     *dcc_meas_row = &state->scratch.dcc_measured[(size_t)new_cl * N];
        uint16_t *dcc_sq16_row = (state->scratch.dcc_sq16 != NULL)
                                 ? &state->scratch.dcc_sq16[(size_t)new_cl * N]
                                 : NULL;

        if (new_cl > 0)
        {
            const float *q_norm = (state->anchor_norms_float != NULL)
                                  ? &state->anchor_norms_float[new_cl]
                                  : NULL;
            const float *cand_norms = state->anchor_norms_float;

            cluster_gemm_dist_float(
                q,
                anchors_mat,
                q_norm,
                cand_norms,
                1,
                new_cl,
                (int)frame_elem,
                NULL,
                dcc_min_row
            );
        }

        int unique_visited = 0;
        char is_temp_index[new_cl > 0 ? new_cl : 1];
        memset(is_temp_index, 0, (new_cl > 0 ? new_cl : 1) * sizeof(char));

        for (int idx = 0; idx < temp_count; idx++)
        {
            int j = temp_indices[idx];
            if (j >= 0 && j < new_cl)
            {
                dcc_min_row[j] = temp_dists[idx];
                if (!is_temp_index[j])
                {
                    is_temp_index[j] = 1;
                    unique_visited++;
                }
            }
        }

        dcc_min_row[new_cl] = 0.0;
        memcpy(dcc_max_row, dcc_min_row, (size_t)(new_cl + 1) * sizeof(double));
        memset(dcc_meas_row, 1, (size_t)(new_cl + 1) * sizeof(char));

        if (dcc_sq16_row != NULL)
        {
            double s = state->scratch.dcc_sq16_scale;
            for (int k = 0; k < new_cl; k++)
            {
                double d = dcc_min_row[k];
                dcc_sq16_row[k] = (d * s >= 65534.0) ? 65534 : (uint16_t)(d * s + 0.5);
            }
            dcc_sq16_row[new_cl] = 0;
        }

        /* Sequential scatter to symmetric columns */
        double   *dcc_min = state->scratch.dcc_min;
        double   *dcc_max = state->scratch.dcc_max;
        char     *dcc_meas = state->scratch.dcc_measured;
        uint16_t *dcc_sq16 = state->scratch.dcc_sq16;

        if (dcc_sq16 != NULL)
        {
            for (int k = 0; k < new_cl; k++)
            {
                size_t col_idx = (size_t)k * N + new_cl;
                double d = dcc_min_row[k];
                dcc_min[col_idx] = d;
                dcc_max[col_idx] = d;
                dcc_meas[col_idx] = 1;
                dcc_sq16[col_idx] = dcc_sq16_row[k];
            }
        }
        else
        {
            for (int k = 0; k < new_cl; k++)
            {
                size_t col_idx = (size_t)k * N + new_cl;
                double d = dcc_min_row[k];
                dcc_min[col_idx] = d;
                dcc_max[col_idx] = d;
                dcc_meas[col_idx] = 1;
            }
        }

        int unvisited_count = new_cl - unique_visited;
        state->telemetry.framedist_calls += (uint64_t)unvisited_count;
        state->telemetry.framedist_calls_intercluster += (uint64_t)unvisited_count;
        state->telemetry.dcc_entries_populated += (uint64_t)new_cl;
    }
    else
    {
        char is_temp_index[new_cl > 0 ? new_cl : 1];
        memset(is_temp_index, 0, (new_cl > 0 ? new_cl : 1) * sizeof(char));

        for (int idx = 0; idx < temp_count; idx++)
        {
            int j = temp_indices[idx];
            if (j >= 0 && j < new_cl)
            {
                double d = temp_dists[idx];
                set_dcc_pair(state, N, new_cl, j, d);
                is_temp_index[j] = 1;
            }
        }

        int unvisited[new_cl > 0 ? new_cl : 1];
        int unvisited_count = 0;
        for (int k = 0; k < new_cl; k++)
        {
            if (!is_temp_index[k])
            {
                unvisited[unvisited_count++] = k;
            }
        }

        long frame_elem = (long)state->clusters[new_cl].anchor.width *
                          (long)state->clusters[new_cl].anchor.height;
        int is_double = state->clusters[new_cl].anchor.is_double;
        int processed_count = 0;

        if (!config->output.distall_mode && config->optim.use_batch_dist)
        {
#if GRIC_HAVE_AVX512_TARGET
            if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && !is_double)
            {
                int b_count16 = unvisited_count / 16;
                #pragma omp parallel for if(b_count16 >= 4) schedule(static)
                for (int b = 0; b < b_count16; b++)
                {
                    int b_idx = b * 16;
                    double batch_dists[16];
                    const float *b_anchors[16];
                    for (int k = 0; k < 16; k++)
                    {
                        int cl_k = unvisited[b_idx + k];
                        b_anchors[k] = (const float *)state->clusters[cl_k].anchor.data;
                    }
                    framedist_batch_1x16_float(
                        (const float *)state->clusters[new_cl].anchor.data,
                        b_anchors,
                        batch_dists,
                        frame_elem);

                    for (int k = 0; k < 16; k++)
                    {
                        int cl_idx = unvisited[b_idx + k];
                        set_dcc_pair(state, N, new_cl, cl_idx, batch_dists[k]);
                    }
                } // for (int b = 0; b < b_count16; b++)
                processed_count = b_count16 * 16;
            }
#endif
            int b_count8 = (unvisited_count - processed_count) / 8;
            #pragma omp parallel for if(b_count8 >= 4) schedule(static)
            for (int b = 0; b < b_count8; b++)
            {
                int b_idx = processed_count + b * 8;
                double batch_dists[8];

                if (is_double)
                {
                    const double *b_anchors[8];
                    for (int k = 0; k < 8; k++)
                    {
                        int cl_k = unvisited[b_idx + k];
                        b_anchors[k] = (const double *)state->clusters[cl_k].anchor.data;
                    }
                    framedist_batch_1x8_double(
                        (const double *)state->clusters[new_cl].anchor.data,
                        b_anchors,
                        batch_dists,
                        frame_elem);
                }
                else
                {
                    const float *b_anchors[8];
                    for (int k = 0; k < 8; k++)
                    {
                        int cl_k = unvisited[b_idx + k];
                        b_anchors[k] = (const float *)state->clusters[cl_k].anchor.data;
                    }
                    framedist_batch_1x8_float(
                        (const float *)state->clusters[new_cl].anchor.data,
                        b_anchors,
                        batch_dists,
                        frame_elem);
                }

                for (int k = 0; k < 8; k++)
                {
                    int cl_idx = unvisited[b_idx + k];
                    set_dcc_pair(state, N, new_cl, cl_idx, batch_dists[k]);
                }
            } // for (int b = 0; b < b_count8; b++)
            processed_count += b_count8 * 8;

            int b_count4 = (unvisited_count - processed_count) / 4;
            for (int b = 0; b < b_count4; b++)
            {
                int b_idx = processed_count + b * 4;
                double batch_dists[4];

                if (is_double)
                {
                    const double *b_anchors[4];
                    for (int k = 0; k < 4; k++)
                    {
                        int cl_k = unvisited[b_idx + k];
                        b_anchors[k] = (const double *)state->clusters[cl_k].anchor.data;
                    }
                    framedist_batch_1x4_double(
                        (const double *)state->clusters[new_cl].anchor.data,
                        b_anchors,
                        batch_dists,
                        frame_elem);
                }
                else
                {
                    const float *b_anchors[4];
                    for (int k = 0; k < 4; k++)
                    {
                        int cl_k = unvisited[b_idx + k];
                        b_anchors[k] = (const float *)state->clusters[cl_k].anchor.data;
                    }
                    framedist_batch_1x4_float(
                        (const float *)state->clusters[new_cl].anchor.data,
                        b_anchors,
                        batch_dists,
                        frame_elem);
                }

                for (int k = 0; k < 4; k++)
                {
                    int cl_idx = unvisited[b_idx + k];
                    set_dcc_pair(state, N, new_cl, cl_idx, batch_dists[k]);
                }
            }
            processed_count += b_count4 * 4;

            for (int i = processed_count; i < unvisited_count; i++)
            {
                int cl_idx = unvisited[i];
                double d = get_dist(&state->clusters[new_cl].anchor,
                                    &state->clusters[cl_idx].anchor, -1, -1.0, -1.0,
                                    config, state);
                set_dcc_pair(state, N, new_cl, cl_idx, d);
            }
            state->telemetry.framedist_calls += (uint64_t)unvisited_count;
            state->telemetry.framedist_calls_intercluster += (uint64_t)unvisited_count;
        }
        else
        {
            for (int i = 0; i < unvisited_count; i++)
            {
                int cl_idx = unvisited[i];
                double d = get_dist(&state->clusters[new_cl].anchor,
                                    &state->clusters[cl_idx].anchor, -1, -1.0, -1.0,
                                    config, state);
                set_dcc_pair(state, N, new_cl, cl_idx, d);
            }
            state->telemetry.framedist_calls += (uint64_t)unvisited_count;
            state->telemetry.framedist_calls_intercluster += (uint64_t)unvisited_count;
        }

        state->scratch.dcc_min[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_max[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_measured[new_cl * N + new_cl] = 1;
        if (state->scratch.dcc_sq16 != NULL)
        {
            state->scratch.dcc_sq16[new_cl * N + new_cl] = 0;
        }
        state->telemetry.dcc_entries_populated += (uint64_t)new_cl;
    }
}

/**
 * handle_new_cluster_creation - Manage cluster creation and eviction limits.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @current_frame: The frame being clustered.
 * @prev_assigned_cluster: The previous frame's cluster index assignment.
 * @temp_indices: Array tracking measured indices in this step.
 * @temp_dists: Array tracking computed distances in this step.
 * @temp_count: Pointer to total measurement count in this step.
 *
 * Checks if cluster capacity maxnbclust is reached. If not, instantiates a new cluster.
 * If reached, executes the configured eviction strategy (Stop, Discard, or Merge).
 *
 * Return: Cluster index assigned to the new frame, or -2 if stop signal is triggered.
 */
static void assign_new_cluster_anchor(
    ClusterState  *state,
    ClusterConfig *config,
    int            cl_idx,
    Frame         *current_frame)
{
    state->clusters[cl_idx].anchor = *current_frame;
    if (config->optim.use_sq8 && state->current_frame_sq8 != NULL)
    {
        long dim = current_frame->width * current_frame->height;
        if (state->anchor_matrix_sq8 != NULL)
        {
            state->clusters[cl_idx].anchor_sq8 =
                state->anchor_matrix_sq8 + (size_t)cl_idx * (size_t)dim;
        }
        else
        {
            state->clusters[cl_idx].anchor_sq8 = (uint8_t *)malloc((size_t)dim);
        }
        if (state->clusters[cl_idx].anchor_sq8 != NULL)
        {
            memcpy(state->clusters[cl_idx].anchor_sq8, state->current_frame_sq8,
                   (size_t)dim);
        }
    }
    else
    {
        state->clusters[cl_idx].anchor_sq8 = NULL;
    }
    if (config->optim.use_eq16 && state->current_frame_eq16 != NULL)
    {
        long dim = current_frame->width * current_frame->height;
        if (state->anchor_matrix_eq16 != NULL)
        {
            state->clusters[cl_idx].anchor_eq16 =
                state->anchor_matrix_eq16 + (size_t)cl_idx * (size_t)dim;
        }
        else
        {
            state->clusters[cl_idx].anchor_eq16 =
                (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        }
        if (state->clusters[cl_idx].anchor_eq16 != NULL)
        {
            memcpy(state->clusters[cl_idx].anchor_eq16, state->current_frame_eq16,
                   (size_t)dim * sizeof(int16_t));
        }
        if (state->anchor_matrix_eq16_interleaved != NULL)
        {
            eq16_set_anchor_interleaved(state->anchor_matrix_eq16_interleaved, cl_idx,
                                        state->current_frame_eq16, dim);
        }
        if (state->anchor_matrix_adc_interleaved != NULL)
        {
            eq16_set_anchor_adc_interleaved(state->anchor_matrix_adc_interleaved, cl_idx,
                                            state->current_frame_eq16, dim);
        }
    }
    else
    {
        state->clusters[cl_idx].anchor_eq16 = NULL;
    }
    if (config->optim.use_sq16 && state->current_frame_sq16 != NULL)
    {
        long dim = current_frame->width * current_frame->height;
        if (state->anchor_matrix_sq16 != NULL)
        {
            state->clusters[cl_idx].anchor_sq16 =
                state->anchor_matrix_sq16 + (size_t)cl_idx * (size_t)dim;
        }
        else
        {
            state->clusters[cl_idx].anchor_sq16 =
                (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        }
        if (state->clusters[cl_idx].anchor_sq16 != NULL)
        {
            memcpy(state->clusters[cl_idx].anchor_sq16, state->current_frame_sq16,
                   (size_t)dim * sizeof(int16_t));
        }
        if (state->anchor_matrix_sq16_interleaved != NULL)
        {
            sq16_set_anchor_interleaved(state->anchor_matrix_sq16_interleaved, cl_idx,
                                        state->current_frame_sq16, dim);
        }
    }
    else
    {
        state->clusters[cl_idx].anchor_sq16 = NULL;
    }
    if (state->anchor_matrix_float != NULL && !current_frame->is_double)
    {
        long dim = current_frame->width * current_frame->height;
        memcpy(state->anchor_matrix_float + (size_t)cl_idx * (size_t)dim,
               state->clusters[cl_idx].anchor.data,
               (size_t)dim * sizeof(float));
        if (state->anchor_norms_float != NULL)
        {
            cluster_compute_l2_norms_float(
                (const float *)state->clusters[cl_idx].anchor.data,
                1,
                (int)dim,
                &state->anchor_norms_float[cl_idx]);
        }
    }
    current_frame->data = NULL;
    state->clusters[cl_idx].id = cl_idx;
    state->clusters[cl_idx].prob = 1.0;
    if (state->scratch.cluster_probs != NULL)
    {
        state->scratch.cluster_probs[cl_idx] = 1.0;
    }
}

int handle_new_cluster_creation(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *prev_assigned_cluster,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count)
{
    if (state->num_clusters < config->algo.maxnbclust)
    {
        int assigned_cluster = state->num_clusters;
        assign_new_cluster_anchor(state, config, state->num_clusters, current_frame);

        init_new_cluster_distances(config, state, state->num_clusters,
                                   temp_indices, temp_dists, *temp_count);

        if (config->output.verbose_level >= 2)
        {
            printf(ANSI_COLOR_GREEN
                   "  [VV] Frame %5ld assigned to Cluster %4d\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, assigned_cluster);
            printf(ANSI_COLOR_ORANGE
                   "  [VV] Frame %5ld created new Cluster %4d\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, state->num_clusters);
        }

        add_visitor(&state->cluster_visitors[state->num_clusters],
                    state->telemetry.total_frames_processed);

        if (*temp_count < config->algo.maxnbclust)
        {
            temp_indices[*temp_count] = state->num_clusters;
            temp_dists[*temp_count] = 0.0;
            (*temp_count)++;
        }
        update_consistency_mask_for_new_cluster(config, state, state->num_clusters);
        state->telemetry.num_new_clusters++;
        state->num_clusters++;
        if (state->trace)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_NEW_CLUSTER);
            if (ev)
            {
                ev->cluster_id = assigned_cluster;
            }
        }
        return assigned_cluster;
    }

    if (config->algo.maxcl_strategy == MAXCL_STOP)
    {
        printf(ANSI_COLOR_ORANGE "Max clusters limit reached.\n" ANSI_COLOR_RESET);
        printf("Frames clustered: %ld\n", state->telemetry.total_frames_processed);
        free_frame(current_frame);
        if (state->trace)
        {
            trace_emit(state->trace, TRACE_EVICT_STOP);
        }
        return -2;
    }
    else if (config->algo.maxcl_strategy == MAXCL_DISCARD)
    {
        int scan_limit = (int)(state->num_clusters * config->algo.discard_fraction);
        if (scan_limit < 1)
        {
            scan_limit = state->num_clusters;
        }

        int min_idx = -1;
        int min_count = -1;

        for (int i = 0; i < scan_limit; i++)
        {
            int count = state->cluster_visitors[i].count;
            if (min_idx == -1 || count < min_count)
            {
                min_count = count;
                min_idx = i;
            }
        }

        if (min_idx != -1)
        {
            if (state->trace)
            {
                TraceEvent *ev = trace_emit(state->trace, TRACE_EVICT_DISCARD);
                if (ev)
                {
                    ev->cluster_id = min_idx;
                }
            }
            remove_cluster(state, config, min_idx, -1);
            if (*prev_assigned_cluster == min_idx)
            {
                *prev_assigned_cluster = -1;
            }
            else if (*prev_assigned_cluster > min_idx)
            {
                (*prev_assigned_cluster)--;
            }
            int assigned_cluster = state->num_clusters;
            assign_new_cluster_anchor(state, config, state->num_clusters, current_frame);

            init_new_cluster_distances(config, state, state->num_clusters,
                                       temp_indices, temp_dists, *temp_count);

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }
            update_consistency_mask_for_new_cluster(config, state, state->num_clusters);
            state->telemetry.num_new_clusters++;
            state->num_clusters++;
            return assigned_cluster;
        }

        free_frame(current_frame);
        return -2;
    }
    else if (config->algo.maxcl_strategy == MAXCL_MERGE)
    {
        int    best_i = -1, best_j = -1;
        double min_d = -1.0;

        for (int i = 0; i < state->num_clusters; i++)
        {
            for (int j = i + 1; j < state->num_clusters; j++)
            {
                double d = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                if (state->scratch.dcc_measured[i * config->algo.maxnbclust + j] &&
                    d >= 0.0 && (min_d < 0.0 || d < min_d))
                {
                    min_d = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i != -1)
        {
            int count_i = state->cluster_visitors[best_i].count;
            int count_j = state->cluster_visitors[best_j].count;
            int target = (count_i >= count_j) ? best_i : best_j;
            int remove = (count_i >= count_j) ? best_j : best_i;

            if (config->output.verbose_level >= 1)
            {
                printf("Merging cluster %d into %d (dist %.4f)\n", remove, target,
                       min_d);
            }

            if (state->trace)
            {
                TraceEvent *ev = trace_emit(state->trace, TRACE_EVICT_MERGE);
                if (ev)
                {
                    ev->cluster_id = remove;
                    ev->distance = min_d;
                    ev->active_remaining = target;
                }
            }

            remove_cluster(state, config, remove, target);
            if (*prev_assigned_cluster == remove)
            {
                if (target > remove)
                {
                    *prev_assigned_cluster = target - 1;
                }
                else
                {
                    *prev_assigned_cluster = target;
                }
            }
            else if (*prev_assigned_cluster > remove)
            {
                (*prev_assigned_cluster)--;
            }

            int assigned_cluster = state->num_clusters;
            assign_new_cluster_anchor(state, config, state->num_clusters, current_frame);

            init_new_cluster_distances(config, state, state->num_clusters,
                                       temp_indices, temp_dists, *temp_count);

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }
            update_consistency_mask_for_new_cluster(config, state, state->num_clusters);
            state->telemetry.num_new_clusters++;
            state->num_clusters++;
            return assigned_cluster;
        }

        free_frame(current_frame);
        return -2;
    }

    free_frame(current_frame);
    return -2;
}

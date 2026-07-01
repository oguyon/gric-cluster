#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include "frameread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * record_step_assignment - Record telemetry and write cluster results.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @current_frame: The frame being clustered.
 * @assigned_cluster: The final cluster index assigned to the current frame.
 * @prev_assigned_cluster: Pointer tracking the previous frame's cluster assignment.
 * @ascii_out: Output file stream for writing frame membership info (NULL if disabled).
 * @temp_indices: Array of measured indices in this step.
 * @temp_dists: Array of computed distances in this step.
 * @temp_count: Number of measurements recorded in this step.
 * @start_pruned_val: Pruning counter before beginning this step.
 *
 * Increments the transition matrix count, saves assignments, logs distance outputs,
 * applies predictive probability reward/decay functions, and frees current_frame.
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
    long           start_pruned_val)
{
    if (state->telemetry.total_frames_processed > 0 && *prev_assigned_cluster != -1 &&
        assigned_cluster != -1)
    {
        state->transition_matrix[*prev_assigned_cluster * config->algo.maxnbclust +
                                 assigned_cluster]++;
    }
    *prev_assigned_cluster = assigned_cluster;

    state->assignments[state->telemetry.total_frames_processed] = assigned_cluster;
    if (ascii_out)
    {
        if (config->input.stream_input_mode)
        {
            fprintf(ascii_out, "%ld %d %lu %ld.%09ld\n",
                    state->telemetry.total_frames_processed,
                    assigned_cluster, current_frame->cnt0, current_frame->atime.tv_sec,
                    current_frame->atime.tv_nsec);
        }
        else
        {
            fprintf(ascii_out, "%ld %d\n",
                    state->telemetry.total_frames_processed, assigned_cluster);
        }
    }

    state->frame_infos[state->telemetry.total_frames_processed].assignment = assigned_cluster;
    state->frame_infos[state->telemetry.total_frames_processed].num_dists = temp_count;
    if (temp_count > 0)
    {
        state->frame_infos[state->telemetry.total_frames_processed].cluster_indices =
            (int *)malloc(temp_count * sizeof(int));
        state->frame_infos[state->telemetry.total_frames_processed].distances =
            (double *)malloc(temp_count * sizeof(double));
        if (state->frame_infos[state->telemetry.total_frames_processed].cluster_indices &&
            state->frame_infos[state->telemetry.total_frames_processed].distances)
        {
            memcpy(state->frame_infos[state->telemetry.total_frames_processed].cluster_indices,
                   temp_indices, temp_count * sizeof(int));
            memcpy(state->frame_infos[state->telemetry.total_frames_processed].distances,
                   temp_dists, temp_count * sizeof(double));
        }
    }
    else
    {
        state->frame_infos[state->telemetry.total_frames_processed].cluster_indices = NULL;
        state->frame_infos[state->telemetry.total_frames_processed].distances = NULL;
    }

    if (config->optim.pred_mode)
    {
        state->clusters[assigned_cluster].prob += 0.3;

        double sum_p = 0.0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            sum_p += state->clusters[i].prob;
        }
        if (sum_p > 0.0)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                state->clusters[i].prob /= sum_p;
            }
        }

        double floor_val = 0.2 / state->num_clusters;
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].prob += floor_val;
        }

        sum_p = 0.0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            sum_p += state->clusters[i].prob;
        }
        if (sum_p > 0.0)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                state->clusters[i].prob /= sum_p;
            }
        }
    }

    state->telemetry.total_frames_processed++;

    if (state->telemetry.dist_counts && temp_count <= config->algo.maxnbclust)
    {
        state->telemetry.dist_counts[temp_count]++;
        state->telemetry.pruned_counts_by_dist[temp_count] +=
            (state->telemetry.clusters_pruned - start_pruned_val);
    }

    free_frame(current_frame);
}

#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include "cluster_core.h"
#include "frameread.h"
#include "cluster_bounds.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANSI_COLOR_GREEN  "\x1b[32m"
#define ANSI_COLOR_RESET  "\x1b[0m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"

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
        }

        state->scratch.dcc_min[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_max[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_measured[new_cl * N + new_cl] = 1;

        // 2. Populate exact distances from the search loop
        char is_temp_index[N];
        memset(is_temp_index, 0, N * sizeof(char));

        for (int idx = 0; idx < temp_count; idx++)
        {
            int j = temp_indices[idx];
            if (j >= 0 && j < new_cl)
            {
                double d = temp_dists[idx];
                state->scratch.dcc_min[new_cl * N + j] = d;
                state->scratch.dcc_min[j * N + new_cl] = d;
                state->scratch.dcc_max[new_cl * N + j] = d;
                state->scratch.dcc_max[j * N + new_cl] = d;
                state->scratch.dcc_measured[new_cl * N + j] = 1;
                state->scratch.dcc_measured[j * N + new_cl] = 1;
                is_temp_index[j] = 1;
            }
        }

        // 3. Propagate bounds to unvisited clusters
        #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
        for (int k = 0; k < new_cl; k++)
        {
            if (is_temp_index[k])
            {
                continue;
            }

            for (int idx = 0; idx < temp_count; idx++)
            {
                int j = temp_indices[idx];
                if (j < 0 || j >= new_cl)
                {
                    continue;
                }

                double d_new_j = temp_dists[idx];

                if (state->scratch.dcc_max[j * N + k] < 1e18)
                {
                    double new_max = d_new_j + state->scratch.dcc_max[j * N + k];
                    if (new_max < state->scratch.dcc_max[new_cl * N + k])
                    {
                        state->scratch.dcc_max[new_cl * N + k] = new_max;
                        state->scratch.dcc_max[k * N + new_cl] = new_max;
                    }
                }

                if (state->scratch.dcc_max[j * N + k] < 1e18)
                {
                    double l1 = d_new_j - state->scratch.dcc_max[j * N + k];
                    if (l1 > state->scratch.dcc_min[new_cl * N + k])
                    {
                        state->scratch.dcc_min[new_cl * N + k] = l1;
                        state->scratch.dcc_min[k * N + new_cl] = l1;
                    }
                }
                double l2 = state->scratch.dcc_min[j * N + k] - d_new_j;
                if (l2 > state->scratch.dcc_min[new_cl * N + k])
                {
                    state->scratch.dcc_min[new_cl * N + k] = l2;
                    state->scratch.dcc_min[k * N + new_cl] = l2;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < new_cl; i++)
        {
            double d = get_dist(&state->clusters[new_cl].anchor,
                                &state->clusters[i].anchor, -1, -1.0, -1.0,
                                config, state);
            state->scratch.dcc_min[new_cl * N + i] = d;
            state->scratch.dcc_min[i * N + new_cl] = d;
            state->scratch.dcc_max[new_cl * N + i] = d;
            state->scratch.dcc_max[i * N + new_cl] = d;
            state->scratch.dcc_measured[new_cl * N + i] = 1;
            state->scratch.dcc_measured[i * N + new_cl] = 1;
        }
        state->scratch.dcc_min[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_max[new_cl * N + new_cl] = 0.0;
        state->scratch.dcc_measured[new_cl * N + new_cl] = 1;
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
        state->clusters[state->num_clusters].anchor = *current_frame;
        current_frame->data = NULL;
        state->clusters[state->num_clusters].id = state->num_clusters;
        state->clusters[state->num_clusters].prob = 1.0;

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
        return assigned_cluster;
    }

    if (config->algo.maxcl_strategy == MAXCL_STOP)
    {
        printf(ANSI_COLOR_ORANGE "Max clusters limit reached.\n" ANSI_COLOR_RESET);
        printf("Frames clustered: %ld\n", state->telemetry.total_frames_processed);
        free_frame(current_frame);
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
            state->clusters[state->num_clusters].anchor = *current_frame;
            current_frame->data = NULL;
            state->clusters[state->num_clusters].id = state->num_clusters;
            state->clusters[state->num_clusters].prob = 1.0;

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
            state->clusters[state->num_clusters].anchor = *current_frame;
            current_frame->data = NULL;
            state->clusters[state->num_clusters].id = state->num_clusters;
            state->clusters[state->num_clusters].prob = 1.0;

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

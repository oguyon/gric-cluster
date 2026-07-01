#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include "cluster_core.h"
#include "frameread.h"
#include <stdio.h>

#define ANSI_COLOR_GREEN  "\x1b[32m"
#define ANSI_COLOR_RESET  "\x1b[0m"
#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"

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

        for (int i = 0; i < state->num_clusters; i++)
        {
            double d =
                get_dist(&state->clusters[state->num_clusters].anchor,
                         &state->clusters[i].anchor, -1, -1.0, -1.0, config, state);
            state->scratch.dccarray[
                state->num_clusters * config->algo.maxnbclust + i] = d;
            state->scratch.dccarray[
                i * config->algo.maxnbclust + state->num_clusters] = d;
        }
        state->scratch.dccarray[
            state->num_clusters * config->algo.maxnbclust + state->num_clusters] = 0.0;

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

            for (int i = 0; i < state->num_clusters; i++)
            {
                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                    config, state);
                state->scratch.dccarray[
                    state->num_clusters * config->algo.maxnbclust + i] = d;
                state->scratch.dccarray[
                    i * config->algo.maxnbclust + state->num_clusters] = d;
            }
            state->scratch.dccarray[state->num_clusters * config->algo.maxnbclust +
                            state->num_clusters] = 0.0;

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }
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
                double d = state->scratch.dccarray[i * config->algo.maxnbclust + j];
                if (d >= 0 && (min_d < 0 || d < min_d))
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

            for (int i = 0; i < state->num_clusters; i++)
            {
                double d = get_dist(&state->clusters[state->num_clusters].anchor,
                                    &state->clusters[i].anchor, -1, -1.0, -1.0,
                                    config, state);
                state->scratch.dccarray[
                    state->num_clusters * config->algo.maxnbclust + i] = d;
                state->scratch.dccarray[
                    i * config->algo.maxnbclust + state->num_clusters] = d;
            }
            state->scratch.dccarray[state->num_clusters * config->algo.maxnbclust +
                            state->num_clusters] = 0.0;

            add_visitor(&state->cluster_visitors[state->num_clusters],
                        state->telemetry.total_frames_processed);

            if (*temp_count < config->algo.maxnbclust)
            {
                temp_indices[*temp_count] = state->num_clusters;
                temp_dists[*temp_count] = 0.0;
                (*temp_count)++;
            }
            state->num_clusters++;
            return assigned_cluster;
        }

        free_frame(current_frame);
        return -2;
    }

    free_frame(current_frame);
    return -2;
}

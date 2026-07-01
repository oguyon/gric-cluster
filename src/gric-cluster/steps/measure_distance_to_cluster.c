#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include "cluster_core.h"
#include <stdio.h>

#define ANSI_COLOR_GREEN  "\x1b[32m"
#define ANSI_COLOR_RESET  "\x1b[0m"

/**
 * measure_distance_to_cluster - Calculate distance from current frame to target cluster.
 * @cj: Cluster index being targeted.
 * @current_frame: The frame being clustered.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @temp_indices: Array tracking measured indices in this step.
 * @temp_dists: Array tracking computed distances in this step.
 * @temp_count: Pointer to total measurement count in this step.
 * @is_prediction: Flag indicating if this candidate is a prediction shortcut.
 *
 * Computes distance via get_dist(), increments telemetry counts, adds visitor entries,
 * and increments cluster probability if matched within the threshold `rlim`.
 *
 * Return: Calculated distance to the target cluster.
 */
double measure_distance_to_cluster(
    int            cj,
    Frame         *current_frame,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int           *temp_count,
    int            is_prediction)
{
    if (*temp_count < state->telemetry.max_steps_recorded && state->num_clusters > 0)
    {
        int pruned_cnt = 0;
        for (int pc = 0; pc < state->num_clusters; pc++)
        {
            if (state->scratch.clmembflag[pc] == 0)
            {
                pruned_cnt++;
            }
        }
        state->telemetry.pruned_fraction_sum[*temp_count] +=
            (double)pruned_cnt / state->num_clusters;
        state->telemetry.step_counts[*temp_count]++;
    }

    double dfc = get_dist(current_frame, &state->clusters[cj].anchor,
                          state->clusters[cj].id, state->clusters[cj].prob,
                          state->scratch.current_gprobs[cj], config, state);

    if (*temp_count < config->algo.maxnbclust)
    {
        temp_indices[*temp_count] = cj;
        temp_dists[*temp_count] = dfc;
        (*temp_count)++;
    }

    add_visitor(&state->cluster_visitors[cj], state->telemetry.total_frames_processed);

    if (dfc < config->algo.rlim)
    {
        if (!config->optim.pred_mode)
        {
            state->clusters[cj].prob += config->algo.deltaprob;
        }
        if (config->output.verbose_level >= 2)
        {
            printf(ANSI_COLOR_GREEN "  [VV] Frame %ld assigned to Cluster %d%s\n" ANSI_COLOR_RESET,
                   state->telemetry.total_frames_processed, cj,
                   is_prediction ? " (Prediction)" : "");
        }
    }

    return dfc;
}

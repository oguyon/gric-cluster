#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_math.h"
#include <math.h>
#include <stdio.h>

#define ANSI_COLOR_RESET  "\x1b[0m"
#define ANSI_BG_GREEN     "\x1b[42m"
#define ANSI_COLOR_BLACK  "\x1b[30m"

/**
 * update_geometric_probabilities - Update geometric priorities of candidate clusters.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @cj: Cluster index measured in the last step.
 * @dfc: Computed distance to cluster index cj.
 *
 * Loops through visitor history to retrieve co-measured frames. Multiplies
 * the running geometric match probabilities with match metric scaling factor.
 */
void update_geometric_probabilities(
    ClusterConfig *config,
    ClusterState  *state,
    int            cj,
    double         dfc)
{
    int match_count = state->cluster_visitors[cj].count;
    if (match_count > 0)
    {
        match_count--;
    }

    if (config->output.verbose_level >= 2)
    {
        printf("  [VV] Distance > rlim. Found %d matches in distinfo for Cluster "
               "%4d (Frame %5d).\n",
               match_count, cj, state->clusters[cj].anchor.id);
    }

    int start_idx = 0;
    if (state->cluster_visitors[cj].count > config->optim.max_gprob_visitors)
    {
        start_idx = state->cluster_visitors[cj].count -
                    config->optim.max_gprob_visitors;
    }
    for (int i = start_idx; i < state->cluster_visitors[cj].count; i++)
    {
        int k_idx = state->cluster_visitors[cj].frames[i];
        if (k_idx == state->telemetry.total_frames_processed)
        {
            continue;
        }

        int target_cl = state->frame_infos[k_idx].assignment;
        if (target_cl < 0 || target_cl >= state->num_clusters)
        {
            continue;
        }
        int is_active = state->scratch.clmembflag[target_cl];

        if (config->output.verbose_level >= 2)
        {
            if (is_active)
            {
                printf(ANSI_BG_GREEN ANSI_COLOR_BLACK
                       "  [VV]   Frame %5d also had distance measurement to "
                       "Cluster %4d (Anchor Frame %5d). Frame %5d cluster "
                       "membership is %4d. " ANSI_COLOR_RESET "\n",
                       k_idx, cj, state->clusters[cj].anchor.id, k_idx, target_cl);
            }
            else
            {
                printf("  [VV]   Frame %5d also had distance measurement to "
                       "Cluster %4d (Anchor Frame %5d). Frame %5d cluster "
                       "membership is %4d.\n",
                       k_idx, cj, state->clusters[cj].anchor.id, k_idx, target_cl);
            }
        }

        if (!is_active)
        {
            continue;
        }

        double dist_k = -1.0;
        for (int d_idx = 0; d_idx < state->frame_infos[k_idx].num_dists; d_idx++)
        {
            if (state->frame_infos[k_idx].cluster_indices[d_idx] == cj)
            {
                dist_k = state->frame_infos[k_idx].distances[d_idx];
                break;
            }
        }

        if (dist_k >= 0)
        {
            double dr = fabs(dfc - dist_k) / config->algo.rlim;
            double val = fmatch(dr, config->optim.fmatch_a, config->optim.fmatch_b);

            if (config->output.verbose_level >= 2)
            {
                printf("    dist %5ld-%-5d = %12.5e  dist %5d-%-5d = %12.5e, "
                       "fmatch=%12.5e, updating GProb(Cluster %4d) from %12.5e to "
                       "%12.5e\n",
                       state->telemetry.total_frames_processed,
                       state->clusters[cj].anchor.id,
                       dfc, k_idx, state->clusters[cj].anchor.id, dist_k, val,
                       target_cl, state->scratch.current_gprobs[target_cl],
                       state->scratch.current_gprobs[target_cl] * val);
            }

            state->scratch.current_gprobs[target_cl] *= val;
        }
    }
}

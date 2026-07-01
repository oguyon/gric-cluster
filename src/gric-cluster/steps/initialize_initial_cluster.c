#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include <stdio.h>

#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"
#define ANSI_COLOR_RESET  "\x1b[0m"

void initialize_initial_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *assigned_cluster)
{
    state->clusters[0].anchor = *current_frame;
    current_frame->data = NULL;
    state->clusters[0].id = 0;
    state->clusters[0].prob = 1.0;
    state->num_clusters = 1;
    state->scratch.dccarray[0] = 0.0;

    add_visitor(&state->cluster_visitors[0], state->telemetry.total_frames_processed);
    *assigned_cluster = 0;

    if (config->output.verbose_level >= 2)
    {
        printf(ANSI_COLOR_ORANGE
               "  [VV] Frame %5ld created initial Cluster    0\n" ANSI_COLOR_RESET,
               state->telemetry.total_frames_processed);
    }
}

#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_mgmt.h"
#include "cluster_trace.h"
#include <stdio.h>
#include <string.h>

#define ANSI_COLOR_ORANGE "\x1b[38;5;208m"
#define ANSI_COLOR_RESET  "\x1b[0m"

/**
 * initialize_initial_cluster - Set up the first cluster in the search space.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @current_frame: The very first frame ingested in the video sequence.
 * @assigned_cluster: Pointer to output destination storing the assigned cluster index (always 0).
 *
 * Configures the first ingested frame as the anchor for cluster index 0,
 * and sets up its initial frequency probability to 1.0. Registers the frame visitor.
 */
void initialize_initial_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    Frame         *current_frame,
    int           *assigned_cluster)
{
    state->clusters[0].anchor = *current_frame;
    if (config->optim.use_sq8 && state->current_frame_sq8 != NULL)
    {
        long dim = current_frame->width * current_frame->height;
        state->clusters[0].anchor_sq8 = (uint8_t *)malloc((size_t)dim);
        if (state->clusters[0].anchor_sq8 != NULL)
        {
            memcpy(state->clusters[0].anchor_sq8, state->current_frame_sq8, (size_t)dim);
        }
    }
    else
    {
        state->clusters[0].anchor_sq8 = NULL;
    }
    current_frame->data = NULL;
    state->clusters[0].id = 0;
    state->clusters[0].prob = 1.0;
    state->num_clusters = 1;
    state->scratch.dcc_min[0] = 0.0;
    state->scratch.dcc_max[0] = 0.0;
    state->scratch.dcc_measured[0] = 1;

    add_visitor(&state->cluster_visitors[0], state->telemetry.total_frames_processed);
    *assigned_cluster = 0;

    if (state->trace)
    {
        TraceEvent *ev = trace_emit(state->trace, TRACE_INITIAL_CLUSTER);
        if (ev)
        {
            ev->cluster_id = 0;
            ev->frame_id = current_frame->id;
        }
    }

    if (config->output.verbose_level >= 2)
    {
        printf(ANSI_COLOR_ORANGE
               "  [VV] Frame %5ld created initial Cluster    0\n" ANSI_COLOR_RESET,
               state->telemetry.total_frames_processed);
    }
}

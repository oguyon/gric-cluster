#include "cluster_mgmt.h"
#include "cluster_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void add_visitor(VisitorList *list, int frame_idx) {
    if (list->count >= list->capacity) {
        int new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        int *new_frames = (int *)realloc(list->frames, new_capacity * sizeof(int));
        if (new_frames) {
            list->frames = new_frames;
            list->capacity = new_capacity;
        } else {
            perror("Failed to realloc visitor list");
            return;
        }
    }
    list->frames[list->count++] = frame_idx;
}

void remove_cluster(ClusterState *state, ClusterConfig *config, int index_to_remove, int index_target) {
    if (index_to_remove < 0 || index_to_remove >= state->num_clusters) return;

    if (config->verbose_level >= 1) {
        printf("Removing cluster %d (Count: %d). Target: %d\n",
               index_to_remove, state->cluster_visitors[index_to_remove].count, index_target);
    }

    // 1. Log or Merge History
    if (index_target == -1 && config->output_discarded) {
        FILE *log = fopen("discarded_frames.txt", "a");
        if (log) {
            fprintf(log, "# Discarded Cluster %d\n", index_to_remove);
            for (int i = 0; i < state->cluster_visitors[index_to_remove].count; i++) {
                fprintf(log, "%d ", state->cluster_visitors[index_to_remove].frames[i]);
            }
            fprintf(log, "\n");
            fclose(log);
        }
    }

    // 2. Shift Clusters Array
    if (state->clusters[index_to_remove].anchor.data) {
        free(state->clusters[index_to_remove].anchor.data);
    }
    // Shift clusters down
    for (int i = index_to_remove; i < state->num_clusters - 1; i++) {
        state->clusters[i] = state->clusters[i+1];
        state->clusters[i].id = i; // Update ID
    }

    // 3. Shift Visitor Lists
    if (state->cluster_visitors[index_to_remove].frames) {
        free(state->cluster_visitors[index_to_remove].frames);
    }
    for (int i = index_to_remove; i < state->num_clusters - 1; i++) {
        state->cluster_visitors[i] = state->cluster_visitors[i+1];
    }
    // Zero out the last one (moved)
    memset(&state->cluster_visitors[state->num_clusters - 1], 0, sizeof(VisitorList));

    // 4. Shift DCC Array (Rows and Cols)
    int N = config->maxnbclust; // Stride is fixed maxnbclust

    // Shift Rows up
    for (int r = index_to_remove; r < state->num_clusters - 1; r++) {
        memcpy(&state->dccarray[r * N], &state->dccarray[(r + 1) * N], config->maxnbclust * sizeof(double));
    }
    // Shift Columns left for ALL rows
    for (int r = 0; r < state->num_clusters - 1; r++) {
        int dest_idx = r * N + index_to_remove;
        int src_idx = r * N + index_to_remove + 1;
        int count = config->maxnbclust - 1 - index_to_remove;
        if (count > 0) {
            memmove(&state->dccarray[dest_idx], &state->dccarray[src_idx], count * sizeof(double));
        }
    }

    // 5. Shift Transition Matrix (Same logic as DCC)
    // Shift Rows
    for (int r = index_to_remove; r < state->num_clusters - 1; r++) {
        memcpy(&state->transition_matrix[r * N], &state->transition_matrix[(r + 1) * N], config->maxnbclust * sizeof(long));
    }
    // Shift Cols
    for (int r = 0; r < state->num_clusters - 1; r++) {
        int dest_idx = r * N + index_to_remove;
        int src_idx = r * N + index_to_remove + 1;
        int count = config->maxnbclust - 1 - index_to_remove;
        if (count > 0) {
            memmove(&state->transition_matrix[dest_idx], &state->transition_matrix[src_idx], count * sizeof(long));
        }
    }

    // 6. Correct Assignments Update Loop
    for (long f = 0; f < state->total_frames_processed; f++) {
        int a = state->assignments[f];
        if (a == index_to_remove) {
            if (index_target == -1) {
                state->assignments[f] = -1;
            } else {
                if (index_target > index_to_remove) state->assignments[f] = index_target - 1;
                else state->assignments[f] = index_target;
            }
        } else if (a > index_to_remove) {
            state->assignments[f] = a - 1;
        }
    }

    // 7. Decrement Num Clusters
    state->num_clusters--;
}

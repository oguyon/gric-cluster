/**
 * @file cluster_mgmt.c
 * @brief Active cluster tracking and lifecycle management.
 *
 * Provides functions to register visitors to clusters and to prune or delete empty
 * clusters during the clustering process.
 *
 * Main Functions:
 * - add_visitor: Records that a frame index has visited/been assigned to a cluster.
 * - remove_cluster: Prunes and completely deletes a cluster from the active set.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_mgmt.h"
#include "cluster_core.h"
#include "cluster_dcc.h"
#include "cluster_steps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * add_visitor() - Safely adds a visitor record to a cluster's visitor history.
 * @list:       Pointer to the VisitorList structure.
 * @frame_idx:  Index of the frame to append.
 * @dist:       Distance from frame to cluster anchor.
 * @assignment: Assigned cluster index of the frame (-1 if not yet assigned).
 *
 * Dynamically resizes the record list capacity as needed (doubles capacity,
 * starting at 16). When the list exceeds VISITOR_COMPACT_THRESHOLD entries,
 * compacts to keep only the most recent VISITOR_COMPACT_KEEP entries to
 * prevent unbounded memory growth in long-running sessions.
 */
#define VISITOR_COMPACT_THRESHOLD 2048
#define VISITOR_COMPACT_KEEP      1024

void add_visitor(
    VisitorList *list,
    int          frame_idx,
    double       dist,
    int          assignment)
{
    /* Compact when the list grows too large */
    if (list->count >= VISITOR_COMPACT_THRESHOLD)
    {
        int keep = VISITOR_COMPACT_KEEP;
        int start = list->count - keep;
        memmove(
            list->records,
            list->records + start,
            (size_t)keep * sizeof(VisitorRecord)
        );
        list->count = keep;
    }

    if (list->count >= list->capacity)
    {
        int new_capacity =
            (list->capacity == 0)
                ? 16 : list->capacity * 2;
        VisitorRecord *new_records = (VisitorRecord *)realloc(
            list->records,
            (size_t)new_capacity * sizeof(VisitorRecord)
        );
        if (new_records)
        {
            list->records = new_records;
            list->capacity = new_capacity;
        }
        else
        {
            perror("Failed to realloc visitor list");
            return;
        }
    }
    list->records[list->count].frame = frame_idx;
    list->records[list->count].dist = dist;
    list->records[list->count].assignment = assignment;
    list->count++;
}

/**
 * remove_cluster() - Deletes a cluster from state, optionally merging its history.
 * @state:           Pointer to the active ClusterState.
 * @config:          Pointer to the active ClusterConfig.
 * @index_to_remove: The index of the cluster being deleted.
 * @index_target:    The merge target index, or -1 to discard completely.
 *
 * Reorganizes the active clusters list:
 * 1. Shifts cluster structs, updating cluster IDs.
 * 2. Shifts visitor history arrays.
 * 3. Compacts and shifts the inter-cluster distance matrix (dccarray).
 * 4. Compacts and shifts the inter-cluster transition probability matrix.
 * 5. Rewrites the assignments logs and maps the deleted cluster assignment
 *    either to a new target cluster index (if merging) or -1 (if discarding).
 */
void remove_cluster(
    ClusterState  *state,
    ClusterConfig *config,
    int            index_to_remove,
    int            index_target)
{
    if (index_to_remove < 0 || index_to_remove >= state->num_clusters)
        return;

    if (config->output.verbose_level >= 1)
    {
        printf("Removing cluster %d (Count: %d). Target: %d\n", index_to_remove,
               state->cluster_visitors[index_to_remove].count, index_target);
    }

    // 1. Log or Merge History
    if (index_target == -1 && config->output.output_discarded)
    {
        FILE *log = fopen("discarded_frames.txt", "a");
        if (log)
        {
            fprintf(log, "# Discarded Cluster %d\n", index_to_remove);
            for (int ii = 0; ii < state->cluster_visitors[index_to_remove].count; ii++)
            {
                fprintf(log, "%d ", state->cluster_visitors[index_to_remove].records[ii].frame);
            }
            fprintf(log, "\n");
            fclose(log);
        }
    } // if (index_target == -1 && config->output.output_discarded)

    // 2. Shift Clusters Array
    if (state->clusters[index_to_remove].anchor.data)
    {
        free(state->clusters[index_to_remove].anchor.data);
    }
    if (state->anchor_matrix_sq8 != NULL)
    {
        size_t dim = (size_t)state->clusters[index_to_remove].anchor.width *
                     (size_t)state->clusters[index_to_remove].anchor.height;
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0 && dim > 0)
        {
            memmove(state->anchor_matrix_sq8 + (size_t)index_to_remove * dim,
                    state->anchor_matrix_sq8 + (size_t)(index_to_remove + 1) * dim,
                    (size_t)remaining * dim * sizeof(uint8_t));
        }
    }
    else if (state->clusters[index_to_remove].anchor_sq8)
    {
        free(state->clusters[index_to_remove].anchor_sq8);
        state->clusters[index_to_remove].anchor_sq8 = NULL;
    }

    if (state->anchor_matrix_eq16 != NULL)
    {
        size_t dim = (size_t)state->clusters[index_to_remove].anchor.width *
                     (size_t)state->clusters[index_to_remove].anchor.height;
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0 && dim > 0)
        {
            memmove(state->anchor_matrix_eq16 + (size_t)index_to_remove * dim,
                    state->anchor_matrix_eq16 + (size_t)(index_to_remove + 1) * dim,
                    (size_t)remaining * dim * sizeof(int16_t));
        }
    }
    else if (state->clusters[index_to_remove].anchor_eq16)
    {
        free(state->clusters[index_to_remove].anchor_eq16);
        state->clusters[index_to_remove].anchor_eq16 = NULL;
    }

    if (state->anchor_matrix_sq16 != NULL)
    {
        size_t dim = (size_t)state->clusters[index_to_remove].anchor.width *
                     (size_t)state->clusters[index_to_remove].anchor.height;
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0 && dim > 0)
        {
            memmove(state->anchor_matrix_sq16 + (size_t)index_to_remove * dim,
                    state->anchor_matrix_sq16 + (size_t)(index_to_remove + 1) * dim,
                    (size_t)remaining * dim * sizeof(int16_t));
        }
    }
    else if (state->clusters[index_to_remove].anchor_sq16)
    {
        free(state->clusters[index_to_remove].anchor_sq16);
        state->clusters[index_to_remove].anchor_sq16 = NULL;
    }

    if (state->anchor_matrix_sq16_interleaved != NULL && state->anchor_matrix_sq16 != NULL)
    {
        long dim = (long)state->clusters[index_to_remove].anchor.width *
                   (long)state->clusters[index_to_remove].anchor.height;
        sq16_rebuild_anchor_interleaved(state->anchor_matrix_sq16_interleaved,
                                        state->anchor_matrix_sq16,
                                        state->num_clusters - 1,
                                        dim);
    }

    if (state->anchor_matrix_eq16_interleaved != NULL && state->anchor_matrix_eq16 != NULL)
    {
        long dim = (long)state->clusters[index_to_remove].anchor.width *
                   (long)state->clusters[index_to_remove].anchor.height;
        eq16_rebuild_anchor_interleaved(state->anchor_matrix_eq16_interleaved,
                                        state->anchor_matrix_eq16,
                                        state->num_clusters - 1,
                                        dim);
    }

    if (state->anchor_matrix_adc_interleaved != NULL && state->anchor_matrix_eq16 != NULL)
    {
        long dim = (long)state->clusters[index_to_remove].anchor.width *
                   (long)state->clusters[index_to_remove].anchor.height;
        eq16_rebuild_anchor_adc_interleaved(state->anchor_matrix_adc_interleaved,
                                            state->anchor_matrix_eq16,
                                            state->num_clusters - 1,
                                            dim);
    }

    if (state->anchor_matrix_float != NULL)
    {
        size_t dim = (size_t)state->clusters[index_to_remove].anchor.width *
                     (size_t)state->clusters[index_to_remove].anchor.height;
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0 && dim > 0)
        {
            memmove(state->anchor_matrix_float + (size_t)index_to_remove * dim,
                    state->anchor_matrix_float + (size_t)(index_to_remove + 1) * dim,
                    (size_t)remaining * dim * sizeof(float));
        }
    }

    if (state->anchor_norms_float != NULL)
    {
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0)
        {
            memmove(state->anchor_norms_float + index_to_remove,
                    state->anchor_norms_float + index_to_remove + 1,
                    (size_t)remaining * sizeof(float));
        }
    }


    if (state->scratch.cluster_probs != NULL)
    {
        int remaining = state->num_clusters - 1 - index_to_remove;
        if (remaining > 0)
        {
            memmove(state->scratch.cluster_probs + index_to_remove,
                    state->scratch.cluster_probs + index_to_remove + 1,
                    (size_t)remaining * sizeof(double));
        }
    }

    // Shift clusters down
    for (int cl_idx = index_to_remove; cl_idx < state->num_clusters - 1; cl_idx++)
    {
        state->clusters[cl_idx] = state->clusters[cl_idx + 1];
        state->clusters[cl_idx].id = cl_idx; // Update ID
        if (state->anchor_matrix_sq8 != NULL)
        {
            size_t dim = (size_t)state->clusters[cl_idx].anchor.width *
                         (size_t)state->clusters[cl_idx].anchor.height;
            state->clusters[cl_idx].anchor_sq8 =
                state->anchor_matrix_sq8 + (size_t)cl_idx * dim;
        }
        if (state->anchor_matrix_eq16 != NULL)
        {
            size_t dim = (size_t)state->clusters[cl_idx].anchor.width *
                         (size_t)state->clusters[cl_idx].anchor.height;
            state->clusters[cl_idx].anchor_eq16 =
                state->anchor_matrix_eq16 + (size_t)cl_idx * dim;
        }
        if (state->anchor_matrix_sq16 != NULL)
        {
            size_t dim = (size_t)state->clusters[cl_idx].anchor.width *
                         (size_t)state->clusters[cl_idx].anchor.height;
            state->clusters[cl_idx].anchor_sq16 =
                state->anchor_matrix_sq16 + (size_t)cl_idx * dim;
        }
    } // for (int cl_idx = index_to_remove; cl_idx < state->num_clusters - 1; cl_idx++)

    // 3. Shift Visitor Lists
    if (state->cluster_visitors[index_to_remove].records)
    {
        free(state->cluster_visitors[index_to_remove].records);
    }
    for (int cl_idx = index_to_remove; cl_idx < state->num_clusters - 1; cl_idx++)
    {
        state->cluster_visitors[cl_idx] = state->cluster_visitors[cl_idx + 1];
    }
    // Zero out the last one (moved)
    memset(&state->cluster_visitors[state->num_clusters - 1], 0, sizeof(VisitorList));

    // 4. Shift DCC Array
    dcc_remove_cluster(state, index_to_remove, config->optim.sparse_dcc_mode);

    // 5. Shift Transition Matrix
    int N = config->algo.maxnbclust;
    // Shift Rows
    for (int r = index_to_remove; r < state->num_clusters - 1; r++)
    {
        memcpy(&state->transition_matrix[r * N], &state->transition_matrix[(r + 1) * N],
               config->algo.maxnbclust * sizeof(long));
    }
    // Shift Cols
    for (int r = 0; r < state->num_clusters - 1; r++)
    {
        int dest_idx = r * N + index_to_remove;
        int src_idx = r * N + index_to_remove + 1;
        int count = config->algo.maxnbclust - 1 - index_to_remove;
        if (count > 0)
        {
            memmove(&state->transition_matrix[dest_idx], &state->transition_matrix[src_idx],
                    count * sizeof(long));
        }
    }

    // Clear the now-unused last row/col in transition matrix
    int last = state->num_clusters - 1;
    for (int r = 0; r < N; r++)
    {
        state->transition_matrix[last * N + r] = 0;
        state->transition_matrix[r * N + last] = 0;
    }
    memset(&state->clusters[last], 0, sizeof(Cluster));

    // 6. Correct Assignments Update Loop
    for (long f = 0; f < state->telemetry.total_frames_processed; f++)
    {
        int a = state->assignments[f];
        if (a == index_to_remove)
        {
            if (index_target == -1)
            {
                state->assignments[f] = -1;
            }
            else
            {
                if (index_target > index_to_remove)
                    state->assignments[f] = index_target - 1;
                else
                    state->assignments[f] = index_target;
            }
        }
        else if (a > index_to_remove)
        {
            state->assignments[f] = a - 1;
        }
    } // for (long f = 0; f < state->telemetry.total_frames_processed; f++)

    if (state->cluster_visitors != NULL)
    {
        for (int cl = 0; cl < state->num_clusters - 1; cl++)
        {
            for (int ii = 0; ii < state->cluster_visitors[cl].count; ii++)
            {
                int a = state->cluster_visitors[cl].records[ii].assignment;
                if (a == index_to_remove)
                {
                    if (index_target == -1)
                    {
                        state->cluster_visitors[cl].records[ii].assignment = -1;
                    }
                    else
                    {
                        state->cluster_visitors[cl].records[ii].assignment =
                            (index_target > index_to_remove)
                                ? index_target - 1
                                : index_target;
                    }
                }
                else if (a > index_to_remove)
                {
                    state->cluster_visitors[cl].records[ii].assignment = a - 1;
                }
            }
        }
    }

    // 7. Decrement Num Clusters
    state->num_clusters--;

    // 8. Recompute Geometric Consistency Mask and update DCC count
    recompute_consistency_mask(config, state);

    state->telemetry.dcc_entries_populated = dcc_count_populated_pairs(state);
    state->scratch.probsorted_count = 0;
}

/**
 * frame_assign_to_anchor() - Assign source frame attributes to cluster anchor.
 * @cluster:      Destination Cluster receiving anchor.
 * @source_frame: Source Frame to assign.
 *
 * Purpose & Context ("What is this used for?"):
 * Transfers frame attributes to a cluster anchor. When source_frame uses zero-copy
 * memory-mapped data (is_mmap = 1), allocates an aligned owned buffer so the anchor
 * maintains an independent buffer that can be freed cleanly during cluster cleanup.
 */
void frame_assign_to_anchor(
    Cluster *cluster,
    Frame   *source_frame)
{
    if (cluster == NULL || source_frame == NULL)
    {
        return;
    }

    cluster->anchor = *source_frame;
    if ((source_frame->is_mmap || source_frame->is_borrowed) && source_frame->data != NULL)
    {
        long dim = source_frame->width * source_frame->height;
        size_t elem_size = source_frame->is_double ? sizeof(double) : sizeof(float);
        void *anchor_buf = NULL;
        if (posix_memalign(&anchor_buf, 64, (size_t)dim * elem_size) == 0)
        {
            memcpy(anchor_buf, source_frame->data, (size_t)dim * elem_size);
            cluster->anchor.data = anchor_buf;
            cluster->anchor.is_mmap = 0;
            cluster->anchor.is_borrowed = 0;
        }
    }

    if (!source_frame->is_borrowed)
    {
        source_frame->data = NULL;
        source_frame->is_mmap = 0;
    }
}

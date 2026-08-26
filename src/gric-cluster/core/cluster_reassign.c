/**
 * @file cluster_reassign.c
 * @brief Second pass closest-anchor clustering and membership reallocation.
 *
 * Reallocates all processed frames to their globally nearest cluster anchor
 * after initial online clustering (Pass 1). Reuses distances already computed
 * in Pass 1, uses triangle-inequality bounds to prune distant candidate anchors,
 * evaluates remaining distances, and updates cluster assignments, transition
 * matrix, and membership logs.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_reassign.h"
#include "cluster_core.h"
#include "frameread.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**
 * run_second_pass_clustering() - Reassign all frames to their nearest cluster anchor.
 * @config: Pointer to the active ClusterConfig.
 * @state:  Pointer to the active ClusterState.
 *
 * Performs a second pass over all processed frames:
 * - Keeps already computed distances from Pass 1 in memory.
 * - Uses triangle-inequality lower bounds against inter-cluster anchor distances
 *   (DCC matrix) to prune anchors guaranteed to be farther than the current best anchor.
 * - Computes distances to remaining unmeasured cluster anchors.
 * - Reassigns each frame to the cluster anchor with the minimum distance.
 * - Updates assignments, frame_infos, transition matrix, and telemetry.
 * - Rewrites frame_membership.txt if membership logging is enabled.
 *
 * Return: Number of frames reassigned to a different cluster, or -1 on error.
 */
long run_second_pass_clustering(
    ClusterConfig *config,
    ClusterState  *state)
{
    if (state == NULL || config == NULL)
    {
        return -1;
    }

    int K = state->num_clusters;
    long N = state->telemetry.total_frames_processed;

    if (K <= 1 || N <= 0)
    {
        return 0;
    }

    /* Scratch buffers allocated once before the frame loop */
    double *frame_dists = (double *)malloc((size_t)K * sizeof(double));
    char   *measured    = (char *)malloc((size_t)K * sizeof(char));

    if (frame_dists == NULL || measured == NULL)
    {
        perror("Memory allocation failed for Pass 2 scratch buffers");
        if (frame_dists != NULL)
        {
            free(frame_dists);
        }
        if (measured != NULL)
        {
            free(measured);
        }
        return -1;
    }

    struct timespec p2_start, p2_end;
    clock_gettime(CLOCK_MONOTONIC, &p2_start);

    long frames_reassigned = 0;
    uint64_t new_dist_evals = 0;
    uint64_t dists_pruned   = 0;

    /* Iterate over each frame and reassign to nearest anchor */
    for (long t = 0; t < N; t++)
    {
        memset(measured, 0, (size_t)K * sizeof(char));
        for (int k = 0; k < K; k++)
        {
            frame_dists[k] = 1e30;
        }

        /* 1. Retrieve all distances computed during Pass 1 for frame t */
        if (state->frame_infos != NULL
            && state->frame_infos[t].cluster_indices != NULL
            && state->frame_infos[t].distances != NULL)
        {
            for (int i = 0; i < state->frame_infos[t].num_dists; i++)
            {
                int c = state->frame_infos[t].cluster_indices[i];
                if (c >= 0 && c < K)
                {
                    frame_dists[c] = state->frame_infos[t].distances[i];
                    measured[c] = 1;
                }
            }
        } // if frame_infos has recorded distances

        /* 2. Determine initial best distance from already measured anchors */
        int best_cl = state->assignments[t];
        double d_best = 1e30;

        if (best_cl >= 0 && best_cl < K && measured[best_cl])
        {
            d_best = frame_dists[best_cl];
        }
        else
        {
            for (int k = 0; k < K; k++)
            {
                if (measured[k] && frame_dists[k] < d_best)
                {
                    d_best = frame_dists[k];
                    best_cl = k;
                }
            }
        }

        /* 3. Check unmeasured anchors using triangle-inequality lower bounding */
        Frame *fr = NULL;
        for (int u = 0; u < K; u++)
        {
            if (measured[u])
            {
                continue;
            }

            /* Calculate lower bound on distance from frame t to anchor u */
            double lb = 0.0;
            for (int m = 0; m < K; m++)
            {
                if (!measured[m])
                {
                    continue;
                }

                size_t dcc_idx = (size_t)m * (size_t)config->algo.maxnbclust + (size_t)u;
                if (state->scratch.dcc_measured != NULL
                    && state->scratch.dcc_measured[dcc_idx]
                    && state->scratch.dcc_min != NULL)
                {
                    double dcc = state->scratch.dcc_min[dcc_idx];
                    if (dcc >= 0.0)
                    {
                        double bound = fabs(frame_dists[m] - dcc);
                        if (bound > lb)
                        {
                            lb = bound;
                        }
                    }
                }
            } // for (int m = 0; m < K; m++)

            if (lb >= d_best)
            {
                /* Anchor u cannot possibly be closer than d_best */
                dists_pruned++;
                continue;
            }

            /* Evaluate distance to anchor u */
            if (fr == NULL)
            {
                fr = getframe_at(t);
                if (fr == NULL)
                {
                    break;
                }
            }

            double d = get_dist(
                fr,
                &state->clusters[u].anchor,
                u,
                0.0,
                0.0,
                config,
                state);

            frame_dists[u] = d;
            measured[u] = 1;
            new_dist_evals++;

            if (d < d_best)
            {
                d_best = d;
                best_cl = u;
            }
        } // for (int u = 0; u < K; u++)

        if (fr != NULL)
        {
            free_frame(fr);
        }

        /* 4. Update assignment if a closer anchor was found */
        if (best_cl >= 0 && best_cl != state->assignments[t])
        {
            frames_reassigned++;
            state->assignments[t] = best_cl;
        }

        /* 5. Update in-memory FrameInfo for frame t with all measured distances */
        if (state->frame_infos != NULL)
        {
            state->frame_infos[t].assignment = best_cl;

            int measured_count = 0;
            for (int k = 0; k < K; k++)
            {
                if (measured[k])
                {
                    measured_count++;
                }
            }

            if (state->frame_infos[t].cluster_indices == NULL
                || state->frame_infos[t].num_dists != measured_count)
            {
                if (state->frame_infos[t].cluster_indices != NULL)
                {
                    free(state->frame_infos[t].cluster_indices);
                }
                if (state->frame_infos[t].distances != NULL)
                {
                    free(state->frame_infos[t].distances);
                }
                state->frame_infos[t].cluster_indices =
                    (int *)malloc((size_t)measured_count * sizeof(int));
                state->frame_infos[t].distances =
                    (double *)malloc((size_t)measured_count * sizeof(double));
            }

            if (state->frame_infos[t].cluster_indices != NULL
                && state->frame_infos[t].distances != NULL)
            {
                int out_idx = 0;
                for (int k = 0; k < K; k++)
                {
                    if (measured[k])
                    {
                        state->frame_infos[t].cluster_indices[out_idx] = k;
                        state->frame_infos[t].distances[out_idx] = frame_dists[k];
                        out_idx++;
                    }
                }
                state->frame_infos[t].num_dists = measured_count;
            }
        } // if state->frame_infos != NULL
    } // for (long t = 0; t < N; t++)

    free(frame_dists);
    free(measured);

    /* 6. Rebuild Transition Matrix to match the updated assignment sequence */
    if (state->transition_matrix != NULL)
    {
        size_t maxcl = (size_t)config->algo.maxnbclust;
        memset(state->transition_matrix, 0, maxcl * maxcl * sizeof(long));

        for (long t = 0; t < N - 1; t++)
        {
            int from = state->assignments[t];
            int to   = state->assignments[t + 1];
            if (from >= 0 && (size_t)from < maxcl
                && to >= 0 && (size_t)to < maxcl)
            {
                state->transition_matrix[(size_t)from * maxcl + (size_t)to]++;
            }
        }
    }

    /* 7. Rewrite frame_membership.txt if enabled */
    if (config->output.output_membership)
    {
        char out_path[1024];
        if (config->output.user_outdir != NULL)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt",
                     config->output.user_outdir);
        }
        else
        {
            snprintf(out_path, sizeof(out_path), "frame_membership.txt");
        }

        FILE *ascii_out = fopen(out_path, "w");
        if (ascii_out != NULL)
        {
            for (long t = 0; t < N; t++)
            {
                double best_d = 0.0;
                if (state->frame_infos != NULL
                    && state->frame_infos[t].distances != NULL)
                {
                    int a = state->assignments[t];
                    for (int i = 0; i < state->frame_infos[t].num_dists; i++)
                    {
                        if (state->frame_infos[t].cluster_indices[i] == a)
                        {
                            best_d = state->frame_infos[t].distances[i];
                            break;
                        }
                    }
                }
                fprintf(ascii_out, "%ld %d %.6f\n", t, state->assignments[t], best_d);
            }
            fclose(ascii_out);
        }
    } // if output_membership

    clock_gettime(CLOCK_MONOTONIC, &p2_end);
    double p2_ms = (p2_end.tv_sec - p2_start.tv_sec) * 1000.0 +
                   (p2_end.tv_nsec - p2_start.tv_nsec) / 1000000.0;

    state->telemetry.time_pass2 = p2_ms;
    state->telemetry.pass2_frames_reassigned = (uint64_t)frames_reassigned;
    state->telemetry.pass2_dist_evals = new_dist_evals;
    state->telemetry.pass2_dist_pruned = dists_pruned;

    double pct_reassigned = (N > 0) ? (100.0 * (double)frames_reassigned / (double)N) : 0.0;
    printf("\nSecond Pass (Nearest Anchor Reallocation):\n");
    printf("  Frames reassigned:     %8ld / %ld (%5.1f%%)\n",
           frames_reassigned, N, pct_reassigned);
    printf("  New distance calls:    %8lu\n", (unsigned long)new_dist_evals);
    printf("  Distance calls pruned: %8lu\n", (unsigned long)dists_pruned);
    printf("  Second pass time:      %9.3f ms\n", p2_ms);

    return frames_reassigned;
}

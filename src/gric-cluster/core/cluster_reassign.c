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
#include "frame_info_arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP


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
#ifdef USE_CUDA
#include "cluster_cuda.h"
#endif

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

#ifdef USE_CUDA
    if (config->optim.use_gpu)
    {
        if (cluster_cuda_is_available())
        {
            long res = cluster_cuda_run_pass2(config, state);
            if (res >= 0)
            {
                return res;
            }
            fprintf(stderr, "Warning: GPU Pass 2 failed, falling back to CPU Pass 2.\n");
        }
        else
        {
            fprintf(stderr, "Warning: CUDA GPU requested for Pass 2 but no available device found. "
                            "Falling back to CPU.\n");
        }
    }
#else
    if (config->optim.use_gpu)
    {
        fprintf(stderr, "Warning: GPU acceleration requested (--gpu), but gric-cluster was built "
                        "without CUDA support (ENABLE_CUDA=OFF). Running on CPU.\n");
    }
#endif

    struct timespec p2_start, p2_end;
    clock_gettime(CLOCK_MONOTONIC, &p2_start);

    long frames_reassigned = 0;
    uint64_t new_dist_evals = 0;
    uint64_t dists_pruned   = 0;

    /* Iterate over each frame and reassign to nearest anchor using OpenMP */
#ifdef _OPENMP
#pragma omp parallel reduction(+:frames_reassigned, new_dist_evals, dists_pruned)
#endif
    {
        double *frame_dists = (double *)malloc((size_t)K * sizeof(double));
        char   *measured    = (char *)malloc((size_t)K * sizeof(char));
        int    *measured_indices = (int *)malloc((size_t)K * sizeof(int));
        Frame   local_fr;
        memset(&local_fr, 0, sizeof(local_fr));

        if (frame_dists != NULL && measured != NULL && measured_indices != NULL)
        {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
            for (long t = 0; t < N; t++)
            {
                memset(measured, 0, (size_t)K * sizeof(char));
                int num_measured = 0;
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
                            measured_indices[num_measured++] = c;
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
                    for (int mi = 0; mi < num_measured; mi++)
                    {
                        int k = measured_indices[mi];
                        if (frame_dists[k] < d_best)
                        {
                            d_best = frame_dists[k];
                            best_cl = k;
                        }
                    } // for (int mi = 0; mi < num_measured; mi++)
                }

                /* 3. Check unmeasured anchors using triangle-inequality lower bounding */
                bool fr_loaded = false;
                for (int u = 0; u < K; u++)
                {
                    if (measured[u])
                    {
                        continue;
                    }

                    /* Calculate lower bound on distance from frame t to anchor u */
                    double lb = 0.0;
                    size_t u_row_offset = (size_t)u * (size_t)config->algo.maxnbclust;
                    const double *dcc_row = (state->scratch.dcc_min != NULL)
                        ? &state->scratch.dcc_min[u_row_offset]
                        : NULL;
                    const char *measured_row = (state->scratch.dcc_measured != NULL)
                        ? &state->scratch.dcc_measured[u_row_offset]
                        : NULL;

                    if (dcc_row != NULL && measured_row != NULL)
                    {
                        for (int mi = 0; mi < num_measured; mi++)
                        {
                            int m = measured_indices[mi];
                            if (measured_row[m])
                            {
                                double dcc = dcc_row[m];
                                if (dcc >= 0.0)
                                {
                                    double bound = fabs(frame_dists[m] - dcc);
                                    if (bound > lb)
                                    {
                                        lb = bound;
                                        if (lb >= d_best)
                                        {
                                            break;
                                        }
                                    }
                                }
                            }
                        } // for (int mi = 0; mi < num_measured; mi++)
                    } // if (dcc_row != NULL && measured_row != NULL)

                    if (lb >= d_best)
                    {
                        /* Anchor u cannot possibly be closer than d_best */
                        dists_pruned++;
                        continue;
                    }

                    /* Evaluate distance to anchor u using lock-free buffer */
                    if (!fr_loaded)
                    {
                        if (getframe_at_buf(&local_fr, t) != 0)
                        {
                            break;
                        }
                        fr_loaded = true;
                    }

                    double d = get_dist(
                        &local_fr,
                        &state->clusters[u].anchor,
                        u,
                        0.0,
                        0.0,
                        config,
                        state);

                    frame_dists[u] = d;
                    measured[u] = 1;
                    measured_indices[num_measured++] = u;
                    new_dist_evals++;

                    if (d < d_best)
                    {
                        d_best = d;
                        best_cl = u;
                    }
                } // for (int u = 0; u < K; u++)

                if (fr_loaded)
                {
                    release_frame_buf(&local_fr);
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
                    } // for (int k = 0; k < K; k++)

                    if (state->frame_infos[t].cluster_indices == NULL
                        || state->frame_infos[t].num_dists != measured_count)
                    {
#ifdef _OPENMP
#pragma omp critical(frame_info_arena)
#endif
                        {
                            frame_info_arena_alloc_records(
                                &state->frame_info_arena,
                                measured_count,
                                &state->frame_infos[t].cluster_indices,
                                &state->frame_infos[t].distances);
                        }
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
                        } // for (int k = 0; k < K; k++)
                        state->frame_infos[t].num_dists = measured_count;
                    }
                } // if (state->frame_infos != NULL)
            } // for (long t = 0; t < N; t++)

            free(frame_dists);
            free(measured);
            free(measured_indices);
        }
        else
        {
            if (frame_dists != NULL)
            {
                free(frame_dists);
            }
            if (measured != NULL)
            {
                free(measured);
            }
            if (measured_indices != NULL)
            {
                free(measured_indices);
            }
        }
    } // OpenMP parallel region

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
    if (config->output.output_membership && !config->output.no_txt)
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
            setvbuf(ascii_out, NULL, _IOFBF, 65536);
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

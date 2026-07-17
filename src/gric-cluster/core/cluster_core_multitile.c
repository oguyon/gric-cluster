/**
 * @file cluster_core_multitile.c
 * @brief Multi-tile clustering orchestrator using OpenMP tasks.
 *
 * Reads full-image frames, scatters pixels into per-tile
 * sub-frames, dispatches parallel Independent Spatial Clustering (Pass 1) tasks
 * via cluster_frame(), and records the assignment tuple for
 * each frame. Joint Trajectory Fusion (Pass 2) is deferred to Phase 6.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_core_multitile.h"
#include "tile_state.h"
#include "frame_scatter.h"
#include "cluster_step.h"
#include "cluster_core.h"
#include "cluster_io.h"
#include "cluster_io_multitile.h"
#include "frameread.h"
#include "tuple_retrieval.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct
{
    long frame_idx;
    const int *tuple;
} TupleSortItem;

static int compare_num_tiles = 0;

static int compare_tuples(
    const void *a,
    const void *b)
{
    const TupleSortItem *itemA = (const TupleSortItem *)a;
    const TupleSortItem *itemB = (const TupleSortItem *)b;
    for (int m = 0; m < compare_num_tiles; m++)
    {
        if (itemA->tuple[m] < itemB->tuple[m])
        {
            return -1;
        }
        if (itemA->tuple[m] > itemB->tuple[m])
        {
            return 1;
        }
    }
    return 0;
}


/**
 * make_task_frame() - Create a heap-allocated Frame for one tile task.
 * @tile_frame: Pre-allocated tile sub-frame (scatter buffer).
 *
 * Allocates a new Frame struct on the heap, copies metadata from
 * @tile_frame, malloc's a fresh data buffer, and memcpy's the
 * tile pixel data into it. The resulting Frame is safe to pass
 * to cluster_frame(), which will call free_frame() on it.
 *
 * Return: Pointer to newly allocated Frame, or NULL on failure.
 */
static Frame *make_task_frame(const Frame *tile_frame)
{
    Frame *tf = malloc(sizeof(*tf));
    if (tf == NULL)
    {
        return NULL;
    }

    tf->id    = tile_frame->id;
    tf->cnt0  = tile_frame->cnt0;
    tf->atime = tile_frame->atime;
    tf->width  = tile_frame->width;
    tf->height = tile_frame->height;

    /*
     * Allocate at the full-image buffer size so that
     * free_frame()'s pool returns correctly-sized
     * buffers to getframe().
     */
    long full_w = get_frame_width();
    long full_h = get_frame_height();
    size_t pool_bytes =
        (size_t)(full_w * full_h) * sizeof(double);
    size_t tile_bytes =
        (size_t)(tile_frame->width
                 * tile_frame->height)
        * sizeof(double);

    if (pool_bytes < tile_bytes)
    {
        pool_bytes = tile_bytes;
    }

    tf->data = calloc(1, pool_bytes);
    if (tf->data == NULL)
    {
        free(tf);
        return NULL;
    }

    memcpy(tf->data, tile_frame->data, tile_bytes);
    return tf;
}


/**
 * run_clustering_multitile() - Multi-tile parallel clustering.
 * @global_config: Global clustering configuration (ncpu, etc.).
 * @mts:           Initialised MultiTileState with per-tile states.
 *
 * Outer loop reads full-image frames sequentially, scatters
 * each into per-tile sub-frames, then dispatches M independent
 * OpenMP tasks for Independent Spatial Clustering (Pass 1).  Each task calls
 * cluster_frame() on a freshly copied Frame so the internal
 * free_frame() does not touch the reusable scatter buffers.
 *
 * After all tasks complete (taskwait barrier), the per-tile
 * assignments are recorded in mts->tuple_history.
 *
 * After all frames are processed, Joint Trajectory Fusion (Pass 2)
 * is executed to resolve tile boundary noise using global
 * historical sequence transitions.
 */
void run_clustering_multitile(
    ClusterConfig  *global_config,
    MultiTileState *mts)
{
    int num_tiles = mts->num_tiles;
    int ncpu      = global_config->optim.ncpu;

    if (ncpu < 1)
    {
        ncpu = 1;
    }

    long actual_frames = get_num_frames();
    if (actual_frames > global_config->input.maxnbfr)
    {
        actual_frames = global_config->input.maxnbfr;
    }

    printf("Multi-tile clustering: %d tiles, %ld frames, "
           "%d threads\n",
           num_tiles, actual_frames, ncpu);

#ifdef _OPENMP
    /*
     * Allow nested parallelism when tiles < cpus so that
     * each tile task can still use intra-tile OpenMP
     * parallelism (e.g. in framedist).
     */
    if (num_tiles < ncpu)
    {
        omp_set_max_active_levels(2);
    }
#endif

    /* Allocate per-tile scatter buffers */
    Frame *scatter_buf =
        calloc((size_t) num_tiles, sizeof(Frame));
    if (scatter_buf == NULL)
    {
        fprintf(stderr,
            "ERROR: scatter buffer alloc failed\n");
        return;
    }
    frame_scatter_alloc(mts->tile_map, scatter_buf);

    /* ---- Open membership file ---- */
    FILE *membership_out = NULL;
    if (global_config->output.output_membership)
    {
        char out_path[1024];
        if (global_config->output.user_outdir)
        {
            snprintf(out_path, sizeof(out_path),
                     "%s/frame_membership.txt",
                     global_config->output.user_outdir);
        }
        else
        {
            char *out_dir = create_output_dir_name(
                global_config->input.fits_filename);
            if (out_dir)
            {
                safe_mkdir(out_dir);
                snprintf(out_path, sizeof(out_path),
                         "%s/frame_membership.txt",
                         out_dir);
                free(out_dir);
            }
            else
            {
                snprintf(out_path, sizeof(out_path),
                         "frame_membership.txt");
            }
        } // if user_outdir
        membership_out = fopen(out_path, "w");
        if (membership_out)
        {
            /* Write header with tile column names */
            fprintf(membership_out, "# frame");
            for (int m = 0; m < num_tiles; m++)
            {
                fprintf(membership_out,
                        "  tile_%d", m);
            }
            fprintf(membership_out, "\n");
        }
    } // if output_membership

    struct timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    long frames_done = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(ncpu)
{
#pragma omp single
{
#endif
    for (long fi = 0; fi < actual_frames; fi++)
    {
        if (stop_requested)
        {
            break;
        }

        /* ---- Read full-image frame ---- */
        Frame *src = getframe();
        if (src == NULL)
        {
            break;
        }

        /* ---- Scatter into per-tile buffers ---- */
        frame_scatter(src, mts->tile_map, scatter_buf);

        /* ---- Predict joint transitions for Pass 1 priors & candidates ---- */
        if (global_config->optim.pred_mode)
        {
            predict_joint_tuples(
                mts,
                global_config->optim.pred_len,
                global_config->optim.pred_h,
                global_config->optim.pred_n);
        }

        /* ---- Dispatch M Pass 1 tasks ---- */
        for (int m = 0; m < num_tiles; m++)
        {
            TileState *ts = &mts->tile_states[m];
            Frame *task_frame =
                make_task_frame(&scatter_buf[m]);
            if (task_frame == NULL)
            {
                fprintf(stderr,
                    "ERROR: make_task_frame tile %d\n",
                    m);
                continue;
            }

#ifdef _OPENMP
#pragma omp task firstprivate(m, ts, task_frame)
{
#endif
            ts->pass1_old_ncl = ts->state.num_clusters;
            for (int i = 0; i < ts->config.algo.maxnbclust; i++)
            {
                ts->temp_indices[i] = -1;
            }
            int res = cluster_frame(
                &ts->config,
                &ts->state,
                task_frame,
                &ts->prev_assigned_cluster,
                ts->ascii_out,
                ts->temp_indices,
                ts->temp_dists,
                ts->sorting_candidates,
                ts->verbose_candidates);

            if (res >= 0)
            {
                ts->pass1_assignment = res;
            }
            else
            {
                ts->pass1_assignment = -1;
            }

            /* Copy or construct posterior for Pass 2 fusion */
            if (ts->pass1_posterior)
            {
                int ncl = ts->state.num_clusters;
                if (ncl > ts->config.algo.maxnbclust)
                {
                    ncl = ts->config.algo.maxnbclust;
                }

                if (ts->config.optim.entropy_mode && ts->state.scratch.entropy_p_current)
                {
                    memcpy(
                        ts->pass1_posterior,
                        ts->state.scratch.entropy_p_current,
                        (size_t) ncl * sizeof(double));
                }
                else
                {
                    /* Construct soft posterior around Pass 1 assignment */
                    int res = ts->pass1_assignment;
                    if (res < 0 || res >= ncl)
                    {
                        /* Flat distribution if no valid assignment */
                        for (int k = 0; k < ncl; k++)
                        {
                            ts->pass1_posterior[k] = 1.0 / ncl;
                        }
                    }
                    else if (ncl == 1)
                    {
                        ts->pass1_posterior[0] = 1.0;
                    }
                    else
                    {
                        double epsilon = 0.1; /* 10% weight open for spatial/temporal corrections */
                        double sum_others = 0.0;
                        for (int k = 0; k < ncl; k++)
                        {
                            if (k != res)
                            {
                                double prior = ts->state.scratch.mixed_probs ? ts->state.scratch.mixed_probs[k] : 1.0;
                                ts->pass1_posterior[k] = prior;
                                sum_others += prior;
                            }
                        }

                        if (sum_others > 0.0)
                        {
                            for (int k = 0; k < ncl; k++)
                            {
                                if (k == res)
                                {
                                    ts->pass1_posterior[k] = 1.0 - epsilon;
                                }
                                else
                                {
                                    ts->pass1_posterior[k] = epsilon * (ts->pass1_posterior[k] / sum_others);
                                }
                            }
                        }
                        else
                        {
                            for (int k = 0; k < ncl; k++)
                            {
                                if (k == res)
                                {
                                    ts->pass1_posterior[k] = 1.0 - epsilon;
                                }
                                else
                                {
                                    ts->pass1_posterior[k] = epsilon / (ncl - 1);
                                }
                            }
                        }
                    }
                }
            }
#ifdef _OPENMP
} // omp task
#endif
        } // for each tile m

#ifdef _OPENMP
#pragma omp taskwait
#endif

        /* ---- Joint Trajectory Fusion (Pass 2) ---- */
        if (mts->tuple_count > 0 && num_tiles > 1 && !global_config->optim.disable_pass2)
        {
            for (int m = 0; m < num_tiles; m++)
            {
#ifdef _OPENMP
#pragma omp task firstprivate(m)
{
#endif
                pass2_fuse(mts, m, &scatter_buf[m]);
#ifdef _OPENMP
} // omp task
#endif
            } // for each tile m (Pass 2)

#ifdef _OPENMP
#pragma omp taskwait
#endif
        } // if tuple_count > 0

        /* ---- Record assignment tuple ---- */
        {
            long base = mts->tuple_count * (long) num_tiles;
            long t = mts->tuple_count;
            int maxcl = mts->tile_states[0].config.algo.maxnbclust;

            for (int m = 0; m < num_tiles; m++)
            {
                int ass = mts->tile_states[m].pass1_assignment;
                mts->tuple_history[base + m] = ass;
                if (ass >= 0 && ass < maxcl)
                {
                    mts->occurrence_prev[t * (long)num_tiles + m] =
                        mts->occurrence_head[m * maxcl + ass];
                    mts->occurrence_head[m * maxcl + ass] = (int)t;
                }
            }
            mts->tuple_count++;
        }

        /* ---- Write membership line ---- */
        if (membership_out)
        {
            fprintf(membership_out, "%ld", fi);
            for (int m = 0; m < num_tiles; m++)
            {
                fprintf(membership_out, "  %d",
                    mts->tile_states[m]
                        .pass1_assignment);
            }
            fprintf(membership_out, "\n");
        }

        /* ---- Free source frame ---- */
        free_frame(src);
        frames_done++;

        /* ---- Progress ---- */
        if (frames_done % 100 == 0
            || frames_done == actual_frames)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed_s =
                (now.tv_sec - wall_start.tv_sec)
                + (now.tv_nsec - wall_start.tv_nsec)
                  / 1.0e9;
            double fps =
                (elapsed_s > 0.0)
                ? (double) frames_done / elapsed_s
                : 0.0;
            printf("\rMulti-tile: frame %ld / %ld  "
                   "(%.1f fps)",
                   frames_done, actual_frames, fps);
            fflush(stdout);
        }
    } // for each frame fi
#ifdef _OPENMP
} // omp single
} // omp parallel
#endif

    printf("\n");

    /* ---- Summary ---- */
    {
        struct timespec wall_end;
        clock_gettime(CLOCK_MONOTONIC, &wall_end);
        double total_ms =
            (wall_end.tv_sec - wall_start.tv_sec)
              * 1000.0
            + (wall_end.tv_nsec - wall_start.tv_nsec)
              / 1.0e6;

        printf("Multi-tile analysis complete.\n");
        printf("Frames processed: %ld\n", frames_done);
        printf("Tuples recorded:  %ld\n",
               mts->tuple_count);
        printf("Wall time:        %.3f ms\n", total_ms);

        long total_dfc = 0;
        long total_dcc = 0;
        for (int m = 0; m < num_tiles; m++)
        {
            ClusterTelemetry *t =
                &mts->tile_states[m].state.telemetry;
            long dfc = t->framedist_calls_sample;
            long dcc =
                t->framedist_calls_intercluster;
            total_dfc += dfc;
            total_dcc += dcc;
            printf("  Tile %3d: %d clusters, "
                   "dfc=%ld, dcc=%ld\n",
                   m,
                   mts->tile_states[m]
                       .state.num_clusters,
                   dfc, dcc);
            print_clustering_metrics(&mts->tile_states[m].state, m);
        }
        printf("Total framedist: %ld "
               "(dfc=%ld, dcc=%ld)\n",
               total_dfc + total_dcc,
               total_dfc, total_dcc);

        if (mts->tuple_count > 0)
        {
            TupleSortItem *items = (TupleSortItem *)malloc((size_t)mts->tuple_count *
                                                          sizeof(TupleSortItem));
            if (items != NULL)
            {
                for (long t = 0; t < mts->tuple_count; t++)
                {
                    items[t].frame_idx = t;
                    items[t].tuple = &mts->tuple_history[t * num_tiles];
                }

                compare_num_tiles = num_tiles;
                qsort(items, (size_t)mts->tuple_count, sizeof(TupleSortItem),
                      compare_tuples);

                long unique_tuples = 0;
                double tuple_entropy = 0.0;
                double sum_tuple_rms = 0.0;
                double total_joint_sum_sq = 0.0;
                long total_assigned_frames = 0;
                double global_joint_max_dist = 0.0;

                long start = 0;
                while (start < mts->tuple_count)
                {
                    long end = start + 1;
                    while (end < mts->tuple_count &&
                           compare_tuples(&items[start], &items[end]) == 0)
                    {
                        end++;
                    }

                    long count = end - start;
                    double sum_sq = 0.0;
                    int valid_tuple = 1;

                    for (long i = start; i < end; i++)
                    {
                        long t = items[i].frame_idx;
                        double frame_dist2 = 0.0;
                        for (int m = 0; m < num_tiles; m++)
                        {
                            int assigned_cl = items[i].tuple[m];
                            if (assigned_cl < 0)
                            {
                                valid_tuple = 0;
                                break;
                            }
                            double d = 0.0;
                            TileState *ts = &mts->tile_states[m];
                            for (int d_idx = 0;
                                 d_idx < ts->state.frame_infos[t].num_dists;
                                 d_idx++)
                            {
                                if (ts->state.frame_infos[t].cluster_indices[d_idx] ==
                                    assigned_cl)
                                {
                                    d = ts->state.frame_infos[t].distances[d_idx];
                                    break;
                                }
                            }
                            frame_dist2 += d * d;
                        }
                        if (valid_tuple)
                        {
                            sum_sq += frame_dist2;
                            double joint_dist = sqrt(frame_dist2);
                            if (joint_dist > global_joint_max_dist)
                            {
                                global_joint_max_dist = joint_dist;
                            }
                        }
                    }

                    if (valid_tuple)
                    {
                        unique_tuples++;
                        total_joint_sum_sq += sum_sq;
                        total_assigned_frames += count;
                        double rms = sqrt(sum_sq / (double)count);
                        sum_tuple_rms += rms;

                        double p = (double)count / (double)mts->tuple_count;
                        tuple_entropy -= p * log2(p);
                    }

                    start = end;
                }

                double global_joint_rms = 0.0;
                if (total_assigned_frames > 0)
                {
                    global_joint_rms = sqrt(total_joint_sum_sq /
                                            (double)total_assigned_frames);
                }
                double avg_tuple_rms = (unique_tuples > 0) ?
                    (sum_tuple_rms / (double)unique_tuples) : 0.0;

                printf("Global Joint System Metrics:\n");
                printf("    Unique Tuples (states): %ld\n", unique_tuples);
                printf("    Global Joint RMS Dist:  %.4f  (Avg tuple RMS: %.4f, Max: %.4f)\n",
                       global_joint_rms, avg_tuple_rms, global_joint_max_dist);
                printf("    Joint Tuple Entropy:    %.4f bits\n", tuple_entropy);

                free(items);
            }
        }
    }

    /* ---- Close membership file ---- */
    if (membership_out)
    {
        fclose(membership_out);
        printf("Multi-tile membership written\n");
    }

    /* ---- Write per-tile results ---- */
    write_results_multitile(global_config, mts);

    /* ---- Cleanup scatter buffers ---- */
    frame_scatter_free(scatter_buf, num_tiles);
    free(scatter_buf);
}

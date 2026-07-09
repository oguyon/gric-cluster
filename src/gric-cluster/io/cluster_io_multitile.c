/**
 * @file cluster_io_multitile.c
 * @brief Results serialization for multi-tile clustering.
 *
 * Creates per-tile output subdirectories and writes per-tile
 * anchor text files, cluster counts, and tuple history.
 */

#include "cluster_io_multitile.h"
#include "cluster_io.h"
#include "tile_state.h"
#include "frameread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif


/**
 * get_out_dir() - Resolve the output directory path.
 * @config: Global clustering configuration.
 *
 * Returns a heap-allocated directory name string using the
 * user-specified output directory if set, otherwise deriving
 * one from the input filename via create_output_dir_name().
 *
 * Return: Heap-allocated string, or NULL on failure.
 *         Caller must free().
 */
static char *get_out_dir(ClusterConfig *config)
{
    if (config->output.user_outdir)
    {
        return strdup(config->output.user_outdir);
    }
    return create_output_dir_name(
        config->input.fits_filename);
}


/**
 * write_tile_anchors() - Write per-tile anchor text file.
 * @tile_dir: Path to the tile subdirectory.
 * @ts:       Per-tile state containing clusters.
 *
 * Writes one line per cluster anchor, with space-separated
 * pixel values. The tile's frame dimensions are used to
 * determine the number of elements per anchor.
 */
static void write_tile_anchors(
    const char      *tile_dir,
    const TileState *ts)
{
    char path[1024];
    snprintf(path, sizeof(path),
             "%s/anchors.txt", tile_dir);

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        return;
    }

    long nelements =
        (long) ts->tile_frame.width
        * (long) ts->tile_frame.height;

    for (int c = 0; c < ts->state.num_clusters; c++)
    {
        const double *data =
            ts->state.clusters[c].anchor.data;
        for (long k = 0; k < nelements; k++)
        {
            fprintf(fp, "%f ", data[k]);
        }
        fprintf(fp, "\n");
    } // for each cluster c

    fclose(fp);
}


/**
 * write_tile_counts() - Write per-tile cluster counts.
 * @tile_dir: Path to the tile subdirectory.
 * @ts:       Per-tile state containing assignments.
 */
static void write_tile_counts(
    const char      *tile_dir,
    const TileState *ts)
{
    char path[1024];
    snprintf(path, sizeof(path),
             "%s/cluster_counts.txt", tile_dir);

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        return;
    }

    int ncl = ts->state.num_clusters;
    int maxcl = ts->config.algo.maxnbclust;

    /* Count assignments per cluster */
    int *counts = calloc((size_t) maxcl, sizeof(int));
    if (counts == NULL)
    {
        fclose(fp);
        return;
    }

    long nframes =
        ts->state.telemetry.total_frames_processed;
    for (long f = 0; f < nframes; f++)
    {
        int a = ts->state.assignments[f];
        if (a >= 0 && a < maxcl)
        {
            counts[a]++;
        }
    }

    for (int c = 0; c < ncl; c++)
    {
        fprintf(fp, "Cluster %d: %d frames\n",
                c, counts[c]);
    }

    free(counts);
    fclose(fp);
}


/**
 * write_tuple_history() - Write tuple history file.
 * @out_dir: Main output directory path.
 * @mts:     Multi-tile state with tuple_history.
 *
 * Writes one line per frame with M columns of per-tile
 * cluster assignments. First column is the frame index.
 */
static void write_tuple_history(
    const char           *out_dir,
    const MultiTileState *mts)
{
    char path[1024];
    snprintf(path, sizeof(path),
             "%s/tuple_history.txt", out_dir);

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        return;
    }

    int num_tiles = mts->num_tiles;

    /* Header */
    fprintf(fp, "# frame");
    for (int m = 0; m < num_tiles; m++)
    {
        fprintf(fp, "  tile_%d", m);
    }
    fprintf(fp, "\n");

    /* Data rows */
    for (long f = 0; f < mts->tuple_count; f++)
    {
        long base = f * (long) num_tiles;
        fprintf(fp, "%ld", f);
        for (int m = 0; m < num_tiles; m++)
        {
            fprintf(fp, "  %d",
                    mts->tuple_history[base + m]);
        }
        fprintf(fp, "\n");
    } // for each frame f

    fclose(fp);
}


/**
 * write_results_multitile() - Write per-tile output files.
 * @config: Global clustering configuration.
 * @mts:    Multi-tile state with per-tile results.
 *
 * Creates the output directory and per-tile subdirectories,
 * then writes anchor text files, cluster counts, and the
 * global tuple history. FITS anchor output is guarded by
 * USE_CFITSIO but currently a stub.
 */
void write_results_multitile(
    ClusterConfig  *config,
    MultiTileState *mts)
{
    char *out_dir = get_out_dir(config);
    if (out_dir == NULL)
    {
        return;
    }

    safe_mkdir(out_dir);

    int num_tiles = mts->num_tiles;

    for (int m = 0; m < num_tiles; m++)
    {
        /* Create tile subdirectory */
        char tile_dir[1024];
        snprintf(tile_dir, sizeof(tile_dir),
                 "%s/tile_%d", out_dir, m);
        safe_mkdir(tile_dir);

        TileState *ts = &mts->tile_states[m];

        if (config->output.output_anchors)
        {
            if (is_ascii_input_mode() && !config->output.fitsout_mode)
            {
                printf("  Tile %d: writing anchors.txt\n", m);
                write_tile_anchors(tile_dir, ts);
            }
            else
            {
#ifdef USE_CFITSIO
                printf("  Tile %d: writing anchors.fits\n", m);
                int status = 0;
                fitsfile *afptr;
                char out_path[2048];
                snprintf(out_path, sizeof(out_path), "!%s/anchors.fits", tile_dir);
                fits_create_file(&afptr, out_path, &status);
                long naxes[3] = {ts->tile_frame.width, ts->tile_frame.height, ts->state.num_clusters};
                fits_create_img(afptr, DOUBLE_IMG, 3, naxes, &status);

                long nelements = ts->tile_frame.width * ts->tile_frame.height;
                for (int i = 0; i < ts->state.num_clusters; i++)
                {
                    long fpixel[3] = {1, 1, i + 1};
                    fits_write_pix(afptr, TDOUBLE, fpixel, nelements,
                                   ts->state.clusters[i].anchor.data, &status);
                }
                fits_close_file(afptr, &status);
#else
                printf("  Tile %d: writing anchors.txt (FITS disabled)\n", m);
                write_tile_anchors(tile_dir, ts);
#endif
            }
        }

        if (config->output.output_counts)
        {
            printf("  Tile %d: writing counts\n", m);
            write_tile_counts(tile_dir, ts);
        }

        if (config->output.output_dcc)
        {
            /* Per-tile dcc.txt */
            char dcc_path[2048];
            snprintf(dcc_path, sizeof(dcc_path),
                     "%s/dcc.txt", tile_dir);
            FILE *dcc_fp = fopen(dcc_path, "w");
            if (dcc_fp)
            {
                int ncl = ts->state.num_clusters;
                int maxcl =
                    ts->config.algo.maxnbclust;
                for (int i = 0; i < ncl; i++)
                {
                    for (int j = 0; j < ncl; j++)
                    {
                        int idx =
                            i * maxcl + j;
                        double d =
                            ts->state.scratch
                                .dcc_min[idx];
                        int meas =
                            ts->state.scratch
                                .dcc_measured[idx];
                        if (meas && d >= 0)
                        {
                            fprintf(dcc_fp,
                                "%d %d %.6f\n",
                                i, j, d);
                        }
                    }
                } // for i
                fclose(dcc_fp);
            } // if dcc_fp
        } // if output_dcc
    } // for each tile m

    /* Write global tuple history */
    printf("Writing tuple_history.txt\n");
    write_tuple_history(out_dir, mts);

    printf("Multi-tile results written to %s/\n",
           out_dir);
    free(out_dir);
}

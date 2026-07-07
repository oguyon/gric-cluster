/**
 * @file tile_state.c
 * @brief Implementation of per-tile clustering state
 *        allocation, teardown, and configuration loading.
 */

#include "tile_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



/**
 * multitile_init - Allocate and initialise multi-tile state.
 * @global:  Global clustering configuration (copied into each tile).
 * @tm:      Tile map describing the spatial partition.
 * @maxnbfr: Maximum number of frames to record.
 *
 * Allocates a MultiTileState and one TileState per tile.
 * Each TileState receives a copy of @global config and
 * pre-allocated scratch buffers sized by maxnbclust.
 * The caller retains ownership of @tm.
 *
 * Return: Pointer to the new MultiTileState, or NULL on
 *         allocation failure.
 */
MultiTileState *multitile_init(
    ClusterConfig *global,
    TileMap       *tm,
    long           maxnbfr)
{
    if (global == NULL || tm == NULL)
    {
        return NULL;
    }

    MultiTileState *mts = calloc(1, sizeof(*mts));
    if (mts == NULL)
    {
        return NULL;
    }

    mts->tile_map  = tm;
    mts->num_tiles = tm->num_tiles;

    /* Allocate array of per-tile states */
    mts->tile_states = calloc(
        (size_t) mts->num_tiles, sizeof(TileState));
    if (mts->tile_states == NULL)
    {
        free(mts);
        return NULL;
    }

    int maxnbc = global->algo.maxnbclust;

    /* Initialise each tile state */
    for (int m = 0; m < mts->num_tiles; m++)
    {
        TileState *ts = &mts->tile_states[m];

        ts->tile_id  = m;
        ts->tile_def = &tm->tiles[m];

        /* Per-tile copy of global config */
        ts->config = *global;

        /* Allocate ClusterState buffers */
        {
            size_t mc = (size_t) maxnbc;
            size_t pairs = mc * mc;
            int words = (maxnbc + 63) / 64;
            size_t cw = pairs * (size_t) words;

            memset(&ts->state, 0, sizeof(ts->state));

            ts->state.clusters = malloc(
                mc * sizeof(Cluster));
            ts->state.cluster_visitors = calloc(
                mc, sizeof(VisitorList));
            ts->state.scratch.dcc_min = malloc(
                pairs * sizeof(double));
            ts->state.scratch.dcc_max = malloc(
                pairs * sizeof(double));
            ts->state.scratch.dcc_measured = malloc(
                pairs * sizeof(char));
            ts->state.scratch.current_gprobs = malloc(
                mc * sizeof(double));
            ts->state.scratch.probsortedclindex = malloc(
                mc * sizeof(int));
            ts->state.scratch.clmembflag = malloc(
                mc * sizeof(int));
            ts->state.scratch.mixed_probs = calloc(
                mc, sizeof(double));
            ts->state.scratch.consistency_mask = calloc(
                cw, sizeof(uint64_t));
            ts->state.scratch.entropy_p_current = malloc(
                mc * sizeof(double));
            ts->state.scratch.entropy_candidates = malloc(
                mc * sizeof(Candidate));
            ts->state.scratch.entropy_prob_scores = malloc(
                mc * sizeof(TargetScore));
            ts->state.scratch.entropy_prune_scores = malloc(
                mc * sizeof(TargetScore));
            ts->state.scratch.entropy_active_indices = malloc(
                mc * sizeof(int));
            ts->state.scratch.entropy_plog2p = malloc(
                mc * sizeof(double));
            ts->state.scratch.entropy_visited = malloc(
                mc * sizeof(uint8_t));
            ts->state.scratch.refine_queue = malloc(
                1024 * sizeof(Candidate));
            ts->state.scratch.refine_queue_capacity = 1024;
            ts->state.scratch.tuple_pred_candidates = malloc(
                mc * sizeof(int));
            ts->state.scratch.tuple_pred_count = 0;

            /* Init DCC bounds */
            for (size_t ii = 0; ii < pairs; ii++)
            {
                ts->state.scratch.dcc_min[ii] = -1.0;
                ts->state.scratch.dcc_max[ii] = -1.0;
                ts->state.scratch.dcc_measured[ii] = 0;
            }

            ts->state.transition_matrix = calloc(
                pairs, sizeof(long));

            /* Assignments and frame info arrays */
            ts->state.assignments = malloc(
                (size_t) maxnbfr * sizeof(int));
            ts->state.frame_infos = calloc(
                (size_t) maxnbfr, sizeof(FrameInfo));

            /* Telemetry tracking arrays */
            ts->state.telemetry.max_steps_recorded = maxnbc;
            ts->state.telemetry.pruned_fraction_sum = calloc(
                mc, sizeof(double));
            ts->state.telemetry.step_counts = calloc(
                mc, sizeof(long));
            ts->state.telemetry.dist_counts = calloc(
                mc + 1, sizeof(long));
            ts->state.telemetry.pruned_counts_by_dist = calloc(
                mc + 1, sizeof(long));
            ts->state.telemetry.cluster_query_counts = calloc(
                mc, sizeof(long));
        } // allocate ClusterState buffers

        /* Tile sub-frame: data allocated later by scatter */
        ts->tile_frame.data   = NULL;
        ts->tile_frame.width  =
            (long) tm->tiles[m].num_pixels;
        ts->tile_frame.height = 1;

        ts->pass1_assignment = -1;

        ts->pass1_posterior = calloc(
            (size_t) maxnbc, sizeof(double));
        ts->temp_indices = malloc(
            (size_t) maxnbc * sizeof(int));
        ts->temp_dists = malloc(
            (size_t) maxnbc * sizeof(double));
        ts->sorting_candidates = malloc(
            (size_t) maxnbc * sizeof(Candidate));

        if (ts->pass1_posterior == NULL
            || ts->temp_indices == NULL
            || ts->temp_dists == NULL
            || ts->sorting_candidates == NULL)
        {
            multitile_free(mts);
            return NULL;
        }

        /* Verbose candidates allocated on demand */
        ts->verbose_candidates = NULL;
        ts->ascii_out          = NULL;
        ts->prev_assigned_cluster = -1;
    } // for each tile m

    /* Tuple history: maxnbfr × num_tiles flat array */
    mts->tuple_history = calloc(
        (size_t)(maxnbfr * mts->num_tiles), sizeof(int));
    if (mts->tuple_history == NULL)
    {
        multitile_free(mts);
        return NULL;
    }

    mts->tuple_count      = 0;
    mts->retrieval_window = global->input.retrieval_window;

    mts->occurrence_head = malloc((size_t)(mts->num_tiles * maxnbc) * sizeof(int));
    mts->occurrence_prev = malloc((size_t)(maxnbfr * mts->num_tiles) * sizeof(int));
    if (mts->occurrence_head == NULL || mts->occurrence_prev == NULL)
    {
        multitile_free(mts);
        return NULL;
    }

    for (int i = 0; i < mts->num_tiles * maxnbc; i++)
    {
        mts->occurrence_head[i] = -1;
    }
    for (long i = 0; i < maxnbfr * mts->num_tiles; i++)
    {
        mts->occurrence_prev[i] = -1;
    }

    return mts;
}

/**
 * multitile_free - Release all multi-tile resources.
 * @mts: Multi-tile state to free (may be NULL).
 *
 * Frees per-tile scratch buffers, the tile_states array,
 * the tuple_history buffer, and the MultiTileState itself.
 * Does NOT free the TileMap (owned by the caller).
 */
void multitile_free(MultiTileState *mts)
{
    if (mts == NULL)
    {
        return;
    }

    if (mts->tile_states != NULL)
    {
        for (int m = 0; m < mts->num_tiles; m++)
        {
            TileState *ts = &mts->tile_states[m];

            free(ts->pass1_posterior);
            free(ts->temp_indices);
            free(ts->temp_dists);
            free(ts->sorting_candidates);
            free(ts->verbose_candidates);
            if (ts->state.scratch.tuple_pred_candidates)
            {
                free(ts->state.scratch.tuple_pred_candidates);
            }
        } // for each tile m

        free(mts->tile_states);
    } // if tile_states != NULL

    free(mts->tuple_history);
    free(mts->occurrence_head);
    free(mts->occurrence_prev);
    free(mts);
}

/**
 * multitile_load_tile_config - Parse per-tile config file.
 * @mts:  Multi-tile state with allocated tile_states.
 * @path: Path to ASCII config file.
 *
 * File format: one line per tile override.
 *   tile_id  rlim  maxnbclust
 * Lines starting with '#' are skipped as comments.
 * Overrides the rlim and maxnbclust fields in the
 * per-tile config for the specified tile_id.
 *
 * Return: 0 on success, -1 on error (file open failure
 *         or invalid tile_id).
 */
int multitile_load_tile_config(
    MultiTileState *mts,
    const char     *path)
{
    if (mts == NULL || path == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,
            "ERROR: cannot open tile config '%s'\n",
            path);
        return -1;
    }

    char line[256];
    int  line_num = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        line_num++;

        /* Skip comments and blank lines */
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        int    tid       = 0;
        double rlim_val  = 0.0;
        int    maxnbc    = 0;

        int nread = sscanf(
            line, "%d %lf %d", &tid, &rlim_val, &maxnbc);
        if (nread != 3)
        {
            fprintf(stderr,
                "WARNING: tile config line %d: "
                "expected 3 fields, got %d\n",
                line_num, nread);
            continue;
        }

        if (tid < 0 || tid >= mts->num_tiles)
        {
            fprintf(stderr,
                "WARNING: tile config line %d: "
                "tile_id %d out of range [0, %d)\n",
                line_num, tid, mts->num_tiles);
            continue;
        }

        mts->tile_states[tid].config.algo.rlim =
            rlim_val;
        mts->tile_states[tid].config.algo.maxnbclust =
            maxnbc;
    } // while reading lines

    fclose(fp);
    return 0;
}

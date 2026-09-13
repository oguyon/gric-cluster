/**
 * @file wasm_multitile.c
 * @brief Multi-tile axis-decomposition clustering WebAssembly bridge API.
 */

#include "wasm_internal.h"
#include "tuple_retrieval.h"

/**
 * tilemap_create_axis_decomposition() - Create a TileMap
 * that decomposes N-dim input into N independent 1D tiles.
 * @ndim: Number of dimensions (2 or 3).
 *
 * Each tile maps to a single coordinate index:
 *   tile 0 → pixel_indices = {0} (X axis)
 *   tile 1 → pixel_indices = {1} (Y axis)
 *   tile 2 → pixel_indices = {2} (Z axis, if 3D)
 *
 * Return: Pointer to allocated TileMap, or NULL on error.
 */
static TileMap *tilemap_create_axis_decomposition(
    int ndim)
{
    TileMap *tm = calloc(1, sizeof(TileMap));
    if (!tm)
    {
        return NULL;
    }

    tm->num_tiles = ndim;
    tm->img_width = ndim;
    tm->img_height = 1;
    tm->tiles = calloc(ndim, sizeof(TileDef));
    if (!tm->tiles)
    {
        free(tm);
        return NULL;
    }

    for (int m = 0; m < ndim; m++)
    {
        tm->tiles[m].num_pixels = 1;
        tm->tiles[m].pixel_indices =
            malloc(sizeof(int));
        if (!tm->tiles[m].pixel_indices)
        {
            tilemap_free(tm);
            return NULL;
        }
        tm->tiles[m].pixel_indices[0] = m;
    }

    return tm;
}

/**
 * wasm_multitile_init() - Allocate and initialize a
 * multi-tile clustering handle for WASM.
 *
 * Creates an axis-decomposition TileMap (ndim tiles),
 * allocates MultiTileState via multitile_init(), and
 * sets up scatter buffers.
 *
 * Return: Opaque handle pointer, or NULL on error.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_multitile_init(
    double rlim,
    int    maxnbclust,
    long   maxnbfr,
    int    ndim,
    int    entropy_mode,
    int    te4_mode,
    int    te5_mode,
    int    pred_mode,
    int    pred_h,
    int    gprob_mode,
    double tm_mixing_coeff,
    int    soft_bayesian_mode,
    int    xtile_mode,
    int    sparse_dcc_mode,
    int    sparse_dcc_extra_evals,
    double entropy_gate_bits,
    double entropy_first_gate_bits,
    int    entropy_fast_mode,
    double soft_bayesian_sigma_coeff,
    int    maxcl_strategy,
    double discard_fraction,
    int    max_gprob_visitors)
{
    if (ndim != 2 && ndim != 3)
    {
        return NULL;
    }

    WasmMultiTileHandle *h = calloc(
        1, sizeof(WasmMultiTileHandle)
    );

    h->ndim = ndim;
    h->maxnbfr = maxnbfr;
    h->current_frame_id = 0;

    /* Build global config */
    h->config.algo.rlim = rlim;
    h->config.algo.maxnbclust = maxnbclust;
    h->config.algo.deltaprob = 0.0;
    h->config.algo.tm_mixing_coeff = tm_mixing_coeff;
    h->config.algo.maxcl_strategy =
        (MaxClustStrategy)maxcl_strategy;
    h->config.algo.discard_fraction = discard_fraction;
    h->config.algo.use_double = 1;

    h->config.input.maxnbfr = maxnbfr;

    h->config.optim.entropy_mode = entropy_mode;
    h->config.optim.te4_mode = te4_mode;
    h->config.optim.te5_mode = te5_mode;
    h->config.optim.pred_mode = pred_mode;
    h->config.optim.pred_len = 4;
    h->config.optim.pred_h = pred_h;
    h->config.optim.pred_n = 3;
    h->config.optim.gprob_mode = gprob_mode;
    h->config.optim.fmatch_a = 2.0;
    h->config.optim.fmatch_b = 0.5;
    h->config.optim.max_gprob_visitors =
        max_gprob_visitors;
    h->config.optim.entropy_max_targets = 24;
    h->config.optim.entropy_min_prob = 0.001;
    h->config.optim.entropy_gate_bits =
        entropy_gate_bits;
    h->config.optim.entropy_first_gate_bits =
        entropy_first_gate_bits;
    h->config.optim.entropy_fast_mode =
        entropy_fast_mode;
    h->config.optim.sparse_dcc_mode = sparse_dcc_mode;
    h->config.optim.sparse_dcc_extra_evals =
        sparse_dcc_extra_evals;
    h->config.optim.soft_bayesian_mode =
        soft_bayesian_mode;
    h->config.optim.soft_bayesian_sigma_coeff =
        soft_bayesian_sigma_coeff;
    h->config.optim.xtile_mode = xtile_mode;
    h->config.optim.xtile_decay = 0.70;

    h->config.output.verbose_level = 0;
    h->config.output.progress_mode = 0;

    /* Create axis-decomposition tile map */
    h->tile_map =
        tilemap_create_axis_decomposition(ndim);
    if (!h->tile_map)
    {
        free(h);
        return NULL;
    }

    /* Allocate MultiTileState */
    h->mts = multitile_init(
        &h->config, h->tile_map, maxnbfr);
    if (!h->mts)
    {
        tilemap_free(h->tile_map);
        free(h);
        return NULL;
    }

    /* Allocate scatter buffers */
    h->scatter_buf = calloc(ndim, sizeof(Frame));
    if (!h->scatter_buf)
    {
        multitile_free(h->mts);
        tilemap_free(h->tile_map);
        free(h);
        return NULL;
    }
    frame_scatter_alloc(h->tile_map, h->scatter_buf, 1);

    /* Reusable source frame */
    h->src_frame.data = calloc(ndim, sizeof(double));
    if (!h->src_frame.data)
    {
        frame_scatter_free(h->scatter_buf, h->ndim);
        free(h->scatter_buf);
        multitile_free(h->mts);
        tilemap_free(h->tile_map);
        free(h);
        return NULL;
    }
    h->src_frame.is_double = 1;
    h->src_frame.width = ndim;
    h->src_frame.height = 1;
    h->src_frame.id = 0;
    h->src_frame.cnt0 = 0;

    return h;
}

/**
 * wasm_multitile_process_frame() - Process one N-dim
 * coordinate frame through the multi-tile pipeline.
 * @ptr:    Opaque WasmMultiTileHandle pointer.
 * @coords: Array of ndim doubles (x, y [, z]).
 * @ndim:   Dimensionality (must match init).
 *
 * Pipeline:
 *   1. Scatter coordinates into per-tile 1D sub-frames
 *   2. predict_joint_tuples() if pred_mode
 *   3. Pass 1: cluster_frame() for each tile
 *   4. Pass 2: pass2_fuse() for each tile
 *   5. Record tuple in history
 *   6. cpt_update_incremental() if xtile_mode
 *
 * Return: Tile 0 assignment, or -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_process_frame(
    void   *ptr,
    double *coords,
    int     ndim)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !coords || ndim != h->ndim)
    {
        return -1;
    }

    int M = h->mts->num_tiles;

    /* Populate source frame */
    memcpy(h->src_frame.data, coords,
           (size_t)ndim * sizeof(double));
    h->src_frame.id = h->current_frame_id;
    h->src_frame.width = ndim;
    h->src_frame.height = 1;

    /* 1. Scatter into per-tile sub-frames */
    frame_scatter(
        &h->src_frame, h->tile_map, h->scatter_buf);

    /* 2. Initialize cross-tile board */
    for (int m = 0; m < M; m++)
    {
        h->mts->xtile_board[m] = -1;
    }

    /* 3. Predict joint tuples for Pass 1 priors */
    if (h->config.optim.pred_mode)
    {
        predict_joint_tuples(
            h->mts,
            h->config.optim.pred_len,
            h->config.optim.pred_h,
            h->config.optim.pred_n);
    }

    /* 4. Pass 1: cluster_frame() for each tile */
    for (int m = 0; m < M; m++)
    {
        TileState *ts = &h->mts->tile_states[m];

        /* Allocate fresh data buffer for task frame */
        Frame task_frame;
        task_frame.id = h->scatter_buf[m].id;
        task_frame.cnt0 = h->scatter_buf[m].cnt0;
        task_frame.width = h->scatter_buf[m].width;
        task_frame.height = h->scatter_buf[m].height;
        task_frame.is_double = 1;
        task_frame.data = malloc(
            (size_t)(task_frame.width
                     * task_frame.height)
            * sizeof(double));
        if (!task_frame.data)
        {
            return -1;
        }
        memcpy(
            task_frame.data,
            h->scatter_buf[m].data,
            (size_t)(task_frame.width
                     * task_frame.height)
            * sizeof(double));

        ts->pass1_old_ncl = ts->state.num_clusters;

        /* Reset per-frame scratch */
        for (int i = 0;
             i < ts->config.algo.maxnbclust; i++)
        {
            ts->temp_indices[i] = -1;
        }
        for (int i = 0; i < M; i++)
        {
            ts->last_injected_assignment[i] = -1;
        }

        /* Wire cross-tile hooks */
        if (h->config.optim.xtile_mode == 1)
        {
            ts->state.cross_tile_hook =
                inject_cross_tile_priors;
            ts->state.cross_tile_ctx = ts;
        }
        else if (h->config.optim.xtile_mode == 2)
        {
            ts->state.cross_tile_hook =
                inject_cross_tile_priors_st;
            ts->state.cross_tile_ctx = ts;
        }
        else
        {
            ts->state.cross_tile_hook = NULL;
            ts->state.cross_tile_ctx = NULL;
        }

        int res = cluster_frame(
            &ts->config,
            &ts->state,
            &task_frame,
            &ts->prev_assigned_cluster,
            ts->ascii_out,
            ts->temp_indices,
            ts->temp_dists,
            ts->sorting_candidates,
            ts->verbose_candidates);

        ts->pass1_assignment = (res >= 0) ? res : -1;
        __atomic_store_n(
            &ts->xtile_board[m], res,
            __ATOMIC_RELEASE);
    } // for each tile m (Pass 1)

    /* 5. Pass 2: Bayesian fusion */
    if (h->mts->tuple_count > 0 && M > 1)
    {
        for (int m = 0; m < M; m++)
        {
            pass2_fuse(
                h->mts, m, &h->scatter_buf[m]);
        }
    }

    /* 6. Record assignment tuple (if history has room) */
    if (h->mts->tuple_count < h->maxnbfr)
    {
        long base =
            h->mts->tuple_count * (long)M;
        long t = h->mts->tuple_count;
        int maxcl =
            h->mts->tile_states[0]
                .config.algo.maxnbclust;

        for (int m = 0; m < M; m++)
        {
            int ass =
                h->mts->tile_states[m]
                    .pass1_assignment;
            h->mts->tuple_history[base + m] = ass;
            if (ass >= 0 && ass < maxcl)
            {
                h->mts->occurrence_prev[
                    t * (long)M + m] =
                    h->mts->occurrence_head[
                        m * maxcl + ass];
                h->mts->occurrence_head[
                    m * maxcl + ass] = (int)t;
            }
        }

        /* 7. CPT update if xtile */
        if (h->config.optim.xtile_mode)
        {
            cpt_update_incremental(
                h->mts->cpt,
                &h->mts->cpt_scale,
                h->mts->tuple_history,
                h->mts->tuple_count,
                M,
                maxcl,
                h->config.optim.xtile_decay);
        }

        h->mts->tuple_count++;
    } // Record assignment tuple

    h->current_frame_id++;

    return h->mts->tile_states[0].pass1_assignment;
}

/**
 * wasm_multitile_get_num_tiles() - Return the number
 * of tiles (== ndim for axis decomposition).
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_num_tiles(void *ptr)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts)
    {
        return 0;
    }
    return h->mts->num_tiles;
}

/**
 * wasm_multitile_get_num_tile_clusters() - Return the
 * current cluster count for a specific tile.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_num_tile_clusters(
    void *ptr,
    int   tile_id)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts ||
        tile_id < 0 || tile_id >= h->mts->num_tiles)
    {
        return 0;
    }
    return h->mts->tile_states[tile_id]
        .state.num_clusters;
}

/**
 * wasm_multitile_get_tile_clusters() - Copy per-tile
 * 1D cluster anchor coordinates and member counts.
 * @ptr:         Opaque handle.
 * @tile_id:     Tile index (0..M-1).
 * @out_coords:  Output double array (1 coord per cluster).
 * @out_members: Output int array (member counts).
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_get_tile_clusters(
    void   *ptr,
    int     tile_id,
    double *out_coords,
    int    *out_members)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts || !out_coords || !out_members ||
        tile_id < 0 || tile_id >= h->mts->num_tiles)
    {
        return;
    }

    TileState *ts = &h->mts->tile_states[tile_id];
    int K = ts->state.num_clusters;

    for (int i = 0; i < K; i++)
    {
        const double *data =
            ts->state.clusters[i].anchor.data;
        out_coords[i] = data ? data[0] : 0.0;
        out_members[i] =
            ts->state.cluster_visitors[i].count;
    }
}

/**
 * wasm_multitile_get_tuples() - Aggregate tuple_history
 * into unique joint tuples with occurrence counts.
 * @ptr:             Opaque handle.
 * @out_flat:        Output flat int array [max_tuples × M]
 *                   of per-tile cluster IDs.
 * @out_counts:      Output int array of occurrence counts.
 * @out_last_active: Output int array of last active frame.
 * @max_tuples:      Capacity of output arrays.
 *
 * Return: Number of unique tuples written.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_tuples(
    void *ptr,
    int  *out_flat,
    int  *out_counts,
    int  *out_last_active,
    int   max_tuples)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts || !out_flat || !out_counts)
    {
        return 0;
    }

    int M = h->mts->num_tiles;
    long total = h->mts->tuple_count;
    if (total == 0)
    {
        return 0;
    }

    /*
     * Simple O(T*U) unique-tuple aggregation.
     * For simulator sizes (T < 100k) this is fast.
     */
    int unique_count = 0;

    for (long t = 0; t < total; t++)
    {
        const int *tuple =
            &h->mts->tuple_history[t * M];

        /* Search for existing unique tuple */
        int found = -1;
        for (int u = 0; u < unique_count; u++)
        {
            int match = 1;
            for (int m = 0; m < M; m++)
            {
                if (out_flat[u * M + m] != tuple[m])
                {
                    match = 0;
                    break;
                }
            }
            if (match)
            {
                found = u;
                break;
            }
        } // search existing

        if (found >= 0)
        {
            out_counts[found]++;
            if (out_last_active)
            {
                out_last_active[found] = (int)t;
            }
        }
        else if (unique_count < max_tuples)
        {
            memcpy(&out_flat[unique_count * M],
                   tuple,
                   (size_t)M * sizeof(int));
            out_counts[unique_count] = 1;
            if (out_last_active)
            {
                out_last_active[unique_count] = (int)t;
            }
            unique_count++;
        }
    } // for each frame

    return unique_count;
}

/**
 * wasm_multitile_get_tile_telemetry() - Retrieve
 * per-tile telemetry counters.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_get_tile_telemetry(
    void   *ptr,
    int     tile_id,
    double *out_stats,
    int    *out_len)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts || !out_stats || !out_len ||
        tile_id < 0 || tile_id >= h->mts->num_tiles)
    {
        if (out_len)
        {
            *out_len = 0;
        }
        return;
    }

    const ClusterTelemetry *t =
        &h->mts->tile_states[tile_id].state.telemetry;

    out_stats[TELEM_FRAMEDIST_CALLS] =
        (double)t->framedist_calls;
    out_stats[TELEM_FRAMEDIST_SAMPLE] =
        (double)t->framedist_calls_sample;
    out_stats[TELEM_FRAMEDIST_INTERCLUSTER] =
        (double)t->framedist_calls_intercluster;
    out_stats[TELEM_CLUSTERS_PRUNED] =
        (double)t->clusters_pruned;
    out_stats[TELEM_TOTAL_FRAMES] =
        (double)t->total_frames_processed;
    out_stats[TELEM_LAST_FRAME_DISTS] =
        (double)t->last_frame_dists;
    out_stats[TELEM_LAST_FRAME_DFC] =
        (double)t->last_frame_dfc;
    out_stats[TELEM_LAST_FRAME_DCC] =
        (double)t->last_frame_dcc;
    out_stats[TELEM_LAST_ASSIGNMENT_DIST] =
        t->last_assignment_dist;
    out_stats[TELEM_NUM_NEW_CLUSTERS] =
        (double)t->num_new_clusters;
    out_stats[TELEM_PRED_ATTEMPTS] =
        (double)t->pred_attempts;
    out_stats[TELEM_PRED_HITS] =
        (double)t->pred_hits;
    out_stats[TELEM_ENTROPY_GATED] =
        (double)t->entropy_frames_gated;
    out_stats[TELEM_ENTROPY_EVALUATED] =
        (double)t->entropy_frames_evaluated;
    out_stats[TELEM_ENTROPY_SUM_INITIAL] =
        t->entropy_sum_initial;
    out_stats[TELEM_ENTROPY_MAX_INITIAL] =
        t->entropy_max_initial;
    out_stats[TELEM_ENTROPY_LAST_INITIAL] =
        t->entropy_last_initial;
    out_stats[TELEM_DCC_ENTRIES_POPULATED] =
        (double)t->dcc_entries_populated;
    out_stats[TELEM_DCC_PAIRS_TOTAL] =
        (double)t->dcc_pairs_total;

    *out_len = TELEM_COUNT;
}

/**
 * wasm_multitile_reset() - Reset all tile states and
 * tuple history while retaining allocations.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_reset(void *ptr)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h || !h->mts)
    {
        return;
    }

    int M = h->mts->num_tiles;
    int maxcl =
        h->mts->tile_states[0]
            .config.algo.maxnbclust;

    h->current_frame_id = 0;

    /* Reset each tile's clustering state */
    for (int m = 0; m < M; m++)
    {
        TileState *ts = &h->mts->tile_states[m];
        int N = ts->config.algo.maxnbclust;

        ts->prev_assigned_cluster = -1;
        ts->pass1_assignment = -1;
        ts->pass1_old_ncl = 0;

        /* Free per-cluster anchor data */
        for (int i = 0; i < N; i++)
        {
            if (ts->state.clusters[i].anchor.data)
            {
                free(
                    ts->state.clusters[i].anchor.data);
                ts->state.clusters[i].anchor.data =
                    NULL;
            }
        }
        memset(ts->state.clusters, 0,
               (size_t)N * sizeof(Cluster));

        /* Free visitor list arrays */
        for (int i = 0; i < N; i++)
        {
            if (ts->state.cluster_visitors[i].frames)
            {
                free(
                    ts->state
                        .cluster_visitors[i].frames);
            }
        }
        memset(ts->state.cluster_visitors, 0,
               (size_t)N * sizeof(VisitorList));

        ts->state.num_clusters = 0;

        /* Reset assignments */
        for (long i = 0; i < h->maxnbfr; i++)
        {
            ts->state.assignments[i] = -1;
        }
        memset(ts->state.frame_infos, 0,
               (size_t)h->maxnbfr * sizeof(FrameInfo));
        memset(ts->state.transition_matrix, 0,
               (size_t)N * N * sizeof(long));

        /* Reset scratch matrices */
        {
            ClusterScratch *s = &ts->state.scratch;
            memset(s->dcc_min, 0,
                   (size_t)N * N * sizeof(double));
            memset(s->dcc_max, 0,
                   (size_t)N * N * sizeof(double));
            memset(s->dcc_measured, 0,
                   (size_t)N * N * sizeof(char));
            size_t mw =
                (size_t)N * N * ((N + 63) / 64);
            memset(s->consistency_mask, 0,
                   mw * sizeof(uint64_t));
            s->refine_queue_size = 0;
            s->refine_queue_idx = 0;
            s->refine_queue_last_num_clusters = 0;
            s->tuple_pred_count = 0;
        } // Reset scratch

        /* Reset telemetry (preserve pointers) */
        {
            ClusterTelemetry *t =
                &ts->state.telemetry;
            double *pfsum = t->pruned_fraction_sum;
            long *scnt = t->step_counts;
            long *dcnt = t->dist_counts;
            long *pcnt = t->pruned_counts_by_dist;
            long *qcnt = t->cluster_query_counts;
            int max_steps = t->max_steps_recorded;

            memset(t, 0, sizeof(ClusterTelemetry));

            t->pruned_fraction_sum = pfsum;
            t->step_counts = scnt;
            t->dist_counts = dcnt;
            t->pruned_counts_by_dist = pcnt;
            t->cluster_query_counts = qcnt;
            t->max_steps_recorded = max_steps;

            if (pfsum)
            {
                memset(pfsum, 0,
                       (size_t)N * sizeof(double));
            }
            if (scnt)
            {
                memset(scnt, 0,
                       (size_t)N * sizeof(long));
            }
            if (dcnt)
            {
                memset(dcnt, 0,
                       (size_t)(N + 1) * sizeof(long));
            }
            if (pcnt)
            {
                memset(pcnt, 0,
                       (size_t)(N + 1) * sizeof(long));
            }
            if (qcnt)
            {
                memset(qcnt, 0,
                       (size_t)N * sizeof(long));
            }
        } // Reset telemetry
    } // for each tile

    /* Reset tuple history */
    h->mts->tuple_count = 0;
    memset(h->mts->occurrence_head, -1,
           (size_t)M * maxcl * sizeof(int));
    memset(h->mts->occurrence_prev, -1,
           (size_t)h->maxnbfr * M * sizeof(int));

    /* Reset CPT */
    {
        size_t cpt_size =
            (size_t)M * maxcl * M * maxcl;
        memset(h->mts->cpt, 0,
               cpt_size * sizeof(double));
        h->mts->cpt_scale = 1.0;
    }

    /* Reset xtile board */
    for (int m = 0; m < M; m++)
    {
        h->mts->xtile_board[m] = -1;
    }
}

/**
 * wasm_multitile_free() - Free all multi-tile
 * resources.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_free(void *ptr)
{
    WasmMultiTileHandle *h =
        (WasmMultiTileHandle *)ptr;
    if (!h)
    {
        return;
    }

    if (h->scatter_buf)
    {
        frame_scatter_free(
            h->scatter_buf, h->ndim);
        free(h->scatter_buf);
    }

    if (h->mts)
    {
        multitile_free(h->mts);
    }

    if (h->tile_map)
    {
        tilemap_free(h->tile_map);
    }

    if (h->src_frame.data)
    {
        free(h->src_frame.data);
    }

    free(h);
}

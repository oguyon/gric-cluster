/**
 * @file wasm_internal.h
 * @brief Internal types and shared declarations for WebAssembly bindings.
 */

#ifndef WASM_INTERNAL_H
#define WASM_INTERNAL_H

#define _POSIX_C_SOURCE 200809L

#include "gric_wasm_api.h"
#include "cluster_defs.h"
#include "cluster_step.h"
#include "cluster_steps.h"
#include "framedistance.h"
#include "cluster_math.h"
#include "cluster_bounds.h"
#include "cluster_trace.h"
#include "knn_defs.h"
#include "knn_engine.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "tile_map.h"
#include "tile_state.h"
#include "frame_scatter.h"

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GRIC_GIT_HASH
#define GRIC_GIT_HASH "unknown"
#endif

#ifndef GRIC_BUILD_DATE
#define GRIC_BUILD_DATE "unknown"
#endif

/* Telemetry layout indices */
enum
{
    TELEM_FRAMEDIST_CALLS = 0,
    TELEM_FRAMEDIST_SAMPLE,
    TELEM_FRAMEDIST_INTERCLUSTER,
    TELEM_CLUSTERS_PRUNED,
    TELEM_TOTAL_FRAMES,
    TELEM_LAST_FRAME_DISTS,
    TELEM_LAST_FRAME_DFC,
    TELEM_LAST_FRAME_DCC,
    TELEM_LAST_ASSIGNMENT_DIST,
    TELEM_NUM_NEW_CLUSTERS,
    TELEM_PRED_ATTEMPTS,
    TELEM_PRED_HITS,
    TELEM_ENTROPY_GATED,
    TELEM_ENTROPY_EVALUATED,
    TELEM_ENTROPY_SUM_INITIAL,
    TELEM_ENTROPY_MAX_INITIAL,
    TELEM_ENTROPY_LAST_INITIAL,
    TELEM_DCC_ENTRIES_POPULATED,
    TELEM_DCC_PAIRS_TOTAL,
    TELEM_COUNT
};

/**
 * struct WasmHandle - Bundled single-tile state for the WASM API.
 */
typedef struct
{
    ClusterConfig  config;
    ClusterState   state;
    Frame          frame;
    int            ndim;
    long           maxnbfr;
    int            current_frame_id;
    int            prev_assigned;
    int            user_maxcl;
    int           *temp_indices;
    double        *temp_dists;
    Candidate     *sorting_candidates;
} WasmHandle;

/**
 * struct WasmMultiTileHandle - Bundled multi-tile state for the WASM API.
 */
typedef struct
{
    ClusterConfig   config;
    MultiTileState *mts;
    TileMap        *tile_map;
    Frame          *scatter_buf;
    int             ndim;
    long            maxnbfr;
    int             current_frame_id;
    Frame           src_frame;
} WasmMultiTileHandle;

extern volatile sig_atomic_t stop_requested;

/**
 * get_dist() - Evaluate Euclidean distance and update telemetry counters.
 */
double get_dist(
    Frame         *a,
    Frame         *b,
    int            cluster_idx,
    double         cluster_prob,
    double         current_gprob,
    ClusterConfig *config,
    ClusterState  *state);

/**
 * free_frame() - Release pixel data buffer associated with a frame.
 */
void free_frame(
    Frame *frame_ptr);

#endif // WASM_INTERNAL_H

#ifndef TILE_STATE_H
#define TILE_STATE_H

/**
 * @file tile_state.h
 * @brief Per-tile clustering state and multi-tile
 *        orchestration structures.
 */

#include "cluster_defs.h"
#include "tile_map.h"

/** Per-tile clustering state and configuration. */
typedef struct
{
    int            tile_id;
    int            num_tiles;          /**< Total number of tiles in the partition */
    TileDef       *tile_def;           /**< Ptr into TileMap */
    ClusterConfig  config;             /**< Per-tile copy */
    ClusterState   state;              /**< Independent state */
    Frame          tile_frame;         /**< Reusable sub-frame */
    int            pass1_assignment;   /**< Best cluster Pass 1 */
    double        *pass1_posterior;    /**< Full posterior dist */
    int           *temp_indices;       /**< Scratch: meas indices */
    double        *temp_dists;         /**< Scratch: meas dists */
    Candidate     *sorting_candidates;
    Candidate     *verbose_candidates;
    FILE          *ascii_out;
    int            prev_assigned_cluster;
    int            pass1_old_ncl;      /**< Number of clusters before Pass 1 */
    volatile int  *xtile_board;        /**< Shared cross-tile assignment board */
    double        *cpt;                /**< Shared Conditional Probability Table */
    double        *cpt_scale;          /**< Shared CPT scale factor pointer */
    int           *last_injected_assignment; /**< Resolved neighbor track list */
    void          *mts;                /**< Ptr to parent MultiTileState (forward declared void* to avoid circular header dependencies) */
} TileState;

/** Top-level multi-tile state. */
typedef struct
{
    TileMap       *tile_map;
    TileState     *tile_states;      /**< Array of M TileStates */
    int            num_tiles;
    int           *tuple_history;    /**< [maxnbfr × M] flat */
    long           tuple_count;      /**< Frames recorded */
    int            retrieval_window; /**< Lookback horizon */
    int           *occurrence_head;  /**< [M × max_clusters] flat */
    int           *occurrence_prev;  /**< [maxnbfr × M] flat */
    volatile int  *xtile_board;      /**< Shared board flat [M] */
    double        *cpt;              /**< Shared CPT flat [M * Kmax * M * Kmax] */
    double         cpt_scale;        /**< Shared CPT scale factor */
} MultiTileState;

/** Allocate and initialise multi-tile state. */
MultiTileState *multitile_init(
    ClusterConfig *global,
    TileMap       *tm,
    long           maxnbfr);

/** Free all multi-tile resources (not the TileMap). */
void multitile_free(MultiTileState *mts);

/** Parse per-tile config overrides from ASCII file. */
int multitile_load_tile_config(
    MultiTileState *mts,
    const char     *path);

#endif // TILE_STATE_H

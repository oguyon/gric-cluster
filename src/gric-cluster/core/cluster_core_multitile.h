#ifndef CLUSTER_CORE_MULTITILE_H
#define CLUSTER_CORE_MULTITILE_H

/**
 * @file cluster_core_multitile.h
 * @brief Multi-tile clustering entry point declaration.
 *
 * Orchestrates parallel per-tile clustering using OpenMP
 * tasks, with tuple recording for later Bayesian fusion.
 */

#include "cluster_defs.h"
#include "tile_state.h"

/**
 * @brief Run multi-tile clustering with OpenMP task parallelism.
 *
 * Reads full-image frames, scatters them into per-tile
 * sub-frames, dispatches parallel Pass 1 clustering tasks,
 * and records the resulting assignment tuples.
 */
void run_clustering_multitile(
    ClusterConfig  *global_config,
    MultiTileState *mts);

#endif // CLUSTER_CORE_MULTITILE_H

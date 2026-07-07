/**
 * @file cluster_io_multitile.h
 * @brief Results serialization for multi-tile clustering.
 */

#ifndef CLUSTER_IO_MULTITILE_H
#define CLUSTER_IO_MULTITILE_H

#include "cluster_defs.h"
#include "tile_state.h"

/** Write per-tile output files. */
void write_results_multitile(
    ClusterConfig  *config,
    MultiTileState *mts);

#endif // CLUSTER_IO_MULTITILE_H

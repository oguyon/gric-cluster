#ifndef CLUSTER_REASSIGN_H
#define CLUSTER_REASSIGN_H

/**
 * @file cluster_reassign.h
 * @brief Second pass closest-anchor clustering and membership reallocation.
 */

#include "cluster_defs.h"

/**
 * @brief Run Pass 2 clustering to reallocate all frames to their nearest cluster anchor.
 */
long run_second_pass_clustering(
    ClusterConfig *config,
    ClusterState  *state);

#endif // CLUSTER_REASSIGN_H

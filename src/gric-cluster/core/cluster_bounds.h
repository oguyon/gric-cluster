/**
 * @file cluster_bounds.h
 * @brief Declarations for inter-cluster distance bound propagation.
 */

#ifndef CLUSTER_BOUNDS_H
#define CLUSTER_BOUNDS_H

#include "cluster_defs.h"

/**
 * update_dcc_bounds - Update bounds matrix with an exact measurement and propagate.
 */
void update_dcc_bounds(
    ClusterState  *state,
    ClusterConfig *config,
    int            i,
    int            j,
    double         d_exact);

/**
 * refine_sparse_bounds - Refine distance bounds by measuring closest unmeasured pairs.
 */
void refine_sparse_bounds(
    ClusterConfig *config,
    ClusterState  *state);

#endif // CLUSTER_BOUNDS_H

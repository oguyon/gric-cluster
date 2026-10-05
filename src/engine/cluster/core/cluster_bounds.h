/**
 * @file cluster_bounds.h
 * @brief Declarations for inter-cluster distance bound propagation.
 */

#ifndef CLUSTER_BOUNDS_H
#define CLUSTER_BOUNDS_H

#include "cluster_defs.h"
#include "cluster_dcc.h"

/**
 * set_dcc_pair() - Update symmetric pairwise distance cache entries in both float and SQ16.
 * @state: Clustering state.
 * @maxnb: Maximum cluster capacity dimension (deprecated, uses state->scratch.dcc_stride).
 * @c1:    First cluster index.
 * @c2:    Second cluster index.
 * @d:     Pairwise Euclidean distance.
 */
static inline void set_dcc_pair(
    ClusterState *state,
    uint64_t      maxnb,
    int           c1,
    int           c2,
    double        d)
{
    (void)maxnb;
    dcc_set_pair(state, c1, c2, d);
}

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

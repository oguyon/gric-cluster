/**
 * @file cluster_bounds.h
 * @brief Declarations for inter-cluster distance bound propagation.
 */

#ifndef CLUSTER_BOUNDS_H
#define CLUSTER_BOUNDS_H

#include "cluster_defs.h"

/**
 * set_dcc_pair() - Update symmetric pairwise distance cache entries in both float and SQ16.
 * @state: Clustering state.
 * @maxnb: Maximum cluster capacity dimension.
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
    state->scratch.dcc_min[c1 * maxnb + c2] = d;
    state->scratch.dcc_min[c2 * maxnb + c1] = d;
    state->scratch.dcc_max[c1 * maxnb + c2] = d;
    state->scratch.dcc_max[c2 * maxnb + c1] = d;
    state->scratch.dcc_measured[c1 * maxnb + c2] = 1;
    state->scratch.dcc_measured[c2 * maxnb + c1] = 1;
    if (state->scratch.dcc_sq16 != NULL)
    {
        double s = state->scratch.dcc_sq16_scale;
        uint16_t q = (d * s >= 65534.0) ? 65534 : (uint16_t)(d * s + 0.5);
        state->scratch.dcc_sq16[c1 * maxnb + c2] = q;
        state->scratch.dcc_sq16[c2 * maxnb + c1] = q;
    }
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

/**
 * @file gric_api_internal.h
 * @brief Internal definitions and helper declarations for libgric API session.
 */

#ifndef GRIC_API_INTERNAL_H
#define GRIC_API_INTERNAL_H

#include "gric/gric.h"
#include "cluster_defs.h"
#include <stddef.h>

/**
 * struct gric_cluster_ctx - Concrete clustering session state.
 */
struct gric_cluster_ctx
{
    ClusterConfig  config;
    ClusterState   state;
    Frame          frame;
    size_t         ndim;
    size_t         maxnbfr;
    int            current_frame_id;
    int            prev_assigned;
    int            user_maxcl;
    int           *temp_indices;
    double        *temp_dists;
    Candidate     *sorting_candidates;
    void          *conv_buf;
};

/**
 * grow_context_capacity() - Double cluster capacity for dynamic clustering sessions.
 * @ctx: Active clustering session.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
int grow_context_capacity(
    gric_cluster_t *ctx);

#endif // GRIC_API_INTERNAL_H

#ifndef CLUSTER_CORE_H
#define CLUSTER_CORE_H

#include "cluster_defs.h"
#include "cluster_math.h"
#include "cluster_mgmt.h"
#include "cluster_prune.h"

extern volatile sig_atomic_t stop_requested;

void run_clustering(ClusterConfig *config, ClusterState *state);
void run_scandist(ClusterConfig *config, char *out_dir);

int compare_candidates(const void *a, const void *b);
int compare_doubles(const void *a, const void *b);

/**
 * @brief High-level distance evaluation between a frame and a cluster anchor.
 *
 * Wraps the raw `framedist` call, records statistics, writes to the distance log if configured,
 * and prints verbose traces if requested.
 */
double get_dist(
    Frame         *a,
    Frame         *b,
    int            cluster_idx,
    double         cluster_prob,
    double         current_gprob,
    ClusterConfig *config,
    ClusterState  *state);

#endif // CLUSTER_CORE_H

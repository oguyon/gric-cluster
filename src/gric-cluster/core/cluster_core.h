#ifndef CLUSTER_CORE_H
#define CLUSTER_CORE_H

#include "cluster_defs.h"
#include "cluster_math.h"
#include "cluster_mgmt.h"
#include "cluster_prune.h"
#include "cluster_bounds.h"

extern volatile sig_atomic_t stop_requested;

/**
 * run_clustering() - Main online streaming clustering loop.
 * @config: Clustering parameters and runtime flags.
 * @state:  Clustering dynamic state, anchors, and counters.
 */
void run_clustering(
    ClusterConfig *config,
    ClusterState  *state);

/**
 * get_dist() - Distance evaluation between a frame and a cluster anchor.
 * @a:             Input frame to classify.
 * @b:             Candidate cluster anchor frame.
 * @cluster_idx:   Candidate cluster index.
 * @cluster_prob:  Prior probability of candidate cluster.
 * @current_gprob: Current greedy probability.
 * @config:        Clustering runtime configuration.
 * @state:         Clustering dynamic state.
 *
 * Return: Distance between frame and anchor.
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
 * print_clustering_metrics() - Display runtime telemetry and distance statistics.
 * @state:   Active cluster state containing telemetry counters.
 * @tile_id: Identifier of current tile (-1 if single-stream mode).
 */
void print_clustering_metrics(
    const ClusterState *state,
    int                 tile_id);

#endif // CLUSTER_CORE_H

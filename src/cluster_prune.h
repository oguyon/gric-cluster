#ifndef CLUSTER_PRUNE_H
#define CLUSTER_PRUNE_H

#include "cluster_defs.h"

/**
 * @brief Predicts potential cluster assignment candidates based on pattern history.
 *
 * Scans recent frame assignment history for matching patterns (markov chain assignment sequences)
 * and ranks candidates by historical transition frequency.
 *
 * @param state Pointer to the active ClusterState.
 * @param config Pointer to the active ClusterConfig.
 * @param candidates Preallocated array to store candidate cluster indices.
 * @param max_candidates Maximum size of candidates to return.
 * @return Number of valid candidates found.
 */
int get_prediction_candidates(ClusterState *state, ClusterConfig *config, int *candidates, int max_candidates);

/**
 * @brief Uses 5-point configuration math to prune non-matching cluster candidates.
 *
 * Eliminates redundant distance calculations by applying geometric triangular bounds.
 *
 * @param config Pointer to the active ClusterConfig.
 * @param state Pointer to the active ClusterState.
 * @param temp_indices Array of candidate indices checked so far.
 * @param temp_dists Array of computed distances corresponding to indices.
 * @param temp_count Number of entries in temp_indices and temp_dists.
 */
void prune_candidates_te5(ClusterConfig *config, ClusterState *state, int *temp_indices, double *temp_dists, int temp_count);

#endif // CLUSTER_PRUNE_H

#ifndef CLUSTER_MGMT_H
#define CLUSTER_MGMT_H

#include "cluster_defs.h"

/**
 * @brief Safely adds a frame index to a cluster's visitor history.
 *
 * Dynamically resizes the integer list capacity as needed (doubles capacity, starting at 16).
 * Prints an error using `perror` if memory reallocation fails.
 *
 * @param list Pointer to the VisitorList structure.
 * @param frame_idx Index of the frame to append.
 */
void add_visitor(VisitorList *list, int frame_idx);

/**
 * @brief Deletes a cluster from state, optionally merging its history.
 *
 * Reorganizes the active clusters list:
 * 1. Shifts cluster structs, updating cluster IDs.
 * 2. Shifts visitor history arrays.
 * 3. Compacts and shifts the inter-cluster distance matrix (dccarray).
 * 4. Compacts and shifts the inter-cluster transition probability matrix.
 * 5. Rewrites the assignments logs and maps the deleted cluster assignment
 *    either to a new target cluster index (if merging) or -1 (if discarding).
 *
 * @param state Pointer to the active ClusterState.
 * @param config Pointer to the active ClusterConfig.
 * @param index_to_remove The index of the cluster being deleted.
 * @param index_target The merge target index, or -1 to discard completely.
 */
void remove_cluster(ClusterState *state, ClusterConfig *config, int index_to_remove,
                    int index_target);

#endif // CLUSTER_MGMT_H

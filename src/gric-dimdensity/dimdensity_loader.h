/**
 * @file dimdensity_loader.h
 * @brief Loader interface for k-NN distance matrices from files or directories.
 */

#ifndef DIMDENSITY_LOADER_H
#define DIMDENSITY_LOADER_H

#include "dimdensity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load k-NN distance matrix from file or cluster directory.
 */
int dimdensity_load_distances(
    const char      *input_path,
    KnnDistanceData *data,
    int              verbose);

/**
 * Free memory associated with KnnDistanceData.
 */
void dimdensity_free_distances(
    KnnDistanceData *data);

#ifdef __cplusplus
}
#endif

#endif // DIMDENSITY_LOADER_H

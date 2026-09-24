/**
 * @file dimdensity_mack.h
 * @brief Mack-Rosenblatt k-NN density estimation with local dimension adaptation.
 */

#ifndef DIMDENSITY_MACK_H
#define DIMDENSITY_MACK_H

#include "dimdensity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute Mack-Rosenblatt density estimates for all samples.
 */
int dimdensity_compute_mack_density(
    const KnnDistanceData  *dist_data,
    const DimDensityConfig *config,
    DimDensityResults      *results);

#ifdef __cplusplus
}
#endif

#endif // DIMDENSITY_MACK_H

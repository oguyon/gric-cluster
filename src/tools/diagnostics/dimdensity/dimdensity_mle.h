/**
 * @file dimdensity_mle.h
 * @brief Maximum Likelihood Estimation (MLE) of Local Intrinsic Dimension (LID).
 */

#ifndef DIMDENSITY_MLE_H
#define DIMDENSITY_MLE_H

#include "dimdensity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute natural log of the volume of a d-dimensional unit Euclidean ball.
 */
double dimdensity_unit_ball_log_volume(
    double d);

/**
 * Compute local intrinsic dimension for a single sample at neighbor count k.
 */
double dimdensity_estimate_single_mle(
    const double *restrict distances,
    int                    k,
    int                    mode);

/**
 * Run MLE intrinsic dimension estimation across all dataset samples.
 */
int dimdensity_compute_mle_dimensions(
    const KnnDistanceData  *dist_data,
    const DimDensityConfig *config,
    DimDensityResults      *results);

#ifdef __cplusplus
}
#endif

#endif // DIMDENSITY_MLE_H

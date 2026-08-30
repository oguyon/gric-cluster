/**
 * @file dimdensity_stats.h
 * @brief Statistical distribution analysis and histogram computation.
 */

#ifndef DIMDENSITY_STATS_H
#define DIMDENSITY_STATS_H

#include "dimdensity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute comprehensive distribution statistics and histograms.
 */
int dimdensity_compute_stats(
    const DimDensityResults *results,
    DimDensityStats         *stats,
    int                      num_hist_bins);

/**
 * Free memory allocations in DimDensityStats.
 */
void dimdensity_free_stats(
    DimDensityStats *stats);

#ifdef __cplusplus
}
#endif

#endif // DIMDENSITY_STATS_H

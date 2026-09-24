/**
 * @file dimdensity_writer.h
 * @brief Output serialization and terminal dashboard reporting for gric-dimdensity.
 */

#ifndef DIMDENSITY_WRITER_H
#define DIMDENSITY_WRITER_H

#include "dimdensity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Save evaluation results into configured file formats (ASCII, BIN, FITS).
 */
int dimdensity_write_results(
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats);

/**
 * Print rich terminal dashboard summary with percentiles and ASCII histograms.
 */
void dimdensity_print_dashboard(
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats,
    double                   elapsed_ms);

/**
 * Output structured JSON report to stream.
 */
int dimdensity_write_json_report(
    FILE                    *stream,
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats,
    double                   elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif // DIMDENSITY_WRITER_H

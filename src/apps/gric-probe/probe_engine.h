#ifndef PROBE_ENGINE_H
#define PROBE_ENGINE_H

/**
 * @file probe_engine.h
 * @brief Analysis engine for dataset geometry, variance, and parameter tuning.
 */

#include "gric_profile.h"
#include <stdbool.h>
#include <stddef.h>

/** Runtime configuration parameters for gric-probe */
typedef struct
{
    char dataset_path[512];      /**< Path to input dataset or stream */
    int  sample_limit;          /**< Frame count limit (0 = adaptive default) */
    int  force_all;             /**< 1 to scan entire dataset without limit */
    int  warmup_frames;         /**< Number of warmup frames for streaming */
    int  use_double;            /**< 1 to force double precision reading */
    int  verbose_level;         /**< 0 = quiet, 1 = normal, 2 = verbose */
    int  show_progress;         /**< 1 to output live progress tags [PROBE: XX%] */
} ProbeConfig;

/** Statistical diagnostics results */
typedef struct
{
    GricProfile profile;        /**< Standardized profile bundle */
    double      noise_floor_est; /**< Estimated local temporal noise floor */
    int         dead_dims_count; /**< Number of zero-variance coordinates */
    int         clipped_dims_count; /**< Number of saturated coordinates */
    double      quadrant_corr;   /**< Cross-quadrant correlation (image mode) */
} ProbeResults;

/**
 * @brief Run the complete dataset profiling pipeline.
 */
int probe_run(
    const ProbeConfig *config,
    ProbeResults      *results);

/**
 * @brief Free resources allocated in a ProbeResults struct.
 */
void probe_results_free(
    ProbeResults *results);

#endif // PROBE_ENGINE_H

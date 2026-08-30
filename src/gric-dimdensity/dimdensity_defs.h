/**
 * @file dimdensity_defs.h
 * @brief Core data structures and configuration types for gric-dimdensity.
 */

#ifndef DIMDENSITY_DEFS_H
#define DIMDENSITY_DEFS_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** Density smoothing kernel type */
typedef enum
{
    DIMDENSITY_KERNEL_UNIFORM      = 0,
    DIMDENSITY_KERNEL_EPANECHNIKOV = 1,
    DIMDENSITY_KERNEL_GAUSSIAN     = 2
} DimDensityKernel;

/** Output format selection */
typedef enum
{
    DIMDENSITY_FORMAT_AUTO = 0,
    DIMDENSITY_FORMAT_TXT  = 1,
    DIMDENSITY_FORMAT_BIN  = 2,
    DIMDENSITY_FORMAT_FITS = 3
} DimDensityOutputFormat;

/** Runtime configuration parameters for gric-dimdensity */
typedef struct
{
    char                  *input_path;
    char                  *output_path;
    int                    k;             /**< Target k neighbors (0 = auto max) */
    int                    k_min;         /**< Min k for range averaging */
    int                    k_max;         /**< Max k for range averaging */
    int                    use_range_avg; /**< 1 if multi-k smoothing is enabled */
    int                    unbiased_mle;  /**< 1 for (k-2) unbiased MLE, 0 for classic (k-1) */
    DimDensityKernel       kernel_type;   /**< Smoothing kernel choice */
    DimDensityOutputFormat output_format; /**< Preferred output format */
    int                    split_output;  /**< 1 to write separate 1D vector files */
    int                    json_mode;     /**< 1 to output structured JSON report */
    int                    progress_mode; /**< 1 to display live progress */
    int                    verbose_level; /**< 0 = quiet, 1 = normal, 2 = verbose */
    int                    nthreads;      /**< Number of OpenMP worker threads */
} DimDensityConfig;

/** Per-sample evaluation results */
typedef struct
{
    uint64_t num_samples;   /**< Total number of query samples N */
    int      k_used;        /**< Target k used for density estimation */
    double  *local_dim;     /**< Estimated local intrinsic dimension [N] */
    double  *density;       /**< Mack-Rosenblatt local density [N] */
    double  *log_density;   /**< Natural log of local density [N] */
    double  *rk_dist;       /**< Distance to k-th nearest neighbor [N] */
} DimDensityResults;

/** Summary percentiles container */
typedef struct
{
    double p10;
    double p25;
    double p50; /**< Median */
    double p75;
    double p90;
} DimDensityPercentiles;

/** Global statistical distribution summary */
typedef struct
{
    double                dim_mean;
    double                dim_std;
    double                dim_min;
    double                dim_max;
    DimDensityPercentiles dim_pct;

    double                dens_mean;
    double                dens_std;
    double                dens_min;
    double                dens_max;
    DimDensityPercentiles dens_pct;

    double                log_dens_mean;
    double                log_dens_std;
    double                log_dens_min;
    double                log_dens_max;
    DimDensityPercentiles log_dens_pct;

    int                   hist_bins;
    long                 *dim_hist;
    long                 *log_dens_hist;
} DimDensityStats;

/** Loaded k-NN distance matrix container */
typedef struct
{
    uint64_t  num_samples;   /**< Number of query points N */
    int       k_available;   /**< Number of neighbor columns in file */
    double   *distances;     /**< Row-major distance array [N * k_available] */
    int       is_fits;       /**< 1 if loaded from FITS, 0 otherwise */
    char     *resolved_path; /**< Path to actual file opened */
} KnnDistanceData;

#endif // DIMDENSITY_DEFS_H

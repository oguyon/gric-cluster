#ifndef GRIC_PROFILE_H
#define GRIC_PROFILE_H

/**
 * @file gric_profile.h
 * @brief Standardized dataset profile container and serialization API.
 *
 * Defines the GricProfile structure and JSON I/O routines used by gric-probe,
 * gric-cluster, and gric-knn for automated parameter configuration and
 * spectral variance dimension re-ordering.
 */

#include "scalar_quant.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GRIC_PROFILE_MAGIC "GRICPROF_1.0"

/**
 * @brief Complete dataset profile and recommended parameter bundle.
 */
typedef struct
{
    /* Dataset Geometry & Format */
    char     dataset_path[512];      /**< Path to dataset or stream name */
    long     num_frames;             /**< Total or sampled frame count */
    long     width;                  /**< Frame width (or dim if height=1) */
    long     height;                 /**< Frame height (1 for 1D vectors) */
    long     dim;                    /**< Total dimension (width * height) */
    int      is_image;               /**< 1 if 2D image (height > 1), 0 for 1D vector */
    int      is_double;              /**< 1 if float64/double, 0 if float32 */

    /* Radius Calibration & Presets (Distance Percentiles) */
    double   rlim_p01;               /**< 1st percentile radius (~D1%, ultra-fine) */
    double   rlim_p03;               /**< 3rd percentile radius (~D3%, very fine) */
    double   rlim_fine;              /**< Fine granularity radius (~D5%) */
    double   rlim_balanced;          /**< Balanced / nominal radius (~D10%) */
    double   rlim_coarse;            /**< Coarse granularity radius (~D25%) */
    double   rlim_recommended;       /**< Primary suggested radius (default: balanced) */
    char     preset_name[32];        /**< Active preset name ("balanced", "fine", etc.) */

    /* Cluster Budget & Spatial Layout */
    int      recommended_maxcl;      /**< Recommended cluster ceiling */
    int      tiles_x;                /**< Recommended horizontal tiles (1 or 2) */
    int      tiles_y;                /**< Recommended vertical tiles (1 or 2) */

    /* Temporal Dynamics & Prediction */
    int      pred_enabled;           /**< 1 if temporal prediction recommended */
    int      pred_len;               /**< Recommended pattern length (default: 2) */
    int      pred_h;                 /**< Recommended lookback horizon H */
    double   continuity_ratio;       /**< Ratio d_seq / median_dist */
    double   tm_mixing_coeff;        /**< Recommended TM mixing weight (0.0 - 0.5) */

    /* Acceleration & Advanced Clustering */
    int      use_sq8;                /**< 1 if 8-bit scalar quantization recommended */
    SQ8Params sq8_params;            /**< Calibrated SQ8 parameters */
    int      te4_enabled;            /**< 1 if 4-point pruning recommended */
    int      te5_enabled;            /**< 1 if 5-point pruning recommended */
    int      sparse_dcc_enabled;     /**< 1 if sparse DCC recommended for memory */
    int      entropy_enabled;        /**< 1 if entropy-guided search recommended */
    double   entropy_gate;           /**< Entropy gating threshold (bits) */
    int      soft_bayesian_enabled;  /**< 1 if soft Gaussian likelihood recommended */
    double   soft_bayesian_sigma_coeff; /**< Multiplier for radius in soft Bayesian */
    int      recommend_double;       /**< 1 if double precision recommended */
    double   noise_floor_est;        /**< Estimated local temporal noise floor */
    int      ncpu;                   /**< Recommended CPU thread count */

    /* Empirical Metric Pruning Telemetry */
    double   te3_prune_rate;         /**< Fraction of candidates pruned by 3-point ineq */
    double   te4_marginal_rate;      /**< Marginal fraction pruned by 4-point ineq */
    double   te5_marginal_rate;      /**< Marginal fraction pruned by 5-point ineq */
    char     recommended_prune_mode[8]; /**< Recommended mode ("3P", "4P", "5P") */

    /* Distance Distribution Spectrum */
    double   dist_min;               /**< Minimum measured pairwise distance */
    double   dist_p01;               /**< 1st percentile distance */
    double   dist_p03;               /**< 3rd percentile distance */
    double   dist_p05;               /**< 5th percentile distance */
    double   dist_p10;               /**< 10th percentile distance */
    double   dist_p25;               /**< 25th percentile distance */
    double   dist_p50;               /**< 50th percentile (median) distance */
    double   dist_p75;               /**< 75th percentile distance */
    double   dist_p90;               /**< 90th percentile distance */
    double   dist_max;               /**< Maximum measured pairwise distance */

    /* Spectral Dimension Ordering & Residual Bounds */
    long    *perm_dim;               /**< Indices sorted by descending variance [dim] */
    double  *var_dim;                /**< Per-dimension coordinate variance [dim] */
    double  *residual_tail;          /**< Precomputed residual tail envelope [dim] */
} GricProfile;

/**
 * @brief Initialize a GricProfile struct with default values.
 */
void gric_profile_init(
    GricProfile *prof,
    long         dim);

/**
 * @brief Free dynamically allocated arrays inside a GricProfile.
 */
void gric_profile_free(
    GricProfile *prof);

/**
 * @brief Write a GricProfile struct to a structured JSON profile file.
 */
int gric_profile_write_json(
    const char        *filepath,
    const GricProfile *prof);

/**
 * @brief Read a GricProfile struct from a JSON profile file.
 */
int gric_profile_read_json(
    const char  *filepath,
    GricProfile *prof);

/**
 * @brief Auto-locate companion .gricprof file for a given dataset path.
 */
int gric_profile_find_auto(
    const char *dataset_path,
    char       *out_prof_path,
    size_t      max_len);

#endif // GRIC_PROFILE_H

/**
 * @file gric.h
 * @brief Public C API for the GRIC clustering and distance geometry engine.
 */

#ifndef GRIC_H
#define GRIC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef GRIC_BUILDING_DLL
        #define GRIC_API __declspec(dllexport)
    #else
        #define GRIC_API __declspec(dllimport)
    #endif
#else
    #define GRIC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * enum gric_status - Return status codes for GRIC library operations.
 */
typedef enum
{
    GRIC_SUCCESS           = 0,
    GRIC_ERR_INVALID_PARAM = -1,
    GRIC_ERR_OUT_OF_MEMORY = -2,
    GRIC_ERR_DIM_MISMATCH  = -3,
    GRIC_ERR_CAPACITY      = -4,
    GRIC_ERR_GENERIC       = -5
} gric_status_t;

/**
 * struct gric_cluster_config - Configuration parameters for clustering session.
 */
typedef struct
{
    double rlim;                   /**< Clustering radius limit */
    int    maxnbclust;             /**< Maximum clusters capacity (0 = dynamic) */
    long   maxnbfr;                /**< Max frames expected (buffer sizing) */
    double tm_mixing_coeff;        /**< Temporal transition mixing weight */
    int    use_double;             /**< 1 = double precision, 0 = single */
    int    use_sq16;               /**< 1 = enable 16-bit scalar quantization */
    int    use_eq16;               /**< 1 = enable 16-bit E8 lattice quantization */
    int    te4_mode;               /**< 1 = 4-point triangle inequality pruning */
    int    te5_mode;               /**< 1 = 5-point inequality pruning */
    int    entropy_mode;           /**< 1 = entropy-guided target evaluation */
    double entropy_gate_bits;      /**< Entropy information gain threshold */
    int    pred_mode;              /**< Temporal Markov prediction mode */
    int    gprob_mode;             /**< Geometric probability guidance mode */
    int    soft_bayesian_mode;     /**< Soft Bayesian likelihood weighting */
    int    sparse_dcc_mode;        /**< Sparse DCC matrix mode */
    int    sparse_dcc_extra_evals; /**< Extra DCC evals in sparse mode */
    int    maxcl_strategy;         /**< 0=stop, 1=discard lowest count */
    double discard_fraction;       /**< Fraction to discard on eviction */
    int    ncpu;                   /**< Thread count (0 = default/OpenMP) */
} gric_cluster_config_t;

/** Opaque handle to an active clustering engine session. */
typedef struct gric_cluster_ctx gric_cluster_t;

/**
 * gric_version() - Retrieve version string of the compiled GRIC library.
 *
 * Return: Null-terminated semantic version string.
 */
GRIC_API const char *gric_version(
    void);

/**
 * gric_cluster_config_default() - Populate configuration struct with standard defaults.
 * @cfg: Pointer to configuration struct to initialize.
 *
 * Return: GRIC_SUCCESS on success, or GRIC_ERR_INVALID_PARAM if cfg is NULL.
 */
GRIC_API gric_status_t gric_cluster_config_default(
    gric_cluster_config_t *cfg);

/**
 * gric_cluster_create() - Allocate and initialize a clustering session.
 * @cfg:  Configuration struct (or NULL for standard defaults).
 * @ndim: Feature vector dimensionality.
 *
 * Return: Pointer to initialized session handle, or NULL on allocation error.
 */
GRIC_API gric_cluster_t *gric_cluster_create(
    const gric_cluster_config_t *cfg,
    size_t                       ndim);

/**
 * gric_cluster_create_simple() - Simplified constructor with default settings.
 * @ndim: Feature vector dimensionality.
 * @rlim: Clustering radius distance threshold.
 *
 * Return: Pointer to initialized session handle, or NULL on error.
 */
GRIC_API gric_cluster_t *gric_cluster_create_simple(
    size_t ndim,
    double rlim);

/**
 * gric_cluster_feed_frame() - Process a single incoming coordinate frame.
 * @ctx:            Active clustering handle.
 * @coords:         Coordinate array [ndim].
 * @out_cluster_id: Pointer where assigned cluster ID will be stored.
 *
 * Return: GRIC_SUCCESS on success, or negative error code on failure.
 */
GRIC_API gric_status_t gric_cluster_feed_frame(
    gric_cluster_t *ctx,
    const double   *coords,
    int64_t        *out_cluster_id);

/**
 * gric_cluster_feed_batch() - Process a contiguous batch of coordinate frames.
 * @ctx:             Active clustering handle.
 * @coords_flat:     Contiguous coordinate buffer [num_frames * ndim].
 * @num_frames:      Number of frames in batch.
 * @out_cluster_ids: Buffer to store assigned cluster IDs [num_frames].
 *
 * Return: GRIC_SUCCESS on success, or negative error code on failure.
 */
GRIC_API gric_status_t gric_cluster_feed_batch(
    gric_cluster_t *ctx,
    const double   *coords_flat,
    size_t          num_frames,
    int64_t        *out_cluster_ids);

/**
 * gric_cluster_get_num_clusters() - Query number of active clusters discovered.
 * @ctx: Active clustering handle.
 *
 * Return: Number of clusters (>= 0), or -1 if ctx is NULL.
 */
GRIC_API int64_t gric_cluster_get_num_clusters(
    const gric_cluster_t *ctx);

/**
 * gric_cluster_get_anchors() - Export cluster anchor coordinate vectors and member counts.
 * @ctx:         Active clustering handle.
 * @out_coords:  Buffer to receive anchor vectors [num_clusters * ndim].
 * @out_members: Buffer to receive member counts [num_clusters] (can be NULL).
 * @max_anchors: Maximum number of cluster vectors out_coords can hold.
 *
 * Return: Number of anchors exported (>= 0), or negative on error.
 */
GRIC_API int64_t gric_cluster_get_anchors(
    const gric_cluster_t *ctx,
    double               *out_coords,
    int                  *out_members,
    size_t                max_anchors);

/**
 * gric_cluster_get_dcc() - Export distance-between-cluster-centers matrix.
 * @ctx:     Active clustering handle.
 * @out_dcc: Buffer to receive square DCC matrix [K * K].
 * @K:       Number of clusters to export.
 *
 * Return: GRIC_SUCCESS on success, or negative error code.
 */
GRIC_API gric_status_t gric_cluster_get_dcc(
    const gric_cluster_t *ctx,
    double               *out_dcc,
    size_t                K);

/**
 * gric_cluster_reset() - Reset clustering state and clear all discovered clusters.
 * @ctx: Active clustering handle.
 *
 * Return: GRIC_SUCCESS on success, or negative on error.
 */
GRIC_API gric_status_t gric_cluster_reset(
    gric_cluster_t *ctx);

/**
 * gric_cluster_destroy() - Free clustering session and all associated memory.
 * @ctx: Clustering handle to destroy (safe to pass NULL).
 */
GRIC_API void gric_cluster_destroy(
    gric_cluster_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_H */

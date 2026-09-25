/**
 * @file cluster_locator.h
 * @brief Standalone shared cluster localization and 3P metric bounding engine.
 */

#ifndef CLUSTER_LOCATOR_H
#define CLUSTER_LOCATOR_H

#include "eq16_quant.h"
#include "scalar_quant.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/** Status return codes for cluster_locate_sample */
#define CLUSTER_LOCATE_SUCCESS   0
#define CLUSTER_LOCATE_REJECTED  1
#define CLUSTER_LOCATE_ERROR    -1

/** Configuration flags matching gric-cluster options */
typedef struct
{
    int                   max_targets;       /**< Max anchor distance evaluations (default: 8) */
    double                rlim;              /**< Matching radius cutoff (0 = disabled) */
    double                tau_max;           /**< Current k-NN upper bound distance */
    double                epsilon;           /**< Slack factor (1 + eps) for approximate bounds */
    int                   entropy_mode;      /**< 1 = Shannon entropy, 0 = greedy / gprob */
    int                   entropy_fast;      /**< 1 = fast entropy early exit */
    int                   gprob_mode;        /**< 1 = geometric probability weighting */
    int                   te4_mode;          /**< 1 = 4-point geometric bounding */
    int                   te5_mode;          /**< 1 = 5-point geometric bounding */
    int                   prev_cluster_id;   /**< Preceding query cluster ID for warm-starting */
    int                   strict_rlim;       /**< 1 = reject sample if no cluster has d <= rlim */
    int                   is_double;         /**< 1 = double precision, 0 = float */
    const int16_t        *query_eq16;        /**< Optional pre-quantized EQ16 query vector */
    const int16_t        *anchors_eq16_buf;  /**< Optional contiguous [M x dim] EQ16 anchors */
    const int16_t *const *anchors_eq16_ptrs; /**< Optional array of M pointers to EQ16 anchors */
    const EQ16Params     *eq16_params;       /**< EQ16 calibration parameters */
    const int16_t        *query_sq16;        /**< Optional pre-quantized SQ16 query vector */
    const int16_t        *anchors_sq16_buf;  /**< Optional contiguous [M x dim] SQ16 anchors */
    const int16_t *const *anchors_sq16_ptrs; /**< Optional array of M pointers to SQ16 anchors */
    const SQ16Params     *sq16_params;       /**< SQ16 calibration parameters */
    const uint8_t        *query_sq8;         /**< Optional pre-quantized SQ8 query vector */
    const uint8_t        *anchors_sq8_buf;   /**< Optional contiguous [M x dim] SQ8 anchors */
    const uint8_t *const *anchors_sq8_ptrs;  /**< Optional array of M pointers to SQ8 anchors */
    const SQ8Params      *sq8_params;        /**< SQ8 calibration parameters */
} ClusterLocatorConfig;

/** Results of the coarse cluster localization */
typedef struct
{
    int      best_cluster_id;        /**< Closest cluster index (-1 if rejected/none) */
    double   best_anchor_dist;       /**< Distance to best cluster anchor */
    int      num_evaluated_anchors;  /**< Number of anchor distances computed */
    int      evaluated_clusters[32]; /**< Array of evaluated cluster indices */
    double   evaluated_dists[32];    /**< Array of computed anchor distances */
    uint8_t *active_cluster_mask;    /**< Array: 1 = surviving, 0 = pruned */
} ClusterLocatorResult;

/** Precomputed reference coordinates for 4-point inequality */
typedef struct
{
    double x4;
    double y4;
    double d12;
    double inv_2d12;
    double d12_sq;
    int    valid;
} TE4Ref;

/** Precomputed reference coordinates for 5-point inequality */
typedef struct
{
    double x3;
    double y3;
    double xF;
    double yF;
    double zF;
    double d12;
    double inv_2d12;
    double inv_2y3;
    double d12_sq;
    double d13_sq;
    int    valid;
} TE5Ref;

/**
 * calc_te4_ref_init() - Initialize precomputed 4-point reference geometry.
 * @ref: Output reference structure.
 * @d14: Distance between point 1 and 4.
 * @d24: Distance between point 2 and 4.
 * @d12: Distance between point 1 and 2.
 */
void calc_te4_ref_init(
    TE4Ref *ref,
    double  d14,
    double  d24,
    double  d12);

/**
 * calc_min_dist_4pt_ref() - Compute min 4-point distance using cached geometry.
 * @ref: Initialized 4-point reference structure.
 * @d13: Distance between point 1 and 3.
 * @d23: Distance between point 2 and 3.
 *
 * Return: Minimum reconstructed distance between points 3 and 4.
 */
static inline double calc_min_dist_4pt_ref(
    const TE4Ref *ref,
    double        d13,
    double        d23)
{
    if (!ref->valid)
    {
        return fabs(ref->x4 - d13);
    }

    double d13_sq = d13 * d13;
    double x3 = (d13_sq + ref->d12_sq - d23 * d23) * ref->inv_2d12;
    double y3_sq = d13_sq - x3 * x3;
    double y3 = (y3_sq > 0.0) ? sqrt(y3_sq) : 0.0;
    double dx = x3 - ref->x4;
    double dy = y3 - ref->y4;
    return sqrt(dx * dx + dy * dy);
}

/**
 * calc_te5_ref_init() - Initialize precomputed 5-point reference geometry.
 * @ref:     Output reference structure.
 * @d_f_c1:  Distance between frame and cluster 1.
 * @d_f_c2:  Distance between frame and cluster 2.
 * @d_f_c3:  Distance between frame and cluster 3.
 * @d_c1_c2: Mutual distance between cluster 1 and 2.
 * @d_c1_c3: Mutual distance between cluster 1 and 3.
 * @d_c2_c3: Mutual distance between cluster 2 and 3.
 */
void calc_te5_ref_init(
    TE5Ref *ref,
    double  d_f_c1,
    double  d_f_c2,
    double  d_f_c3,
    double  d_c1_c2,
    double  d_c1_c3,
    double  d_c2_c3);

/**
 * calc_min_dist_5pt_ref() - Compute min 5-point distance using cached geometry.
 * @ref:    Initialized 5-point reference structure.
 * @d_t_c1: Distance between target and cluster 1.
 * @d_t_c2: Distance between target and cluster 2.
 * @d_t_c3: Distance between target and cluster 3.
 *
 * Return: Minimum reconstructed 3D distance between frame and target.
 */
double calc_min_dist_5pt_ref(
    const TE5Ref *ref,
    double        d_t_c1,
    double        d_t_c2,
    double        d_t_c3);

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * calc_min_dist_4pt_batch4_avx2() - AVX2 vectorized 4-way 4-point evaluation.
 * @ref:       Initialized 4-point reference structure.
 * @d13:       Array of 4 distance values to point 1.
 * @d23:       Array of 4 distance values to point 2.
 * @out_dists: Output array for 4 computed min distances.
 */
void calc_min_dist_4pt_batch4_avx2(
    const TE4Ref          *ref,
    const double *restrict d13,
    const double *restrict d23,
    double       *restrict out_dists);

/**
 * calc_min_dist_5pt_batch4_avx2() - AVX2 vectorized 4-way 5-point evaluation.
 * @ref:       Initialized 5-point reference structure.
 * @d_t_c1:    Array of 4 distance values to cluster 1.
 * @d_t_c2:    Array of 4 distance values to cluster 2.
 * @d_t_c3:    Array of 4 distance values to cluster 3.
 * @out_dists: Output array for 4 computed min distances.
 */
void calc_min_dist_5pt_batch4_avx2(
    const TE5Ref          *ref,
    const double *restrict d_t_c1,
    const double *restrict d_t_c2,
    const double *restrict d_t_c3,
    double       *restrict out_dists);

/**
 * calc_min_dist_4pt_batch8_avx512() - AVX-512 vectorized 8-way 4-point evaluation.
 * @ref:       Initialized 4-point reference structure.
 * @d13:       Array of 8 distance values to point 1.
 * @d23:       Array of 8 distance values to point 2.
 * @out_dists: Output array for 8 computed min distances.
 */
void calc_min_dist_4pt_batch8_avx512(
    const TE4Ref          *ref,
    const double *restrict d13,
    const double *restrict d23,
    double       *restrict out_dists);

/**
 * calc_min_dist_5pt_batch8_avx512() - AVX-512 vectorized 8-way 5-point evaluation.
 * @ref:       Initialized 5-point reference structure.
 * @d_t_c1:    Array of 8 distance values to cluster 1.
 * @d_t_c2:    Array of 8 distance values to cluster 2.
 * @d_t_c3:    Array of 8 distance values to cluster 3.
 * @out_dists: Output array for 8 computed min distances.
 */
void calc_min_dist_5pt_batch8_avx512(
    const TE5Ref          *ref,
    const double *restrict d_t_c1,
    const double *restrict d_t_c2,
    const double *restrict d_t_c3,
    double       *restrict out_dists);
#endif

/**
 * calc_min_dist_4pt() - Computes minimum distance using a 4-point configuration.
 * @d14: Distance between point 1 and 4.
 * @d24: Distance between point 2 and 4.
 * @d12: Distance between point 1 and 2.
 * @d13: Distance between point 1 and 3.
 * @d23: Distance between point 2 and 3.
 *
 * Return: Min distance between point 3 and 4 in reconstructed 2D space.
 */
double calc_min_dist_4pt(
    double d14,
    double d24,
    double d12,
    double d13,
    double d23);

/**
 * calc_min_dist_5pt() - Computes minimum distance using a 5-point configuration.
 * @d_f_c1:  Distance from frame F to C1.
 * @d_f_c2:  Distance from frame F to C2.
 * @d_f_c3:  Distance from frame F to C3.
 * @d_t_c1:  Distance from target T to C1.
 * @d_t_c2:  Distance from target T to C2.
 * @d_t_c3:  Distance from target T to C3.
 * @d_c1_c2: Distance between C1 and C2.
 * @d_c1_c3: Distance between C1 and C3.
 * @d_c2_c3: Distance between C2 and C3.
 *
 * Return: Computed distance between F and T in reconstructed 3D space.
 */
double calc_min_dist_5pt(
    double d_f_c1,
    double d_f_c2,
    double d_f_c3,
    double d_t_c1,
    double d_t_c2,
    double d_t_c3,
    double d_c1_c2,
    double d_c1_c3,
    double d_c2_c3);

/**
 * cluster_locate_sample() - Locates the matching/closest cluster for sample q.
 * @query_data:       Pointer to sample/query frame vector.
 * @frame_elements:   Number of elements per vector.
 * @num_clusters:     Total clusters M in model.
 * @cluster_anchors:  Array of pointers to anchor vectors [M].
 * @cluster_radii:    Array of cluster radii [M].
 * @dcc_matrix:       Dense M x M inter-cluster distance matrix.
 * @config:           Tuning and runtime configuration.
 * @result:           Output structure to populate.
 *
 * Return: CLUSTER_LOCATE_SUCCESS (0), CLUSTER_LOCATE_REJECTED (1), or error (-1).
 */
int cluster_locate_sample(
    const void                 *query_data,
    long                        frame_elements,
    int                         num_clusters,
    const void *const          *cluster_anchors,
    const double               *cluster_radii,
    const double               *dcc_matrix,
    const ClusterLocatorConfig *config,
    ClusterLocatorResult       *result);

/**
 * @brief Computes Euclidean distance between two vectors.
 */
double compute_vector_distance(
    const void *restrict a,
    const void *restrict b,
    long                 n,
    int                  is_double);

#endif // CLUSTER_LOCATOR_H

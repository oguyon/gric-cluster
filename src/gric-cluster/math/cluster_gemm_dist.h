/**
 * @file cluster_gemm_dist.h
 * @brief High-throughput batched matrix Euclidean distance engine (BLAS Level 3).
 *
 * Implements the FAISS-style matrix multiplication distance formulation:
 *   ||Q_i - X_j||^2 = ||Q_i||^2 + ||X_j||^2 - 2 * (Q * X^T)_ij
 * using an in-tree register-tiled fused micro-kernel (AVX2/AVX-512) and optional
 * external BLAS (cblas_sgemm/cblas_dgemm).
 */

#ifndef CLUSTER_GEMM_DIST_H
#define CLUSTER_GEMM_DIST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cluster_compute_l2_norms_float() - Precompute squared L2 norms of float vectors.
 */
int cluster_compute_l2_norms_float(
    const float *restrict mat,
    int                   count,
    int                   dim,
    float       *restrict out_norms);

/**
 * cluster_compute_l2_norms_double() - Precompute squared L2 norms of double vectors.
 */
int cluster_compute_l2_norms_double(
    const double *restrict mat,
    int                    count,
    int                    dim,
    double       *restrict out_norms);

/**
 * cluster_gemm_dist_float() - Batched Euclidean distance between contiguous matrices.
 */
int cluster_gemm_dist_float(
    const float *restrict Q,
    const float *restrict X,
    const float *restrict q_norms,
    const float *restrict x_norms,
    int                   M,
    int                   N,
    int                   D,
    float       *restrict out_dist_sq,
    double      *restrict out_dists);

/**
 * cluster_gemm_dist_double() - Batched Euclidean distance between contiguous matrices.
 */
int cluster_gemm_dist_double(
    const double *restrict Q,
    const double *restrict X,
    const double *restrict q_norms,
    const double *restrict x_norms,
    int                    M,
    int                    N,
    int                    D,
    double       *restrict out_dist_sq,
    double       *restrict out_dists);

/**
 * cluster_gemm_dist_ptrs_float() - Batched Euclidean distance with non-contiguous candidates.
 */
int cluster_gemm_dist_ptrs_float(
    const float *restrict        Q,
    const float *const *restrict cand_ptrs,
    const float *restrict        q_norms,
    const float *restrict        x_norms,
    int                          M,
    int                          N,
    int                          D,
    float       *restrict        out_dist_sq,
    double      *restrict        out_dists);

/**
 * cluster_gemm_dist_ptrs_double() - Batched Euclidean distance with non-contiguous candidates.
 */
int cluster_gemm_dist_ptrs_double(
    const double *restrict        Q,
    const double *const *restrict cand_ptrs,
    const double *restrict        q_norms,
    const double *restrict        x_norms,
    int                           M,
    int                           N,
    int                           D,
    double       *restrict        out_dist_sq,
    double       *restrict        out_dists);

#ifdef __cplusplus
}
#endif

#endif /* CLUSTER_GEMM_DIST_H */

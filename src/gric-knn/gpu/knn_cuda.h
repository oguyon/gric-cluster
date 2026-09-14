/**
 * @file knn_cuda.h
 * @brief GPU-accelerated k-NN search engine for gric-knn.
 */

#ifndef KNN_CUDA_H
#define KNN_CUDA_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if CUDA GPU acceleration is available for k-NN.
 *
 * @return 1 if GPU is available, 0 otherwise.
 */
int knn_cuda_is_available(void);

/**
 * @brief Execute GPU-accelerated batched k-NN search.
 *
 * @param config    Active KnnConfig configuration.
 * @param model     Active KnnModel containing cluster model or dataset.
 * @param results   Pointer to KnnResults structure to receive sorted indices and distances.
 * @param telemetry Pointer to KnnTelemetry structure to receive performance counters.
 *
 * @return 0 on success, -1 on failure/fallback.
 */
int knn_cuda_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry);

/**
 * @brief Locate top-M closest cluster anchors for query frames using GPU cuBLAS GEMM.
 *
 * @param model          Active KnnModel containing cluster anchors.
 * @param host_queries   Host buffer containing contiguous query frames [num_queries x D].
 * @param num_queries    Number of query frames.
 * @param is_double      1 if queries are double precision, 0 if single precision.
 * @param m              Number of nearest anchors to return per query frame (1..32).
 * @param out_cl_indices Output buffer [num_queries x m] to receive nearest cluster indices.
 * @param out_cl_dists   Output buffer [num_queries x m] to receive distances to anchors.
 *
 * @return 0 on success, -1 on error.
 */
int knn_cuda_locate_anchors(
    const KnnModel *model,
    const void     *host_queries,
    int             num_queries,
    int             is_double,
    int             m,
    int            *out_cl_indices,
    float          *out_cl_dists);

#ifdef __cplusplus
}
#endif

#endif /* KNN_CUDA_H */

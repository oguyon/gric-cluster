/**
 * @file knn_cuda_ivf.h
 * @brief GPU Inverted-File (IVF) hierarchical metric pruned k-NN search engine.
 *
 * Declares the GPU Inverted-File (IVF) search entry point knn_cuda_run_ivf_search().
 * Dispatches queries to GPU kernels that perform coarse cluster anchor routing followed
 * by fine-grained inverted-list member evaluations with dynamic metric bounds on CUDA devices.
 */

#ifndef KNN_CUDA_IVF_H
#define KNN_CUDA_IVF_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_cuda_run_ivf_search() - Execute GPU hierarchical metric pruned k-NN search.
 * @config:    Active KnnConfig.
 * @model:     Active KnnModel with clusters and dataset vectors.
 * @results:   Output KnnResults structure to populate.
 * @telemetry: Output KnnTelemetry structure for timing and eviction stats.
 *
 * Return: 0 on success, -1 on failure/fallback.
 */
int knn_cuda_run_ivf_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry);

/**
 * knn_cuda_ivf_search_single_frame() - Ultra-low-latency single query streaming search.
 * @model:       Active KnnModel.
 * @query_frame: Pointer to single query frame data (float or double).
 * @is_double:   1 if query frame is double precision, 0 if float.
 * @k:           Number of nearest neighbors to retrieve.
 * @out_indices: Output buffer [k] to receive nearest neighbor frame IDs.
 * @out_dists:   Output buffer [k] to receive nearest neighbor distances.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_cuda_ivf_search_single_frame(
    const KnnModel *model,
    const void     *query_frame,
    int             is_double,
    int             k,
    int            *out_indices,
    float          *out_dists);

#ifdef __cplusplus
}
#endif

#endif /* KNN_CUDA_IVF_H */

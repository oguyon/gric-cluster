/**
 * @file cuda_anchor_store.h
 * @brief Persistent GPU-resident cluster anchor store and fast nearest-anchor engine.
 */

#ifndef CUDA_ANCHOR_STORE_H
#define CUDA_ANCHOR_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuAnchorStore GpuAnchorStore;

/**
 * GpuAnchorStoreConfig - Configuration parameters for GPU anchor store allocation.
 * @max_clusters:   Maximum number of cluster anchors (capacity).
 * @dim:            Dimensionality D of each anchor vector.
 * @device_id:      GPU device index (0 for default).
 * @max_batch_size: Maximum batch size of query frames processed per GEMM call.
 * @rlim:           Metric cutoff radius threshold (<= 0.0f to disable).
 */
typedef struct
{
    int   max_clusters;
    int   dim;
    int   device_id;
    int   max_batch_size;
    float rlim;
} GpuAnchorStoreConfig;

/**
 * gpu_anchor_store_set_rlim() - Set distance cutoff radius for metric pruning.
 * @store: Pointer to active GpuAnchorStore.
 * @rlim:  Metric cutoff radius (<= 0.0f to disable).
 */
void gpu_anchor_store_set_rlim(
    GpuAnchorStore *store,
    float           rlim);

/**
 * gpu_anchor_store_create() - Allocate and initialize a persistent GPU anchor store.
 * @config: Pointer to configuration parameters.
 *
 * Return: Pointer to allocated GpuAnchorStore on success, NULL on error.
 */
GpuAnchorStore *gpu_anchor_store_create(
    const GpuAnchorStoreConfig *config);

/**
 * gpu_anchor_store_destroy() - Free all device and host resources of an anchor store.
 * @store: Pointer to GpuAnchorStore to destroy.
 */
void gpu_anchor_store_destroy(
    GpuAnchorStore *store);

/**
 * gpu_anchor_store_reset() - Reset the active cluster count to 0 without freeing memory.
 * @store: Pointer to active GpuAnchorStore.
 */
void gpu_anchor_store_reset(
    GpuAnchorStore *store);

/**
 * gpu_anchor_store_get_count() - Query current number of active clusters in the store.
 * @store: Pointer to active GpuAnchorStore.
 *
 * Return: Number of active clusters K.
 */
int gpu_anchor_store_get_count(
    const GpuAnchorStore *store);

/**
 * gpu_anchor_store_append_anchor() - Append a new cluster anchor from host memory.
 * @store:     Pointer to active GpuAnchorStore.
 * @data:      Pointer to host vector data.
 * @is_double: 1 if vector is double precision, 0 if single precision.
 *
 * Return: New cluster index (0 .. K-1), or -1 on capacity exhaustion or error.
 */
int gpu_anchor_store_append_anchor(
    GpuAnchorStore *store,
    const void     *data,
    int             is_double);

/**
 * gpu_anchor_store_find_nearest() - Find closest cluster anchor for a batch of frames.
 * @store:         Pointer to active GpuAnchorStore.
 * @host_frames:   Pointer to contiguous host frame buffer [batch_size x dim].
 * @batch_size:    Number of frames in the batch.
 * @is_double:     1 if frames are double precision, 0 if float.
 * @out_best_cl:   Output array [batch_size] to receive winning cluster IDs.
 * @out_best_dist: Output array [batch_size] to receive distances to winning anchors.
 *
 * Return: 0 on success, -1 on error.
 */
int gpu_anchor_store_find_nearest(
    GpuAnchorStore *store,
    const void     *host_frames,
    int             batch_size,
    int             is_double,
    int            *out_best_cl,
    float          *out_best_dist);

/**
 * gpu_anchor_store_find_top_m() - Find top-M closest cluster anchors for query frames.
 * @store:          Pointer to active GpuAnchorStore.
 * @host_queries:   Pointer to contiguous host query buffer [num_queries x dim].
 * @num_queries:    Number of query frames.
 * @is_double:      1 if queries are double precision, 0 if float.
 * @m:              Number of nearest anchors to return per query (e.g. 1 to 32).
 * @out_cl_indices: Output array [num_queries x m] for anchor indices.
 * @out_cl_dists:   Output array [num_queries x m] for anchor distances.
 *
 * Return: 0 on success, -1 on error.
 */
int gpu_anchor_store_find_top_m(
    GpuAnchorStore *store,
    const void     *host_queries,
    int             num_queries,
    int             is_double,
    int             m,
    int            *out_cl_indices,
    float          *out_cl_dists);

/**
 * gpu_anchor_store_get_anchors_host() - Download all resident anchors to host memory.
 * @store:             Pointer to active GpuAnchorStore.
 * @out_host_anchors:  Output buffer of size [max_clusters x dim].
 * @max_clusters:      Capacity of output buffer in anchors.
 *
 * Return: Number of anchors copied, or -1 on error.
 */
int gpu_anchor_store_get_anchors_host(
    const GpuAnchorStore *store,
    float                *out_host_anchors,
    int                   max_clusters);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ANCHOR_STORE_H */

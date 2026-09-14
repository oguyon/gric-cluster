/**
 * @file cuda_ivf_index.h
 * @brief GPU-resident Inverted File (IVF) index for hierarchical metric pruned k-NN.
 */

#ifndef CUDA_IVF_INDEX_H
#define CUDA_IVF_INDEX_H

#include <stddef.h>
#include <stdint.h>
#include "gric-knn/knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GpuIvfIndex - Inverted-file index resident in GPU global memory.
 * @num_clusters:        Number of cluster partitions K.
 * @dim:                 Vector dimensionality D.
 * @total_members:       Total candidate frames indexed across all clusters.
 * @total_dataset_frames: Total dataset frames N.
 * @device_id:           CUDA device index.
 * @d_cluster_offsets:   Device array [K + 1] of member start offsets.
 * @d_cluster_sizes:     Device array [K] of member counts per cluster.
 * @d_cluster_radii:     Device array [K] of cluster radii.
 * @d_member_frame_ids:  Device array [total_members] of original frame indices.
 * @d_member_r_anchors:  Device array [total_members] of member distances to anchor.
 * @d_vectors_ivf:       Device array [total_members x D] of candidate vectors.
 */
typedef struct
{
    int    num_clusters;
    int    dim;
    long   total_members;
    long   total_dataset_frames;
    int    device_id;
    int   *d_cluster_offsets;
    int   *d_cluster_sizes;
    float *d_cluster_radii;
    int   *d_member_frame_ids;
    float *d_member_r_anchors;
    float *d_vectors_ivf;
    int    cluster_graph_k;
    int   *d_cluster_graph_adj;
    int    has_rq8;
    int    dim_words;
    int32_t *d_vectors_rq8_words;
    int32_t *d_member_norms_sq;
    float *d_cluster_scales;
    float *d_cluster_inv_scales;
    float *d_cluster_err_radii;
} GpuIvfIndex;

/**
 * gpu_ivf_index_create() - Build GPU inverted-file index from active KnnModel.
 * @model:     Pointer to active KnnModel with clusters and dataset buffer.
 * @device_id: GPU device index (0 for default).
 *
 * Return: Pointer to allocated GpuIvfIndex on success, NULL on error.
 */
GpuIvfIndex *gpu_ivf_index_create(
    const KnnModel *model,
    int             device_id);

/**
 * gpu_ivf_index_destroy() - Free all GPU resources associated with GpuIvfIndex.
 * @idx: Pointer to GpuIvfIndex to deallocate.
 */
void gpu_ivf_index_destroy(
    GpuIvfIndex *idx);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_IVF_INDEX_H */

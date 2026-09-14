/**
 * @file cuda_ivf_index.cu
 * @brief GPU-resident Inverted File (IVF) index for hierarchical metric pruned k-NN.
 */

#include "cuda_ivf_index.h"
#include "cuda_common.h"
#include "gric-knn/knn_tree.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK_GOTO(call, label)                                          \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            goto label;                                                       \
        }                                                                     \
    } while (0)

/**
 * gpu_ivf_index_create() - Build GPU inverted-file index from active KnnModel.
 * @model:     Pointer to active KnnModel with clusters and dataset buffer.
 * @device_id: GPU device index (0 for default).
 *
 * Return: Pointer to allocated GpuIvfIndex on success, NULL on error.
 */
GpuIvfIndex *gpu_ivf_index_create(
    const KnnModel *model,
    int             device_id)
{
    if (model == NULL || model->num_clusters <= 0 || model->frame_elements <= 0)
    {
        return NULL;
    }

    if (gric_cuda_init(device_id) != 0)
    {
        return NULL;
    }

    int K = model->num_clusters;
    int D = (int)model->frame_elements;

    int   *h_offsets = (int *)malloc((size_t)(K + 1) * sizeof(int));
    int   *h_sizes = (int *)malloc((size_t)K * sizeof(int));
    float *h_radii = (float *)malloc((size_t)K * sizeof(float));

    if (h_offsets == NULL || h_sizes == NULL || h_radii == NULL)
    {
        free(h_offsets);
        free(h_sizes);
        free(h_radii);
        return NULL;
    }

    h_offsets[0] = 0;
    for (int c = 0; c < K; c++)
    {
        h_sizes[c] = model->clusters[c].num_members;
        h_radii[c] = (float)model->clusters[c].radius;
        h_offsets[c + 1] = h_offsets[c] + h_sizes[c];
    }

    long total_members = (long)h_offsets[K];
    if (total_members <= 0)
    {
        free(h_offsets);
        free(h_sizes);
        free(h_radii);
        return NULL;
    }

    int   *h_member_ids = (int *)malloc((size_t)total_members * sizeof(int));
    float *h_member_r_anchors = (float *)malloc((size_t)total_members * sizeof(float));
    float *h_vectors_ivf = (float *)malloc((size_t)total_members * (size_t)D * sizeof(float));

    if (h_member_ids == NULL || h_member_r_anchors == NULL || h_vectors_ivf == NULL)
    {
        free(h_offsets);
        free(h_sizes);
        free(h_radii);
        free(h_member_ids);
        free(h_member_r_anchors);
        free(h_vectors_ivf);
        return NULL;
    }

    /* Populate contiguous member vectors, r_anchor and frame IDs ordered by cluster */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int c = 0; c < K; c++)
    {
        int off = h_offsets[c];
        int count = h_sizes[c];

        for (int j = 0; j < count; j++)
        {
            int fid = (int)model->clusters[c].members[j].frame_id;
            h_member_ids[off + j] = fid;
            h_member_r_anchors[off + j] = (float)model->clusters[c].members[j].r_anchor;

            float *dst = h_vectors_ivf + (size_t)(off + j) * (size_t)D;

            if (model->dataset_buffer != NULL)
            {
                if (model->is_double)
                {
                    const double *src = (const double *)model->dataset_buffer +
                                        (size_t)fid * (size_t)D;
                    for (int d = 0; d < D; d++)
                    {
                        dst[d] = (float)src[d];
                    }
                }
                else
                {
                    const float *src = (const float *)model->dataset_buffer +
                                       (size_t)fid * (size_t)D;
                    memcpy(dst, src, (size_t)D * sizeof(float));
                }
            }
            else
            {
                memset(dst, 0, (size_t)D * sizeof(float));
            }
        }
    } // for (int c = 0; ...)

    GpuIvfIndex *idx = (GpuIvfIndex *)calloc(1, sizeof(GpuIvfIndex));
    if (idx == NULL)
    {
        free(h_offsets);
        free(h_sizes);
        free(h_radii);
        free(h_member_ids);
        free(h_member_r_anchors);
        free(h_vectors_ivf);
        return NULL;
    }

    idx->num_clusters = K;
    idx->dim = D;
    idx->total_members = total_members;
    idx->total_dataset_frames = model->total_dataset_frames;
    idx->device_id = device_id;

    /* Allocate device global memory buffers */
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_cluster_offsets,
                               (size_t)(K + 1) * sizeof(int)), error_cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_cluster_sizes,
                               (size_t)K * sizeof(int)), error_cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_cluster_radii,
                               (size_t)K * sizeof(float)), error_cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_member_frame_ids,
                               (size_t)total_members * sizeof(int)), error_cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_member_r_anchors,
                               (size_t)total_members * sizeof(float)), error_cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_vectors_ivf,
                               (size_t)total_members * (size_t)D * sizeof(float)), error_cleanup);

    /* Transfer index structures and sorted candidate matrix to VRAM */
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_cluster_offsets, h_offsets,
                               (size_t)(K + 1) * sizeof(int),
                               cudaMemcpyHostToDevice), error_cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_cluster_sizes, h_sizes,
                               (size_t)K * sizeof(int),
                               cudaMemcpyHostToDevice), error_cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_cluster_radii, h_radii,
                               (size_t)K * sizeof(float),
                               cudaMemcpyHostToDevice), error_cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_member_frame_ids, h_member_ids,
                               (size_t)total_members * sizeof(int),
                               cudaMemcpyHostToDevice), error_cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_member_r_anchors, h_member_r_anchors,
                               (size_t)total_members * sizeof(float),
                               cudaMemcpyHostToDevice), error_cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(idx->d_vectors_ivf, h_vectors_ivf,
                               (size_t)total_members * (size_t)D * sizeof(float),
                               cudaMemcpyHostToDevice), error_cleanup);

    /* Upload cluster proximity graph if available */
    if (model->cluster_graph_adj == NULL && model->dcc_matrix != NULL)
    {
        knn_build_cluster_graph((KnnModel *)model);
    }

    if (model->cluster_graph_adj != NULL && model->cluster_graph_k > 0)
    {
        int k_adj = model->cluster_graph_k;
        idx->cluster_graph_k = k_adj;
        size_t graph_sz = (size_t)K * (size_t)k_adj * sizeof(int);
        CUDA_CHECK_GOTO(cudaMalloc((void **)&idx->d_cluster_graph_adj, graph_sz),
                        error_cleanup);
        CUDA_CHECK_GOTO(cudaMemcpy(idx->d_cluster_graph_adj, model->cluster_graph_adj,
                                   graph_sz, cudaMemcpyHostToDevice),
                        error_cleanup);
    }

    /* Build INT8 residual quantization structures if dimension is divisible by 4 */
    if (D >= 4 && (D % 4 == 0))
    {
        int dim_w = D / 4;
        float   *h_scales = (float *)malloc((size_t)K * sizeof(float));
        float   *h_inv_scales = (float *)malloc((size_t)K * sizeof(float));
        float   *h_err_radii = (float *)malloc((size_t)K * sizeof(float));
        int32_t *h_rq8_words = (int32_t *)malloc((size_t)total_members *
                                                 (size_t)dim_w * sizeof(int32_t));
        int32_t *h_norms_sq = (int32_t *)malloc((size_t)total_members * sizeof(int32_t));

        if (h_scales != NULL && h_inv_scales != NULL && h_err_radii != NULL &&
            h_rq8_words != NULL && h_norms_sq != NULL)
        {
            float *temp_anchor = (float *)malloc((size_t)D * sizeof(float));
            for (int c = 0; c < K; c++)
            {
                float rlim = (h_radii[c] > 0.0f) ? h_radii[c] : 1.0f;
                float scale = rlim / 127.0f;
                float inv_scale = 127.0f / rlim;
                float err_radius = sqrtf((float)D) * scale * 0.5f;

                h_scales[c] = scale;
                h_inv_scales[c] = inv_scale;
                h_err_radii[c] = err_radius;

                if (model->clusters[c].anchor_data != NULL && temp_anchor != NULL)
                {
                    if (model->is_double)
                    {
                        const double *src_a = (const double *)model->clusters[c].anchor_data;
                        for (int d = 0; d < D; d++)
                        {
                            temp_anchor[d] = (float)src_a[d];
                        }
                    }
                    else
                    {
                        memcpy(temp_anchor, model->clusters[c].anchor_data,
                               (size_t)D * sizeof(float));
                    }
                }
                else if (temp_anchor != NULL)
                {
                    memset(temp_anchor, 0, (size_t)D * sizeof(float));
                }

                int off = h_offsets[c];
                int count = h_sizes[c];
                for (int j = 0; j < count; j++)
                {
                    int cand_idx = off + j;
                    const float *cand_v = h_vectors_ivf + (size_t)cand_idx * (size_t)D;
                    int32_t total_norm_sq = 0;

                    for (int w = 0; w < dim_w; w++)
                    {
                        int8_t b[4];
                        for (int sub = 0; sub < 4; sub++)
                        {
                            int d = w * 4 + sub;
                            float a_val = (temp_anchor != NULL) ? temp_anchor[d] : 0.0f;
                            float res = (cand_v[d] - a_val) * inv_scale;
                            int val = (int)roundf(res);
                            if (val < -127)
                            {
                                val = -127;
                            }
                            if (val > 127)
                            {
                                val = 127;
                            }
                            b[sub] = (int8_t)val;
                            total_norm_sq += val * val;
                        }
                        int32_t packed = ((uint32_t)(uint8_t)b[0]) |
                                         (((uint32_t)(uint8_t)b[1]) << 8) |
                                         (((uint32_t)(uint8_t)b[2]) << 16) |
                                         (((uint32_t)(uint8_t)b[3]) << 24);
                        h_rq8_words[(size_t)cand_idx * (size_t)dim_w + (size_t)w] = packed;
                    }
                    h_norms_sq[cand_idx] = total_norm_sq;
                }
            } // for (int c = 0; c < K; c++)
            free(temp_anchor);

            /* Allocate GPU buffers for INT8 DP4A FastScan */
            if (cudaMalloc((void **)&idx->d_cluster_scales,
                           (size_t)K * sizeof(float)) == cudaSuccess &&
                cudaMalloc((void **)&idx->d_cluster_inv_scales,
                           (size_t)K * sizeof(float)) == cudaSuccess &&
                cudaMalloc((void **)&idx->d_cluster_err_radii,
                           (size_t)K * sizeof(float)) == cudaSuccess &&
                cudaMalloc((void **)&idx->d_vectors_rq8_words,
                           (size_t)total_members * (size_t)dim_w *
                           sizeof(int32_t)) == cudaSuccess &&
                cudaMalloc((void **)&idx->d_member_norms_sq,
                           (size_t)total_members * sizeof(int32_t)) == cudaSuccess)
            {
                cudaMemcpy(idx->d_cluster_scales, h_scales,
                           (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(idx->d_cluster_inv_scales, h_inv_scales,
                           (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(idx->d_cluster_err_radii, h_err_radii,
                           (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(idx->d_vectors_rq8_words, h_rq8_words,
                           (size_t)total_members * (size_t)dim_w * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(idx->d_member_norms_sq, h_norms_sq,
                           (size_t)total_members * sizeof(int32_t), cudaMemcpyHostToDevice);

                idx->has_rq8 = 1;
                idx->dim_words = dim_w;
            }
        }

        free(h_scales);
        free(h_inv_scales);
        free(h_err_radii);
        free(h_rq8_words);
        free(h_norms_sq);
    } // if (D >= 4 && (D % 4 == 0))

    free(h_offsets);
    free(h_sizes);
    free(h_radii);
    free(h_member_ids);
    free(h_member_r_anchors);
    free(h_vectors_ivf);
    return idx;

error_cleanup:
    free(h_offsets);
    free(h_sizes);
    free(h_radii);
    free(h_member_ids);
    free(h_member_r_anchors);
    free(h_vectors_ivf);
    gpu_ivf_index_destroy(idx);
    return NULL;
}

/**
 * gpu_ivf_index_destroy() - Free all GPU resources associated with GpuIvfIndex.
 * @idx: Pointer to GpuIvfIndex to deallocate.
 */
void gpu_ivf_index_destroy(
    GpuIvfIndex *idx)
{
    if (idx == NULL)
    {
        return;
    }

    if (idx->d_cluster_scales != NULL)
    {
        cudaFree(idx->d_cluster_scales);
        idx->d_cluster_scales = NULL;
    }
    if (idx->d_cluster_inv_scales != NULL)
    {
        cudaFree(idx->d_cluster_inv_scales);
        idx->d_cluster_inv_scales = NULL;
    }
    if (idx->d_cluster_err_radii != NULL)
    {
        cudaFree(idx->d_cluster_err_radii);
        idx->d_cluster_err_radii = NULL;
    }
    if (idx->d_vectors_rq8_words != NULL)
    {
        cudaFree(idx->d_vectors_rq8_words);
        idx->d_vectors_rq8_words = NULL;
    }
    if (idx->d_member_norms_sq != NULL)
    {
        cudaFree(idx->d_member_norms_sq);
        idx->d_member_norms_sq = NULL;
    }
    if (idx->d_cluster_graph_adj != NULL)
    {
        cudaFree(idx->d_cluster_graph_adj);
        idx->d_cluster_graph_adj = NULL;
    }
    if (idx->d_cluster_offsets != NULL)
    {
        cudaFree(idx->d_cluster_offsets);
        idx->d_cluster_offsets = NULL;
    }
    if (idx->d_cluster_sizes != NULL)
    {
        cudaFree(idx->d_cluster_sizes);
        idx->d_cluster_sizes = NULL;
    }
    if (idx->d_cluster_radii != NULL)
    {
        cudaFree(idx->d_cluster_radii);
        idx->d_cluster_radii = NULL;
    }
    if (idx->d_member_frame_ids != NULL)
    {
        cudaFree(idx->d_member_frame_ids);
        idx->d_member_frame_ids = NULL;
    }
    if (idx->d_member_r_anchors != NULL)
    {
        cudaFree(idx->d_member_r_anchors);
        idx->d_member_r_anchors = NULL;
    }
    if (idx->d_vectors_ivf != NULL)
    {
        cudaFree(idx->d_vectors_ivf);
        idx->d_vectors_ivf = NULL;
    }

    free(idx);
}

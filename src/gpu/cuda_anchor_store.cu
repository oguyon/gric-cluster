/**
 * @file cuda_anchor_store.cu
 * @brief Persistent GPU-resident cluster anchor store and fast nearest-anchor engine.
 */

#include "cuda_anchor_store.h"
#include "cuda_common.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do                                                                        \
    {                                                                         \
        cublasStatus_t st__ = (call);                                         \
        if (st__ != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",             \
                    __FILE__, __LINE__, (int)st__);                           \
            return -1;                                                        \
        }                                                                     \
    } while (0)

struct GpuAnchorStore
{
    int             max_clusters;
    int             dim;
    int             num_clusters;
    int             num_clusters_norms;
    int             max_batch_size;
    int             device_id;
    cublasHandle_t  cublas;

    float          *d_anchors;
    float          *d_anchor_norms;

    cudaStream_t    streams[2];
    float          *d_frames[2];
    float          *d_frame_norms[2];
    float          *d_P[2];
    int            *d_best_cl[2];
    float          *d_best_dist[2];

    float          *h_frames_pinned[2];
    int            *h_best_cl_pinned[2];
    float          *h_best_dist_pinned[2];
    float          *h_anchor_staging;

    int            *d_top_cl;
    float          *d_top_dist;

    float           rlim;
};


static __global__ void compute_norms_kernel(
    const float *__restrict__ mat,
    int                       count,
    int                       dim,
    float       *__restrict__ out_norms)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int vec_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (vec_idx < count)
    {
        const float *v = mat + (size_t)vec_idx * (size_t)dim;
        float sum = 0.0f;
        for (int d = lane; d < dim; d += 32)
        {
            float val = v[d];
            sum += val * val;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0)
        {
            out_norms[vec_idx] = sum;
        }
    }
}

static __global__ void argmin_distance_kernel(
    const float *__restrict__ d_F_norms,
    const float *__restrict__ d_A_norms,
    const float *__restrict__ d_P,
    int         *__restrict__ d_best_cl,
    float       *__restrict__ d_best_dist,
    int                       B,
    int                       K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int i = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (i < B)
    {
        float f_norm = d_F_norms[i];
        const float *p_row = d_P + (size_t)i * (size_t)K;

        float min_dist_sq = 1e38f;
        int   min_k = -1;

        for (int k = lane; k < K; k += 32)
        {
            float dist_sq = f_norm + d_A_norms[k] - 2.0f * p_row[k];
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                min_k = k;
            }
        } // for (int k = lane; k < K; k += 32)

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float other_dist = __shfl_down_sync(0xffffffff, min_dist_sq, offset);
            int   other_k = __shfl_down_sync(0xffffffff, min_k, offset);
            bool  take_other = (other_dist < min_dist_sq) ||
                (other_dist == min_dist_sq && other_k >= 0 &&
                 (min_k < 0 || other_k < min_k));
            if (take_other)
            {
                min_dist_sq = other_dist;
                min_k = other_k;
            }
        } // for (int offset = 16; offset > 0; offset /= 2)

        if (lane == 0)
        {
            d_best_cl[i] = min_k;
            d_best_dist[i] = sqrtf(min_dist_sq);
        }
    }
}

#define TOP_M_WARPS_PER_BLOCK 4

static __global__ void top_m_distance_kernel(
    const float *__restrict__ d_F_norms,
    const float *__restrict__ d_A_norms,
    const float *__restrict__ d_P,
    int         *__restrict__ d_top_cl,
    float       *__restrict__ d_top_dist,
    int                       B,
    int                       K,
    int                       m)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int i = blockIdx.x * (blockDim.x / 32) + warp_id;

    __shared__ float s_top_d[TOP_M_WARPS_PER_BLOCK][32];
    __shared__ int   s_top_c[TOP_M_WARPS_PER_BLOCK][32];

    int eff_m = (m < 32) ? m : 32;

    /* Initialize shared memory top-m lists */
    if (lane < 32)
    {
        s_top_d[warp_id][lane] = 1e38f;
        s_top_c[warp_id][lane] = -1;
    }
    __syncwarp();

    if (i < B)
    {
        float f_norm = d_F_norms[i];
        const float *p_row = d_P + (size_t)i * (size_t)K;

        for (int k_base = 0; k_base < K; k_base += 32)
        {
            int k = k_base + lane;
            float dist_sq = 1e38f;
            if (k < K)
            {
                float a_norm = d_A_norms[k];
                float p_val = p_row[k];
                dist_sq = f_norm + a_norm - 2.0f * p_val;
                if (dist_sq < 0.0f)
                {
                    dist_sq = 0.0f;
                }
            }

            float tau = s_top_d[warp_id][eff_m - 1];
            bool cand_valid = (k < K && dist_sq < tau);
            uint32_t mask = __ballot_sync(0xffffffff, cand_valid);

            while (mask != 0)
            {
                int winner_lane = __ffs(mask) - 1;
                float winner_dist = __shfl_sync(0xffffffff, dist_sq, winner_lane);
                int winner_k = k_base + winner_lane;

                if (lane == 0)
                {
                    if (winner_dist < s_top_d[warp_id][eff_m - 1])
                    {
                        int pos = eff_m - 1;
                        while (pos > 0 && s_top_d[warp_id][pos - 1] > winner_dist)
                        {
                            s_top_d[warp_id][pos] = s_top_d[warp_id][pos - 1];
                            s_top_c[warp_id][pos] = s_top_c[warp_id][pos - 1];
                            pos--;
                        }
                        s_top_d[warp_id][pos] = winner_dist;
                        s_top_c[warp_id][pos] = winner_k;
                    }
                } // if (lane == 0)

                mask &= mask - 1;
            } // while (mask != 0)
            __syncwarp();
        } // for (int k_base = 0; k_base < K; k_base += 32)

        int *out_c = d_top_cl + (size_t)i * (size_t)m;
        float *out_d = d_top_dist + (size_t)i * (size_t)m;

        for (int j = lane; j < eff_m; j += 32)
        {
            float d_val = s_top_d[warp_id][j];
            out_c[j] = s_top_c[warp_id][j];
            out_d[j] = (d_val < 1e37f) ? sqrtf(d_val) : 1e38f;
        }
    } // if (i < B)
}

GpuAnchorStore *gpu_anchor_store_create(
    const GpuAnchorStoreConfig *config)
{
    if (config == NULL || config->max_clusters <= 0 || config->dim <= 0)
    {
        return NULL;
    }

    if (gric_cuda_init(config->device_id) != 0)
    {
        return NULL;
    }

    GpuAnchorStore *store = (GpuAnchorStore *)calloc(1, sizeof(GpuAnchorStore));
    if (store == NULL)
    {
        return NULL;
    }

    store->max_clusters = config->max_clusters;
    store->dim = config->dim;
    store->num_clusters = 0;
    store->device_id = config->device_id;
    store->max_batch_size = (config->max_batch_size > 0) ? config->max_batch_size : 64;
    store->rlim = config->rlim;

    if (cublasCreate(&store->cublas) != CUBLAS_STATUS_SUCCESS)
    {
        free(store);
        return NULL;
    }
    cublasSetMathMode(store->cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    size_t anc_bytes = (size_t)store->max_clusters * (size_t)store->dim * sizeof(float);
    size_t anc_norms_bytes = (size_t)store->max_clusters * sizeof(float);
    size_t frames_bytes = (size_t)store->max_batch_size * (size_t)store->dim * sizeof(float);
    size_t frame_norms_bytes = (size_t)store->max_batch_size * sizeof(float);
    size_t p_bytes = (size_t)store->max_batch_size * (size_t)store->max_clusters * sizeof(float);
    size_t best_cl_bytes = (size_t)store->max_batch_size * sizeof(int);
    size_t best_dist_bytes = (size_t)store->max_batch_size * sizeof(float);

    if (cudaMalloc((void **)&store->d_anchors, anc_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_anchor_norms, anc_norms_bytes) != cudaSuccess ||
        cudaMallocHost((void **)&store->h_anchor_staging, (size_t)store->dim * sizeof(float))
            != cudaSuccess ||
        cudaMalloc((void **)&store->d_top_cl, (size_t)store->max_batch_size * 32 * sizeof(int))
            != cudaSuccess ||
        cudaMalloc((void **)&store->d_top_dist, (size_t)store->max_batch_size * 32 * sizeof(float))
            != cudaSuccess)
    {
        gpu_anchor_store_destroy(store);
        return NULL;
    }

    for (int s = 0; s < 2; s++)
    {
        if (cudaStreamCreate(&store->streams[s]) != cudaSuccess ||
            cudaMalloc((void **)&store->d_frames[s], frames_bytes) != cudaSuccess ||
            cudaMalloc((void **)&store->d_frame_norms[s], frame_norms_bytes) != cudaSuccess ||
            cudaMalloc((void **)&store->d_P[s], p_bytes) != cudaSuccess ||
            cudaMalloc((void **)&store->d_best_cl[s], best_cl_bytes) != cudaSuccess ||
            cudaMalloc((void **)&store->d_best_dist[s], best_dist_bytes) != cudaSuccess ||
            cudaMallocHost((void **)&store->h_frames_pinned[s], frames_bytes) != cudaSuccess ||
            cudaMallocHost((void **)&store->h_best_cl_pinned[s], best_cl_bytes) != cudaSuccess ||
            cudaMallocHost((void **)&store->h_best_dist_pinned[s], best_dist_bytes) != cudaSuccess)
        {
            gpu_anchor_store_destroy(store);
            return NULL;
        }
    }

    return store;
}

void gpu_anchor_store_destroy(
    GpuAnchorStore *store)
{
    if (store == NULL)
    {
        return;
    }

    if (store->d_anchors != NULL)
    {
        cudaFree(store->d_anchors);
    }
    if (store->d_anchor_norms != NULL)
    {
        cudaFree(store->d_anchor_norms);
    }
    if (store->h_anchor_staging != NULL)
    {
        cudaFreeHost(store->h_anchor_staging);
    }

    for (int s = 0; s < 2; s++)
    {
        if (store->d_frames[s] != NULL)
        {
            cudaFree(store->d_frames[s]);
        }
        if (store->d_frame_norms[s] != NULL)
        {
            cudaFree(store->d_frame_norms[s]);
        }
        if (store->d_P[s] != NULL)
        {
            cudaFree(store->d_P[s]);
        }
        if (store->d_best_cl[s] != NULL)
        {
            cudaFree(store->d_best_cl[s]);
        }
        if (store->d_best_dist[s] != NULL)
        {
            cudaFree(store->d_best_dist[s]);
        }
        if (store->h_frames_pinned[s] != NULL)
        {
            cudaFreeHost(store->h_frames_pinned[s]);
        }
        if (store->h_best_cl_pinned[s] != NULL)
        {
            cudaFreeHost(store->h_best_cl_pinned[s]);
        }
        if (store->h_best_dist_pinned[s] != NULL)
        {
            cudaFreeHost(store->h_best_dist_pinned[s]);
        }
        if (store->streams[s] != NULL)
        {
            cudaStreamDestroy(store->streams[s]);
        }
    }

    if (store->d_top_cl != NULL)
    {
        cudaFree(store->d_top_cl);
    }
    if (store->d_top_dist != NULL)
    {
        cudaFree(store->d_top_dist);
    }
    if (store->cublas != NULL)
    {
        cublasDestroy(store->cublas);
    }

    free(store);
}

void gpu_anchor_store_reset(
    GpuAnchorStore *store)
{
    if (store != NULL)
    {
        store->num_clusters = 0;
        store->num_clusters_norms = 0;
    }
}

void gpu_anchor_store_set_rlim(
    GpuAnchorStore *store,
    float           rlim)
{
    if (store != NULL)
    {
        store->rlim = rlim;
    }
}

int gpu_anchor_store_get_count(
    const GpuAnchorStore *store)
{
    return (store != NULL) ? store->num_clusters : 0;
}

float *gpu_anchor_store_get_pinned_frames(
    GpuAnchorStore *store,
    int             slot)
{
    if (store == NULL || slot < 0 || slot >= 2)
    {
        return NULL;
    }
    return store->h_frames_pinned[slot];
}

int gpu_anchor_store_append_anchor(
    GpuAnchorStore *store,
    const void     *data,
    int             is_double)
{
    if (store == NULL || data == NULL)
    {
        return -1;
    }

    if (store->num_clusters >= store->max_clusters)
    {
        fprintf(stderr, "GpuAnchorStore capacity reached (%d clusters)\n",
                store->max_clusters);
        return -1;
    }

    int idx = store->num_clusters;
    int D = store->dim;
    float *h_vec = store->h_anchor_staging;

    if (is_double)
    {
        const double *src = (const double *)data;
        for (int d = 0; d < D; d++)
        {
            h_vec[d] = (float)src[d];
        }
    }
    else
    {
        const float *src = (const float *)data;
        memcpy(h_vec, src, (size_t)D * sizeof(float));
    }

    float *d_dst = store->d_anchors + (size_t)idx * (size_t)D;
    CUDA_CHECK(cudaMemcpy(d_dst, h_vec, (size_t)D * sizeof(float), cudaMemcpyHostToDevice));

    store->num_clusters++;
    return idx;
}

int gpu_anchor_store_async_find_nearest(
    GpuAnchorStore *store,
    int             slot,
    int             batch_size)
{
    if (store == NULL || slot < 0 || slot >= 2 || batch_size <= 0)
    {
        return -1;
    }

    int K = store->num_clusters;
    int D = store->dim;
    int B = batch_size;

    if (K <= 0 || B > store->max_batch_size)
    {
        return -1;
    }

    /* Batch compute missing anchor norms on device */
    if (store->num_clusters > store->num_clusters_norms)
    {
        int new_k = store->num_clusters - store->num_clusters_norms;
        int threads = 128;
        int blocks = (new_k + (threads / 32) - 1) / (threads / 32);
        compute_norms_kernel<<<blocks, threads, 0, store->streams[slot]>>>(
            store->d_anchors + (size_t)store->num_clusters_norms * (size_t)D,
            new_k, D,
            store->d_anchor_norms + store->num_clusters_norms);
        CUDA_CHECK(cudaGetLastError());
        store->num_clusters_norms = store->num_clusters;
    }

    /* Asynchronous H2D transfer */
    CUDA_CHECK(cudaMemcpyAsync(
        store->d_frames[slot], store->h_frames_pinned[slot],
        (size_t)B * (size_t)D * sizeof(float),
        cudaMemcpyHostToDevice, store->streams[slot]));

    /* Compute frame norms */
    {
        int threads = 128;
        int blocks = (B + (threads / 32) - 1) / (threads / 32);
        compute_norms_kernel<<<blocks, threads, 0, store->streams[slot]>>>(
            store->d_frames[slot], B, D, store->d_frame_norms[slot]);
        CUDA_CHECK(cudaGetLastError());
    }

    /* GEMM: d_P[B x K] = d_frames[B x D] * (d_anchors[K x D])^T */
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(store->cublas, store->streams[slot]));
    CUBLAS_CHECK(cublasSgemm(store->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, D,
                             &alpha,
                             store->d_anchors, D,
                             store->d_frames[slot], D,
                             &beta,
                             store->d_P[slot], K));

    /* Argmin reduction kernel */
    {
        int r_threads = 128;
        int warps_per_block = r_threads / 32;
        int r_blocks = (B + warps_per_block - 1) / warps_per_block;
        argmin_distance_kernel<<<r_blocks, r_threads, 0, store->streams[slot]>>>(
            store->d_frame_norms[slot], store->d_anchor_norms, store->d_P[slot],
            store->d_best_cl[slot], store->d_best_dist[slot],
            B, K);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Asynchronous D2H transfer */
    CUDA_CHECK(cudaMemcpyAsync(
        store->h_best_cl_pinned[slot], store->d_best_cl[slot],
        (size_t)B * sizeof(int),
        cudaMemcpyDeviceToHost, store->streams[slot]));
    CUDA_CHECK(cudaMemcpyAsync(
        store->h_best_dist_pinned[slot], store->d_best_dist[slot],
        (size_t)B * sizeof(float),
        cudaMemcpyDeviceToHost, store->streams[slot]));

    return 0;
}

int gpu_anchor_store_sync_nearest(
    GpuAnchorStore  *store,
    int              slot,
    const int      **out_best_cl,
    const float    **out_best_dist)
{
    if (store == NULL || slot < 0 || slot >= 2)
    {
        return -1;
    }

    CUDA_CHECK(cudaStreamSynchronize(store->streams[slot]));

    if (out_best_cl != NULL)
    {
        *out_best_cl = store->h_best_cl_pinned[slot];
    }
    if (out_best_dist != NULL)
    {
        *out_best_dist = store->h_best_dist_pinned[slot];
    }

    return 0;
}

int gpu_anchor_store_find_nearest(
    GpuAnchorStore *store,
    const void     *host_frames,
    int             batch_size,
    int             is_double,
    int            *out_best_cl,
    float          *out_best_dist)
{
    if (store == NULL || out_best_cl == NULL || out_best_dist == NULL)
    {
        return -1;
    }

    if (batch_size <= 0 || batch_size > store->max_batch_size)
    {
        return -1;
    }

    if (store->num_clusters <= 0)
    {
        return -1;
    }

    /* If host_frames is provided and not already in slot 0 pinned buffer, copy it */
    if (host_frames != NULL && host_frames != store->h_frames_pinned[0])
    {
        float *dst = store->h_frames_pinned[0];
        int D = store->dim;
        if (is_double)
        {
            const double *src = (const double *)host_frames;
            size_t total = (size_t)batch_size * (size_t)D;
            for (size_t i = 0; i < total; i++)
            {
                dst[i] = (float)src[i];
            }
        }
        else
        {
            memcpy(dst, host_frames, (size_t)batch_size * (size_t)D * sizeof(float));
        }
    }

    if (gpu_anchor_store_async_find_nearest(store, 0, batch_size) != 0)
    {
        return -1;
    }

    const int *res_cl = NULL;
    const float *res_dist = NULL;
    if (gpu_anchor_store_sync_nearest(store, 0, &res_cl, &res_dist) != 0)
    {
        return -1;
    }

    if (out_best_cl != res_cl)
    {
        memcpy(out_best_cl, res_cl, (size_t)batch_size * sizeof(int));
    }
    if (out_best_dist != res_dist)
    {
        memcpy(out_best_dist, res_dist, (size_t)batch_size * sizeof(float));
    }

    return 0;
}

int gpu_anchor_store_find_top_m(
    GpuAnchorStore *store,
    const void     *host_queries,
    int             num_queries,
    int             is_double,
    int             m,
    int            *out_cl_indices,
    float          *out_cl_dists)
{
    if (store == NULL || host_queries == NULL || out_cl_indices == NULL || out_cl_dists == NULL)
    {
        return -1;
    }

    int K = store->num_clusters;
    int D = store->dim;
    int B = num_queries;

    if (K <= 0 || B <= 0 || m <= 0)
    {
        return -1;
    }

    if (m > 32)
    {
        m = 32;
    }
    if (m > K)
    {
        m = K;
    }

    if (B > store->max_batch_size)
    {
        fprintf(stderr, "GpuAnchorStore num_queries %d exceeds allocated batch size %d\n",
                B, store->max_batch_size);
        return -1;
    }

    /* Batch compute missing anchor norms on device */
    if (store->num_clusters > store->num_clusters_norms)
    {
        int new_k = store->num_clusters - store->num_clusters_norms;
        int threads = 128;
        int blocks = (new_k + (threads / 32) - 1) / (threads / 32);
        compute_norms_kernel<<<blocks, threads>>>(
            store->d_anchors + (size_t)store->num_clusters_norms * (size_t)D,
            new_k, D,
            store->d_anchor_norms + store->num_clusters_norms);
        CUDA_CHECK(cudaGetLastError());
        store->num_clusters_norms = store->num_clusters;
    }

    float *h_f = store->h_frames_pinned[0];
    if (is_double)
    {
        const double *src = (const double *)host_queries;
        size_t total = (size_t)B * (size_t)D;
        for (size_t i = 0; i < total; i++)
        {
            h_f[i] = (float)src[i];
        }
    }
    else
    {
        memcpy(h_f, host_queries, (size_t)B * (size_t)D * sizeof(float));
    }

    CUDA_CHECK(cudaMemcpy(store->d_frames[0], h_f, (size_t)B * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int blocks = (B + (threads / 32) - 1) / (threads / 32);
        compute_norms_kernel<<<blocks, threads>>>(store->d_frames[0], B, D,
                                                  store->d_frame_norms[0]);
        CUDA_CHECK(cudaGetLastError());
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(store->cublas, NULL));
    CUBLAS_CHECK(cublasSgemm(store->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, D,
                             &alpha,
                             store->d_anchors, D,
                             store->d_frames[0], D,
                             &beta,
                             store->d_P[0], K));

    {
        int threads = 128;
        int warps_per_block = threads / 32;
        int blocks = (B + warps_per_block - 1) / warps_per_block;
        top_m_distance_kernel<<<blocks, threads>>>(
            store->d_frame_norms[0], store->d_anchor_norms, store->d_P[0],
            store->d_top_cl, store->d_top_dist,
            B, K, m);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(out_cl_indices, store->d_top_cl, (size_t)B * (size_t)m * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_cl_dists, store->d_top_dist, (size_t)B * (size_t)m * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return 0;
}

int gpu_anchor_store_get_anchors_host(
    const GpuAnchorStore *store,
    float                *out_host_anchors,
    int                   max_clusters)
{
    if (store == NULL || out_host_anchors == NULL)
    {
        return -1;
    }

    int count = (store->num_clusters < max_clusters) ? store->num_clusters : max_clusters;
    if (count <= 0)
    {
        return 0;
    }

    size_t bytes = (size_t)count * (size_t)store->dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(out_host_anchors, store->d_anchors, bytes, cudaMemcpyDeviceToHost));

    return count;
}

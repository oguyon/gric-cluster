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
    int             max_batch_size;
    int             device_id;
    cublasHandle_t  cublas;

    float          *d_anchors;
    float          *d_anchor_norms;
    float          *d_frames;
    float          *d_frame_norms;
    float          *d_P;
    int            *d_best_cl;
    float          *d_best_dist;

    int            *d_top_cl;
    float          *d_top_dist;

    float          *h_frames_float;
};

static __global__ void compute_norms_kernel(
    const float *__restrict__ mat,
    int                       count,
    int                       dim,
    float       *__restrict__ out_norms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        const float *v = mat + (size_t)idx * (size_t)dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            float val = v[d];
            sum += val * val;
        }
        out_norms[idx] = sum;
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B)
    {
        float f_norm = d_F_norms[i];
        const float *p_row = d_P + (size_t)i * (size_t)K;

        float min_dist_sq = 1e38f;
        int min_k = -1;

        for (int k = 0; k < K; k++)
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
        }

        d_best_cl[i] = min_k;
        d_best_dist[i] = sqrtf(min_dist_sq);
    }
}

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B)
    {
        float f_norm = d_F_norms[i];
        const float *p_row = d_P + (size_t)i * (size_t)K;

        int *out_c = d_top_cl + (size_t)i * (size_t)m;
        float *out_d = d_top_dist + (size_t)i * (size_t)m;

        /* Local register arrays for top-m tracking */
        float local_d[32];
        int   local_c[32];
        int eff_m = (m < 32) ? m : 32;

        for (int j = 0; j < eff_m; j++)
        {
            local_d[j] = 1e38f;
            local_c[j] = -1;
        }

        for (int k = 0; k < K; k++)
        {
            float dist_sq = f_norm + d_A_norms[k] - 2.0f * p_row[k];
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }

            if (dist_sq < local_d[eff_m - 1])
            {
                int pos = eff_m - 1;
                while (pos > 0 && dist_sq < local_d[pos - 1])
                {
                    local_d[pos] = local_d[pos - 1];
                    local_c[pos] = local_c[pos - 1];
                    pos--;
                }
                local_d[pos] = dist_sq;
                local_c[pos] = k;
            }
        }

        for (int j = 0; j < eff_m; j++)
        {
            out_c[j] = local_c[j];
            out_d[j] = (local_d[j] < 1e37f) ? sqrtf(local_d[j]) : 1e38f;
        }
    }
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
        cudaMalloc((void **)&store->d_frames, frames_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_frame_norms, frame_norms_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_P, p_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_best_cl, best_cl_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_best_dist, best_dist_bytes) != cudaSuccess ||
        cudaMalloc((void **)&store->d_top_cl, (size_t)store->max_batch_size * 32 * sizeof(int))
            != cudaSuccess ||
        cudaMalloc((void **)&store->d_top_dist, (size_t)store->max_batch_size * 32 * sizeof(float))
            != cudaSuccess)
    {
        gpu_anchor_store_destroy(store);
        return NULL;
    }

    store->h_frames_float = (float *)malloc(frames_bytes);
    if (store->h_frames_float == NULL)
    {
        gpu_anchor_store_destroy(store);
        return NULL;
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
    if (store->d_frames != NULL)
    {
        cudaFree(store->d_frames);
    }
    if (store->d_frame_norms != NULL)
    {
        cudaFree(store->d_frame_norms);
    }
    if (store->d_P != NULL)
    {
        cudaFree(store->d_P);
    }
    if (store->d_best_cl != NULL)
    {
        cudaFree(store->d_best_cl);
    }
    if (store->d_best_dist != NULL)
    {
        cudaFree(store->d_best_dist);
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
    if (store->h_frames_float != NULL)
    {
        free(store->h_frames_float);
    }

    free(store);
}

void gpu_anchor_store_reset(
    GpuAnchorStore *store)
{
    if (store != NULL)
    {
        store->num_clusters = 0;
    }
}

int gpu_anchor_store_get_count(
    const GpuAnchorStore *store)
{
    return (store != NULL) ? store->num_clusters : 0;
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
        fprintf(stderr, "GpuAnchorStore capacity reached (%d clusters)\n", store->max_clusters);
        return -1;
    }

    int idx = store->num_clusters;
    int D = store->dim;
    float *h_vec = store->h_frames_float;

    float norm_sq = 0.0f;
    if (is_double)
    {
        const double *src = (const double *)data;
        for (int d = 0; d < D; d++)
        {
            float val = (float)src[d];
            h_vec[d] = val;
            norm_sq += val * val;
        }
    }
    else
    {
        const float *src = (const float *)data;
        for (int d = 0; d < D; d++)
        {
            float val = src[d];
            h_vec[d] = val;
            norm_sq += val * val;
        }
    }

    float *d_dst = store->d_anchors + (size_t)idx * (size_t)D;
    CUDA_CHECK(cudaMemcpy(d_dst, h_vec, (size_t)D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(store->d_anchor_norms + idx, &norm_sq, sizeof(float),
                          cudaMemcpyHostToDevice));

    store->num_clusters++;
    return idx;
}

int gpu_anchor_store_find_nearest(
    GpuAnchorStore *store,
    const void     *host_frames,
    int             batch_size,
    int             is_double,
    int            *out_best_cl,
    float          *out_best_dist)
{
    if (store == NULL || host_frames == NULL || out_best_cl == NULL || out_best_dist == NULL)
    {
        return -1;
    }

    int K = store->num_clusters;
    int D = store->dim;
    int B = batch_size;

    if (K <= 0 || B <= 0)
    {
        return -1;
    }

    if (B > store->max_batch_size)
    {
        fprintf(stderr, "GpuAnchorStore batch size %d exceeds allocated maximum %d\n",
                B, store->max_batch_size);
        return -1;
    }

    /* Prepare host float buffer */
    float *h_f = store->h_frames_float;
    if (is_double)
    {
        const double *src = (const double *)host_frames;
        size_t total = (size_t)B * (size_t)D;
        for (size_t i = 0; i < total; i++)
        {
            h_f[i] = (float)src[i];
        }
    }
    else
    {
        memcpy(h_f, host_frames, (size_t)B * (size_t)D * sizeof(float));
    }

    CUDA_CHECK(cudaMemcpy(store->d_frames, h_f, (size_t)B * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Compute frame norms */
    {
        int threads = 128;
        int blocks = (B + threads - 1) / threads;
        compute_norms_kernel<<<blocks, threads>>>(store->d_frames, B, D, store->d_frame_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    /* GEMM: d_P[B x K] = d_frames[B x D] * (d_anchors[K x D])^T */
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(store->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, D,
                             &alpha,
                             store->d_anchors, D,
                             store->d_frames, D,
                             &beta,
                             store->d_P, K));

    /* Argmin reduction kernel */
    {
        int threads = 128;
        int blocks = (B + threads - 1) / threads;
        argmin_distance_kernel<<<blocks, threads>>>(
            store->d_frame_norms, store->d_anchor_norms, store->d_P,
            store->d_best_cl, store->d_best_dist,
            B, K);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(out_best_cl, store->d_best_cl, (size_t)B * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_best_dist, store->d_best_dist, (size_t)B * sizeof(float),
                          cudaMemcpyDeviceToHost));

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

    float *h_f = store->h_frames_float;
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

    CUDA_CHECK(cudaMemcpy(store->d_frames, h_f, (size_t)B * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int blocks = (B + threads - 1) / threads;
        compute_norms_kernel<<<blocks, threads>>>(store->d_frames, B, D, store->d_frame_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(store->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K, B, D,
                             &alpha,
                             store->d_anchors, D,
                             store->d_frames, D,
                             &beta,
                             store->d_P, K));

    {
        int threads = 128;
        int blocks = (B + threads - 1) / threads;
        top_m_distance_kernel<<<blocks, threads>>>(
            store->d_frame_norms, store->d_anchor_norms, store->d_P,
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

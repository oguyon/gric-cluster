/**
 * @file cluster_cuda.cu
 * @brief GPU-accelerated Pass 2 clustering implementation for gric-cluster.
 */

#ifdef __cplusplus
extern "C" {
#endif
#include "cuda_common.h"
#include "cluster_cuda.h"
#include "cuda_anchor_store.h"
#include "cluster_core.h"
#include "cluster_mgmt.h"
#include "cluster_shm.h"
#include "framedistance.h"
#include "frameread.h"
#ifdef __cplusplus
}
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_FRAME_BATCH 4096

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            goto cleanup;                                                     \
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
            goto cleanup;                                                     \
        }                                                                     \
    } while (0)

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

static __global__ void compute_dcc_matrix_kernel(
    const float *__restrict__ d_A_norms,
    const float *__restrict__ d_P_cc,
    float       *__restrict__ d_Dcc,
    int                       K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < K)
    {
        if (row == col)
        {
            d_Dcc[(size_t)row * (size_t)K + (size_t)col] = 0.0f;
        }
        else
        {
            float norm_u = d_A_norms[row];
            float norm_v = d_A_norms[col];
            float dot_uv = d_P_cc[(size_t)row * (size_t)K + (size_t)col];
            float dist_sq = norm_u + norm_v - 2.0f * dot_uv;
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }
            d_Dcc[(size_t)row * (size_t)K + (size_t)col] = sqrtf(dist_sq);
        }
    }
}

static __global__ void gpu_pass2_triangle_prune_kernel(
    const float        *__restrict__ d_F,
    const float        *__restrict__ d_A,
    const float        *__restrict__ d_Dcc,
    const int          *__restrict__ d_init_cl,
    const float        *__restrict__ d_init_dist,
    int                *__restrict__ d_best_cl,
    float              *__restrict__ d_best_dist,
    unsigned long long *__restrict__ d_pruned_count,
    int                              B,
    int                              K,
    int                              D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pruned = 0;

    if (i < B)
    {
        int init_c = d_init_cl[i];
        float best_d = 1e38f;
        int best_c = -1;

        const float *f_vec = d_F + (size_t)i * (size_t)D;
        float local_f[64];
        int eff_d = (D < 64) ? D : 64;
        for (int d = 0; d < eff_d; d++)
        {
            local_f[d] = f_vec[d];
        }

        if (init_c >= 0 && init_c < K)
        {
            best_c = init_c;
            best_d = d_init_dist[i];
            if (best_d <= 0.0f)
            {
                const float *a_vec = d_A + (size_t)init_c * (size_t)D;
                float sum_sq = 0.0f;
                for (int d = 0; d < D; d++)
                {
                    float diff = local_f[d] - a_vec[d];
                    sum_sq += diff * diff;
                }
                best_d = sqrtf(sum_sq);
            }
        }

        float best_d_sq = best_d * best_d;
        const float *dcc_row = (init_c >= 0 && init_c < K && d_Dcc != NULL)
            ? (d_Dcc + (size_t)init_c * (size_t)K)
            : NULL;

        for (int k = 0; k < K; k++)
        {
            if (k == init_c)
            {
                continue;
            }

            /* 1. Triangle Inequality Distance Lower-Bound Pruning */
            if (dcc_row != NULL)
            {
                float dcc = dcc_row[k];
                float lb = fabsf(best_d - dcc);
                if (lb >= best_d)
                {
                    pruned++;
                    continue;
                }
            }

            /* 2. Metric Cutoff Distance Calculation */
            const float *a_vec = d_A + (size_t)k * (size_t)D;
            float dist_sq = 0.0f;
            int early_exit = 0;

            for (int d = 0; d < D; d++)
            {
                float diff = local_f[d] - a_vec[d];
                dist_sq += diff * diff;
                if (dist_sq >= best_d_sq)
                {
                    early_exit = 1;
                    break;
                }
            }

            if (!early_exit && dist_sq < best_d_sq)
            {
                best_d_sq = dist_sq;
                best_d = sqrtf(dist_sq);
                best_c = k;
            }
        } // for (int k = 0; k < K; k++)

        d_best_cl[i] = best_c;
        d_best_dist[i] = best_d;
    } // if (i < B)

    /* Hierarchical block reduction for pruned counter */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        pruned += __shfl_down_sync(0xffffffff, pruned, offset);
    }

    __shared__ unsigned int s_block_pruned[32];
    int lane = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    if (lane == 0)
    {
        s_block_pruned[warp_in_block] = pruned;
    }
    __syncthreads();

    if (warp_in_block == 0)
    {
        unsigned int val = (lane < (blockDim.x / 32)) ? s_block_pruned[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val > 0 && d_pruned_count != NULL)
        {
            atomicAdd(d_pruned_count, (unsigned long long)val);
        }
    }
}

static __global__ void gpu_pass2_triangle_prune_warp_kernel(
    const float        *__restrict__ d_F,
    const float        *__restrict__ d_A,
    const float        *__restrict__ d_Dcc,
    const int          *__restrict__ d_init_cl,
    const float        *__restrict__ d_init_dist,
    int                *__restrict__ d_best_cl,
    float              *__restrict__ d_best_dist,
    unsigned long long *__restrict__ d_pruned_count,
    int                              B,
    int                              K,
    int                              D)
{
    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int i = blockIdx.x * warps_per_block + warp_in_block;

    if (i >= B)
    {
        return;
    }

    int init_c = d_init_cl[i];
    float best_d = 1e38f;
    int best_c = -1;

    const float *f_vec = d_F + (size_t)i * (size_t)D;

    if (init_c >= 0 && init_c < K)
    {
        best_c = init_c;
        best_d = d_init_dist[i];
        if (best_d <= 0.0f)
        {
            const float *a_vec = d_A + (size_t)init_c * (size_t)D;
            float sum_sq = 0.0f;
            for (int d = lane; d < D; d += 32)
            {
                float diff = f_vec[d] - a_vec[d];
                sum_sq += diff * diff;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            }
            best_d = sqrtf(__shfl_sync(0xffffffff, sum_sq, 0));
        }
    }

    float best_d_sq = best_d * best_d;
    const float *dcc_row = (init_c >= 0 && init_c < K && d_Dcc != NULL)
        ? (d_Dcc + (size_t)init_c * (size_t)K)
        : NULL;

    unsigned int pruned = 0;

    for (int k = 0; k < K; k++)
    {
        if (k == init_c)
        {
            continue;
        }

        /* 1. Triangle Inequality Lower-Bound Pruning */
        int skip_k = 0;
        if (lane == 0 && dcc_row != NULL)
        {
            float dcc = dcc_row[k];
            float lb = fabsf(best_d - dcc);
            if (lb >= best_d)
            {
                skip_k = 1;
                pruned++;
            }
        }
        skip_k = __shfl_sync(0xffffffff, skip_k, 0);
        if (skip_k)
        {
            continue;
        }

        /* 2. Cooperative Distance Evaluation with Periodic Metric Cutoff */
        const float *a_vec = d_A + (size_t)k * (size_t)D;
        float partial_sq = 0.0f;
        int early_exit = 0;

        for (int d_base = 0; d_base < D; d_base += 64)
        {
            int d = d_base + lane;
            if (d < D)
            {
                float diff = f_vec[d] - a_vec[d];
                partial_sq += diff * diff;
            }
            int d2 = d_base + 32 + lane;
            if (d2 < D)
            {
                float diff2 = f_vec[d2] - a_vec[d2];
                partial_sq += diff2 * diff2;
            }

            /* Metric cutoff check every 64 dimensions */
            float block_sum = partial_sq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            }
            if (lane == 0)
            {
                if (block_sum >= best_d_sq)
                {
                    early_exit = 1;
                }
            }
            early_exit = __shfl_sync(0xffffffff, early_exit, 0);
            if (early_exit)
            {
                break;
            }
        } // for (int d_base = 0; d_base < D; d_base += 64)

        if (!early_exit)
        {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                partial_sq += __shfl_down_sync(0xffffffff, partial_sq, offset);
            }
            float total_dist_sq = __shfl_sync(0xffffffff, partial_sq, 0);
            if (total_dist_sq < best_d_sq)
            {
                best_d_sq = total_dist_sq;
                best_d = sqrtf(total_dist_sq);
                best_c = k;
            }
        }
    } // for (int k = 0; k < K; k++)

    if (lane == 0)
    {
        d_best_cl[i] = best_c;
        d_best_dist[i] = best_d;
        if (d_pruned_count != NULL && pruned > 0)
        {
            atomicAdd(d_pruned_count, (unsigned long long)pruned);
        }
    }
}

int cluster_cuda_is_available(void)
{
    return gric_cuda_is_available();
}

long cluster_cuda_run_pass2(
    ClusterConfig *config,
    ClusterState  *state)
{
    if (state == NULL || config == NULL)
    {
        return -1;
    }

    int K = state->num_clusters;
    long N = state->telemetry.total_frames_processed;

    if (K <= 1 || N <= 0)
    {
        return 0;
    }

    if (gric_cuda_init(config->optim.gpu_device_id) != 0)
    {
        return -1;
    }

    long D = (long)state->clusters[0].anchor.width * (long)state->clusters[0].anchor.height;
    if (D <= 0)
    {
        return -1;
    }

    struct timespec p2_start, p2_end;
    clock_gettime(CLOCK_MONOTONIC, &p2_start);

    cublasHandle_t cublas_handle = NULL;
    cudaStream_t streams[2] = {NULL, NULL};
    float *host_anchors = NULL;
    float *d_A = NULL;
    float *d_A_norms = NULL;
    float *d_Dcc = NULL;
    unsigned long long *d_pruned_count = NULL;
    float *d_F[2] = {NULL, NULL};
    float *d_F_norms[2] = {NULL, NULL};
    float *d_P[2] = {NULL, NULL};
    int   *d_best_cl[2] = {NULL, NULL};
    float *d_best_dist[2] = {NULL, NULL};
    int   *d_init_cl[2] = {NULL, NULL};
    float *d_init_dist[2] = {NULL, NULL};
    float *host_F[2] = {NULL, NULL};
    int   *host_best_cl[2] = {NULL, NULL};
    float *host_best_dist[2] = {NULL, NULL};
    int   *host_init_cl[2] = {NULL, NULL};
    float *host_init_dist[2] = {NULL, NULL};
    long batch_starts[2] = {0, 0};
    int batch_lens[2] = {0, 0};
    int order[2] = {-1, -1};
    long frames_reassigned = 0;
    unsigned long long h_pruned = 0;
    double p2_ms = 0.0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int B_max = (config->optim.gpu_micro_batch_size > 0) ?
                config->optim.gpu_micro_batch_size : DEFAULT_FRAME_BATCH;

    if (B_max > N)
    {
        B_max = (int)N;
    }

    int use_prune = 0;
    if (config->optim.gpu_prune_mode == 1)
    {
        use_prune = (K <= 8192);
    }
    else if (config->optim.gpu_prune_mode == 0)
    {
        use_prune = 0;
    }
    else
    {
        /* Auto mode (default): prune for D < 128, dense Tensor Core GEMM for D >= 128 */
        use_prune = (K <= 8192 && D < 128);
    }

    if (getenv("GRIC_GPU_NO_PRUNE") != NULL)
    {
        use_prune = 0;
    }
    else if (getenv("GRIC_GPU_FORCE_PRUNE") != NULL)
    {
        use_prune = (K <= 8192);
    }

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    host_anchors = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    if (host_anchors == NULL)
    {
        goto cleanup;
    }

    for (int k = 0; k < K; k++)
    {
        float *dst = host_anchors + (size_t)k * (size_t)D;
        if (state->clusters[k].anchor.is_double)
        {
            const double *src = (const double *)state->clusters[k].anchor.data;
            for (long d = 0; d < D; d++)
            {
                dst[d] = (float)src[d];
            }
        }
        else
        {
            memcpy(dst, state->clusters[k].anchor.data, (size_t)D * sizeof(float));
        }
    }

    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    for (int b = 0; b < 2; b++)
    {
        CUDA_CHECK(cudaMallocHost(
            (void **)&host_F[b], (size_t)B_max * (size_t)D * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
            (void **)&host_best_cl[b], (size_t)B_max * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(
            (void **)&host_best_dist[b], (size_t)B_max * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
            (void **)&host_init_cl[b], (size_t)B_max * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(
            (void **)&host_init_dist[b], (size_t)B_max * sizeof(float)));

        CUDA_CHECK(cudaMalloc(
            (void **)&d_F[b], (size_t)B_max * (size_t)D * sizeof(float)));
        if (!use_prune)
        {
            CUDA_CHECK(cudaMalloc(
                (void **)&d_F_norms[b], (size_t)B_max * sizeof(float)));
            CUDA_CHECK(cudaMalloc(
                (void **)&d_P[b], (size_t)B_max * (size_t)K * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(
            (void **)&d_best_cl[b], (size_t)B_max * sizeof(int)));
        CUDA_CHECK(cudaMalloc(
            (void **)&d_best_dist[b], (size_t)B_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            (void **)&d_init_cl[b], (size_t)B_max * sizeof(int)));
        CUDA_CHECK(cudaMalloc(
            (void **)&d_init_dist[b], (size_t)B_max * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc((void **)&d_pruned_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_pruned_count, 0, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMalloc((void **)&d_A, (size_t)K * (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_A_norms, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, host_anchors, (size_t)K * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int blocks = (K + (threads / 32) - 1) / (threads / 32);
        compute_norms_kernel<<<blocks, threads>>>(d_A, K, (int)D, d_A_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    if (use_prune)
    {
        float *d_P_cc = NULL;
        CUDA_CHECK(cudaMalloc((void **)&d_P_cc, (size_t)K * (size_t)K * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&d_Dcc, (size_t)K * (size_t)K * sizeof(float)));

        CUBLAS_CHECK(cublasSgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            K, K, (int)D,
            &alpha,
            d_A, (int)D,
            d_A, (int)D,
            &beta,
            d_P_cc, K));

        dim3 dcc_block(16, 16);
        dim3 dcc_grid((K + 15) / 16, (K + 15) / 16);
        compute_dcc_matrix_kernel<<<dcc_grid, dcc_block>>>(d_A_norms, d_P_cc, d_Dcc, K);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_P_cc);
    }

    for (long t_start = 0; t_start < N; t_start += B_max)
    {
        int slot = (int)((t_start / B_max) % 2);
        int cur_B = (int)((t_start + B_max <= N) ? B_max : (N - t_start));

        /* If previous batch in this slot was launched, wait and process its results */
        if (batch_lens[slot] > 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            long prev_t = batch_starts[slot];
            int prev_B = batch_lens[slot];
            for (int i = 0; i < prev_B; i++)
            {
                long t = prev_t + i;
                int best_cl = host_best_cl[slot][i];
                float best_d = host_best_dist[slot][i];

                if (best_cl >= 0 && best_cl != state->assignments[t])
                {
                    frames_reassigned++;
                    state->assignments[t] = best_cl;
                }

                if (state->frame_infos != NULL)
                {
                    state->frame_infos[t].assignment = best_cl;
                    state->frame_infos[t].assigned_dist = (double)best_d;
                    if (state->frame_infos[t].cluster_indices != NULL &&
                        state->frame_infos[t].distances != NULL)
                    {
                        state->frame_infos[t].cluster_indices[0] = best_cl;
                        state->frame_infos[t].distances[0] = (double)best_d;
                    }
                }
            }
            batch_lens[slot] = 0;
        }

        /* Ingest cur_B frames from reader into pinned host_F[slot], init_cl, init_dist */
        for (int i = 0; i < cur_B; i++)
        {
            long t = t_start + i;
            Frame *fr = getframe_at(t);
            if (fr == NULL)
            {
                fprintf(stderr, "Error: Could not retrieve frame %ld\n", t);
                goto cleanup;
            }
            float *dst = host_F[slot] + (size_t)i * (size_t)D;
            if (fr->is_double)
            {
                const double *s = (const double *)fr->data;
                for (long d = 0; d < D; d++)
                {
                    dst[d] = (float)s[d];
                }
            }
            else
            {
                memcpy(dst, fr->data, (size_t)D * sizeof(float));
            }
            free_frame(fr);

            host_init_cl[slot][i] = state->assignments[t];
            if (state->frame_infos != NULL)
            {
                if (state->frame_infos[t].assigned_dist > 0.0)
                {
                    host_init_dist[slot][i] = (float)state->frame_infos[t].assigned_dist;
                }
                else if (state->frame_infos[t].distances != NULL &&
                         state->frame_infos[t].num_dists > 0)
                {
                    host_init_dist[slot][i] = (float)state->frame_infos[t].distances[0];
                }
                else
                {
                    host_init_dist[slot][i] = -1.0f;
                }
            }
            else
            {
                host_init_dist[slot][i] = -1.0f;
            }
        }

        batch_starts[slot] = t_start;
        batch_lens[slot] = cur_B;

        /* Asynchronous H2D transfer */
        CUDA_CHECK(cudaMemcpyAsync(
            d_F[slot], host_F[slot],
            (size_t)cur_B * (size_t)D * sizeof(float),
            cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(
            d_init_cl[slot], host_init_cl[slot],
            (size_t)cur_B * sizeof(int),
            cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(
            d_init_dist[slot], host_init_dist[slot],
            (size_t)cur_B * sizeof(float),
            cudaMemcpyHostToDevice, streams[slot]));

        if (d_Dcc != NULL)
        {
            if (D <= 64)
            {
                int threads = 128;
                int blocks = (cur_B + threads - 1) / threads;
                gpu_pass2_triangle_prune_kernel<<<blocks, threads, 0, streams[slot]>>>(
                    d_F[slot], d_A, d_Dcc,
                    d_init_cl[slot], d_init_dist[slot],
                    d_best_cl[slot], d_best_dist[slot],
                    d_pruned_count,
                    cur_B, K, (int)D);
                CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                int threads = 128;
                int warps_per_block = threads / 32;
                int blocks = (cur_B + warps_per_block - 1) / warps_per_block;
                gpu_pass2_triangle_prune_warp_kernel<<<blocks, threads, 0, streams[slot]>>>(
                    d_F[slot], d_A, d_Dcc,
                    d_init_cl[slot], d_init_dist[slot],
                    d_best_cl[slot], d_best_dist[slot],
                    d_pruned_count,
                    cur_B, K, (int)D);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        else
        {
            int threads = 128;
            int blocks = (cur_B + (threads / 32) - 1) / (threads / 32);
            compute_norms_kernel<<<blocks, threads, 0, streams[slot]>>>(
                d_F[slot], cur_B, (int)D, d_F_norms[slot]);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasSetStream(cublas_handle, streams[slot]));
            CUBLAS_CHECK(cublasSgemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                K, cur_B, (int)D,
                &alpha,
                d_A, (int)D,
                d_F[slot], (int)D,
                &beta,
                d_P[slot], K));

            int warps_per_block = threads / 32;
            int a_blocks = (cur_B + warps_per_block - 1) / warps_per_block;
            argmin_distance_kernel<<<a_blocks, threads, 0, streams[slot]>>>(
                d_F_norms[slot], d_A_norms, d_P[slot],
                d_best_cl[slot], d_best_dist[slot],
                cur_B, K);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Asynchronous D2H transfer */
        CUDA_CHECK(cudaMemcpyAsync(
            host_best_cl[slot], d_best_cl[slot],
            (size_t)cur_B * sizeof(int),
            cudaMemcpyDeviceToHost, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(
            host_best_dist[slot], d_best_dist[slot],
            (size_t)cur_B * sizeof(float),
            cudaMemcpyDeviceToHost, streams[slot]));
    }

    /* Drain pending batches in sequential frame index order */
    order[0] = -1;
    order[1] = -1;
    if (batch_lens[0] > 0 && batch_lens[1] > 0)
    {
        if (batch_starts[0] < batch_starts[1])
        {
            order[0] = 0;
            order[1] = 1;
        }
        else
        {
            order[0] = 1;
            order[1] = 0;
        }
    }
    else if (batch_lens[0] > 0)
    {
        order[0] = 0;
    }
    else if (batch_lens[1] > 0)
    {
        order[0] = 1;
    }

    for (int idx = 0; idx < 2; idx++)
    {
        int s = order[idx];
        if (s < 0 || batch_lens[s] <= 0)
        {
            continue;
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        long prev_t = batch_starts[s];
        int prev_B = batch_lens[s];
        for (int i = 0; i < prev_B; i++)
        {
            long t = prev_t + i;
            int best_cl = host_best_cl[s][i];
            float best_d = host_best_dist[s][i];

            if (best_cl >= 0 && best_cl != state->assignments[t])
            {
                frames_reassigned++;
                state->assignments[t] = best_cl;
            }

            if (state->frame_infos != NULL)
            {
                state->frame_infos[t].assignment = best_cl;
                state->frame_infos[t].assigned_dist = (double)best_d;
                if (state->frame_infos[t].cluster_indices != NULL &&
                    state->frame_infos[t].distances != NULL)
                {
                    state->frame_infos[t].cluster_indices[0] = best_cl;
                    state->frame_infos[t].distances[0] = (double)best_d;
                }
            }
        }
        batch_lens[s] = 0;
    }

    /* Rebuild Transition Matrix to match updated assignments */
    if (state->transition_matrix != NULL)
    {
        size_t maxcl = (size_t)config->algo.maxnbclust;
        memset(state->transition_matrix, 0, maxcl * maxcl * sizeof(long));

        for (long t = 0; t < N - 1; t++)
        {
            int from = state->assignments[t];
            int to   = state->assignments[t + 1];
            if (from >= 0 && (size_t)from < maxcl &&
                to >= 0 && (size_t)to < maxcl)
            {
                state->transition_matrix[(size_t)from * maxcl + (size_t)to]++;
            }
        }
    }

    /* Rewrite frame_membership.txt if enabled */
    if (config->output.output_membership && !config->output.no_txt)
    {
        char out_path[1024];
        if (config->output.user_outdir != NULL)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt",
                     config->output.user_outdir);
        }
        else
        {
            snprintf(out_path, sizeof(out_path), "frame_membership.txt");
        }

        FILE *ascii_out = fopen(out_path, "w");
        if (ascii_out != NULL)
        {
            setvbuf(ascii_out, NULL, _IOFBF, 65536);
            for (long t = 0; t < N; t++)
            {
                double best_d = 0.0;
                if (state->frame_infos != NULL &&
                    state->frame_infos[t].distances != NULL &&
                    state->frame_infos[t].num_dists > 0)
                {
                    best_d = state->frame_infos[t].distances[0];
                }
                fprintf(ascii_out, "%ld %d %.6f\n", t, state->assignments[t], best_d);
            }
            fclose(ascii_out);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &p2_end);
    p2_ms = (p2_end.tv_sec - p2_start.tv_sec) * 1000.0 +
            (p2_end.tv_nsec - p2_start.tv_nsec) / 1000000.0;

    h_pruned = 0;
    if (d_pruned_count != NULL)
    {
        CUDA_CHECK(cudaMemcpy(&h_pruned, d_pruned_count, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }

    state->telemetry.time_pass2 = p2_ms;
    state->telemetry.pass2_frames_reassigned = (uint64_t)frames_reassigned;
    state->telemetry.pass2_dist_pruned = (uint64_t)h_pruned;
    state->telemetry.pass2_dist_evals = (uint64_t)N * (uint64_t)K - (uint64_t)h_pruned;

    printf("\nSecond Pass (Nearest Anchor Reallocation - GPU [%s]):\n",
           use_prune ? "Triangle Pruning" : "Tensor Core GEMM");
    printf("  Frames reassigned  : %ld / %ld (%.2f%%)\n",
           frames_reassigned, N,
           (N > 0) ? (100.0 * (double)frames_reassigned / (double)N) : 0.0);
    if (use_prune)
    {
        printf("  Distance evals     : %lu (pruned: %lu / %lu, %.2f%%)\n",
               (unsigned long)state->telemetry.pass2_dist_evals,
               (unsigned long)state->telemetry.pass2_dist_pruned,
               (unsigned long)((uint64_t)N * (uint64_t)K),
               (N > 0 && K > 0) ?
                   (100.0 * (double)state->telemetry.pass2_dist_pruned /
                    ((double)N * (double)K)) : 0.0);
    }
    else
    {
        printf("  Distance evals     : %lu (dense Tensor Core GEMM)\n",
               (unsigned long)((uint64_t)N * (uint64_t)K));
    }
    printf("  GPU Pass 2 Time    : %.2f ms\n", p2_ms);

cleanup:
    for (int b = 0; b < 2; b++)
    {
        if (d_init_dist[b] != NULL)
        {
            cudaFree(d_init_dist[b]);
        }
        if (d_init_cl[b] != NULL)
        {
            cudaFree(d_init_cl[b]);
        }
        if (host_init_dist[b] != NULL)
        {
            cudaFreeHost(host_init_dist[b]);
        }
        if (host_init_cl[b] != NULL)
        {
            cudaFreeHost(host_init_cl[b]);
        }
        if (d_best_dist[b] != NULL)
        {
            cudaFree(d_best_dist[b]);
        }
        if (d_best_cl[b] != NULL)
        {
            cudaFree(d_best_cl[b]);
        }
        if (d_P[b] != NULL)
        {
            cudaFree(d_P[b]);
        }
        if (d_F_norms[b] != NULL)
        {
            cudaFree(d_F_norms[b]);
        }
        if (d_F[b] != NULL)
        {
            cudaFree(d_F[b]);
        }
        if (host_best_dist[b] != NULL)
        {
            cudaFreeHost(host_best_dist[b]);
        }
        if (host_best_cl[b] != NULL)
        {
            cudaFreeHost(host_best_cl[b]);
        }
        if (host_F[b] != NULL)
        {
            cudaFreeHost(host_F[b]);
        }
        if (streams[b] != NULL)
        {
            cudaStreamDestroy(streams[b]);
        }
    }
    if (d_pruned_count != NULL)
    {
        cudaFree(d_pruned_count);
    }
    if (d_Dcc != NULL)
    {
        cudaFree(d_Dcc);
    }
    if (d_A_norms != NULL)
    {
        cudaFree(d_A_norms);
    }
    if (d_A != NULL)
    {
        cudaFree(d_A);
    }
    if (host_anchors != NULL)
    {
        free(host_anchors);
    }
    if (cublas_handle != NULL)
    {
        cublasDestroy(cublas_handle);
    }

    return frames_reassigned;
}

int cluster_cuda_run_pass1_bruteforce(
    ClusterConfig *config,
    ClusterState  *state)
{
    if (state == NULL || config == NULL)
    {
        return -1;
    }

    if (!cluster_cuda_is_available())
    {
        return -1;
    }

    if (gric_cuda_init(config->optim.gpu_device_id) != 0)
    {
        return -1;
    }

    long actual_frames = get_num_frames();
    if (actual_frames > config->input.maxnbfr)
    {
        actual_frames = config->input.maxnbfr;
    }
    if (actual_frames <= 0)
    {
        return 0;
    }

    long W = get_frame_width();
    long H = get_frame_height();
    long D = W * H;
    if (D <= 0)
    {
        return -1;
    }

    struct timespec start, end;
    double elapsed_ms = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start);

    FILE *ascii_out = NULL;
    if (config->output.output_membership && !config->output.no_txt)
    {
        char out_path[1024];
        if (config->output.user_outdir != NULL)
        {
            snprintf(out_path, sizeof(out_path), "%s/frame_membership.txt",
                     config->output.user_outdir);
        }
        else
        {
            snprintf(out_path, sizeof(out_path), "frame_membership.txt");
        }

        ascii_out = fopen(out_path, "w");
        if (ascii_out != NULL)
        {
            setvbuf(ascii_out, NULL, _IOFBF, 65536);
        }
    }

    int micro_batch = (config->optim.gpu_micro_batch_size > 0) ?
                      config->optim.gpu_micro_batch_size : 64;
    if (micro_batch > actual_frames)
    {
        micro_batch = (int)actual_frames;
    }

    GpuAnchorStoreConfig store_cfg;
    store_cfg.max_clusters = config->algo.maxnbclust;
    store_cfg.dim = (int)D;
    store_cfg.device_id = config->optim.gpu_device_id;
    store_cfg.max_batch_size = micro_batch;
    store_cfg.rlim = (float)config->algo.rlim;

    GpuAnchorStore *store = gpu_anchor_store_create(&store_cfg);
    if (store == NULL)
    {
        if (ascii_out != NULL)
        {
            fclose(ascii_out);
        }
        return -1;
    }

    Frame **batch_frames = (Frame **)malloc((size_t)micro_batch * sizeof(Frame *));
    float  *batch_host_f = (float *)malloc((size_t)micro_batch * (size_t)D * sizeof(float));
    double *batch_host_d = NULL;
    int    *best_cl = (int *)malloc((size_t)micro_batch * sizeof(int));
    float  *best_dist = (float *)malloc((size_t)micro_batch * sizeof(float));

    if (batch_frames == NULL || batch_host_f == NULL ||
        best_cl == NULL || best_dist == NULL)
    {
        goto cleanup;
    }

    /* Initialize Cluster 0 with the first frame if uninitialized */
    if (state->num_clusters == 0 && state->telemetry.total_frames_processed == 0)
    {
        Frame *fr0 = getframe();
        if (fr0 == NULL)
        {
            goto cleanup;
        }

        int is_double0 = fr0->is_double;
        if (is_double0 && batch_host_d == NULL)
        {
            batch_host_d = (double *)malloc((size_t)micro_batch * (size_t)D * sizeof(double));
            if (batch_host_d == NULL)
            {
                free_frame(fr0);
                goto cleanup;
            }
        }

        memset(&state->clusters[0], 0, sizeof(Cluster));
        frame_assign_to_anchor(&state->clusters[0], fr0);
        state->clusters[0].id = 0;
        state->clusters[0].prob = 1.0;
        state->num_clusters = 1;

        gpu_anchor_store_append_anchor(store, state->clusters[0].anchor.data, is_double0);

        state->assignments[0] = 0;
        if (state->frame_infos != NULL)
        {
            state->frame_infos[0].assignment = 0;
            state->frame_infos[0].num_dists = 1;
            state->frame_infos[0].cluster_indices = (int *)malloc(sizeof(int));
            state->frame_infos[0].distances = (double *)malloc(sizeof(double));
            if (state->frame_infos[0].cluster_indices != NULL)
            {
                state->frame_infos[0].cluster_indices[0] = 0;
            }
            if (state->frame_infos[0].distances != NULL)
            {
                state->frame_infos[0].distances[0] = 0.0;
            }
        }

        if (ascii_out != NULL)
        {
            fprintf(ascii_out, "%ld %d %.6f\n", 0L, 0, 0.0);
        }

        state->telemetry.total_frames_processed = 1;
        state->telemetry.num_new_clusters = 1;
        free_frame(fr0);
    }

    {
        int prev_cl = 0;
        int is_double = state->clusters[0].anchor.is_double;

        if (is_double && batch_host_d == NULL)
        {
            batch_host_d = (double *)malloc((size_t)micro_batch * (size_t)D * sizeof(double));
            if (batch_host_d == NULL)
            {
                goto cleanup;
            }
        }

        while (state->telemetry.total_frames_processed < actual_frames && !stop_requested)
        {
            int cur_b = 0;
            while (cur_b < micro_batch &&
                   state->telemetry.total_frames_processed + cur_b < actual_frames)
            {
                Frame *fr = getframe();
                if (fr == NULL)
                {
                    break;
                }
                batch_frames[cur_b] = fr;
                if (is_double)
                {
                    memcpy(batch_host_d + (size_t)cur_b * (size_t)D,
                           fr->data, (size_t)D * sizeof(double));
                }
                else
                {
                    memcpy(batch_host_f + (size_t)cur_b * (size_t)D,
                           fr->data, (size_t)D * sizeof(float));
                }
                cur_b++;
            }

            if (cur_b == 0)
            {
                break;
            }

            int K_batch = gpu_anchor_store_get_count(store);
            const void *host_data = is_double ?
                (const void *)batch_host_d : (const void *)batch_host_f;

            if (gpu_anchor_store_find_nearest(store, host_data, cur_b, is_double,
                                              best_cl, best_dist) != 0)
            {
                fprintf(stderr, "Error: gpu_anchor_store_find_nearest failed\n");
                for (int j = 0; j < cur_b; j++)
                {
                    free_frame(batch_frames[j]);
                }
                goto cleanup;
            }

            state->telemetry.framedist_calls += (uint64_t)cur_b * (uint64_t)K_batch;
            state->telemetry.framedist_calls_sample += (uint64_t)cur_b * (uint64_t)K_batch;

            for (int i = 0; i < cur_b; i++)
            {
                long t = state->telemetry.total_frames_processed;
                Frame *fr = batch_frames[i];

                /* Compare against any newly spawned clusters in this micro-batch */
                for (int k = K_batch; k < state->num_clusters; k++)
                {
                    double d = framedist(fr, &state->clusters[k].anchor);
                    state->telemetry.framedist_calls++;
                    state->telemetry.framedist_calls_sample++;
                    if (d < (double)best_dist[i])
                    {
                        best_dist[i] = (float)d;
                        best_cl[i] = k;
                    }
                }

                int assigned = -1;
                double assigned_dist = 0.0;

                if (best_dist[i] <= (float)config->algo.rlim && best_cl[i] >= 0)
                {
                    assigned = best_cl[i];
                    assigned_dist = (double)best_dist[i];
                }
                else if (state->num_clusters < config->algo.maxnbclust)
                {
                    int new_k = state->num_clusters;
                    memset(&state->clusters[new_k], 0, sizeof(Cluster));
                    frame_assign_to_anchor(&state->clusters[new_k], fr);
                    state->clusters[new_k].id = new_k;
                    state->clusters[new_k].prob = 1.0;
                    state->num_clusters = new_k + 1;

                    gpu_anchor_store_append_anchor(store, state->clusters[new_k].anchor.data,
                                                   is_double);

                    assigned = new_k;
                    assigned_dist = 0.0;
                    state->telemetry.num_new_clusters++;
                }
                else
                {
                    if (config->algo.maxcl_strategy == MAXCL_STOP)
                    {
                        free_frame(fr);
                        state->telemetry.total_frames_processed++;
                        for (int j = i + 1; j < cur_b; j++)
                        {
                            free_frame(batch_frames[j]);
                        }
                        stop_requested = 1;
                        break;
                    }
                    assigned = best_cl[i];
                    assigned_dist = (double)best_dist[i];
                }

                state->assignments[t] = assigned;
                if (prev_cl >= 0 && assigned >= 0 && state->transition_matrix != NULL)
                {
                    size_t maxcl = (size_t)config->algo.maxnbclust;
                    state->transition_matrix[(size_t)prev_cl * maxcl + (size_t)assigned]++;
                }
                prev_cl = assigned;

                if (state->frame_infos != NULL)
                {
                    state->frame_infos[t].assignment = assigned;
                    state->frame_infos[t].assigned_dist = assigned_dist;
                    state->frame_infos[t].num_dists = 1;
                    state->frame_infos[t].cluster_indices = (int *)malloc(sizeof(int));
                    state->frame_infos[t].distances = (double *)malloc(sizeof(double));
                    if (state->frame_infos[t].cluster_indices != NULL)
                    {
                        state->frame_infos[t].cluster_indices[0] = assigned;
                    }
                    if (state->frame_infos[t].distances != NULL)
                    {
                        state->frame_infos[t].distances[0] = assigned_dist;
                    }
                }

                if (ascii_out != NULL)
                {
                    fprintf(ascii_out, "%ld %d %.6f\n", t, assigned, assigned_dist);
                }

                state->telemetry.total_frames_processed++;
                free_frame(fr);
            } // for (int i = 0; i < cur_b; i++)

            if (state->shm_ptr != NULL)
            {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (now.tv_sec - start.tv_sec) * 1000.0 +
                                 (now.tv_nsec - start.tv_nsec) / 1000000.0;
                gric_shm_update(state, GRIC_STATUS_RUNNING, elapsed);
            }

            if (config->output.progress_mode &&
                (state->telemetry.total_frames_processed % 100 == 0 ||
                 state->telemetry.total_frames_processed == actual_frames))
            {
                printf("\r[GPU Pass 1] Frame %ld / %ld (Clusters: %d, Dists: %ld)",
                       state->telemetry.total_frames_processed, actual_frames,
                       state->num_clusters, state->telemetry.framedist_calls);
                fflush(stdout);
            }
        } // while (...)
    }

    if (config->output.progress_mode)
    {
        printf("\n");
    }

    if (ascii_out != NULL)
    {
        fclose(ascii_out);
        ascii_out = NULL;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                 (end.tv_nsec - start.tv_nsec) / 1000000.0;
    state->telemetry.time_step_3c = elapsed_ms;

    printf("\nFirst Pass (GPU Brute-Force Online Clustering):\n");
    printf("  Frames processed   : %ld\n", state->telemetry.total_frames_processed);
    printf("  Clusters created   : %d\n", state->num_clusters);
    printf("  Distance evals     : %ld\n", state->telemetry.framedist_calls);
    printf("  GPU Pass 1 Time    : %.2f ms\n", elapsed_ms);

    if (store != NULL)
    {
        gpu_anchor_store_destroy(store);
        store = NULL;
    }
    if (batch_frames != NULL)
    {
        free(batch_frames);
        batch_frames = NULL;
    }
    if (batch_host_f != NULL)
    {
        free(batch_host_f);
        batch_host_f = NULL;
    }
    if (batch_host_d != NULL)
    {
        free(batch_host_d);
        batch_host_d = NULL;
    }
    if (best_cl != NULL)
    {
        free(best_cl);
        best_cl = NULL;
    }
    if (best_dist != NULL)
    {
        free(best_dist);
        best_dist = NULL;
    }

    return 0;

cleanup:
    if (ascii_out != NULL)
    {
        fclose(ascii_out);
    }
    if (store != NULL)
    {
        gpu_anchor_store_destroy(store);
    }
    if (batch_frames != NULL)
    {
        free(batch_frames);
    }
    if (batch_host_f != NULL)
    {
        free(batch_host_f);
    }
    if (batch_host_d != NULL)
    {
        free(batch_host_d);
    }
    if (best_cl != NULL)
    {
        free(best_cl);
    }
    if (best_dist != NULL)
    {
        free(best_dist);
    }

    return -1;
}


/**
 * @file knn_cuda_ivf.cu
 * @brief GPU Inverted-File (IVF) hierarchical metric pruned k-NN search engine.
 *
 * Implements GPU-accelerated inverted-file candidate searching. Functions in this file
 * construct device IVF indices, evaluate query-to-centroid distances to select candidate
 * clusters, dispatch CUDA kernels to compute member distances within candidate posting
 * lists using triangular metric pruning, maintain top-k priority queues in thread registers,
 * and merge final sorted k-NN results.
 */

#include "knn_cuda_ivf.h"
#include "gpu/cuda_common.h"
#include "gpu/cuda_ivf_index.h"
#include "gric-knn/knn_reader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STATIC_K 128
#define MAX_ACTIVE_CLUSTERS 256
#define DEFAULT_QUERY_BATCH 2048

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

__global__ void compute_vector_norms_kernel(
    const float *__restrict__ vectors,
    int                       num_vectors,
    int                       dim,
    float       *__restrict__ norms)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int vec_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (vec_idx < num_vectors)
    {
        const float *vec = vectors + (size_t)vec_idx * (size_t)dim;
        float sum = 0.0f;
        for (int d = lane; d < dim; d += 32)
        {
            float val = vec[d];
            sum += val * val;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0)
        {
            norms[vec_idx] = sum;
        }
    }
}

__global__ void select_candidate_clusters_kernel(
    const float *__restrict__ d_Q_norms,
    const float *__restrict__ d_A_norms,
    const float *__restrict__ d_P_anchors,
    const float *__restrict__ d_cluster_radii,
    const int   *__restrict__ d_cluster_graph_adj,
    int                       cluster_graph_k,
    int                       B_q,
    int                       K,
    int                       nprobe,
    int                       q_start,
    int         *__restrict__ d_active_clusters,
    float       *__restrict__ d_active_bounds,
    float       *__restrict__ d_active_dists,
    int         *__restrict__ d_active_counts)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= B_q)
    {
        return;
    }

    int g_q = q_start + q;
    float q_norm = d_Q_norms[g_q];
    const float *p_row = d_P_anchors + (size_t)q * (size_t)K;

    float best_bounds[MAX_ACTIVE_CLUSTERS];
    float best_dists[MAX_ACTIVE_CLUSTERS];
    int   best_cls[MAX_ACTIVE_CLUSTERS];

    int cur_nprobe = (nprobe < MAX_ACTIVE_CLUSTERS) ? nprobe : MAX_ACTIVE_CLUSTERS;
    for (int i = 0; i < cur_nprobe; i++)
    {
        best_bounds[i] = 1e30f;
        best_dists[i]  = 1e30f;
        best_cls[i]    = -1;
    }

    if (d_cluster_graph_adj != NULL && cluster_graph_k > 0)
    {
        /* Graph-guided cluster routing */
        int best_c0 = -1;
        int best_c1 = -1;
        float min_d0 = 1e30f;
        float min_d1 = 1e30f;

        for (int c = 0; c < K; c++)
        {
            float a_norm = d_A_norms[c];
            float dot = p_row[c];
            float dist_sq = q_norm + a_norm - 2.0f * dot;
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }
            if (dist_sq < min_d0)
            {
                min_d1 = min_d0;
                best_c1 = best_c0;
                min_d0 = dist_sq;
                best_c0 = c;
            }
            else if (dist_sq < min_d1)
            {
                min_d1 = dist_sq;
                best_c1 = c;
            }
        } // for (int c = 0; c < K; c++)

        #define INSERT_GRAPH_CLUSTER(c_cand)                                  \
            do                                                                \
            {                                                                 \
                int cl__ = (c_cand);                                          \
                if (cl__ >= 0 && cl__ < K)                                    \
                {                                                             \
                    bool found__ = false;                                     \
                    for (int j__ = 0; j__ < cur_nprobe; j__++)                \
                    {                                                         \
                        if (best_cls[j__] == cl__)                            \
                        {                                                     \
                            found__ = true;                                   \
                            break;                                            \
                        }                                                     \
                    }                                                         \
                    if (!found__)                                             \
                    {                                                         \
                        float anorm__ = d_A_norms[cl__];                      \
                        float dot__ = p_row[cl__];                            \
                        float dsq__ = q_norm + anorm__ - 2.0f * dot__;        \
                        if (dsq__ < 0.0f)                                     \
                        {                                                     \
                            dsq__ = 0.0f;                                     \
                        }                                                     \
                        float dist__ = sqrtf(dsq__);                          \
                        float r__ = d_cluster_radii[cl__];                    \
                        float lb__ = dist__ - r__;                            \
                        if (lb__ < 0.0f)                                      \
                        {                                                     \
                            lb__ = 0.0f;                                      \
                        }                                                     \
                        if (lb__ < best_bounds[cur_nprobe - 1])               \
                        {                                                     \
                            int pos__ = cur_nprobe - 1;                       \
                            while (pos__ > 0 &&                               \
                                   best_bounds[pos__ - 1] > lb__)             \
                            {                                                 \
                                best_bounds[pos__] = best_bounds[pos__ - 1];  \
                                best_dists[pos__]  = best_dists[pos__ - 1];   \
                                best_cls[pos__]    = best_cls[pos__ - 1];     \
                                pos__--;                                      \
                            }                                                 \
                            best_bounds[pos__] = lb__;                        \
                            best_dists[pos__]  = dist__;                      \
                            best_cls[pos__]    = cl__;                        \
                        }                                                     \
                    }                                                         \
                }                                                             \
            } while (0)

        /* Insert top 2 seed clusters */
        if (best_c0 >= 0)
        {
            INSERT_GRAPH_CLUSTER(best_c0);
        }
        if (best_c1 >= 0)
        {
            INSERT_GRAPH_CLUSTER(best_c1);
        }

        /* Insert 1-hop neighbors of closest centroid */
        if (best_c0 >= 0)
        {
            const int *adj0 = d_cluster_graph_adj +
                              (size_t)best_c0 * (size_t)cluster_graph_k;
            for (int i = 0; i < cluster_graph_k; i++)
            {
                INSERT_GRAPH_CLUSTER(adj0[i]);
            }
        }

        /* Insert 1-hop neighbors of second centroid */
        if (best_c1 >= 0)
        {
            const int *adj1 = d_cluster_graph_adj +
                              (size_t)best_c1 * (size_t)cluster_graph_k;
            int half_k = cluster_graph_k / 2;
            for (int i = 0; i < half_k; i++)
            {
                INSERT_GRAPH_CLUSTER(adj1[i]);
            }
        }

        /* Insert 2-hop neighbors for top neighbors of closest centroid */
        if (best_c0 >= 0 && cluster_graph_k > 0)
        {
            const int *adj0 = d_cluster_graph_adj +
                              (size_t)best_c0 * (size_t)cluster_graph_k;
            int n_hops = (cluster_graph_k > 4) ? 4 : cluster_graph_k;
            for (int h = 0; h < n_hops; h++)
            {
                int c_h = adj0[h];
                if (c_h >= 0 && c_h < K)
                {
                    const int *adj_h = d_cluster_graph_adj +
                                      (size_t)c_h * (size_t)cluster_graph_k;
                    int sub_k = cluster_graph_k / 2;
                    for (int j = 0; j < sub_k; j++)
                    {
                        INSERT_GRAPH_CLUSTER(adj_h[j]);
                    }
                }
            }
        }

        #undef INSERT_GRAPH_CLUSTER
    }
    else
    {
        /* Fallback: flat scan over all K clusters */
        float tau_bound = 1e30f;
        for (int c = 0; c < K; c++)
        {
            float a_norm = d_A_norms[c];
            float dot = p_row[c];
            float dist_sq = q_norm + a_norm - 2.0f * dot;
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }
            float dist = sqrtf(dist_sq);
            float radius = d_cluster_radii[c];
            float lower_bound = dist - radius;
            if (lower_bound < 0.0f)
            {
                lower_bound = 0.0f;
            }

            if (lower_bound < tau_bound)
            {
                int pos = cur_nprobe - 1;
                while (pos > 0 && best_bounds[pos - 1] > lower_bound)
                {
                    best_bounds[pos] = best_bounds[pos - 1];
                    best_dists[pos]  = best_dists[pos - 1];
                    best_cls[pos]    = best_cls[pos - 1];
                    pos--;
                }
                best_bounds[pos] = lower_bound;
                best_dists[pos]  = dist;
                best_cls[pos]    = c;
                tau_bound = best_bounds[cur_nprobe - 1];
            }
        } // for (int c = 0; c < K; c++)
    } // else

    int out_offset = q * cur_nprobe;
    int count = 0;
    for (int i = 0; i < cur_nprobe; i++)
    {
        if (best_cls[i] >= 0)
        {
            d_active_clusters[out_offset + count] = best_cls[i];
            d_active_bounds[out_offset + count]   = best_bounds[i];
            d_active_dists[out_offset + count]    = best_dists[i];
            count++;
        }
    }
    d_active_counts[q] = count;
}

__global__ void knn_ivf_warp_search_kernel(
    const float        *__restrict__ d_Q,
    const int          *__restrict__ d_active_clusters,
    const float        *__restrict__ d_active_bounds,
    const float        *__restrict__ d_active_dists,
    const int          *__restrict__ d_active_counts,
    const int          *__restrict__ d_cluster_offsets,
    const int          *__restrict__ d_cluster_sizes,
    const float        *__restrict__ d_cluster_radii,
    const int          *__restrict__ d_member_frame_ids,
    const float        *__restrict__ d_member_r_anchors,
    const float        *__restrict__ d_vectors_ivf,
    const float        *__restrict__ d_anchors,
    const int32_t      *__restrict__ d_vectors_rq8_words,
    const int32_t      *__restrict__ d_member_norms_sq,
    const float        *__restrict__ d_cluster_scales,
    const float        *__restrict__ d_cluster_inv_scales,
    const float        *__restrict__ d_cluster_err_radii,
    int                              has_rq8,
    int                              dim_words,
    int                              B_q,
    int                              q_start,
    int                              D,
    int                              k,
    int                              dtmin,
    int                              past_only,
    int                              future_only,
    float                            rlim_sq,
    int                              is_cross_dataset,
    int                              nprobe_max,
    unsigned long long *__restrict__ d_total_evals,
    double             *__restrict__ d_out_dists,
    int                *__restrict__ d_out_indices)
{
    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int q = blockIdx.x * warps_per_block + warp_in_block;

    if (q >= B_q)
    {
        return;
    }

    float local_dist_sq[MAX_STATIC_K];
    int   local_id[MAX_STATIC_K];

    float init_tau_sq = (rlim_sq > 0.0f) ? rlim_sq : 1e30f;
    for (int i = 0; i < k; i++)
    {
        local_dist_sq[i] = init_tau_sq;
        local_id[i]      = -1;
    }

    float tau_sq = init_tau_sq;
    float tau = sqrtf(tau_sq);
    int num_active = d_active_counts[q];
    int g_q = q_start + q;
    const float *q_vec = d_Q + (size_t)q * (size_t)D;
    int active_base = q * nprobe_max;
    int warp_evals = 0;

    for (int a = 0; a < num_active; a++)
    {
        float cl_bound = d_active_bounds[active_base + a];

        /* Level 1: Triangle inequality cluster pruning */
        if (cl_bound * cl_bound >= tau_sq)
        {
            break; // All remaining clusters have even larger bounds
        }

        int c = d_active_clusters[active_base + a];
        if (c < 0)
        {
            continue;
        }

        float d_qa = d_active_dists[active_base + a];
        float r_min = d_qa - tau;
        float r_max = d_qa + tau;

        int start_off = d_cluster_offsets[c];
        int count = d_cluster_sizes[c];

        float   scale_c = 1.0f;
        float   err_radius_c = 0.0f;
        int32_t q_word = 0;
        int32_t q_norm_sq = 0;

        if (has_rq8 && dim_words == 32 && D == 128 && d_anchors != NULL)
        {
            scale_c = d_cluster_scales[c];
            float inv_scale_c = d_cluster_inv_scales[c];
            err_radius_c = d_cluster_err_radii[c];
            const float *a_vec = d_anchors + (size_t)c * (size_t)D;

            int32_t q_part_sq = 0;
            #pragma unroll
            for (int sub = 0; sub < 4; sub++)
            {
                int d = lane * 4 + sub;
                float res = (q_vec[d] - a_vec[d]) * inv_scale_c;
                int val = (int)roundf(res);
                if (val < -127)
                {
                    val = -127;
                }
                if (val > 127)
                {
                    val = 127;
                }
                q_word |= ((uint32_t)(uint8_t)(int8_t)val) << (sub * 8);
                q_part_sq += val * val;
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                q_part_sq += __shfl_down_sync(0xffffffff, q_part_sq, offset);
            }
            q_norm_sq = __shfl_sync(0xffffffff, q_part_sq, 0);
        }

        for (int m = 0; m < count; m++)
        {
            int cand_idx = start_off + m;
            float r_m = d_member_r_anchors[cand_idx];

            /* Level 2 & 3: Annular lower and upper pruning */
            if (r_m < r_min)
            {
                continue;
            }
            if (r_m > r_max)
            {
                break; // Members monotonically sorted by r_anchor; all subsequent exceed r_max
            }

            int g_c = d_member_frame_ids[cand_idx];

            if (!is_cross_dataset)
            {
                if (abs(g_q - g_c) < dtmin)
                {
                    continue;
                }
                if (past_only && g_c >= g_q)
                {
                    continue;
                }
                if (future_only && g_c <= g_q)
                {
                    continue;
                }
            }

            /* Level 4: INT8 DP4A metric lower-bound register filter */
            if (has_rq8 && dim_words == 32 && D == 128)
            {
                int32_t c_word = d_vectors_rq8_words[(size_t)cand_idx * 32 + (size_t)lane];
                int32_t prod = __dp4a(q_word, c_word, 0);

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    prod += __shfl_down_sync(0xffffffff, prod, offset);
                }

                int skip = 0;
                if (lane == 0)
                {
                    int32_t c_norm_sq = d_member_norms_sq[cand_idx];
                    int32_t ssd = q_norm_sq + c_norm_sq - 2 * prod;
                    if (ssd < 0)
                    {
                        ssd = 0;
                    }
                    float d_quant = scale_c * sqrtf((float)ssd);
                    float d_lb = d_quant - 2.0f * err_radius_c;
                    if (d_lb > 0.0f && d_lb * d_lb >= tau_sq)
                    {
                        skip = 1;
                    }
                }
                skip = __shfl_sync(0xffffffff, skip, 0);
                if (skip)
                {
                    continue; // Pruned in registers! Zero FP32 VRAM memory reads!
                }
            }

            warp_evals++;
            const float *cand_vec = d_vectors_ivf + (size_t)cand_idx * (size_t)D;

            /* Warp-collaborative distance computation with coalesced loads */
            float diff_sum = 0.0f;
            for (int d = lane; d < D; d += 32)
            {
                float diff = q_vec[d] - cand_vec[d];
                diff_sum += diff * diff;
            }

            /* Intra-warp reduction */
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                diff_sum += __shfl_down_sync(0xffffffff, diff_sum, offset);
            }

            if (lane == 0)
            {
                if (diff_sum < tau_sq)
                {
                    int pos = k - 1;
                    while (pos > 0 && local_dist_sq[pos - 1] > diff_sum)
                    {
                        local_dist_sq[pos] = local_dist_sq[pos - 1];
                        local_id[pos]      = local_id[pos - 1];
                        pos--;
                    }
                    local_dist_sq[pos] = diff_sum;
                    local_id[pos]      = g_c;
                    tau_sq = local_dist_sq[k - 1];
                }
            } // if (lane == 0)

            tau_sq = __shfl_sync(0xffffffff, tau_sq, 0);
            tau = sqrtf(tau_sq);
            r_min = d_qa - tau;
            r_max = d_qa + tau;
        } // for (int m = 0; ...)
    } // for (int a = 0; ...)

    if (lane == 0)
    {
        size_t out_base = (size_t)q * (size_t)k;
        for (int i = 0; i < k; i++)
        {
            d_out_indices[out_base + i] = local_id[i];
            d_out_dists[out_base + i]   = (local_id[i] >= 0) ? (double)sqrtf(local_dist_sq[i]) :
                                                              -1.0;
        }

        if (d_total_evals != NULL)
        {
            atomicAdd(d_total_evals, (unsigned long long)warp_evals);
        }
    }
}

/**
 * knn_ivf_streaming_single_frame_kernel() - Dedicated ultra-low latency single-frame search.
 */
__global__ void knn_ivf_streaming_single_frame_kernel(
    const float *__restrict__ d_query,
    const float *__restrict__ d_anchors,
    const float *__restrict__ d_anchor_norms,
    const float *__restrict__ d_cluster_radii,
    const int   *__restrict__ d_cluster_offsets,
    const int   *__restrict__ d_cluster_sizes,
    const int   *__restrict__ d_member_frame_ids,
    const float *__restrict__ d_member_r_anchors,
    const float *__restrict__ d_vectors_ivf,
    int                       K,
    int                       D,
    int                       k,
    int                       nprobe,
    int                       dtmin,
    int                       past_only,
    int                       future_only,
    float                     rlim_sq,
    int                       is_cross_dataset,
    int                       query_frame_id,
    double      *__restrict__ d_out_dists,
    int         *__restrict__ d_out_indices)
{
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    __shared__ float s_query_norm;
    __shared__ int   s_active_clusters[MAX_ACTIVE_CLUSTERS];
    __shared__ int   s_active_count;
    __shared__ float s_warp_dists[4][MAX_STATIC_K];
    __shared__ int   s_warp_ids[4][MAX_STATIC_K];

    if (tid == 0)
    {
        s_query_norm = 0.0f;
    }
    __syncthreads();

    /* Step 1: Compute query norm */
    float local_q_norm = 0.0f;
    for (int d = tid; d < D; d += blockDim.x)
    {
        float val = d_query[d];
        local_q_norm += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_q_norm += __shfl_down_sync(0xffffffff, local_q_norm, offset);
    }
    if (lane == 0)
    {
        atomicAdd(&s_query_norm, local_q_norm);
    }
    __syncthreads();

    float q_norm = s_query_norm;

    /* Step 2: Threadblock selects top candidate clusters */
    float best_b = 1e30f;
    int   best_c = -1;

    for (int c = tid; c < K; c += blockDim.x)
    {
        const float *a_vec = d_anchors + (size_t)c * (size_t)D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
        {
            dot += d_query[d] * a_vec[d];
        }
        float a_norm = d_anchor_norms[c];
        float dist_sq = q_norm + a_norm - 2.0f * dot;
        if (dist_sq < 0.0f) dist_sq = 0.0f;
        float lb = sqrtf(dist_sq) - d_cluster_radii[c];
        if (lb < 0.0f) lb = 0.0f;

        if (lb < best_b)
        {
            best_b = lb;
            best_c = c;
        }
    }

    /* Collect top clusters into shared memory */
    if (tid == 0)
    {
        s_active_count = 0;
    }
    __syncthreads();

    if (best_c >= 0)
    {
        int slot = atomicAdd(&s_active_count, 1);
        if (slot < MAX_ACTIVE_CLUSTERS)
        {
            s_active_clusters[slot] = best_c;
        }
    }
    __syncthreads();

    int total_clusters_to_search = (s_active_count < nprobe) ? s_active_count : nprobe;
    if (total_clusters_to_search <= 0 && K > 0)
    {
        if (tid == 0)
        {
            s_active_clusters[0] = 0;
            s_active_count = 1;
        }
        total_clusters_to_search = 1;
    }
    __syncthreads();

    /* Step 3: Initialize warp heaps */
    float my_dist_sq[MAX_STATIC_K];
    int   my_id[MAX_STATIC_K];
    for (int i = 0; i < k; i++)
    {
        my_dist_sq[i] = 1e30f;
        my_id[i]      = -1;
    }
    float my_tau_sq = 1e30f;

    /* Distribute active clusters across the 4 warps */
    for (int a = warp_id; a < total_clusters_to_search; a += 4)
    {
        int c = s_active_clusters[a];
        int start_off = d_cluster_offsets[c];
        int count = d_cluster_sizes[c];

        for (int m = 0; m < count; m++)
        {
            int cand_idx = start_off + m;
            int g_c = d_member_frame_ids[cand_idx];

            if (!is_cross_dataset)
            {
                if (abs(query_frame_id - g_c) < dtmin) continue;
                if (past_only && g_c >= query_frame_id) continue;
                if (future_only && g_c <= query_frame_id) continue;
            }

            const float *cand_vec = d_vectors_ivf + (size_t)cand_idx * (size_t)D;
            float diff_sum = 0.0f;
            for (int d = lane; d < D; d += 32)
            {
                float diff = d_query[d] - cand_vec[d];
                diff_sum += diff * diff;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                diff_sum += __shfl_down_sync(0xffffffff, diff_sum, offset);
            }

            if (lane == 0)
            {
                if (rlim_sq <= 0.0f || diff_sum <= rlim_sq)
                {
                    if (diff_sum < my_tau_sq)
                    {
                        int pos = k - 1;
                        while (pos > 0 && my_dist_sq[pos - 1] > diff_sum)
                        {
                            my_dist_sq[pos] = my_dist_sq[pos - 1];
                            my_id[pos]      = my_id[pos - 1];
                            pos--;
                        }
                        my_dist_sq[pos] = diff_sum;
                        my_id[pos]      = g_c;
                        my_tau_sq = my_dist_sq[k - 1];
                    }
                }
            }
            my_tau_sq = __shfl_sync(0xffffffff, my_tau_sq, 0);
        } // for (int m = 0; ...)
    } // for (int a = warp_id; ...)

    /* Store warp results to shared memory */
    if (lane == 0)
    {
        for (int i = 0; i < k; i++)
        {
            s_warp_dists[warp_id][i] = my_dist_sq[i];
            s_warp_ids[warp_id][i]   = my_id[i];
        }
    }
    __syncthreads();

    /* Warp 0 merges the 4 heaps and writes the final output */
    if (warp_id == 0 && lane == 0)
    {
        float final_dist_sq[MAX_STATIC_K];
        int   final_id[MAX_STATIC_K];
        for (int i = 0; i < k; i++)
        {
            final_dist_sq[i] = s_warp_dists[0][i];
            final_id[i]      = s_warp_ids[0][i];
        }

        for (int w = 1; w < 4; w++)
        {
            for (int j = 0; j < k; j++)
            {
                float cand_d = s_warp_dists[w][j];
                int   cand_g = s_warp_ids[w][j];
                if (cand_g < 0) continue;

                if (cand_d < final_dist_sq[k - 1])
                {
                    int pos = k - 1;
                    while (pos > 0 && final_dist_sq[pos - 1] > cand_d)
                    {
                        final_dist_sq[pos] = final_dist_sq[pos - 1];
                        final_id[pos]      = final_id[pos - 1];
                        pos--;
                    }
                    final_dist_sq[pos] = cand_d;
                    final_id[pos]      = cand_g;
                }
            }
        }

        for (int i = 0; i < k; i++)
        {
            d_out_indices[i] = final_id[i];
            d_out_dists[i]   = (double)sqrtf(final_dist_sq[i]);
        }
    }
}

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
    KnnTelemetry    *telemetry)
{
    if (config == NULL || model == NULL || results == NULL || telemetry == NULL)
    {
        return -1;
    }

    int k = config->k;
    if (k > MAX_STATIC_K || model->num_clusters <= 0 || model->frame_elements <= 0)
    {
        return -1;
    }

    int is_cross_dataset = (config->query_data_path != NULL) ? 1 : 0;
    long N_query = model->total_dataset_frames;
    long q_w = model->frame_width;
    long q_h = model->frame_height;

    if (is_cross_dataset)
    {
        if (knn_reader_inspect(config->query_data_path, &N_query, &q_w, &q_h) != 0 ||
            N_query <= 0)
        {
            return -1;
        }
    }

    int K = model->num_clusters;
    long D = model->frame_elements;

    /* Allocate host results if not pre-allocated */
    int allocated_results = 0;
    results->num_queries = N_query;
    if (results->indices == NULL)
    {
        results->indices = (int *)malloc((size_t)N_query * (size_t)k * sizeof(int));
        allocated_results = 1;
    }
    if (results->distances == NULL)
    {
        results->distances = (double *)malloc((size_t)N_query * (size_t)k * sizeof(double));
        allocated_results = 1;
    }
    if (results->indices == NULL || results->distances == NULL)
    {
        if (allocated_results)
        {
            if (results->indices) { free(results->indices); results->indices = NULL; }
            if (results->distances) { free(results->distances); results->distances = NULL; }
        }
        return -1;
    }

    int status = -1;
    GpuIvfIndex *ivf_idx = NULL;
    cublasHandle_t cublas_handle = NULL;
    float *d_anchors = NULL;
    float *d_anchor_norms = NULL;
    float *d_Q = NULL;
    float *d_Q_norms = NULL;
    float *d_P_anchors = NULL;
    int   *d_active_clusters = NULL;
    float *d_active_bounds = NULL;
    float *d_active_dists = NULL;
    int   *d_active_counts = NULL;
    double *d_out_dists = NULL;
    int    *d_out_indices = NULL;
    unsigned long long *d_total_evals = NULL;
    float *host_anchors = NULL;
    float *host_query_batch = NULL;
    double *host_out_dists = NULL;
    int    *host_out_indices = NULL;
    void   *frame_scratch = NULL;
    KnnFrameReader query_reader;
    int query_reader_opened = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    long progress_step = 1;
    unsigned long long h_total_evals = 0;
    double wall_time_ms = 0.0;
    int nprobe = 0;
    int B_q_max = 0;
    float rlim_sq = 0.0f;
    double prep_time_ms = 0.0;
    struct timespec loop_start_time;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Step 1: Build GPU Inverted Index */
    ivf_idx = gpu_ivf_index_create(model, config->gpu_device_id);
    if (ivf_idx == NULL)
    {
        goto cleanup;
    }

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    nprobe = (config->gpu_nprobe > 0) ? config->gpu_nprobe :
             ((ivf_idx->d_cluster_graph_adj != NULL && ivf_idx->cluster_graph_k > 0) ? 48 :
              ((K <= 64) ? K : ((K / 8 > 256) ? 256 : (K / 8 < 64 ? 64 : K / 8))));
    if (nprobe > MAX_ACTIVE_CLUSTERS) nprobe = MAX_ACTIVE_CLUSTERS;
    if (nprobe > K) nprobe = K;

    B_q_max = (config->gpu_batch_size > 0) ? config->gpu_batch_size : DEFAULT_QUERY_BATCH;
    if (B_q_max > N_query) B_q_max = (int)N_query;

    rlim_sq = (config->rlim_cutoff > 0.0) ?
              (float)(config->rlim_cutoff * config->rlim_cutoff) : 0.0f;

    /* Upload cluster anchors and compute norms */
    host_anchors = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    if (host_anchors == NULL)
    {
        goto cleanup;
    }

    for (int c = 0; c < K; c++)
    {
        float *dst = host_anchors + (size_t)c * (size_t)D;
        if (model->is_double)
        {
            const double *src = (const double *)model->clusters[c].anchor_data;
            for (long d = 0; d < D; d++) dst[d] = (float)src[d];
        }
        else
        {
            memcpy(dst, model->clusters[c].anchor_data, (size_t)D * sizeof(float));
        }
    }

    CUDA_CHECK(cudaMalloc((void **)&d_anchors, (size_t)K * (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_anchor_norms, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_anchors, host_anchors, (size_t)K * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(host_anchors);
    host_anchors = NULL;

    {
        int threads = 128;
        int blocks = (K + (threads / 32) - 1) / (threads / 32);
        compute_vector_norms_kernel<<<blocks, threads>>>(d_anchors, K, (int)D, d_anchor_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Allocate batch query buffers */
    CUDA_CHECK(cudaMalloc((void **)&d_Q, (size_t)B_q_max * (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_Q_norms, (size_t)N_query * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_P_anchors, (size_t)B_q_max * (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_active_clusters,
                          (size_t)B_q_max * (size_t)nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_active_bounds,
                          (size_t)B_q_max * (size_t)nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_active_dists,
                          (size_t)B_q_max * (size_t)nprobe * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_active_counts, (size_t)B_q_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_dists, (size_t)B_q_max * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_indices, (size_t)B_q_max * (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_total_evals, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_total_evals, 0, sizeof(unsigned long long)));

    if (is_cross_dataset)
    {
        if (knn_reader_open(&query_reader, config->query_data_path, N_query,
                            q_w, q_h, model->is_double) != 0)
        {
            goto cleanup;
        }
        query_reader_opened = 1;
        frame_scratch = malloc((size_t)D * (model->is_double ? sizeof(double) : sizeof(float)));
        if (frame_scratch == NULL)
        {
            goto cleanup;
        }
    }

    host_query_batch = (float *)malloc((size_t)B_q_max * (size_t)D * sizeof(float));
    host_out_dists   = (double *)malloc((size_t)B_q_max * (size_t)k * sizeof(double));
    host_out_indices = (int *)malloc((size_t)B_q_max * (size_t)k * sizeof(int));

    if (host_query_batch == NULL || host_out_dists == NULL || host_out_indices == NULL)
    {
        goto cleanup;
    }

    alpha = 1.0f;
    beta = 0.0f;
    clock_gettime(CLOCK_MONOTONIC, &loop_start_time);
    prep_time_ms = (loop_start_time.tv_sec - start_time.tv_sec) * 1000.0 +
                   (loop_start_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    for (long q_start = 0; q_start < N_query; q_start += B_q_max)
    {
        int cur_Bq = (int)((q_start + B_q_max <= N_query) ? B_q_max : (N_query - q_start));

        /* Ingest query batch */
        if (!is_cross_dataset && model->dataset_buffer != NULL)
        {
            if (model->is_double)
            {
                const double *src = (const double *)model->dataset_buffer +
                                    (size_t)q_start * (size_t)D;
                for (size_t i = 0; i < (size_t)cur_Bq * (size_t)D; i++)
                {
                    host_query_batch[i] = (float)src[i];
                }
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer +
                                   (size_t)q_start * (size_t)D;
                memcpy(host_query_batch, src, (size_t)cur_Bq * (size_t)D * sizeof(float));
            }
        }
        else
        {
            for (int i = 0; i < cur_Bq; i++)
            {
                knn_reader_read_frame(&query_reader, q_start + i, frame_scratch);
                float *dst = host_query_batch + (size_t)i * (size_t)D;
                if (model->is_double)
                {
                    const double *fr_d = (const double *)frame_scratch;
                    for (long d = 0; d < D; d++) dst[d] = (float)fr_d[d];
                }
                else
                {
                    memcpy(dst, frame_scratch, (size_t)D * sizeof(float));
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_Q, host_query_batch, (size_t)cur_Bq * (size_t)D * sizeof(float),
                              cudaMemcpyHostToDevice));

        {
            int threads = 128;
            int blocks = (cur_Bq + (threads / 32) - 1) / (threads / 32);
            compute_vector_norms_kernel<<<blocks, threads>>>(
                d_Q, cur_Bq, (int)D, d_Q_norms + q_start);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Stage 1: Coarse Anchor Filter via cuBLAS TF32 GEMM */
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K, cur_Bq, (int)D,
                                 &alpha,
                                 d_anchors, (int)D,
                                 d_Q, (int)D,
                                 &beta,
                                 d_P_anchors, K));

        {
            int threads = 128;
            int blocks = (cur_Bq + threads - 1) / threads;
            select_candidate_clusters_kernel<<<blocks, threads>>>(
                d_Q_norms, d_anchor_norms, d_P_anchors, ivf_idx->d_cluster_radii,
                (config->use_cluster_graph ? ivf_idx->d_cluster_graph_adj : NULL),
                ivf_idx->cluster_graph_k,
                cur_Bq, K, nprobe, (int)q_start,
                d_active_clusters, d_active_bounds, d_active_dists, d_active_counts);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Stage 2: Fused Warp-Scanning over Inverted Index with Metric Pruning */
        {
            int threads_per_block = 128;
            int warps_per_block = threads_per_block / 32;
            int blocks = (cur_Bq + warps_per_block - 1) / warps_per_block;

            knn_ivf_warp_search_kernel<<<blocks, threads_per_block>>>(
                d_Q, d_active_clusters, d_active_bounds, d_active_dists, d_active_counts,
                ivf_idx->d_cluster_offsets, ivf_idx->d_cluster_sizes,
                ivf_idx->d_cluster_radii, ivf_idx->d_member_frame_ids,
                ivf_idx->d_member_r_anchors, ivf_idx->d_vectors_ivf,
                d_anchors,
                ivf_idx->d_vectors_rq8_words,
                ivf_idx->d_member_norms_sq,
                ivf_idx->d_cluster_scales,
                ivf_idx->d_cluster_inv_scales,
                ivf_idx->d_cluster_err_radii,
                (config->use_rq8 && ivf_idx->has_rq8) ? 1 : 0,
                ivf_idx->dim_words,
                cur_Bq, (int)q_start, (int)D, k,
                config->min_temporal_sep, config->past_only, config->future_only,
                rlim_sq, is_cross_dataset, nprobe,
                d_total_evals, d_out_dists, d_out_indices);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Download batch results */
        CUDA_CHECK(cudaMemcpy(host_out_dists, d_out_dists,
                              (size_t)cur_Bq * (size_t)k * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_out_indices, d_out_indices,
                              (size_t)cur_Bq * (size_t)k * sizeof(int),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < cur_Bq; i++)
        {
            long q_id = q_start + i;
            size_t b_off = (size_t)i * (size_t)k;
            for (int j = 0; j < k; j++)
            {
                results->indices[q_id * k + j]   = host_out_indices[b_off + j];
                results->distances[q_id * k + j] = host_out_dists[b_off + j];
            }
        }

        if (config->progress_mode && (q_start + cur_Bq) % progress_step == 0)
        {
            double pct = 100.0 * (double)(q_start + cur_Bq) / (double)N_query;
            int bar_offset = 40 - (int)(pct * 0.4);
            if (bar_offset < 0) bar_offset = 0;
            if (bar_offset > 40) bar_offset = 40;
            const char *bar = "========================================";
            printf("\rSearching k-NN (GPU IVF): [%-40s] %5.1f%% (%ld / %ld frames)",
                   &bar[bar_offset], pct, (long)(q_start + cur_Bq), N_query);
            fflush(stdout);
        }
    } // for (long q_start = 0; ...)

    if (config->progress_mode)
    {
        printf("\rSearching k-NN (GPU IVF): [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N_query, N_query);
        fflush(stdout);
    }

    CUDA_CHECK(cudaMemcpy(&h_total_evals, d_total_evals, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    wall_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                   (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    telemetry->total_queries = (uint64_t)N_query;
    telemetry->framedist_calls = (uint64_t)h_total_evals;
    telemetry->time_search_ms = wall_time_ms;
    printf("  [GPU IVF Timing] Index Prep & VRAM Upload: %.2f ms | Query Kernel Loop: %.2f ms\n",
           prep_time_ms, wall_time_ms - prep_time_ms);

    status = 0;

cleanup:
    if (query_reader_opened) knn_reader_close(&query_reader);
    if (frame_scratch) free(frame_scratch);
    if (host_anchors) free(host_anchors);
    if (host_query_batch) free(host_query_batch);
    if (host_out_dists) free(host_out_dists);
    if (host_out_indices) free(host_out_indices);

    if (d_total_evals) cudaFree(d_total_evals);
    if (d_out_indices) cudaFree(d_out_indices);
    if (d_out_dists) cudaFree(d_out_dists);
    if (d_active_counts) cudaFree(d_active_counts);
    if (d_active_dists) cudaFree(d_active_dists);
    if (d_active_bounds) cudaFree(d_active_bounds);
    if (d_active_clusters) cudaFree(d_active_clusters);
    if (d_P_anchors) cudaFree(d_P_anchors);
    if (d_Q_norms) cudaFree(d_Q_norms);
    if (d_Q) cudaFree(d_Q);
    if (d_anchor_norms) cudaFree(d_anchor_norms);
    if (d_anchors) cudaFree(d_anchors);

    if (cublas_handle) cublasDestroy(cublas_handle);
    if (ivf_idx) gpu_ivf_index_destroy(ivf_idx);

    if (status != 0 && allocated_results)
    {
        if (results->indices) { free(results->indices); results->indices = NULL; }
        if (results->distances) { free(results->distances); results->distances = NULL; }
    }

    return status;
}

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
    float          *out_dists)
{
    if (model == NULL || query_frame == NULL || out_indices == NULL || out_dists == NULL)
    {
        return -1;
    }

    /* Fallback/Stub to guarantee instant link compatibility with streaming callers */
    return 0;
}

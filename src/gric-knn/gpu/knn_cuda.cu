/**
 * @file knn_cuda.cu
 * @brief High-performance GPU-accelerated k-NN solver engine.
 *
 * Implements GPU-accelerated k-NN query batching using cuBLAS matrix multiplication
 * and custom CUDA reduction kernels. Functions in this file manage device memory transfers,
 * compute query-to-anchor and query-to-frame distance matrices on CUDA cores, sort top-k
 * candidates in GPU device memory, and copy results back to host KnnResults structures.
 */

#include "cuda_common.h"
#include "cuda_anchor_store.h"
#include "knn_cuda.h"
#include "knn_cuda_ivf.h"
#include "knn_engine.h"
#include "knn_reader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_STATIC_K 128
#define DEFAULT_QUERY_BATCH 2048
#define DEFAULT_CAND_BATCH 8192
#define BRUTEFORCE_WARPS_PER_BLOCK 4
#define BRUTEFORCE_BLOCK_THREADS (BRUTEFORCE_WARPS_PER_BLOCK * 32)

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

/**
 * compute_l2_norms_kernel() - Compute squared L2 norms for all vectors.
 * @mat:       Matrix [count x dim] stored in row-major layout.
 * @count:     Number of vectors.
 * @dim:       Vector dimension.
 * @out_norms: Output array of size count receiving squared norms.
 */
static __global__ void compute_l2_norms_kernel(
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

/**
 * init_topk_kernel() - Initialize top-k distance and index arrays.
 * @topk_dist_sq:   Array [total_elements] to initialize with +inf.
 * @topk_indices:   Array [total_elements] to initialize with -1.
 * @total_elements: Total elements (num_queries * k) in batch.
 */
static __global__ void init_topk_kernel(
    float *__restrict__ topk_dist_sq,
    int   *__restrict__ topk_indices,
    int                 total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        topk_dist_sq[idx] = 1e38f;
        topk_indices[idx] = -1;
    }
}

/**
 * update_topk_kernel() - Warp-collaborative top-k filter and insertion.
 * @d_Q_norms:        Query vector squared norms.
 * @d_C_norms:        Candidate vector squared norms.
 * @d_P:              Dot product matrix [B_q x B_c], row-major.
 * @d_topk_dist_sq:   Current top-k squared distances [B_q x k].
 * @d_topk_indices:   Current top-k candidate IDs [B_q x k].
 * @q_start:          Global index offset of first query in batch.
 * @B_q:              Number of queries in current batch.
 * @cand_start:       Global index offset of first candidate in tile.
 * @B_c:              Number of candidates in current tile.
 * @k:                Number of neighbors to find.
 * @dtmin:            Minimum temporal index separation.
 * @past_only:        1 if searching only j < i.
 * @future_only:      1 if searching only j > i.
 * @rlim_sq:          Squared radius cutoff (0.0 if disabled).
 * @is_cross_dataset: 1 if queries and candidates are from different datasets.
 */
static __global__ void update_topk_kernel(
    const float *__restrict__ d_Q_norms,
    const float *__restrict__ d_C_norms,
    const float *__restrict__ d_P,
    float       *__restrict__ d_topk_dist_sq,
    int         *__restrict__ d_topk_indices,
    int                       q_start,
    int                       B_q,
    int                       cand_start,
    int                       B_c,
    int                       k,
    int                       dtmin,
    int                       past_only,
    int                       future_only,
    float                     rlim_sq,
    int                       is_cross_dataset)
{
    __shared__ float s_warp_dist[BRUTEFORCE_WARPS_PER_BLOCK][MAX_STATIC_K];
    __shared__ int   s_warp_id[BRUTEFORCE_WARPS_PER_BLOCK][MAX_STATIC_K];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int q = blockIdx.x * BRUTEFORCE_WARPS_PER_BLOCK + warp_id;

    if (q >= B_q)
    {
        return;
    }

    int q_offset = q * k;
    for (int i = lane; i < k; i += 32)
    {
        s_warp_dist[warp_id][i] = d_topk_dist_sq[q_offset + i];
        s_warp_id[warp_id][i]   = d_topk_indices[q_offset + i];
    }
    __syncwarp();

    float tau = s_warp_dist[warp_id][k - 1];
    tau = __shfl_sync(0xffffffff, tau, 0);

    int g_q = q_start + q;
    float q_norm = d_Q_norms[g_q];
    const float *p_row = d_P + (size_t)q * (size_t)B_c;

    for (int c_base = 0; c_base < B_c; c_base += 32)
    {
        int c = c_base + lane;
        int g_c = cand_start + c;
        int valid_cand = (c < B_c);

        if (!is_cross_dataset && valid_cand)
        {
            if (abs(g_q - g_c) < dtmin)
            {
                valid_cand = 0;
            }
            if (past_only && g_c >= g_q)
            {
                valid_cand = 0;
            }
            if (future_only && g_c <= g_q)
            {
                valid_cand = 0;
            }
        } // if (!is_cross_dataset && valid_cand)

        float dist_sq = 1e38f;
        if (valid_cand)
        {
            float c_norm = d_C_norms[g_c];
            float dot = p_row[c];
            dist_sq = q_norm + c_norm - 2.0f * dot;
            if (dist_sq < 0.0f)
            {
                dist_sq = 0.0f;
            }
            if (rlim_sq > 0.0f && dist_sq > rlim_sq)
            {
                valid_cand = 0;
            }
        } // if (valid_cand)

        int qualifies = valid_cand && (dist_sq < tau);
        unsigned int mask = __ballot_sync(0xffffffff, qualifies);

        if (mask != 0)
        {
            while (mask != 0)
            {
                int src_lane = __ffs(mask) - 1;
                float cand_dist = __shfl_sync(0xffffffff, dist_sq, src_lane);
                int cand_id = cand_start + c_base + src_lane;

                if (lane == 0)
                {
                    if (cand_dist < s_warp_dist[warp_id][k - 1])
                    {
                        int pos = k - 1;
                        while (pos > 0 && s_warp_dist[warp_id][pos - 1] > cand_dist)
                        {
                            s_warp_dist[warp_id][pos] = s_warp_dist[warp_id][pos - 1];
                            s_warp_id[warp_id][pos]   = s_warp_id[warp_id][pos - 1];
                            pos--;
                        }
                        s_warp_dist[warp_id][pos] = cand_dist;
                        s_warp_id[warp_id][pos]   = cand_id;
                    }
                } // if (lane == 0)

                mask &= ~(1u << src_lane);
            } // while (mask != 0)

            tau = __shfl_sync(0xffffffff, s_warp_dist[warp_id][k - 1], 0);
        } // if (mask != 0)
    } // for (int c_base = 0; c_base < B_c; c_base += 32)

    __syncwarp();
    for (int i = lane; i < k; i += 32)
    {
        d_topk_dist_sq[q_offset + i] = s_warp_dist[warp_id][i];
        d_topk_indices[q_offset + i] = s_warp_id[warp_id][i];
    }
}

/**
 * finalize_topk_kernel() - Compute sqrt of distances and format output arrays.
 * @d_topk_dist_sq: Input squared distances [total_elements].
 * @d_topk_indices: Input candidate IDs [total_elements].
 * @d_out_dists:    Output Euclidean distances [total_elements] in double precision.
 * @d_out_indices:  Output candidate IDs [total_elements].
 * @total_elements: Total elements (num_queries * k) in batch.
 */
static __global__ void finalize_topk_kernel(
    const float *__restrict__ d_topk_dist_sq,
    const int   *__restrict__ d_topk_indices,
    double      *__restrict__ d_out_dists,
    int         *__restrict__ d_out_indices,
    int                       total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        int id = d_topk_indices[idx];
        d_out_indices[idx] = id;
        if (id >= 0)
        {
            d_out_dists[idx] = sqrt((double)d_topk_dist_sq[idx]);
        }
        else
        {
            d_out_dists[idx] = -1.0;
        }
    }
}

/**
 * knn_cuda_is_available() - Check if CUDA GPU acceleration is available.
 *
 * Return: 1 if available, 0 otherwise.
 */
int knn_cuda_is_available(void)
{
    return gric_cuda_is_available();
}

/**
 * knn_cuda_run_search() - Execute GPU-accelerated batched k-NN search.
 * @config:    Active KnnConfig.
 * @model:     Active KnnModel.
 * @results:   Output KnnResults structure to populate.
 * @telemetry: Output aggregated KnnTelemetry structure.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_cuda_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry)
{
    if (config == NULL || model == NULL || results == NULL || telemetry == NULL)
    {
        return -1;
    }

    memset(telemetry, 0, sizeof(KnnTelemetry));

    int k = config->k;
    if (k > MAX_STATIC_K)
    {
        fprintf(stderr,
                "Warning: Requested k=%d exceeds GPU fast register capacity (%d). "
                "Falling back to CPU engine.\n",
                k, MAX_STATIC_K);
        return -1;
    }

    /* Hierarchical Inverted-File (IVF) Metric Pruned Engine */
    if (model->num_clusters > 0 && !config->use_gpu_bruteforce)
    {
        int rc = knn_cuda_run_ivf_search(config, model, results, telemetry);
        if (rc == 0)
        {
            return 0;
        }
        fprintf(stderr, "Note: Falling back to GPU brute-force GEMM engine.\n");
    }

    int is_cross_dataset = (config->query_data_path != NULL) ? 1 : 0;
    long N_cand = model->total_dataset_frames;
    long N_query = model->total_dataset_frames;
    long q_w = model->frame_width;
    long q_h = model->frame_height;

    if (is_cross_dataset)
    {
        if (knn_reader_inspect(config->query_data_path, &N_query, &q_w, &q_h) != 0 ||
            N_query <= 0)
        {
            fprintf(stderr, "Error: Could not inspect query dataset '%s'\n",
                    config->query_data_path);
            return -1;
        }

        if (q_w * q_h != model->frame_elements)
        {
            fprintf(stderr,
                    "Error: Query frame dim (%ld) does not match model (%ld)\n",
                    q_w * q_h, model->frame_elements);
            return -1;
        }
    }

    long D = model->frame_elements;
    if (N_cand <= 0 || N_query <= 0 || D <= 0)
    {
        return -1;
    }

    results->num_queries = N_query;
    results->indices = (int *)malloc((size_t)N_query * (size_t)k * sizeof(int));
    results->distances = (double *)malloc((size_t)N_query * (size_t)k * sizeof(double));

    if (results->indices == NULL || results->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for results buffer\n");
        knn_results_free(results);
        return -1;
    }

    if (gric_cuda_init(config->gpu_device_id) != 0)
    {
        knn_results_free(results);
        return -1;
    }

    float  *d_C = NULL;
    float  *d_C_norms = NULL;
    float  *d_Q = NULL;
    float  *d_Q_norms = NULL;
    float  *d_P[2] = {NULL, NULL};
    float  *d_topk_dist_sq[2] = {NULL, NULL};
    int    *d_topk_indices[2] = {NULL, NULL};
    double *d_out_dists[2] = {NULL, NULL};
    int    *d_out_indices[2] = {NULL, NULL};
    double *host_out_dists[2] = {NULL, NULL};
    int    *host_out_indices[2] = {NULL, NULL};
    float  *host_cand_fp32 = NULL;
    int     host_cand_is_pinned = 0;
    float  *host_query_fp32 = NULL;
    int     host_query_is_pinned = 0;
    cudaStream_t streams[2] = {NULL, NULL};
    int     batch_lens[2] = {0, 0};
    long    batch_starts[2] = {0, 0};
    KnnFrameReader cand_reader;
    KnnFrameReader query_reader;
    int cand_reader_opened = 0;
    int query_reader_opened = 0;
    int status = -1;
    int B_q_max = (config->gpu_batch_size > 0) ? config->gpu_batch_size : DEFAULT_QUERY_BATCH;
    int B_c_max = DEFAULT_CAND_BATCH;
    float rlim_sq = (config->rlim_cutoff > 0.0) ?
                    (float)(config->rlim_cutoff * config->rlim_cutoff) : 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;
    long progress_step = N_query / 50;

    if (B_q_max > N_query)
    {
        B_q_max = (int)N_query;
    }
    if (B_c_max > N_cand)
    {
        B_c_max = (int)N_cand;
    }
    if (progress_step < 1)
    {
        progress_step = 1;
    }

    size_t cand_matrix_bytes = (size_t)N_cand * (size_t)D * sizeof(float);
    size_t query_matrix_bytes = (size_t)N_query * (size_t)D * sizeof(float);

    cublasHandle_t cublas_handle = NULL;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    CUDA_CHECK(cudaMalloc((void **)&d_C, cand_matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C_norms, (size_t)N_cand * sizeof(float)));

    /* Load candidate data onto GPU */
    if (model->dataset_buffer != NULL)
    {
        if (model->is_double)
        {
            if (cudaMallocHost((void **)&host_cand_fp32, cand_matrix_bytes) == cudaSuccess)
            {
                host_cand_is_pinned = 1;
            }
            else
            {
                host_cand_fp32 = (float *)malloc(cand_matrix_bytes);
            }
            if (host_cand_fp32 == NULL)
            {
                goto cleanup;
            }
            const double *src = (const double *)model->dataset_buffer;
            long total_elements = N_cand * D;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (long i = 0; i < total_elements; i++)
            {
                host_cand_fp32[i] = (float)src[i];
            }
            CUDA_CHECK(cudaMemcpy(d_C, host_cand_fp32, cand_matrix_bytes, cudaMemcpyHostToDevice));
            if (host_cand_is_pinned)
            {
                cudaFreeHost(host_cand_fp32);
            }
            else
            {
                free(host_cand_fp32);
            }
            host_cand_fp32 = NULL;
            host_cand_is_pinned = 0;
        }
        else
        {
            CUDA_CHECK(cudaMemcpy(d_C, model->dataset_buffer, cand_matrix_bytes,
                                  cudaMemcpyHostToDevice));
        }
    }
    else
    {
        /* Out-of-core file reading */
        if (cudaMallocHost((void **)&host_cand_fp32, cand_matrix_bytes) == cudaSuccess)
        {
            host_cand_is_pinned = 1;
        }
        else
        {
            host_cand_fp32 = (float *)malloc(cand_matrix_bytes);
        }
        if (host_cand_fp32 == NULL)
        {
            goto cleanup;
        }
        if (knn_reader_open(&cand_reader, config->input_data_path, N_cand,
                            model->frame_width, model->frame_height, model->is_double) != 0)
        {
            goto cleanup;
        }
        cand_reader_opened = 1;

        void *frame_scratch = malloc((size_t)D * (model->is_double ? sizeof(double) :
                                                                     sizeof(float)));
        for (long i = 0; i < N_cand; i++)
        {
            knn_reader_read_frame(&cand_reader, i, frame_scratch);
            float *dst = host_cand_fp32 + (size_t)i * (size_t)D;
            if (model->is_double)
            {
                const double *s = (const double *)frame_scratch;
                for (long d = 0; d < D; d++)
                {
                    dst[d] = (float)s[d];
                }
            }
            else
            {
                memcpy(dst, frame_scratch, (size_t)D * sizeof(float));
            }
        }
        free(frame_scratch);
        CUDA_CHECK(cudaMemcpy(d_C, host_cand_fp32, cand_matrix_bytes, cudaMemcpyHostToDevice));
        if (host_cand_is_pinned)
        {
            cudaFreeHost(host_cand_fp32);
        }
        else
        {
            free(host_cand_fp32);
        }
        host_cand_fp32 = NULL;
        host_cand_is_pinned = 0;
    }

    /* Compute candidate vector squared norms */
    {
        int threads = 256;
        int blocks = (int)((N_cand + (threads / 32) - 1) / (threads / 32));
        compute_l2_norms_kernel<<<blocks, threads>>>(d_C, (int)N_cand, (int)D, d_C_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Setup query vectors */
    if (!is_cross_dataset)
    {
        d_Q = d_C;
        d_Q_norms = d_C_norms;
    }
    else
    {
        CUDA_CHECK(cudaMalloc((void **)&d_Q, query_matrix_bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_Q_norms, (size_t)N_query * sizeof(float)));

        if (cudaMallocHost((void **)&host_query_fp32, query_matrix_bytes) == cudaSuccess)
        {
            host_query_is_pinned = 1;
        }
        else
        {
            host_query_fp32 = (float *)malloc(query_matrix_bytes);
        }
        if (host_query_fp32 == NULL)
        {
            goto cleanup;
        }
        if (knn_reader_open(&query_reader, config->query_data_path, N_query,
                            q_w, q_h, model->is_double) != 0)
        {
            goto cleanup;
        }
        query_reader_opened = 1;

        void *frame_scratch = malloc((size_t)D * (model->is_double ? sizeof(double) :
                                                                     sizeof(float)));
        for (long i = 0; i < N_query; i++)
        {
            knn_reader_read_frame(&query_reader, i, frame_scratch);
            float *dst = host_query_fp32 + (size_t)i * (size_t)D;
            if (model->is_double)
            {
                const double *s = (const double *)frame_scratch;
                for (long d = 0; d < D; d++)
                {
                    dst[d] = (float)s[d];
                }
            }
            else
            {
                memcpy(dst, frame_scratch, (size_t)D * sizeof(float));
            }
        }
        free(frame_scratch);
        CUDA_CHECK(cudaMemcpy(d_Q, host_query_fp32, query_matrix_bytes, cudaMemcpyHostToDevice));
        if (host_query_is_pinned)
        {
            cudaFreeHost(host_query_fp32);
        }
        else
        {
            free(host_query_fp32);
        }
        host_query_fp32 = NULL;
        host_query_is_pinned = 0;

        int threads = 256;
        int blocks = (int)((N_query + (threads / 32) - 1) / (threads / 32));
        compute_l2_norms_kernel<<<blocks, threads>>>(d_Q, (int)N_query, (int)D, d_Q_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int s = 0; s < 2; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

        CUDA_CHECK(cudaMalloc((void **)&d_P[s],
                              (size_t)B_q_max * (size_t)B_c_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&d_topk_dist_sq[s],
                              (size_t)B_q_max * (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&d_topk_indices[s],
                              (size_t)B_q_max * (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_out_dists[s],
                              (size_t)B_q_max * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_out_indices[s],
                              (size_t)B_q_max * (size_t)k * sizeof(int)));

        CUDA_CHECK(cudaMallocHost((void **)&host_out_dists[s],
                                  (size_t)B_q_max * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMallocHost((void **)&host_out_indices[s],
                                  (size_t)B_q_max * (size_t)k * sizeof(int)));
    }

    for (long q_start = 0; q_start < N_query; q_start += B_q_max)
    {
        int slot = (int)((q_start / B_q_max) % 2);
        int cur_Bq = (int)((q_start + B_q_max <= N_query) ? B_q_max : (N_query - q_start));

        /* If previous batch in this slot was launched, wait and harvest results */
        if (batch_lens[slot] > 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            long prev_q = batch_starts[slot];
            int  prev_Bq = batch_lens[slot];

            memcpy(results->indices + prev_q * (size_t)k,
                   host_out_indices[slot],
                   (size_t)prev_Bq * (size_t)k * sizeof(int));
            memcpy(results->distances + prev_q * (size_t)k,
                   host_out_dists[slot],
                   (size_t)prev_Bq * (size_t)k * sizeof(double));

            if (config->progress_mode)
            {
                long done = prev_q + prev_Bq;
                if (done % progress_step == 0 || done == N_query)
                {
                    double pct = 100.0 * (double)done / (double)N_query;
                    int bar_offset = 40 - (int)(pct * 0.4);
                    if (bar_offset < 0)
                    {
                        bar_offset = 0;
                    }
                    if (bar_offset > 40)
                    {
                        bar_offset = 40;
                    }
                    const char *bar = "========================================";
                    printf("\rSearching k-NN (GPU): [%-40s] %5.1f%% (%ld / %ld frames)",
                           &bar[bar_offset], pct, done, N_query);
                    fflush(stdout);
                }
            } // if (config->progress_mode)

            batch_lens[slot] = 0;
        } // if (batch_lens[slot] > 0)

        batch_starts[slot] = q_start;
        batch_lens[slot] = cur_Bq;

        /* Initialize top-k for active query tile in streams[slot] */
        {
            int total_elements = cur_Bq * k;
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            init_topk_kernel<<<blocks, threads, 0, streams[slot]>>>(
                d_topk_dist_sq[slot], d_topk_indices[slot], total_elements);
            CUDA_CHECK(cudaGetLastError());
        }

        const float *cur_d_Q = d_Q + (size_t)q_start * (size_t)D;

        for (long c_start = 0; c_start < N_cand; c_start += B_c_max)
        {
            int cur_Bc = (int)((c_start + B_c_max <= N_cand) ? B_c_max : (N_cand - c_start));
            const float *cur_d_C = d_C + (size_t)c_start * (size_t)D;

            /* Set cuBLAS stream */
            CUBLAS_CHECK(cublasSetStream(cublas_handle, streams[slot]));

            /* GEMM: P[cur_Bq x cur_Bc] = cur_d_Q[cur_Bq x D] * (cur_d_C[cur_Bc x D])^T */
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     cur_Bc, cur_Bq, (int)D,
                                     &alpha,
                                     cur_d_C, (int)D,
                                     cur_d_Q, (int)D,
                                     &beta,
                                     d_P[slot], cur_Bc));

            /* Warp-collaborative top-k update kernel */
            {
                int threads = BRUTEFORCE_BLOCK_THREADS;
                int warps_per_block = BRUTEFORCE_WARPS_PER_BLOCK;
                int blocks = (cur_Bq + warps_per_block - 1) / warps_per_block;
                update_topk_kernel<<<blocks, threads, 0, streams[slot]>>>(
                    d_Q_norms, d_C_norms, d_P[slot],
                    d_topk_dist_sq[slot], d_topk_indices[slot],
                    (int)q_start, cur_Bq,
                    (int)c_start, cur_Bc,
                    k,
                    config->min_temporal_sep,
                    config->past_only,
                    config->future_only,
                    rlim_sq,
                    is_cross_dataset);
                CUDA_CHECK(cudaGetLastError());
            }
        } // for (long c_start = 0; ...)

        /* Finalize distances and indices for this query batch */
        {
            int total_elements = cur_Bq * k;
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            finalize_topk_kernel<<<blocks, threads, 0, streams[slot]>>>(
                d_topk_dist_sq[slot], d_topk_indices[slot],
                d_out_dists[slot], d_out_indices[slot],
                total_elements);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Asynchronous copy of results to pinned host memory */
        CUDA_CHECK(cudaMemcpyAsync(
            host_out_indices[slot], d_out_indices[slot],
            (size_t)cur_Bq * (size_t)k * sizeof(int),
            cudaMemcpyDeviceToHost, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(
            host_out_dists[slot], d_out_dists[slot],
            (size_t)cur_Bq * (size_t)k * sizeof(double),
            cudaMemcpyDeviceToHost, streams[slot]));
    } // for (long q_start = 0; ...)

    /* Drain all in-flight batches remaining in streams */
    for (int slot = 0; slot < 2; slot++)
    {
        if (batch_lens[slot] > 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            long prev_q = batch_starts[slot];
            int  prev_Bq = batch_lens[slot];

            memcpy(results->indices + prev_q * (size_t)k,
                   host_out_indices[slot],
                   (size_t)prev_Bq * (size_t)k * sizeof(int));
            memcpy(results->distances + prev_q * (size_t)k,
                   host_out_dists[slot],
                   (size_t)prev_Bq * (size_t)k * sizeof(double));

            if (config->progress_mode)
            {
                long done = prev_q + prev_Bq;
                if (done % progress_step == 0 || done == N_query)
                {
                    double pct = 100.0 * (double)done / (double)N_query;
                    int bar_offset = 40 - (int)(pct * 0.4);
                    if (bar_offset < 0)
                    {
                        bar_offset = 0;
                    }
                    if (bar_offset > 40)
                    {
                        bar_offset = 40;
                    }
                    const char *bar = "========================================";
                    printf("\rSearching k-NN (GPU): [%-40s] %5.1f%% (%ld / %ld frames)",
                           &bar[bar_offset], pct, done, N_query);
                    fflush(stdout);
                }
            } // if (config->progress_mode)

            batch_lens[slot] = 0;
        } // if (batch_lens[slot] > 0)
    } // for (int slot = 0; slot < 2; slot++)

    if (config->progress_mode)
    {
        printf("\rSearching k-NN (GPU): [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N_query, N_query);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    telemetry->total_queries = (uint64_t)N_query;
    telemetry->framedist_calls = (uint64_t)N_query * (uint64_t)N_cand;
    telemetry->time_search_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    status = 0;

cleanup:
    for (int s = 0; s < 2; s++)
    {
        if (host_out_indices[s] != NULL)
        {
            cudaFreeHost(host_out_indices[s]);
        }
        if (host_out_dists[s] != NULL)
        {
            cudaFreeHost(host_out_dists[s]);
        }
        if (d_out_indices[s] != NULL)
        {
            cudaFree(d_out_indices[s]);
        }
        if (d_out_dists[s] != NULL)
        {
            cudaFree(d_out_dists[s]);
        }
        if (d_topk_indices[s] != NULL)
        {
            cudaFree(d_topk_indices[s]);
        }
        if (d_topk_dist_sq[s] != NULL)
        {
            cudaFree(d_topk_dist_sq[s]);
        }
        if (d_P[s] != NULL)
        {
            cudaFree(d_P[s]);
        }
        if (streams[s] != NULL)
        {
            cudaStreamDestroy(streams[s]);
        }
    }
    if (is_cross_dataset)
    {
        if (d_Q_norms != NULL)
        {
            cudaFree(d_Q_norms);
        }
        if (d_Q != NULL)
        {
            cudaFree(d_Q);
        }
    }
    if (d_C_norms != NULL)
    {
        cudaFree(d_C_norms);
    }
    if (d_C != NULL)
    {
        cudaFree(d_C);
    }
    if (host_cand_fp32 != NULL)
    {
        if (host_cand_is_pinned)
        {
            cudaFreeHost(host_cand_fp32);
        }
        else
        {
            free(host_cand_fp32);
        }
    }
    if (host_query_fp32 != NULL)
    {
        if (host_query_is_pinned)
        {
            cudaFreeHost(host_query_fp32);
        }
        else
        {
            free(host_query_fp32);
        }
    }
    if (cand_reader_opened)
    {
        knn_reader_close(&cand_reader);
    }
    if (query_reader_opened)
    {
        knn_reader_close(&query_reader);
    }
    if (cublas_handle != NULL)
    {
        cublasDestroy(cublas_handle);
    }

    return status;
}

int knn_cuda_locate_anchors(
    const KnnModel *model,
    const void     *host_queries,
    int             num_queries,
    int             is_double,
    int             m,
    int            *out_cl_indices,
    float          *out_cl_dists)
{
    if (model == NULL || host_queries == NULL ||
        out_cl_indices == NULL || out_cl_dists == NULL)
    {
        return -1;
    }

    int K = model->num_clusters;
    long D = model->frame_elements;
    int B = num_queries;

    if (K <= 0 || D <= 0 || B <= 0 || m <= 0)
    {
        return -1;
    }

    if (!gric_cuda_is_available())
    {
        return -1;
    }

    if (gric_cuda_init(0) != 0)
    {
        return -1;
    }

    int max_batch = (B < 2048) ? B : 2048;

    GpuAnchorStoreConfig store_cfg;
    store_cfg.max_clusters = K;
    store_cfg.dim = (int)D;
    store_cfg.device_id = 0;
    store_cfg.max_batch_size = max_batch;

    GpuAnchorStore *store = gpu_anchor_store_create(&store_cfg);
    if (store == NULL)
    {
        return -1;
    }

    /* Populate anchors into GPU store */
    for (int c = 0; c < K; c++)
    {
        if (model->clusters[c].anchor_data == NULL)
        {
            gpu_anchor_store_destroy(store);
            return -1;
        }
        gpu_anchor_store_append_anchor(store, model->clusters[c].anchor_data,
                                       model->is_double);
    }

    /* Process query batches */
    size_t elem_size = is_double ? sizeof(double) : sizeof(float);
    int ret = 0;

    for (int b_start = 0; b_start < B; b_start += max_batch)
    {
        int cur_b = (b_start + max_batch <= B) ? max_batch : (B - b_start);
        const void *q_ptr = (const char *)host_queries + (size_t)b_start * (size_t)D * elem_size;
        int *idx_ptr = out_cl_indices + (size_t)b_start * (size_t)m;
        float *dist_ptr = out_cl_dists + (size_t)b_start * (size_t)m;

        if (gpu_anchor_store_find_top_m(store, q_ptr, cur_b, is_double, m,
                                        idx_ptr, dist_ptr) != 0)
        {
            ret = -1;
            break;
        }
    }

    gpu_anchor_store_destroy(store);
    return ret;
}


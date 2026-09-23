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

/**
 * init_topk_kernel() - Initialize top-k distance and index arrays.
 * @topk_dist_sq: Array [num_queries x k] to initialize with +inf.
 * @topk_indices: Array [num_queries x k] to initialize with -1.
 * @num_queries: Number of queries in batch.
 * @k:           Number of neighbors.
 */
static __global__ void init_topk_kernel(
    float *__restrict__ topk_dist_sq,
    int   *__restrict__ topk_indices,
    int                 num_queries,
    int                 k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries)
    {
        int offset = idx * k;
        for (int i = 0; i < k; i++)
        {
            topk_dist_sq[offset + i] = 1e38f;
            topk_indices[offset + i] = -1;
        }
    }
}

/**
 * update_topk_kernel() - Fused metric constraint filter and top-k insertion.
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
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= B_q)
    {
        return;
    }

    float local_dist[MAX_STATIC_K];
    int   local_id[MAX_STATIC_K];

    int q_offset = q * k;
    for (int i = 0; i < k; i++)
    {
        local_dist[i] = d_topk_dist_sq[q_offset + i];
        local_id[i]   = d_topk_indices[q_offset + i];
    }

    float tau = local_dist[k - 1];
    int g_q = q_start + q;
    float q_norm = d_Q_norms[g_q];
    const float *p_row = d_P + (size_t)q * (size_t)B_c;

    for (int c = 0; c < B_c; c++)
    {
        int g_c = cand_start + c;

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

        float c_norm = d_C_norms[g_c];
        float dot = p_row[c];
        float dist_sq = q_norm + c_norm - 2.0f * dot;
        if (dist_sq < 0.0f)
        {
            dist_sq = 0.0f;
        }

        if (rlim_sq > 0.0f && dist_sq > rlim_sq)
        {
            continue;
        }

        if (dist_sq < tau)
        {
            int pos = k - 1;
            while (pos > 0 && local_dist[pos - 1] > dist_sq)
            {
                local_dist[pos] = local_dist[pos - 1];
                local_id[pos]   = local_id[pos - 1];
                pos--;
            }
            local_dist[pos] = dist_sq;
            local_id[pos]   = g_c;
            tau = local_dist[k - 1];
        }
    }

    for (int i = 0; i < k; i++)
    {
        d_topk_dist_sq[q_offset + i] = local_dist[i];
        d_topk_indices[q_offset + i] = local_id[i];
    }
}

/**
 * finalize_topk_kernel() - Compute sqrt of distances and format output arrays.
 * @d_topk_dist_sq: Input squared distances [num_queries x k].
 * @d_topk_indices: Input candidate IDs [num_queries x k].
 * @d_out_dists:    Output Euclidean distances [num_queries x k] in double precision.
 * @d_out_indices:  Output candidate IDs [num_queries x k].
 * @num_queries:    Number of queries.
 * @k:              Number of neighbors.
 */
static __global__ void finalize_topk_kernel(
    const float *__restrict__ d_topk_dist_sq,
    const int   *__restrict__ d_topk_indices,
    double      *__restrict__ d_out_dists,
    int         *__restrict__ d_out_indices,
    int                       num_queries,
    int                       k)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < num_queries)
    {
        int offset = q * k;
        for (int i = 0; i < k; i++)
        {
            int id = d_topk_indices[offset + i];
            d_out_indices[offset + i] = id;
            if (id >= 0)
            {
                d_out_dists[offset + i] = sqrt((double)d_topk_dist_sq[offset + i]);
            }
            else
            {
                d_out_dists[offset + i] = -1.0;
            }
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

    float *d_C = NULL;
    float *d_C_norms = NULL;
    float *d_Q = NULL;
    float *d_Q_norms = NULL;
    float *d_P = NULL;
    float *d_topk_dist_sq = NULL;
    int   *d_topk_indices = NULL;
    double *d_out_dists = NULL;
    int    *d_out_indices = NULL;
    float  *host_cand_fp32 = NULL;
    float  *host_query_fp32 = NULL;
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
            host_cand_fp32 = (float *)malloc(cand_matrix_bytes);
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
            free(host_cand_fp32);
            host_cand_fp32 = NULL;
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
        host_cand_fp32 = (float *)malloc(cand_matrix_bytes);
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
        free(host_cand_fp32);
        host_cand_fp32 = NULL;
    }

    /* Compute candidate vector squared norms */
    {
        int threads = 256;
        int blocks = (int)((N_cand + threads - 1) / threads);
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

        host_query_fp32 = (float *)malloc(query_matrix_bytes);
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
        free(host_query_fp32);
        host_query_fp32 = NULL;

        int threads = 256;
        int blocks = (int)((N_query + threads - 1) / threads);
        compute_l2_norms_kernel<<<blocks, threads>>>(d_Q, (int)N_query, (int)D, d_Q_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMalloc((void **)&d_P, (size_t)B_q_max * (size_t)B_c_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_topk_dist_sq, (size_t)B_q_max * (size_t)k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_topk_indices, (size_t)B_q_max * (size_t)k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_dists, (size_t)B_q_max * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_indices, (size_t)B_q_max * (size_t)k * sizeof(int)));

    for (long q_start = 0; q_start < N_query; q_start += B_q_max)
    {
        int cur_Bq = (int)((q_start + B_q_max <= N_query) ? B_q_max : (N_query - q_start));

        /* Initialize top-k for active query tile */
        {
            int threads = 128;
            int blocks = (cur_Bq + threads - 1) / threads;
            init_topk_kernel<<<blocks, threads>>>(d_topk_dist_sq, d_topk_indices, cur_Bq, k);
            CUDA_CHECK(cudaGetLastError());
        }

        const float *cur_d_Q = d_Q + (size_t)q_start * (size_t)D;

        for (long c_start = 0; c_start < N_cand; c_start += B_c_max)
        {
            int cur_Bc = (int)((c_start + B_c_max <= N_cand) ? B_c_max : (N_cand - c_start));
            const float *cur_d_C = d_C + (size_t)c_start * (size_t)D;

            /* GEMM: P[cur_Bq x cur_Bc] = cur_d_Q[cur_Bq x D] * (cur_d_C[cur_Bc x D])^T */
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     cur_Bc, cur_Bq, (int)D,
                                     &alpha,
                                     cur_d_C, (int)D,
                                     cur_d_Q, (int)D,
                                     &beta,
                                     d_P, cur_Bc));

            /* Fused update kernel */
            {
                int threads = 128;
                int blocks = (cur_Bq + threads - 1) / threads;
                update_topk_kernel<<<blocks, threads>>>(
                    d_Q_norms, d_C_norms, d_P,
                    d_topk_dist_sq, d_topk_indices,
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
            int threads = 128;
            int blocks = (cur_Bq + threads - 1) / threads;
            finalize_topk_kernel<<<blocks, threads>>>(
                d_topk_dist_sq, d_topk_indices,
                d_out_dists, d_out_indices,
                cur_Bq, k);
            CUDA_CHECK(cudaGetLastError());
        }

        /* Copy batch results back to host */
        CUDA_CHECK(cudaMemcpy(results->indices + q_start * k,
                              d_out_indices,
                              (size_t)cur_Bq * (size_t)k * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(results->distances + q_start * k,
                              d_out_dists,
                              (size_t)cur_Bq * (size_t)k * sizeof(double),
                              cudaMemcpyDeviceToHost));

        if (config->progress_mode && (q_start + cur_Bq) % progress_step == 0)
        {
            double pct = 100.0 * (double)(q_start + cur_Bq) / (double)N_query;
            int bar_offset = 40 - (int)(pct * 0.4);
            if (bar_offset < 0) bar_offset = 0;
            if (bar_offset > 40) bar_offset = 40;
            const char *bar = "========================================";
            printf("\rSearching k-NN (GPU): [%-40s] %5.1f%% (%ld / %ld frames)",
                   &bar[bar_offset], pct, (long)(q_start + cur_Bq), N_query);
            fflush(stdout);
        }
    } // for (long q_start = 0; ...)

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
    if (d_out_indices != NULL) cudaFree(d_out_indices);
    if (d_out_dists != NULL) cudaFree(d_out_dists);
    if (d_topk_indices != NULL) cudaFree(d_topk_indices);
    if (d_topk_dist_sq != NULL) cudaFree(d_topk_dist_sq);
    if (d_P != NULL) cudaFree(d_P);
    if (is_cross_dataset)
    {
        if (d_Q_norms != NULL) cudaFree(d_Q_norms);
        if (d_Q != NULL) cudaFree(d_Q);
    }
    if (d_C_norms != NULL) cudaFree(d_C_norms);
    if (d_C != NULL) cudaFree(d_C);
    if (host_cand_fp32 != NULL) free(host_cand_fp32);
    if (host_query_fp32 != NULL) free(host_query_fp32);
    if (cand_reader_opened) knn_reader_close(&cand_reader);
    if (query_reader_opened) knn_reader_close(&query_reader);
    if (cublas_handle != NULL) cublasDestroy(cublas_handle);

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


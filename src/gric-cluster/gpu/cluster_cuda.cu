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
    float *host_anchors = NULL;
    float *d_A = NULL;
    float *d_A_norms = NULL;
    float *d_F = NULL;
    float *d_F_norms = NULL;
    float *d_P = NULL;
    int   *d_best_cl = NULL;
    float *d_best_dist = NULL;
    float *host_F = NULL;
    int   *host_best_cl = NULL;
    float *host_best_dist = NULL;
    long frames_reassigned = 0;
    double p2_ms = 0.0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int B_max = (config->optim.gpu_micro_batch_size > 0) ?
                config->optim.gpu_micro_batch_size : DEFAULT_FRAME_BATCH;

    if (B_max > N)
    {
        B_max = (int)N;
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

    host_F = (float *)malloc((size_t)B_max * (size_t)D * sizeof(float));
    host_best_cl = (int *)malloc((size_t)B_max * sizeof(int));
    host_best_dist = (float *)malloc((size_t)B_max * sizeof(float));

    if (host_F == NULL || host_best_cl == NULL || host_best_dist == NULL)
    {
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_A, (size_t)K * (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_A_norms, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, host_anchors, (size_t)K * (size_t)D * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int blocks = (K + threads - 1) / threads;
        compute_norms_kernel<<<blocks, threads>>>(d_A, K, (int)D, d_A_norms);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMalloc((void **)&d_F, (size_t)B_max * (size_t)D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_F_norms, (size_t)B_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_P, (size_t)B_max * (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_best_cl, (size_t)B_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_best_dist, (size_t)B_max * sizeof(float)));

    for (long t_start = 0; t_start < N; t_start += B_max)
    {
        int cur_B = (int)((t_start + B_max <= N) ? B_max : (N - t_start));

        /* Ingest cur_B frames from reader */
        for (int i = 0; i < cur_B; i++)
        {
            long t = t_start + i;
            Frame *fr = getframe_at(t);
            if (fr == NULL)
            {
                fprintf(stderr, "Error: Could not retrieve frame %ld\n", t);
                goto cleanup;
            }
            float *dst = host_F + (size_t)i * (size_t)D;
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
        }

        CUDA_CHECK(cudaMemcpy(d_F, host_F, (size_t)cur_B * (size_t)D * sizeof(float),
                              cudaMemcpyHostToDevice));

        {
            int threads = 128;
            int blocks = (cur_B + threads - 1) / threads;
            compute_norms_kernel<<<blocks, threads>>>(d_F, cur_B, (int)D, d_F_norms);
            CUDA_CHECK(cudaGetLastError());
        }

        /* GEMM: d_P[cur_B x K] = d_F[cur_B x D] * (d_A[K x D])^T */
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K, cur_B, (int)D,
                                 &alpha,
                                 d_A, (int)D,
                                 d_F, (int)D,
                                 &beta,
                                 d_P, K));

        {
            int threads = 128;
            int blocks = (cur_B + threads - 1) / threads;
            argmin_distance_kernel<<<blocks, threads>>>(
                d_F_norms, d_A_norms, d_P,
                d_best_cl, d_best_dist,
                cur_B, K);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemcpy(host_best_cl, d_best_cl, (size_t)cur_B * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_best_dist, d_best_dist, (size_t)cur_B * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < cur_B; i++)
        {
            long t = t_start + i;
            int best_cl = host_best_cl[i];
            float best_d = host_best_dist[i];

            if (best_cl >= 0 && best_cl != state->assignments[t])
            {
                frames_reassigned++;
                state->assignments[t] = best_cl;
            }

            if (state->frame_infos != NULL)
            {
                state->frame_infos[t].assignment = best_cl;
                if (state->frame_infos[t].cluster_indices == NULL)
                {
                    state->frame_infos[t].cluster_indices = (int *)malloc(sizeof(int));
                    state->frame_infos[t].distances = (double *)malloc(sizeof(double));
                    state->frame_infos[t].num_dists = 1;
                }
                if (state->frame_infos[t].cluster_indices != NULL &&
                    state->frame_infos[t].distances != NULL)
                {
                    state->frame_infos[t].cluster_indices[0] = best_cl;
                    state->frame_infos[t].distances[0] = (double)best_d;
                }
            }
        }
    } // for (long t_start = 0; ...)

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

    state->telemetry.time_pass2 = p2_ms;
    state->telemetry.pass2_frames_reassigned = (uint64_t)frames_reassigned;
    state->telemetry.pass2_dist_evals = (uint64_t)N * (uint64_t)K;
    state->telemetry.pass2_dist_pruned = 0;

    printf("\nSecond Pass (Nearest Anchor Reallocation - GPU):\n");
    printf("  Frames reassigned  : %ld / %ld (%.2f%%)\n",
           frames_reassigned, N,
           (N > 0) ? (100.0 * (double)frames_reassigned / (double)N) : 0.0);
    printf("  GPU Pass 2 Time    : %.2f ms\n", p2_ms);

cleanup:
    if (d_best_dist != NULL) cudaFree(d_best_dist);
    if (d_best_cl != NULL) cudaFree(d_best_cl);
    if (d_P != NULL) cudaFree(d_P);
    if (d_F_norms != NULL) cudaFree(d_F_norms);
    if (d_F != NULL) cudaFree(d_F);
    if (d_A_norms != NULL) cudaFree(d_A_norms);
    if (d_A != NULL) cudaFree(d_A);
    if (host_best_dist != NULL) free(host_best_dist);
    if (host_best_cl != NULL) free(host_best_cl);
    if (host_F != NULL) free(host_F);
    if (host_anchors != NULL) free(host_anchors);
    if (cublas_handle != NULL) cublasDestroy(cublas_handle);

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

        state->clusters[0].anchor = *fr0;
        fr0->data = NULL;
        state->clusters[0].id = 0;
        state->clusters[0].prob = 1.0;
        state->clusters[0].anchor_sq8 = NULL;
        state->clusters[0].anchor_sq16 = NULL;
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
                    state->clusters[new_k].anchor = *fr;
                    fr->data = NULL;
                    state->clusters[new_k].id = new_k;
                    state->clusters[new_k].prob = 1.0;
                    state->clusters[new_k].anchor_sq8 = NULL;
                    state->clusters[new_k].anchor_sq16 = NULL;
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


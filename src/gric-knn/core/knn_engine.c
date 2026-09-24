/**
 * @file knn_engine.c
 * @brief High-performance metric-pruned k-NN solver engine coordinator.
 *
 * Implements the top-level execution coordinator for the k-NN solver. Functions in
 * this file manage memory allocation for output results, set up thread-local scratch
 * heaps and file readers, invoke GPU kernels or OpenMP multi-threaded CPU routines,
 * sort top-k candidates, and reduce thread-local telemetry into global performance metrics.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_engine.h"
#include "knn_engine_internal.h"
#include "knn_pruning.h"
#include "knn_cluster_search.h"
#include "knn_cross_dataset.h"
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "knn_cuda.h"
#endif

/**
 * knn_search_validate_and_alloc() - Validate inputs and allocate search heap structures.
 * @config:       Active k-NN search configuration.
 * @model:        Pre-loaded cluster model.
 * @results:      Results structure receiving allocated indices and distances buffers.
 * @out_heaps:    Output pointer receiving array of per-query KnnMaxHeap structures.
 * @out_is_cross: Output boolean flag (1 for cross-dataset, 0 for single-dataset).
 * @out_n_query:  Output total number of queries to search.
 * @out_n_cand:   Output total number of candidates in candidate pool.
 * @out_q_w:      Output query frame width in pixels.
 * @out_q_h:      Output query frame height in pixels.
 *
 * Purpose & Context ("What is this used for?"):
 * Validates configuration parameters and dataset dimensions, checks query frame size
 * consistency in cross-dataset mode, allocates result storage for N_query * k items,
 * and initializes bounded max-heaps with search_k capacity for every query.
 *
 * Return: 0 on success, -1 on failure.
 */
static int knn_search_validate_and_alloc(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnMaxHeap     **out_heaps,
    int             *out_is_cross,
    long            *out_n_query,
    long            *out_n_cand,
    long            *out_q_w,
    long            *out_q_h)
{
    int is_cross_dataset = (config->query_data_path != NULL) ? 1 : 0;
    long N_query = model->total_dataset_frames;
    long N_cand = model->total_dataset_frames;
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

    int k = config->k;

    results->num_queries = N_query;
    results->indices = (int *)malloc((size_t)N_query * (size_t)k * sizeof(int));
    results->distances = (double *)malloc((size_t)N_query * (size_t)k * sizeof(double));

    if (results->indices == NULL || results->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for results buffer\n");
        return -1;
    }

    KnnMaxHeap *all_heaps = NULL;
    if (posix_memalign((void **)&all_heaps, 32, (size_t)N_query * sizeof(KnnMaxHeap)) != 0)
    {
        fprintf(stderr, "Error: Memory allocation failed for heaps array\n");
        knn_results_free(results);
        return -1;
    }

    int search_k = (config->ef_search > 0) ? config->ef_search :
                   (config->approx_mode ? 2 * k : k);
    if (search_k < k)
    {
        search_k = k;
    }

    for (long i = 0; i < N_query; i++)
    {
        if (knn_heap_init(&all_heaps[i], search_k) != 0)
        {
            fprintf(stderr, "Error: Failed to init heap %ld\n", i);
            for (long j = 0; j < i; j++)
            {
                knn_heap_free(&all_heaps[j]);
            }
            free(all_heaps);
            knn_results_free(results);
            return -1;
        }
    }

    *out_heaps = all_heaps;
    *out_is_cross = is_cross_dataset;
    *out_n_query = N_query;
    *out_n_cand = N_cand;
    *out_q_w = q_w;
    *out_q_h = q_h;
    return 0;
}

/**
 * knn_search_open_readers() - Open candidate and query frame readers.
 * @config:              Active search configuration.
 * @model:               Pre-loaded cluster model.
 * @results:             Results structure to free if opening readers fails.
 * @is_cross_dataset:    Non-zero if query dataset is distinct.
 * @N_cand:              Number of candidate frames.
 * @N_query:             Number of query frames.
 * @q_w:                 Query frame width.
 * @q_h:                 Query frame height.
 * @master_cand_reader:  Output candidate reader handle.
 * @master_query_reader: Output query reader handle.
 *
 * Purpose & Context ("What is this used for?"):
 * Prepares dataset access handles (memory buffer or disk file) for the search phase.
 * Opens the master reader structures that worker threads clone per-thread.
 *
 * Return: 0 on success, -1 on failure.
 */
static int knn_search_open_readers(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    int              is_cross_dataset,
    long             N_cand,
    long             N_query,
    long             q_w,
    long             q_h,
    KnnFrameReader  *master_cand_reader,
    KnnFrameReader  *master_query_reader)
{
    if (config->memory_data != NULL)
    {
        if (knn_reader_open_memory(master_cand_reader, config->memory_data, N_cand,
                                   model->frame_elements, model->is_double) != 0)
        {
            knn_results_free(results);
            return -1;
        }
    }
    else if (knn_reader_open(master_cand_reader, config->input_data_path, N_cand,
                             model->frame_width, model->frame_height, model->is_double) != 0)
    {
        knn_results_free(results);
        return -1;
    }

    if (is_cross_dataset)
    {
        if (knn_reader_open(master_query_reader, config->query_data_path, N_query,
                            q_w, q_h, model->is_double) != 0)
        {
            knn_reader_close(master_cand_reader);
            knn_results_free(results);
            return -1;
        }
    }

    return 0;
}

/**
 * knn_search_extract_and_finalize() - Extract sorted neighbors, close readers, and set time.
 * @config:              Active search configuration.
 * @results:             Output results structure.
 * @telemetry:           Output telemetry structure.
 * @all_heaps:           Array of per-query max-heaps.
 * @N_query:             Number of queries.
 * @is_cross_dataset:    Cross-dataset flag.
 * @bucket_locks:        OpenMP bucket locks array (single-dataset only).
 * @master_cand_reader:  Master candidate reader to close.
 * @master_query_reader: Master query reader to close if cross-dataset.
 * @start_time:          Start timestamp.
 *
 * Purpose & Context ("What is this used for?"):
 * Drains all per-query max-heaps in parallel to extract sorted nearest-neighbor lists,
 * frees heap allocations, destroys OpenMP synchronization locks, closes file readers,
 * and records total wall-clock search duration into telemetry.
 */
static void knn_search_extract_and_finalize(
    const KnnConfig *config,
    KnnResults      *results,
    KnnTelemetry    *telemetry,
    KnnMaxHeap      *all_heaps,
    long             N_query,
    int              is_cross_dataset,
#ifdef _OPENMP
    omp_lock_t      *bucket_locks,
#endif
    KnnFrameReader  *master_cand_reader,
    KnnFrameReader  *master_query_reader,
    struct timespec  start_time)
{
    int k = config->k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < N_query; i++)
    {
        knn_heap_extract_sorted(&all_heaps[i], &results->indices[i * k],
                                &results->distances[i * k], k);
        knn_heap_free(&all_heaps[i]);
    }
    free(all_heaps);

#ifdef _OPENMP
    if (!is_cross_dataset)
    {
        for (int b = 0; b < KNN_NUM_BUCKET_LOCKS; b++)
        {
            omp_destroy_lock(&bucket_locks[b]);
        }
    }
#endif

    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    if (is_cross_dataset)
    {
        knn_reader_close(master_query_reader);
    }
    knn_reader_close(master_cand_reader);

    if (config->progress_mode)
    {
        printf("\rSearching k-NN: [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N_query, N_query);
        fflush(stdout);
    }

    telemetry->time_search_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
}

/**
 * knn_run_search() - Multi-threaded driver executing k-NN search across all frames
 * @config:    Active KnnConfig specifying query mode, thread count, and pruning flags.
 * @model:     Active KnnModel containing cluster anchors, graphs, and quantization codes.
 * @results:   Output KnnResults structure to populate with top-k indices and distances.
 * @telemetry: Output aggregated KnnTelemetry structure tracking evaluations and prune counts.
 *
 * Coordinates execution of exact or approximate k-nearest neighbor search across all
 * requested query frames. The algorithm proceeds through the following phases:
 *
 * 1. Hardware Dispatch:
 *    If GPU acceleration is requested and CUDA support is compiled in, dispatches to
 *    the CUDA k-NN pipeline via knn_cuda_run_search(). If GPU execution fails or is
 *    unavailable, gracefully falls back to multi-threaded CPU execution.
 *
 * 2. Result Buffer Allocation:
 *    Allocates contiguous memory for query neighbor indices (int) and distances (double)
 *    sized to total_queries * k elements.
 *
 * 3. Thread-Local Context Setup:
 *    Spawns OpenMP threads, allocating per-thread scratch structures (bounded max-heaps,
 *    visited bitsets, cluster candidate arrays, and independent file reader handles).
 *
 * 4. Query Frame Processing:
 *    Distributes query frames across threads. For each query:
 *    - In cross-dataset mode, reads query vector from query dataset and routes through
 *      cluster graph via knn_search_cross_dataset_frame().
 *    - In single-dataset mode, calls knn_search_single_frame() utilizing intra-cluster
 *      and inter-cluster metric pruning, pivot bounds, and SIMD/quantized distance filters.
 *    - Extracts sorted top-k nearest neighbors from thread max-heap into output arrays.
 *
 * 5. Telemetry & Cleanup:
 *    Reduces per-thread telemetry metrics (pruned clusters, distance calculations, timing)
 *    into the global telemetry structure and releases thread-local scratch memory.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry)
{
    if (config == NULL || model == NULL || results == NULL || telemetry == NULL)
    {
        return -1;
    }

#ifdef USE_CUDA
    if (config->use_gpu)
    {
        if (knn_cuda_is_available())
        {
            int rc = knn_cuda_run_search(config, model, results, telemetry);
            if (rc == 0)
            {
                return 0;
            }
            fprintf(stderr, "Warning: GPU search failed, falling back to CPU engine.\n");
        }
        else
        {
            fprintf(stderr, "Warning: CUDA GPU requested but no available device found. "
                            "Falling back to CPU engine.\n");
        }
    }
#else
    if (config->use_gpu)
    {
        fprintf(stderr, "Warning: GPU acceleration requested (--gpu), but gric-knn was built "
                        "without CUDA support (ENABLE_CUDA=OFF). Running on CPU.\n");
    }
#endif

    memset(telemetry, 0, sizeof(KnnTelemetry));

    int         is_cross_dataset = 0;
    long        N_query = 0;
    long        N_cand = 0;
    long        q_w = 0;
    long        q_h = 0;
    KnnMaxHeap *all_heaps = NULL;

    if (knn_search_validate_and_alloc(config, model, results, &all_heaps,
                                      &is_cross_dataset, &N_query, &N_cand,
                                      &q_w, &q_h) != 0)
    {
        return -1;
    }

#ifdef _OPENMP
    omp_lock_t bucket_locks[KNN_NUM_BUCKET_LOCKS];
    if (!is_cross_dataset)
    {
        for (int b = 0; b < KNN_NUM_BUCKET_LOCKS; b++)
        {
            omp_init_lock(&bucket_locks[b]);
        }
    }
#endif

    KnnFrameReader master_cand_reader;
    KnnFrameReader master_query_reader;

    if (knn_search_open_readers(config, model, results, is_cross_dataset,
                                N_cand, N_query, q_w, q_h,
                                &master_cand_reader, &master_query_reader) != 0)
    {
        for (long j = 0; j < N_query; j++)
        {
            knn_heap_free(&all_heaps[j]);
        }
        free(all_heaps);
        return -1;
    }

    int nthreads = config->nthreads;
#ifdef _OPENMP
    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
    }
    else
    {
        nthreads = omp_get_max_threads();
    }
#else
    nthreads = 1;
#endif

    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long progress_step = N_query / 100;
    if (progress_step < 1)
    {
        progress_step = 1;
    }

    uint64_t global_telem_calls = 0;
    uint64_t global_telem_l1 = 0;
    uint64_t global_telem_l2 = 0;
    uint64_t global_telem_l3 = 0;
    uint64_t global_telem_temp = 0;
    uint64_t global_telem_recip = 0;
    uint64_t global_telem_graph_seeds = 0;
    uint64_t global_telem_graph_edges = 0;
    uint64_t global_telem_multi_pivot = 0;
    uint64_t global_telem_angular = 0;
    uint64_t global_telem_containment = 0;
    uint64_t global_telem_cand = 0;
    uint64_t global_telem_traj = 0;
    uint64_t global_telem_rejected = 0;
    uint64_t global_telem_sq8_evals = 0;
    uint64_t global_telem_sq8_pruned = 0;
    uint64_t global_telem_sq8_graph_pruned = 0;
    uint64_t global_telem_sq16_evals = 0;
    uint64_t global_telem_sq16_pruned = 0;
    uint64_t global_telem_sq16_graph_pruned = 0;
    uint64_t global_telem_eq16_evals = 0;
    uint64_t global_telem_eq16_pruned = 0;
    uint64_t global_telem_eq16_graph_pruned = 0;
    uint64_t global_telem_rq8_evals = 0;
    uint64_t global_telem_rq8_pruned = 0;
    uint64_t global_telem_rq8_graph_pruned = 0;
    uint64_t global_telem_pq_evals = 0;
    uint64_t global_telem_pq_pruned = 0;
    uint64_t global_telem_rabitq_evals = 0;
    uint64_t global_telem_rabitq_pruned = 0;
    uint64_t global_telem_graph_clusters = 0;
    uint64_t global_telem_two_hop_evals = 0;
    uint64_t global_telem_two_hop_pruned = 0;
    uint64_t global_telem_two_hop_injected = 0;
    uint64_t global_telem_memo_hits = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:global_telem_calls, global_telem_l1,                   \
                                 global_telem_l2, global_telem_l3, global_telem_temp,   \
                                 global_telem_recip, global_telem_graph_seeds,          \
                                 global_telem_graph_edges, global_telem_multi_pivot,    \
                                 global_telem_angular, global_telem_containment,        \
                                 global_telem_cand, global_telem_traj,                  \
                                 global_telem_rejected,                                 \
                                 global_telem_sq8_evals, global_telem_sq8_pruned,      \
                                 global_telem_sq8_graph_pruned,                         \
                                 global_telem_sq16_evals, global_telem_sq16_pruned,    \
                                 global_telem_sq16_graph_pruned,                        \
                                 global_telem_eq16_evals, global_telem_eq16_pruned,    \
                                 global_telem_eq16_graph_pruned,                        \
                                 global_telem_rq8_evals, global_telem_rq8_pruned,      \
                                 global_telem_rq8_graph_pruned,                         \
                                 global_telem_pq_evals, global_telem_pq_pruned,        \
                                 global_telem_rabitq_evals, global_telem_rabitq_pruned, \
                                 global_telem_graph_clusters,                           \
                                 global_telem_two_hop_evals,                            \
                                 global_telem_two_hop_pruned,                           \
                                 global_telem_two_hop_injected,                         \
                                 global_telem_memo_hits)
#endif
    {
        KnnFrameReader thread_cand_reader;
        KnnFrameReader thread_query_reader;

        knn_reader_clone_thread(&master_cand_reader, &thread_cand_reader);
        if (is_cross_dataset)
        {
            knn_reader_clone_thread(&master_query_reader, &thread_query_reader);
        }

        size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
        void *query_buffer =
            malloc((size_t)model->frame_elements * elem_size);
        void *cand_buffer =
            malloc((size_t)4 * model->frame_elements * elem_size);
        uint8_t *query_sq8 =
            config->use_sq8 ? (uint8_t *)malloc((size_t)model->frame_elements) : NULL;
        int16_t *query_sq16 =
            config->use_sq16 ? (int16_t *)malloc((size_t)model->frame_elements *
                                                 sizeof(int16_t)) : NULL;
        int16_t *query_eq16 =
            config->use_eq16 ? (int16_t *)malloc((size_t)model->frame_elements *
                                                 sizeof(int16_t)) : NULL;
        float *query_eq16_adc =
            (config->use_eq16 && config->use_eq16_adc) ?
                (float *)malloc((size_t)model->frame_elements * sizeof(float)) : NULL;
        int16_t *query_rq8 =
            config->use_rq8 ? (int16_t *)malloc((size_t)model->frame_elements *
                                                sizeof(int16_t)) : NULL;
        float *query_rq8_adc =
            (config->use_rq8 && config->use_rq8_adc) ?
                (float *)malloc((size_t)model->frame_elements * sizeof(float)) : NULL;
        uint8_t *query_pq_lut =
            (config->use_pq && model->pq_codebook != NULL) ?
                (uint8_t *)malloc((size_t)model->pq_codebook->m *
                                  (size_t)model->pq_codebook->k_centroids) : NULL;
        float *query_rabitq_rot =
            (config->use_rabitq && model->rabitq_dataset_buffer != NULL) ?
                (float *)malloc((size_t)model->rabitq_params.dim_pad * sizeof(float)) : NULL;
        int rabitq_num_nibbles = (model->rabitq_params.bits == 2) ?
            (int)(model->rabitq_params.dim_pad / 2) :
            (int)(model->rabitq_params.dim_pad / 4);
        int8_t *query_rabitq_lut_buf =
            (config->use_rabitq && model->rabitq_dataset_buffer != NULL) ?
                (int8_t *)malloc((size_t)rabitq_num_nibbles * 16 *
                                 sizeof(int8_t)) : NULL;
        double *anchor_dists =
            (double *)malloc((size_t)model->num_clusters * sizeof(double));
        ClusterScore *scores_buf =
            (ClusterScore *)malloc((size_t)model->num_clusters * sizeof(ClusterScore));

        KnnClusterGraphScratch graph_scratch;
        memset(&graph_scratch, 0, sizeof(KnnClusterGraphScratch));
        if (config->use_cluster_graph && model->cluster_graph_adj != NULL)
        {
            graph_scratch.pq = (ClusterPqNode *)malloc(
                (size_t)model->num_clusters * sizeof(ClusterPqNode)
            );
            graph_scratch.enqueued_tags = (uint32_t *)calloc(
                (size_t)model->num_clusters, sizeof(uint32_t)
            );
            graph_scratch.enqueued_epoch = 1;
            graph_scratch.anchor_dists = anchor_dists;
            graph_scratch.anchor_is_sq16 = (uint8_t *)malloc(
                (size_t)model->num_clusters * sizeof(uint8_t)
            );
        }

        KnnTelemetry thread_telem;
        memset(&thread_telem, 0, sizeof(KnnTelemetry));
        KnnTrajectoryTracker thread_tracker;
        memset(&thread_tracker, 0, sizeof(KnnTrajectoryTracker));
        thread_tracker.prev_query_id = -999;
        thread_tracker.prev_cluster_id = -1;
        thread_tracker.prev_best_seed = -1;
        thread_tracker.prev_best_dist = 1e20;

        KnnVisitedTracker visited;
        visited.tags = (uint32_t *)calloc((size_t)N_cand, sizeof(uint32_t));
        visited.epoch = 1;
        visited.query_sq8 = query_sq8;
        visited.query_sq16 = query_sq16;
        visited.query_eq16 = query_eq16;
        visited.query_eq16_adc = query_eq16_adc;
        visited.query_rq8 = query_rq8;
        visited.query_rq8_adc = query_rq8_adc;
        visited.query_rq8_clipped = 0;
        visited.query_pq_lut = query_pq_lut;
        memset(&visited.query_pq_table, 0, sizeof(PQLookupTable));
        if (query_pq_lut != NULL && model->pq_codebook != NULL)
        {
            visited.query_pq_table.lut_u8 = query_pq_lut;
            visited.query_pq_table.m = model->pq_codebook->m;
            visited.query_pq_table.k_centroids = model->pq_codebook->k_centroids;
        }
        visited.query_rabitq_rot = query_rabitq_rot;
        visited.query_rabitq_norm = 0.0f;
        memset(&visited.query_rabitq_lut, 0, sizeof(RaBitQLookupTable));
        if (query_rabitq_lut_buf != NULL)
        {
            visited.query_rabitq_lut.lut_i8 = query_rabitq_lut_buf;
        }
        visited.rep_tags = NULL;
        visited.rep_dists = NULL;
        if (config->use_memo && model->frame_to_unique_map != NULL)
        {
            visited.rep_tags = (uint32_t *)calloc((size_t)N_cand, sizeof(uint32_t));
            visited.rep_dists = (float *)malloc((size_t)N_cand * sizeof(float));
        }

        int16_t *sq16_scratch = NULL;
        KnnSparseLruCache *sq16_lru = NULL;
        if (config->use_sq16 && config->use_sq16_sparse)
        {
            if (config->use_sq16_sparse_lru)
            {
                sq16_lru = (KnnSparseLruCache *)calloc(1, sizeof(KnnSparseLruCache));
                if (sq16_lru != NULL)
                {
                    size_t blk_elems = (size_t)model->frame_elements * SQ16_FASTSCAN_BLOCK_SIZE;
                    sq16_lru->buffer_pool = (int16_t *)malloc(
                        (size_t)KNN_SPARSE_LRU_BLOCKS * blk_elems * sizeof(int16_t));
                    for (int s = 0; s < KNN_SPARSE_LRU_BLOCKS; s++)
                    {
                        sq16_lru->blocks[s].cl_id = -1;
                        sq16_lru->blocks[s].block_id = -1;
                        sq16_lru->blocks[s].access_seq = 0;
                        if (sq16_lru->buffer_pool != NULL)
                        {
                            sq16_lru->blocks[s].data = sq16_lru->buffer_pool + s * blk_elems;
                        }
                        else
                        {
                            sq16_lru->blocks[s].data = NULL;
                        }
                    }
                }
            }
            else
            {
                sq16_scratch = (int16_t *)malloc((size_t)model->frame_elements *
                                                 SQ16_FASTSCAN_BLOCK_SIZE * sizeof(int16_t));
            }
        }
        visited.sq16_scratch_block = sq16_scratch;
        visited.sq16_lru_cache = sq16_lru;

#ifdef _OPENMP
#pragma omp for schedule(guided, 16)
#endif
        for (long i = 0; i < N_query; i++)
        {
            visited.epoch++;
            if (visited.epoch == 0)
            {
                if (visited.tags != NULL)
                {
                    memset(visited.tags, 0, (size_t)N_cand * sizeof(uint32_t));
                }
                if (visited.rep_tags != NULL)
                {
                    memset(visited.rep_tags, 0, (size_t)N_cand * sizeof(uint32_t));
                }
                visited.epoch = 1;
            }

            KnnFrameReader *active_qreader = is_cross_dataset ? &thread_query_reader :
                                                                &thread_cand_reader;
            visited.query_rq8_clipped = 0;

            if (knn_reader_read_frame(active_qreader, i, query_buffer) == 0)
            {
                if (config->use_sq8 && query_sq8 != NULL)
                {
                    if (model->is_double)
                    {
                        sq8_quantize_double(
                            (const double *)query_buffer, query_sq8, &model->sq8_params);
                    }
                    else
                    {
                        sq8_quantize_float(
                            (const float *)query_buffer, query_sq8, &model->sq8_params);
                    }
                }

                if (config->use_eq16)
                {
                    if (config->use_eq16_adc && query_eq16_adc != NULL)
                    {
                        if (model->is_double)
                        {
                            eq16_prepare_query_adc_double(
                                (const double *)query_buffer, query_eq16_adc, &model->eq16_params);
                        }
                        else
                        {
                            eq16_prepare_query_adc_float(
                                (const float *)query_buffer, query_eq16_adc, &model->eq16_params);
                        }
                    }
                    else if (query_eq16 != NULL)
                    {
                        if (model->is_double)
                        {
                            eq16_quantize_double(
                                (const double *)query_buffer, query_eq16, &model->eq16_params);
                        }
                        else
                        {
                            eq16_quantize_float(
                                (const float *)query_buffer, query_eq16, &model->eq16_params);
                        }
                    }
                }
                else if (config->use_sq16 && query_sq16 != NULL)
                {
                    if (model->is_double)
                    {
                        sq16_quantize_double(
                            (const double *)query_buffer, query_sq16, &model->sq16_params);
                    }
                    else
                    {
                        sq16_quantize_float(
                            (const float *)query_buffer, query_sq16, &model->sq16_params);
                    }
                }

                if (config->use_pq && query_pq_lut != NULL && model->pq_codebook != NULL)
                {
                    if (model->is_double)
                    {
                        pq_build_query_lut_double(
                            (const double *)query_buffer, model->pq_codebook,
                            &visited.query_pq_table, config->rlim_cutoff);
                    }
                    else
                    {
                        pq_build_query_lut_float(
                            (const float *)query_buffer, model->pq_codebook,
                            &visited.query_pq_table, config->rlim_cutoff);
                    }
                }

                if (config->use_rabitq && query_rabitq_rot != NULL &&
                    model->rabitq_dataset_buffer != NULL)
                {
                    if (model->is_double)
                    {
                        rabitq_rotate_vector_double(
                            (const double *)query_buffer, query_rabitq_rot,
                            &model->rabitq_params);
                    }
                    else
                    {
                        rabitq_rotate_vector_float(
                            (const float *)query_buffer, query_rabitq_rot,
                            &model->rabitq_params);
                    }

                    double q_norm_sq = 0.0;
                    long d = model->frame_elements;
                    if (model->is_double)
                    {
                        const double *q_d = (const double *)query_buffer;
                        for (long k = 0; k < d; k++)
                        {
                            q_norm_sq += q_d[k] * q_d[k];
                        }
                    }
                    else
                    {
                        const float *q_f = (const float *)query_buffer;
                        for (long k = 0; k < d; k++)
                        {
                            q_norm_sq += (double)(q_f[k] * q_f[k]);
                        }
                    }
                    visited.query_rabitq_norm = (float)sqrt(q_norm_sq);

                    rabitq_build_query_lut(
                        query_rabitq_rot, visited.query_rabitq_norm,
                        &model->rabitq_params, &visited.query_rabitq_lut);
                }

                if (is_cross_dataset)
                {
                    knn_search_cross_dataset_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, anchor_dists, scores_buf, &all_heaps[i],
                        &thread_tracker, &visited, &thread_telem);
                }
                else
                {
                    knn_search_single_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, scores_buf, &graph_scratch, all_heaps,
#ifdef _OPENMP
                        bucket_locks,
#endif
                        &visited, &thread_telem);
                }
                thread_telem.total_queries++;
            }

            if (config->progress_mode && i % progress_step == 0)
            {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0)
#endif
                {
                    double pct = 100.0 * (double)i / (double)N_query;
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
                    printf("\rSearching k-NN: [%-40s] %5.1f%% (%ld / %ld frames)",
                           &bar[bar_offset], pct, i, N_query);
                    fflush(stdout);
                }
            }
        } // for (long i = 0; ...)

        global_telem_calls += thread_telem.framedist_calls;
        global_telem_l1 += thread_telem.level1_clusters_pruned;
        global_telem_l2 += thread_telem.level2_anchors_pruned;
        global_telem_l3 += thread_telem.level3_annular_pruned;
        global_telem_temp += thread_telem.temporal_pruned;
        global_telem_recip += thread_telem.reciprocal_reused;
        global_telem_graph_seeds += thread_telem.graph_seeds_evaluated;
        global_telem_graph_edges += thread_telem.graph_edges_pruned;
        global_telem_multi_pivot += thread_telem.multi_pivot_pruned;
        global_telem_angular += thread_telem.angular_pruned;
        global_telem_containment += thread_telem.global_containment_hits;
        global_telem_cand += thread_telem.total_candidates_considered;
        global_telem_traj += thread_telem.trajectory_warmstarts;
        global_telem_rejected += thread_telem.out_of_cluster_rejected;
        global_telem_sq8_evals += thread_telem.sq8_evaluations;
        global_telem_sq8_pruned += thread_telem.sq8_members_pruned;
        global_telem_sq8_graph_pruned += thread_telem.sq8_graph_pruned;
        global_telem_sq16_evals += thread_telem.sq16_evaluations;
        global_telem_sq16_pruned += thread_telem.sq16_members_pruned;
        global_telem_sq16_graph_pruned += thread_telem.sq16_graph_pruned;
        global_telem_eq16_evals += thread_telem.eq16_evaluations;
        global_telem_eq16_pruned += thread_telem.eq16_members_pruned;
        global_telem_eq16_graph_pruned += thread_telem.eq16_graph_pruned;
        global_telem_rq8_evals += thread_telem.rq8_evaluations;
        global_telem_rq8_pruned += thread_telem.rq8_members_pruned;
        global_telem_rq8_graph_pruned += thread_telem.rq8_graph_pruned;
        global_telem_pq_evals += thread_telem.pq_evaluations;
        global_telem_pq_pruned += thread_telem.pq_members_pruned;
        global_telem_rabitq_evals += thread_telem.rabitq_evaluations;
        global_telem_rabitq_pruned += thread_telem.rabitq_members_pruned;
        global_telem_graph_clusters += thread_telem.clusters_graph_evaluated;
        global_telem_two_hop_evals += thread_telem.two_hop_evaluations;
        global_telem_two_hop_pruned += thread_telem.two_hop_pruned;
        global_telem_two_hop_injected += thread_telem.two_hop_injected;
        global_telem_memo_hits += thread_telem.memo_hits;

        if (visited.tags != NULL)
        {
            free(visited.tags);
        }

        if (visited.rep_tags != NULL)
        {
            free(visited.rep_tags);
        }

        if (visited.rep_dists != NULL)
        {
            free(visited.rep_dists);
        }

        if (sq16_lru != NULL)
        {
            if (sq16_lru->buffer_pool != NULL)
            {
                free(sq16_lru->buffer_pool);
            }
            free(sq16_lru);
        }

        if (sq16_scratch != NULL)
        {
            free(sq16_scratch);
        }

        if (query_sq8 != NULL)
        {
            free(query_sq8);
        }

        if (query_sq16 != NULL)
        {
            free(query_sq16);
        }

        if (query_eq16 != NULL)
        {
            free(query_eq16);
        }

        if (query_eq16_adc != NULL)
        {
            free(query_eq16_adc);
        }

        if (query_rq8 != NULL)
        {
            free(query_rq8);
        }

        if (query_rq8_adc != NULL)
        {
            free(query_rq8_adc);
        }

        if (query_pq_lut != NULL)
        {
            free(query_pq_lut);
        }

        if (query_rabitq_rot != NULL)
        {
            free(query_rabitq_rot);
        }

        if (query_rabitq_lut_buf != NULL)
        {
            free(query_rabitq_lut_buf);
        }

        if (graph_scratch.pq != NULL)
        {
            free(graph_scratch.pq);
        }
        if (graph_scratch.enqueued_tags != NULL)
        {
            free(graph_scratch.enqueued_tags);
        }
        if (graph_scratch.anchor_is_sq16 != NULL)
        {
            free(graph_scratch.anchor_is_sq16);
        }

        free(scores_buf);
        free(anchor_dists);
        free(cand_buffer);
        free(query_buffer);

        if (is_cross_dataset)
        {
            knn_reader_close_thread(&thread_query_reader);
        }
        knn_reader_close_thread(&thread_cand_reader);
    } // OpenMP parallel block

    telemetry->total_queries = (uint64_t)N_query;
    telemetry->framedist_calls = global_telem_calls;
    telemetry->level1_clusters_pruned = global_telem_l1;
    telemetry->level2_anchors_pruned = global_telem_l2;
    telemetry->level3_annular_pruned = global_telem_l3;
    telemetry->temporal_pruned = global_telem_temp;
    telemetry->reciprocal_reused = global_telem_recip;
    telemetry->graph_seeds_evaluated = global_telem_graph_seeds;
    telemetry->graph_edges_pruned = global_telem_graph_edges;
    telemetry->multi_pivot_pruned = global_telem_multi_pivot;
    telemetry->angular_pruned = global_telem_angular;
    telemetry->global_containment_hits = global_telem_containment;
    telemetry->trajectory_warmstarts = global_telem_traj;
    telemetry->out_of_cluster_rejected = global_telem_rejected;
    telemetry->sq8_evaluations = global_telem_sq8_evals;
    telemetry->sq8_members_pruned = global_telem_sq8_pruned;
    telemetry->sq8_graph_pruned = global_telem_sq8_graph_pruned;
    telemetry->sq16_evaluations = global_telem_sq16_evals;
    telemetry->sq16_members_pruned = global_telem_sq16_pruned;
    telemetry->sq16_graph_pruned = global_telem_sq16_graph_pruned;
    telemetry->eq16_evaluations = global_telem_eq16_evals;
    telemetry->eq16_members_pruned = global_telem_eq16_pruned;
    telemetry->eq16_graph_pruned = global_telem_eq16_graph_pruned;
    telemetry->rq8_evaluations = global_telem_rq8_evals;
    telemetry->rq8_members_pruned = global_telem_rq8_pruned;
    telemetry->rq8_graph_pruned = global_telem_rq8_graph_pruned;
    telemetry->pq_evaluations = global_telem_pq_evals;
    telemetry->pq_members_pruned = global_telem_pq_pruned;
    telemetry->rabitq_evaluations = global_telem_rabitq_evals;
    telemetry->rabitq_members_pruned = global_telem_rabitq_pruned;
    telemetry->clusters_graph_evaluated = global_telem_graph_clusters;
    telemetry->two_hop_evaluations = global_telem_two_hop_evals;
    telemetry->two_hop_pruned = global_telem_two_hop_pruned;
    telemetry->two_hop_injected = global_telem_two_hop_injected;
    telemetry->memo_hits = global_telem_memo_hits;
    telemetry->memo_unique_frames = (uint64_t)model->num_unique_frames;
    telemetry->total_candidates_considered = global_telem_cand;

    knn_search_extract_and_finalize(config, results, telemetry, all_heaps,
                                    N_query, is_cross_dataset,
#ifdef _OPENMP
                                    bucket_locks,
#endif
                                    &master_cand_reader, &master_query_reader,
                                    start_time);

    return 0;
}

/**
 * knn_results_free() - Clean up KnnResults arrays.
 * @results: Pointer to KnnResults structure to free.
 *
 * Purpose & Context ("What is this used for?"):
 * Releases dynamic memory allocated for top-k neighbor indices and distances
 * arrays across all queries. Resets buffer pointers to NULL to prevent dangling
 * references. Called by the gric-knn CLI and pipeline orchestrators once search
 * results are exported or verified.
 */
void knn_results_free(
    KnnResults *results)
{
    if (results == NULL)
    {
        return;
    }

    if (results->indices != NULL)
    {
        free(results->indices);
        results->indices = NULL;
    }

    if (results->distances != NULL)
    {
        free(results->distances);
        results->distances = NULL;
    }
}

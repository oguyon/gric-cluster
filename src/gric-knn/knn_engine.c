/**
 * @file knn_engine.c
 * @brief High-performance metric-pruned k-NN solver engine.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_engine.h"
#include "knn_heap.h"
#include "knn_reader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/** Cluster candidate record for sorting by ascending lower bound and center proximity */
typedef struct
{
    int    id;
    double lb;
    double dcc;
} ClusterScore;

/**
 * compare_cluster_scores() - Sort cluster candidates by ascending lower bound and center proximity.
 * @a: Pointer to first ClusterScore.
 * @b: Pointer to second ClusterScore.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_cluster_scores(
    const void *a,
    const void *b)
{
    const ClusterScore *ca = (const ClusterScore *)a;
    const ClusterScore *cb = (const ClusterScore *)b;
    if (ca->lb < cb->lb)
    {
        return -1;
    }
    if (ca->lb > cb->lb)
    {
        return 1;
    }
    if (ca->dcc < cb->dcc)
    {
        return -1;
    }
    if (ca->dcc > cb->dcc)
    {
        return 1;
    }
    return 0;
}

/**
 * find_member_lower_bound() - Binary search for first member with r_anchor >= val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element >= val.
 */
static inline int find_member_lower_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor < val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * find_member_upper_bound() - Binary search for first member with r_anchor > val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element > val.
 */
static inline int find_member_upper_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor <= val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * compute_euclidean_distance() - Vectorized Euclidean distance between frames.
 * @da:   Pointer to first pixel array.
 * @db:   Pointer to second pixel array.
 * @size: Number of elements in frame.
 *
 * Return: Euclidean L2 distance.
 */
static inline double compute_euclidean_distance(
    const double *restrict da,
    const double *restrict db,
    long                   size)
{
    double sum = 0.0;
    long i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 4)
    {
        __m256d sum_vec = _mm256_setzero_pd();
        for (; i <= size - 4; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&da[i]);
            __m256d vb = _mm256_loadu_pd(&db[i]);
            __m256d diff = _mm256_sub_pd(va, vb);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);
#else
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
#endif
        }
        __m256d hsum = _mm256_hadd_pd(sum_vec, sum_vec);
        sum += ((double *)&hsum)[0] + ((double *)&hsum)[2];
    }
#endif

    for (; i < size; i++)
    {
        double diff = da[i] - db[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

/**
 * check_temporal_separation() - Verify if candidate satisfies temporal criteria.
 * @query_id:     Frame ID of query.
 * @candidate_id: Frame ID of candidate.
 * @config:       Active KnnConfig.
 *
 * Return: 1 if candidate is temporally valid, 0 if excluded.
 */
static inline int check_temporal_separation(
    long             query_id,
    long             candidate_id,
    const KnnConfig *config)
{
    if (query_id == candidate_id)
    {
        return 0; // Self-match exclusion
    }

    if (config->past_only && candidate_id > query_id)
    {
        return 0;
    }

    if (config->future_only && candidate_id < query_id)
    {
        return 0;
    }

    long diff = query_id - candidate_id;
    if (diff < 0)
    {
        diff = -diff;
    }

    if (diff < (long)config->min_temporal_sep)
    {
        return 0;
    }

    return 1;
}

/**
 * knn_search_single_frame() - Execute multi-level metric pruned search for one query frame.
 * @query_id:       Index of query frame.
 * @query_data:     Pixel buffer of query frame.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @reader:         Thread-local KnnFrameReader.
 * @cand_buffer:    Scratch buffer for candidate frame pixels.
 * @scores_buffer:  Scratch buffer for cluster sorting.
 * @heap:           Thread-local KnnMaxHeap.
 * @telem:          Thread-local KnnTelemetry.
 * @out_indices:    Output array for neighbor indices.
 * @out_distances:  Output array for neighbor distances.
 */
static void knn_search_single_frame(
    long                  query_id,
    const double *restrict query_data,
    const KnnModel       *model,
    const KnnConfig      *config,
    KnnFrameReader       *reader,
    double       *restrict cand_buffer,
    ClusterScore *restrict scores_buffer,
    KnnMaxHeap           *heap,
    KnnTelemetry *restrict telem,
    int          *restrict out_indices,
    double       *restrict out_distances)
{
    knn_heap_reset(heap);

    int home_cluster_id = model->frame_cluster_map[query_id];
    double r_home = (double)model->frame_r_anchor[query_id];
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    // Step 1: Intra-Cluster Search (Home cluster c_p)
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        const KnnCluster *home_cl = &model->clusters[home_cluster_id];
        telem->total_candidates_considered += (uint64_t)home_cl->num_members;

        // Determine radial slice interval for home cluster
        int start_m = 0;
        int end_m = home_cl->num_members;
        double tau_init = knn_heap_peek_max_dist(heap);
        if (tau_init < 1e20)
        {
            float r_min = (float)fmax(0.0, r_home - tau_init / eps_factor);
            float r_max = (float)(r_home + tau_init / eps_factor);
            start_m = find_member_lower_bound(home_cl->members, home_cl->num_members, r_min);
            end_m = find_member_upper_bound(home_cl->members, home_cl->num_members, r_max);
            telem->level3_annular_pruned +=
                (uint64_t)start_m + (uint64_t)(home_cl->num_members - end_m);
        }

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)home_cl->members[m].frame_id;

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            double r_cand = (double)home_cl->members[m].r_anchor;
            double lb_annular = fabs(r_home - r_cand);

            if (lb_annular >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(query_data, cand_buffer, frame_elem);
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // Intra-Cluster Search

    // Step 2: Rank and Sort other candidate clusters (Heuristic proximity order)
    int num_cand_clusters = 0;
    for (int q = 0; q < M; q++)
    {
        if (q == home_cluster_id || model->clusters[q].num_members == 0)
        {
            continue;
        }

        double dcc = model->dcc_matrix[home_cluster_id * M + q];
        double r_q = model->clusters[q].radius;
        double lb = dcc - r_home - r_q;
        if (lb < 0.0)
        {
            lb = 0.0;
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = dcc;
        num_cand_clusters++;
    } // for (int q = 0; ...)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    // Step 3 & 4: Inter-Cluster Pruning and Verification
    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        double current_tau = knn_heap_peek_max_dist(heap);

        // Level 1: Cluster-level DCC bound
        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned += (uint64_t)(num_cand_clusters - idx);
            break; // Since sorted by lb, all remaining clusters are also pruned
        }

        // Level 2: Query-to-Anchor evaluation
        const KnnCluster *cl = &model->clusters[q];
        telem->framedist_calls++;
        double d_anchor = compute_euclidean_distance(query_data, cl->anchor_data, frame_elem);
        double lb_anchor = d_anchor - cl->radius;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
            continue;
        }

        // Level 3: Radially-Sorted Annular Window Slicing & Multi-Pivot Filter
        telem->total_candidates_considered += (uint64_t)cl->num_members;

        double tau_eff = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff = config->rlim_cutoff;
        }

        float r_min = (float)fmax(0.0, d_anchor - tau_eff);
        float r_max = (float)(d_anchor + tau_eff);

        int start_m = find_member_lower_bound(cl->members, cl->num_members, r_min);
        int end_m = find_member_upper_bound(cl->members, cl->num_members, r_max);

        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(cl->num_members - end_m);

        double dcc_home = model->dcc_matrix[home_cluster_id * M + q];

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double r_cand = (double)cl->members[m].r_anchor;

            // Primary pivot lower bound: anchor A_q
            double lb1 = fabs(d_anchor - r_cand);
            if (lb1 >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Secondary pivot lower bound: home anchor A_home
            double lb_home = dcc_home - r_cand - r_home;
            if (lb_home >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            if (config->rlim_cutoff > 0.0 &&
                (lb1 >= config->rlim_cutoff || lb_home >= config->rlim_cutoff))
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Level 4: Exact Distance Evaluation
            if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(query_data, cand_buffer, frame_elem);
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // for (int idx = 0; ...)

    // Extract sorted results
    knn_heap_extract_sorted(heap, out_indices, out_distances, config->k);
}

/**
 * knn_run_search() - Multi-threaded driver executing k-NN search across all frames.
 * @config:    Active KnnConfig.
 * @model:     Active KnnModel.
 * @results:   Output KnnResults structure to populate.
 * @telemetry: Output aggregated KnnTelemetry structure.
 *
 * Return: 0 on success, -1 on error.
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

    memset(telemetry, 0, sizeof(KnnTelemetry));

    long N = model->total_dataset_frames;
    int k = config->k;

    results->indices = (int *)malloc((size_t)N * (size_t)k * sizeof(int));
    results->distances = (double *)malloc((size_t)N * (size_t)k * sizeof(double));

    if (results->indices == NULL || results->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for results buffer\n");
        return -1;
    }

    KnnFrameReader master_reader;
    if (config->memory_data != NULL)
    {
        if (knn_reader_open_memory(&master_reader, config->memory_data, N,
                                   model->frame_elements) != 0)
        {
            knn_results_free(results);
            return -1;
        }
    }
    else if (knn_reader_open(&master_reader, config->input_data_path, N,
                             model->frame_width, model->frame_height) != 0)
    {
        knn_results_free(results);
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

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long progress_step = N / 100;
    if (progress_step < 1)
    {
        progress_step = 1;
    }

    uint64_t global_telem_calls = 0;
    uint64_t global_telem_l1 = 0;
    uint64_t global_telem_l2 = 0;
    uint64_t global_telem_l3 = 0;
    uint64_t global_telem_temp = 0;
    uint64_t global_telem_cand = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:global_telem_calls, global_telem_l1, global_telem_l2, \
                                 global_telem_l3, global_telem_temp, global_telem_cand)
#endif
    {
        KnnFrameReader thread_reader;
        knn_reader_clone_thread(&master_reader, &thread_reader);

        KnnMaxHeap thread_heap;
        knn_heap_init(&thread_heap, k);

        double *query_buffer = (double *)malloc((size_t)model->frame_elements * sizeof(double));
        double *cand_buffer = (double *)malloc((size_t)model->frame_elements * sizeof(double));
        ClusterScore *scores_buf =
            (ClusterScore *)malloc((size_t)model->num_clusters * sizeof(ClusterScore));

        KnnTelemetry thread_telem;
        memset(&thread_telem, 0, sizeof(KnnTelemetry));

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&thread_reader, i, query_buffer) == 0)
            {
                knn_search_single_frame(
                    i, query_buffer, model, config, &thread_reader,
                    cand_buffer, scores_buf, &thread_heap, &thread_telem,
                    &results->indices[i * k], &results->distances[i * k]);
                thread_telem.total_queries++;
            }

            if (config->progress_mode && i % progress_step == 0)
            {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0)
#endif
                {
                    double pct = 100.0 * (double)i / (double)N;
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
                           &bar[bar_offset], pct, i, N);
                    fflush(stdout);
                }
            }
        } // for (long i = 0; ...)

        global_telem_calls += thread_telem.framedist_calls;
        global_telem_l1 += thread_telem.level1_clusters_pruned;
        global_telem_l2 += thread_telem.level2_anchors_pruned;
        global_telem_l3 += thread_telem.level3_annular_pruned;
        global_telem_temp += thread_telem.temporal_pruned;
        global_telem_cand += thread_telem.total_candidates_considered;

        free(scores_buf);
        free(cand_buffer);
        free(query_buffer);
        knn_heap_free(&thread_heap);
        knn_reader_close_thread(&thread_reader);
    } // OpenMP parallel block

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    knn_reader_close(&master_reader);

    if (config->progress_mode)
    {
        printf("\rSearching k-NN: [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N, N);
        fflush(stdout);
    }

    telemetry->total_queries = (uint64_t)N;
    telemetry->framedist_calls = global_telem_calls;
    telemetry->level1_clusters_pruned = global_telem_l1;
    telemetry->level2_anchors_pruned = global_telem_l2;
    telemetry->level3_annular_pruned = global_telem_l3;
    telemetry->temporal_pruned = global_telem_temp;
    telemetry->total_candidates_considered = global_telem_cand;
    telemetry->time_search_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return 0;
}

/**
 * knn_results_free() - Clean up KnnResults arrays.
 * @results: Pointer to KnnResults.
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

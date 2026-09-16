/**
 * @file knn_cluster_search.c
 * @brief Intra-dataset cluster candidate scoring, graph routing, and search coordination.
 */

#include "knn_cluster_search.h"
#include "knn_member_search.h"
#include "knn_pruning.h"
#include "framedistance.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * knn_search_intra_cluster() - Search members of the query's home cluster.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance from query to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_search_intra_cluster(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    double                 r_home,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    KnnCandidateBatch batch;
    knn_batch_init(&batch);

    knn_eval_cluster_members(
        home_cluster_id,
        r_home,
        home_cluster_id,
        r_home,
        0, /* anchor_is_sq16 */
        query_id,
        query_data,
        model,
        config,
        reader,
        cand_buffer,
        heap,
        all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        NULL,
        0,
        visited,
        &batch,
        telem
    );

    knn_batch_flush(
        &batch,
        query_id,
        query_data,
        model,
        config,
        heap,
        all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        telem
    );
}

/**
 * knn_warm_start_nearest_cluster() - Pre-seed heap using nearest neighbor clusters.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate frame pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @pivots:          Pivot array.
 * @num_pivots:      Pointer to pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 *
 * Return: Best warm-start cluster index, or -1 if skipped.
 */
static int knn_warm_start_nearest_cluster(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    if (home_cluster_id < 0 || home_cluster_id >= model->num_clusters)
    {
        return -1;
    }

    /* Fast Graph Warm-Start (if precomputed k-NN graph is resident) */
    if (model->has_knn_graph && model->graph_indices != NULL &&
        query_id >= 0 && query_id < model->total_dataset_frames)
    {
        int g_k = model->graph_k;
        const uint32_t *nb_indices = &model->graph_indices[(size_t)query_id * (size_t)g_k];
        const float    *nb_dists = (model->graph_distances != NULL) ?
            &model->graph_distances[(size_t)query_id * (size_t)g_k] : NULL;

        for (int j = 0; j < g_k; j++)
        {
            long nb = (long)nb_indices[j];
            if (nb >= 0 && nb < model->total_dataset_frames && nb != query_id)
            {
                if (!knn_visited_is_visited(visited, nb))
                {
                    if (check_temporal_separation(query_id, nb, config))
                    {
                        double d = (nb_dists != NULL) ? (double)nb_dists[j] : -1.0;
                        if (d < 0.0)
                        {
                            const void *cand_data = NULL;
                            if (reader->memory_data != NULL)
                            {
                                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                                cand_data = (const char *)reader->memory_data +
                                            (size_t)nb * (size_t)model->frame_elements * el_sz;
                            }
                            else if (knn_reader_read_frame(reader, nb, cand_buffer) == 0)
                            {
                                cand_data = cand_buffer;
                            }
                            if (cand_data != NULL)
                            {
                                telem->framedist_calls++;
                                d = compute_euclidean_distance(
                                    query_data, cand_data, model->frame_elements, model->is_double
                                );
                            }
                        }
                        if (d >= 0.0)
                        {
                            knn_visited_check_and_mark(visited, nb);
                            knn_heap_push(heap, (int)nb, d);
                        }
                    }
                }
            } // if (nb >= 0 ...)
        } // for (int j = 0; j < g_k; j++)
        if (heap->count >= heap->k)
        {
            return -1;
        }
    } // if (model->has_knn_graph ...)

    if (heap->count >= heap->k)
    {
        return -1;
    }

    int M = model->num_clusters;
    int visited_warm[8];
    double warm_dcc[8];
    int num_warm_found = 0;
    int first_warm_c = -1;
    double eps_factor = 1.0 + config->epsilon;

    const double *home_dcc = &model->dcc_matrix[home_cluster_id * M];
    for (int c = 0; c < M; c++)
    {
        if (c == home_cluster_id || model->clusters[c].num_members == 0)
        {
            continue;
        }

        double dcc = home_dcc[c];
        if (dcc <= 0.0)
        {
            continue;
        }

        /* Maintain up to 8 closest clusters in sorted order */
        if (num_warm_found < 8)
        {
            int pos = num_warm_found;
            while (pos > 0 && warm_dcc[pos - 1] > dcc)
            {
                warm_dcc[pos] = warm_dcc[pos - 1];
                visited_warm[pos] = visited_warm[pos - 1];
                pos--;
            }
            warm_dcc[pos] = dcc;
            visited_warm[pos] = c;
            num_warm_found++;
        }
        else if (dcc < warm_dcc[7])
        {
            int pos = 7;
            while (pos > 0 && warm_dcc[pos - 1] > dcc)
            {
                warm_dcc[pos] = warm_dcc[pos - 1];
                visited_warm[pos] = visited_warm[pos - 1];
                pos--;
            }
            warm_dcc[pos] = dcc;
            visited_warm[pos] = c;
        }
    } // for (int c = 0; c < M; c++)

    for (int w = 0; w < num_warm_found; w++)
    {
        if (heap->count >= heap->k)
        {
            break;
        }

        int best_c = visited_warm[w];
        if (first_warm_c < 0)
        {
            first_warm_c = best_c;
        }

        const KnnCluster *warm_cl = &model->clusters[best_c];
        telem->total_candidates_considered += (uint64_t)warm_cl->num_members;
        telem->framedist_calls++;

        double d_anchor = compute_euclidean_distance(
            query_data, warm_cl->anchor_data, model->frame_elements, model->is_double
        );

        if (config->use_rq8 && model->rq8_dataset_buffer != NULL && visited->query_rq8 != NULL)
        {
            if (model->is_double)
            {
                visited->query_rq8_clipped = rq8_quantize_query_residual_double(
                    (const double *)query_data,
                    (const double *)warm_cl->anchor_data,
                    visited->query_rq8,
                    &model->rq8_params);
            }
            else
            {
                visited->query_rq8_clipped = rq8_quantize_query_residual_float(
                    (const float *)query_data,
                    (const float *)warm_cl->anchor_data,
                    visited->query_rq8,
                    &model->rq8_params);
            }
        }

        for (int m = 0; m < warm_cl->num_members; m++)
        {
            long cand_id = (long)warm_cl->members[m].frame_id;

            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }
            double r_cand = (double)warm_cl->members[m].r_anchor;
            double lb1 = fabs(d_anchor - r_cand);
            double current_tau = knn_heap_peek_max_dist(heap);
            if (lb1 >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }
            if ((!visited->query_rq8_clipped &&
                 is_member_pruned_by_rq8(visited->query_rq8, cand_id, current_tau,
                                         model, config, telem)) ||
                (!config->use_rq8 &&
                 is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                          model, config, telem)) ||
                (!config->use_rq8 &&
                 is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                         model, config, telem)))
            {
                continue;
            }
            if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
            {
                telem->reciprocal_reused++;
                continue;
            }
            long rep_id = (model->frame_to_unique_map != NULL) ?
                          model->frame_to_unique_map[cand_id] : cand_id;
            if (visited != NULL && visited->rep_tags != NULL &&
                visited->rep_tags[rep_id] == visited->epoch)
            {
                telem->memo_hits++;
                double d = (double)visited->rep_dists[rep_id];
                record_neighbor_and_reciprocal(
                    query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
                continue;
            }

            const void *cand_data = NULL;
            if (reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)reader->memory_data +
                            (size_t)cand_id * (size_t)model->frame_elements * el_sz;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, model->frame_elements, model->is_double
                );
                if (visited != NULL && visited->rep_tags != NULL)
                {
                    visited->rep_dists[rep_id] = (float)d;
                    visited->rep_tags[rep_id] = visited->epoch;
                }
                record_neighbor_and_reciprocal(
                    query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        } // for (int m = 0; ...)

        if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS)
        {
            pivots[*num_pivots].cluster_id = best_c;
            pivots[*num_pivots].d_anchor = d_anchor;
            (*num_pivots)++;
        }
    } // for (int w = 0; w < num_warm_found; w++)

    return first_warm_c;
}

/**
 * knn_inject_two_hop_candidates() - Expand 2-hop neighbors from top heap seeds.
 * @query_id:     Index of query frame.
 * @query_data:   Query frame pixel data.
 * @model:        Active KnnModel.
 * @config:       Active KnnConfig.
 * @reader:       KnnFrameReader context.
 * @cand_buffer:  Scratch buffer for candidate frame pixels.
 * @heap:         Max-heap for current query.
 * @all_heaps:    Array of all frame heaps.
 * @bucket_locks: OpenMP locks (if multithreaded).
 * @visited:      Per-query frame visited tracker.
 * @telem:        Telemetry record.
 */
static void knn_inject_two_hop_candidates(
    long                   query_id,
    const void *restrict   query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void *restrict         cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    KnnVisitedTracker     *visited,
    KnnTelemetry *restrict telem)
{
    if (!config->use_two_hop || heap == NULL || heap->count == 0)
    {
        return;
    }

    int max_seeds = (config->two_hop_seeds > 0) ? config->two_hop_seeds : 2;
    if (max_seeds > 4)
    {
        max_seeds = 4;
    }
    if (max_seeds > heap->count)
    {
        max_seeds = heap->count;
    }

    long   seed_ids[4];
    double seed_dists[4];
    int    num_seeds = 0;

    for (int j = 0; j < heap->count; j++)
    {
        long cand_id = (long)knn_heap_get_id(heap, j);
        double d = knn_heap_get_dist(heap, j);

        int pos = num_seeds;
        while (pos > 0 && seed_dists[pos - 1] > d)
        {
            if (pos < max_seeds)
            {
                seed_ids[pos] = seed_ids[pos - 1];
                seed_dists[pos] = seed_dists[pos - 1];
            }
            pos--;
        }
        if (pos < max_seeds)
        {
            seed_ids[pos] = cand_id;
            seed_dists[pos] = d;
            if (num_seeds < max_seeds)
            {
                num_seeds++;
            }
        }
    } // for (int j = 0; j < heap->count; j++)

    int max_hop2 = (config->two_hop_max_cands > 0) ? config->two_hop_max_cands : 16;
    int hop2_evals = 0;

    double eps_factor = 1.0 + config->epsilon;
    size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
    long frame_elem = model->frame_elements;

    for (int s = 0; s < num_seeds && hop2_evals < max_hop2; s++)
    {
        long   seed_id = seed_ids[s];
        double d_query_seed = seed_dists[s];

        if (model->has_knn_graph && model->graph_indices != NULL)
        {
            int g_k = model->graph_k;
            const uint32_t *nb_indices = &model->graph_indices[(size_t)seed_id * (size_t)g_k];
            const float    *nb_dists = (model->graph_distances != NULL) ?
                &model->graph_distances[(size_t)seed_id * (size_t)g_k] : NULL;

            for (int j = 0; j < g_k && hop2_evals < max_hop2; j++)
            {
                long nb = (long)nb_indices[j];
                if (nb < 0 || nb >= model->total_dataset_frames || nb == query_id)
                {
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb))
                {
                    continue;
                }

                if (!check_temporal_separation(query_id, nb, config))
                {
                    telem->temporal_pruned++;
                    continue;
                }

                double cur_tau = knn_heap_peek_max_dist(heap);
                if (nb_dists != NULL)
                {
                    double d_seed_nb = (double)nb_dists[j];
                    double lb_tri = fabs(d_query_seed - d_seed_nb);
                    if (lb_tri >= cur_tau / eps_factor)
                    {
                        telem->level3_annular_pruned++;
                        continue;
                    }
                }

                if (is_member_pruned_by_sq16(
                        visited->query_sq16, nb, cur_tau, model, config, telem))
                {
                    continue;
                }
                if (is_member_pruned_by_sq8(visited->query_sq8, nb, cur_tau, model, config, telem))
                {
                    continue;
                }

                const void *cand_data = NULL;
                if (reader->memory_data != NULL)
                {
                    cand_data = (const char *)reader->memory_data +
                                (size_t)nb * (size_t)frame_elem * el_sz;
                }
                else if (knn_reader_read_frame(reader, nb, cand_buffer) == 0)
                {
                    cand_data = cand_buffer;
                }

                if (cand_data != NULL)
                {
                    telem->framedist_calls++;
                    hop2_evals++;
                    double d = compute_euclidean_distance(
                        query_data, cand_data, frame_elem, model->is_double
                    );
                    record_neighbor_and_reciprocal(
                        query_id, nb, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            } // for (int j = 0; j < g_k ... )
        }
        else
        {
            int seed_cl = model->frame_cluster_map[seed_id];
            if (seed_cl < 0 || seed_cl >= model->num_clusters)
            {
                continue;
            }
            const KnnCluster *cl = &model->clusters[seed_cl];
            for (int m = 0; m < cl->num_members && hop2_evals < max_hop2; m++)
            {
                long nb = (long)cl->members[m].frame_id;
                if (nb == query_id || knn_visited_check_and_mark(visited, nb))
                {
                    continue;
                }
                if (!check_temporal_separation(query_id, nb, config))
                {
                    telem->temporal_pruned++;
                    continue;
                }

                double cur_tau = knn_heap_peek_max_dist(heap);
                if (is_member_pruned_by_sq16(
                        visited->query_sq16, nb, cur_tau, model, config, telem))
                {
                    continue;
                }
                if (is_member_pruned_by_sq8(visited->query_sq8, nb, cur_tau, model, config, telem))
                {
                    continue;
                }

                const void *cand_data = NULL;
                if (reader->memory_data != NULL)
                {
                    cand_data = (const char *)reader->memory_data +
                                (size_t)nb * (size_t)frame_elem * el_sz;
                }
                else if (knn_reader_read_frame(reader, nb, cand_buffer) == 0)
                {
                    cand_data = cand_buffer;
                }

                if (cand_data != NULL)
                {
                    telem->framedist_calls++;
                    hop2_evals++;
                    double d = compute_euclidean_distance(
                        query_data, cand_data, frame_elem, model->is_double
                    );
                    record_neighbor_and_reciprocal(
                        query_id, nb, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            } // for (int m = 0; ...)
        } // if-else has_knn_graph
    } // for (int s = 0; ...)
}

/**
 * knn_score_candidate_clusters() - Compute lower bounds and sort candidate clusters.
 */
static int knn_score_candidate_clusters(
    int                     home_cluster_id,
    const MeasuredPivot    *pivots,
    int                     num_pivots,
    double                  r_home,
    const KnnModel         *model,
    const KnnConfig        *config,
    const KnnMaxHeap       *heap,
    ClusterScore *restrict  scores_buffer,
    KnnTelemetry *restrict  telem)
{
    int M = model->num_clusters;
    int num_cand_clusters = 0;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);

    TE4Ref te4_ref;
    const double *d1_row = NULL;
    const double *d2_row = NULL;
    if (num_pivots >= 2)
    {
        int    c1 = pivots[0].cluster_id;
        int    c2 = pivots[1].cluster_id;
        double d1 = pivots[0].d_anchor;
        double d2 = pivots[1].d_anchor;
        double d12 = model->dcc_matrix[c1 * M + c2];
        calc_te4_ref_init(&te4_ref, d1, d2, d12);
        d1_row = &model->dcc_matrix[c1 * M];
        d2_row = &model->dcc_matrix[c2 * M];
    }

    const double *home_dcc = &model->dcc_matrix[home_cluster_id * M];
    for (int q = 0; q < M; q++)
    {
        if (q == home_cluster_id)
        {
            continue;
        }

        double dcc_home_q = home_dcc[q];
        if (dcc_home_q < 0.0)
        {
            continue;
        }

        double r_q = model->clusters[q].radius;
        double lb1 = fabs(dcc_home_q - r_home) - r_q;

        double lb = lb1;
        if (config->use_multi_pivot && num_pivots > 0)
        {
            for (int p = 0; p < num_pivots; p++)
            {
                int    p_cl = pivots[p].cluster_id;
                double d_qp = pivots[p].d_anchor;
                double dcc_pq = model->dcc_matrix[p_cl * M + q];
                if (dcc_pq > 0.0)
                {
                    double lb_p = fabs(dcc_pq - d_qp) - r_q;
                    if (lb_p > lb)
                    {
                        lb = lb_p;
                    }
                }
            } // for (int p = 0; p < num_pivots; p++)

            if (d1_row != NULL && d2_row != NULL)
            {
                double d1q = d1_row[q];
                double d2q = d2_row[q];
                double min_d = calc_min_dist_4pt_ref(&te4_ref, d1q, d2q);
                double lb_te4 = min_d - r_q;
                if (lb_te4 > lb)
                {
                    lb = lb_te4;
                }
            }
        } // if (config->use_multi_pivot ...)

        if (lb >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = dcc_home_q;
        num_cand_clusters++;
    } // for (int q = 0; q < M; q++)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore), compare_cluster_scores);
    return num_cand_clusters;
}

/**
 * knn_search_inter_clusters() - Search scored candidate clusters in ascending lower bound.
 */
static void knn_search_inter_clusters(
    long                   query_id,
    const void *restrict   query_data,
    int                    home_cluster_id,
    double                 r_home,
    int                    num_cand_clusters,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    void          *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    double eps_factor = 1.0 + config->epsilon;

    KnnCandidateBatch batch;
    knn_batch_init(&batch);

    double sq16_delta = (config->use_sq16 && model->sq16_dataset_buffer != NULL)
                        ? 2.0 * (double)model->sq16_params.err_radius
                        : 0.0;
    TE4Ref te4_pivots[8][8];
    int n_p_cached = 0;

    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        double current_tau = knn_heap_peek_max_dist(heap);

        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        const KnnCluster *cl = &model->clusters[q];
        if (num_pivots != NULL && *num_pivots >= 2)
        {
            int n_p = (*num_pivots > 8) ? 8 : *num_pivots;
            if (n_p > n_p_cached)
            {
                for (int p1 = 0; p1 < n_p - 1; p1++)
                {
                    int    c1 = pivots[p1].cluster_id;
                    double d1 = pivots[p1].d_anchor;
                    for (int p2 = (p1 >= n_p_cached ? p1 + 1 : n_p_cached); p2 < n_p; p2++)
                    {
                        int    c2 = pivots[p2].cluster_id;
                        double d2 = pivots[p2].d_anchor;
                        double d12 = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)c2];
                        calc_te4_ref_init(&te4_pivots[p1][p2], d1, d2, d12);
                    }
                }
                n_p_cached = n_p;
            }

            int pruned_by_te4 = 0;
            for (int p1 = 0; p1 < n_p - 1; p1++)
            {
                int c1 = pivots[p1].cluster_id;
                for (int p2 = p1 + 1; p2 < n_p; p2++)
                {
                    int    c2 = pivots[p2].cluster_id;
                    double d1q = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)q];
                    double d2q = model->dcc_matrix[(size_t)c2 * (size_t)M + (size_t)q];

                    double min_d = calc_min_dist_4pt_ref(&te4_pivots[p1][p2], d1q, d2q);
                    if (min_d - cl->radius >= current_tau / eps_factor)
                    {
                        pruned_by_te4 = 1;
                        break;
                    }
                }
                if (pruned_by_te4)
                {
                    break;
                }
            }
            if (pruned_by_te4)
            {
                telem->level1_clusters_pruned++;
                continue;
            }
        }

        int    anchor_is_sq16 = 0;
        double d_anchor = knn_compute_anchor_distance(
            query_data, q, model, config, visited, &anchor_is_sq16, telem
        );

        if (!anchor_is_sq16)
        {
            if (config->use_multi_pivot)
            {
                for (int j = idx + 1; j < num_cand_clusters; j++)
                {
                    int other_q = scores_buffer[j].id;
                    double dcc_pivot = model->dcc_matrix[(size_t)q * (size_t)M + (size_t)other_q];
                    if (dcc_pivot > 0.0)
                    {
                        double r_other = model->clusters[other_q].radius;
                        double lb_pivot = dcc_pivot - d_anchor - r_other;
                        if (lb_pivot > scores_buffer[j].lb)
                        {
                            scores_buffer[j].lb = lb_pivot;
                        }
                    }
                } // for (int j = idx + 1; ...)
            }

            if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS)
            {
                int already_present = 0;
                for (int p = 0; p < *num_pivots; p++)
                {
                    if (pivots[p].cluster_id == q)
                    {
                        already_present = 1;
                        break;
                    }
                }
                if (!already_present)
                {
                    pivots[*num_pivots].cluster_id = q;
                    pivots[*num_pivots].d_anchor = d_anchor;
                    (*num_pivots)++;
                }
            }
        } // if (!anchor_is_sq16)

        double sq_err = anchor_is_sq16 ? (double)model->sq16_params.err_radius : 0.0;
        double lb_anchor = d_anchor - cl->radius - sq_err;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
            continue;
        }

        if (is_cluster_pruned_by_pivots(
                q, cl->radius, current_tau, eps_factor, model, config,
                pivots, (num_pivots != NULL) ? *num_pivots : 0, sq16_delta))
        {
            telem->level1_clusters_pruned++;
            telem->multi_pivot_pruned++;
            continue;
        }

        knn_eval_cluster_members(
            q, d_anchor, home_cluster_id, r_home, anchor_is_sq16,
            query_id, query_data, model, config, reader, cand_buffer,
            heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, (num_pivots != NULL) ? *num_pivots : 0, visited,
            &batch, telem
        );
    } // for (int idx = 0; idx < num_cand_clusters; idx++)

    knn_batch_flush(
        &batch, query_id, query_data, model, config, heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        telem);
}

/**
 * knn_search_cluster_graph() - Execute best-first graph routing over cluster proximity graph.
 */
static void knn_search_cluster_graph(
    long                    query_id,
    const void *restrict    query_data,
    int                     home_cluster_id,
    double                  r_home,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnClusterGraphScratch *scratch,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    MeasuredPivot          *pivots,
    int                    *num_pivots,
    KnnVisitedTracker      *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    int k_adj = model->cluster_graph_k;
    double eps_factor = 1.0 + config->epsilon;
    double rlim = (config->rlim_cutoff > 0.0) ? config->rlim_cutoff : model->model_rlim;
    if (rlim <= 0.0)
    {
        rlim = 1e9;
    }

    scratch->enqueued_epoch++;
    if (scratch->enqueued_epoch == 0)
    {
        memset(scratch->enqueued_tags, 0, (size_t)M * sizeof(uint32_t));
        scratch->enqueued_epoch = 1;
    }
    uint32_t epoch = scratch->enqueued_epoch;

    int pq_size = 0;
    ClusterPqNode *pq = scratch->pq;

    /* Mark home cluster as enqueued so it is not re-evaluated */
    scratch->enqueued_tags[home_cluster_id] = epoch;
    scratch->anchor_dists[home_cluster_id] = r_home;
    scratch->anchor_is_sq16[home_cluster_id] = 0;

    /* Mark any clusters already evaluated in warm-start / pivots */
    if (num_pivots != NULL)
    {
        for (int p = 0; p < *num_pivots; p++)
        {
            int p_cl = pivots[p].cluster_id;
            if (p_cl >= 0 && p_cl < M)
            {
                scratch->enqueued_tags[p_cl] = epoch;
                scratch->anchor_dists[p_cl] = pivots[p].d_anchor;
                scratch->anchor_is_sq16[p_cl] = 0;
            }
        }
    }

    /* Seed priority queue with graph neighbors of home cluster and pivots */
    int n_seed_pivots = (num_pivots != NULL) ? *num_pivots : 1;
    for (int p = 0; p < n_seed_pivots; p++)
    {
        int p_cl = (num_pivots != NULL) ? pivots[p].cluster_id : home_cluster_id;
        if (p_cl < 0 || p_cl >= M)
        {
            continue;
        }
        const int *adj = model->cluster_graph_adj + (size_t)p_cl * (size_t)k_adj;
        for (int i = 0; i < k_adj; i++)
        {
            int nbr = adj[i];
            if (nbr < 0 || nbr >= M || scratch->enqueued_tags[nbr] == epoch)
            {
                continue;
            }
            scratch->enqueued_tags[nbr] = epoch;

            int is_sq16 = 0;
            double d_nbr = knn_compute_anchor_distance(
                query_data, nbr, model, config, visited, &is_sq16, telem
            );
            scratch->anchor_dists[nbr] = d_nbr;
            scratch->anchor_is_sq16[nbr] = (uint8_t)is_sq16;
            cluster_pq_push(pq, &pq_size, nbr, d_nbr);
        }
    }

    KnnCandidateBatch batch;
    knn_batch_init(&batch);

    int ef_limit = config->ef_cluster;
    if (ef_limit <= 0)
    {
        double avg_size = (model->avg_cluster_size > 0.0)
                          ? model->avg_cluster_size : 50.0;
        int c_needed = (int)ceil((double)config->k / avg_size);
        int target = 4 * c_needed;
        if (target < 12)
        {
            target = 12;
        }
        if (target > 48)
        {
            target = 48;
        }
        if (target > M)
        {
            target = M;
        }
        ef_limit = target;
    }
    else if (ef_limit > M)
    {
        ef_limit = M;
    }
    int clusters_evaluated = 0;
    double sq16_delta = (config->use_sq16 && model->sq16_dataset_buffer != NULL)
                        ? 2.0 * (double)model->sq16_params.err_radius
                        : 0.0;

    while (pq_size > 0 && clusters_evaluated < ef_limit)
    {
        ClusterPqNode top = cluster_pq_pop(pq, &pq_size);
        int c = top.cluster_id;
        double d_anchor = top.dist;
        int anchor_is_sq16 = (int)scratch->anchor_is_sq16[c];

        double current_tau = knn_heap_peek_max_dist(heap);
        const KnnCluster *cl = &model->clusters[c];

        double sq_err = anchor_is_sq16 ? (double)model->sq16_params.err_radius : 0.0;
        double lb_anchor = d_anchor - cl->radius - sq_err;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        /* Early termination: if heap is full and anchor distance is beyond tau + rlim */
        if (heap->count >= heap->k && d_anchor > current_tau / eps_factor + rlim)
        {
            break;
        }

        /* Record measured anchor pivot if useful */
        if (num_pivots != NULL && *num_pivots < MAX_MEASURED_PIVOTS)
        {
            if (!is_in_pivots(c, pivots, *num_pivots))
            {
                pivots[*num_pivots].cluster_id = c;
                pivots[*num_pivots].d_anchor = d_anchor;
                (*num_pivots)++;
            }
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
        }
        else if (is_cluster_pruned_by_pivots(
                     c, cl->radius, current_tau, eps_factor, model, config,
                     pivots, (num_pivots != NULL) ? *num_pivots : 0, sq16_delta))
        {
            telem->level1_clusters_pruned++;
            telem->multi_pivot_pruned++;
        }
        else
        {
            clusters_evaluated++;
            telem->clusters_graph_evaluated++;

            /* Evaluate members of cluster c */
            knn_eval_cluster_members(
                c, d_anchor, home_cluster_id, r_home, anchor_is_sq16,
                query_id, query_data, model, config, reader, cand_buffer,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                pivots, (num_pivots != NULL) ? *num_pivots : 0, visited,
                &batch, telem
            );
        }

        /* Expand neighbors into priority queue if within search horizon */
        if (d_anchor <= current_tau / eps_factor + rlim)
        {
            const int *adj = model->cluster_graph_adj + (size_t)c * (size_t)k_adj;
            for (int i = 0; i < k_adj; i++)
            {
                int nbr = adj[i];
                if (nbr < 0 || nbr >= M || scratch->enqueued_tags[nbr] == epoch)
                {
                    continue;
                }
                scratch->enqueued_tags[nbr] = epoch;

                int is_sq16 = 0;
                double d_nbr = knn_compute_anchor_distance(
                    query_data, nbr, model, config, visited, &is_sq16, telem
                );
                scratch->anchor_dists[nbr] = d_nbr;
                scratch->anchor_is_sq16[nbr] = (uint8_t)is_sq16;
                cluster_pq_push(pq, &pq_size, nbr, d_nbr);
            }
        }
    } // while (pq_size > 0 && ...)

    /* Flush any pending batched distance computations */
    knn_batch_flush(
        &batch, query_id, query_data, model, config, heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        telem);
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
 * @graph_scratch:  Scratch buffers for cluster graph search.
 * @all_heaps:      Array of KnnMaxHeap structures for all frames.
 * @bucket_locks:   Array of OpenMP bucket locks (if OpenMP enabled).
 * @visited:        Per-query frame visited tracker.
 * @telem:          Thread-local KnnTelemetry.
 */
void knn_search_single_frame(
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnClusterGraphScratch *graph_scratch,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnTelemetry  *restrict telem)
{
    KnnMaxHeap *heap = &all_heaps[query_id];
    int home_cluster_id = model->frame_cluster_map[query_id];
    double r_home = (double)model->frame_r_anchor[query_id];
    int M = model->num_clusters;

    /* Pre-seed visited tracker with frames already in heap (from reciprocal pushes) */
    for (int h = 0; h < heap->count; h++)
    {
        knn_visited_check_and_mark(visited, (long)knn_heap_get_id(heap, h));
    }

    MeasuredPivot pivots[MAX_MEASURED_PIVOTS];
    int           num_pivots = 0;
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        pivots[num_pivots].cluster_id = home_cluster_id;
        pivots[num_pivots].d_anchor = r_home;
        num_pivots++;
    }

    /* Fast Graph Warm-Start: if precomputed k-NN graph is resident, load it first */
    if (model->has_knn_graph && model->graph_indices != NULL)
    {
        knn_warm_start_nearest_cluster(
            query_id, query_data, home_cluster_id, model, config, reader,
            cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }

    // Step 1: Intra-Cluster Search (Home cluster c_p)
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        knn_search_intra_cluster(
            query_id, query_data, home_cluster_id, r_home, model, config,
            reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, telem
        );
    }

    /* If heap is still not full, search nearest neighbor clusters */
    if (heap->count < heap->k)
    {
        knn_warm_start_nearest_cluster(
            query_id, query_data, home_cluster_id, model, config, reader,
            cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }

    // 2-Hop Candidate Injection: Expand neighbors of top seeds to collapse tau early
    if (config->use_two_hop)
    {
        knn_inject_two_hop_candidates(
            query_id, query_data, model, config, reader, cand_buffer,
            heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, telem
        );
    }

    if (config->approx_mode)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        return;
    }

    // Step 2: Search remaining clusters (graph-guided routing or linear inter-cluster scan)
    if (config->use_cluster_graph && model->cluster_graph_adj != NULL &&
        model->cluster_graph_k > 0 && home_cluster_id >= 0 && home_cluster_id < M)
    {
        knn_search_cluster_graph(
            query_id, query_data, home_cluster_id, r_home, model, config,
            reader, cand_buffer, graph_scratch, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }
    else
    {
        int num_cand_clusters = knn_score_candidate_clusters(
            home_cluster_id, pivots, num_pivots, r_home, model, config,
            heap, scores_buffer, telem
        );

        knn_search_inter_clusters(
            query_id, query_data, home_cluster_id, r_home, num_cand_clusters,
            model, config, reader, cand_buffer, scores_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, &num_pivots, visited, telem
        );
    }
}

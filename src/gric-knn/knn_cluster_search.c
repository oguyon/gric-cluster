/**
 * @file knn_cluster_search.c
 * @brief Intra-dataset cluster candidate scoring, graph routing, and member search.
 */

#include "knn_cluster_search.h"
#include <alloca.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

static inline int knn_popcount32(
    uint32_t value)
{
#ifdef _MSC_VER
    return (int)__popcnt(value);
#else
    return __builtin_popcount(value);
#endif
}

static inline int knn_ctz32(
    uint32_t value)
{
#ifdef _MSC_VER
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return (int)index;
#else
    return __builtin_ctz(value);
#endif
}

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
    const KnnCluster *home_cl = &model->clusters[home_cluster_id];
    telem->total_candidates_considered += (uint64_t)home_cl->num_members;
    double eps_factor = 1.0 + config->epsilon;

    int num_m = home_cl->num_members;
    int mid = find_member_lower_bound(home_cl->members, num_m, (float)r_home);
    int left = mid - 1;
    int right = mid;

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;
    long frame_elem = model->frame_elements;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    int sq16_active = (config->use_sq16 && model->sq16_dataset_buffer != NULL &&
                       visited->query_sq16 != NULL);

    if (sq16_active && home_cl->sq16_transposed != NULL && home_cl->num_sq16_blocks > 0)
    {
        int num_b = home_cl->num_sq16_blocks;
        int mid_b = 0;
        int b_lo = 0;
        int b_hi = num_b - 1;
        while (b_lo <= b_hi)
        {
            int b_m = b_lo + (b_hi - b_lo) / 2;
            if (home_cl->members[b_m * SQ16_FASTSCAN_BLOCK_SIZE].r_anchor <= (float)r_home)
            {
                mid_b = b_m;
                b_lo = b_m + 1;
            }
            else
            {
                b_hi = b_m - 1;
            }
        } // while (b_lo <= b_hi)
        int left_b = mid_b - 1;
        int right_b = mid_b;

        while (left_b >= 0 || right_b < num_b)
        {
            double d_left_b = 1e30;
            if (left_b >= 0)
            {
                int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                    (left_b * SQ16_FASTSCAN_BLOCK_SIZE + SQ16_FASTSCAN_BLOCK_SIZE - 1);
                float r_max_l = home_cl->members[m_end_l].r_anchor;
                d_left_b = (r_home > (double)r_max_l) ? (r_home - (double)r_max_l) : 0.0;
            }

            double d_right_b = 1e30;
            if (right_b < num_b)
            {
                float r_min_r = home_cl->members[right_b * SQ16_FASTSCAN_BLOCK_SIZE].r_anchor;
                d_right_b = ((double)r_min_r > r_home) ?
                    ((double)r_min_r - r_home) : 0.0;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            double tau_thresh = current_tau / eps_factor;
            if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
            {
                tau_thresh = config->rlim_cutoff;
            }

            if (left_b >= 0 && d_left_b >= tau_thresh)
            {
                int pruned_count = (left_b + 1) * SQ16_FASTSCAN_BLOCK_SIZE;
                if (pruned_count > num_m)
                {
                    pruned_count = num_m;
                }
                telem->level3_annular_pruned += (uint64_t)pruned_count;
                left_b = -1;
                d_left_b = 1e30;
            }
            if (right_b < num_b && d_right_b >= tau_thresh)
            {
                int pruned_count = num_m - right_b * SQ16_FASTSCAN_BLOCK_SIZE;
                if (pruned_count > 0)
                {
                    telem->level3_annular_pruned += (uint64_t)pruned_count;
                }
                right_b = num_b;
                d_right_b = 1e30;
            }
            if (left_b < 0 && right_b >= num_b)
            {
                break;
            }

            int b;
            if (d_left_b <= d_right_b)
            {
                b = left_b--;
            }
            else
            {
                b = right_b++;
            }

            int m_start = b * SQ16_FASTSCAN_BLOCK_SIZE;
            int m_count = num_m - m_start;
            if (m_count > SQ16_FASTSCAN_BLOCK_SIZE)
            {
                m_count = SQ16_FASTSCAN_BLOCK_SIZE;
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }

            const int16_t *b_coords = home_cl->sq16_transposed +
                (size_t)b * (size_t)frame_elem * SQ16_FASTSCAN_BLOCK_SIZE;
            telem->sq16_evaluations += (uint64_t)m_count;

            uint32_t pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }

            if (!pass_mask)
            {
                telem->sq16_members_pruned += (uint64_t)m_count;
                continue;
            }

            int passed_count = knn_popcount32(pass_mask);
            telem->sq16_members_pruned += (uint64_t)(m_count - passed_count);

            while (pass_mask)
            {
                int lane = knn_ctz32(pass_mask);
                pass_mask &= pass_mask - 1;

                int m = m_start + lane;
                long cand_id = (long)home_cl->members[m].frame_id;
                double r_cand = (double)home_cl->members[m].r_anchor;

                if (knn_visited_check_and_mark(visited, cand_id))
                {
                    continue;
                }

                if (!check_temporal_separation(query_id, cand_id, config))
                {
                    telem->temporal_pruned++;
                    continue;
                }

                if (fabs(r_home - r_cand) >= tau_thresh)
                {
                    telem->level3_annular_pruned++;
                    continue;
                }

                if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                            model, config, telem))
                {
                    continue;
                }

                if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
                {
                    telem->reciprocal_reused++;
                    continue;
                }

                if (!config->use_batch_dist)
                {
                    long rep_id = (model->frame_to_unique_map != NULL) ?
                                  model->frame_to_unique_map[cand_id] : cand_id;
                    double c_tau = (config->rlim_cutoff > 0.0 &&
                                    config->rlim_cutoff < current_tau)
                                   ? config->rlim_cutoff
                                   : current_tau;
                    if (visited != NULL && visited->rep_tags != NULL &&
                        visited->rep_tags[rep_id] == visited->epoch)
                    {
                        telem->memo_hits++;
                        double d = (double)visited->rep_dists[rep_id];
                        if (d <= c_tau)
                        {
                            record_neighbor_and_reciprocal(
                                query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                                , bucket_locks
#endif
                            );
                        }
                        continue;
                    }

                    const void *cand_ptr = NULL;
                    if (reader->memory_data != NULL)
                    {
                        cand_ptr = (const char *)reader->memory_data +
                                   (size_t)cand_id * frame_bytes;
                    }
                    else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
                    {
                        cand_ptr = cand_buffer;
                    }

                    if (cand_ptr != NULL)
                    {
                        telem->framedist_calls++;
                        double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                        double d = compute_euclidean_distance_cutoff(
                            query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                        );
                        if (visited != NULL && visited->rep_tags != NULL)
                        {
                            visited->rep_dists[rep_id] = (float)d;
                            visited->rep_tags[rep_id] = visited->epoch;
                        }
                        if (d <= c_tau)
                        {
                            record_neighbor_and_reciprocal(
                                query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                                , bucket_locks
#endif
                            );
                        }
                    }
                    continue;
                }

                const void *dest = NULL;
                if (reader->memory_data != NULL)
                {
                    dest = (const char *)reader->memory_data +
                           (size_t)cand_id * frame_bytes;
                }
                else
                {
                    void *buf_dest = (char *)cand_buffer +
                                     (size_t)batch_count * frame_bytes;
                    if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
                    {
                        dest = buf_dest;
                    }
                }
                if (dest != NULL)
                {
                    batch_cand_ids[batch_count] = cand_id;
                    batch_ptrs[batch_count] = dest;
                    batch_count++;
                    if (batch_count == 4)
                    {
                        double dists[4];
                        if (model->is_double)
                        {
                            framedist_batch_1x4_double(
                                (const double *)query_data,
                                (const double *const *)batch_ptrs,
                                dists,
                                frame_elem);
                        }
                        else
                        {
                            framedist_batch_1x4_float(
                                (const float *)query_data,
                                (const float *const *)batch_ptrs,
                                dists,
                                frame_elem);
                        }
                        telem->framedist_calls += 4;
                        for (int b_idx = 0; b_idx < 4; b_idx++)
                        {
                            record_neighbor_and_reciprocal(
                                query_id, batch_cand_ids[b_idx], dists[b_idx],
                                config, model, heap, all_heaps
#ifdef _OPENMP
                                , bucket_locks
#endif
                            );
                        }
                        batch_count = 0;
                    }
                }
            } // while (pass_mask)
        } // while (left_b >= 0 || right_b < num_b)

        if (batch_count > 0)
        {
            double dists[4];
            if (model->is_double)
            {
                framedist_batch_double(
                    (const double *)query_data,
                    (const double *const *)batch_ptrs,
                    batch_count,
                    dists,
                    frame_elem);
            }
            else
            {
                framedist_batch_float(
                    (const float *)query_data,
                    (const float *const *)batch_ptrs,
                    batch_count,
                    dists,
                    frame_elem);
            }
            telem->framedist_calls += (uint64_t)batch_count;
            for (int b_idx = 0; b_idx < batch_count; b_idx++)
            {
                record_neighbor_and_reciprocal(
                    query_id, batch_cand_ids[b_idx], dists[b_idx],
                    config, model, heap, all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        }

        return;
    } // if (sq16_active && home_cl->sq16_transposed != NULL)

    while (left >= 0 || right < num_m)
    {
        double d_left = (left >= 0) ?
            fabs(r_home - (double)home_cl->members[left].r_anchor) : 1e30;
        double d_right = (right < num_m) ?
            fabs(r_home - (double)home_cl->members[right].r_anchor) : 1e30;

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left >= 0 && d_left >= tau_thresh)
        {
            telem->level3_annular_pruned += (uint64_t)(left + 1);
            left = -1;
            d_left = 1e30;
        }
        if (right < num_m && d_right >= tau_thresh)
        {
            telem->level3_annular_pruned += (uint64_t)(num_m - right);
            right = num_m;
            d_right = 1e30;
        }
        if (left < 0 && right >= num_m)
        {
            break;
        }

        int m;
        if (d_left <= d_right)
        {
            m = left--;
        }
        else
        {
            m = right++;
        }

        long cand_id = (long)home_cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }

        if (sq16_active)
        {
            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }
            if (is_member_pruned_by_sq16_cached(
                    visited->query_sq16, cand_id, cached_ssd_cutoff, model, telem
                ))
            {
                continue;
            }
        }
        else if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                          model, config, telem))
        {
            continue;
        }

        if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                    model, config, telem))
        {
            continue;
        }

        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }

        if (!config->use_batch_dist)
        {
            long rep_id = (model->frame_to_unique_map != NULL) ?
                          model->frame_to_unique_map[cand_id] : cand_id;
            double c_tau = (config->rlim_cutoff > 0.0 && config->rlim_cutoff < current_tau)
                           ? config->rlim_cutoff
                           : current_tau;
            if (visited != NULL && visited->rep_tags != NULL &&
                visited->rep_tags[rep_id] == visited->epoch)
            {
                telem->memo_hits++;
                double d = (double)visited->rep_dists[rep_id];
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
                continue;
            }

            const void *cand_ptr = NULL;
            if (reader->memory_data != NULL)
            {
                cand_ptr = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_ptr = cand_buffer;
            }
            if (cand_ptr != NULL)
            {
                telem->framedist_calls++;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                );
                if (visited != NULL && visited->rep_tags != NULL)
                {
                    visited->rep_dists[rep_id] = (float)d;
                    visited->rep_tags[rep_id] = visited->epoch;
                }
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            }
            continue;
        }

        const void *dest = NULL;
        if (reader->memory_data != NULL)
        {
            dest = (const char *)reader->memory_data + (size_t)cand_id * frame_bytes;
        }
        else
        {
            void *buf_dest = (char *)cand_buffer + (size_t)batch_count * frame_bytes;
            if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
            {
                dest = buf_dest;
            }
        }
        if (dest != NULL)
        {
            batch_cand_ids[batch_count] = cand_id;
            batch_ptrs[batch_count] = dest;
            batch_count++;
            if (batch_count == 4)
            {
                double dists[4];
                if (model->is_double)
                {
                    framedist_batch_1x4_double(
                        (const double *)query_data,
                        (const double *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                else
                {
                    framedist_batch_1x4_float(
                        (const float *)query_data,
                        (const float *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                telem->framedist_calls += 4;
                for (int b = 0; b < 4; b++)
                {
                    record_neighbor_and_reciprocal(
                        query_id, batch_cand_ids[b], dists[b],
                        config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
                batch_count = 0;
            }
        }
    } // while (left >= 0 || right < num_m)

    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
}

/**
 * knn_warm_start_nearest_cluster() - Search nearest neighboring cluster if heap is not full.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @pivots:          Output pivot array.
 * @num_pivots:      Output pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 *
 * Return: Cluster ID of nearest cluster used for warm start, or -1.
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

    // Fast Graph Warm-Start (if precomputed k-NN graph is resident)
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
            }
        }
        if (heap->count >= heap->k)
        {
            return -1;
        }
    }

    if (heap->count >= heap->k)
    {
        return -1;
    }

    // Multi-Cluster Nearest Warm-Start until heap saturates
    int M = model->num_clusters;
    int visited_warm[8];
    int num_warm = 0;
    int first_warm_c = -1;
    double eps_factor = 1.0 + config->epsilon;

    while (heap->count < heap->k && num_warm < 8)
    {
        int best_c = -1;
        double min_dcc = 1e19;

        for (int c = 0; c < M; c++)
        {
            if (c == home_cluster_id || model->clusters[c].num_members == 0)
            {
                continue;
            }
            int already_w = 0;
            for (int w = 0; w < num_warm; w++)
            {
                if (visited_warm[w] == c)
                {
                    already_w = 1;
                    break;
                }
            }
            if (already_w)
            {
                continue;
            }

            double dcc = model->dcc_matrix[home_cluster_id * M + c];
            if (dcc > 0.0 && dcc < min_dcc)
            {
                min_dcc = dcc;
                best_c = c;
            }
        }

        if (best_c < 0)
        {
            break;
        }
        visited_warm[num_warm++] = best_c;
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
            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
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
    } // while (heap->count < heap->k && num_warm < 8)

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
        long cand_id = (long)heap->data[j].frame_id;
        double d = heap->data[j].dist;

        if (cand_id < 0 || cand_id >= model->total_dataset_frames || cand_id == query_id)
        {
            continue;
        }

        if (num_seeds < max_seeds)
        {
            int insert_pos = num_seeds;
            while (insert_pos > 0 && d < seed_dists[insert_pos - 1])
            {
                seed_ids[insert_pos] = seed_ids[insert_pos - 1];
                seed_dists[insert_pos] = seed_dists[insert_pos - 1];
                insert_pos--;
            }
            seed_ids[insert_pos] = cand_id;
            seed_dists[insert_pos] = d;
            num_seeds++;
        }
        else if (d < seed_dists[max_seeds - 1])
        {
            int insert_pos = max_seeds - 1;
            while (insert_pos > 0 && d < seed_dists[insert_pos - 1])
            {
                seed_ids[insert_pos] = seed_ids[insert_pos - 1];
                seed_dists[insert_pos] = seed_dists[insert_pos - 1];
                insert_pos--;
            }
            seed_ids[insert_pos] = cand_id;
            seed_dists[insert_pos] = d;
        }
    }

    if (num_seeds == 0)
    {
        return;
    }

    int max_evals = (config->two_hop_max_cands > 0) ? config->two_hop_max_cands : 32;
    int evals_done = 0;
    double eps_factor = 1.0 + config->epsilon;
    long frame_elem = model->frame_elements;
    size_t frame_bytes = (size_t)frame_elem *
                         (model->is_double ? sizeof(double) : sizeof(float));

    for (int s = 0; s < num_seeds; s++)
    {
        long u = seed_ids[s];
        double d_qu = seed_dists[s];

        long   cands_2hop[64];
        double dists_2hop[64];
        int    num_2hop = 0;

        if (all_heaps != NULL)
        {
#ifdef _OPENMP
            if (bucket_locks != NULL)
            {
                omp_set_lock(&bucket_locks[u & KNN_BUCKET_LOCK_MASK]);
            }
#endif
            int u_cnt = all_heaps[u].count;
            if (u_cnt > 32)
            {
                u_cnt = 32;
            }
            for (int c = 0; c < u_cnt; c++)
            {
                cands_2hop[num_2hop] = (long)all_heaps[u].data[c].frame_id;
                dists_2hop[num_2hop] = all_heaps[u].data[c].dist;
                num_2hop++;
            }
#ifdef _OPENMP
            if (bucket_locks != NULL)
            {
                omp_unset_lock(&bucket_locks[u & KNN_BUCKET_LOCK_MASK]);
            }
#endif
        }

        if (model->has_knn_graph && model->graph_indices != NULL)
        {
            int g_k = model->graph_k;
            int add_k = (g_k < 32) ? g_k : 32;
            const uint32_t *nb_idx = &model->graph_indices[(size_t)u * (size_t)g_k];
            const float    *nb_dst = (model->graph_distances != NULL) ?
                &model->graph_distances[(size_t)u * (size_t)g_k] : NULL;

            for (int c = 0; c < add_k && num_2hop < 64; c++)
            {
                long nb = (long)nb_idx[c];
                if (nb >= 0 && nb < model->total_dataset_frames &&
                    nb != query_id && nb != u)
                {
                    cands_2hop[num_2hop] = nb;
                    dists_2hop[num_2hop] = (nb_dst != NULL) ? (double)nb_dst[c] : -1.0;
                    num_2hop++;
                }
            }
        }

        for (int c = 0; c < num_2hop; c++)
        {
            long v = cands_2hop[c];
            double d_uv = dists_2hop[c];

            if (v < 0 || v >= model->total_dataset_frames || v == query_id)
            {
                continue;
            }

            if (knn_visited_check_and_mark(visited, v))
            {
                continue;
            }

            if (!check_temporal_separation(query_id, v, config))
            {
                telem->temporal_pruned++;
                continue;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            if (d_uv >= 0.0)
            {
                double lb_2hop = fabs(d_qu - d_uv);
                if (heap->count >= heap->k && lb_2hop >= current_tau / eps_factor)
                {
                    telem->two_hop_pruned++;
                    continue;
                }
            }

            if (is_member_pruned_by_sq16(visited->query_sq16, v, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, v, current_tau,
                                        model, config, telem))
            {
                continue;
            }

            if (config->use_reciprocal && knn_heap_contains(heap, (int)v))
            {
                telem->reciprocal_reused++;
                continue;
            }

            const void *cand_data = NULL;
            if (reader->memory_data != NULL)
            {
                cand_data = (const char *)reader->memory_data + (size_t)v * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, v, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }

            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                telem->two_hop_evaluations++;
                evals_done++;

                double c_tau = (config->rlim_cutoff > 0.0 &&
                                config->rlim_cutoff < current_tau)
                               ? config->rlim_cutoff
                               : current_tau;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_data, frame_elem, model->is_double, cutoff_sq
                );

                if (d < current_tau)
                {
                    telem->two_hop_injected++;
                }

                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, v, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }

                if (evals_done >= max_evals)
                {
                    return;
                }
            }
        } // for (int c = 0; c < num_2hop; ...)
    } // for (int s = 0; s < num_seeds; ...)
}

/**
 * knn_score_candidate_clusters() - Filter super-clusters and score candidate clusters.
 * @home_cluster_id: Home cluster index.
 * @pivots:          Array of measured anchor pivots.
 * @num_pivots:      Number of measured anchor pivots.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @heap:            Max-heap for current query.
 * @scores_buffer:   Output array for scored candidate clusters.
 * @telem:           Telemetry record.
 *
 * Return: Number of candidate clusters populated in scores_buffer.
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

    for (int q = 0; q < M; q++)
    {
        if (q == home_cluster_id || is_in_pivots(q, pivots, num_pivots) ||
            model->clusters[q].num_members == 0)
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

        // Level 1: Cluster-level DCC bound
        if (current_tau < 1e18 && lb >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // Multi-Point TE4 Triangulation against measured pivots
        if (num_pivots >= 2)
        {
            int    c1 = pivots[0].cluster_id;
            int    c2 = pivots[1].cluster_id;
            double d1 = pivots[0].d_anchor;
            double d2 = pivots[1].d_anchor;
            double d12 = model->dcc_matrix[c1 * M + c2];
            double d1q = model->dcc_matrix[c1 * M + q];
            double d2q = model->dcc_matrix[c2 * M + q];

            double min_d = calc_min_dist_4pt(d1, d2, d12, d1q, d2q);
            if (min_d - r_q - 1e-5 >= current_tau / eps_factor)
            {
                telem->level1_clusters_pruned++;
                continue;
            }
            if (min_d - r_q > lb)
            {
                lb = min_d - r_q;
            }
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = dcc;
        num_cand_clusters++;
    } // for (int q = 0; ...)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    return num_cand_clusters;
}


/**
 * knn_eval_candidate_cluster_members() - Evaluate members of a single candidate cluster.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @q:               Candidate cluster index.
 * @d_anchor:        Query-to-anchor distance for cluster q.
 * @anchor_is_sq16:  1 if d_anchor was computed using SQ16.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all heaps.
 * @bucket_locks:    OpenMP bucket locks.
 * @pivots:          Pivot array.
 * @num_pivots:      Pointer to pivot count.
 * @visited:         Per-query frame visited tracker.
 * @batch_count:     Pointer to current batch count.
 * @batch_cand_ids:  Batch candidate frame IDs buffer.
 * @batch_ptrs:      Batch candidate pointers buffer.
 * @telem:           Telemetry record.
 */
static void knn_eval_candidate_cluster_members(
    long                   query_id,
    const void *restrict   query_data,
    int                    q,
    double                 d_anchor,
    int                    anchor_is_sq16,
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
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    int                   *batch_count,
    long                   batch_cand_ids[4],
    const void            *batch_ptrs[4],
    KnnTelemetry  *restrict telem)
{
    const KnnCluster *cl = &model->clusters[q];
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));
    double current_tau = knn_heap_peek_max_dist(heap);

    int num_m = cl->num_members;
    if (num_m <= 0)
    {
        return;
    }

    // Level 3: Center-Outward Annular Window Expansion & Multi-Pivot Filter
    telem->total_candidates_considered += (uint64_t)num_m;

    double sq16_delta = anchor_is_sq16 ? 2.0 * (double)model->sq16_params.err_radius : 0.0;
    int mid = find_member_lower_bound(cl->members, num_m, (float)d_anchor);
    int left = mid - 1;
    int right = mid;

    double dcc_home = (home_cluster_id >= 0 && home_cluster_id < M) ?
        model->dcc_matrix[(size_t)home_cluster_id * (size_t)M + (size_t)q] : 0.0;
    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    int sq16_active = (config->use_sq16 && model->sq16_dataset_buffer != NULL &&
                       visited->query_sq16 != NULL);

    int    num_active_pivots = 0;
    double pivot_diffs[MAX_MEASURED_PIVOTS];
    if (config->use_multi_pivot && num_pivots != NULL && *num_pivots > 0)
    {
        for (int p = 0; p < *num_pivots; p++)
        {
            int p_cl = pivots[p].cluster_id;
            if (p_cl == q || p_cl == home_cluster_id)
            {
                continue;
            }
            double dcc_pq = model->dcc_matrix[(size_t)p_cl * (size_t)M + (size_t)q];
            if (dcc_pq > 0.0)
            {
                pivot_diffs[num_active_pivots++] = fabs(dcc_pq - pivots[p].d_anchor);
            }
        }
    }

    if (sq16_active && cl->sq16_transposed != NULL && cl->num_sq16_blocks > 0)
    {
        int num_b = cl->num_sq16_blocks;
        int mid_b = 0;
        int b_lo = 0;
        int b_hi = num_b - 1;
        while (b_lo <= b_hi)
        {
            int b_m = b_lo + (b_hi - b_lo) / 2;
            if (cl->members[b_m * SQ16_FASTSCAN_BLOCK_SIZE].r_anchor <= (float)d_anchor)
            {
                mid_b = b_m;
                b_lo = b_m + 1;
            }
            else
            {
                b_hi = b_m - 1;
            }
        } // while (b_lo <= b_hi)
        int left_b = mid_b - 1;
        int right_b = mid_b;

        while (left_b >= 0 || right_b < num_b)
        {
            double d_left_b = 1e30;
            if (left_b >= 0)
            {
                int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                    (left_b * SQ16_FASTSCAN_BLOCK_SIZE + SQ16_FASTSCAN_BLOCK_SIZE - 1);
                float r_max_l = cl->members[m_end_l].r_anchor;
                d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
            }

            double d_right_b = 1e30;
            if (right_b < num_b)
            {
                float r_min_r = cl->members[right_b * SQ16_FASTSCAN_BLOCK_SIZE].r_anchor;
                d_right_b = ((double)r_min_r > d_anchor) ?
                    ((double)r_min_r - d_anchor) : 0.0;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double tau_thresh = current_tau / eps_factor;
            if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
            {
                tau_thresh = config->rlim_cutoff;
            }

            if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
            {
                int pruned_count = (left_b + 1) * SQ16_FASTSCAN_BLOCK_SIZE;
                if (pruned_count > num_m)
                {
                    pruned_count = num_m;
                }
                telem->level3_annular_pruned += (uint64_t)pruned_count;
                left_b = -1;
                d_left_b = 1e30;
            }
            if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
            {
                int pruned_count = num_m - right_b * SQ16_FASTSCAN_BLOCK_SIZE;
                if (pruned_count > 0)
                {
                    telem->level3_annular_pruned += (uint64_t)pruned_count;
                }
                right_b = num_b;
                d_right_b = 1e30;
            }
            if (left_b < 0 && right_b >= num_b)
            {
                break;
            }

            int b;
            if (d_left_b <= d_right_b)
            {
                b = left_b--;
            }
            else
            {
                b = right_b++;
            }

            int m_start = b * SQ16_FASTSCAN_BLOCK_SIZE;
            int m_count = num_m - m_start;
            if (m_count > SQ16_FASTSCAN_BLOCK_SIZE)
            {
                m_count = SQ16_FASTSCAN_BLOCK_SIZE;
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }

            const int16_t *b_coords = cl->sq16_transposed +
                (size_t)b * (size_t)frame_elem * SQ16_FASTSCAN_BLOCK_SIZE;
            telem->sq16_evaluations += (uint64_t)m_count;

            uint32_t pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }

            if (!pass_mask)
            {
                telem->sq16_members_pruned += (uint64_t)m_count;
                continue;
            }

            int passed_count = knn_popcount32(pass_mask);
            telem->sq16_members_pruned += (uint64_t)(m_count - passed_count);

            while (pass_mask)
            {
                int lane = knn_ctz32(pass_mask);
                pass_mask &= pass_mask - 1;

                int m = m_start + lane;
                long cand_id = (long)cl->members[m].frame_id;
                double r_cand = (double)cl->members[m].r_anchor;

                if (knn_visited_check_and_mark(visited, cand_id))
                {
                    continue;
                }

                if (!check_temporal_separation(query_id, cand_id, config))
                {
                    telem->temporal_pruned++;
                    continue;
                }

                double lb1 = fabs(d_anchor - r_cand) - sq16_delta;
                if (lb1 < 0.0)
                {
                    lb1 = 0.0;
                }
                if (lb1 >= tau_thresh)
                {
                    telem->level3_annular_pruned++;
                    continue;
                }

                if (dcc_home > 0.0)
                {
                    double diff_home = fabs(dcc_home - r_cand);
                    if (diff_home - r_home - sq16_delta >= tau_thresh)
                    {
                        telem->level3_annular_pruned++;
                        continue;
                    }
                }

                if (num_active_pivots > 0)
                {
                    int pruned_by_pivot = 0;
                    double target_thresh = tau_thresh + sq16_delta;
                    for (int p = 0; p < num_active_pivots; p++)
                    {
                        double diff = fabs(pivot_diffs[p] - r_cand);
                        if (diff >= target_thresh)
                        {
                            pruned_by_pivot = 1;
                            break;
                        }
                    }
                    if (pruned_by_pivot)
                    {
                        telem->level3_annular_pruned++;
                        telem->multi_pivot_pruned++;
                        continue;
                    }
                }

                if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                            model, config, telem))
                {
                    continue;
                }

                if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
                {
                    telem->reciprocal_reused++;
                    continue;
                }

                if (!config->use_batch_dist)
                {
                    const void *cand_ptr = NULL;
                    if (reader->memory_data != NULL)
                    {
                        cand_ptr = (const char *)reader->memory_data +
                                   (size_t)cand_id * frame_bytes;
                    }
                    else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
                    {
                        cand_ptr = cand_buffer;
                    }

                    if (cand_ptr != NULL)
                    {
                        telem->framedist_calls++;
                        double c_tau = (config->rlim_cutoff > 0.0 &&
                                        config->rlim_cutoff < current_tau)
                                       ? config->rlim_cutoff
                                       : current_tau;
                        double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                        double d = compute_euclidean_distance_cutoff(
                            query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                        );
                        if (d <= c_tau)
                        {
                            record_neighbor_and_reciprocal(
                                query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                                , bucket_locks
#endif
                            );
                        }
                    }
                    continue;
                }

                const void *dest = NULL;
                if (reader->memory_data != NULL)
                {
                    dest = (const char *)reader->memory_data +
                           (size_t)cand_id * frame_bytes;
                }
                else
                {
                    void *buf_dest = (char *)cand_buffer +
                                     (size_t)(*batch_count) * frame_bytes;
                    if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
                    {
                        dest = buf_dest;
                    }
                }
                if (dest != NULL)
                {
                    batch_cand_ids[*batch_count] = cand_id;
                    batch_ptrs[*batch_count] = dest;
                    (*batch_count)++;
                    if (*batch_count == 4)
                    {
                        double dists[4];
                        if (model->is_double)
                        {
                            framedist_batch_1x4_double(
                                (const double *)query_data,
                                (const double *const *)batch_ptrs,
                                dists,
                                frame_elem);
                        }
                        else
                        {
                            framedist_batch_1x4_float(
                                (const float *)query_data,
                                (const float *const *)batch_ptrs,
                                dists,
                                frame_elem);
                        }
                        telem->framedist_calls += 4;
                        for (int b_idx = 0; b_idx < 4; b_idx++)
                        {
                            record_neighbor_and_reciprocal(
                                query_id, batch_cand_ids[b_idx], dists[b_idx],
                                config, model, heap, all_heaps
#ifdef _OPENMP
                                , bucket_locks
#endif
                            );
                        }
                        *batch_count = 0;
                    }
                }
            } // while (pass_mask)
        } // while (left_b >= 0 || right_b < num_b)

        return;
    } // if (sq16_active && cl->sq16_transposed != NULL)

    while (left >= 0 || right < num_m)
    {
        double d_left = (left >= 0) ?
            fabs(d_anchor - (double)cl->members[left].r_anchor) : 1e30;
        double d_right = (right < num_m) ?
            fabs(d_anchor - (double)cl->members[right].r_anchor) : 1e30;

        current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left >= 0 && (d_left - sq16_delta >= tau_thresh))
        {
            telem->level3_annular_pruned += (uint64_t)(left + 1);
            left = -1;
            d_left = 1e30;
        }
        if (right < num_m && (d_right - sq16_delta >= tau_thresh))
        {
            telem->level3_annular_pruned += (uint64_t)(num_m - right);
            right = num_m;
            d_right = 1e30;
        }
        if (left < 0 && right >= num_m)
        {
            break;
        }

        int m;
        if (d_left <= d_right)
        {
            m = left--;
        }
        else
        {
            m = right++;
        }

        long cand_id = (long)cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }

        double r_cand = (double)cl->members[m].r_anchor;

        // Primary pivot lower bound: anchor A_q
        double lb1 = fabs(d_anchor - r_cand) - sq16_delta;
        if (lb1 < 0.0)
        {
            lb1 = 0.0;
        }
        if (lb1 >= tau_thresh)
        {
            telem->level3_annular_pruned++;
            continue;
        }

        // Secondary pivot lower bound: home anchor A_home
        if (dcc_home > 0.0)
        {
            double diff_home = dcc_home - r_cand;
            double lb_home = fabs(diff_home) - r_home;
            if (lb_home - sq16_delta >= tau_thresh)
            {
                telem->level3_annular_pruned++;
                continue;
            }
        }

        // Multi-Anchor Pivot Bounding (AESA / LAESA Indexing) - Member Level
        if (num_active_pivots > 0)
        {
            int pruned_by_pivot = 0;
            double target_thresh = tau_thresh + sq16_delta;
            for (int p = 0; p < num_active_pivots; p++)
            {
                double diff = pivot_diffs[p] - r_cand;
                if (diff < 0.0)
                {
                    diff = -diff;
                }
                if (diff >= target_thresh)
                {
                    pruned_by_pivot = 1;
                    break;
                }
            }

            if (pruned_by_pivot)
            {
                telem->level3_annular_pruned++;
                telem->multi_pivot_pruned++;
                continue;
            }
        }

        if (sq16_active)
        {
            if (left >= 0)
            {
                long pref_l = (long)cl->members[left].frame_id;
                GRIC_PREFETCH_T0(
                    model->sq16_dataset_buffer + (size_t)pref_l * (size_t)frame_elem
                );
            }
            if (right < num_m)
            {
                long pref_r = (long)cl->members[right].frame_id;
                GRIC_PREFETCH_T0(
                    model->sq16_dataset_buffer + (size_t)pref_r * (size_t)frame_elem
                );
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
            }

            const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                                       (size_t)cand_id * (size_t)frame_elem;
            telem->sq16_evaluations++;
            uint64_t ssd = sq16_dist_squared_cutoff_i16(
                visited->query_sq16, cand_sq16, frame_elem, cached_ssd_cutoff
            );

            if (ssd > cached_ssd_cutoff)
            {
                telem->sq16_members_pruned++;
                continue;
            }
        }
        else if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                          model, config, telem))
        {
            continue;
        }

        if (is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                    model, config, telem))
        {
            continue;
        }

        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }

        if (!config->use_batch_dist)
        {
            const void *cand_ptr = NULL;
            if (reader->memory_data != NULL)
            {
                cand_ptr = (const char *)reader->memory_data +
                           (size_t)cand_id * frame_bytes;
            }
            else if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                cand_ptr = cand_buffer;
            }

            if (cand_ptr != NULL)
            {
                telem->framedist_calls++;
                double c_tau = (config->rlim_cutoff > 0.0 &&
                                config->rlim_cutoff < current_tau)
                               ? config->rlim_cutoff
                               : current_tau;
                double cutoff_sq = (c_tau > 0.0) ? (c_tau * c_tau) : 0.0;
                double d = compute_euclidean_distance_cutoff(
                    query_data, cand_ptr, frame_elem, model->is_double, cutoff_sq
                );
                if (d <= c_tau)
                {
                    record_neighbor_and_reciprocal(
                        query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
            }
            continue;
        }

        const void *dest = NULL;
        if (reader->memory_data != NULL)
        {
            dest = (const char *)reader->memory_data +
                   (size_t)cand_id * frame_bytes;
        }
        else
        {
            void *buf_dest = (char *)cand_buffer +
                             (size_t)(*batch_count) * frame_bytes;
            if (knn_reader_read_frame(reader, cand_id, buf_dest) == 0)
            {
                dest = buf_dest;
            }
        }
        if (dest != NULL)
        {
            batch_cand_ids[*batch_count] = cand_id;
            batch_ptrs[*batch_count] = dest;
            (*batch_count)++;
            if (*batch_count == 4)
            {
                double dists[4];
                if (model->is_double)
                {
                    framedist_batch_1x4_double(
                        (const double *)query_data,
                        (const double *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                else
                {
                    framedist_batch_1x4_float(
                        (const float *)query_data,
                        (const float *const *)batch_ptrs,
                        dists,
                        frame_elem);
                }
                telem->framedist_calls += 4;
                for (int b = 0; b < 4; b++)
                {
                    record_neighbor_and_reciprocal(
                        query_id, batch_cand_ids[b], dists[b],
                        config, model, heap, all_heaps
#ifdef _OPENMP
                        , bucket_locks
#endif
                    );
                }
                *batch_count = 0;
            }
        }
    } // while (left >= 0 || right < num_m)
}

/**
 * knn_search_inter_clusters() - Evaluate inter-cluster candidate members.
 * @query_id:          Index of query frame.
 * @query_data:        Query frame pixel data.
 * @home_cluster_id:   Home cluster index.
 * @r_home:            Distance to home anchor.
 * @num_cand_clusters: Number of candidate clusters.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            KnnFrameReader context.
 * @cand_buffer:       Candidate pixel buffer.
 * @scores_buffer:     Candidate cluster records.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Array of all heaps.
 * @bucket_locks:      OpenMP bucket locks.
 * @pivots:            Pivot array.
 * @num_pivots:        Pointer to pivot count.
 * @visited:           Per-query frame visited tracker.
 * @telem:             Telemetry record.
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
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;
    double sq16_delta = (config->use_sq16 && model->sq16_dataset_buffer != NULL)
                        ? 2.0 * (double)model->sq16_params.err_radius
                        : 0.0;

    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        double current_tau = knn_heap_peek_max_dist(heap);

        // Level 1: Cluster-level DCC bound
        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // Multi-Point TE4 Triangulation against measured pivots
        const KnnCluster *cl = &model->clusters[q];
        if (num_pivots != NULL && *num_pivots >= 2)
        {
            int pruned_by_te4 = 0;
            int n_p = (*num_pivots > 8) ? 8 : *num_pivots;
            for (int p1 = 0; p1 < n_p - 1; p1++)
            {
                int    c1 = pivots[p1].cluster_id;
                double d1 = pivots[p1].d_anchor;
                for (int p2 = p1 + 1; p2 < n_p; p2++)
                {
                    int    c2 = pivots[p2].cluster_id;
                    double d2 = pivots[p2].d_anchor;
                    double d12 = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)c2];
                    double d1q = model->dcc_matrix[(size_t)c1 * (size_t)M + (size_t)q];
                    double d2q = model->dcc_matrix[(size_t)c2 * (size_t)M + (size_t)q];

                    double min_d = calc_min_dist_4pt(d1, d2, d12, d1q, d2q);
                    if (min_d - cl->radius - 1e-5 >= current_tau / eps_factor)
                    {
                        pruned_by_te4 = 1;
                        break;
                    }
                    if (min_d - cl->radius > lb_cluster)
                    {
                        lb_cluster = min_d - cl->radius;
                        scores_buffer[idx].lb = lb_cluster;
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

        // Level 2: Query-to-Anchor evaluation
        int    anchor_is_sq16 = 0;
        double d_anchor = knn_compute_anchor_distance(
            query_data, q, model, config, visited, &anchor_is_sq16, telem
        );

        if (!anchor_is_sq16)
        {
            // Dynamic Bound Tightening (Multi-Pivot)
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

            // Record measured anchor pivot for Multi-Anchor Pivot Bounding (AESA)
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
        }

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

        // Level 3 & 4: Evaluate candidate cluster members
        knn_eval_candidate_cluster_members(
            query_id, query_data, q, d_anchor, anchor_is_sq16,
            home_cluster_id, r_home, model, config, reader, cand_buffer,
            heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            pivots, num_pivots, visited,
            &batch_count, batch_cand_ids, batch_ptrs, telem
        );
    } // for (int idx = 0; idx < num_cand_clusters; idx++)

    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count,
                dists,
                frame_elem);
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
}

/**
 * knn_search_cluster_graph() - Execute best-first graph routing over cluster proximity graph.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @scratch:         Thread-local cluster graph scratch buffers.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all heaps.
 * @bucket_locks:    OpenMP bucket locks.
 * @pivots:          Pivot array.
 * @num_pivots:      Pointer to pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
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
    long frame_elem = model->frame_elements;
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

    long batch_cand_ids[4];
    const void *batch_ptrs[4];
    int batch_count = 0;

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
            knn_eval_candidate_cluster_members(
                query_id, query_data, c, d_anchor, anchor_is_sq16,
                home_cluster_id, r_home, model, config, reader,
                cand_buffer, heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                pivots, num_pivots, visited,
                &batch_count, batch_cand_ids, batch_ptrs, telem
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
    if (batch_count > 0)
    {
        double dists[4];
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch_ptrs,
                batch_count, dists, frame_elem
            );
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch_ptrs,
                batch_count, dists, frame_elem
            );
        }
        telem->framedist_calls += (uint64_t)batch_count;
        for (int b = 0; b < batch_count; b++)
        {
            record_neighbor_and_reciprocal(
                query_id, batch_cand_ids[b], dists[b],
                config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    }
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
        knn_visited_check_and_mark(visited, (long)heap->data[h].frame_id);
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

/**
 * knn_cross_seed_frontier() - Populate dynamic search frontier with nearest evaluated anchor seeds.
 * @loc_res:         Cluster locator result with evaluated anchors.
 * @best_c:          Best anchor cluster ID.
 * @query_data:      Query frame pixel data.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @cand_reader:     Candidate frame reader.
 * @cand_buffer:     Candidate frame pixel buffer.
 * @heap:            Max-heap for current query.
 * @frontier:        Output frontier node array.
 * @frontier_count:  Pointer to frontier count.
 * @best_seed_id:    Pointer to best seed frame ID.
 * @best_seed_dist:  Pointer to best seed distance.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */

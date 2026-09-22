/**
 * @file knn_cross_dataset.c
 * @brief Cross-dataset k-NN search driver and orchestrator.
 */

#include "knn_cross_dataset.h"
#include "knn_cross_route.h"
#include "knn_cross_eval.h"
#include <alloca.h>

/**
 * knn_update_trajectory_tracker() - Update thread-local trajectory tracker from heap.
 * @tracker:        Pointer to KnnTrajectoryTracker.
 * @query_id:       Current query index.
 * @best_c:         Best cluster ID.
 * @best_seed_id:   Best seed node frame ID.
 * @best_seed_dist: Distance to best seed node.
 * @heap:           Current query top-k heap.
 */
static inline void knn_update_trajectory_tracker(
    KnnTrajectoryTracker *tracker,
    long                  query_id,
    int                   best_c,
    long                  best_seed_id,
    double                best_seed_dist,
    const KnnMaxHeap     *heap)
{
    if (tracker == NULL)
    {
        return;
    }

    tracker->prev_query_id = query_id;
    tracker->prev_cluster_id = best_c;
    tracker->prev_best_seed = best_seed_id;
    tracker->prev_best_dist = best_seed_dist;

    int copy_k = (heap->count < 64) ? heap->count : 64;
    tracker->num_cached = copy_k;
    for (int j = 0; j < copy_k; j++)
    {
        tracker->cached_ids[j] = knn_heap_get_id(heap, j);
        tracker->cached_dists[j] = knn_heap_get_dist(heap, j);
    }
}

void knn_search_cross_dataset_frame(
    long                  query_id,
    const void *restrict  query_data,
    const KnnModel       *model,
    const KnnConfig      *config,
    KnnFrameReader       *cand_reader,
    void       *restrict  cand_buffer,
    double     *restrict  anchor_dists,
    ClusterScore *restrict scores_buffer,
    KnnMaxHeap           *heap,
    KnnTrajectoryTracker *tracker,
    KnnVisitedTracker    *visited,
    KnnTelemetry *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int best_c = -1;
    double min_d_anchor = 1e20;
    long best_seed_id = -1;
    double best_seed_dist = 1e20;
    int num_seed_pivots = 0;
    long seed_pivot_ids[8];
    double seed_pivot_dists[8];

    for (int c = 0; c < M; c++)
    {
        anchor_dists[c] = -1.0;
    }

    double effective_rlim = 0.0;
    if (config->rlim_cutoff > 0.0)
    {
        effective_rlim = config->rlim_cutoff;
    }
    else if (config->refuse_unclustered)
    {
        effective_rlim = model->model_rlim;
    }

    uint8_t *active_mask = (uint8_t *)alloca((size_t)M);
    int prev_c = (tracker != NULL) ? tracker->prev_cluster_id : -1;
    ClusterLocatorConfig loc_cfg;
    memset(&loc_cfg, 0, sizeof(loc_cfg));
    loc_cfg.max_targets = (config->approx_mode) ? 24 : 32;
    loc_cfg.strict_rlim = (effective_rlim > 0.0) ? config->refuse_unclustered : 0;
    loc_cfg.te4_mode = 1;
    loc_cfg.te5_mode = 1;
    loc_cfg.rlim = effective_rlim;
    loc_cfg.tau_max = knn_heap_peek_max_dist(heap);
    loc_cfg.epsilon = config->epsilon;
    loc_cfg.prev_cluster_id = prev_c;
    loc_cfg.is_double = model->is_double;
    loc_cfg.query_eq16 = visited->query_eq16;
    loc_cfg.eq16_params =
        (config->use_eq16 && model->anchor_eq16_buffer != NULL) ?
            &model->eq16_params : NULL;
    loc_cfg.anchors_eq16_buf = model->anchor_eq16_buffer;
    loc_cfg.query_sq16 = visited->query_sq16;
    loc_cfg.query_sq8 = visited->query_sq8;
    loc_cfg.sq16_params =
        (config->use_sq16 && model->anchor_sq16_buffer != NULL) ?
            &model->sq16_params : NULL;
    loc_cfg.sq8_params =
        (config->use_sq8 && model->anchor_sq8_buffer != NULL) ?
            &model->sq8_params : NULL;
    loc_cfg.anchors_sq16_buf = model->anchor_sq16_buffer;
    loc_cfg.anchors_sq8_buf = model->anchor_sq8_buffer;

    ClusterLocatorResult loc_res;
    loc_res.active_cluster_mask = active_mask;

    int loc_ret = cluster_locate_sample(
        query_data, frame_elem, M, model->anchor_ptrs,
        model->cluster_radii, model->dcc_matrix, &loc_cfg, &loc_res
    );

    telem->framedist_calls += (uint64_t)loc_res.num_evaluated_anchors;
    for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
    {
        anchor_dists[loc_res.evaluated_clusters[e]] = loc_res.evaluated_dists[e];
    }

    if (loc_ret == CLUSTER_LOCATE_REJECTED ||
        (loc_cfg.strict_rlim && loc_res.best_cluster_id < 0))
    {
        telem->out_of_cluster_rejected++;
        return;
    }

    best_c = loc_res.best_cluster_id;
    min_d_anchor = loc_res.best_anchor_dist;
    if (best_c < 0)
    {
        best_c = 0;
    }

    // Push evaluated anchor frames into heap and seed_pivot_ids
    for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
    {
        int c = loc_res.evaluated_clusters[e];
        double d_anc = loc_res.evaluated_dists[e];
        if (c >= 0 && c < M && model->clusters[c].num_members > 0)
        {
            if (model->clusters[c].members[0].r_anchor < 1e-4f)
            {
                long anc_frame = (long)model->clusters[c].members[0].frame_id;
                knn_visited_check_and_mark(visited, anc_frame);
                if (config->rlim_cutoff <= 0.0 || d_anc <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)anc_frame, d_anc);
                }
                if (num_seed_pivots < 8)
                {
                    seed_pivot_ids[num_seed_pivots] = anc_frame;
                    seed_pivot_dists[num_seed_pivots] = d_anc;
                    num_seed_pivots++;
                }
                if (c == best_c)
                {
                    best_seed_id = anc_frame;
                    best_seed_dist = d_anc;
                }
            }
        }
    } // for (int e = 0; ...)

    // Select additional seed in best_c near d_anchor_best
    if (best_c >= 0 && best_c < M)
    {
        const KnnCluster *best_cl = &model->clusters[best_c];
        double d_anchor_best = (anchor_dists[best_c] >= 0.0) ?
            anchor_dists[best_c] : min_d_anchor;

        int mid = find_member_lower_bound(
            best_cl->members, best_cl->num_members, (float)d_anchor_best
        );
        if (mid >= best_cl->num_members)
        {
            mid = best_cl->num_members - 1;
        }
        if (mid > 0)
        {
            long seed_0 = (long)best_cl->members[mid].frame_id;
            if (!knn_visited_check_and_mark(visited, seed_0))
            {
                if (knn_reader_read_frame(cand_reader, seed_0, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    double d0 = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d0 <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)seed_0, d0);
                    }
                    if (d0 < best_seed_dist)
                    {
                        best_seed_dist = d0;
                        best_seed_id = seed_0;
                    }
                    if (num_seed_pivots < 8)
                    {
                        seed_pivot_ids[num_seed_pivots] = seed_0;
                        seed_pivot_dists[num_seed_pivots] = d0;
                        num_seed_pivots++;
                    }
                }
            }
        }
    } // if (best_c >= 0 ...)

    if (best_seed_id < 0 && best_c >= 0 && best_c < M)
    {
        const KnnCluster *best_cl = &model->clusters[best_c];
        if (best_cl->num_members > 0)
        {
            long seed_0 = (long)best_cl->members[0].frame_id;
            knn_visited_check_and_mark(visited, seed_0);
            if (knn_reader_read_frame(cand_reader, seed_0, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d0 = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d0 <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)seed_0, d0);
                }
                best_seed_id = seed_0;
                best_seed_dist = d0;
                if (num_seed_pivots < 8)
                {
                    seed_pivot_ids[num_seed_pivots] = seed_0;
                    seed_pivot_dists[num_seed_pivots] = d0;
                    num_seed_pivots++;
                }
            }
        }
    }

    // Option 2: Sequential Trajectory Seed Warm-Start
    if (config->use_trajectory && tracker != NULL &&
        tracker->prev_query_id == query_id - 1 && tracker->prev_best_seed >= 0)
    {
        long prev_seed = tracker->prev_best_seed;
        if (!knn_visited_check_and_mark(visited, prev_seed))
        {
            if (knn_reader_read_frame(cand_reader, prev_seed, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d_prev = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d_prev <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)prev_seed, d_prev);
                }
                if (d_prev < best_seed_dist)
                {
                    best_seed_dist = d_prev;
                    best_seed_id = prev_seed;
                    if (num_seed_pivots < 8)
                    {
                        seed_pivot_ids[num_seed_pivots] = prev_seed;
                        seed_pivot_dists[num_seed_pivots] = d_prev;
                        num_seed_pivots++;
                    }
                    telem->trajectory_warmstarts++;
                }
                if (prev_seed >= 0 && prev_seed < model->total_dataset_frames)
                {
                    int prev_seed_c = model->frame_cluster_map[prev_seed];
                    if (prev_seed_c >= 0 && prev_seed_c < M && d_prev < min_d_anchor)
                    {
                        best_c = prev_seed_c;
                        min_d_anchor = d_prev;
                    }
                }
            }
        }
    }

    // Graph Search: Dynamic Frontier Search across candidate seeds
    if (model->has_knn_graph && best_seed_id >= 0 && model->graph_indices != NULL &&
        model->graph_distances != NULL)
    {
        // Phase 1 & 2 (Approx mode): Fast Greedy 1-NN Routing + Direct Basin Expansion
        if (config->approx_mode)
        {
            long last_p_id = -1;
            double last_p_dist = -1.0;
            knn_greedy_route_to_basin(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, &last_p_id, &last_p_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

            knn_direct_basin_expansion(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, last_p_id, last_p_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

        }

        // Opportunity 1: In approx mode, if basin expansion has saturated the top-k heap,
        // verify whether any other evaluated candidate cluster could overlap the search ball.
        int skip_frontier = 0;
        if (config->approx_mode && heap->count >= heap->k)
        {
            double current_tau = knn_heap_peek_max_dist(heap);
            double eps_factor = 1.0 + config->epsilon;
            double tau_eff = current_tau / eps_factor;
            double min_other_lb = 1e30;
            for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
            {
                int c = loc_res.evaluated_clusters[e];
                if (c != best_c && c >= 0 && c < M)
                {
                    double d_anc = loc_res.evaluated_dists[e];
                    double r_c = model->cluster_radii[c];
                    double lb = d_anc - r_c;
                    if (lb < min_other_lb)
                    {
                        min_other_lb = lb;
                    }
                }
            }
            if (tau_eff <= min_other_lb)
            {
                skip_frontier = 1;
            }
        }

        if (!skip_frontier)
        {
            FrontierNode frontier[GRAPH_FRONTIER_MAX];
            int frontier_count = 0;

            frontier[0].frame_id = best_seed_id;
            frontier[0].dist = best_seed_dist;
            frontier[0].parent_id = -1;
            frontier[0].parent_dist = -1.0;
            frontier[0].expanded = 0;
            frontier_count = 1;

            knn_cross_seed_frontier(
                &loc_res, best_c, query_data, model, config, cand_reader,
                cand_buffer, heap, frontier, &frontier_count,
                &best_seed_id, &best_seed_dist, visited, telem
            );

            // Phase 2: Local Basin Top-k Collection with Angular Prioritization
            int contained = knn_cross_explore_graph_frontier(
                query_data, model, config, cand_reader, cand_buffer, heap,
                frontier, frontier_count, &best_seed_id, &best_seed_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

            if (contained)
            {
                knn_update_trajectory_tracker(
                    tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
                );
                return;
            }
        }
    }

    // Intra-Cluster Member Search in target cluster (containing best_seed_id)
    knn_cross_eval_intra_cluster(
        best_c, best_seed_id, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists, heap, visited, telem
    );

    for (int j = 0; j < heap->count; j++)
    {
        double d = knn_heap_get_dist(heap, j);
        if (d < best_seed_dist)
        {
            best_seed_dist = d;
            best_seed_id = knn_heap_get_id(heap, j);
        }
    }

    if (best_seed_id >= 0 && best_seed_id < model->total_dataset_frames)
    {
        int seed_c = model->frame_cluster_map[best_seed_id];
        if (seed_c >= 0 && seed_c < M)
        {
            best_c = seed_c;
        }
    }

    if (config->approx_mode && heap->count >= heap->k && best_seed_dist < 1e-5)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        knn_update_trajectory_tracker(
            tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
        );
        return;
    }

    // Inter-Cluster Search on surviving candidate clusters
    knn_cross_eval_inter_clusters(
        &loc_res, best_c, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists,
        scores_buffer, active_mask, heap, visited, telem
    );

    for (int j = 0; j < heap->count; j++)
    {
        double d = knn_heap_get_dist(heap, j);
        if (d < best_seed_dist)
        {
            best_seed_dist = d;
            best_seed_id = knn_heap_get_id(heap, j);
        }
    }

    if (best_seed_id >= 0 && best_seed_id < model->total_dataset_frames)
    {
        int seed_c = model->frame_cluster_map[best_seed_id];
        if (seed_c >= 0 && seed_c < M)
        {
            best_c = seed_c;
        }
    }

    knn_update_trajectory_tracker(
        tracker, query_id, best_c, best_seed_id, best_seed_dist, heap
    );
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

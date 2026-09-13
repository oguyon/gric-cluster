/**
 * @file knn_cross_dataset.c
 * @brief Cross-dataset k-NN query routing, basin expansion, and trajectory tracking.
 */

#include "knn_cross_dataset.h"
#include <alloca.h>

static void knn_cross_seed_frontier(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    const void *restrict        query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    void *restrict              cand_buffer,
    KnnMaxHeap                 *heap,
    FrontierNode               *frontier,
    int                        *frontier_count,
    long                       *best_seed_id,
    double                     *best_seed_dist,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem)
{
    int sorted_eval_clusters[32];
    double sorted_eval_dists[32];
    int num_eval = loc_res->num_evaluated_anchors;
    if (num_eval > 32)
    {
        num_eval = 32;
    }
    for (int e = 0; e < num_eval; e++)
    {
        sorted_eval_clusters[e] = loc_res->evaluated_clusters[e];
        sorted_eval_dists[e] = loc_res->evaluated_dists[e];
    }
    for (int i = 0; i < num_eval - 1; i++)
    {
        for (int j = i + 1; j < num_eval; j++)
        {
            if (sorted_eval_dists[j] < sorted_eval_dists[i])
            {
                double tmp_d = sorted_eval_dists[i];
                sorted_eval_dists[i] = sorted_eval_dists[j];
                sorted_eval_dists[j] = tmp_d;
                int tmp_c = sorted_eval_clusters[i];
                sorted_eval_clusters[i] = sorted_eval_clusters[j];
                sorted_eval_clusters[j] = tmp_c;
            }
        }
    }

    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int max_frontier_seeds = (config->approx_mode) ? 2 : 8;
    for (int e = 0; e < num_eval && *frontier_count < max_frontier_seeds; e++)
    {
        int anc_c = sorted_eval_clusters[e];
        if (anc_c >= 0 && anc_c < M && anc_c != best_c)
        {
            const KnnCluster *anc_cl = &model->clusters[anc_c];
            if (anc_cl->num_members > 0)
            {
                for (int mi = 0; mi < anc_cl->num_members && mi < 2; mi++)
                {
                    long s_frame = (long)anc_cl->members[mi].frame_id;
                    if (knn_visited_check_and_mark(visited, s_frame))
                    {
                        continue;
                    }
                    if (!knn_heap_contains(heap, (int)s_frame))
                    {
                        double tau = knn_heap_peek_max_dist(heap);
                        if (best_seed_dist != NULL && *best_seed_dist > tau)
                        {
                            tau = *best_seed_dist;
                        }
                        if (is_graph_pruned_by_sq16(visited->query_sq16, s_frame, tau,
                                                    model, config, telem) ||
                            is_graph_pruned_by_sq8(visited->query_sq8, s_frame, tau,
                                                   model, config, telem))
                        {
                            continue;
                        }

                        if (knn_reader_read_frame(cand_reader, s_frame, cand_buffer) == 0)
                        {
                            telem->framedist_calls++;
                            telem->graph_seeds_evaluated++;
                            double d = compute_euclidean_distance(
                                query_data, cand_buffer, frame_elem, model->is_double
                            );
                            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                            {
                                knn_heap_push(heap, (int)s_frame, d);
                            }
                            if (d < *best_seed_dist)
                            {
                                *best_seed_id = s_frame;
                                *best_seed_dist = d;
                            }
                            if (frontier_count != NULL && *frontier_count < GRAPH_FRONTIER_MAX)
                            {
                                frontier[*frontier_count].frame_id = s_frame;
                                frontier[*frontier_count].dist = d;
                                frontier[*frontier_count].parent_id = -1;
                                frontier[*frontier_count].parent_dist = -1.0;
                                frontier[*frontier_count].expanded = 0;
                                (*frontier_count)++;
                            }
                        }
                    }
                } // for (int mi = 0; ...)
            }
        }
    } // for (int e = 0; ...)
}

/**
 * knn_greedy_route_to_basin() - Fast Phase 1 greedy 1-NN routing along graph.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_greedy_route_to_basin(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                *last_parent_id,
    double              *last_parent_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem)
{
    long curr_u = *best_seed_id;
    double curr_d = *best_seed_dist;
    long parent_u = (last_parent_id != NULL) ? *last_parent_id : -1;
    double parent_d = (last_parent_dist != NULL) ? *last_parent_dist : -1.0;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    int max_hops = 16;
    int check_k = (graph_k < 16) ? graph_k : 16;

    for (int hop = 0; hop < max_hops; hop++)
    {
        if (curr_d <= 1e-9)
        {
            break;
        }

        const uint32_t *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];
        int improved = 0;
        long next_u = curr_u;
        double next_d = curr_d;

        int cand_order[16];
        float cand_alignment[16];
        for (int i = 0; i < check_k; i++)
        {
            cand_order[i] = i;
            cand_alignment[i] = 0.0f;
        }

        // Inherit direction from parent hop to prioritize forward descent
        if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
            parent_u >= 0 && parent_d > 1e-9 && curr_d > 1e-9)
        {
            int p_idx = -1;
            for (int k = 0; k < graph_k; k++)
            {
                if ((long)neighbors[k] == parent_u)
                {
                    p_idx = k;
                    break;
                }
            }

            if (p_idx >= 0)
            {
                double d_edge_p = (double)n_dists[p_idx];
                if (d_edge_p > 1e-9)
                {
                    double cos_alpha = (curr_d * curr_d + d_edge_p * d_edge_p -
                                        parent_d * parent_d) /
                                       (2.0 * curr_d * d_edge_p);
                    cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                    double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                    for (int m = 0; m < check_k; m++)
                    {
                        if (m == p_idx)
                        {
                            cand_alignment[m] = -1.0f;
                            continue;
                        }

                        double r_m = (double)n_dists[m];
                        float d_mut = knn_get_mutual_dist(model, curr_u, p_idx, m);
                        if (d_mut > 0.0f && r_m > 1e-9)
                        {
                            double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                              (double)d_mut * d_mut) /
                                             (2.0 * d_edge_p * r_m);
                            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                            double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                            cand_alignment[m] = (float)(cos_alpha * cos_phi +
                                                        sin_alpha * sin_phi);
                        }
                    }

                    for (int i = 0; i < check_k - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < check_k; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                }
            }
        }

        for (int step = 0; step < check_k; step++)
        {
            int k_idx = cand_order[step];
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            if (knn_visited_is_visited(visited, nb_id))
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_d - d_edge);
            if (lb_edge >= curr_d)
            {
                telem->graph_edges_pruned++;
                continue;
            }

            double routing_tau = next_d;
            double tau_heap = knn_heap_peek_max_dist(heap);
            if (tau_heap > routing_tau)
            {
                routing_tau = tau_heap;
            }
            if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, routing_tau,
                                        model, config, telem) ||
                is_graph_pruned_by_sq8(visited->query_sq8, nb_id, routing_tau,
                                       model, config, telem))
            {
                continue;
            }

            knn_visited_mark(visited, nb_id);

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_dist = d;
                    *best_seed_id = nb_id;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (d < next_d)
                {
                    next_u = nb_id;
                    next_d = d;
                    improved = 1;
                    break;
                }
            }
        } // for (int step = 0; ...)

        if (!improved)
        {
            break;
        }

        parent_u = curr_u;
        parent_d = curr_d;
        curr_u = next_u;
        curr_d = next_d;
    } // for (int hop = 0; ...)

    *best_seed_id = curr_u;
    *best_seed_dist = curr_d;
    if (last_parent_id != NULL)
    {
        *last_parent_id = parent_u;
    }
    if (last_parent_dist != NULL)
    {
        *last_parent_dist = parent_d;
    }
}

/**
 * knn_direct_basin_expansion() - Direct Option B1 neighbor expansion around 1-NN basin.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID (u*).
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_direct_basin_expansion(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                 parent_id,
    double               parent_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem)
{
    long center_u = *best_seed_id;
    double center_d = *best_seed_dist;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    const uint32_t *neighbors = &model->graph_indices[center_u * (long)graph_k];
    const float    *n_dists = &model->graph_distances[center_u * (long)graph_k];

    float local_lb2[64];
    int cand_order[64];
    float cand_alignment[64];
    int eval_lim = (graph_k < 64) ? graph_k : 64;

    for (int i = 0; i < eval_lim; i++)
    {
        cand_order[i] = i;
        cand_alignment[i] = 0.0f;
        local_lb2[i] = 0.0f;
    }

    int first_eval_done = 0;

    // Option 1: Inter-Node Direction Inheritance from greedy routing hop
    if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
        parent_id >= 0 && parent_dist > 1e-9 && center_d > 1e-9)
    {
        int p_idx = -1;
        for (int k = 0; k < graph_k; k++)
        {
            if ((long)neighbors[k] == parent_id)
            {
                p_idx = k;
                break;
            }
        }

        if (p_idx >= 0)
        {
            double d_edge_p = (double)n_dists[p_idx];
            if (d_edge_p > 1e-9)
            {
                double cos_alpha = (center_d * center_d + d_edge_p * d_edge_p -
                                    parent_dist * parent_dist) /
                                   (2.0 * center_d * d_edge_p);
                cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                for (int m = 0; m < eval_lim; m++)
                {
                    if (m == p_idx)
                    {
                        cand_alignment[m] = -1.0f;
                        continue;
                    }

                    double r_m = (double)n_dists[m];
                    float d_mut = knn_get_mutual_dist(model, center_u, p_idx, m);
                    if (d_mut > 0.0f && r_m > 1e-9)
                    {
                        double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                          (double)d_mut * d_mut) /
                                         (2.0 * d_edge_p * r_m);
                        cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                        double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                        double cos_th = cos_alpha * cos_phi + sin_alpha * sin_phi;
                        cand_alignment[m] = (float)fmin(1.0, cos_th);

                        double lb2_ang = center_d * center_d + r_m * r_m -
                                         2.0 * center_d * r_m * fmin(1.0, cos_th);
                        double lb_tri = fabs(parent_dist - (double)d_mut);
                        double lb2_tri = lb_tri * lb_tri;
                        local_lb2[m] = (float)fmax(lb2_ang, lb2_tri);
                    }
                }

                for (int i = 0; i < eval_lim - 1; i++)
                {
                    int best_p = i;
                    float max_a = cand_alignment[cand_order[i]];
                    for (int j = i + 1; j < eval_lim; j++)
                    {
                        if (cand_alignment[cand_order[j]] > max_a)
                        {
                            max_a = cand_alignment[cand_order[j]];
                            best_p = j;
                        }
                    }
                    if (best_p != i)
                    {
                        int tmp = cand_order[i];
                        cand_order[i] = cand_order[best_p];
                        cand_order[best_p] = tmp;
                    }
                }
                first_eval_done = 1;
            }
        }
    }

    long improving_node = -1;
    int non_improving_streak = 0;
    const int max_streak = 8;

    for (int step = 0; step < eval_lim; step++)
    {
        int k_idx = cand_order[step];
        long nb_id = (long)neighbors[k_idx];
        if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == center_u)
        {
            continue;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_eff = current_tau / eps_factor;
        double tau_eff2 = tau_eff * tau_eff;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
        }

        if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
            (double)local_lb2[k_idx] >= tau_eff2)
        {
            telem->angular_pruned++;
            if (heap->count >= heap->k)
            {
                non_improving_streak++;
                if (non_improving_streak >= max_streak)
                {
                    break;
                }
            }
            continue;
        }

        if (knn_visited_check_and_mark(visited, nb_id))
        {
            continue;
        }

        double d_edge = (double)n_dists[k_idx];
        double lb_edge = fabs(center_d - d_edge);
        if (heap->count >= heap->k &&
            (lb_edge >= current_tau / eps_factor ||
             (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
        {
            telem->graph_edges_pruned++;
            non_improving_streak++;
            if (non_improving_streak >= max_streak)
            {
                break;
            }
            continue;
        }

        if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
                                    model, config, telem) ||
            is_graph_pruned_by_sq8(visited->query_sq8, nb_id, current_tau,
                                   model, config, telem))
        {
            if (heap->count >= heap->k)
            {
                non_improving_streak++;
                if (non_improving_streak >= max_streak)
                {
                    break;
                }
            }
            continue;
        }

        if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
        {
            telem->framedist_calls++;
            telem->graph_seeds_evaluated++;
            double d = compute_euclidean_distance(
                query_data, cand_buffer, frame_elem, model->is_double
            );
            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
            {
                knn_heap_push(heap, (int)nb_id, d);
            }
            if (heap->count >= heap->k)
            {
                if (d < current_tau)
                {
                    non_improving_streak = 0;
                }
                else
                {
                    non_improving_streak++;
                    if (non_improving_streak >= max_streak)
                    {
                        break;
                    }
                }
            }
            if (d < *best_seed_dist)
            {
                *best_seed_dist = d;
                *best_seed_id = nb_id;
                improving_node = nb_id;
            }
            if (*num_seed_pivots < 8)
            {
                seed_pivot_ids[*num_seed_pivots] = nb_id;
                seed_pivot_dists[*num_seed_pivots] = d;
                (*num_seed_pivots)++;
            }

            if (model->graph_mutual_dists != NULL)
            {
                for (int rem = step + 1; rem < eval_lim; rem++)
                {
                    int m_idx = cand_order[rem];
                    if (m_idx < 64)
                    {
                        double r_m = (double)n_dists[m_idx];
                        float d_mut = knn_get_mutual_dist(
                            model, center_u, k_idx, m_idx
                        );
                        if (d_mut > 0.0f)
                        {
                            // Opportunity 3: Mutual distance triangle lower bound
                            double lb_tri = fabs(d - (double)d_mut);
                            double lb2_tri = lb_tri * lb_tri;
                            if (lb2_tri > (double)local_lb2[m_idx])
                            {
                                local_lb2[m_idx] = (float)lb2_tri;
                            }

                            if (config->use_angular_bound)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    center_d, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }
                            }

                            if (!first_eval_done && config->use_angular_bound &&
                                center_d > 1e-9 && d_edge > 1e-9 && r_m > 1e-9)
                            {
                                double cos_th = (center_d * center_d +
                                                 d_edge * d_edge - d * d) /
                                                (2.0 * center_d * d_edge);
                                cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                  (double)d_mut * d_mut) /
                                                 (2.0 * d_edge * r_m);
                                cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                sin_th * sin_phi);
                            }
                        }
                    }
                } // for (int rem = step + 1; ...)

                if (!first_eval_done)
                {
                    first_eval_done = 1;
                    for (int i = step + 1; i < eval_lim - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < eval_lim; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                }
            } // if (config->use_angular_bound ...)
        }
    } // for (int step = 0; ...)

    // Phase 2b: Also check top neighbors of the closest discovered nodes in heap
    int top_seeds[3];
    int num_top_seeds = 0;
    if (improving_node >= 0)
    {
        top_seeds[num_top_seeds++] = (int)improving_node;
    }

    for (int i = 0; i < heap->count && num_top_seeds < 3; i++)
    {
        int cand = knn_heap_get_id(heap, i);
        if (cand != (int)center_u && (num_top_seeds == 0 || cand != top_seeds[0]))
        {
            top_seeds[num_top_seeds++] = cand;
        }
    }

    for (int ts = 0; ts < num_top_seeds; ts++)
    {
        long s_node = (long)top_seeds[ts];
        if (s_node >= 0 && s_node < model->total_dataset_frames)
        {
            double s_dist = -1.0;
            for (int h = 0; h < heap->count; h++)
            {
                if (knn_heap_get_id(heap, h) == (int)s_node)
                {
                    s_dist = knn_heap_get_dist(heap, h);
                    break;
                }
            }

            const uint32_t *s_nb = &model->graph_indices[s_node * (long)graph_k];
            const float    *s_dst = &model->graph_distances[s_node * (long)graph_k];
            int check_k = (graph_k < 12) ? graph_k : 12;

            int cand_order_s[12];
            float cand_alignment_s[12];
            float local_lb2_s[12];
            for (int i = 0; i < check_k; i++)
            {
                cand_order_s[i] = i;
                cand_alignment_s[i] = 0.0f;
                local_lb2_s[i] = 0.0f;
            }

            if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
                center_u >= 0 && center_d > 1e-9 && s_dist > 1e-9)
            {
                int p_idx = -1;
                for (int k = 0; k < graph_k; k++)
                {
                    if ((long)s_nb[k] == center_u)
                    {
                        p_idx = k;
                        break;
                    }
                }

                if (p_idx >= 0)
                {
                    double d_edge_p = (double)s_dst[p_idx];
                    if (d_edge_p > 1e-9)
                    {
                        double cos_alpha = (s_dist * s_dist + d_edge_p * d_edge_p -
                                            center_d * center_d) /
                                           (2.0 * s_dist * d_edge_p);
                        cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                        double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                        for (int m = 0; m < check_k; m++)
                        {
                            if (m == p_idx)
                            {
                                cand_alignment_s[m] = -1.0f;
                                continue;
                            }

                            double r_m = (double)s_dst[m];
                            float d_mut = knn_get_mutual_dist(model, s_node, p_idx, m);
                            if (d_mut > 0.0f && r_m > 1e-9)
                            {
                                double cos_phi = (d_edge_p * d_edge_p + r_m * r_m -
                                                  (double)d_mut * d_mut) /
                                                 (2.0 * d_edge_p * r_m);
                                cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                double cos_th = cos_alpha * cos_phi + sin_alpha * sin_phi;
                                cand_alignment_s[m] = (float)fmin(1.0, cos_th);

                                double lb2_ang = s_dist * s_dist + r_m * r_m -
                                                 2.0 * s_dist * r_m * fmin(1.0, cos_th);
                                double lb_tri = fabs(center_d - (double)d_mut);
                                double lb2_tri = lb_tri * lb_tri;
                                local_lb2_s[m] = (float)fmax(lb2_ang, lb2_tri);
                            }
                        }

                        for (int i = 0; i < check_k - 1; i++)
                        {
                            int best_p = i;
                            float max_a = cand_alignment_s[cand_order_s[i]];
                            for (int j = i + 1; j < check_k; j++)
                            {
                                if (cand_alignment_s[cand_order_s[j]] > max_a)
                                {
                                    max_a = cand_alignment_s[cand_order_s[j]];
                                    best_p = j;
                                }
                            }
                            if (best_p != i)
                            {
                                int tmp = cand_order_s[i];
                                cand_order_s[i] = cand_order_s[best_p];
                                cand_order_s[best_p] = tmp;
                            }
                        }
                    }
                }
            }

            for (int step = 0; step < check_k; step++)
            {
                int i = cand_order_s[step];
                long nb2 = (long)s_nb[i];
                if (nb2 < 0 || nb2 >= model->total_dataset_frames)
                {
                    continue;
                }

                double current_tau = knn_heap_peek_max_dist(heap);
                double tau_eff = current_tau / eps_factor;
                double tau_eff2 = tau_eff * tau_eff;
                if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
                {
                    tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
                }

                if ((heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
                    (double)local_lb2_s[i] >= tau_eff2)
                {
                    telem->angular_pruned++;
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb2))
                {
                    continue;
                }

                double d_edge = (double)s_dst[i];
                if (heap->count >= heap->k && d_edge >= tau_eff)
                {
                    telem->graph_edges_pruned++;
                    continue;
                }

                if (is_graph_pruned_by_sq16(visited->query_sq16, nb2, current_tau,
                                            model, config, telem) ||
                    is_graph_pruned_by_sq8(visited->query_sq8, nb2, current_tau,
                                           model, config, telem))
                {
                    continue;
                }

                if (knn_reader_read_frame(cand_reader, nb2, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    telem->graph_seeds_evaluated++;
                    double d2 = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d2 <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)nb2, d2);
                    }
                    if (d2 < *best_seed_dist)
                    {
                        *best_seed_dist = d2;
                        *best_seed_id = nb2;
                    }
                }
            } // for (int step = 0; ...)
        }
    } // for (int ts = 0; ...)
}

/**
 * knn_cross_explore_graph_frontier() - Explore KNN graph frontier for cross-dataset query.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @heap:             Max-heap for current query.
 * @frontier:         Frontier node array.
 * @frontier_count:   Frontier node count.
 * @best_seed_id:     Pointer to best seed ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed pivot IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Return: 1 if global containment criterion was satisfied, 0 otherwise.
 */
static int knn_cross_explore_graph_frontier(
    const void *restrict query_data,
    const KnnModel      *model,
    const KnnConfig     *config,
    KnnFrameReader      *cand_reader,
    void *restrict       cand_buffer,
    KnnMaxHeap          *heap,
    FrontierNode        *frontier,
    int                  frontier_count,
    long                *best_seed_id,
    double              *best_seed_dist,
    long                *seed_pivot_ids,
    double              *seed_pivot_dists,
    int                 *num_seed_pivots,
    KnnVisitedTracker   *visited,
    KnnTelemetry *restrict telem)
{
    int graph_k = model->graph_k;
    int target_k = heap->k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int max_expansions = (config->approx_mode) ? 4 : 16;
    int max_eval_k = (config->approx_mode) ? ((graph_k < 12) ? graph_k : 12) : graph_k;
    int expansions_done = 0;
    int stagnation_count = 0;
    double prev_tau = 1e20;

    while ((expansions_done < max_expansions || heap->count < target_k) &&
           expansions_done < GRAPH_FRONTIER_MAX)
    {
        int best_idx = -1;
        double min_f_dist = 1e20;
        for (int fi = 0; fi < frontier_count; fi++)
        {
            if (!frontier[fi].expanded && frontier[fi].dist < min_f_dist)
            {
                min_f_dist = frontier[fi].dist;
                best_idx = fi;
            }
        }

        if (best_idx < 0)
        {
            break;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        if (heap->count >= target_k && min_f_dist > current_tau / eps_factor)
        {
            break;
        }

        if (config->approx_mode && heap->count >= target_k)
        {
            if (current_tau >= prev_tau * 0.9999)
            {
                stagnation_count++;
                if (stagnation_count >= 3)
                {
                    break;
                }
            }
            else
            {
                stagnation_count = 0;
                prev_tau = current_tau;
            }
        }

        frontier[best_idx].expanded = 1;
        expansions_done++;

        long curr_u = frontier[best_idx].frame_id;
        double curr_u_dist = frontier[best_idx].dist;
        long parent_id = frontier[best_idx].parent_id;
        double parent_dist = frontier[best_idx].parent_dist;
        if (curr_u_dist < *best_seed_dist)
        {
            *best_seed_id = curr_u;
            *best_seed_dist = curr_u_dist;
        }
        uint32_t const *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];

        float local_lb2[64];
        int cand_order[64];
        float cand_alignment[64];
        int eval_lim = (max_eval_k < 64) ? max_eval_k : 64;
        for (int i = 0; i < eval_lim; i++)
        {
            cand_order[i] = i;
            cand_alignment[i] = 0.0f;
            local_lb2[i] = 0.0f;
        }

        int first_eval_done = 0;

        // Option 1: Inter-Node Direction Inheritance from parent hop
        if (config->use_angular_bound && model->graph_mutual_dists != NULL &&
            parent_id >= 0 && parent_dist > 1e-9 && curr_u_dist > 1e-9)
        {
            int p_idx = -1;
            for (int k = 0; k < graph_k; k++)
            {
                if ((long)neighbors[k] == parent_id)
                {
                    p_idx = k;
                    break;
                }
            }

            if (p_idx >= 0)
            {
                double d_edge_p = (double)n_dists[p_idx];
                if (d_edge_p > 1e-9)
                {
                    double cos_alpha = (curr_u_dist * curr_u_dist +
                                        d_edge_p * d_edge_p -
                                        parent_dist * parent_dist) /
                                       (2.0 * curr_u_dist * d_edge_p);
                    cos_alpha = fmax(-1.0, fmin(1.0, cos_alpha));
                    double sin_alpha = sqrt(fmax(0.0, 1.0 - cos_alpha * cos_alpha));

                    for (int m = 0; m < eval_lim; m++)
                    {
                        if (m == p_idx)
                        {
                            cand_alignment[m] = -1.0f;
                            continue;
                        }

                        double r_m = (double)n_dists[m];
                        float d_mut = knn_get_mutual_dist(
                            model, curr_u, p_idx, m
                        );
                        if (d_mut > 0.0f && r_m > 1e-9)
                        {
                            double cos_phi = (d_edge_p * d_edge_p +
                                              r_m * r_m -
                                              (double)d_mut * d_mut) /
                                             (2.0 * d_edge_p * r_m);
                            cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                            double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                            double cos_th = cos_alpha * cos_phi +
                                            sin_alpha * sin_phi;
                            cand_alignment[m] = (float)fmin(1.0, cos_th);

                            double lb2_ang = curr_u_dist * curr_u_dist +
                                             r_m * r_m -
                                             2.0 * curr_u_dist * r_m *
                                             fmin(1.0, cos_th);
                            double lb_tri = fabs(parent_dist - (double)d_mut);
                            double lb2_tri = lb_tri * lb_tri;
                            local_lb2[m] = (float)fmax(lb2_ang, lb2_tri);
                        }
                    }

                    for (int i = 0; i < eval_lim - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < eval_lim; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                    first_eval_done = 1;
                }
            }
        }

        for (int step = 0; step < eval_lim; step++)
        {
            int k_idx = cand_order[step];
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double tau_eff2 = tau_eff * tau_eff;
            if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
            {
                tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
            }

            if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
                (double)local_lb2[k_idx] >= tau_eff2)
            {
                telem->angular_pruned++;
                continue;
            }

            if (knn_visited_check_and_mark(visited, nb_id))
            {
                continue;
            }

            int already_in_frontier = 0;
            for (int fi = 0; fi < frontier_count; fi++)
            {
                if (frontier[fi].frame_id == nb_id)
                {
                    already_in_frontier = 1;
                    break;
                }
            }
            if (already_in_frontier)
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_u_dist - d_edge);
            current_tau = knn_heap_peek_max_dist(heap);

            if (heap->count >= heap->k &&
                (lb_edge >= current_tau / eps_factor ||
                 (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            double sq_tau = current_tau;
            if (best_seed_dist != NULL && *best_seed_dist > sq_tau)
            {
                sq_tau = *best_seed_dist;
            }
            if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, sq_tau,
                                        model, config, telem) ||
                is_graph_pruned_by_sq8(visited->query_sq8, nb_id, sq_tau,
                                       model, config, telem))
            {
                continue;
            }

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem, model->is_double
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_id = nb_id;
                    *best_seed_dist = d;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                {
                    for (int rem = step + 1; rem < eval_lim; rem++)
                    {
                        int m_idx = cand_order[rem];
                        if (m_idx < 64)
                        {
                            double r_m = (double)n_dists[m_idx];
                            float d_mut = knn_get_mutual_dist(
                                model, curr_u, k_idx, m_idx
                            );
                            if (d_mut > 0.0f)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    curr_u_dist, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }

                                if (!first_eval_done && curr_u_dist > 1e-9 &&
                                    d_edge > 1e-9 && r_m > 1e-9)
                                {
                                    double cos_th = (curr_u_dist * curr_u_dist +
                                                     d_edge * d_edge - d * d) /
                                                    (2.0 * curr_u_dist * d_edge);
                                    cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                    double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                    double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                      (double)d_mut * d_mut) /
                                                     (2.0 * d_edge * r_m);
                                    cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                    cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                    sin_th * sin_phi);
                                }
                            }
                        }
                    } // for (int rem = step + 1; ...)

                    if (!first_eval_done)
                    {
                        first_eval_done = 1;
                        for (int i = step + 1; i < eval_lim - 1; i++)
                        {
                            int best_p = i;
                            float max_a = cand_alignment[cand_order[i]];
                            for (int j = i + 1; j < eval_lim; j++)
                            {
                                if (cand_alignment[cand_order[j]] > max_a)
                                {
                                    max_a = cand_alignment[cand_order[j]];
                                    best_p = j;
                                }
                            }
                            if (best_p != i)
                            {
                                int tmp = cand_order[i];
                                cand_order[i] = cand_order[best_p];
                                cand_order[best_p] = tmp;
                            }
                        }
                    }
                }

                current_tau = knn_heap_peek_max_dist(heap);
                if ((d < current_tau / eps_factor || heap->count < heap->k) &&
                    frontier_count < GRAPH_FRONTIER_MAX)
                {
                    frontier[frontier_count].frame_id = nb_id;
                    frontier[frontier_count].dist = d;
                    frontier[frontier_count].parent_id = curr_u;
                    frontier[frontier_count].parent_dist = curr_u_dist;
                    frontier[frontier_count].expanded = 0;
                    frontier_count++;
                }
            }
        } // for (int step = 0; ...)
    } // while (expansions_done ...)

    // Global Containment Criterion via Local k-Ball Radius R_k(best_seed_id)
    if (*best_seed_id >= 0 && heap->count >= heap->k)
    {
        double current_tau = knn_heap_peek_max_dist(heap);
        double r_graph_ball =
            (double)model->graph_distances[*best_seed_id * (long)graph_k + (graph_k - 1)];

        if (r_graph_ball > 0.0 && (*best_seed_dist + current_tau / eps_factor) <= r_graph_ball)
        {
            const uint32_t *neighbors = &model->graph_indices[*best_seed_id * (long)graph_k];
            const float *n_dists = &model->graph_distances[*best_seed_id * (long)graph_k];

            float local_lb2_gc[64];
            int eval_gc_lim = (graph_k < 64) ? graph_k : 64;
            for (int i = 0; i < eval_gc_lim; i++)
            {
                local_lb2_gc[i] = 0.0f;
            }

            for (int k_idx = 0; k_idx < graph_k; k_idx++)
            {
                long nb_id = (long)neighbors[k_idx];
                if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == *best_seed_id)
                {
                    continue;
                }

                current_tau = knn_heap_peek_max_dist(heap);
                double tau_eff = current_tau / eps_factor;
                double tau_eff2 = tau_eff * tau_eff;
                if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
                {
                    tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
                }

                if (k_idx < 64 && (double)local_lb2_gc[k_idx] >= tau_eff2)
                {
                    telem->angular_pruned++;
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb_id))
                {
                    continue;
                }
                if (knn_heap_contains(heap, (int)nb_id))
                {
                    continue;
                }

                double d_edge = (double)n_dists[k_idx];
                double lb_edge = fabs(*best_seed_dist - d_edge);
                current_tau = knn_heap_peek_max_dist(heap);

                if (lb_edge >= current_tau / eps_factor ||
                    (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff))
                {
                    telem->graph_edges_pruned++;
                    continue;
                }

                if (is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
                                            model, config, telem) ||
                    is_graph_pruned_by_sq8(visited->query_sq8, nb_id, current_tau,
                                           model, config, telem))
                {
                    continue;
                }

                if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    telem->graph_seeds_evaluated++;
                    double d = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem, model->is_double
                    );
                    if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)nb_id, d);
                    }

                    if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                    {
                        for (int m_idx = k_idx + 1; m_idx < graph_k; m_idx++)
                        {
                            if (m_idx < 64)
                            {
                                double r_m = (double)n_dists[m_idx];
                                float d_mut = knn_get_mutual_dist(
                                    model, *best_seed_id, k_idx, m_idx
                                );
                                if (d_mut > 0.0f)
                                {
                                    double lb2_m = compute_angular_lower_bound_sq(
                                        *best_seed_dist, d_edge, r_m, d, (double)d_mut
                                    );
                                    if (lb2_m > (double)local_lb2_gc[m_idx])
                                    {
                                        local_lb2_gc[m_idx] = (float)lb2_m;
                                    }
                                }
                            }
                        } // for (int m_idx = ...)
                    }
                }
            } // for (int k_idx = 0; ...)

            telem->level1_clusters_pruned += (uint64_t)model->num_clusters;
            telem->global_containment_hits++;
            return 1;
        } // if contained
    } // Global Containment

    return 0;
}

/**
 * knn_cross_eval_intra_cluster() - Search target cluster containing best seed for cross dataset.
 * @best_c:           Initial best anchor cluster.
 * @best_seed_id:     Best seed frame ID.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Computed distances to seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_cross_eval_intra_cluster(
    int                    best_c,
    long                   best_seed_id,
    double                 min_d_anchor,
    int                    num_seed_pivots,
    const long            *seed_pivot_ids,
    const double          *seed_pivot_dists,
    const void *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    void *restrict         cand_buffer,
    double        *restrict anchor_dists,
    KnnMaxHeap            *heap,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int target_c = best_c;
    if (best_seed_id >= 0 && model->frame_cluster_map != NULL)
    {
        int mapped_c = (int)model->frame_cluster_map[best_seed_id];
        if (mapped_c >= 0 && mapped_c < M)
        {
            target_c = mapped_c;
        }
    }

    if (target_c >= 0 && target_c < M)
    {
        const KnnCluster *target_cl = &model->clusters[target_c];
        telem->total_candidates_considered += (uint64_t)target_cl->num_members;

        double d_anchor_target = anchor_dists[target_c];
        if (d_anchor_target < 0.0)
        {
            telem->framedist_calls++;
            d_anchor_target = compute_euclidean_distance(
                query_data, target_cl->anchor_data, frame_elem, model->is_double
            );
            anchor_dists[target_c] = d_anchor_target;
        }

        int start_m = 0;
        int end_m = target_cl->num_members;
        double tau_init = knn_heap_peek_max_dist(heap);
        if (tau_init < 1e20)
        {
            float r_min = (float)fmax(0.0, d_anchor_target - tau_init / eps_factor);
            float r_max = (float)(d_anchor_target + tau_init / eps_factor);
            start_m = find_member_lower_bound(
                target_cl->members, target_cl->num_members, r_min
            );
            end_m = find_member_upper_bound(
                target_cl->members, target_cl->num_members, r_max
            );
            telem->level3_annular_pruned +=
                (uint64_t)start_m + (uint64_t)(target_cl->num_members - end_m);
        }

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)target_cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                continue;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double r_cand = (double)target_cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor_target + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor_target + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor_target, r_cand, 0.0, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
            {
                continue;
            }

            long rep_id = (model->frame_to_unique_map != NULL) ?
                          model->frame_to_unique_map[cand_id] : cand_id;
            if (visited != NULL && visited->rep_tags != NULL &&
                visited->rep_tags[rep_id] == visited->epoch)
            {
                telem->memo_hits++;
                double d = (double)visited->rep_dists[rep_id];
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
                continue;
            }

            const void *cand_data = NULL;
            if (cand_reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)cand_reader->memory_data +
                            (size_t)cand_id * (size_t)frame_elem * el_sz;
            }
            else if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, frame_elem, model->is_double
                );
                if (visited != NULL && visited->rep_tags != NULL)
                {
                    visited->rep_dists[rep_id] = (float)d;
                    visited->rep_tags[rep_id] = visited->epoch;
                }
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // if (target_c >= 0 ...)
}

/**
 * knn_cross_eval_inter_clusters() - Evaluate inter-cluster candidate members for cross dataset.
 * @loc_res:          Cluster locator results.
 * @best_c:           Best anchor cluster index.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Distances to evaluated seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @scores_buffer:    Scratch buffer for candidate cluster sorting.
 * @active_mask:      Mask of surviving clusters.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_cross_eval_inter_clusters(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    double                      min_d_anchor,
    int                         num_seed_pivots,
    const long                 *seed_pivot_ids,
    const double               *seed_pivot_dists,
    const void *restrict        query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    void *restrict              cand_buffer,
    double *restrict            anchor_dists,
    ClusterScore *restrict      scores_buffer,
    uint8_t                    *active_mask,
    KnnMaxHeap                 *heap,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);
    int num_cand_clusters = 0;

    for (int q = 0; q < M; q++)
    {
        if (q == best_c || model->clusters[q].num_members == 0)
        {
            continue;
        }

        if (config->rlim_cutoff > 0.0 && active_mask != NULL && !active_mask[q])
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // 3P test against all evaluated anchor pivots using tightened tau_k
        int pruned_3p = 0;
        double r_q = model->cluster_radii[q];
        for (int e = 0; e < loc_res->num_evaluated_anchors; e++)
        {
            int p = loc_res->evaluated_clusters[e];
            double d_p = loc_res->evaluated_dists[e];
            double dcc = model->dcc_matrix[p * M + q];
            double lb1 = dcc - r_q - d_p;
            double lb2 = d_p - dcc - r_q;
            double lb = (lb1 > lb2) ? lb1 : lb2;

            if (lb >= current_tau / eps_factor ||
                (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
            {
                active_mask[q] = 0;
                pruned_3p = 1;
                break;
            }
        }

        if (pruned_3p)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (anchor_dists[q] < 0.0)
        {
            telem->framedist_calls++;
            anchor_dists[q] = compute_euclidean_distance(
                query_data, model->clusters[q].anchor_data, frame_elem, model->is_double
            );
        }

        double d_a = anchor_dists[q];
        double lb = d_a - r_q;
        if (lb < 0.0)
        {
            lb = 0.0;
        }

        if (lb >= current_tau / eps_factor ||
            (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = d_a;
        num_cand_clusters++;
    } // for (int q = 0; ...)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    // Inter-Cluster Search on surviving candidate clusters
    int max_cands = (config->approx_mode) ? 4 : num_cand_clusters;
    for (int idx = 0; idx < num_cand_clusters && idx < max_cands; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        current_tau = knn_heap_peek_max_dist(heap);

        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (config->rlim_cutoff > 0.0 && lb_cluster >= config->rlim_cutoff)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        const KnnCluster *cl = &model->clusters[q];
        double d_anchor = anchor_dists[q];
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

        double dcc_best = model->dcc_matrix[best_c * M + q];

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            tau_eff = current_tau / eps_factor;
            double r_cand = (double)cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor, r_cand, dcc_best, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                         model, config, telem) ||
                is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                        model, config, telem))
            {
                continue;
            }

            long rep_id = (model->frame_to_unique_map != NULL) ?
                          model->frame_to_unique_map[cand_id] : cand_id;
            if (visited != NULL && visited->rep_tags != NULL &&
                visited->rep_tags[rep_id] == visited->epoch)
            {
                telem->memo_hits++;
                double d = (double)visited->rep_dists[rep_id];
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
                continue;
            }

            const void *cand_data = NULL;
            if (cand_reader->memory_data != NULL)
            {
                size_t el_sz = model->is_double ? sizeof(double) : sizeof(float);
                cand_data = (const char *)cand_reader->memory_data +
                            (size_t)cand_id * (size_t)frame_elem * el_sz;
            }
            else if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                cand_data = cand_buffer;
            }
            if (cand_data != NULL)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_data, frame_elem, model->is_double
                );
                if (visited != NULL && visited->rep_tags != NULL)
                {
                    visited->rep_dists[rep_id] = (float)d;
                    visited->rep_tags[rep_id] = visited->epoch;
                }
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // for (int idx = 0; ...)
}

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

/**
 * knn_search_cross_dataset_frame() - Metric-pruned k-NN search for external query frame.
 * @query_id:      Index of query frame.
 * @query_data:    Pixel buffer of query frame.
 * @model:         Active KnnModel.
 * @config:        Active KnnConfig.
 * @cand_reader:   Thread-local candidate frame reader.
 * @cand_buffer:   Scratch buffer for candidate frame pixels.
 * @anchor_dists:  Scratch buffer for query-to-anchor distances [num_clusters].
 * @scores_buffer: Scratch buffer for cluster sorting [num_clusters].
 * @heap:          KnnMaxHeap structure for this query frame.
 * @tracker:       Thread-local trajectory tracker for sequential continuity.
 * @visited:       Per-query frame visited tracker.
 * @telem:         Thread-local KnnTelemetry.
 */
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

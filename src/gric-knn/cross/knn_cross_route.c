/**
 * @file knn_cross_route.c
 * @brief Cross-dataset graph frontier routing and basin expansion.
 *
 * Implements graph-guided path finding for locating nearest cluster regions in
 * cross-dataset search. Functions in this file seed the routing priority queue with
 * evaluated anchor nodes, execute greedy downhill hops along the cluster proximity graph,
 * expand candidate basins, and traverse multi-hop graph edges while updating pivot bounds.
 */

#include "knn_cross_route.h"
#include <alloca.h>
#include <string.h>

/**
 * knn_cross_seed_frontier() - Populate search frontier with nearest evaluated anchor seeds.
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
 *
 * Populates the initial search frontier with seed frames from nearest evaluated cluster anchors:
 * 1. Anchor sorting: Sorts evaluated anchor clusters from the locator in ascending distance order.
 * 2. Seed selection: Extracts representative frames (cluster medoid or high-density member) from
 *    the top-ranked clusters.
 * 3. Distance evaluation: Computes exact query-to-seed distances and adds unvisited seeds to the
 *    frontier array and query max-heap.
 * 4. Tracking: Updates best_seed_id and best_seed_dist with the closest seed discovered.
 */
void knn_cross_seed_frontier(
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
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int max_frontier_seeds = (config->approx_mode) ? 2 : 8;
    for (int e = 0; e < num_eval && *frontier_count < max_frontier_seeds; e++)
    {
        /* Partial selection sort: locate smallest remaining element and swap once */
        int min_idx = e;
        for (int j = e + 1; j < num_eval; j++)
        {
            if (sorted_eval_dists[j] < sorted_eval_dists[min_idx])
            {
                min_idx = j;
            }
        }
        if (min_idx != e)
        {
            double tmp_d = sorted_eval_dists[e];
            sorted_eval_dists[e] = sorted_eval_dists[min_idx];
            sorted_eval_dists[min_idx] = tmp_d;
            int tmp_c = sorted_eval_clusters[e];
            sorted_eval_clusters[e] = sorted_eval_clusters[min_idx];
            sorted_eval_clusters[min_idx] = tmp_c;
        }

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
                        if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                                    s_frame, tau, model, config, telem) ||
                            is_graph_pruned_by_sq16(visited->query_sq16, s_frame, tau,
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
 * @best_seed_id:     In/out pointer to best seed frame ID.
 * @best_seed_dist:   In/out pointer to best seed distance.
 * @last_parent_id:   Output pointer to previous parent frame ID.
 * @last_parent_dist: Output pointer to previous parent distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Traverses the proximity graph greedily from initial seeds toward the nearest neighbor basin:
 * 1. Neighborhood scan: Inspects all adjacent neighbors of the current best_seed_id in the graph.
 * 2. Distance computation: Evaluates exact query distances for all unvisited adjacent frames.
 * 3. Gradient descent: If an adjacent neighbor is closer to query_data than best_seed_dist,
 *    moves the search to that neighbor and repeats.
 * 4. Local minimum termination: Stops when no adjacent neighbor yields a lower distance than the
 *    current node, having converged to a local distance basin.
 */
void knn_greedy_route_to_basin(
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
            if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                        nb_id, routing_tau, model, config, telem) ||
                is_graph_pruned_by_sq16(visited->query_sq16, nb_id, routing_tau,
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
 * @best_seed_id:     In/out pointer to best seed frame ID (u*).
 * @best_seed_dist:   In/out pointer to best seed distance.
 * @parent_id:        Parent frame ID from greedy routing.
 * @parent_dist:      Parent distance from query.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Dense graph expansion around the converged 1-NN basin:
 * 1. Neighborhood gathering: Gathers 1-hop and 2-hop graph neighbors of best_seed_id and parent_id.
 * 2. Unvisited evaluation: Computes exact query distances for all unvisited candidate nodes.
 * 3. Heap updating: Inserts viable candidates into the query heap to populate nearest neighbors.
 * 4. Seed registration: Adds evaluated candidates into seed_pivot_ids[] to serve as metric pivots
 *    during subsequent cluster filtering.
 */
void knn_direct_basin_expansion(
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

        if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                    nb_id, current_tau, model, config, telem) ||
            is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
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

                if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                            nb2, current_tau, model, config, telem) ||
                    is_graph_pruned_by_sq16(visited->query_sq16, nb2, current_tau,
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
 * Explores graph frontier nodes to expand the neighborhood for cross-dataset queries:
 * 1. Frontier traversal: Iterates over frontier nodes ordered by distance to query.
 * 2. Neighbor exploration: For each node, examines adjacent neighbors in model->graph.
 * 3. Distance evaluation: Evaluates unvisited candidates using early cutoff distance computation
 *    and inserts surviving frames into the heap.
 * 4. Containment stopping: Maintains seed pivots and stops when containment criterion is met
 *    or maximum expansions are reached.
 *
 * Return: 1 if global containment criterion was satisfied, 0 otherwise.
 */
int knn_cross_explore_graph_frontier(
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
            if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                        nb_id, sq_tau, model, config, telem) ||
                is_graph_pruned_by_sq16(visited->query_sq16, nb_id, sq_tau,
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

                if (is_graph_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                            nb_id, current_tau, model, config, telem) ||
                    is_graph_pruned_by_sq16(visited->query_sq16, nb_id, current_tau,
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


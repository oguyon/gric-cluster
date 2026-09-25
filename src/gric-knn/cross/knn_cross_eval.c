/**
 * @file knn_cross_eval.c
 * @brief Cross-dataset intra-cluster and inter-cluster candidate evaluation.
 *
 * Implements member-level evaluation and metric lower-bound filtering for cross-dataset
 * searching. Functions in this file evaluate frames within the nearest cluster basin
 * (knn_cross_eval_intra_cluster) and scan adjacent clusters in distance priority order
 * (knn_cross_eval_inter_clusters), using pivot distance bounds to prune unnecessary checks.
 */

#include "knn_cross_eval.h"
#include <alloca.h>
#include <string.h>

/**
 * knn_cross_eval_intra_cluster() - Evaluate frames inside best cluster and prune by radius.
 * @best_c:           Cluster ID containing best seed.
 * @best_seed_id:     Frame ID of best seed found during routing.
 * @min_d_anchor:     Distance from query frame to target cluster anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Array of evaluated pivot frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @query_data:       Raw pixel buffer for query frame.
 * @model:            Target KnnModel containing candidate dataset.
 * @config:           Active KnnConfig specifying search options, epsilon slack, and cutoffs.
 * @cand_reader:      Candidate frame reader context for loading frame vectors.
 * @cand_buffer:      Thread-local scratch buffer for reading candidate frame data.
 * @anchor_dists:     Thread-local query-to-anchor distance cache.
 * @heap:             Per-query max-heap tracking the top-k nearest neighbors found so far.
 * @visited:          Per-query visited tracker preventing duplicate frame evaluations.
 * @telem:            Thread-local telemetry record accumulating search counters and prunes.
 *
 * Evaluates candidate frames within the winning cluster partition for a cross-dataset query:
 * 1. Target cluster resolution: Maps best_seed_id to its host cluster partition, defaulting to
 *    best_c if no cluster map exists.
 * 2. Anchor distance resolution: Retrieves or computes query-to-anchor Euclidean distance.
 * 3. Annular member scan: Traverses cluster members, checking annular triangle lower bounds
 *    |d_anchor - r_cand| against current search horizon tau / (1.0 + epsilon).
 * 4. Multi-pivot seed filtering: Tests candidate against seed pivots via
 *    is_member_pruned_by_pointwise_pivots().
 * 5. Exact distance & heap insertion: Evaluates exact Euclidean distance for survivors and
 *    inserts viable neighbors into the heap.
 */
void knn_cross_eval_intra_cluster(
    int                    best_c,
    long                   best_seed_id,
    double                 min_d_anchor,
    int                    num_seed_pivots,
    const long            *seed_pivot_ids,
    const double          *seed_pivot_dists,
    const void *restrict   query_data,
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

            if (is_member_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                         cand_id, current_tau, model, config, telem) ||
                is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
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
 * @loc_res:          Cluster locator results containing evaluated anchors.
 * @best_c:           Best anchor cluster index.
 * @min_d_anchor:     Distance to closest cluster anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Array of evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Array of distances from query to seed pivots.
 * @query_data:       Raw pixel buffer for query frame.
 * @model:            Target KnnModel containing candidate dataset.
 * @config:           Active KnnConfig specifying search options, epsilon slack, and cutoffs.
 * @cand_reader:      Candidate frame reader context for loading frame vectors.
 * @cand_buffer:      Thread-local scratch buffer for reading candidate frame data.
 * @anchor_dists:     Thread-local query-to-anchor distance cache.
 * @scores_buffer:    Thread-local scratch buffer for candidate cluster sorting.
 * @active_mask:      Bitmask of surviving candidate clusters.
 * @heap:             Per-query max-heap tracking the top-k nearest neighbors found so far.
 * @visited:          Per-query visited tracker preventing duplicate frame evaluations.
 * @telem:            Thread-local telemetry record accumulating search counters and prunes.
 *
 * Evaluates non-home candidate clusters for a cross-dataset query in prioritized order:
 * 1. Lower-bound scoring: Computes distance lower bounds for all clusters q != best_c using
 *    DCC matrix bounds and measured seed pivots.
 * 2. Pruning: Discards clusters whose lower bound exceeds tau / (1.0 + epsilon).
 * 3. Sorting: Sorts surviving clusters ascending by lower bound in scores_buffer.
 * 4. Priority traversal: Searches candidate clusters in order, terminating early once the
 *    next cluster's lower bound exceeds the dynamic search horizon.
 */
void knn_cross_eval_inter_clusters(
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

    // Inter-Cluster Search on surviving candidate clusters
    int max_cands = (config->approx_mode) ? 4 : num_cand_clusters;

    if (config->approx_mode && num_cand_clusters > max_cands)
    {
        for (int i = 0; i < max_cands; i++)
        {
            int min_idx = i;
            for (int j = i + 1; j < num_cand_clusters; j++)
            {
                if (compare_cluster_scores(&scores_buffer[j], &scores_buffer[min_idx]) < 0)
                {
                    min_idx = j;
                }
            }
            if (min_idx != i)
            {
                ClusterScore tmp = scores_buffer[i];
                scores_buffer[i] = scores_buffer[min_idx];
                scores_buffer[min_idx] = tmp;
            }
        }
    }
    else
    {
        qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
              compare_cluster_scores);
    }

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

            if (is_member_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                         cand_id, current_tau, model, config, telem) ||
                is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
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


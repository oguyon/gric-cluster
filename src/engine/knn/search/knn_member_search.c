/**
 * @file knn_member_search.c
 * @brief Member candidate evaluation, annular traversal, and distance batching.
 *
 * Implements frame-level distance evaluation and candidate pruning within clusters.
 * Functions in this file perform annular member filtering sorted by distance to cluster
 * anchor, batch distance calculations to maximize SIMD throughput and cache locality,
 * and dispatch member scanning across full-precision, float, and quantized formats.
 */

#include "knn_member_search.h"
#include "knn_member_quant.h"
#include "residual_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * knn_batch_init() - Zero-initialize a candidate evaluation batch.
 * @batch: Pointer to KnnCandidateBatch struct to initialize.
 *
 * Resets the candidate counter to zero. The batch structure buffers up to 8 candidate
 * vectors to allow SIMD 1x8 batched distance kernels (AVX2 / AVX-512) to compute exact
 * Euclidean distances concurrently against the query vector.
 */
void knn_batch_init(
    KnnCandidateBatch *batch)
{
    if (batch != NULL)
    {
        batch->count = 0;
    }
}

/**
 * knn_batch_flush() - Compute distances for accumulated candidates and update heaps.
 * @batch:        Pointer to active KnnCandidateBatch containing queued candidate vectors.
 * @query_id:     Global frame index of current query.
 * @query_data:   Raw pixel vector for query frame.
 * @model:        Active KnnModel containing frame dimensions and precision configuration.
 * @config:       Active KnnConfig specifying search options, epsilon slack, and cutoffs.
 * @heap:         Per-query max-heap tracking the top-k nearest neighbors found so far.
 * @all_heaps:    Global array of per-query max-heaps (used for mutual neighbor insertion).
 * @bucket_locks: Bucket locks array synchronizing concurrent heap updates (or NULL).
 * @telem:        Thread-local telemetry record accumulating search counters and prunes.
 *
 * Flushes all accumulated candidate frames in the batch:
 * 1. Vectorized 1x8 batch execution: If exactly 8 candidates are buffered, dispatches
 *    to compute_framedist_1x8() to compute Euclidean distances concurrently across all 8
 *    vectors against the query in vector registers.
 * 2. Scalar fallback / tail handling: Computes distances for remaining candidates using
 *    compute_euclidean_distance_cutoff() with early termination on squared distance bounds.
 * 3. Heap update & reciprocal insertion: For any candidate whose distance d <= tau_thresh,
 *    invokes record_neighbor_and_reciprocal() to insert into the query's heap and optionally
 *    update the candidate's heap under OpenMP bucket lock synchronization.
 * 4. Resets the batch count to 0 for subsequent accumulation.
 */
void knn_batch_flush(
    KnnCandidateBatch      *batch,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnTelemetry  *restrict telem)
{
    if (batch == NULL || batch->count <= 0)
    {
        return;
    }

    int count = batch->count;
    long frame_elem = model->frame_elements;

    double eps_factor = 1.0 + (double)config->epsilon;
    double tau_thresh = (double)heap->tau / eps_factor;
    if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
    {
        tau_thresh = config->rlim_cutoff;
    }

    double cutoff_sq = 0.0;
    if (heap->count >= heap->k || config->rlim_cutoff > 0.0)
    {
        cutoff_sq = tau_thresh * tau_thresh;
    }

    int b = 0;
    for (; b <= count - 8; b += 8)
    {
        double chunk_dists[8];
        int pruned_mask = 0;

        if (model->is_double)
        {
            pruned_mask = framedist_batch_cutoff_1x8_double(
                (const double *)query_data,
                (const double *const *)(batch->ptrs + b),
                frame_elem,
                cutoff_sq,
                chunk_dists);
        }
        else
        {
            pruned_mask = framedist_batch_cutoff_1x8_float(
                (const float *)query_data,
                (const float *const *)(batch->ptrs + b),
                frame_elem,
                cutoff_sq,
                chunk_dists);
        }
        telem->framedist_calls += 8;

        for (int k = 0; k < 8; k++)
        {
            if ((pruned_mask & (1 << k)) == 0)
            {
                record_neighbor_and_reciprocal(
                    query_id,
                    batch->cand_ids[b + k],
                    chunk_dists[k],
                    config,
                    model,
                    heap,
                    all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        } // for (int k = 0; k < 8; k++)
    } // for (; b <= count - 8; b += 8)

    for (; b <= count - 4; b += 4)
    {
        double chunk_dists[4];
        int pruned_mask = 0;

        if (model->is_double)
        {
            pruned_mask = framedist_batch_cutoff_1x4_double(
                (const double *)query_data,
                (const double *const *)(batch->ptrs + b),
                frame_elem,
                cutoff_sq,
                chunk_dists);
        }
        else
        {
            pruned_mask = framedist_batch_cutoff_1x4_float(
                (const float *)query_data,
                (const float *const *)(batch->ptrs + b),
                frame_elem,
                cutoff_sq,
                chunk_dists);
        }
        telem->framedist_calls += 4;

        for (int k = 0; k < 4; k++)
        {
            if ((pruned_mask & (1 << k)) == 0)
            {
                record_neighbor_and_reciprocal(
                    query_id,
                    batch->cand_ids[b + k],
                    chunk_dists[k],
                    config,
                    model,
                    heap,
                    all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        } // for (int k = 0; k < 4; k++)
    } // for (; b <= count - 4; b += 4)

    for (; b < count; b++)
    {
        telem->framedist_calls++;
        double d_exact = compute_euclidean_distance_cutoff(
            query_data,
            batch->ptrs[b],
            frame_elem,
            model->is_double,
            cutoff_sq);

        if (cutoff_sq <= 0.0 || d_exact <= tau_thresh)
        {
            record_neighbor_and_reciprocal(
                query_id,
                batch->cand_ids[b],
                d_exact,
                config,
                model,
                heap,
                all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    } // for (; b < count; b++)

    batch->count = 0;
}

/**
 * knn_eval_members_annular() - Center-outward annular scalar search over cluster members.
 * @cl:                Candidate cluster being searched.
 * @d_anchor:          Distance from query vector to candidate cluster anchor.
 * @r_home:            Distance from query vector to its home cluster anchor.
 * @dcc_home:          Inter-anchor distance between query home cluster and candidate cluster.
 * @sq16_delta:        Quantization error bound margin for conservative metric bounds.
 * @eq16_active:       1 if member-level EQ16 quantization filtering is active.
 * @sq16_active:       1 if member-level SQ16 quantization filtering is active.
 * @rq8_active:        1 if member-level RQ8 quantization filtering is active.
 * @num_active_pivots: Number of active measured anchor pivots for multi-pivot filtering.
 * @pivot_diffs:       Array of precomputed query-to-pivot distance differentials.
 * @query_id:          Global frame index of current query.
 * @query_data:        Raw pixel buffer of query frame (float or double).
 * @model:             Active KnnModel containing cluster metadata and quantization structures.
 * @config:            Active KnnConfig specifying search options, epsilon slack, and cutoffs.
 * @reader:            Streaming frame reader context for fetching candidate vectors if unmapped.
 * @cand_buffer:       Thread-local scratch buffer for reading candidate frame data.
 * @heap:              Per-query max-heap tracking the top-k nearest neighbors found so far.
 * @all_heaps:         Global array of per-query max-heaps (used for mutual neighbor insertion).
 * @bucket_locks:      OpenMP bucket locks array synchronizing concurrent heap updates (or NULL).
 * @visited:           Per-query visited tracker preventing duplicate frame evaluations.
 * @batch:             Pointer to shared KnnCandidateBatch for vectorized 1x8 exact distances.
 * @telem:             Thread-local telemetry record accumulating search counters and prunes.
 *
 * Traverses cluster members radially outward from the query's projection d_anchor:
 * 1. Binary search initialization: Locates initial member index where r_anchor matches d_anchor
 *    via find_member_lower_bound(), establishing center-outward left and right pointers.
 * 2. Annular bounds pruning: At each step, checks lower bound |d_anchor - r_member| against
 *    tau_thresh = tau / (1.0 + epsilon). Truncates leftward or rightward search when the lower
 *    bound guarantees no member in that direction can improve on the current heap cutoff.
 * 3. Pointwise quantization screening: Applies scalar SQ16, EQ16, or RQ8 lower bounding on
 *    individual candidate coordinates before fetching raw pixel data.
 * 4. Pivot and temporal validation: Evaluates metric pivots and temporal separation bounds
 *    using knn_is_candidate_pruned().
 * 5. Vector resolution: Loads candidate data and dispatches to batch or exact evaluation.
 */
static void knn_eval_members_annular(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     eq16_active,
    int                     sq16_active,
    int                     rq8_active,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    int mid = find_member_lower_bound(cl->members, num_m, (float)d_anchor);
    int left = mid - 1;
    int right = mid;

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    float cached_cutoff_adc = 1e30f;

    while (left >= 0 || right < num_m)
    {
        double d_left = 1e30;
        if (left >= 0)
        {
            float r_l = cl->members[left].r_anchor;
            d_left = (d_anchor > (double)r_l) ? (d_anchor - (double)r_l) : 0.0;
        }

        double d_right = 1e30;
        if (right < num_m)
        {
            float r_r = cl->members[right].r_anchor;
            d_right = ((double)r_r > d_anchor) ? ((double)r_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
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

        int m = (d_left <= d_right) ? left-- : right++;
        long cand_id = (long)cl->members[m].frame_id;
        double r_cand = (double)cl->members[m].r_anchor;

        if (knn_is_candidate_pruned(
                query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                config, heap, visited, telem))
        {
            continue;
        }

        if (rq8_active)
        {
            if (cl->rq8_vectors != NULL)
            {
                if (left >= 0)
                {
                    KNN_PREFETCH_T0(cl->rq8_vectors + (size_t)left * (size_t)frame_elem);
                }
                if (right < num_m)
                {
                    KNN_PREFETCH_T0(cl->rq8_vectors + (size_t)right * (size_t)frame_elem);
                }
            }
            else
            {
                if (left >= 0)
                {
                    long pref_l = (long)cl->members[left].frame_id;
                    const int8_t *p_l = model->rq8_dataset_buffer +
                        (size_t)pref_l * (size_t)frame_elem;
                    KNN_PREFETCH_T0(p_l);
                }
                if (right < num_m)
                {
                    long pref_r = (long)cl->members[right].frame_id;
                    const int8_t *p_r = model->rq8_dataset_buffer +
                        (size_t)pref_r * (size_t)frame_elem;
                    KNN_PREFETCH_T0(p_r);
                }
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
                {
                    cached_cutoff_adc = compute_rq8_cutoff_thresh_adc_cluster(
                        current_tau, &cl->rq8_params, config
                    );
                }
                else
                {
                    cached_ssd_cutoff = compute_rq8_cutoff_thresh_cluster(
                        current_tau, &cl->rq8_params, config
                    );
                }
            }

            if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
            {
                if (is_member_pruned_by_rq8_adc_cached(
                        visited->query_rq8_adc, cand_id, cached_cutoff_adc, model, telem))
                {
                    continue;
                }
            }
            else
            {
                const int8_t *cand_rq8 = (cl->rq8_vectors != NULL)
                    ? (cl->rq8_vectors + (size_t)m * (size_t)frame_elem)
                    : (model->rq8_dataset_buffer + (size_t)cand_id * (size_t)frame_elem);
                telem->rq8_evaluations++;
                uint64_t ssd = rq8_dist_squared_cutoff_i8(
                    visited->query_rq8, cand_rq8, frame_elem, cached_ssd_cutoff
                );
                if (ssd > cached_ssd_cutoff)
                {
                    telem->rq8_members_pruned++;
                    continue;
                }
            }
        }
        else if (!config->use_rq8 && eq16_active)
        {
            if (cl->eq16_vectors != NULL)
            {
                if (left >= 0)
                {
                    KNN_PREFETCH_T0(cl->eq16_vectors + (size_t)left * (size_t)frame_elem);
                }
                if (right < num_m)
                {
                    KNN_PREFETCH_T0(cl->eq16_vectors + (size_t)right * (size_t)frame_elem);
                }
            }
            else
            {
                if (left >= 0)
                {
                    long pref_l = (long)cl->members[left].frame_id;
                    const int16_t *p_l = model->eq16_dataset_buffer +
                        (size_t)pref_l * (size_t)frame_elem;
                    KNN_PREFETCH_T0(p_l);
                }
                if (right < num_m)
                {
                    long pref_r = (long)cl->members[right].frame_id;
                    const int16_t *p_r = model->eq16_dataset_buffer +
                        (size_t)pref_r * (size_t)frame_elem;
                    KNN_PREFETCH_T0(p_r);
                }
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_eq16_cutoff_thresh(current_tau, model, config);
            }

            if (is_member_pruned_by_eq16_cached(
                    visited->query_eq16, visited->query_eq16_adc,
                    cand_id, cached_ssd_cutoff, model, telem))
            {
                continue;
            }
        }
        else if (!config->use_rq8 && sq16_active)
        {
            if (left >= 0)
            {
                long pref_l = (long)cl->members[left].frame_id;
                KNN_PREFETCH_T0(model->sq16_dataset_buffer + (size_t)pref_l * (size_t)frame_elem);
            }
            if (right < num_m)
            {
                long pref_r = (long)cl->members[right].frame_id;
                KNN_PREFETCH_T0(model->sq16_dataset_buffer + (size_t)pref_r * (size_t)frame_elem);
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
        else if (!config->use_rq8 &&
                 (is_member_pruned_by_eq16(visited->query_eq16, visited->query_eq16_adc,
                                           cand_id, current_tau, model, config, telem) ||
                  is_member_pruned_by_sq16(visited->query_sq16, cand_id, current_tau,
                                           model, config, telem)))
        {
            continue;
        }

        if (!config->use_rq8 &&
            is_member_pruned_by_sq8(visited->query_sq8, cand_id, current_tau,
                                    model, config, telem))
        {
            continue;
        }

        void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
        const void *cand_ptr = knn_resolve_candidate_data(
            cand_id, m, cl, model, reader, slot, frame_bytes
        );
        if (cand_ptr == NULL)
        {
            continue;
        }

        knn_append_or_eval_candidate(
            cand_id, cand_ptr, query_id, query_data, model, config,
            heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            batch, telem);
    } // while (left >= 0 || right < num_m)
}

/**
 * knn_eval_cluster_members() - Search all members of a cluster with metric and SIMD pruning.
 * @c:                 Index of target cluster to evaluate.
 * @d_anchor:          Computed distance from query frame to target cluster anchor.
 * @home_cluster_id:   Query's home cluster index (or -1 if unassigned).
 * @r_home:            Distance to query's home cluster anchor.
 * @anchor_is_sq16:    1 if d_anchor was computed via SQ16, 0 otherwise.
 * @query_id:          Index of query frame.
 * @query_data:        Raw pixel buffer for query frame.
 * @model:             Active KnnModel containing cluster metadata and quantization tables.
 * @config:            Active KnnConfig specifying search options, epsilon slack, and cutoffs.
 * @reader:            KnnFrameReader context for reading on-disk frames.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Global array of all frame heaps (for mutual updates).
 * @bucket_locks:      OpenMP bucket locks array for thread synchronization (or NULL).
 * @pivots:            Array of measured anchor pivots for multi-pivot filtering (or NULL).
 * @num_pivots:        Number of active anchor pivots.
 * @visited:           Per-query frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch for 1x8 distance batching.
 * @telem:             Thread-local telemetry record accumulating search counters and prunes.
 *
 * Coordinates member candidate evaluation for cluster c against the query vector:
 * 1. Precomputes triangle inequality slack factors from quantization error radii (quant_err).
 * 2. Computes or looks up inter-anchor distance (dcc_home) to the query's home cluster.
 * 3. Identifies active metric pivots and precomputes query-to-pivot distance differences.
 * 4. Quantization fast-path dispatch: If cluster c has block-quantized representations,
 *    delegates to the specialized SIMD FastScan kernel:
 *    - knn_eval_members_rabitq() for RaBitQ binary quantization.
 *    - knn_eval_members_pq() for Product Quantization.
 *    - knn_eval_members_rq8_blocks() for 8-bit Residual Quantization.
 *    - knn_eval_members_eq16_blocks() for E8 lattice Gosset quantization.
 *    - knn_eval_members_sq16_blocks() for 16-bit Scalar Quantization.
 * 5. Scalar fallback: If no block quantization is available, executes two-pointer annular
 *    scalar traversal via knn_eval_members_annular().
 * 6. Flushes any remaining batched candidate distances via knn_batch_flush().
 */
void knn_eval_cluster_members(
    int                     c,
    double                  d_anchor,
    int                     home_cluster_id,
    double                  r_home,
    int                     anchor_is_sq16,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    const MeasuredPivot    *pivots,
    int                     num_pivots,
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    const KnnCluster *cl = &model->clusters[c];
    int num_m = cl->num_members;
    if (num_m <= 0)
    {
        return;
    }

    telem->total_candidates_considered += (uint64_t)num_m;

    int M = model->num_clusters;
    double quant_err = (config->use_eq16 && model->anchor_eq16_buffer != NULL)
                       ? (double)model->eq16_params.err_radius
                       : (double)model->sq16_params.err_radius;
    double sq16_delta = 0.0;
    if (anchor_is_sq16)
    {
        sq16_delta = (config->use_eq16 && config->use_eq16_adc) ? quant_err : 2.0 * quant_err;
    }
    double dcc_home = (home_cluster_id >= 0 && home_cluster_id < M && home_cluster_id != c) ?
        model->dcc_matrix[(size_t)home_cluster_id * (size_t)M + (size_t)c] : 0.0;

    int eq16_active = (config->use_eq16 && model->eq16_dataset_buffer != NULL &&
                       ((config->use_eq16_adc && visited->query_eq16_adc != NULL) ||
                        visited->query_eq16 != NULL));
    int sq16_active = (config->use_sq16 && model->sq16_dataset_buffer != NULL &&
                       visited->query_sq16 != NULL);
    int rq8_active = (config->use_rq8 && model->rq8_dataset_buffer != NULL &&
                      ((config->use_rq8_adc && visited->query_rq8_adc != NULL) ||
                       (visited->query_rq8 != NULL && !visited->query_rq8_clipped)));

    if (rq8_active)
    {
        if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
        {
            if (model->is_double)
            {
                rq8_prepare_query_residual_double(
                    (const double *)query_data,
                    (const double *)cl->anchor_data,
                    visited->query_rq8_adc,
                    &cl->rq8_params);
            }
            else
            {
                rq8_prepare_query_residual_float(
                    (const float *)query_data,
                    (const float *)cl->anchor_data,
                    visited->query_rq8_adc,
                    &cl->rq8_params);
            }
            visited->query_rq8_clipped = 0;
        }
        else if (visited->query_rq8 != NULL)
        {
            if (model->is_double)
            {
                visited->query_rq8_clipped = rq8_quantize_query_residual_double(
                    (const double *)query_data,
                    (const double *)cl->anchor_data,
                    visited->query_rq8,
                    &cl->rq8_params);
            }
            else
            {
                visited->query_rq8_clipped = rq8_quantize_query_residual_float(
                    (const float *)query_data,
                    (const float *)cl->anchor_data,
                    visited->query_rq8,
                    &cl->rq8_params);
            }
            rq8_active = !visited->query_rq8_clipped;
        }
    }

    int num_active_pivots = 0;
    double pivot_diffs[MAX_MEASURED_PIVOTS];
    if (config->use_multi_pivot && pivots != NULL && num_pivots > 0)
    {
        for (int p = 0; p < num_pivots; p++)
        {
            int p_cl = pivots[p].cluster_id;
            if (p_cl == c || p_cl == home_cluster_id)
            {
                continue;
            }
            double dcc_pc = model->dcc_matrix[(size_t)p_cl * (size_t)M + (size_t)c];
            if (dcc_pc > 0.0)
            {
                pivot_diffs[num_active_pivots++] = fabs(dcc_pc - pivots[p].d_anchor);
            }
        }
    }

    int rabitq_active = (config->use_rabitq &&
                         cl->rabitq_transposed != NULL &&
                         cl->num_rabitq_blocks > 0 &&
                         visited->query_rabitq_lut.lut_i8 != NULL);

    if (rabitq_active)
    {
        knn_eval_members_rabitq(
            cl, d_anchor, r_home, dcc_home, sq16_delta,
            num_active_pivots, pivot_diffs, query_id, query_data, model,
            config, reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, batch, telem);
        return;
    }

    int pq_active = (config->use_pq && model->pq_codebook != NULL &&
                     cl->pq_transposed != NULL && cl->num_pq_blocks > 0 &&
                     visited->query_pq_lut != NULL);

    if (pq_active)
    {
        knn_eval_members_pq(
            cl, d_anchor, r_home, dcc_home, sq16_delta,
            num_active_pivots, pivot_diffs, query_id, query_data, model,
            config, reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, batch, telem);
        return;
    }

    if (rq8_active && cl->num_rq8_blocks > 0 &&
        (cl->rq8_transposed != NULL || config->use_rq8_sparse))
    {
        knn_eval_members_rq8_blocks(
            cl, d_anchor, r_home, dcc_home, sq16_delta,
            num_active_pivots, pivot_diffs, query_id, query_data, model,
            config, reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, batch, telem);
        return;
    }

    if (eq16_active && cl->num_eq16_blocks > 0 &&
        (cl->eq16_transposed != NULL || config->use_eq16_sparse))
    {
        knn_eval_members_eq16_blocks(
            cl, d_anchor, r_home, dcc_home, sq16_delta,
            num_active_pivots, pivot_diffs, query_id, query_data, model,
            config, reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, batch, telem);
        return;
    }

    if (sq16_active && cl->num_sq16_blocks > 0 &&
        (cl->sq16_transposed != NULL || config->use_sq16_sparse))
    {
        knn_eval_members_sq16_blocks(
            cl, d_anchor, r_home, dcc_home, sq16_delta,
            num_active_pivots, pivot_diffs, query_id, query_data, model,
            config, reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, batch, telem);
        return;
    }

    knn_eval_members_annular(
        cl, d_anchor, r_home, dcc_home, sq16_delta,
        eq16_active, sq16_active, rq8_active, num_active_pivots, pivot_diffs,
        query_id, query_data, model, config, reader, cand_buffer,
        heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        visited, batch, telem);
}

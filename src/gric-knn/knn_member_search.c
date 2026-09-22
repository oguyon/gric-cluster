/**
 * @file knn_member_search.c
 * @brief Member candidate evaluation, SIMD fastscan, and batched distance calculation.
 */

#include "knn_member_search.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define KNN_PREFETCH_T0(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define KNN_PREFETCH_T0(addr) __builtin_prefetch((const void *)(addr), 0, 3)
#else
#define KNN_PREFETCH_T0(addr) ((void)0)
#endif

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
 * knn_batch_init() - Zero-initialize a candidate evaluation batch.
 * @batch: Pointer to KnnCandidateBatch struct.
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
 * @batch:        Pointer to active KnnCandidateBatch.
 * @query_id:     Frame index of current query.
 * @query_data:   Raw pixel vector for query frame.
 * @model:        Active KnnModel.
 * @config:       Active KnnConfig.
 * @heap:         Per-query max-heap.
 * @all_heaps:    Global array of all frame heaps (for mutual updates).
 * @bucket_locks: Bucket locks array for thread synchronization (or NULL).
 * @telem:        Thread-local telemetry record.
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
    double dists[8];

    if (count == 8)
    {
        if (model->is_double)
        {
            framedist_batch_1x8_double(
                (const double *)query_data,
                (const double *const *)batch->ptrs,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_1x8_float(
                (const float *)query_data,
                (const float *const *)batch->ptrs,
                dists,
                frame_elem);
        }
        telem->framedist_calls += 8;
    }
    else
    {
        if (model->is_double)
        {
            framedist_batch_double(
                (const double *)query_data,
                (const double *const *)batch->ptrs,
                count,
                dists,
                frame_elem);
        }
        else
        {
            framedist_batch_float(
                (const float *)query_data,
                (const float *const *)batch->ptrs,
                count,
                dists,
                frame_elem);
        }
        telem->framedist_calls += (uint64_t)count;
    }

    for (int b = 0; b < count; b++)
    {
        record_neighbor_and_reciprocal(
            query_id,
            batch->cand_ids[b],
            dists[b],
            config,
            model,
            heap,
            all_heaps
#ifdef _OPENMP
            , bucket_locks
#endif
        );
    } // for (int b = 0; b < count; b++)

    batch->count = 0;
}

/**
 * knn_resolve_candidate_data() - Fetch candidate pixel pointer with optional disk read.
 * @cand_id:     Index of candidate frame.
 * @m_idx:       Index of member within cluster.
 * @cl:          Pointer to candidate cluster.
 * @model:       Active KnnModel.
 * @reader:      Streaming frame reader.
 * @slot_buffer: Target buffer slot if reading from disk.
 * @frame_bytes: Size of frame in bytes.
 *
 * Return: Pointer to frame pixel data, or NULL on read failure.
 */
static inline const void *knn_resolve_candidate_data(
    long              cand_id,
    int               m_idx,
    const KnnCluster *cl,
    const KnnModel   *model,
    KnnFrameReader   *reader,
    void             *slot_buffer,
    size_t            frame_bytes)
{
    if (cl->ivf_vectors != NULL)
    {
        const void *ptr = (const char *)cl->ivf_vectors + (size_t)m_idx * frame_bytes;
        KNN_PREFETCH_T0(ptr);
        return ptr;
    }

    if (model->dataset_buffer != NULL)
    {
        const void *ptr = (const char *)model->dataset_buffer + (size_t)cand_id * frame_bytes;
        KNN_PREFETCH_T0(ptr);
        return ptr;
    }

    if (knn_reader_read_frame(reader, cand_id, slot_buffer) != 0)
    {
        return NULL;
    }

    return slot_buffer;
}

/**
 * knn_is_candidate_pruned() - Check visited, temporal, annular, and pivot bounds.
 *
 * Return: 1 if candidate is pruned, 0 if viable.
 */
static inline int knn_is_candidate_pruned(
    long                 query_id,
    long                 cand_id,
    double               r_cand,
    double               d_anchor,
    double               r_home,
    double               dcc_home,
    double               tau_thresh,
    double               sq16_delta,
    int                  num_active_pivots,
    const double        *pivot_diffs,
    const KnnConfig     *config,
    KnnMaxHeap          *heap,
    KnnVisitedTracker   *visited,
    KnnTelemetry        *telem)
{
    if (knn_visited_check_and_mark(visited, cand_id))
    {
        return 1;
    }

    if (!check_temporal_separation(query_id, cand_id, config))
    {
        telem->temporal_pruned++;
        return 1;
    }

    double lb1 = fabs(d_anchor - r_cand) - sq16_delta;
    if (lb1 < 0.0)
    {
        lb1 = 0.0;
    }
    if (lb1 >= tau_thresh)
    {
        telem->level3_annular_pruned++;
        return 1;
    }

    if (dcc_home > 0.0)
    {
        double diff_home = fabs(dcc_home - r_cand);
        if (diff_home - r_home - sq16_delta >= tau_thresh)
        {
            telem->level3_annular_pruned++;
            return 1;
        }
    }

    if (num_active_pivots > 0)
    {
        double target_thresh = tau_thresh + sq16_delta;
        for (int p = 0; p < num_active_pivots; p++)
        {
            double diff = pivot_diffs[p] - r_cand;
            if (diff >= target_thresh)
            {
                telem->level3_annular_pruned++;
                telem->multi_pivot_pruned++;
                return 1;
            }
        } // for (int p = 0; p < num_active_pivots; p++)
    }

    if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
    {
        telem->reciprocal_reused++;
        return 1;
    }

    return 0;
}

/**
 * knn_append_or_eval_candidate() - Add candidate to batch or compute distance immediately.
 */
static inline void knn_append_or_eval_candidate(
    long                    cand_id,
    const void             *cand_ptr,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    if (config->use_batch_dist)
    {
        batch->cand_ids[batch->count] = cand_id;
        batch->ptrs[batch->count] = cand_ptr;
        batch->count++;
        if (batch->count == 8)
        {
            knn_batch_flush(
                batch, query_id, query_data, model, config, heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                telem);
        }
    }
    else
    {
        telem->framedist_calls++;
        double d_exact = compute_euclidean_distance(
            query_data, cand_ptr, model->frame_elements, model->is_double
        );
        record_neighbor_and_reciprocal(
            query_id, cand_id, d_exact, config, model, heap, all_heaps
#ifdef _OPENMP
            , bucket_locks
#endif
        );
    }
}

/**
 * knn_init_annular_block_pointers() - Initialize two-pointer annular block search.
 * @members:     Array of member metadata sorted by r_anchor.
 * @num_m:       Total number of members.
 * @num_b:       Total number of candidate blocks.
 * @block_size:  Number of candidates per block (e.g. 32).
 * @d_anchor:    Distance from query to cluster anchor.
 * @out_left_b:  Output pointer to left block index.
 * @out_right_b: Output pointer to right block index.
 */
static inline void knn_init_annular_block_pointers(
    const MemberMeta *members,
    int               num_m,
    int               num_b,
    int               block_size,
    double            d_anchor,
    int              *out_left_b,
    int              *out_right_b)
{
    int m_star = find_member_lower_bound(members, num_m, (float)d_anchor);
    if (m_star >= num_m)
    {
        /* All members have r_anchor < d_anchor: traverse leftward from last block */
        *out_left_b = num_b - 1;
        *out_right_b = num_b;
    }
    else if (m_star == 0)
    {
        /* All members have r_anchor >= d_anchor: traverse rightward from first block */
        *out_left_b = -1;
        *out_right_b = 0;
    }
    else
    {
        /* d_anchor falls within member radius range: start at enclosing block */
        int mid_b = m_star / block_size;
        *out_left_b = mid_b - 1;
        *out_right_b = mid_b;
    }
}

/**
 * knn_eval_members_rabitq() - Evaluate cluster members using RaBitQ FastScan.
 */
static void knn_eval_members_rabitq(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
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
    int num_b = cl->num_rabitq_blocks;
    long frame_elem = model->frame_elements;
    int num_nibbles = (model->rabitq_params.bits == 2) ?
        (int)(model->rabitq_params.dim_pad / 2) :
        (int)(model->rabitq_params.dim_pad / 4);
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, RABITQ_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * RABITQ_FASTSCAN_BLOCK_SIZE + RABITQ_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * RABITQ_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && d_left_b >= tau_thresh)
        {
            int pruned = (left_b + 1) * RABITQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && d_right_b >= tau_thresh)
        {
            int pruned = num_m - right_b * RABITQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * RABITQ_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > RABITQ_FASTSCAN_BLOCK_SIZE)
        {
            m_count = RABITQ_FASTSCAN_BLOCK_SIZE;
        }

        const uint8_t *b_codes = cl->rabitq_transposed +
            (size_t)b * (size_t)num_nibbles * 16;
        const RaBitQMeta *b_meta = &cl->rabitq_meta[m_start];

        telem->rabitq_evaluations += (uint64_t)m_count;

        uint32_t pass_mask = rabitq_fastscan_32x(
            &visited->query_rabitq_lut,
            b_codes,
            b_meta,
            tau_thresh,
            config->epsilon
        );
        if (m_count < RABITQ_FASTSCAN_BLOCK_SIZE)
        {
            pass_mask &= ((1U << m_count) - 1);
        }

        if (!pass_mask)
        {
            telem->rabitq_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->rabitq_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
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
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_pq() - Evaluate cluster members using Product Quantization FastScan.
 */
static void knn_eval_members_pq(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
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
    int num_b = cl->num_pq_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, PQ_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * PQ_FASTSCAN_BLOCK_SIZE + PQ_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * PQ_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && d_left_b >= tau_thresh)
        {
            int pruned = (left_b + 1) * PQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && d_right_b >= tau_thresh)
        {
            int pruned = num_m - right_b * PQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * PQ_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > PQ_FASTSCAN_BLOCK_SIZE)
        {
            m_count = PQ_FASTSCAN_BLOCK_SIZE;
        }

        const uint8_t *b_codes = cl->pq_transposed +
            (size_t)b * (size_t)model->pq_codebook->m * PQ_FASTSCAN_BLOCK_SIZE;
        telem->pq_evaluations += (uint64_t)m_count;

        uint8_t cutoff_u8 = 255;
        if (current_tau > 0.0)
        {
            double eff_tau = current_tau / eps_factor;
            double tau_scaled = eff_tau * eff_tau * (double)visited->query_pq_table.scale;
            if (tau_scaled < 255.0)
            {
                cutoff_u8 = (uint8_t)tau_scaled;
            }
        }

        uint32_t pass_mask = pq_fastscan_32x(
            visited->query_pq_table.lut_u8,
            b_codes,
            model->pq_codebook->m,
            cutoff_u8
        );
        if (m_count < PQ_FASTSCAN_BLOCK_SIZE)
        {
            pass_mask &= ((1U << m_count) - 1);
        }

        if (!pass_mask)
        {
            telem->pq_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->pq_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
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
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_rq8_blocks() - Evaluate cluster members using RQ8 FastScan blocks.
 */
static void knn_eval_members_rq8_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
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
    int num_b = cl->num_rq8_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, RQ8_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * RQ8_FASTSCAN_BLOCK_SIZE + RQ8_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * RQ8_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * RQ8_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * RQ8_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * RQ8_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > RQ8_FASTSCAN_BLOCK_SIZE)
        {
            m_count = RQ8_FASTSCAN_BLOCK_SIZE;
        }

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            cached_ssd_cutoff = compute_rq8_cutoff_thresh_cluster(
                current_tau, &cl->rq8_params, config
            );
        }

        const int8_t *b_coords = cl->rq8_transposed +
            (size_t)b * (size_t)frame_elem * RQ8_FASTSCAN_BLOCK_SIZE;
        telem->rq8_evaluations += (uint64_t)m_count;

        uint32_t pass_mask = rq8_fastscan_32x(
            visited->query_rq8, b_coords, frame_elem, cached_ssd_cutoff
        );
        if (m_count < RQ8_FASTSCAN_BLOCK_SIZE)
        {
            pass_mask &= ((1U << m_count) - 1);
        }

        if (!pass_mask)
        {
            telem->rq8_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->rq8_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
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
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_eq16_blocks() - Evaluate cluster members using EQ16 FastScan blocks.
 */
static void knn_eval_members_eq16_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
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
    int num_b = cl->num_eq16_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, EQ16_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * EQ16_FASTSCAN_BLOCK_SIZE + EQ16_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * EQ16_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * EQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * EQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * EQ16_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > EQ16_FASTSCAN_BLOCK_SIZE)
        {
            m_count = EQ16_FASTSCAN_BLOCK_SIZE;
        }

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            cached_ssd_cutoff = compute_eq16_cutoff_thresh(current_tau, model, config);
        }

        uint32_t pass_mask = 0;
        if (cl->eq16_transposed != NULL)
        {
            const int16_t *b_coords = cl->eq16_transposed +
                (size_t)b * (size_t)frame_elem * EQ16_FASTSCAN_BLOCK_SIZE;
            telem->eq16_evaluations += (uint64_t)m_count;

            if (config->use_eq16_adc && visited->query_eq16_adc != NULL)
            {
                pass_mask = eq16_fastscan_32x_adc(
                    visited->query_eq16_adc, b_coords, frame_elem, (float)cached_ssd_cutoff
                );
            }
            else if (visited->query_eq16 != NULL)
            {
                pass_mask = eq16_fastscan_32x_i16(
                    visited->query_eq16, b_coords, frame_elem, cached_ssd_cutoff
                );
            }

            if (m_count < EQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_eq16_sparse && model->eq16_dataset_buffer != NULL)
        {
            telem->eq16_evaluations += (uint64_t)m_count;
            if (config->use_eq16_adc && visited->query_eq16_adc != NULL)
            {
                int i = 0;
                for (; i <= m_count - 4; i += 4)
                {
                    const int16_t *cands[4];
                    cands[0] = model->eq16_dataset_buffer +
                        (size_t)cl->members[m_start + i + 0].frame_id * (size_t)frame_elem;
                    cands[1] = model->eq16_dataset_buffer +
                        (size_t)cl->members[m_start + i + 1].frame_id * (size_t)frame_elem;
                    cands[2] = model->eq16_dataset_buffer +
                        (size_t)cl->members[m_start + i + 2].frame_id * (size_t)frame_elem;
                    cands[3] = model->eq16_dataset_buffer +
                        (size_t)cl->members[m_start + i + 3].frame_id * (size_t)frame_elem;

                    float dsq[4];
                    eq16_dist_asym_cutoff_batch_1x4(
                        visited->query_eq16_adc, cands, frame_elem,
                        (float)cached_ssd_cutoff, dsq
                    );
                    if (dsq[0] <= (float)cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << (i + 0));
                    }
                    if (dsq[1] <= (float)cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << (i + 1));
                    }
                    if (dsq[2] <= (float)cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << (i + 2));
                    }
                    if (dsq[3] <= (float)cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << (i + 3));
                    }
                } // for (; i <= m_count - 4; i += 4)

                for (; i < m_count; i++)
                {
                    long cand_id = (long)cl->members[m_start + i].frame_id;
                    const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                        (size_t)cand_id * (size_t)frame_elem;
                    float dist_sq = eq16_dist_asym_cutoff_f32(
                        visited->query_eq16_adc, cand_eq16, frame_elem,
                        (float)cached_ssd_cutoff
                    );
                    if (dist_sq <= (float)cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << i);
                    }
                }
            }
            else if (visited->query_eq16 != NULL)
            {
                for (int i = 0; i < m_count; i++)
                {
                    long cand_id = (long)cl->members[m_start + i].frame_id;
                    const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                        (size_t)cand_id * (size_t)frame_elem;
                    uint64_t ssd = eq16_dist_squared_cutoff_i16(
                        visited->query_eq16, cand_eq16, frame_elem, cached_ssd_cutoff
                    );
                    if (ssd <= cached_ssd_cutoff)
                    {
                        pass_mask |= (1U << i);
                    }
                }
            }
        }
        else
        {
            continue;
        }

        if (!pass_mask)
        {
            telem->eq16_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->eq16_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
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
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_sq16_blocks() - Evaluate cluster members using SQ16 FastScan blocks.
 */
static void knn_eval_members_sq16_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
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
    int num_b = cl->num_sq16_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, SQ16_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

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
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * SQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * SQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
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

        uint32_t pass_mask = 0;
        if (cl->sq16_transposed != NULL)
        {
            const int16_t *b_coords = cl->sq16_transposed +
                (size_t)b * (size_t)frame_elem * SQ16_FASTSCAN_BLOCK_SIZE;
            telem->sq16_evaluations += (uint64_t)m_count;
            pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_sq16_sparse && config->use_sq16_sparse_lru &&
                 visited->sq16_lru_cache != NULL)
        {
            /* 16-block LRU Transposed FastScan in L2 cache */
            KnnSparseLruCache *lru = visited->sq16_lru_cache;
            int cl_idx = (int)(cl - model->clusters);
            int hit_slot = -1;
            int lru_slot = 0;
            uint64_t min_seq = UINT64_MAX;

            for (int s = 0; s < KNN_SPARSE_LRU_BLOCKS; s++)
            {
                if (lru->blocks[s].cl_id == cl_idx && lru->blocks[s].block_id == b)
                {
                    hit_slot = s;
                    break;
                }
                if (lru->blocks[s].access_seq < min_seq)
                {
                    min_seq = lru->blocks[s].access_seq;
                    lru_slot = s;
                }
            }

            int16_t *b_coords = NULL;
            if (hit_slot >= 0)
            {
                lru->blocks[hit_slot].access_seq = ++lru->access_clock;
                b_coords = lru->blocks[hit_slot].data;
            }
            else
            {
                b_coords = lru->blocks[lru_slot].data;
                lru->blocks[lru_slot].cl_id = cl_idx;
                lru->blocks[lru_slot].block_id = b;
                lru->blocks[lru_slot].access_seq = ++lru->access_clock;

                const int16_t *cand_ptrs[SQ16_FASTSCAN_BLOCK_SIZE];
                for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
                {
                    if (i < m_count)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        cand_ptrs[i] = model->sq16_dataset_buffer + cand_id * frame_elem;
                    }
                    else
                    {
                        cand_ptrs[i] = NULL;
                    }
                }

                for (long d = 0; d < frame_elem; d++)
                {
                    int16_t *dst = b_coords + d * SQ16_FASTSCAN_BLOCK_SIZE;
                    for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
                    {
                        dst[i] = cand_ptrs[i] ? cand_ptrs[i][d] : 32767;
                    }
                }
            }

            telem->sq16_evaluations += (uint64_t)m_count;
            pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_sq16_sparse && model->sq16_dataset_buffer != NULL)
        {
            /* Direct Row-Major SIMD with early cutoff (0 MB resident index) */
            telem->sq16_evaluations += (uint64_t)m_count;
            for (int i = 0; i < m_count; i++)
            {
                long cand_id = (long)cl->members[m_start + i].frame_id;
                const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                                           (size_t)cand_id * (size_t)frame_elem;
                uint64_t ssd = sq16_dist_squared_cutoff_i16(
                    visited->query_sq16, cand_sq16, frame_elem, cached_ssd_cutoff
                );
                if (ssd <= cached_ssd_cutoff)
                {
                    pass_mask |= (1U << i);
                }
            }
        }
        else
        {
            continue;
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

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
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
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_annular() - Center-outward annular scalar search over cluster members.
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
            if (left >= 0)
            {
                long pref_l = (long)cl->members[left].frame_id;
                KNN_PREFETCH_T0(model->rq8_dataset_buffer + (size_t)pref_l * (size_t)frame_elem);
            }
            if (right < num_m)
            {
                long pref_r = (long)cl->members[right].frame_id;
                KNN_PREFETCH_T0(model->rq8_dataset_buffer + (size_t)pref_r * (size_t)frame_elem);
            }

            if (current_tau != last_tau)
            {
                last_tau = current_tau;
                cached_ssd_cutoff = compute_rq8_cutoff_thresh_cluster(
                    current_tau, &cl->rq8_params, config
                );
            }

            const int8_t *cand_rq8 = model->rq8_dataset_buffer +
                                     (size_t)cand_id * (size_t)frame_elem;
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
        else if (!config->use_rq8 && eq16_active)
        {
            if (left >= 0)
            {
                long pref_l = (long)cl->members[left].frame_id;
                KNN_PREFETCH_T0(model->eq16_dataset_buffer + (size_t)pref_l * (size_t)frame_elem);
            }
            if (right < num_m)
            {
                long pref_r = (long)cl->members[right].frame_id;
                KNN_PREFETCH_T0(model->eq16_dataset_buffer + (size_t)pref_r * (size_t)frame_elem);
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
                      visited->query_rq8 != NULL && !visited->query_rq8_clipped);

    if (rq8_active)
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

    if (rq8_active && cl->rq8_transposed != NULL && cl->num_rq8_blocks > 0)
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

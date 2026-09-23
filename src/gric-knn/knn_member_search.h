/**
 * @file knn_member_search.h
 * @brief Intra-cluster and candidate-cluster member evaluation, fastscan, and batching.
 *
 * Declares member evaluation workflows and data structures. Defines KnnDistanceBatch for
 * buffering raw frame distance computations and provides helpers for candidate lower-bound
 * filtering, annular distance interval traversal, and batch execution.
 */

#ifndef KNN_MEMBER_SEARCH_H
#define KNN_MEMBER_SEARCH_H

#include "knn_engine_internal.h"
#include "knn_pruning.h"
#include "knn_reader.h"
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_popcount32() - Count the number of set bits in a 32-bit unsigned integer.
 * @value: 32-bit bitmask to inspect.
 *
 * Return: Number of bits set to 1 (population count).
 */
static inline int knn_popcount32(
    uint32_t value)
{
#ifdef _MSC_VER
    return (int)__popcnt(value);
#else
    return __builtin_popcount(value);
#endif
}

/**
 * knn_ctz32() - Count trailing zero bits in a non-zero 32-bit integer.
 * @value: 32-bit integer bitmask (must be non-zero).
 *
 * Return: Number of trailing zero bits (0 to 31), identifying the lowest set bit index.
 */
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
 * struct KnnCandidateBatch - Accumulator for batched distance evaluations.
 * @cand_ids: Array of up to 8 candidate frame indices.
 * @ptrs:     Array of up to 8 pointers to candidate pixel buffers.
 * @count:    Number of valid entries currently accumulated in the batch [0..8].
 */
typedef struct
{
    long        cand_ids[8];
    const void *ptrs[8];
    int         count;
} KnnCandidateBatch;

/**
 * knn_batch_init() - Zero-initialize a candidate evaluation batch.
 * @batch: Pointer to KnnCandidateBatch struct.
 */
void knn_batch_init(
    KnnCandidateBatch *batch);

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
    KnnTelemetry  *restrict telem);

/**
 * knn_resolve_candidate_data() - Resolve candidate frame vector pointer.
 * @cand_id:     Global index of candidate frame.
 * @m_idx:       Cluster-local member offset.
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
 * @query_id:          Index of query frame.
 * @cand_id:           Index of candidate frame.
 * @r_cand:            Precomputed distance from cluster anchor to candidate.
 * @d_anchor:          Distance from query to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @tau_thresh:        Current maximum distance threshold in heap.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @config:            Active KnnConfig.
 * @heap:              Per-query max-heap.
 * @visited:           Frame visited tracker.
 * @telem:             Thread-local telemetry record.
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
 * @cand_id:      Global index of candidate frame.
 * @cand_ptr:     Pointer to candidate frame data.
 * @query_id:     Frame index of current query.
 * @query_data:   Raw pixel vector for query frame.
 * @model:        Active KnnModel.
 * @config:       Active KnnConfig.
 * @heap:         Per-query max-heap.
 * @all_heaps:    Global array of all heaps.
 * @bucket_locks: OpenMP bucket locks array.
 * @batch:        Pointer to KnnCandidateBatch.
 * @telem:        Thread-local telemetry record.
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
        double eps_factor = 1.0 + (double)config->epsilon;
        double tau_thresh = (double)heap->tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }
        double cutoff_sq = (heap->count >= heap->k || config->rlim_cutoff > 0.0)
                         ? (tau_thresh * tau_thresh) : 0.0;
        double d_exact = compute_euclidean_distance_cutoff(
            query_data, cand_ptr, model->frame_elements, model->is_double, cutoff_sq
        );
        if (cutoff_sq <= 0.0 || d_exact <= tau_thresh)
        {
            record_neighbor_and_reciprocal(
                query_id,
                cand_id,
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
 * knn_eval_cluster_members() - Search all members of a cluster with metric and SIMD pruning.
 * @c:                 Index of target cluster to evaluate.
 * @d_anchor:          Computed distance from query frame to target cluster anchor.
 * @home_cluster_id:   Query's home cluster index (or -1 if unassigned).
 * @r_home:            Distance to query's home cluster anchor.
 * @anchor_is_sq16:    1 if d_anchor was computed via SQ16, 0 otherwise.
 * @query_id:          Index of query frame.
 * @query_data:        Raw pixel buffer for query frame.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            KnnFrameReader context for reading on-disk frames.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Global array of all frame heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @pivots:            Array of measured anchor pivots (or NULL).
 * @num_pivots:        Number of active anchor pivots.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
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
    KnnTelemetry  *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_MEMBER_SEARCH_H

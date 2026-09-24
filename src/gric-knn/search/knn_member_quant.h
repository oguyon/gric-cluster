/**
 * @file knn_member_quant.h
 * @brief Quantization-accelerated member candidate search routines (RaBitQ, PQ, RQ8, EQ16, SQ16).
 *
 * Declares high-throughput member evaluation kernels that search candidate frames within
 * clusters using quantized codes. These vectorized routines evaluate 32-vector blocks at a
 * time using SIMD Look-Up Tables (LUTs) or integer distance instructions to identify nearest
 * neighbor candidates while minimizing memory bandwidth.
 */

#ifndef KNN_MEMBER_QUANT_H
#define KNN_MEMBER_QUANT_H

#include "knn_member_search.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_eval_members_rabitq() - Evaluate cluster members using RaBitQ FastScan.
 * @cl:                Pointer to candidate cluster.
 * @d_anchor:          Distance to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @query_id:          Index of query frame.
 * @query_data:        Raw query vector.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            Frame reader context.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Per-query max-heap.
 * @all_heaps:         Global array of all heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_members_rabitq(
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
    KnnTelemetry  *restrict telem);

/**
 * knn_eval_members_pq() - Evaluate cluster members using Product Quantization FastScan.
 * @cl:                Pointer to candidate cluster.
 * @d_anchor:          Distance to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @query_id:          Index of query frame.
 * @query_data:        Raw query vector.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            Frame reader context.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Per-query max-heap.
 * @all_heaps:         Global array of all heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_members_pq(
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
    KnnTelemetry  *restrict telem);

/**
 * knn_eval_members_rq8_blocks() - Evaluate cluster members using 8-bit Residual Quantization.
 * @cl:                Pointer to candidate cluster.
 * @d_anchor:          Distance to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @query_id:          Index of query frame.
 * @query_data:        Raw query vector.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            Frame reader context.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Per-query max-heap.
 * @all_heaps:         Global array of all heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_members_rq8_blocks(
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
    KnnTelemetry  *restrict telem);

/**
 * knn_eval_members_eq16_blocks() - Evaluate cluster members using E8-lattice Quantization.
 * @cl:                Pointer to candidate cluster.
 * @d_anchor:          Distance to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @query_id:          Index of query frame.
 * @query_data:        Raw query vector.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            Frame reader context.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Per-query max-heap.
 * @all_heaps:         Global array of all heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_members_eq16_blocks(
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
    KnnTelemetry  *restrict telem);

/**
 * knn_eval_members_sq16_blocks() - Evaluate cluster members using 16-bit Scalar Quantization.
 * @cl:                Pointer to candidate cluster.
 * @d_anchor:          Distance to candidate cluster anchor.
 * @r_home:            Distance to query's home cluster anchor.
 * @dcc_home:          Inter-anchor distance between home and candidate cluster.
 * @sq16_delta:        Quantization error bound margin.
 * @num_active_pivots: Number of active measured anchor pivots.
 * @pivot_diffs:       Array of pivot distance differentials.
 * @query_id:          Index of query frame.
 * @query_data:        Raw query vector.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            Frame reader context.
 * @cand_buffer:       Thread-local scratch buffer for reading frames.
 * @heap:              Per-query max-heap.
 * @all_heaps:         Global array of all heaps.
 * @bucket_locks:      OpenMP bucket locks array.
 * @visited:           Frame visited tracker.
 * @batch:             Pointer to shared KnnCandidateBatch.
 * @telem:             Thread-local telemetry record.
 */
void knn_eval_members_sq16_blocks(
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
    KnnTelemetry  *restrict telem);

#ifdef __cplusplus
}
#endif

#endif // KNN_MEMBER_QUANT_H

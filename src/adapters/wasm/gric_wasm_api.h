#ifndef GRIC_WASM_API_H
#define GRIC_WASM_API_H

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * wasm_cluster_init() - Instantiate and initialize a single-stream clustering engine.
 * @rlim:                       Radius limit for cluster creation.
 * @maxnbclust:                 Maximum capacity of cluster table.
 * @maxnbfr:                    Maximum frame count capacity.
 * @ndim:                       Dimensionality of input frames.
 * @entropy_mode:               Entropy-based candidate pruning mode.
 * @te4_mode:                   4-point triangle inequality pruning mode.
 * @te5_mode:                   5-point tetrahedron inequality pruning mode.
 * @pred_mode:                  Predictive candidate ranking mode.
 * @pred_h:                     Prediction history horizon.
 * @gprob_mode:                 Greedy probability guidance mode.
 * @tm_mixing_coeff:            Transition matrix mixing coefficient.
 * @soft_bayesian_mode:         Soft Bayesian probability mode.
 * @xtile_mode:                 Cross-tile correlation mode.
 * @sparse_dcc_mode:            Sparse DCC matrix update mode.
 * @sparse_dcc_extra_evals:     Extra evaluations per frame in sparse DCC mode.
 * @entropy_gate_bits:          Information gain threshold for entropy gating.
 * @entropy_first_gate_bits:    First-stage information gain threshold.
 * @entropy_fast_mode:          Accelerated entropy evaluation mode.
 * @soft_bayesian_sigma_coeff:  Standard deviation scale for soft Bayesian.
 * @maxcl_strategy:             Cluster eviction strategy on capacity exhaustion.
 * @discard_fraction:           Fraction of clusters to discard when full.
 * @max_gprob_visitors:         Max visitor count for greedy probability search.
 *
 * Return: Opaque cluster handle pointer, or NULL on allocation error.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_cluster_init(
    double rlim,
    int    maxnbclust,
    long   maxnbfr,
    int    ndim,
    int    entropy_mode,
    int    te4_mode,
    int    te5_mode,
    int    pred_mode,
    int    pred_h,
    int    gprob_mode,
    double tm_mixing_coeff,
    int    soft_bayesian_mode,
    int    xtile_mode,
    int    sparse_dcc_mode,
    int    sparse_dcc_extra_evals,
    double entropy_gate_bits,
    double entropy_first_gate_bits,
    int    entropy_fast_mode,
    double soft_bayesian_sigma_coeff,
    int    maxcl_strategy,
    double discard_fraction,
    int    max_gprob_visitors);

/**
 * wasm_cluster_process_frame() - Cluster a single incoming frame.
 * @handle: Opaque cluster engine handle.
 * @coords: Pointer to frame coordinate array [ndim].
 * @ndim:   Dimensionality of coordinate array.
 *
 * Return: Cluster index assigned to frame, or -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_frame(
    void   *handle,
    double *coords,
    int     ndim);

/**
 * wasm_cluster_process_batch() - Process multiple frames in sequence.
 * @handle:          Opaque cluster engine handle.
 * @coords_flat:     Contiguous array of frame coordinates [num_frames * ndim].
 * @out_assignments: Output array for cluster assignment indices [num_frames].
 * @num_frames:      Number of frames in batch.
 * @ndim:            Dimensionality of coordinates.
 *
 * Return: 0 on success, -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_batch(
    void   *handle,
    double *coords_flat,
    int    *out_assignments,
    int     num_frames,
    int     ndim);

/**
 * wasm_cluster_get_num_clusters() - Query current number of active clusters.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Current cluster count.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_num_clusters(
    void *handle);

/**
 * wasm_cluster_get_anchors() - Export cluster anchor vectors and member counts.
 * @handle:      Opaque cluster engine handle.
 * @out_coords:  Output buffer for anchor coordinates [num_clusters * ndim].
 * @out_members: Output buffer for cluster member counts [num_clusters].
 * @ndim:        Dimensionality of coordinate space.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_anchors(
    void   *handle,
    double *out_coords,
    int    *out_members,
    int     ndim);

/**
 * wasm_cluster_get_dcc() - Export distance-between-cluster-centers matrix.
 * @handle:  Opaque cluster engine handle.
 * @out_dcc: Output buffer for square DCC matrix [K * K].
 * @K:       Number of clusters.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_dcc(
    void   *handle,
    double *out_dcc,
    int     K);

/**
 * wasm_cluster_get_transition_matrix() - Export temporal transition count matrix.
 * @handle: Opaque cluster engine handle.
 * @out_tm: Output buffer for square transition matrix [K * K].
 * @K:      Number of clusters.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_transition_matrix(
    void *handle,
    long *out_tm,
    int   K);

/**
 * wasm_cluster_get_telemetry() - Export cumulative runtime telemetry counters.
 * @handle:    Opaque cluster engine handle.
 * @out_stats: Output buffer for telemetry doubles.
 * @out_len:   Output pointer to number of stats written.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_telemetry(
    void   *handle,
    double *out_stats,
    int    *out_len);

/**
 * wasm_cluster_get_probs() - Export current posterior cluster probability distribution.
 * @handle:    Opaque cluster engine handle.
 * @out_probs: Output buffer for probability array [K].
 * @K:         Number of clusters.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_probs(
    void   *handle,
    double *out_probs,
    int     K);

/**
 * wasm_cluster_get_evaluations() - Export distance evaluations performed for last frame.
 * @handle:      Opaque cluster engine handle.
 * @out_indices: Output buffer for evaluated cluster indices [max_evals].
 * @out_dists:   Output buffer for measured distances [max_evals].
 * @max_evals:   Capacity of output buffers.
 *
 * Return: Number of distance evaluations written.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_evaluations(
    void   *handle,
    int    *out_indices,
    double *out_dists,
    int     max_evals);

/**
 * wasm_cluster_reset() - Reset clustering state and clear all clusters.
 * @handle: Opaque cluster engine handle.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_reset(
    void *handle);

/**
 * wasm_cluster_free() - Destroy cluster engine instance and release all memory.
 * @handle: Opaque cluster engine handle.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_free(
    void *handle);

/**
 * wasm_cluster_set_trace() - Enable or disable detailed step-by-step event tracing.
 * @handle:   Opaque cluster engine handle.
 * @enabled:  1 to enable tracing, 0 to disable.
 * @capacity: Maximum number of events to retain in ring buffer.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_set_trace(
    void *handle,
    int   enabled,
    int   capacity);

/**
 * wasm_cluster_get_trace_count() - Query total number of recorded trace events.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Number of trace events currently buffered.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_count(
    void *handle);

/**
 * wasm_cluster_get_trace_events() - Get direct pointer to ring buffer trace events.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Pointer to internal TraceEvent array.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_cluster_get_trace_events(
    void *handle);

/**
 * wasm_cluster_get_trace_event_size() - Query size of TraceEvent struct in bytes.
 *
 * Return: sizeof(TraceEvent).
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_event_size(
    void);

/**
 * wasm_cluster_get_trace_head() - Get current write head offset in trace ring buffer.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Ring buffer head index.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_head(
    void *handle);

/**
 * wasm_cluster_get_trace_frame_start() - Get start index of events for current frame.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Trace index where current frame began.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_frame_start(
    void *handle);

/**
 * wasm_cluster_clear_trace() - Clear all recorded events in trace ring buffer.
 * @handle: Opaque cluster engine handle.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_clear_trace(
    void *handle);

/**
 * wasm_cluster_set_unlimited() - Enable or disable dynamic cluster table reallocation.
 * @handle:    Opaque cluster engine handle.
 * @unlimited: Non-zero to allow unlimited automatic growth.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_set_unlimited(
    void *handle,
    int   unlimited);

/**
 * wasm_cluster_get_capacity() - Query current cluster table allocation capacity.
 * @handle: Opaque cluster engine handle.
 *
 * Return: Current maximum number of clusters before reallocation.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_capacity(
    void *handle);

/**
 * wasm_cluster_get_version() - Return build version string of WASM engine.
 *
 * Return: Static null-terminated version string.
 */
EMSCRIPTEN_KEEPALIVE
const char *wasm_cluster_get_version(
    void);

/**
 * wasm_cluster_reassign_nearest() - Reassign all frames to strictly nearest cluster anchor.
 * @handle:          Opaque cluster engine handle.
 * @coords_flat:     Contiguous coordinates array [num_frames * ndim].
 * @num_frames:      Number of frames to reassign.
 * @ndim:            Dimensionality of coordinates.
 * @out_assignments: Output array for assigned cluster indices [num_frames].
 *
 * Return: 0 on success, -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_reassign_nearest(
    void         *handle,
    const double *coords_flat,
    int           num_frames,
    int           ndim,
    int          *out_assignments);

/* ---- Multi-tile WASM API ---- */

/**
 * wasm_multitile_init() - Instantiate a multi-tile clustered stream engine.
 * @rlim:                       Radius limit for cluster creation.
 * @maxnbclust:                 Maximum cluster capacity per tile.
 * @maxnbfr:                    Maximum frame count capacity.
 * @ndim:                       Total frame dimensionality.
 * @entropy_mode:               Entropy-based candidate pruning mode.
 * @te4_mode:                   4-point triangle inequality pruning mode.
 * @te5_mode:                   5-point tetrahedron inequality pruning mode.
 * @pred_mode:                  Predictive candidate ranking mode.
 * @pred_h:                     Prediction history horizon.
 * @gprob_mode:                 Greedy probability guidance mode.
 * @tm_mixing_coeff:            Transition matrix mixing coefficient.
 * @soft_bayesian_mode:         Soft Bayesian probability mode.
 * @xtile_mode:                 Cross-tile correlation mode.
 * @sparse_dcc_mode:            Sparse DCC matrix update mode.
 * @sparse_dcc_extra_evals:     Extra evaluations per frame in sparse DCC mode.
 * @entropy_gate_bits:          Information gain threshold for entropy gating.
 * @entropy_first_gate_bits:    First-stage information gain threshold.
 * @entropy_fast_mode:          Accelerated entropy evaluation mode.
 * @soft_bayesian_sigma_coeff:  Standard deviation scale for soft Bayesian.
 * @maxcl_strategy:             Cluster eviction strategy on capacity exhaustion.
 * @discard_fraction:           Fraction of clusters to discard when full.
 * @max_gprob_visitors:         Max visitor count for greedy probability search.
 *
 * Return: Opaque multi-tile engine handle, or NULL on error.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_multitile_init(
    double rlim,
    int    maxnbclust,
    long   maxnbfr,
    int    ndim,
    int    entropy_mode,
    int    te4_mode,
    int    te5_mode,
    int    pred_mode,
    int    pred_h,
    int    gprob_mode,
    double tm_mixing_coeff,
    int    soft_bayesian_mode,
    int    xtile_mode,
    int    sparse_dcc_mode,
    int    sparse_dcc_extra_evals,
    double entropy_gate_bits,
    double entropy_first_gate_bits,
    int    entropy_fast_mode,
    double soft_bayesian_sigma_coeff,
    int    maxcl_strategy,
    double discard_fraction,
    int    max_gprob_visitors);

/**
 * wasm_multitile_process_frame() - Feed next multi-tile frame into engine.
 * @handle: Opaque multi-tile engine handle.
 * @coords: Array of coordinates across all tiles.
 * @ndim:   Total coordinate count.
 *
 * Return: 0 on success, -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_process_frame(
    void   *handle,
    double *coords,
    int     ndim);

/**
 * wasm_multitile_get_num_tiles() - Query configured number of tiles.
 * @handle: Opaque multi-tile engine handle.
 *
 * Return: Number of tiles.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_num_tiles(
    void *handle);

/**
 * wasm_multitile_get_num_tile_clusters() - Query active cluster count for specific tile.
 * @handle:  Opaque multi-tile engine handle.
 * @tile_id: Tile index.
 *
 * Return: Cluster count in specified tile.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_num_tile_clusters(
    void *handle,
    int   tile_id);

/**
 * wasm_multitile_get_tile_clusters() - Export cluster anchors for given tile.
 * @handle:      Opaque multi-tile engine handle.
 * @tile_id:     Tile index.
 * @out_coords:  Output buffer for anchor coordinates.
 * @out_members: Output buffer for member counts.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_get_tile_clusters(
    void   *handle,
    int     tile_id,
    double *out_coords,
    int    *out_members);

/**
 * wasm_multitile_get_tuples() - Export cross-tile active tuple states.
 * @handle:          Opaque multi-tile engine handle.
 * @out_flat:        Output buffer for flattened tuple cluster IDs.
 * @out_counts:      Output buffer for tuple frequencies.
 * @out_last_active: Output buffer for frame index of last activation.
 * @max_tuples:      Capacity of output arrays.
 *
 * Return: Number of tuples written.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_multitile_get_tuples(
    void *handle,
    int  *out_flat,
    int  *out_counts,
    int  *out_last_active,
    int   max_tuples);

/**
 * wasm_multitile_get_tile_telemetry() - Export telemetry metrics for specific tile.
 * @handle:    Opaque multi-tile engine handle.
 * @tile_id:   Tile index.
 * @out_stats: Output buffer for stats doubles.
 * @out_len:   Output pointer to count of stats written.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_get_tile_telemetry(
    void   *handle,
    int     tile_id,
    double *out_stats,
    int    *out_len);

/**
 * wasm_multitile_reset() - Reset all tiles to initial unclustered state.
 * @handle: Opaque multi-tile engine handle.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_reset(
    void *handle);

/**
 * wasm_multitile_free() - Destroy multi-tile engine and release all resources.
 * @handle: Opaque multi-tile engine handle.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_multitile_free(
    void *handle);

/* ---- k-NN WASM API ---- */

/**
 * wasm_knn_run_search() - Execute k-NN graph and cluster-accelerated search in WASM.
 * @handle:           Opaque cluster/knn engine handle.
 * @dataset_points:   Dataset frames array [total_frames * ndim].
 * @total_frames:     Total number of frames in dataset.
 * @ndim:             Dimension of vector space.
 * @k:                Number of nearest neighbors to find.
 * @min_temporal_sep: Minimum frame separation to prevent auto-correlation.
 * @past_only:        Restricted to past neighbors only.
 * @future_only:      Restricted to future neighbors only.
 * @epsilon:          Graph search exploration slack factor.
 * @rlim_cutoff:      Cluster radius cutoff threshold.
 * @use_rq8:          Use 8-bit residual quantization.
 * @use_multi_pivot:  Use multi-pivot triangular pruning.
 * @out_indices:      Output neighbor index array [total_frames * k].
 * @out_distances:    Output neighbor distance array [total_frames * k].
 * @out_telemetry:    Output array for search telemetry statistics.
 *
 * Return: 0 on success, non-zero on failure.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_knn_run_search(
    void         *handle,
    const double *dataset_points,
    long          total_frames,
    int           ndim,
    int           k,
    int           min_temporal_sep,
    int           past_only,
    int           future_only,
    double        epsilon,
    double        rlim_cutoff,
    int           use_rq8,
    int           use_multi_pivot,
    int          *out_indices,
    double       *out_distances,
    double       *out_telemetry);

#ifdef __cplusplus

}
#endif

#endif /* GRIC_WASM_API_H */

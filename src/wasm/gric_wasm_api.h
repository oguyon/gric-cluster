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
    double entropy_gate_bits,
    double entropy_first_gate_bits,
    int    entropy_fast_mode,
    double soft_bayesian_sigma_coeff,
    int    maxcl_strategy,
    double discard_fraction,
    int    max_gprob_visitors);

EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_frame(void *handle, double *coords, int ndim);

EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_batch(
    void   *handle,
    double *coords_flat,
    int    *out_assignments,
    int     num_frames,
    int     ndim);

EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_num_clusters(void *handle);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_anchors(void *handle, double *out_coords, int *out_members, int ndim);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_dcc(void *handle, double *out_dcc, int K);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_transition_matrix(void *handle, long *out_tm, int K);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_telemetry(void *handle, double *out_stats, int *out_len);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_probs(void *handle, double *out_probs, int K);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_reset(void *handle);

EMSCRIPTEN_KEEPALIVE
void wasm_cluster_free(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_WASM_API_H */

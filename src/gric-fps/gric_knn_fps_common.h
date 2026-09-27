/**
 * @file gric_knn_fps_common.h
 * @brief Common adapter declarations for GRIC k-NN Milk streaming integration.
 */

#ifndef GRIC_KNN_FPS_COMMON_H
#define GRIC_KNN_FPS_COMMON_H

#include <libmilkcommon/milk_compiler.h>
#include "fps.h"
#include "fps_procinfo_macros.h"
#include "processinfo.h"
#include <ImageStreamIO/ImageStreamIO.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Global parameter variables mapped by FPS */
extern char     fps_knn_in_name[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_knn_out_name[FUNCTION_PARAMETER_STRMAXLEN];
extern uint64_t fps_knn_cnt2sync;
extern uint64_t fps_knn_allow_frame_drop;
extern char     fps_knn_cluster_dir[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_knn_ref_data[FUNCTION_PARAMETER_STRMAXLEN];
extern uint32_t fps_knn_k;
extern double   fps_knn_rlim;
extern double   fps_knn_eps;
extern uint64_t fps_knn_max_frames;
extern uint32_t fps_knn_ncpu;
extern uint64_t fps_knn_use_double;
extern uint64_t fps_knn_use_sq8;
extern uint64_t fps_knn_use_sq16;
extern uint64_t fps_knn_use_eq16;
extern uint64_t fps_knn_use_rq8;
extern uint64_t fps_knn_approx_mode;
extern uint64_t fps_knn_use_cluster_graph;

/* Status and telemetry parameter variables */
extern uint64_t fps_knn_status_queries;
extern uint32_t fps_knn_status_k;
extern uint64_t fps_knn_status_dist_evals;
extern double   fps_knn_status_latency_us;
extern double   fps_knn_status_fps;
extern double   fps_knn_status_min_dist;
extern double   fps_knn_status_max_dist;
extern int64_t  fps_knn_status_stream_lag;
extern double   fps_knn_status_memory_rss_mb;
extern char     fps_knn_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN];

/**
 * @brief Validate configuration parameters and check input stream.
 *
 * @return 0 on success, non-zero error code on invalid parameters.
 */
errno_t gric_knn_fps_custom_conf_check(void);

/**
 * @brief Allocate and initialize the resident k-NN model and search buffers.
 *
 * @param ndim Feature vector dimensionality (e.g. width * height).
 * @return 0 on success, non-zero on failure.
 */
errno_t gric_knn_fps_init_engine(
    uint32_t ndim);

/**
 * @brief Free resident k-NN model and per-frame search scratch buffers.
 */
void gric_knn_fps_cleanup_engine(void);

/**
 * @brief Create output ImageStreamIO stream for k-NN matches [k x 2].
 *
 * @param k       Number of nearest neighbors.
 * @param out_knn Pointer to output IMAGE struct.
 * @return 0 on success, non-zero on error.
 */
errno_t gric_knn_fps_init_output_streams(
    uint32_t  k,
    IMAGE    *out_knn);

/**
 * @brief Safely close output ImageStreamIO stream.
 *
 * @param out_knn Pointer to output IMAGE struct.
 */
void gric_knn_fps_close_output_streams(
    IMAGE *out_knn);

/**
 * @brief Search k-nearest neighbors for incoming frame and publish results.
 *
 * @param raw_pixels  Raw buffer pointer from ImageStreamIO input.
 * @param datatype    ImageStreamIO data type code (_DATATYPE_FLOAT, UINT16, etc.).
 * @param ndim        Number of pixels per frame.
 * @param frame_index Frame counter (cnt0).
 * @param frame_time  Frame arrival timestamp.
 * @param latency_us  Measured algorithm execution time in microseconds.
 * @param out_knn     Output k-NN matches stream.
 * @return 0 on success, non-zero on failure.
 */
errno_t gric_knn_fps_process_frame(
    const void      *raw_pixels,
    int              datatype,
    uint32_t         ndim,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_knn);

/**
 * @brief Initialize the status shared memory bridge file.
 *
 * @param fps_name    Name of the active FPS daemon instance.
 * @param custom_path Optional custom SHM status file path.
 * @return 0 on success, non-zero on error.
 */
errno_t gric_knn_fps_status_init(
    const char *fps_name,
    const char *custom_path);

/**
 * @brief Update real-time status parameters and bridge shared memory.
 *
 * @param frame_index Frame counter (cnt0).
 * @param latency_us  Measured algorithm execution time in microseconds.
 * @param stream_lag  Lag in frame count between stream write and read heads.
 * @param write_slice Current ring buffer write slice index.
 * @param read_slice  Current ring buffer read slice index.
 */
void gric_knn_fps_status_update(
    uint64_t frame_index,
    double   latency_us,
    long     stream_lag,
    long     write_slice,
    long     read_slice);

/**
 * @brief Finalize and unmap the status shared memory bridge file.
 *
 * @param exit_state Exit status code (0 for success, non-zero for error).
 */
void gric_knn_fps_status_close(
    int exit_state);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_KNN_FPS_COMMON_H */

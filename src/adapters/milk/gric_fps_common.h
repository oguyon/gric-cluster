/**
 * @file gric_fps_common.h
 * @brief Common adapter declarations for GRIC Milk streaming integration.
 */

#ifndef GRIC_FPS_COMMON_H
#define GRIC_FPS_COMMON_H

#include <libmilkcommon/milk_compiler.h>
#include "fps.h"
#include "fps_procinfo_macros.h"
#include "processinfo.h"
#include "shared/gric_stream_layout.h"
#include <ImageStreamIO/ImageStreamIO.h>
#include <gric/gric.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Global parameter variables mapped by FPS */
extern char     fps_in_name[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_out_assign_name[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_out_anchors_name[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_out_counts_name[FUNCTION_PARAMETER_STRMAXLEN];
extern uint64_t fps_stream_anchors;
extern uint64_t fps_stream_counts;
extern uint64_t fps_allow_frame_drop;
extern uint64_t fps_cnt2sync;
extern uint64_t fps_query_mode;
extern char     fps_load_anchors[FUNCTION_PARAMETER_STRMAXLEN];
extern double   fps_rlim;
extern double   fps_deltaprob;
extern uint32_t fps_maxnbclust;
extern int64_t  fps_maxcl_strategy;
extern uint64_t fps_max_frames;
extern uint32_t fps_ncpu;
extern uint64_t fps_use_double;
extern uint64_t fps_use_sq16;
extern uint64_t fps_entropy_mode;
extern uint64_t fps_reset_state;

/* Status and telemetry parameter variables */
extern uint64_t fps_status_frames_processed;
extern uint32_t fps_status_num_clusters;
extern uint64_t fps_status_new_clusters;
extern uint64_t fps_status_distance_evals;
extern double   fps_status_pruning_ratio;
extern double   fps_status_latency_us;
extern double   fps_status_fps;
extern int64_t  fps_status_stream_lag;
extern double   fps_status_memory_rss_mb;
extern char     fps_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_save_dir[FUNCTION_PARAMETER_STRMAXLEN];

/* Telemetry packet size for <out>_assign */
#define GRIC_ASSIGN_PACKET_ELEMS GRIC_ASSIGN_NFIELDS

/**
 * @brief Validate configuration parameters and check input stream.
 *
 * @return 0 on success, non-zero error code on invalid parameters.
 */
errno_t gric_fps_custom_conf_check(void);

/**
 * @brief Allocate and initialize the underlying libgric clustering session.
 *
 * @param ndim Feature vector dimensionality (e.g. width * height).
 * @return 0 on success, non-zero on failure.
 */
errno_t gric_fps_init_engine(
    uint32_t ndim);

/**
 * @brief Destroy the active libgric clustering session and conversion buffers.
 */
void gric_fps_cleanup_engine(void);

/**
 * @brief Export clustering results to disk in GRIC binary format.
 *
 * @param out_dir Destination output directory path.
 * @return 0 on success, non-zero on error.
 */
errno_t gric_fps_save_results(
    const char *out_dir);

/**
 * @brief Create shared memory output streams (_assign, _anchors, _counts).
 *
 * @param xsize        Input image width.
 * @param ysize        Input image height.
 * @param max_clusters Maximum clusters capacity.
 * @param out_assign   Pointer to assignment output IMAGE struct.
 * @param out_anchors  Pointer to anchors output IMAGE struct.
 * @param out_counts   Pointer to counts output IMAGE struct.
 * @return 0 on success, non-zero on error.
 */
errno_t gric_fps_init_output_streams(
    uint32_t xsize,
    uint32_t ysize,
    uint32_t max_clusters,
    IMAGE   *out_assign,
    IMAGE   *out_anchors,
    IMAGE   *out_counts);

/**
 * @brief Safely close all output streams.
 *
 * @param out_assign   Pointer to assignment output IMAGE struct.
 * @param out_anchors  Pointer to anchors output IMAGE struct.
 * @param out_counts   Pointer to counts output IMAGE struct.
 */
void gric_fps_close_output_streams(
    IMAGE *out_assign,
    IMAGE *out_anchors,
    IMAGE *out_counts);

/**
 * @brief Process an incoming frame and write results to output streams.
 *
 * @param raw_pixels  Raw buffer pointer from ImageStreamIO input.
 * @param datatype    ImageStreamIO data type code (_DATATYPE_FLOAT, UINT16, etc.).
 * @param ndim        Number of pixels per frame.
 * @param frame_index Frame counter (cnt0).
 * @param frame_time  Frame arrival timestamp.
 * @param latency_us  Measured algorithm execution time in microseconds.
 * @param out_assign  Assignment telemetry output stream.
 * @param out_anchors Centroids / anchors output stream.
 * @param out_counts  Cluster member counts output stream.
 * @return 0 on success, non-zero on failure.
 */
errno_t gric_fps_process_frame(
    const void      *raw_pixels,
    int              datatype,
    uint32_t         ndim,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_assign,
    IMAGE           *out_anchors,
    IMAGE           *out_counts);

/**
 * @brief Query total active clusters discovered in the current session.
 *
 * @return Number of active clusters (>= 0).
 */
int64_t gric_fps_get_cluster_count(void);

/**
 * @brief Initialize the status shared memory bridge file.
 *
 * @param fps_name    Name of the active FPS daemon instance.
 * @param custom_path Optional custom SHM status file path.
 * @return 0 on success, non-zero on error.
 */
errno_t gric_fps_status_init(
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
void gric_fps_status_update(
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
void gric_fps_status_close(
    int exit_state);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_FPS_COMMON_H */

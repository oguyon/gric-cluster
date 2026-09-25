/**
 * @file gric_fps_common.h
 * @brief Common adapter declarations for GRIC Milk streaming integration.
 */

#ifndef GRIC_FPS_COMMON_H
#define GRIC_FPS_COMMON_H

#include "milk_compiler.h"
#include "fps.h"
#include "fps_procinfo_macros.h"
#include "processinfo.h"
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
extern int32_t  fps_stream_anchors;
extern int32_t  fps_stream_counts;
extern int32_t  fps_allow_frame_drop;
extern double   fps_rlim;
extern double   fps_deltaprob;
extern uint32_t fps_maxnbclust;
extern int64_t  fps_maxcl_strategy;
extern uint32_t fps_ncpu;
extern int32_t  fps_use_double;
extern int32_t  fps_use_sq16;
extern int32_t  fps_entropy_mode;
extern int32_t  fps_reset_state;

/* Telemetry packet size for <out>_assign */
#define GRIC_ASSIGN_PACKET_ELEMS 8

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

#ifdef __cplusplus
}
#endif

#endif /* GRIC_FPS_COMMON_H */

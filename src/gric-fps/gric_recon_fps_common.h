/**
 * @file gric_recon_fps_common.h
 * @brief Common adapter header for GRIC dataset reconstruction Milk streaming integration.
 */

#ifndef GRIC_RECON_FPS_COMMON_H
#define GRIC_RECON_FPS_COMMON_H

#include <libmilkcommon/milk_compiler.h>
#include "fps.h"
#include "fps_procinfo_macros.h"
#include "processinfo.h"
#include <ImageStreamIO/ImageStreamIO.h>
#include <errno.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

extern char     fps_recon_in_name[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_recon_out_name[FUNCTION_PARAMETER_STRMAXLEN];
extern uint64_t fps_recon_cnt2sync;
extern uint64_t fps_recon_allow_frame_drop;
extern char     fps_recon_target_data[FUNCTION_PARAMETER_STRMAXLEN];
extern char     fps_recon_weight_mode[FUNCTION_PARAMETER_STRMAXLEN];
extern double   fps_recon_alpha;
extern uint32_t fps_recon_k;
extern uint64_t fps_recon_max_frames;
extern char     fps_recon_out_file[FUNCTION_PARAMETER_STRMAXLEN];

extern uint64_t fps_recon_status_frames;
extern double   fps_recon_status_latency_us;
extern double   fps_recon_status_fps;
extern double   fps_recon_status_variance;
extern double   fps_recon_status_k_eff;
extern int64_t  fps_recon_status_stream_lag;
extern double   fps_recon_status_memory_rss_mb;
extern char     fps_recon_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN];

errno_t gric_recon_fps_custom_conf_check(void);

errno_t gric_recon_fps_init_engine(
    uint32_t *out_b_dim);

void gric_recon_fps_cleanup_engine(void);

errno_t gric_recon_fps_init_output_streams(
    uint32_t  b_dim,
    IMAGE    *out_recon);

void gric_recon_fps_close_output_streams(
    IMAGE *out_recon);

errno_t gric_recon_fps_process_frame(
    const void      *raw_matches,
    int              datatype,
    uint32_t         match_k,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_recon);

void gric_recon_fps_status_init(
    const char *instance_name,
    const char *custom_status_path);

void gric_recon_fps_status_update(
    uint64_t frame_index,
    double   latency_us,
    int64_t  stream_lag,
    long     write_slice,
    long     read_slice);

void gric_recon_fps_status_close(
    int unlink_shm);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_RECON_FPS_COMMON_H */

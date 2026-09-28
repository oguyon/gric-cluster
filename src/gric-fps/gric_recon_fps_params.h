/**
 * @file gric_recon_fps_params.h
 * @brief X-Macro parameter definitions for GRIC dataset reconstruction Milk FPS integration.
 */

#ifndef GRIC_RECON_FPS_PARAMS_H
#define GRIC_RECON_FPS_PARAMS_H

#include "fps.h"

/**
 * @brief GRIC Reconstruction Parameter Definition Macro.
 *
 * Defines all parameters used by the GRIC Reconstruction Milk streaming processor.
 */
#define GRIC_RECON_FPS_PARAMS(X) \
    /* Stream bindings */ \
    X(".in_name", fps_recon_in_name, FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_TRIGGER_STREAM, "Input k-NN matches stream [k x 2]") \
    X(".out_name", fps_recon_out_name, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Output reconstructed stream name") \
    X(".cnt2sync", &fps_recon_cnt2sync, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Enable cnt2 flow-control handshaking") \
    X(".allow_frame_drop", &fps_recon_allow_frame_drop, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Drop frames if lagging (1=latest)") \
    /* Target dataset & weights */ \
    X(".target_data", fps_recon_target_data, FPTYPE_FILENAME, 1, \
      FPFLAG_DEFAULT_INPUT, "Target dataset B file (.bin, .fits, .txt)") \
    X(".weight_mode", fps_recon_weight_mode, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Weighting mode: uniform or idw") \
    X(".alpha", &fps_recon_alpha, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Inverse distance weighting exponent alpha") \
    X(".k", &fps_recon_k, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "Neighbors to average (0 = all in stream)") \
    X(".max_frames", &fps_recon_max_frames, FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Maximum frames to reconstruct (0=unbounded)") \
    X(".out_file", fps_recon_out_file, FPTYPE_FILENAME, 1, \
      FPFLAG_DEFAULT_INPUT, "Optional output reconstructed ASCII file (.txt)") \
    /* Status & Telemetry outputs (read-only in FPS) */ \
    X(".status.frames_processed", &fps_recon_status_frames, FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Total frames reconstructed") \
    X(".status.latency_us", &fps_recon_status_latency_us, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Per-frame reconstruction latency (us)") \
    X(".status.fps", &fps_recon_status_fps, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Instantaneous processing rate (FPS)") \
    X(".status.variance", &fps_recon_status_variance, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Target reconstruction dispersion / variance") \
    X(".status.k_eff", &fps_recon_status_k_eff, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Effective number of NN points averaged") \
    X(".status.stream_lag", &fps_recon_status_stream_lag, FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Input stream lag (write - read)") \
    X(".status.memory_rss_mb", &fps_recon_status_memory_rss_mb, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Process RSS memory consumption (MB)") \
    X(".shm_status_file", fps_recon_shm_status_file, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Bridge status SHM file path (blank=default)")

#endif /* GRIC_RECON_FPS_PARAMS_H */

/**
 * @file gric_fps_params.h
 * @brief X-Macro parameter definitions for GRIC Milk FPS integration.
 */

#ifndef GRIC_FPS_PARAMS_H
#define GRIC_FPS_PARAMS_H

#include "fps.h"

/**
 * @brief GRIC Parameter Definition Macro.
 *
 * Defines all parameters used by the GRIC Milk streaming processor.
 * Columns:
 * 1. KEY:       The hierarchical parameter key (e.g. ".in_name").
 * 2. PTR:       Address of the local variable / pointer holding the value.
 * 3. FPTYPE:    FPS data type.
 * 4. FPFLAG:    Parameter behavior flags (1 = active/used).
 * 5. CLIFLAG:   CLI input flags (FPFLAG_DEFAULT_INPUT).
 * 6. DESCR:     Human-readable description displayed in milk-fpsCTRL and help.
 */
#define GRIC_FPS_PARAMS(X) \
    /* Stream bindings */ \
    X(".in_name", fps_in_name, FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_TRIGGER_STREAM, "Input ImageStreamIO stream") \
    X(".out_name", fps_out_assign_name, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Output assignment stream name") \
    X(".out_anchors", fps_out_anchors_name, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Output anchors stream name") \
    X(".out_counts", fps_out_counts_name, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Output counts stream name") \
    /* Stream toggles & lag policy */ \
    X(".stream_anchors", &fps_stream_anchors, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Publish anchors stream live") \
    X(".stream_counts", &fps_stream_counts, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Publish counts stream live") \
    X(".allow_frame_drop", &fps_allow_frame_drop, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Drop frames if lagging (1=latest)") \
    /* Clustering parameters */ \
    X(".rlim", &fps_rlim, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Cluster radius limit threshold") \
    X(".deltaprob", &fps_deltaprob, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Neighbor search probability cutoff") \
    X(".maxnbclust", &fps_maxnbclust, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "Maximum cluster capacity") \
    X(".maxcl_strategy", &fps_maxcl_strategy, FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Capacity strategy (0=stop, 1=discard)") \
    /* Compute & Acceleration */ \
    X(".ncpu", &fps_ncpu, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "OpenMP thread count") \
    X(".use_double", &fps_use_double, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Use double precision compute") \
    X(".use_sq16", &fps_use_sq16, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "16-bit scalar quantization filter") \
    X(".entropy_mode", &fps_entropy_mode, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Shannon entropy target selection") \
    /* Dynamic Triggers */ \
    X(".reset_state", &fps_reset_state, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Dynamic trigger: reset clusters")

#endif /* GRIC_FPS_PARAMS_H */

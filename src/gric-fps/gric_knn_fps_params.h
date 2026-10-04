/**
 * @file gric_knn_fps_params.h
 * @brief X-Macro parameter definitions for GRIC k-NN Milk FPS integration.
 */

#ifndef GRIC_KNN_FPS_PARAMS_H
#define GRIC_KNN_FPS_PARAMS_H

/* Dependency-free header: consumer must include fps.h if using Milk symbols */
/**
 * @brief GRIC k-NN Parameter Definition Macro.
 *
 * Defines all parameters used by the GRIC k-NN Milk streaming processor.
 */
#define GRIC_KNN_FPS_PARAMS(X) \
    /* Stream bindings */ \
    X(".in_name", fps_knn_in_name, FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_TRIGGER_STREAM, "Input ImageStreamIO query stream") \
    X(".out_name", fps_knn_out_name, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Output k-NN matches stream name") \
    X(".cnt2sync", &fps_knn_cnt2sync, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Enable cnt2 flow-control handshaking") \
    X(".allow_frame_drop", &fps_knn_allow_frame_drop, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Drop frames if lagging (1=latest)") \
    /* Model & Reference dataset */ \
    X(".cluster_dir", fps_knn_cluster_dir, FPTYPE_DIRNAME, 1, \
      FPFLAG_DEFAULT_INPUT, "Pass 1 cluster directory (.clusterdat)") \
    X(".ref_data", fps_knn_ref_data, FPTYPE_FILENAME, 1, \
      FPFLAG_DEFAULT_INPUT, "Reference dataset file (.bin, .fits, .txt)") \
    /* k-NN search parameters */ \
    X(".k", &fps_knn_k, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "Target number of nearest neighbors (k)") \
    X(".rlim", &fps_knn_rlim, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Max distance cutoff (0 = unbounded)") \
    X(".eps", &fps_knn_eps, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Relaxation factor for (1+eps)-ANN") \
    X(".max_frames", &fps_knn_max_frames, FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_INPUT, "Maximum frames to query (0=unbounded)") \
    /* Compute & Acceleration */ \
    X(".ncpu", &fps_knn_ncpu, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, "OpenMP thread count") \
    X(".use_double", &fps_knn_use_double, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Use double precision compute") \
    X(".use_sq8", &fps_knn_use_sq8, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "8-bit scalar quantization filter") \
    X(".use_sq16", &fps_knn_use_sq16, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "16-bit scalar quantization filter") \
    X(".use_eq16", &fps_knn_use_eq16, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "16-bit E8 lattice quantization filter") \
    X(".use_rq8", &fps_knn_use_rq8, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "8-bit residual quantization filter") \
    X(".approx_mode", &fps_knn_approx_mode, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Fast approximate graph search") \
    X(".use_cluster_graph", &fps_knn_use_cluster_graph, FPTYPE_ONOFF, 1, \
      FPFLAG_DEFAULT_INPUT, "Graph-guided cluster routing") \
    /* Status & Telemetry outputs (read-only in FPS) */ \
    X(".status.queries_processed", &fps_knn_status_queries, FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Total queries processed") \
    X(".status.k", &fps_knn_status_k, FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Active neighbor count k") \
    X(".status.distance_evals", &fps_knn_status_dist_evals, FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Total distance evaluations") \
    X(".status.latency_us", &fps_knn_status_latency_us, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Per-frame query latency (us)") \
    X(".status.fps", &fps_knn_status_fps, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Instantaneous processing rate (FPS)") \
    X(".status.min_dist", &fps_knn_status_min_dist, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Distance to 1-NN nearest neighbor") \
    X(".status.max_dist", &fps_knn_status_max_dist, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Distance to kth neighbor (tau)") \
    X(".status.stream_lag", &fps_knn_status_stream_lag, FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Input stream lag (write - read)") \
    X(".status.memory_rss_mb", &fps_knn_status_memory_rss_mb, FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_OUTPUT, "Process RSS memory consumption (MB)") \
    X(".shm_status_file", fps_knn_shm_status_file, FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "Bridge status SHM file path (blank=default)")

#endif /* GRIC_KNN_FPS_PARAMS_H */

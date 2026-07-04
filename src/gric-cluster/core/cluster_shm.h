/**
 * @file cluster_shm.h
 * @brief Declarations for file-mapped shared memory real-time status tracking.
 */

#ifndef CLUSTER_SHM_H
#define CLUSTER_SHM_H

#include "cluster_defs.h"
#include <stdint.h>

#define GRIC_SHM_MAGIC   0x47524943 // 'GRIC'
#define GRIC_SHM_VERSION 3

typedef enum
{
    GRIC_STATUS_INIT    = 0,
    GRIC_STATUS_RUNNING = 1,
    GRIC_STATUS_SUCCESS = 2,
    GRIC_STATUS_ERROR   = 3,
    GRIC_STATUS_ABORTED = 4
} GricClusterStatusState;

typedef struct
{
    /* Version 1 Fields */
    uint32_t magic;                    // Magic bytes for validation
    uint32_t version;                  // Structure version layout
    uint32_t pid;                      // PID of the running gric-cluster process
    uint32_t status_state;             // Current run state (GricClusterStatusState)

    uint64_t total_frames;             // Total frames to process
    uint64_t total_frames_processed;   // Samples processed so far
    uint32_t num_clusters;             // Number of active clusters
    uint32_t padding;                  // Structure alignment padding

    uint64_t framedist_calls;          // Distance computations
    uint64_t framedist_calls_sample;   // Sample-to-cluster computations
    uint64_t framedist_calls_intercluster; // Inter-cluster computations
    uint64_t clusters_pruned;          // Total pruned candidates
    uint64_t total_missed_frames;      // Missed frames (for streaming)

    double   elapsed_ms;               // Current run time elapsed
    char     input_source[256];        // Input file path or stream name
    uint64_t last_update_time;         // Unix timestamp of the last update

    /* Version 2 Expanded Fields */
    double   stream_wait_time_sec;     // Total blocked time waiting for frames
    long     stream_read_slice;        // Current circular buffer read index
    long     stream_write_slice;       // Current circular buffer write index
    long     stream_lag;               // Buffer lag (write_slice - read_slice)

    double   last_assignment_dist;     // Distance of last sample to its anchor
    uint64_t num_new_clusters;         // Count of spawned clusters

    uint64_t memory_rss_kb;            // Process RAM usage (Resident Set Size)
    uint32_t active_threads;           // Thread count used by OpenMP
    uint32_t padding2;                 // Align struct to 64-bit boundaries

    /* Version 3 Expanded Fields */
    double   config_rlim;              // Distance threshold
    uint32_t config_maxnbclust;        // Limit of cluster instances
    double   config_dprob;             // Delta probability threshold
    uint32_t config_maxcl_strategy;    // 0: stop, 1: discard, 2: merge
    uint32_t config_te4_mode;          // TE4 triangle pruning boolean flag
    uint32_t config_te5_mode;          // TE5 triangle pruning boolean flag
    uint32_t config_gprob_mode;        // Geometrical probability boolean flag
    uint32_t config_sparse_dcc;        // Sparse DCC mode boolean flag
    uint32_t config_entropy_mode;      // Entropy vs greedy mode boolean flag
    char     config_cwd[256];          // CWD of the running process

    uint64_t last_frame_dists;         // Distance calls for the last frame
    uint64_t last_frame_dfc;           // Frame-to-cluster distance calls for the last frame
    uint64_t last_frame_dcc;           // Inter-cluster distance calls for the last frame

    double   time_io_ms;               // Time spent in frame I/O (ms)
    double   time_step_1;              // Step 1: Base case setup (ms)
    double   time_step_2;              // Step 2: Retrieve prediction candidates (ms)
    double   time_step_3a;             // Step 3a: Compute mixed priors & pruning (ms)
    double   time_step_3b;             // Step 3b: Select next target (ms)
    double   time_step_3b_score;       // Step 3b score calculation and sorting (ms)
    double   time_step_3b_filter;      // Step 3b candidate filtering & selection (ms)
    double   time_step_3b_eval;        // Step 3b expected entropy evaluation (ms)
    double   time_step_3c;             // Step 3c: Measure distance to target (ms)
    double   time_step_4;              // Step 4: Handle new cluster creation & eviction (ms)
    double   time_step_5;              // Step 5: Telemetry & file serialization (ms)
    double   time_step_refine;         // DCC bounds refinement (ms)

    /* Entropy telemetry fields */
    double   entropy_last_initial;     // H at meas_idx==0 for last frame
    double   entropy_avg_initial;      // Running average H at meas_idx==0
    double   entropy_gate_ratio;       // Fraction of calls gated
} GricClusterShmStatus;

/**
 * @brief Initialize the file-mapped shared memory file.
 */
int gric_shm_init(
    ClusterConfig *config,
    ClusterState  *state);

/**
 * @brief Update the mapped shared memory telemetry variables.
 */
void gric_shm_update(
    ClusterState *state,
    int           status_state,
    double        elapsed_ms);

/**
 * @brief Clean up and unmap the status shared memory.
 */
void gric_shm_cleanup(
    ClusterState *state);

#endif // CLUSTER_SHM_H

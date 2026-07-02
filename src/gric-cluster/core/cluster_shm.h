/**
 * @file cluster_shm.h
 * @brief Declarations for file-mapped shared memory real-time status tracking.
 */

#ifndef CLUSTER_SHM_H
#define CLUSTER_SHM_H

#include "cluster_defs.h"
#include <stdint.h>

#define GRIC_SHM_MAGIC   0x47524943 // 'GRIC'
#define GRIC_SHM_VERSION 1

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

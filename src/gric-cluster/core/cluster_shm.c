/**
 * @file cluster_shm.c
 * @brief Implementation of file-mapped shared memory status reporting.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_shm.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/**
 * @brief Initialize the file-mapped shared memory file.
 *
 * @param config Pointer to the active ClusterConfig.
 * @param state  Pointer to the active ClusterState.
 * @return 0 on success, -1 on error.
 */
int gric_shm_init(
    ClusterConfig *config,
    ClusterState  *state)
{
    if (config->output.shm_filename == NULL)
    {
        state->shm_ptr = NULL;
        return 0;
    }

    int fd = open(config->output.shm_filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
        perror("Failed to open status SHM file");
        return -1;
    }

    if (ftruncate(fd, sizeof(GricClusterShmStatus)) < 0)
    {
        perror("Failed to truncate status SHM file");
        close(fd);
        return -1;
    }

    void *ptr = mmap(NULL, sizeof(GricClusterShmStatus),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); /* fd is no longer needed after mmap */

    if (ptr == MAP_FAILED)
    {
        perror("Failed to mmap status SHM file");
        return -1;
    }

    GricClusterShmStatus *status = (GricClusterShmStatus *)ptr;
    memset(status, 0, sizeof(GricClusterShmStatus));
    status->magic = GRIC_SHM_MAGIC;
    status->version = GRIC_SHM_VERSION;
    status->pid = (uint32_t)getpid();
    status->status_state = GRIC_STATUS_INIT;
    status->total_frames = (uint64_t)config->input.maxnbfr;

    if (config->input.fits_filename != NULL)
    {
        strncpy(status->input_source, config->input.fits_filename,
                sizeof(status->input_source) - 1);
    }
    else
    {
        strcpy(status->input_source, "N/A");
    }

    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    status->last_update_time = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;

    state->shm_ptr = ptr;
    return 0;
}

/**
 * @brief Update the mapped shared memory telemetry variables.
 *
 * @param state        Pointer to the active ClusterState.
 * @param status_state Current state (GricClusterStatusState).
 * @param elapsed_ms   Current elapsed time in milliseconds.
 */
void gric_shm_update(
    ClusterState *state,
    int           status_state,
    double        elapsed_ms)
{
    if (state->shm_ptr == NULL)
    {
        return;
    }

    GricClusterShmStatus *status = (GricClusterShmStatus *)state->shm_ptr;
    status->status_state = (uint32_t)status_state;
    status->total_frames_processed = (uint64_t)state->telemetry.total_frames_processed;
    status->num_clusters = (uint32_t)state->num_clusters;
    status->framedist_calls = (uint64_t)state->telemetry.framedist_calls;
    status->framedist_calls_sample = (uint64_t)state->telemetry.framedist_calls_sample;
    status->framedist_calls_intercluster = (uint64_t)state->telemetry.framedist_calls_intercluster;
    status->clusters_pruned = (uint64_t)state->telemetry.clusters_pruned;
    status->total_missed_frames = (uint64_t)state->telemetry.total_missed_frames;
    status->elapsed_ms = elapsed_ms;

    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    status->last_update_time = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

/**
 * @brief Clean up and unmap the status shared memory.
 *
 * @param state Pointer to the active ClusterState.
 */
void gric_shm_cleanup(
    ClusterState *state)
{
    if (state->shm_ptr != NULL)
    {
        munmap(state->shm_ptr, sizeof(GricClusterShmStatus));
        state->shm_ptr = NULL;
    }
}

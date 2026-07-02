/**
 * @file cluster_shm.c
 * @brief Implementation of file-mapped shared memory status reporting.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_shm.h"
#include "frameread.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Query current process Resident Set Size (RSS) in KB.
 *
 * @return Current RSS in KB, or 0 on error.
 */
static uint64_t get_current_rss_kb(void)
{
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f)
    {
        return 0;
    }
    long pages = 0;
    if (fscanf(f, "%*d %ld", &pages) != 1)
    {
        fclose(f);
        return 0;
    }
    fclose(f);
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0)
    {
        page_size = 4096;
    }
    return (uint64_t)pages * (uint64_t)page_size / 1024ULL;
}

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

    /* Version 3 Configuration Parameters */
    status->config_rlim = config->algo.rlim;
    status->config_maxnbclust = (uint32_t)config->algo.maxnbclust;
    status->config_dprob = config->algo.deltaprob;
    status->config_maxcl_strategy = (uint32_t)config->algo.maxcl_strategy;
    status->config_te4_mode = (uint32_t)config->optim.te4_mode;
    status->config_te5_mode = (uint32_t)config->optim.te5_mode;
    status->config_gprob_mode = (uint32_t)config->optim.gprob_mode;
    status->config_sparse_dcc = (uint32_t)config->optim.sparse_dcc_mode;
    status->config_entropy_mode = (uint32_t)config->optim.entropy_mode;

    if (getcwd(status->config_cwd, sizeof(status->config_cwd)) == NULL)
    {
        strcpy(status->config_cwd, "N/A");
    }

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

    /* Version 2 Telemetry fields */
    status->stream_wait_time_sec = get_stream_wait_time();
    status->stream_read_slice = get_stream_read_slice();
    status->stream_write_slice = get_stream_write_slice();
    status->stream_lag = get_stream_lag();
    status->last_assignment_dist = state->telemetry.last_assignment_dist;
    status->num_new_clusters = (uint64_t)state->telemetry.num_new_clusters;

    /* Update memory RSS and OpenMP thread count */
    status->memory_rss_kb = get_current_rss_kb();
#ifdef _OPENMP
    status->active_threads = (uint32_t)omp_get_max_threads();
#else
    status->active_threads = 1;
#endif

    /* Version 3 Telemetry fields */
    status->last_frame_dists = (uint64_t)state->telemetry.last_frame_dists;
    status->last_frame_dfc = (uint64_t)state->telemetry.last_frame_dfc;
    status->last_frame_dcc = (uint64_t)state->telemetry.last_frame_dcc;
    status->time_io_ms = state->telemetry.time_io_ms;
    status->time_step_1 = state->telemetry.time_step_1;
    status->time_step_2 = state->telemetry.time_step_2;
    status->time_step_3a = state->telemetry.time_step_3a;
    status->time_step_3b = state->telemetry.time_step_3b;
    status->time_step_3b_score = state->telemetry.time_step_3b_score;
    status->time_step_3b_filter = state->telemetry.time_step_3b_filter;
    status->time_step_3b_eval = state->telemetry.time_step_3b_eval;
    status->time_step_3c = state->telemetry.time_step_3c;
    status->time_step_4 = state->telemetry.time_step_4;
    status->time_step_5 = state->telemetry.time_step_5;
    status->time_step_refine = state->telemetry.time_step_refine;

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

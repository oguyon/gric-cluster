/**
 * @file gric-status.c
 * @brief Command-line monitoring client for gric-cluster shared memory status.
 *
 * Attaches to the file-mapped status structure and displays progress in real-time.
 */

#define _POSIX_C_SOURCE 200809L
#include "gric-cluster/core/cluster_shm.h"
#include <inttypes.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <signal.h>
#include <errno.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

static int is_pid_alive(
    pid_t pid)
{
    if (kill(pid, 0) == 0)
    {
        return 1;
    }
    if (errno == ESRCH)
    {
        return 0;
    }
    return 1;
}

static const char *get_state_string(
    uint32_t state,
    pid_t    pid)
{
    if (state == GRIC_STATUS_INIT || state == GRIC_STATUS_RUNNING)
    {
        if (!is_pid_alive(pid))
        {
            return ANSI_COLOR_RED "DEAD / CRASHED" ANSI_COLOR_RESET;
        }
    }

    switch (state)
    {
        case GRIC_STATUS_INIT:
            return ANSI_COLOR_YELLOW "INITIALIZING" ANSI_COLOR_RESET;
        case GRIC_STATUS_RUNNING:
            return ANSI_COLOR_BLUE "RUNNING" ANSI_COLOR_RESET;
        case GRIC_STATUS_SUCCESS:
            return ANSI_COLOR_GREEN "SUCCESS" ANSI_COLOR_RESET;
        case GRIC_STATUS_ERROR:
            return ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET;
        case GRIC_STATUS_ABORTED:
            return ANSI_COLOR_RED "ABORTED" ANSI_COLOR_RESET;
        default:
            return "UNKNOWN";
    }
}

static void print_status(
    const GricClusterShmStatus *status)
{
    printf("\n%s--- gric-cluster telemetry status ---%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("PID:                  %u\n", status->pid);
    printf("State:                %s\n", get_state_string(status->status_state, (pid_t)status->pid));
    printf("Input Source:         %s\n", status->input_source);
    printf("Samples Processed:    %" PRIu64 " / %" PRIu64 "\n",
           status->total_frames_processed, status->total_frames);
    printf("Active Clusters:      %u\n", status->num_clusters);
    printf("Elapsed Time:         %.2f ms (%.3f s)\n",
           status->elapsed_ms, status->elapsed_ms / 1000.0);
    printf("Distance Computations:%" PRIu64 " (sample: %" PRIu64 ", inter-cluster: %" PRIu64 ")\n",
           status->framedist_calls,
           status->framedist_calls_sample,
           status->framedist_calls_intercluster);
    printf("Candidates Pruned:    %" PRIu64 "\n", status->clusters_pruned);
    printf("Missed Frames:        %" PRIu64 "\n", status->total_missed_frames);

    double avg_dists = (status->total_frames_processed > 0)
                           ? (double)status->framedist_calls / status->total_frames_processed
                           : 0.0;
    printf("Avg Dists per Sample: %.3f\n", avg_dists);

    /* Last update timestamp formatting */
    time_t sec = (time_t)(status->last_update_time / 1000000000ULL);
    struct tm tm_info;
    localtime_r(&sec, &tm_info);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_info);
    printf("Last Update:          %s\n", time_str);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <shm_file_path> [-w|--watch]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int watch_mode = 0;
    if (argc > 2 && (strcmp(argv[2], "-w") == 0 || strcmp(argv[2], "--watch") == 0))
    {
        watch_mode = 1;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        perror("Failed to open status SHM file");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0)
    {
        perror("Failed to stat status SHM file");
        close(fd);
        return 1;
    }

    if (st.st_size < (off_t)sizeof(GricClusterShmStatus))
    {
        fprintf(stderr, "Error: File size too small to be a status SHM file.\n");
        close(fd);
        return 1;
    }

    void *ptr = mmap(NULL, sizeof(GricClusterShmStatus), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED)
    {
        perror("Failed to mmap status SHM file");
        return 1;
    }

    const GricClusterShmStatus *status = (const GricClusterShmStatus *)ptr;

    if (status->magic != GRIC_SHM_MAGIC)
    {
        fprintf(stderr, "Error: Invalid magic bytes. Not a gric status file.\n");
        munmap(ptr, sizeof(GricClusterShmStatus));
        return 1;
    }
    if (status->version != GRIC_SHM_VERSION)
    {
        fprintf(stderr, "Error: Unsupported SHM version %u (expected %u).\n",
                status->version, GRIC_SHM_VERSION);
        munmap(ptr, sizeof(GricClusterShmStatus));
        return 1;
    }

    if (!watch_mode)
    {
        print_status(status);
    }
    else
    {
        printf("\033[2J"); /* Clear screen */
        for (;;)
        {
            printf("\033[H"); /* Move cursor to top-left */
            print_status(status);

            uint32_t state = status->status_state;
            if (state == GRIC_STATUS_SUCCESS ||
                state == GRIC_STATUS_ERROR ||
                state == GRIC_STATUS_ABORTED ||
                !is_pid_alive((pid_t)status->pid))
            {
                break;
            }

            struct timespec req = {0, 250000000L}; /* 250ms */
            nanosleep(&req, NULL);
        }
    }

    munmap(ptr, sizeof(GricClusterShmStatus));
    return 0;
}

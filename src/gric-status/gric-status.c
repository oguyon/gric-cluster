/**
 * @file gric-status.c
 * @brief Main entry point and CLI option handling for the gric-status client.
 */

#define _POSIX_C_SOURCE 200809L
#include "status_internal.h"
#include "shared/cli_colors.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Allocate storage for global variables declared in status_internal.h */
int sc_rows = MAX_ROWS;
int sc_cols = MAX_COLS;
ScreenCell front_buf[MAX_ROWS][MAX_COLS];
ScreenCell shadow_buf[MAX_ROWS][MAX_COLS];
struct termios orig_termios;
int raw_active = 0;
int ov__color_level = 0;
volatile sig_atomic_t sig_quit = 0;

FpsSample fps_history[FPS_SAMPLES];
int fps_history_count = 0;
int fps_history_index = 0;

FrameStatsHistory frame_history[FRAME_HISTORY_SIZE];
int frame_history_count = 0;
int frame_history_index = 0;
uint64_t last_tracked_frame = 0;
uint64_t last_num_new_clusters = 0;

double last_cum_io   = 0.0;
double last_cum_s1   = 0.0;
double last_cum_s2   = 0.0;
double last_cum_s3a  = 0.0;
double last_cum_s3b_score = 0.0;
double last_cum_s3b_filter = 0.0;
double last_cum_s3b_eval  = 0.0;
double last_cum_s3c  = 0.0;
double last_cum_s4   = 0.0;
double last_cum_s5   = 0.0;
double last_cum_ref  = 0.0;

int main(
    int   argc,
    char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("gric-status: Monitor real-time clustering telemetry "
                   "and process status from shared memory.\n");
            return 0;
        }
        if (strcmp(argv[i], "-h2") == 0 || strcmp(argv[i], "--help-description") == 0)
        {
            printf("This program attaches to the file-mapped shared memory created by "
                   "gric-cluster\nand displays real-time telemetry metrics in either "
                   "classic text or interactive\ndashboard modes. It helps track "
                   "progress, active clusters, distance computation\ncounts, pruning "
                   "ratios, RSS memory consumption, and OpenMP thread activity.\n");
            return 0;
        }
    }

    if (argc < 2)
    {
        fprintf(stderr, "Error: missing required argument <shm_file_path>\n");
        fprintf(stderr, "Usage: %s <shm_file_path> [options]\n", argv[0]);
        return 1;
    }

    const char *path = NULL;
    int watch_mode = 0;
    double rate_hz = 15.0;
    int help_mono = 0;
    int show_help = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            show_help = 1;
        }
        else if (strcmp(argv[i], "-hm") == 0 || strcmp(argv[i], "--help-mono") == 0)
        {
            help_mono = 1;
            show_help = 1;
        }
        else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--watch") == 0)
        {
            watch_mode = 1;
        }
        else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rate") == 0)
        {
            if (i + 1 < argc)
            {
                rate_hz = atof(argv[++i]);
                if (rate_hz <= 0.1)
                {
                    rate_hz = 0.1;
                }
                else if (rate_hz > 100.0)
                {
                    rate_hz = 100.0;
                }
            }
        }
        else
        {
            path = argv[i];
        }
    }

    if (show_help)
    {
        int use_color = !help_mono && isatty(STDOUT_FILENO);
        print_help_standard(argv[0], use_color);
        return 0;
    }

    if (path == NULL)
    {
        fprintf(stderr, "Error: No status SHM file path specified.\n");
        print_help_standard(argv[0], isatty(STDOUT_FILENO));
        return 1;
    }

    ov_detect_color_level(help_mono);

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

    if (!watch_mode)
    {
        print_status_classic(status);
        munmap(ptr, sizeof(GricClusterShmStatus));
        return 0;
    }

    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))
    {
        fprintf(stderr, "Error: watch mode requires an interactive terminal.\n");
        munmap(ptr, sizeof(GricClusterShmStatus));
        return 1;
    }

    int ret = run_status_watch(status, rate_hz);
    munmap(ptr, sizeof(GricClusterShmStatus));
    return ret;
} // main

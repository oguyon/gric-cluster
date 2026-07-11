/**
 * @file gric-status.c
 * @brief btop-style Terminal User Interface (TUI) client for gric-cluster.
 */

#define _POSIX_C_SOURCE 200809L
#include "shared/cli_colors.h"
#include "status_internal.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
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

/**
 * print_status_classic() - Print a simple one-shot snapshot of telemetry.
 * @status: Pointer to the read-only mapped GricClusterShmStatus struct.
 */
static void print_status_classic(
    const GricClusterShmStatus *status)
{
    int use_color = (ov__color_level > 0 && isatty(STDOUT_FILENO));
    printf("\n%s--- gric-cluster telemetry status ---%s\n",
           use_color ? ANSI_BOLD : "", use_color ? ANSI_COLOR_RESET : "");
    printf("PID:                  %u\n", status->pid);
    printf("CWD:                  %s\n", status->config_cwd);
    printf("State:                %s\n",
           get_state_string(status->status_state, (pid_t)status->pid));
    printf("Input Source:         %s\n", status->input_source);
    printf("Samples Processed:    %" PRIu64 " / %" PRIu64 "\n",
           status->total_frames_processed, status->total_frames);
    printf("Active Clusters:      %u\n", status->num_clusters);
    printf("Elapsed Time:         %.2f ms (%.3f s)\n",
           status->elapsed_ms, status->elapsed_ms / 1000.0);
    printf("Distance Computations:%" PRIu64 " (sample: %" PRIu64
           ", inter-cluster: %" PRIu64 ")\n",
           status->framedist_calls,
           status->framedist_calls_sample,
           status->framedist_calls_intercluster);
    printf("Candidates Pruned:    %" PRIu64 "\n", status->clusters_pruned);
    printf("Missed Frames:        %" PRIu64 "\n", status->total_missed_frames);

    double avg_dists = (status->total_frames_processed > 0)
                           ? (double)status->framedist_calls / status->total_frames_processed
                           : 0.0;
    printf("Avg Dists per Sample: %.3f\n", avg_dists);

    /* Version 3 metrics */
    printf("Config parameters:    rlim=%.4f, maxcl=%u, dprob=%.4f, strategy=%u\n",
           status->config_rlim, status->config_maxnbclust, status->config_dprob,
           status->config_maxcl_strategy);
    printf("Optimizations:        te4=%u, te5=%u, gprob=%u, sparse=%u, entropy=%u\n",
           status->config_te4_mode, status->config_te5_mode, status->config_gprob_mode,
           status->config_sparse_dcc, status->config_entropy_mode);
    printf("Last Frame Dists:     %" PRIu64 "\n", status->last_frame_dists);
    printf("Step-by-step Timers:  IO=%.2f ms, S1=%.2f ms, S2=%.2f ms, S3a=%.2f ms, S3b=%.2f ms,\n"
           "                      S3c=%.2f ms, S4=%.2f ms, S5=%.2f ms, Ref=%.2f ms\n",
           status->time_io_ms, status->time_step_1, status->time_step_2, status->time_step_3a,
           status->time_step_3b, status->time_step_3c, status->time_step_4, status->time_step_5,
           status->time_step_refine);

    time_t sec = (time_t)(status->last_update_time / 1000000000ULL);
    struct tm tm_info;
    localtime_r(&sec, &tm_info);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_info);
    printf("Last Update:          %s\n", time_str);
}

/**
 * print_help_standard() - Output help information with color support.
 * @progname: Name of the executable.
 * @color:    Flag indicating whether color output should be enabled.
 */
static void print_help_standard_raw(
    const char *progname,
    int         color)
{
#define C_STR(code, txt) (color ? (code txt MH_RST) : (txt))

    /* Build a colored progname string since C_STR requires string literals */
    char pn_colored[512];
    if (color)
    {
        snprintf(pn_colored, sizeof(pn_colored), MH_CMD "%s" MH_RST, progname);
    }
    else
    {
        snprintf(pn_colored, sizeof(pn_colored), "%s", progname);
    }

    printf("\n%s\n", C_STR(MH_TITLE, "NAME"));
    printf("  %s - Monitor shared-memory telemetry from %s\n",
           C_STR(MH_CMD, "gric-status"),
           C_STR(MH_CMD, "gric-cluster"));
    printf("\n%s\n", C_STR(MH_HDR, "USAGE"));
    printf("  %s %s [%s]\n",
           pn_colored,
           C_STR(MH_ARG, "<shm_file_path>"),
           C_STR(MH_OPT, "[options]"));

    printf("\n%s\n", C_STR(MH_HDR, "DESCRIPTION"));
    printf("  Connects to a file-mapped shared memory telemetry file produced by a running\n");
    printf("  gric-cluster process. Reads and reports real-time metrics, including frame\n");
    printf("  counts, active/spawned clusters, distance computations, pruning statistics,\n");
    printf("  and process resource limits.\n");

    printf("\n%s\n", C_STR(MH_HDR, "OPTIONS"));
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-w, --watch"),
           "Enter real-time interactive terminal monitoring dashboard mode");
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-r, --rate <hz>"),
           "Set TUI update rate frequency in Hz (default: 15)");
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-h, --help"),
           "Show this full help message with color (if supported)");
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-hm, --help-mono"),
           "Show full help message forced to monochrome");
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-h1, --help-oneline"),
           "Show a brief one-line description and exit");
    printf("  %-30s %s\n",
           C_STR(MH_OPT, "-h2, --help-description"),
           "Show a verbose plain-text description and exit");

    printf("\n%s\n", C_STR(MH_HDR, "INTERACTIVE CONTROLS"));
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[space]"), "Pause / Resume dashboard telemetry refresh");
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[+], [=]"), "Increase refresh rate by 1 Hz");
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[-]"), "Decrease refresh rate by 1 Hz");
    printf("  %-30s %s\n",
           C_STR(MH_BOLD, "[q], [Esc]"),
           "Cleanly quit the utility and restore terminal settings");

    printf("\n%s\n", C_STR(MH_HDR, "EXAMPLES"));
    printf("  %s %s\n", pn_colored, C_STR(MH_ARG, "/tmp/gric_status.shm"));
    printf("    Print a one-shot telemetry snapshot to stdout.\n");
    printf("  %s %s %s\n", pn_colored, C_STR(MH_ARG, "/tmp/gric_status.shm"), C_STR(MH_OPT, "-w"));
    printf("    Launch interactive TUI dashboard at the default 15 Hz refresh rate.\n");
    printf("  %s %s %s %s\n",
           pn_colored,
           C_STR(MH_ARG, "/tmp/gric_status.shm"),
           C_STR(MH_OPT, "-w -r"),
           C_STR(MH_ARG, "30"));
    printf("    Launch interactive TUI dashboard at 30 Hz refresh rate.\n");

    printf("\n%s\n", C_STR(MH_HDR, "COLOR MODE"));
    if (color)
    {
        printf("  Color output is %s.\n", C_STR(MH_CMD, "enabled"));
    }
    else
    {
        printf("  Color output is disabled (monochrome mode or NO_COLOR set).\n");
    }
    printf("  Set the %s environment variable to disable all ANSI color output.\n",
           C_STR(MH_OPT, "NO_COLOR"));
    printf("\n");
}

static void print_help_standard(
    const char *progname,
    int         color)
{
    FILE *tmp = tmpfile();
    if (tmp != NULL)
    {
        int saved_stdout = dup(STDOUT_FILENO);
        int tmp_fd = fileno(tmp);
        dup2(tmp_fd, STDOUT_FILENO);

        print_help_standard_raw(progname, color);
        fflush(stdout);

        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);

        fseek(tmp, 0, SEEK_END);
        long sz = ftell(tmp);
        fseek(tmp, 0, SEEK_SET);

        char *buf = malloc((size_t)sz + 1);
        if (buf != NULL)
        {
            size_t read_bytes = fread(buf, 1, (size_t)sz, tmp);
            buf[read_bytes] = '\0';
            cli_print_pager(buf);
            free(buf);
        }
        fclose(tmp);
    }
    else
    {
        print_help_standard_raw(progname, color);
    }
}

/**
 * main() - Application entry point.
 * @argc: Number of CLI arguments.
 * @argv: Array of CLI argument strings.
 *
 * Return: 0 on success, 1 on error.
 */
int main(
    int   argc,
    char *argv[])
{
    /* 1. Handle -h1 or --help-oneline first before anything else */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("gric-status: Monitor real-time clustering telemetry "
                   "and process status from shared memory.\n");
            return 0;
        }
    }

    /* 2. Handle -h2 or --help-description next */
    for (int i = 1; i < argc; i++)
    {
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

    /* Watch TUI Mode: require an interactive terminal */
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))
    {
        fprintf(stderr, "Error: watch mode requires an interactive terminal.\n");
        munmap(ptr, sizeof(GricClusterShmStatus));
        return 1;
    }

    /* Watch TUI Mode */
    setup_signals();
    tui_enter_raw_mode();

    /* Clear and Hide Cursor */
    printf("\033[?1049h\033[?25l\033[2J");
    fflush(stdout);

    /* Initialize front buffer */
    for (int r = 0; r < sc_rows; r++)
    {
        for (int c = 0; c < sc_cols; c++)
        {
            front_buf[r][c].utf8[0] = '\0';
            front_buf[r][c].fg      = -2;
            front_buf[r][c].bg      = -2;
            front_buf[r][c].bold    = -1;
        }
    }

    double   inst_fps = 0.0;
    double   prev_calc_time = 0.0;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    prev_calc_time = ts.tv_sec + ts.tv_nsec / 1e9;

    int is_paused = 0;

    while (!sig_quit)
    {
        /* Check keyboard input (non-blocking) */
        char ch;
        int nread = read(STDIN_FILENO, &ch, 1);
        if (nread > 0)
        {
            if (ch == 'q' || ch == 27) /* Esc or 'q' */
            {
                break;
            }
            else if (ch == ' ')
            {
                is_paused = !is_paused;
            }
            else if (ch == '+' || ch == '=')
            {
                rate_hz += 1.0;
                if (rate_hz > 100.0)
                {
                    rate_hz = 100.0;
                }
            }
            else if (ch == '-')
            {
                rate_hz -= 1.0;
                if (rate_hz < 1.0)
                {
                    rate_hz = 1.0;
                }
            }
        }

        /* Resize detection */
        struct winsize w;
        if (ioctl(STDIN_FILENO, TIOCGWINSZ, &w) >= 0)
        {
            if (w.ws_row > 0 && w.ws_col > 0)
            {
                int new_rows = w.ws_row < MAX_ROWS ? w.ws_row : MAX_ROWS;
                int new_cols = w.ws_col < MAX_COLS ? w.ws_col : MAX_COLS;
                if (new_rows != sc_rows || new_cols != sc_cols)
                {
                    sc_rows = new_rows;
                    sc_cols = new_cols;
                    printf("\033[2J");
                    fflush(stdout);
                    /* Force re-draw by clearing front buffer */
                    for (int r = 0; r < sc_rows; r++)
                    {
                        for (int c = 0; c < sc_cols; c++)
                        {
                            front_buf[r][c].utf8[0] = '\0';
                            front_buf[r][c].fg      = -2;
                            front_buf[r][c].bg      = -2;
                            front_buf[r][c].bold    = -1;
                        }
                    }
                }
            }
        }

        if (!is_paused)
        {
            static double          last_cluster_ticks = 0.0;
            static double          last_status_ticks  = 0.0;
            static struct timespec last_cpu_time      = {0, 0};
            static double          cluster_cpu        = 0.0;
            static double          status_cpu         = 0.0;

            struct timespec now_cpu;
            clock_gettime(CLOCK_MONOTONIC, &now_cpu);
            double elapsed_sec = 0.1;
            if (last_cpu_time.tv_sec > 0)
            {
                elapsed_sec = (now_cpu.tv_sec - last_cpu_time.tv_sec) +
                              (now_cpu.tv_nsec - last_cpu_time.tv_nsec) / 1000000000.0;
            }
            if (elapsed_sec < 0.01)
            {
                elapsed_sec = 0.01;
            }
            long clk_tck = sysconf(_SC_CLK_TCK);

            /* For gric-cluster */
            pid_t p_pid = (pid_t)status->pid;
            if (is_pid_alive(p_pid))
            {
                unsigned long utime = 0, stime = 0;
                if (get_process_cpu_ticks(p_pid, &utime, &stime) == 0)
                {
                    double ticks = (double)(utime + stime);
                    if (last_cluster_ticks > 0.0 && ticks >= last_cluster_ticks)
                    {
                        cluster_cpu = 100.0 * ((ticks - last_cluster_ticks) / clk_tck)
                                      / elapsed_sec;
                    }
                    last_cluster_ticks = ticks;
                }
            }
            else
            {
                cluster_cpu = 0.0;
            }

            /* For gric-status (self) */
            {
                unsigned long utime = 0, stime = 0;
                if (get_process_cpu_ticks(getpid(), &utime, &stime) == 0)
                {
                    double ticks = (double)(utime + stime);
                    if (last_status_ticks > 0.0 && ticks >= last_status_ticks)
                    {
                        status_cpu = 100.0 * ((ticks - last_status_ticks) / clk_tck) / elapsed_sec;
                    }
                    last_status_ticks = ticks;
                }
            }
            last_cpu_time = now_cpu;

            /* Calculate rolling FPS based on the last 10 samples */
            struct timespec current_ts;
            clock_gettime(CLOCK_MONOTONIC, &current_ts);
            double current_calc_time = current_ts.tv_sec + current_ts.tv_nsec / 1e9;
            double time_delta = current_calc_time - prev_calc_time;

            if (time_delta >= 0.1) /* Store a sample every 100ms */
            {
                uint64_t current_processed = status->total_frames_processed;

                fps_history[fps_history_index].time = current_calc_time;
                fps_history[fps_history_index].processed = current_processed;

                fps_history_index = (fps_history_index + 1) % FPS_SAMPLES;
                if (fps_history_count < FPS_SAMPLES)
                {
                    fps_history_count++;
                }

                if (fps_history_count >= 2)
                {
                    int oldest_idx = (fps_history_count == FPS_SAMPLES) ? fps_history_index : 0;
                    int latest_idx = (fps_history_index - 1 + FPS_SAMPLES) % FPS_SAMPLES;

                    double dt = fps_history[latest_idx].time - fps_history[oldest_idx].time;
                    uint64_t dp = fps_history[latest_idx].processed -
                                  fps_history[oldest_idx].processed;

                    if (dt > 0.001)
                    {
                        inst_fps = (double)dp / dt;
                    }
                    else
                    {
                        inst_fps = 0.0;
                    }
                }
                else
                {
                    inst_fps = 0.0;
                }

                prev_calc_time = current_calc_time;
            }

            tui_clear();

            int w_left = sc_cols / 2;
            if (w_left < 40)
            {
                w_left = 40;
            }
            int w_right = sc_cols - w_left;
            if (w_right < 40)
            {
                w_right = 40;
            }

            int h_top = sc_rows / 2;
            if (h_top < 10)
            {
                h_top = 10;
            }
            int h_bottom = sc_rows - h_top - 1;
            if (h_bottom < 11)
            {
                h_bottom = 11;
            }

            uint64_t processed = status->total_frames_processed;
            if (processed != last_tracked_frame)
            {
                frame_history[frame_history_index].solve_dfc = status->last_frame_dfc;
                frame_history[frame_history_index].solve_dcc = status->last_frame_dcc;
                frame_history[frame_history_index].assign_dist = status->last_assignment_dist;
                frame_history[frame_history_index].created_cluster =
                    (status->num_new_clusters > last_num_new_clusters) ? 1 : 0;

                /* Individual frame timings calculation */
                double current_cum[11] = {
                    status->time_io_ms, status->time_step_1, status->time_step_2,
                    status->time_step_3a, status->time_step_3b_score, status->time_step_3b_filter,
                    status->time_step_3b_eval, status->time_step_3c, status->time_step_4,
                    status->time_step_5, status->time_step_refine
                };
                double last_cum[11] = {
                    last_cum_io, last_cum_s1, last_cum_s2, last_cum_s3a,
                    last_cum_s3b_score, last_cum_s3b_filter, last_cum_s3b_eval,
                    last_cum_s3c, last_cum_s4, last_cum_s5, last_cum_ref
                };

                for (int step = 0; step < 11; step++)
                {
                    double diff = current_cum[step] - last_cum[step];
                    if (diff < 0.0 || last_tracked_frame == 0)
                    {
                        diff = (last_tracked_frame == 0) ? current_cum[step] : 0.0;
                    }
                    frame_history[frame_history_index].step_times[step] = diff;
                }

                last_cum_io   = status->time_io_ms;
                last_cum_s1   = status->time_step_1;
                last_cum_s2   = status->time_step_2;
                last_cum_s3a  = status->time_step_3a;
                last_cum_s3b_score = status->time_step_3b_score;
                last_cum_s3b_filter = status->time_step_3b_filter;
                last_cum_s3b_eval  = status->time_step_3b_eval;
                last_cum_s3c  = status->time_step_3c;
                last_cum_s4   = status->time_step_4;
                last_cum_s5   = status->time_step_5;
                last_cum_ref  = status->time_step_refine;

                frame_history_index = (frame_history_index + 1) % FRAME_HISTORY_SIZE;
                if (frame_history_count < FRAME_HISTORY_SIZE)
                {
                    frame_history_count++;
                }
                last_tracked_frame = processed;
                last_num_new_clusters = status->num_new_clusters;
            }

            /* PANEL 1: INFO (Top Left) */
            tui_draw_box(0, 0, w_left, h_top, "INFO", COLOR_CYAN, COLOR_DEFAULT, 1);
            char buf[512];
            snprintf(buf, sizeof(buf), "PID:    %u", status->pid);
            tui_draw_string(2, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "CWD:    %.*s", w_left - 10, status->config_cwd);
            tui_draw_string(3, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "Input:  %.*s", w_left - 10, status->input_source);
            tui_draw_string(4, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "RLim:   %.4f", status->config_rlim);
            tui_draw_string(6, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "MaxCl:  %u", status->config_maxnbclust);
            tui_draw_string(7, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "DProb:  %.4f", status->config_dprob);
            tui_draw_string(8, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            const char *strat = "STOP";
            if (status->config_maxcl_strategy == 1)
            {
                strat = "DISCARD";
            }
            else if (status->config_maxcl_strategy == 2)
            {
                strat = "MERGE";
            }
            snprintf(buf, sizeof(buf), "Strat:  %s", strat);
            tui_draw_string(9, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            /* Row 10: TE4, TE5, and Entropy buttons */
            tui_draw_label_button(10, 2, "TE4", status->config_te4_mode);
            tui_draw_label_button(10, 11, "TE5", status->config_te5_mode);
            tui_draw_label_button(10, 20, "Entropy", status->config_entropy_mode);

            /* Row 11: Sparse DCC and Geom Prob buttons */
            tui_draw_label_button(11, 2, "Sparse DCC", status->config_sparse_dcc);
            tui_draw_label_button(11, 17, "Geom Prob", status->config_gprob_mode);

            /* PANEL 2: FRAME STATS (Top Right) */
            tui_draw_box(0, w_left, w_right, h_top, "FRAME STATS", COLOR_CYAN, COLOR_DEFAULT, 1);

            uint64_t total = status->total_frames;
            snprintf(buf, sizeof(buf), "Frame Index:      %" PRIu64 " / %" PRIu64, processed, total);
            tui_draw_string(2, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            /* Solve DFC rolling history list (newest on left, oldest on right) */
            tui_draw_string(4, w_left + 2, "Solve DFC:        ", COLOR_WHITE, COLOR_DEFAULT, 0);
            for (int i = 0; i < frame_history_count; i++)
            {
                int idx = (frame_history_index - 1 - i + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
                int col = w_left + 20 + i * 6;
                if (col + 5 >= w_left + w_right - 1)
                {
                    break;
                }
                char val[8];
                format_5char_int(val, frame_history[idx].solve_dfc);
                int color = frame_history[idx].created_cluster ? COLOR_GREEN : COLOR_WHITE;
                tui_draw_string(4, col, val, color, COLOR_DEFAULT,
                                frame_history[idx].created_cluster);
            }

            /* Solve DCC rolling history list (newest on left, oldest on right) */
            tui_draw_string(5, w_left + 2, "Solve DCC:        ", COLOR_WHITE, COLOR_DEFAULT, 0);
            for (int i = 0; i < frame_history_count; i++)
            {
                int idx = (frame_history_index - 1 - i + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
                int col = w_left + 20 + i * 6;
                if (col + 5 >= w_left + w_right - 1)
                {
                    break;
                }
                char val[8];
                format_5char_int(val, frame_history[idx].solve_dcc);
                int color = frame_history[idx].created_cluster ? COLOR_GREEN : COLOR_WHITE;
                tui_draw_string(5, col, val, color, COLOR_DEFAULT,
                                frame_history[idx].created_cluster);
            }

            /* Assign Dists rolling history list (newest on left, oldest on right) */
            tui_draw_string(6, w_left + 2, "Assign Dists:     ", COLOR_WHITE, COLOR_DEFAULT, 0);
            for (int i = 0; i < frame_history_count; i++)
            {
                int idx = (frame_history_index - 1 - i + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
                int col = w_left + 20 + i * 6;
                if (col + 5 >= w_left + w_right - 1)
                {
                    break;
                }
                char val[8];
                format_5char_float(val, frame_history[idx].assign_dist);
                int color = frame_history[idx].created_cluster ? COLOR_GREEN : COLOR_WHITE;
                tui_draw_string(6, col, val, color, COLOR_DEFAULT,
                                frame_history[idx].created_cluster);
            }

            pid_t pid = (pid_t)status->pid;
            const char *state_str = get_state_string(status->status_state, pid);
            int state_color = get_state_color(status->status_state, pid);
            tui_draw_string(8, w_left + 2, "State:            ", COLOR_WHITE, COLOR_DEFAULT, 0);
            tui_draw_string(8, w_left + 20, state_str, state_color, COLOR_DEFAULT, 1);

            /* PANEL 3: CLUSTER STATS (Bottom Left) */
            tui_draw_box(h_top, 0, w_left, h_bottom, "CLUSTER STATS", COLOR_CYAN, COLOR_DEFAULT, 1);

            snprintf(buf, sizeof(buf), "Active Clusters:   %u", status->num_clusters);
            tui_draw_string(h_top + 2, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "Spawned Clusters:  %" PRIu64, status->num_new_clusters);
            tui_draw_string(h_top + 3, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            double avg_dists = (processed > 0)
                               ? (double)status->framedist_calls / processed
                               : 0.0;
            snprintf(buf, sizeof(buf), "Avg Dists/Sample:  %.2f", avg_dists);
            tui_draw_string(h_top + 4, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            uint64_t total_evals = status->framedist_calls;
            uint64_t total_possible = processed * status->num_clusters;
            double prune_ratio = 0.0;
            if (total_possible > 0)
            {
                prune_ratio = 100.0 * (1.0 - (double)total_evals / (double)total_possible);
                if (prune_ratio < 0.0)
                {
                    prune_ratio = 0.0;
                }
            }
            snprintf(buf, sizeof(buf), "Pruning Ratio:     %.2f%%", prune_ratio);
            tui_draw_string(h_top + 5, 2, buf, COLOR_GREEN, COLOR_DEFAULT, 1);

            snprintf(buf, sizeof(buf), "Pruned Telemetry:  %" PRIu64, status->clusters_pruned);
            tui_draw_string(h_top + 6, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "Total Solve Dists: %" PRIu64, status->framedist_calls);
            tui_draw_string(h_top + 7, 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            /* Progress Bar displaying current sample index (processed / total) */
            double pct = 0.0;
            if (total > 0)
            {
                pct = 100.0 * (double)processed / (double)total;
            }
            if (pct > 100.0)
            {
                pct = 100.0;
            }

            char prog_lbl[128];
            snprintf(prog_lbl, sizeof(prog_lbl), "Progress (%" PRIu64 "/%" PRIu64 "): ",
                     processed, total);
            tui_draw_string(h_top + 8, 2, prog_lbl, COLOR_WHITE, COLOR_DEFAULT, 0);

            int start_col = 2 + strlen(prog_lbl);
            tui_draw_string(h_top + 8, start_col, "[", COLOR_WHITE, COLOR_DEFAULT, 0);

            int bar_width = w_left - start_col - 9;
            if (bar_width < 10)
            {
                bar_width = 10;
            }

            int filled = (int)(pct * bar_width / 100.0);
            for (int b = 0; b < bar_width; b++)
            {
                if (b < filled)
                {
                    tui_draw_string(h_top + 8, start_col + 1 + b, "█", COLOR_GREEN, COLOR_DEFAULT,
                                    0);
                }
                else
                {
                    tui_draw_string(h_top + 8, start_col + 1 + b, "░", COLOR_GREY, COLOR_DEFAULT,
                                    0);
                }
            }
            snprintf(buf, sizeof(buf), "] %.1f%%", pct);
            tui_draw_string(h_top + 8, start_col + 1 + bar_width, buf, COLOR_WHITE, COLOR_DEFAULT,
                            0);

            /* FPS Counter */
            double cum_fps = 0.0;
            if (status->elapsed_ms > 0.0)
            {
                cum_fps = (double)processed / (status->elapsed_ms / 1000.0);
            }
            snprintf(buf, sizeof(buf), "Speed: %.1f FPS (roll) | %.1f FPS (avg)",
                     inst_fps, cum_fps);
            tui_draw_string(h_top + 9, 2, buf, COLOR_ORANGE, COLOR_DEFAULT, 0);

            /* PANEL 4: RESOURCES (Bottom Right) */
            tui_draw_box(h_top, w_left, w_right, h_bottom, "RESOURCES", COLOR_CYAN, COLOR_DEFAULT,
                         1);

            snprintf(buf, sizeof(buf), "CPU Threads:  %u OpenMP threads", status->active_threads);
            tui_draw_string(h_top + 2, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "CPU:          %5.1f%% | %5.1f%% (self)",
                     cluster_cpu, status_cpu);
            tui_draw_string(h_top + 3, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            double rss_mb = (double)status->memory_rss_kb / 1024.0;
            snprintf(buf, sizeof(buf), "Memory RSS:   %.2f MB", rss_mb);
            tui_draw_string(h_top + 4, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);
            snprintf(buf, sizeof(buf), "Stream Lag:   %ld frames", status->stream_lag);
            tui_draw_string(h_top + 5, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            snprintf(buf, sizeof(buf), "Wait Time:    %.3f s", status->stream_wait_time_sec);
            tui_draw_string(h_top + 6, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

            /* High resolution performance timers by algorithm step */
            double total_prof = status->time_io_ms + status->time_step_1 + status->time_step_2 +
                                status->time_step_3a + status->time_step_3b_score +
                                status->time_step_3b_filter + status->time_step_3b_eval +
                                status->time_step_3c + status->time_step_4 +
                                status->time_step_5 + status->time_step_refine;
            double p_io = 0.0, p_s1 = 0.0, p_s2 = 0.0, p_s3a = 0.0, p_s3b_score = 0.0;
            double p_s3b_filter = 0.0, p_s3b_eval = 0.0, p_s3c = 0.0, p_s4 = 0.0;
            double p_s5 = 0.0, p_ref = 0.0;

            if (total_prof > 0.0)
            {
                p_io  = 100.0 * status->time_io_ms / total_prof;
                p_s1  = 100.0 * status->time_step_1 / total_prof;
                p_s2  = 100.0 * status->time_step_2 / total_prof;
                p_s3a = 100.0 * status->time_step_3a / total_prof;
                p_s3b_score = 100.0 * status->time_step_3b_score / total_prof;
                p_s3b_filter = 100.0 * status->time_step_3b_filter / total_prof;
                p_s3b_eval  = 100.0 * status->time_step_3b_eval / total_prof;
                p_s3c = 100.0 * status->time_step_3c / total_prof;
                p_s4  = 100.0 * status->time_step_4 / total_prof;
                p_s5  = 100.0 * status->time_step_5 / total_prof;
                p_ref = 100.0 * status->time_step_refine / total_prof;
            }

            tui_draw_string(h_top + 7, w_left + 2, "Profile (Total / Rolling 100):",
                            COLOR_ORANGE, COLOR_DEFAULT, 1);

            const char *descriptions[] = {
                "Retrieve prediction candidates",
                "Prior mixing & candidate pruning",
                "Target score prep & sorting",
                "Candidate filtering & selection",
                "Expected entropy evaluation",
                "Distance calculations",
                "New cluster spawn / eviction",
                "Telemetry log writeout",
                "DCC bounds refinement"
            };
            int mapped_indices[] = {
                2, 3, 4, 5, 6, 7, 8, 9, 10
            };
            double percentages[11] = {
                p_io, p_s1, p_s2, p_s3a, p_s3b_score, p_s3b_filter,
                p_s3b_eval, p_s3c, p_s4, p_s5, p_ref
            };

            /* Calculate rolling average of the last 100 frames */
            double rolling_sums[11] = {0.0};
            double total_rolling = 0.0;
            for (int i = 0; i < frame_history_count; i++)
            {
                int idx = (frame_history_index - 1 - i + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
                for (int step = 0; step < 11; step++)
                {
                    rolling_sums[step] += frame_history[idx].step_times[step];
                    total_rolling += frame_history[idx].step_times[step];
                }
            }

            double rolling_p[11] = {0.0};
            if (total_rolling > 0.0)
            {
                for (int step = 0; step < 11; step++)
                {
                    rolling_p[step] = 100.0 * rolling_sums[step] / total_rolling;
                }
            }
            else
            {
                for (int step = 0; step < 11; step++)
                {
                    rolling_p[step] = percentages[step];
                }
            }

            for (int i = 0; i < 9; i++)
            {
                int step = mapped_indices[i];
                int row = h_top + 8 + i;
                if (row >= h_top + h_bottom)
                {
                    break;
                }
                int max_desc_len = w_right - 22;
                if (max_desc_len < 5)
                {
                    max_desc_len = 5;
                }
                snprintf(buf, sizeof(buf), " %5.1f%% / %5.1f%% | %.*s",
                         percentages[step], rolling_p[step],
                         max_desc_len, descriptions[i]);
                tui_draw_string(row, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);
            }

            /* Help info line at the very bottom */
            char help_buf[256];
            snprintf(help_buf, sizeof(help_buf),
                     "[q] Quit TUI    [space] Pause/Resume    [+]/[-] Rate: %.0f Hz",
                     rate_hz);
            tui_draw_string(sc_rows - 1, 2, help_buf, COLOR_GREY, COLOR_DEFAULT, 0);

            tui_flush();
        }

        double interval_sec = 1.0 / rate_hz;
        long nsec = (long)(interval_sec * 1000000000.0);
        struct timespec req = {nsec / 1000000000L, nsec % 1000000000L};
        nanosleep(&req, NULL);
    }

    tui_exit_raw_mode();
    printf("\033[?1049l\033[?25h\033[0m\n"); /* Restore screen and show cursor */
    fflush(stdout);

    munmap(ptr, sizeof(GricClusterShmStatus));
    return 0;
}

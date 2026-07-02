/**
 * @file gric-status.c
 * @brief btop-style Terminal User Interface (TUI) client for gric-cluster.
 */

#define _POSIX_C_SOURCE 200809L
#include "gric-cluster/core/cluster_shm.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
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

#define MAX_ROWS 60
#define MAX_COLS 180

#define ANSI_BOLD        "\033[1m"
#define ANSI_COLOR_RESET "\033[0m"

#define MH_RST   "\033[0m"
#define MH_BOLD  "\033[1m"
#define MH_DIM   "\033[2m"
#define MH_TITLE "\033[1;36m"
#define MH_HDR   "\033[1;34m"
#define MH_CMD   "\033[1;32m"
#define MH_OPT   "\033[33m"
#define MH_ARG   "\033[1;35m"
#define MH_DFLT  "\033[36m"
#define MH_NOTE  "\033[1;33m"
#define MH_ERR   "\033[1;31m"

#define FPS_SAMPLES 10

typedef struct
{
    double   time;
    uint64_t processed;
} FpsSample;

static FpsSample fps_history[FPS_SAMPLES];
static int       fps_history_count = 0;
static int       fps_history_index = 0;

#define FRAME_HISTORY_SIZE 100

typedef struct
{
    uint64_t solve_dfc;
    uint64_t solve_dcc;
    double   assign_dist;
    int      created_cluster;
    double   step_times[11];
} FrameStatsHistory;

static FrameStatsHistory frame_history[FRAME_HISTORY_SIZE];
static int               frame_history_count = 0;
static int               frame_history_index = 0;
static uint64_t          last_tracked_frame  = 0;
static uint64_t          last_num_new_clusters = 0;

static double last_cum_io   = 0.0;
static double last_cum_s1   = 0.0;
static double last_cum_s2   = 0.0;
static double last_cum_s3a  = 0.0;
static double last_cum_s3b_score = 0.0;
static double last_cum_s3b_filter = 0.0;
static double last_cum_s3b_eval  = 0.0;
static double last_cum_s3c  = 0.0;
static double last_cum_s4   = 0.0;
static double last_cum_s5   = 0.0;
static double last_cum_ref  = 0.0;

enum
{
    COLOR_DEFAULT = -1,
    COLOR_BLACK   = 0,
    COLOR_RED     = 1,
    COLOR_GREEN   = 2,
    COLOR_YELLOW  = 3,
    COLOR_BLUE    = 4,
    COLOR_MAGENTA = 5,
    COLOR_CYAN    = 6,
    COLOR_WHITE   = 7,
    COLOR_GREY    = 8,
    COLOR_ORANGE  = 9
};

typedef struct
{
    char utf8[5]; /* 4-byte UTF-8 sequence + NUL terminator */
    int  fg;
    int  bg;
    int  bold;
} ScreenCell;

static ScreenCell front_buf[MAX_ROWS][MAX_COLS];
static ScreenCell shadow_buf[MAX_ROWS][MAX_COLS];
static int        sc_rows = MAX_ROWS;
static int        sc_cols = MAX_COLS;

static struct termios orig_termios;
static int            raw_active      = 0;
static int            ov__color_level = 0;

static volatile sig_atomic_t sig_quit = 0;

static void handle_sigint(int sig)
{
    (void)sig;
    sig_quit = 1;
}

static void tui_exit_raw_mode(void)
{
    if (raw_active)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_active = 0;
    }
}

static void tui_enter_raw_mode(void)
{
    if (tcgetattr(STDIN_FILENO, &orig_termios) < 0)
    {
        return;
    }
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG | IEXTEN);
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_cflag |= (CS8);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) >= 0)
    {
        raw_active = 1;
    }
}

static void crash_handler(int sig)
{
    tui_exit_raw_mode();
    printf("\033[?1049l\033[?25h\033[0m\n");
    fflush(stdout);

    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, NULL);
    raise(sig);
}

static void setup_signals(void)
{
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    struct sigaction sa_crash;
    sa_crash.sa_handler = crash_handler;
    sigemptyset(&sa_crash.sa_mask);
    sa_crash.sa_flags = 0;
    sigaction(SIGSEGV, &sa_crash, NULL);
    sigaction(SIGABRT, &sa_crash, NULL);
}

static void ov_detect_color_level(
    int help_mono)
{
    if (help_mono)
    {
        ov__color_level = 0;
        return;
    }
    if (getenv("NO_COLOR") != NULL)
    {
        ov__color_level = 0;
        return;
    }
    const char *colorterm = getenv("COLORTERM");
    if (colorterm && (strcmp(colorterm, "truecolor") == 0 || strcmp(colorterm, "24bit") == 0))
    {
        ov__color_level = 3;
        return;
    }
    const char *term = getenv("TERM");
    if (term && (strstr(term, "256color") || strstr(term, "xterm-256")))
    {
        ov__color_level = 2;
        return;
    }
    ov__color_level = 1;
}

static void tui_apply_colors(
    int fg,
    int bg,
    int bold)
{
    if (bold)
    {
        printf("\033[1m");
    }

    if (ov__color_level == 3) /* TrueColor */
    {
        int fg_r = 248, fg_g = 249, fg_b = 250;
        int bg_r = 20, bg_g = 22, bg_b = 28;

        if (fg == COLOR_BLACK)       { fg_r = 20;  fg_g = 22;  fg_b = 28;  }
        else if (fg == COLOR_RED)    { fg_r = 220; fg_g = 53;  fg_b = 69;  }
        else if (fg == COLOR_GREEN)  { fg_r = 40;  fg_g = 167; fg_b = 69;  }
        else if (fg == COLOR_YELLOW) { fg_r = 255; fg_g = 193; fg_b = 7;   }
        else if (fg == COLOR_BLUE)   { fg_r = 0;   fg_g = 123; fg_b = 255; }
        else if (fg == COLOR_MAGENTA){ fg_r = 111; fg_g = 66;  fg_b = 193; }
        else if (fg == COLOR_CYAN)   { fg_r = 0;   fg_g = 180; fg_b = 216; }
        else if (fg == COLOR_WHITE)  { fg_r = 248; fg_g = 249; fg_b = 250; }
        else if (fg == COLOR_GREY)   { fg_r = 100; fg_g = 110; fg_b = 120; }
        else if (fg == COLOR_ORANGE) { fg_r = 253; fg_g = 126; fg_b = 20;  }

        if (bg == COLOR_BLACK)       { bg_r = 20;  bg_g = 22;  bg_b = 28;  }
        else if (bg == COLOR_RED)    { bg_r = 220; bg_g = 53;  bg_b = 69;  }
        else if (bg == COLOR_GREEN)  { bg_r = 40;  bg_g = 167; bg_b = 69;  }
        else if (bg == COLOR_YELLOW) { bg_r = 255; bg_g = 193; bg_b = 7;   }
        else if (bg == COLOR_BLUE)   { bg_r = 0;   bg_g = 123; bg_b = 255; }
        else if (bg == COLOR_MAGENTA){ bg_r = 111; bg_g = 66;  bg_b = 193; }
        else if (bg == COLOR_CYAN)   { bg_r = 0;   bg_g = 180; bg_b = 216; }
        else if (bg == COLOR_WHITE)  { bg_r = 248; bg_g = 249; bg_b = 250; }
        else if (bg == COLOR_GREY)   { bg_r = 30;  bg_g = 34;  bg_b = 40;  }
        else if (bg == COLOR_ORANGE) { bg_r = 253; bg_g = 126; bg_b = 20;  }

        if (fg != COLOR_DEFAULT)
        {
            printf("\033[38;2;%d;%d;%dm", fg_r, fg_g, fg_b);
        }
        if (bg != COLOR_DEFAULT)
        {
            printf("\033[48;2;%d;%d;%dm", bg_r, bg_g, bg_b);
        }
    }
    else if (ov__color_level == 2) /* 256-color */
    {
        int fg_256 = -1, bg_256 = -1;
        if (fg == COLOR_BLACK)       fg_256 = 234;
        else if (fg == COLOR_RED)    fg_256 = 196;
        else if (fg == COLOR_GREEN)  fg_256 = 40;
        else if (fg == COLOR_YELLOW) fg_256 = 220;
        else if (fg == COLOR_BLUE)   fg_256 = 33;
        else if (fg == COLOR_MAGENTA)fg_256 = 129;
        else if (fg == COLOR_CYAN)   fg_256 = 38;
        else if (fg == COLOR_WHITE)  fg_256 = 255;
        else if (fg == COLOR_GREY)   fg_256 = 244;
        else if (fg == COLOR_ORANGE) fg_256 = 208;

        if (bg == COLOR_BLACK)       bg_256 = 234;
        else if (bg == COLOR_RED)    bg_256 = 196;
        else if (bg == COLOR_GREEN)  bg_256 = 40;
        else if (bg == COLOR_YELLOW) bg_256 = 220;
        else if (bg == COLOR_BLUE)   bg_256 = 33;
        else if (bg == COLOR_MAGENTA)bg_256 = 129;
        else if (bg == COLOR_CYAN)   bg_256 = 38;
        else if (bg == COLOR_WHITE)  bg_256 = 255;
        else if (bg == COLOR_GREY)   bg_256 = 236;
        else if (bg == COLOR_ORANGE) bg_256 = 208;

        if (fg_256 != -1)
        {
            printf("\033[38;5;%dm", fg_256);
        }
        if (bg_256 != -1)
        {
            printf("\033[48;5;%dm", bg_256);
        }
    }
    else if (ov__color_level == 1) /* 16-color */
    {
        if (fg != COLOR_DEFAULT)
        {
            int fg_16 = 37;
            if (fg == COLOR_BLACK)       fg_16 = 30;
            else if (fg == COLOR_RED)    fg_16 = 31;
            else if (fg == COLOR_GREEN)  fg_16 = 32;
            else if (fg == COLOR_YELLOW) fg_16 = 33;
            else if (fg == COLOR_BLUE)   fg_16 = 34;
            else if (fg == COLOR_MAGENTA)fg_16 = 35;
            else if (fg == COLOR_CYAN)   fg_16 = 36;
            else if (fg == COLOR_WHITE)  fg_16 = 37;
            else if (fg == COLOR_GREY)   fg_16 = 90;
            else if (fg == COLOR_ORANGE) fg_16 = 91;
            printf("\033[%dm", fg_16);
        }
        if (bg != COLOR_DEFAULT)
        {
            int bg_16 = 40;
            if (bg == COLOR_BLACK)       bg_16 = 40;
            else if (bg == COLOR_RED)    bg_16 = 41;
            else if (bg == COLOR_GREEN)  bg_16 = 42;
            else if (bg == COLOR_YELLOW) bg_16 = 43;
            else if (bg == COLOR_BLUE)   bg_16 = 44;
            else if (bg == COLOR_MAGENTA)bg_16 = 45;
            else if (bg == COLOR_CYAN)   bg_16 = 46;
            else if (bg == COLOR_WHITE)  bg_16 = 47;
            else if (bg == COLOR_GREY)   bg_16 = 100;
            else if (bg == COLOR_ORANGE) bg_16 = 101;
            printf("\033[%dm", bg_16);
        }
    }
}

static void tui_clear(void)
{
    for (int r = 0; r < sc_rows; r++)
    {
        for (int c = 0; c < sc_cols; c++)
        {
            shadow_buf[r][c].utf8[0] = ' ';
            shadow_buf[r][c].utf8[1] = '\0';
            shadow_buf[r][c].fg      = COLOR_DEFAULT;
            shadow_buf[r][c].bg      = COLOR_DEFAULT;
            shadow_buf[r][c].bold    = 0;
        }
    }
}

static void tui_draw_string(
    int         r,
    int         c,
    const char *str,
    int         fg,
    int         bg,
    int         bold)
{
    int col = c;
    const unsigned char *p = (const unsigned char *)str;
    while (*p && col < sc_cols)
    {
        if (r < 0 || r >= sc_rows)
        {
            break;
        }
        ScreenCell *cell = &shadow_buf[r][col];
        cell->fg   = fg;
        cell->bg   = bg;
        cell->bold = bold;

        int len = 1;
        if ((*p & 0x80) == 0) len = 1;
        else if ((*p & 0xE0) == 0xC0) len = 2;
        else if ((*p & 0xF0) == 0xE0) len = 3;
        else if ((*p & 0xF8) == 0xF0) len = 4;

        int i;
        for (i = 0; i < len && *p; i++)
        {
            cell->utf8[i] = *p;
            p++;
        }
        cell->utf8[i] = '\0';
        col++;
    }
}

static void tui_draw_box(
    int         r,
    int         c,
    int         w,
    int         h,
    const char *title,
    int         fg,
    int         bg,
    int         bold)
{
    tui_draw_string(r, c, "┌", fg, bg, bold);
    for (int col = 1; col < w - 1; col++)
    {
        tui_draw_string(r, c + col, "─", fg, bg, bold);
    }
    tui_draw_string(r, c + w - 1, "┐", fg, bg, bold);

    for (int row = 1; row < h - 1; row++)
    {
        tui_draw_string(r + row, c, "│", fg, bg, bold);
        tui_draw_string(r + row, c + w - 1, "│", fg, bg, bold);
    }

    tui_draw_string(r + h - 1, c, "└", fg, bg, bold);
    for (int col = 1; col < w - 1; col++)
    {
        tui_draw_string(r + h - 1, c + col, "─", fg, bg, bold);
    }
    tui_draw_string(r + h - 1, c + w - 1, "┘", fg, bg, bold);

    if (title && strlen(title) > 0)
    {
        char temp_title[128];
        snprintf(temp_title, sizeof(temp_title), " %s ", title);
        tui_draw_string(r, c + 2, temp_title, fg, bg, bold);
    }
}

static int get_process_cpu_ticks(
    pid_t          pid,
    unsigned long *utime,
    unsigned long *stime)
{
    char path[128];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        return -1;
    }

    char buf[1024];
    if (fgets(buf, sizeof(buf), f) == NULL)
    {
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Find last closing parenthesis */
    char *p = strrchr(buf, ')');
    if (p == NULL || *(p + 1) == '\0')
    {
        return -1;
    }
    p += 2; /* Skip ') ' */

    char          state;
    int           ppid, pgrp, session, tty_nr, tpgid;
    unsigned int  flags;
    unsigned long minflt, cminflt, majflt, cmajflt;

    if (sscanf(p, "%c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu",
               &state, &ppid, &pgrp, &session, &tty_nr, &tpgid, &flags,
               &minflt, &cminflt, &majflt, &cmajflt, utime, stime) != 13)
    {
        return -1;
    }

    return 0;
}

static void tui_draw_label_button(
    int         r,
    int         c,
    const char *label,
    int         is_on)
{
    char buf[64];
    snprintf(buf, sizeof(buf), " %s ", label);
    tui_draw_string(r, c, "[", COLOR_WHITE, COLOR_DEFAULT, 0);
    if (is_on)
    {
        tui_draw_string(r, c + 1, buf, COLOR_BLACK, COLOR_GREEN, 1);
    }
    else
    {
        tui_draw_string(r, c + 1, buf, COLOR_WHITE, COLOR_GREY, 1);
    }
    tui_draw_string(r, c + 1 + strlen(buf), "]", COLOR_WHITE, COLOR_DEFAULT, 0);
}

static void format_5char_float(
    char   *out,
    double  val)
{
    char temp[64];
    if (val >= 1000.0)
    {
        snprintf(temp, sizeof(temp), "%5.0f", val);
    }
    else if (val >= 100.0)
    {
        snprintf(temp, sizeof(temp), "%5.1f", val);
    }
    else if (val >= 10.0)
    {
        snprintf(temp, sizeof(temp), "%5.2f", val);
    }
    else
    {
        snprintf(temp, sizeof(temp), "%5.3f", val);
    }
    temp[5] = '\0';
    int len = strlen(temp);
    if (len < 5)
    {
        snprintf(out, 6, "%-5s", temp);
    }
    else
    {
        strcpy(out, temp);
    }
}

static void format_5char_int(
    char     *out,
    uint64_t  val)
{
    char temp[64];
    snprintf(temp, sizeof(temp), "%5lu", (unsigned long)val);
    temp[5] = '\0';
    int len = strlen(temp);
    if (len < 5)
    {
        snprintf(out, 6, "%-5s", temp);
    }
    else
    {
        strcpy(out, temp);
    }
}

static void tui_flush(void)
{
    printf("\033[?2026h"); /* Start synchronized update */

    int last_fg   = -2;
    int last_bg   = -2;
    int last_bold = -1;

    for (int r = 0; r < sc_rows; r++)
    {
        for (int c = 0; c < sc_cols; c++)
        {
            ScreenCell shadow = shadow_buf[r][c];
            ScreenCell front  = front_buf[r][c];

            if (strcmp(shadow.utf8, front.utf8) != 0 || shadow.fg != front.fg ||
                shadow.bg != front.bg || shadow.bold != front.bold)
            {
                printf("\033[%d;%dH", r + 1, c + 1);

                if (shadow.fg != last_fg || shadow.bg != last_bg || shadow.bold != last_bold)
                {
                    printf("\033[0m");
                    tui_apply_colors(shadow.fg, shadow.bg, shadow.bold);
                    last_fg   = shadow.fg;
                    last_bg   = shadow.bg;
                    last_bold = shadow.bold;
                }

                printf("%s", shadow.utf8);
                front_buf[r][c] = shadow;
            }
        }
    }

    printf("\033[?2026l"); /* End synchronized update */
    fflush(stdout);
}

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
            return "DEAD / CRASHED";
        }
    }

    switch (state)
    {
        case GRIC_STATUS_INIT:    return "INITIALIZING";
        case GRIC_STATUS_RUNNING: return "RUNNING";
        case GRIC_STATUS_SUCCESS: return "SUCCESS";
        case GRIC_STATUS_ERROR:   return "ERROR";
        case GRIC_STATUS_ABORTED: return "ABORTED";
        default:                  return "UNKNOWN";
    }
}

static int get_state_color(
    uint32_t state,
    pid_t    pid)
{
    if (state == GRIC_STATUS_INIT || state == GRIC_STATUS_RUNNING)
    {
        if (!is_pid_alive(pid))
        {
            return COLOR_RED;
        }
    }

    switch (state)
    {
        case GRIC_STATUS_INIT:    return COLOR_YELLOW;
        case GRIC_STATUS_RUNNING: return COLOR_BLUE;
        case GRIC_STATUS_SUCCESS: return COLOR_GREEN;
        case GRIC_STATUS_ERROR:   return COLOR_RED;
        case GRIC_STATUS_ABORTED: return COLOR_RED;
        default:                  return COLOR_WHITE;
    }
}

static void print_status_classic(
    const GricClusterShmStatus *status)
{
    int use_color = (ov__color_level > 0 && isatty(STDOUT_FILENO));
    printf("\n%s--- gric-cluster telemetry status ---%s\n",
           use_color ? ANSI_BOLD : "", use_color ? ANSI_COLOR_RESET : "");
    printf("PID:                  %u\n", status->pid);
    printf("CWD:                  %s\n", status->config_cwd);
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

static void print_help_standard(
    const char *progname,
    int         color)
{
    #define C_STR(code, txt) (color ? (code txt MH_RST) : (txt))

    /* Build a colored progname string since C_STR requires string literals */
    char pn_colored[512];
    if (color)
        snprintf(pn_colored, sizeof(pn_colored), MH_CMD "%s" MH_RST, progname);
    else
        snprintf(pn_colored, sizeof(pn_colored), "%s", progname);

    printf("\n%s\n", C_STR(MH_TITLE, "NAME"));
    printf("  %s\n", C_STR(MH_CMD, "gric-status - Monitor shared-memory telemetry from gric-cluster"));
    printf("\n%s\n", C_STR(MH_HDR, "USAGE"));
    printf("  %s %s [%s]\n", pn_colored, C_STR(MH_ARG, "<shm_file_path>"), C_STR(MH_OPT, "[options]"));

    printf("\n%s\n", C_STR(MH_HDR, "DESCRIPTION"));
    printf("  Connects to a file-mapped shared memory telemetry file produced by a running\n");
    printf("  gric-cluster process. Reads and reports real-time metrics, including frame\n");
    printf("  counts, active/spawned clusters, distance computations, pruning statistics,\n");
    printf("  and process resource limits.\n");

    printf("\n%s\n", C_STR(MH_HDR, "OPTIONS"));
    printf("  %-30s %s\n", C_STR(MH_OPT, "-w, --watch"), "Enter real-time interactive terminal monitoring dashboard mode");
    printf("  %-30s %s\n", C_STR(MH_OPT, "-r, --rate <hz>"), "Set TUI update rate frequency in Hz (default: 15)");
    printf("  %-30s %s\n", C_STR(MH_OPT, "-h, --help"), "Show this full help message with color (if supported)");
    printf("  %-30s %s\n", C_STR(MH_OPT, "-hm, --help-mono"), "Show full help message forced to monochrome");
    printf("  %-30s %s\n", C_STR(MH_OPT, "-h1, --help-oneline"), "Show a brief one-line description and exit");
    printf("  %-30s %s\n", C_STR(MH_OPT, "-h2, --help-description"), "Show a verbose plain-text description and exit");

    printf("\n%s\n", C_STR(MH_HDR, "INTERACTIVE CONTROLS"));
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[space]"), "Pause / Resume dashboard telemetry refresh");
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[+], [=]"), "Increase refresh rate by 1 Hz");
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[-]"), "Decrease refresh rate by 1 Hz");
    printf("  %-30s %s\n", C_STR(MH_BOLD, "[q], [Esc]"), "Cleanly quit the utility and restore terminal settings");

    printf("\n%s\n", C_STR(MH_HDR, "EXAMPLES"));
    printf("  %s %s\n", pn_colored, C_STR(MH_ARG, "/tmp/gric_status.shm"));
    printf("    Print a one-shot telemetry snapshot to stdout.\n");
    printf("  %s %s %s\n", pn_colored, C_STR(MH_ARG, "/tmp/gric_status.shm"), C_STR(MH_OPT, "-w"));
    printf("    Launch interactive TUI dashboard at the default 15 Hz refresh rate.\n");
    printf("  %s %s %s %s\n", pn_colored, C_STR(MH_ARG, "/tmp/gric_status.shm"),
           C_STR(MH_OPT, "-w -r"), C_STR(MH_ARG, "30"));
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

int main(int argc, char *argv[])
{
    /* 1. Handle -h1 or --help-oneline first before anything else */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("gric-status: Monitor real-time clustering telemetry and process status from shared memory.\n");
            return 0;
        }
    }

    /* 2. Handle -h2 or --help-description next */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h2") == 0 || strcmp(argv[i], "--help-description") == 0)
        {
            printf("This program attaches to the file-mapped shared memory created by gric-cluster\n");
            printf("and displays real-time telemetry metrics in either classic text or interactive\n");
            printf("dashboard modes. It helps track progress, active clusters, distance computation\n");
            printf("counts, pruning ratios, RSS memory consumption, and OpenMP thread activity.\n");
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
    uint64_t prev_processed = 0;
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
                        cluster_cpu = 100.0 * ((ticks - last_cluster_ticks) / clk_tck) / elapsed_sec;
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
                    uint64_t dp = fps_history[latest_idx].processed - fps_history[oldest_idx].processed;

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
                tui_draw_string(4, col, val, color, COLOR_DEFAULT, frame_history[idx].created_cluster);
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
                tui_draw_string(5, col, val, color, COLOR_DEFAULT, frame_history[idx].created_cluster);
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
                tui_draw_string(6, col, val, color, COLOR_DEFAULT, frame_history[idx].created_cluster);
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
            snprintf(prog_lbl, sizeof(prog_lbl), "Progress (%" PRIu64 "/%" PRIu64 "): ", processed, total);
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
                    tui_draw_string(h_top + 8, start_col + 1 + b, "█", COLOR_GREEN, COLOR_DEFAULT, 0);
                }
                else
                {
                    tui_draw_string(h_top + 8, start_col + 1 + b, "░", COLOR_GREY, COLOR_DEFAULT, 0);
                }
            }
            snprintf(buf, sizeof(buf), "] %.1f%%", pct);
            tui_draw_string(h_top + 8, start_col + 1 + bar_width, buf, COLOR_WHITE, COLOR_DEFAULT, 0);

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
            tui_draw_box(h_top, w_left, w_right, h_bottom, "RESOURCES", COLOR_CYAN, COLOR_DEFAULT, 1);

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
                                status->time_step_3a + status->time_step_3b_score + status->time_step_3b_filter +
                                status->time_step_3b_eval + status->time_step_3c + status->time_step_4 +
                                status->time_step_5 + status->time_step_refine;
            double p_io = 0.0, p_s1 = 0.0, p_s2 = 0.0, p_s3a = 0.0, p_s3b_score = 0.0,
                   p_s3b_filter = 0.0, p_s3b_eval = 0.0, p_s3c = 0.0, p_s4 = 0.0, p_s5 = 0.0, p_ref = 0.0;
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

            tui_draw_string(h_top + 7, w_left + 2, "Profile (Total / Rolling 100):", COLOR_ORANGE, COLOR_DEFAULT, 1);

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
                p_io, p_s1, p_s2, p_s3a, p_s3b_score, p_s3b_filter, p_s3b_eval, p_s3c, p_s4, p_s5, p_ref
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
                         percentages[step], rolling_p[step], max_desc_len, descriptions[i]);
                tui_draw_string(row, w_left + 2, buf, COLOR_WHITE, COLOR_DEFAULT, 0);
            }

            /* Help info line at the very bottom */
            char help_buf[256];
            snprintf(help_buf, sizeof(help_buf),
                     "[q] Quit TUI    [space] Pause/Resume    [+]/[-] Rate: %.0f Hz", rate_hz);
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

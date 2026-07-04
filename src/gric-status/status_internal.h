/**
 * @file status_internal.h
 * @brief Internal definitions and declarations for the gric-status client.
 */

#ifndef STATUS_INTERNAL_H
#define STATUS_INTERNAL_H

#include "gric-cluster/core/cluster_shm.h"
#include <inttypes.h>
#include <signal.h>
#include <termios.h>

#define MAX_ROWS 60
#define MAX_COLS 180
#define FPS_SAMPLES 10
#define FRAME_HISTORY_SIZE 100

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
    double   time;
    uint64_t processed;
} FpsSample;

typedef struct
{
    uint64_t solve_dfc;
    uint64_t solve_dcc;
    double   assign_dist;
    int      created_cluster;
    double   step_times[11];
} FrameStatsHistory;

typedef struct
{
    char utf8[5]; /* 4-byte UTF-8 sequence + NUL terminator */
    int  fg;
    int  bg;
    int  bold;
} ScreenCell;

/* Global state variables */
extern int sc_rows;
extern int sc_cols;
extern ScreenCell front_buf[MAX_ROWS][MAX_COLS];
extern ScreenCell shadow_buf[MAX_ROWS][MAX_COLS];
extern struct termios orig_termios;
extern int raw_active;
extern int ov__color_level;
extern volatile sig_atomic_t sig_quit;

extern FpsSample fps_history[FPS_SAMPLES];
extern int fps_history_count;
extern int fps_history_index;

extern FrameStatsHistory frame_history[FRAME_HISTORY_SIZE];
extern int frame_history_count;
extern int frame_history_index;
extern uint64_t last_tracked_frame;
extern uint64_t last_num_new_clusters;

extern double last_cum_io;
extern double last_cum_s1;
extern double last_cum_s2;
extern double last_cum_s3a;
extern double last_cum_s3b_score;
extern double last_cum_s3b_filter;
extern double last_cum_s3b_eval;
extern double last_cum_s3c;
extern double last_cum_s4;
extern double last_cum_s5;
extern double last_cum_ref;

/* Terminal Control Functions */

/**
 * tui_enter_raw_mode() - Put terminal into raw mode.
 */
void tui_enter_raw_mode(void);

/**
 * tui_exit_raw_mode() - Restore original terminal settings.
 */
void tui_exit_raw_mode(void);

/**
 * crash_handler() - Restore terminal and re-raise crash signal.
 * @sig: Signal number causing the crash.
 */
void crash_handler(
    int sig);

/**
 * setup_signals() - Configure signal handlers for TUI client.
 */
void setup_signals(void);

/**
 * handle_sigint() - Set exit flag on SIGINT/SIGTERM.
 * @sig: Signal number received.
 */
void handle_sigint(
    int sig);

/* Shared Memory & Process Functions */

/**
 * is_pid_alive() - Check if a process ID is currently running.
 * @pid: Process ID to check.
 */
int is_pid_alive(
    pid_t pid);

/**
 * get_process_cpu_ticks() - Read CPU ticks for a process from /proc.
 * @pid:   Process ID to query.
 * @utime: Pointer to store user space CPU ticks.
 * @stime: Pointer to store kernel space CPU ticks.
 */
int get_process_cpu_ticks(
    pid_t          pid,
    unsigned long *utime,
    unsigned long *stime);

/* TUI Rendering Functions */

/**
 * ov_detect_color_level() - Auto-detect terminal color support.
 * @help_mono: Force monochrome flag.
 */
void ov_detect_color_level(
    int help_mono);

/**
 * tui_apply_colors() - Output ANSI sequence to set text color.
 * @fg:   Foreground color constant.
 * @bg:   Background color constant.
 * @bold: Bold flag (1 or 0).
 */
void tui_apply_colors(
    int fg,
    int bg,
    int bold);

/**
 * tui_clear() - Clear shadow screen buffer.
 */
void tui_clear(void);

/**
 * tui_draw_string() - Write string to shadow screen buffer.
 * @r:    Row position (0-indexed).
 * @c:    Column position (0-indexed).
 * @str:  String to draw.
 * @fg:   Foreground color constant.
 * @bg:   Background color constant.
 * @bold: Bold flag (1 or 0).
 */
void tui_draw_string(
    int         r,
    int         c,
    const char *str,
    int         fg,
    int         bg,
    int         bold);

/**
 * tui_draw_box() - Draw a bordered box with optional title.
 * @r:     Row position (0-indexed).
 * @c:     Column position (0-indexed).
 * @w:     Width of box.
 * @h:     Height of box.
 * @title: Optional title string to display.
 * @fg:    Border/Title foreground color.
 * @bg:    Border/Title background color.
 * @bold:  Bold borders flag (1 or 0).
 */
void tui_draw_box(
    int         r,
    int         c,
    int         w,
    int         h,
    const char *title,
    int         fg,
    int         bg,
    int         bold);

/**
 * tui_draw_label_button() - Draw a labeled toggle status indicator.
 * @r:     Row position.
 * @c:     Column position.
 * @label: Button label text.
 * @is_on: State flag (active/inactive).
 */
void tui_draw_label_button(
    int         r,
    int         c,
    const char *label,
    int         is_on);

/**
 * format_5char_float() - Format double into exactly 5-character string.
 * @out: Output buffer of size >= 6.
 * @val: Value to format.
 */
void format_5char_float(
    char   *out,
    double  val);

/**
 * format_5char_int() - Format uint64_t into exactly 5-character string.
 * @out: Output buffer of size >= 6.
 * @val: Value to format.
 */
void format_5char_int(
    char     *out,
    uint64_t  val);

/**
 * tui_flush() - Diff shadow buffer against front buffer and update screen.
 */
void tui_flush(void);

/**
 * get_state_string() - Map status state to display string.
 * @state: Status state value.
 * @pid:   Target process ID.
 */
const char *get_state_string(
    uint32_t state,
    pid_t    pid);

/**
 * get_state_color() - Map status state to UI color constant.
 * @state: Status state value.
 * @pid:   Target process ID.
 */
int get_state_color(
    uint32_t state,
    pid_t    pid);

#endif /* STATUS_INTERNAL_H */

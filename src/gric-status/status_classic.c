/**
 * @file status_classic.c
 * @brief One-shot classic CLI telemetry and help display for gric-status.
 */

#define _POSIX_C_SOURCE 200809L
#include "status_internal.h"
#include "shared/cli_colors.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void print_status_classic(
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
    printf("DCC Matrix Populated: %" PRIu64 " / %" PRIu64 " pairs (explicit calls: %" PRIu64 ")\n",
           status->dcc_entries_populated,
           status->dcc_pairs_total,
           status->framedist_calls_intercluster);
    printf("Candidates Pruned:    %" PRIu64 "\n", status->clusters_pruned);
    printf("Missed Frames:        %" PRIu64 "\n", status->total_missed_frames);

    double avg_dists = (status->total_frames_processed > 0)
                           ? (double)status->framedist_calls / (double)status->total_frames_processed
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
} // print_status_classic

static void print_help_standard_raw(
    const char *progname,
    int         color)
{
#define C_STR(code, txt) (color ? (code txt MH_RST) : (txt))

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
#undef C_STR
} // print_help_standard_raw

void print_help_standard(
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
} // print_help_standard

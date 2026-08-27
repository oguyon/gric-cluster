/**
 * @file server_job_mgmt.h
 * @brief Background CLI job management and process table.
 */

#ifndef SERVER_JOB_MGMT_H
#define SERVER_JOB_MGMT_H

#include "http_server.h"
#include <limits.h>
#include <sys/types.h>
#include <time.h>

#define MAX_JOBS 16

typedef struct
{
    char   id[64];
    pid_t  pid;
    pid_t  streamer_pid;
    char   stream_name[256];
    int    active;
    int    finished;
    int    exit_code;
    char   log_path[PATH_MAX];
    char   shm_path[PATH_MAX];
    time_t start_time;
} CliJob;

extern CliJob s_jobs[MAX_JOBS];
extern int    s_job_count;

int ensure_tmux_session(
    const ServerConfig *config);

void stop_tmux_session(
    void);

int format_shm_telemetry_json(
    const char *shm_path,
    char       *out_buf,
    size_t      out_size);

void handle_api_cli_session_init(
    int                 client_fd,
    const ServerConfig *config);

void handle_api_cli_session_stop(
    int client_fd);

void handle_api_cli_session_status(
    int client_fd);

void handle_api_cli_status(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

void handle_api_cli_kill(
    int         client_fd,
    const char *body);

void server_jobs_cleanup(
    void);

#endif /* SERVER_JOB_MGMT_H */

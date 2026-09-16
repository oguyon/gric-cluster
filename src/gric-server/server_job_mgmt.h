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

/**
 * ensure_tmux_session() - Ensure dedicated background tmux session is alive.
 * @config: Server configuration.
 *
 * Return: 0 on success, -1 on failure.
 */
int ensure_tmux_session(
    const ServerConfig *config);

/**
 * stop_tmux_session() - Gracefully stop and kill background tmux session.
 */
void stop_tmux_session(
    void);

/**
 * format_shm_telemetry_json() - Read ImageStreamIO SHM telemetry into JSON string.
 * @shm_path: Path to shared memory file or name.
 * @out_buf:  Output JSON buffer.
 * @out_size: Output buffer capacity in bytes.
 *
 * Return: Number of bytes written, or -1 on error.
 */
int format_shm_telemetry_json(
    const char *shm_path,
    char       *out_buf,
    size_t      out_size);

/**
 * handle_api_cli_session_init() - Handle POST /api/cli/session/init request.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration.
 */
void handle_api_cli_session_init(
    int                 client_fd,
    const ServerConfig *config);

/**
 * handle_api_cli_session_stop() - Handle POST /api/cli/session/stop request.
 * @client_fd: Connected client socket descriptor.
 */
void handle_api_cli_session_stop(
    int client_fd);

/**
 * handle_api_cli_session_status() - Handle GET /api/cli/session/status request.
 * @client_fd: Connected client socket descriptor.
 */
void handle_api_cli_session_status(
    int client_fd);

/**
 * handle_api_cli_status() - Handle GET /api/cli/status polling job progress.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration.
 * @query:     HTTP query string with job ID.
 */
void handle_api_cli_status(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

/**
 * handle_api_cli_kill() - Handle POST /api/cli/kill terminating active job.
 * @client_fd: Connected client socket descriptor.
 * @body:      JSON body containing job ID to kill.
 */
void handle_api_cli_kill(
    int         client_fd,
    const char *body);

/**
 * server_jobs_cleanup() - Terminate running child processes on server shutdown.
 */
void server_jobs_cleanup(
    void);

#endif /* SERVER_JOB_MGMT_H */

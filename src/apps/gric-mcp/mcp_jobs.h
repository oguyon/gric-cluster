/**
 * @file mcp_jobs.h
 * @brief Async background job execution, monitoring, and lifecycle management for gric-mcp.
 */

#ifndef MCP_JOBS_H
#define MCP_JOBS_H

#include <stddef.h>
#include <sys/types.h>
#include "shared/cjson/cJSON.h"

enum mcp_job_status
{
    MCP_JOB_STATUS_UNKNOWN   = 0,
    MCP_JOB_STATUS_RUNNING   = 1,
    MCP_JOB_STATUS_COMPLETED = 2,
    MCP_JOB_STATUS_FAILED    = 3,
    MCP_JOB_STATUS_CANCELLED = 4,
};

struct mcp_job_progress
{
    long   current_frame;
    long   total_frames;
    int    clusters;
    long   dists;
    long   pruned;
    double progress_pct;
    double elapsed_sec;
};

struct mcp_job_info
{
    char                    job_id[64];
    pid_t                   pid;
    enum mcp_job_status     status;
    int                     exit_code;
    char                    program[64];
    char                    log_path[1024];
    char                    outdir[1024];
    double                  start_time_sec;
    double                  end_time_sec;
    struct mcp_job_progress progress;
    char                    log_tail[1024];
};

/**
 * mcp_jobs_init() - Initialize the job manager subsystem and base job directories.
 *
 * Return: 0 on success, -1 on failure.
 */
int mcp_jobs_init(void);

/**
 * mcp_job_spawn() - Launch an asynchronous background job with stdout/stderr redirection.
 * @program:     Short name of program (e.g. "gric-cluster", "gric-knn").
 * @argv:        NULL-terminated argument vector for execvp.
 * @output_dir:  Optional output directory path (for tracking/cleanup), or NULL.
 * @out_job_id:  Buffer to store generated unique job ID.
 * @job_id_size: Capacity of out_job_id buffer.
 *
 * Return: 0 on success, -1 on failure.
 */
int mcp_job_spawn(
    const char        *program,
    const char *const  argv[],
    const char        *output_dir,
    char              *out_job_id,
    size_t             job_id_size);

/**
 * mcp_job_get_info() - Retrieve current status and live progress for a job.
 * @job_id: Identifier of the job.
 * @info:   Pointer to struct mcp_job_info to populate.
 *
 * Return: 0 on success, -1 if job not found or read error.
 */
int mcp_job_get_info(
    const char          *job_id,
    struct mcp_job_info *info);

/**
 * mcp_job_cancel() - Terminate a running job and optionally clean up its outputs.
 * @job_id:      Identifier of the job to cancel.
 * @clean_output: 1 to delete output directory, 0 to preserve partial output.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_job_cancel(
    const char *job_id,
    int         clean_output);

/**
 * mcp_job_status_to_string() - Convert job status enum to human-readable string.
 * @status: Enum status value.
 *
 * Return: Static string representation ("RUNNING", "COMPLETED", etc.).
 */
const char *mcp_job_status_to_string(
    enum mcp_job_status status);

#endif // MCP_JOBS_H

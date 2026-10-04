/**
 * @file mcp_exec.h
 * @brief Shell-free process execution and process inspection for gric-mcp.
 */

#ifndef MCP_EXEC_H
#define MCP_EXEC_H

#include <stddef.h>
#include <sys/types.h>

/**
 * mcp_exec_capture() - Run executable without shell, capturing stdout and stderr.
 * @argv:        NUL-terminated array of arguments (argv[0] resolved via PATH).
 * @out:         Output buffer to store merged stdout and stderr.
 * @out_size:    Size of output buffer in bytes.
 * @timeout_ms:  Execution timeout in milliseconds (<0 for no timeout).
 * @exit_status: Pointer to store child exit status or termination code.
 *
 * Return: 0 on success, -2 on timeout, -1 on fork/exec failure.
 */
int mcp_exec_capture(
    const char *const argv[],
    char              *out,
    size_t             out_size,
    int                timeout_ms,
    int               *exit_status);

/**
 * mcp_exec_spawn_detached() - Spawn detached background daemon process without a shell.
 * @argv:     NUL-terminated array of arguments (argv[0] resolved via PATH).
 * @log_path: Path to log file for stdout/stderr (or NULL for /dev/null).
 * @pid_out:  Pointer to store the spawned child PID.
 *
 * Return: 0 on success, -1 on failure.
 */
int mcp_exec_spawn_detached(
    const char *const argv[],
    const char        *log_path,
    pid_t             *pid_out);

/**
 * mcp_find_process() - Scan /proc to find PID of running process matching criteria.
 * @exe_basename: Expected basename of executable (e.g. "milk-fpsexec-gric-cluster"), or NULL.
 * @token:        Substring required to be in process cmdline, or NULL.
 * @pid_out:      Pointer to store discovered PID.
 *
 * Return: 0 if found, -1 if no matching process found.
 */
int mcp_find_process(
    const char *exe_basename,
    const char *token,
    pid_t      *pid_out);

#endif // MCP_EXEC_H

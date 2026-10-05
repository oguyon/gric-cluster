/**
 * @file status_shm.c
 * @brief Process status and CPU telemetry functions for gric-status.
 */

#include "status_internal.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

/**
 * is_pid_alive() - Checks if a process ID is currently running.
 * @pid: The process ID to query.
 *
 * Sends signal 0 to the target process. Returns 1 if alive/accessible,
 * or 0 if it has definitely exited.
 *
 * Return: 1 if the process is alive, 0 otherwise.
 */
int is_pid_alive(
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

/**
 * get_process_cpu_ticks() - Read process user and system CPU ticks from /proc.
 * @pid:   The process ID to query.
 * @utime: Output pointer for user-mode CPU ticks.
 * @stime: Output pointer for system-mode CPU ticks.
 *
 * Parses /proc/<pid>/stat to extract the 14th (utime) and 15th (stime) fields.
 *
 * Return: 0 on success, -1 on failure (e.g., process exited, invalid stat format).
 */
int get_process_cpu_ticks(
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

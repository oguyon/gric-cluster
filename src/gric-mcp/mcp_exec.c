/**
 * @file mcp_exec.c
 * @brief Shell-free process execution and process inspection for gric-mcp.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mcp_exec.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/**
 * close_inherited_fds() - Close open file descriptors above STDERR_FILENO.
 */
static void close_inherited_fds(void)
{
    DIR *dir = opendir("/proc/self/fd");
    if (dir != NULL)
    {
        int dfd = dirfd(dir);
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL)
        {
            int fd = atoi(ent->d_name);
            if (fd > STDERR_FILENO && fd != dfd)
            {
                close(fd);
            }
        }
        closedir(dir);
    }
    else
    {
        long max_fd = sysconf(_SC_OPEN_MAX);
        if (max_fd < 0 || max_fd > 1024)
        {
            max_fd = 1024;
        }
        for (int fd = STDERR_FILENO + 1; fd < (int)max_fd; fd++)
        {
            close(fd);
        }
    }
} // close_inherited_fds

int mcp_exec_capture(
    const char *const argv[],
    char              *out,
    size_t             out_size,
    int                timeout_ms,
    int               *exit_status)
{
    if (argv == NULL || argv[0] == NULL)
    {
        return -1;
    }

    if (out != NULL && out_size > 0)
    {
        out[0] = '\0';
    }

    int pipefd[2];
#if defined(__APPLE__) || !defined(O_CLOEXEC)
    if (pipe(pipefd) != 0)
    {
        return -1;
    }
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
#else
    if (pipe2(pipefd, O_CLOEXEC) != 0)
    {
        return -1;
    }
#endif

    pid_t pid = fork();
    if (pid < 0)
    {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

    if (pid == 0)
    {
        /* Child process: never inherit standard JSON-RPC streams */
        close(pipefd[0]);

        if (dup2(pipefd[1], STDOUT_FILENO) < 0 ||
            dup2(pipefd[1], STDERR_FILENO) < 0)
        {
            _exit(127);
        }
        close(pipefd[1]);

        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            if (devnull > STDERR_FILENO)
            {
                close(devnull);
            }
        }

        close_inherited_fds();
        execvp(argv[0], (char *const *)argv);
        _exit(127);
    }

    /* Parent process */
    close(pipefd[1]);

    size_t bytes_read = 0;
    int timed_out = 0;
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    struct pollfd pfd;
    pfd.fd = pipefd[0];
    pfd.events = POLLIN | POLLHUP | POLLERR;

    while (1)
    {
        int remaining_ms = -1;
        if (timeout_ms >= 0)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long elapsed_ms = (now.tv_sec - start_time.tv_sec) * 1000 +
                              (now.tv_nsec - start_time.tv_nsec) / 1000000;
            remaining_ms = timeout_ms - (int)elapsed_ms;
            if (remaining_ms <= 0)
            {
                timed_out = 1;
                break;
            }
        }

        int poll_ret = poll(&pfd, 1, remaining_ms);
        if (poll_ret < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (poll_ret == 0)
        {
            timed_out = 1;
            break;
        }

        if (pfd.revents & POLLIN)
        {
            char buf[512];
            ssize_t n = read(pipefd[0], buf, sizeof(buf));
            if (n <= 0)
            {
                break;
            }

            if (out != NULL && out_size > 1)
            {
                size_t avail = out_size - 1 - bytes_read;
                if (avail > 0)
                {
                    size_t to_copy = ((size_t)n < avail) ? (size_t)n : avail;
                    memcpy(out + bytes_read, buf, to_copy);
                    bytes_read += to_copy;
                    out[bytes_read] = '\0';
                }
            }
        }
        else if (pfd.revents & (POLLHUP | POLLERR))
        {
            break;
        }
    } // while (1)

    close(pipefd[0]);

    if (timed_out)
    {
        kill(pid, SIGTERM);
        struct timespec sleep_ts = { .tv_sec = 0, .tv_nsec = 50000000 }; /* 50ms */
        nanosleep(&sleep_ts, NULL);
        if (waitpid(pid, exit_status, WNOHANG) == 0)
        {
            nanosleep(&sleep_ts, NULL);
            if (waitpid(pid, exit_status, WNOHANG) == 0)
            {
                kill(pid, SIGKILL);
                waitpid(pid, exit_status, 0);
            }
        }
        return -2;
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
    {
        if (errno != EINTR)
        {
            break;
        }
    }

    if (exit_status != NULL)
    {
        if (WIFEXITED(status))
        {
            *exit_status = WEXITSTATUS(status);
        }
        else if (WIFSIGNALED(status))
        {
            *exit_status = 128 + WTERMSIG(status);
        }
        else
        {
            *exit_status = status;
        }
    }

    return 0;
} // mcp_exec_capture

int mcp_exec_spawn_detached(
    const char *const argv[],
    const char        *log_path,
    pid_t             *pid_out)
{
    if (argv == NULL || argv[0] == NULL)
    {
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        return -1;
    }

    if (pid == 0)
    {
        /* Child: detach from terminal session */
        if (setsid() < 0)
        {
            _exit(127);
        }

        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            if (devnull > STDERR_FILENO)
            {
                close(devnull);
            }
        }

        int logfd = -1;
        if (log_path != NULL)
        {
            logfd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        }
        if (logfd < 0)
        {
            logfd = open("/dev/null", O_WRONLY);
        }

        if (logfd >= 0)
        {
            dup2(logfd, STDOUT_FILENO);
            dup2(logfd, STDERR_FILENO);
            if (logfd > STDERR_FILENO)
            {
                close(logfd);
            }
        }

        close_inherited_fds();
        execvp(argv[0], (char *const *)argv);
        _exit(127);
    }

    if (pid_out != NULL)
    {
        *pid_out = pid;
    }

    return 0;
} // mcp_exec_spawn_detached

int mcp_find_process(
    const char *exe_basename,
    const char *token,
    pid_t      *pid_out)
{
    DIR *proc = opendir("/proc");
    if (proc == NULL)
    {
        return -1;
    }

    pid_t self_pid = getpid();
    struct dirent *ent;

    while ((ent = readdir(proc)) != NULL)
    {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9')
        {
            continue;
        }

        pid_t pid = (pid_t)atoi(ent->d_name);
        if (pid <= 0 || pid == self_pid)
        {
            continue;
        }

        char path[256];
        snprintf(path, sizeof(path), "/proc/%d/cmdline", (int)pid);

        int fd = open(path, O_RDONLY);
        if (fd < 0)
        {
            continue;
        }

        char cmdline[2048];
        ssize_t n = read(fd, cmdline, sizeof(cmdline) - 1);
        close(fd);

        if (n <= 0)
        {
            continue;
        }
        cmdline[n] = '\0';

        /* Check executable basename if specified */
        if (exe_basename != NULL)
        {
            const char *base = strrchr(cmdline, '/');
            base = (base != NULL) ? base + 1 : cmdline;
            if (strcmp(base, exe_basename) != 0)
            {
                continue;
            }
        }

        /* Check token in full cmdline */
        if (token != NULL)
        {
            /* Replace NUL separators with spaces for substring searching */
            for (ssize_t ii = 0; ii < n - 1; ii++)
            {
                if (cmdline[ii] == '\0')
                {
                    cmdline[ii] = ' ';
                }
            }
            if (strstr(cmdline, token) == NULL)
            {
                continue;
            }
        }

        closedir(proc);
        if (pid_out != NULL)
        {
            *pid_out = pid;
        }
        return 0;
    } // while readdir

    closedir(proc);
    return -1;
} // mcp_find_process

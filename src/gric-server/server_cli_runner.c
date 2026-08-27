/**
 * @file server_cli_runner.c
 * @brief Dispatch and execution of asynchronous CLI tasks into tmux sessions.
 */

#define _GNU_SOURCE
#include "server_cli_runner.h"
#include "server_http_utils.h"
#include "server_job_mgmt.h"
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void handle_api_cli_run(
    int                 client_fd,
    const ServerConfig *config,
    const char         *body,
    size_t              body_len)
{
    if (!body || body_len == 0)
    {
        api_send_json(client_fd, 400, "{\"error\":\"Empty command specification\"}");
        return;
    }

    char cmd_name[64] = "gric-cluster";
    const char *cmd_key = "\"cmd\":";
    const char *c_pos = strstr(body, cmd_key);
    if (c_pos)
    {
        c_pos += strlen(cmd_key);
        while (*c_pos == ' ' || *c_pos == '"') c_pos++;
        const char *c_end = strchr(c_pos, '"');
        if (c_end)
        {
            size_t clen = (size_t)(c_end - c_pos);
            if (clen < sizeof(cmd_name))
            {
                memcpy(cmd_name, c_pos, clen);
                cmd_name[clen] = '\0';
            }
        }
    }

    char bin_path[PATH_MAX];
    if (!check_binary_available(config, cmd_name, bin_path, sizeof(bin_path)))
    {
        char err[512];
        snprintf(err, sizeof(err), "{\"error\":\"Executable '%s' not found\"}", cmd_name);
        api_send_json(client_fd, 400, err);
        return;
    }

    char *argv[64];
    int argc = 0;
    argv[argc++] = strdup(bin_path);

    const char *args_key = "\"args\":";
    const char *a_pos = strstr(body, args_key);
    if (a_pos)
    {
        const char *bracket = strchr(a_pos, '[');
        if (bracket)
        {
            const char *cur = bracket + 1;
            while (*cur && *cur != ']' && argc < 60)
            {
                while (*cur == ' ' || *cur == ',' || *cur == '\n' || *cur == '\r') cur++;
                if (*cur == '"')
                {
                    cur++;
                    const char *arg_end = strchr(cur, '"');
                    if (arg_end)
                    {
                        size_t arg_len = (size_t)(arg_end - cur);
                        char *arg_val = (char *)malloc(arg_len + 1);
                        if (arg_val)
                        {
                            memcpy(arg_val, cur, arg_len);
                            arg_val[arg_len] = '\0';
                            argv[argc++] = arg_val;
                        }
                        cur = arg_end + 1;
                    }
                    else
                    {
                        break;
                    }
                }
                else if (*cur == ']')
                {
                    break;
                }
                else
                {
                    cur++;
                }
            }
        }
    }
    argv[argc] = NULL;

    int slot = -1;
    for (int ii = 0; ii < MAX_JOBS; ii++)
    {
        if (!s_jobs[ii].active && (s_jobs[ii].finished || s_jobs[ii].pid == 0))
        {
            slot = ii;
            break;
        }
    }
    if (slot == -1 && s_job_count < MAX_JOBS)
    {
        slot = s_job_count++;
    }

    if (slot == -1)
    {
        for (int i = 0; i < argc; i++) free(argv[i]);
        api_send_json(client_fd, 500, "{\"error\":\"Max concurrent job limit reached\"}");
        return;
    }

    CliJob *job = &s_jobs[slot];
    memset(job, 0, sizeof(CliJob));

    time_t now = time(NULL);
    char job_id_buf[64];
    snprintf(job_id_buf, sizeof(job_id_buf), "job_%ld_%d", (long)now, slot);
    snprintf(job->id, sizeof(job->id), "%s", job_id_buf);
    snprintf(job->log_path, sizeof(job->log_path), "/tmp/gric_%s.log", job_id_buf);
    int init_log_fd = open(job->log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (init_log_fd >= 0)
    {
        close(init_log_fd);
    }
    unlink("/tmp/gric_latest.log");
    int sym_ret = symlink(job->log_path, "/tmp/gric_latest.log");
    (void)sym_ret;

    if (strcmp(cmd_name, "gric-cluster") == 0)
    {
        int has_shm_arg = 0;
        for (int i = 0; i < argc; i++)
        {
            if (strcmp(argv[i], "-shm") == 0 || strcmp(argv[i], "-shm-file") == 0)
            {
                has_shm_arg = 1;
                if (i + 1 < argc)
                {
                    strncpy(job->shm_path, argv[i + 1], sizeof(job->shm_path) - 1);
                }
                break;
            }
        }
        if (!has_shm_arg)
        {
            snprintf(job->shm_path, sizeof(job->shm_path), "/tmp/gric_shm_%s.shm", job_id_buf);
            if (argc + 2 < 127)
            {
                argv[argc++] = strdup("-shm");
                argv[argc++] = strdup(job->shm_path);
                argv[argc] = NULL;
            }
        }
    }

    char stream_file[PATH_MAX] = {0};
    char stream_name[256] = {0};
    double stream_fps = 100.0;
    int stream_loop = 0;
    int stream_cnt2sync = 0;
    int is_stream = 0;

    const char *sf_pos = strstr(body, "\"stream_file\":");
    if (sf_pos)
    {
        sf_pos += strlen("\"stream_file\":");
        while (*sf_pos == ' ' || *sf_pos == '"') sf_pos++;
        const char *sf_end = strchr(sf_pos, '"');
        if (sf_end)
        {
            size_t sflen = (size_t)(sf_end - sf_pos);
            if (sflen < sizeof(stream_file))
            {
                memcpy(stream_file, sf_pos, sflen);
                stream_file[sflen] = '\0';
                is_stream = 1;
            }
        }
    }

    const char *sn_pos = strstr(body, "\"stream_name\":");
    if (sn_pos)
    {
        sn_pos += strlen("\"stream_name\":");
        while (*sn_pos == ' ' || *sn_pos == '"') sn_pos++;
        const char *sn_end = strchr(sn_pos, '"');
        if (sn_end)
        {
            size_t snlen = (size_t)(sn_end - sn_pos);
            if (snlen < sizeof(stream_name))
            {
                memcpy(stream_name, sn_pos, snlen);
                stream_name[snlen] = '\0';
            }
        }
    }

    const char *fps_pos = strstr(body, "\"stream_fps\":");
    if (fps_pos)
    {
        fps_pos += strlen("\"stream_fps\":");
        while (*fps_pos == ' ' || *fps_pos == ':') fps_pos++;
        stream_fps = atof(fps_pos);
    }

    const char *loop_pos = strstr(body, "\"stream_loop\":");
    if (loop_pos)
    {
        loop_pos += strlen("\"stream_loop\":");
        while (*loop_pos == ' ' || *loop_pos == ':') loop_pos++;
        if (strncmp(loop_pos, "true", 4) == 0) stream_loop = 1;
    }

    const char *cnt2_pos = strstr(body, "\"stream_cnt2sync\":");
    if (cnt2_pos)
    {
        cnt2_pos += strlen("\"stream_cnt2sync\":");
        while (*cnt2_pos == ' ' || *cnt2_pos == ':') cnt2_pos++;
        if (strncmp(cnt2_pos, "true", 4) == 0) stream_cnt2sync = 1;
    }

    pid_t streamer_pid = 0;
    if (is_stream && stream_file[0] != '\0')
    {
        if (stream_name[0] == '\0')
        {
            snprintf(stream_name, sizeof(stream_name), "gric_sim_%s", job_id_buf);
        }
        char txt2stream_bin[PATH_MAX];
        if (check_binary_available(config, "gric-txt2stream",
                                   txt2stream_bin, sizeof(txt2stream_bin)))
        {
            char streamer_log[PATH_MAX];
            snprintf(streamer_log, sizeof(streamer_log), "/tmp/gric_streamer_%s.log", job_id_buf);

            char fps_str[32];
            snprintf(fps_str, sizeof(fps_str), "%.1f", stream_fps);

            char *s_argv[16];
            int s_argc = 0;
            s_argv[s_argc++] = txt2stream_bin;
            s_argv[s_argc++] = stream_file;
            s_argv[s_argc++] = stream_name;
            s_argv[s_argc++] = "-fps";
            s_argv[s_argc++] = fps_str;
            if (stream_loop)
            {
                s_argv[s_argc++] = "-loop";
            }
            if (stream_cnt2sync)
            {
                s_argv[s_argc++] = "-cnt2sync";
            }
            s_argv[s_argc] = NULL;

            streamer_pid = fork();
            if (streamer_pid == 0)
            {
                int s_fd = open(streamer_log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (s_fd >= 0)
                {
                    dup2(s_fd, STDOUT_FILENO);
                    dup2(s_fd, STDERR_FILENO);
                    close(s_fd);
                }
                int cd_ret = chdir(config->workdir);
                (void)cd_ret;
                execv(txt2stream_bin, s_argv);
                _exit(127);
            }
            usleep(60000);
        }
    }
    job->streamer_pid = streamer_pid;
    snprintf(job->stream_name, sizeof(job->stream_name), "%s", stream_name);

    ensure_tmux_session(config);

    char done_path[PATH_MAX];
    snprintf(done_path, sizeof(done_path), "/tmp/gric_%s.done", job->id);
    unlink(done_path);

    char cmd_str[4096] = "";
    for (int i = 0; i < argc; i++)
    {
        if (i > 0)
        {
            strncat(cmd_str, " ", sizeof(cmd_str) - strlen(cmd_str) - 1);
        }
        strncat(cmd_str, argv[i], sizeof(cmd_str) - strlen(cmd_str) - 1);
    }

    job->start_time = now;
    job->active = 1;

    pid_t pid = fork();
    if (pid < 0)
    {
        job->active = 0;
        if (job->streamer_pid > 0)
        {
            kill(job->streamer_pid, SIGTERM);
            job->streamer_pid = 0;
        }
        for (int i = 0; i < argc; i++) free(argv[i]);
        api_send_json(client_fd, 500, "{\"error\":\"fork() failed\"}");
        return;
    }

    if (pid == 0)
    {
        int cd_ret = chdir(config->workdir);
        (void)cd_ret;

        char tmux_cmd[8192];
        snprintf(tmux_cmd, sizeof(tmux_cmd),
                 "tmux send-keys -t gric_cli:0 \"clear; "
                 "echo -e '\\033[1;36m[GRIC Native Runner] %s\\033[0m'; "
                 "%s 2>&1 | tee %s; echo \\$? > %s\" C-m",
                 cmd_str, cmd_str, job->log_path, done_path);
        int sys_ret = system(tmux_cmd);
        (void)sys_ret;

        while (access(done_path, F_OK) != 0)
        {
            usleep(50000);
        }

        int exit_code = 0;
        FILE *df = fopen(done_path, "r");
        if (df)
        {
            if (fscanf(df, "%d", &exit_code) != 1)
            {
                exit_code = 0;
            }
            fclose(df);
            unlink(done_path);
        }
        _exit(exit_code);
    }

    job->pid = pid;
    for (int i = 0; i < argc; i++)
    {
        free(argv[i]);
    }

    char resp[512];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"job_id\":\"%s\",\"pid\":%d,\"log\":\"%s\",\"shm\":\"%s\"}",
             job->id, (int)pid, job->log_path, job->shm_path);
    api_send_json(client_fd, 200, resp);
} // handle_api_cli_run

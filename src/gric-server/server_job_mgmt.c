/**
 * @file server_job_mgmt.c
 * @brief Background CLI job management and process table.
 */

#define _GNU_SOURCE
#include "server_job_mgmt.h"
#include "server_http_utils.h"
#include "cluster_shm.h"
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

CliJob s_jobs[MAX_JOBS];
int    s_job_count = 0;

int ensure_tmux_session(
    const ServerConfig *config)
{
    if (system("tmux has-session -t gric_cli 2>/dev/null") == 0)
    {
        return 1;
    }

    char create_cmd[PATH_MAX + 128];
    snprintf(create_cmd, sizeof(create_cmd),
             "tmux new-session -d -s gric_cli -c \"%s\" \"bash\"",
             config->workdir);
    int ret = system(create_cmd);
    return (ret == 0);
} // ensure_tmux_session

void stop_tmux_session(
    void)
{
    int ret = system("tmux kill-session -t gric_cli 2>/dev/null");
    (void)ret;
} // stop_tmux_session

void handle_api_cli_session_init(
    int                 client_fd,
    const ServerConfig *config)
{
    int ok = ensure_tmux_session(config);
    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"session\":\"gric_cli\",\"active\":%s}",
             ok ? "true" : "false");
    api_send_json(client_fd, 200, resp);
} // handle_api_cli_session_init

void handle_api_cli_session_stop(
    int client_fd)
{
    stop_tmux_session();
    api_send_json(client_fd, 200,
                  "{\"status\":\"ok\",\"session\":\"gric_cli\",\"active\":false}");
} // handle_api_cli_session_stop

void handle_api_cli_session_status(
    int client_fd)
{
    int exists = (system("tmux has-session -t gric_cli 2>/dev/null") == 0);
    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"session\":\"gric_cli\",\"exists\":%s}",
             exists ? "true" : "false");
    api_send_json(client_fd, 200, resp);
} // handle_api_cli_session_status

int format_shm_telemetry_json(
    const char *shm_path,
    char       *out_buf,
    size_t      out_size)
{
    if (!shm_path || shm_path[0] == '\0')
    {
        return 0;
    }

    int fd = open(shm_path, O_RDONLY);
    if (fd < 0)
    {
        return 0;
    }

    GricClusterShmStatus st;
    ssize_t rd = read(fd, &st, sizeof(st));
    close(fd);

    if (rd < (ssize_t)sizeof(st) || st.magic != GRIC_SHM_MAGIC)
    {
        return 0;
    }

    double progress = (st.total_frames > 0)
                          ? ((double)st.total_frames_processed / (double)st.total_frames)
                          : 0.0;
    if (progress > 1.0)
    {
        progress = 1.0;
    }

    const char *state_str = "init";
    if (st.status_state == GRIC_STATUS_RUNNING)
    {
        state_str = "running";
    }
    else if (st.status_state == GRIC_STATUS_SUCCESS)
    {
        state_str = "success";
    }
    else if (st.status_state == GRIC_STATUS_ERROR)
    {
        state_str = "error";
    }
    else if (st.status_state == GRIC_STATUS_ABORTED)
    {
        state_str = "aborted";
    }

    return snprintf(
        out_buf, out_size,
        "  \"telemetry\": {\n"
        "    \"version\": %u,\n"
        "    \"pid\": %u,\n"
        "    \"state\": \"%s\",\n"
        "    \"total_frames\": %llu,\n"
        "    \"processed_frames\": %llu,\n"
        "    \"progress\": %.4f,\n"
        "    \"num_clusters\": %u,\n"
        "    \"num_new_clusters\": %llu,\n"
        "    \"framedist_calls\": %llu,\n"
        "    \"framedist_sample\": %llu,\n"
        "    \"framedist_intercluster\": %llu,\n"
        "    \"clusters_pruned\": %llu,\n"
        "    \"total_missed_frames\": %llu,\n"
        "    \"elapsed_ms\": %.2f,\n"
        "    \"last_assignment_dist\": %.6f,\n"
        "    \"memory_rss_kb\": %llu,\n"
        "    \"active_threads\": %u,\n"
        "    \"last_frame_dists\": %llu,\n"
        "    \"last_frame_dfc\": %llu,\n"
        "    \"last_frame_dcc\": %llu,\n"
        "    \"stream_lag\": %ld,\n"
        "    \"step_timers\": {\n"
        "      \"io_ms\": %.2f,\n"
        "      \"step_1\": %.2f,\n"
        "      \"step_2\": %.2f,\n"
        "      \"step_3a\": %.2f,\n"
        "      \"step_3b\": %.2f,\n"
        "      \"step_3b_score\": %.2f,\n"
        "      \"step_3b_filter\": %.2f,\n"
        "      \"step_3b_eval\": %.2f,\n"
        "      \"step_3c\": %.2f,\n"
        "      \"step_4\": %.2f,\n"
        "      \"step_5\": %.2f,\n"
        "      \"refine_ms\": %.2f\n"
        "    },\n"
        "    \"entropy\": {\n"
        "      \"last_initial\": %.4f,\n"
        "      \"avg_initial\": %.4f,\n"
        "      \"gate_ratio\": %.4f\n"
        "    }\n"
        "  }",
        st.version,
        st.pid,
        state_str,
        (unsigned long long)st.total_frames,
        (unsigned long long)st.total_frames_processed,
        progress,
        st.num_clusters,
        (unsigned long long)st.num_new_clusters,
        (unsigned long long)st.framedist_calls,
        (unsigned long long)st.framedist_calls_sample,
        (unsigned long long)st.framedist_calls_intercluster,
        (unsigned long long)st.clusters_pruned,
        (unsigned long long)st.total_missed_frames,
        st.elapsed_ms,
        st.last_assignment_dist,
        (unsigned long long)st.memory_rss_kb,
        st.active_threads,
        (unsigned long long)st.last_frame_dists,
        (unsigned long long)st.last_frame_dfc,
        (unsigned long long)st.last_frame_dcc,
        st.stream_lag,
        st.time_io_ms,
        st.time_step_1,
        st.time_step_2,
        st.time_step_3a,
        st.time_step_3b,
        st.time_step_3b_score,
        st.time_step_3b_filter,
        st.time_step_3b_eval,
        st.time_step_3c,
        st.time_step_4,
        st.time_step_5,
        st.time_step_refine,
        st.entropy_last_initial,
        st.entropy_avg_initial,
        st.entropy_gate_ratio);
} // format_shm_telemetry_json

void handle_api_cli_status(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query)
{
    (void)config;
    char job_id[64];
    if (!get_query_param(query, "job_id", job_id, sizeof(job_id)))
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing job_id parameter\"}");
        return;
    }

    char offset_str[32] = "0";
    get_query_param(query, "offset", offset_str, sizeof(offset_str));
    long offset = atol(offset_str);
    if (offset < 0)
    {
        offset = 0;
    }

    CliJob *job = NULL;
    for (int ii = 0; ii < MAX_JOBS; ii++)
    {
        if (s_jobs[ii].pid > 0 && strcmp(s_jobs[ii].id, job_id) == 0)
        {
            job = &s_jobs[ii];
            break;
        }
    }

    if (!job)
    {
        api_send_json(client_fd, 404, "{\"error\":\"Job not found\"}");
        return;
    }

    if (job->active)
    {
        int status = 0;
        pid_t res = waitpid(job->pid, &status, WNOHANG);
        if (res == job->pid)
        {
            job->active = 0;
            job->finished = 1;
            if (WIFEXITED(status))
            {
                job->exit_code = WEXITSTATUS(status);
            }
            else if (WIFSIGNALED(status))
            {
                job->exit_code = 128 + WTERMSIG(status);
            }

            if (job->streamer_pid > 0)
            {
                kill(job->streamer_pid, SIGTERM);
                usleep(20000);
                kill(job->streamer_pid, SIGKILL);
                job->streamer_pid = 0;
            }
            if (job->stream_name[0] != '\0')
            {
                char shm_im[PATH_MAX];
                char shm_sem[PATH_MAX];
                snprintf(shm_im, sizeof(shm_im), "/dev/shm/%s.im.shm", job->stream_name);
                snprintf(shm_sem, sizeof(shm_sem), "/dev/shm/%s.sem.shm", job->stream_name);
                unlink(shm_im);
                unlink(shm_sem);
                job->stream_name[0] = '\0';
            }
        }
    }

    size_t chunk_cap = 65536;
    char *chunk = (char *)malloc(chunk_cap);
    size_t chunk_len = 0;
    long next_offset = offset;

    if (chunk)
    {
        FILE *lf = fopen(job->log_path, "rb");
        if (lf)
        {
            fseek(lf, 0, SEEK_END);
            long total_log_len = ftell(lf);
            if (offset < total_log_len)
            {
                fseek(lf, offset, SEEK_SET);
                size_t to_read = (size_t)(total_log_len - offset);
                if (to_read > chunk_cap - 1)
                {
                    to_read = chunk_cap - 1;
                }
                chunk_len = fread(chunk, 1, to_read, lf);
                next_offset = offset + (long)chunk_len;
            }
            fclose(lf);
        }
        chunk[chunk_len] = '\0';
    }

    char telemetry_json[4096] = "";
    int has_telemetry = 0;
    if (job->shm_path[0] != '\0')
    {
        has_telemetry = format_shm_telemetry_json(
            job->shm_path, telemetry_json, sizeof(telemetry_json));
    }

    size_t json_cap = chunk_len * 2 + sizeof(telemetry_json) + 1024;
    char *json_resp = (char *)malloc(json_cap);
    if (!json_resp)
    {
        free(chunk);
        api_send_json(client_fd, 500, "{\"error\":\"Memory allocation failed\"}");
        return;
    }

    size_t jlen = (size_t)snprintf(
        json_resp, json_cap,
        "{\n"
        "  \"job_id\": \"%s\",\n"
        "  \"status\": \"%s\",\n"
        "  \"active\": %s,\n"
        "  \"exit_code\": %d,\n"
        "  \"offset\": %ld,\n"
        "  \"output\": \"",
        job->id,
        job->active ? "running" : (job->exit_code == 0 ? "completed" : "failed"),
        job->active ? "true" : "false",
        job->exit_code,
        next_offset);

    if (chunk)
    {
        for (size_t i = 0; i < chunk_len && jlen + 8 < json_cap; i++)
        {
            unsigned char c = (unsigned char)chunk[i];
            if (c == '"') { json_resp[jlen++] = '\\'; json_resp[jlen++] = '"'; }
            else if (c == '\\') { json_resp[jlen++] = '\\'; json_resp[jlen++] = '\\'; }
            else if (c == '\n') { json_resp[jlen++] = '\\'; json_resp[jlen++] = 'n'; }
            else if (c == '\r') { json_resp[jlen++] = '\\'; json_resp[jlen++] = 'r'; }
            else if (c == '\t') { json_resp[jlen++] = '\\'; json_resp[jlen++] = 't'; }
            else if (c < 32) { /* skip non-printable */ }
            else { json_resp[jlen++] = (char)c; }
        }
    }
    json_resp[jlen++] = '"';

    if (has_telemetry)
    {
        jlen += (size_t)snprintf(json_resp + jlen, json_cap - jlen, ",\n%s\n", telemetry_json);
    }
    else
    {
        json_resp[jlen++] = '\n';
    }

    json_resp[jlen++] = '}';
    json_resp[jlen] = '\0';

    api_send_json(client_fd, 200, json_resp);
    free(chunk);
    free(json_resp);
} // handle_api_cli_status

void handle_api_cli_kill(
    int         client_fd,
    const char *body)
{
    char job_id[64] = {0};
    const char *id_key = "\"job_id\":";
    const char *pos = strstr(body, id_key);
    if (pos)
    {
        pos += strlen(id_key);
        while (*pos == ' ' || *pos == '"') pos++;
        const char *end = strchr(pos, '"');
        if (end)
        {
            size_t len = (size_t)(end - pos);
            if (len < sizeof(job_id))
            {
                memcpy(job_id, pos, len);
                job_id[len] = '\0';
            }
        }
    }

    if (job_id[0] == '\0')
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing job_id field\"}");
        return;
    }

    for (int ii = 0; ii < MAX_JOBS; ii++)
    {
        if (s_jobs[ii].active && s_jobs[ii].pid > 0 &&
            strcmp(s_jobs[ii].id, job_id) == 0)
        {
            if (s_jobs[ii].streamer_pid > 0)
            {
                kill(s_jobs[ii].streamer_pid, SIGTERM);
                usleep(20000);
                kill(s_jobs[ii].streamer_pid, SIGKILL);
                s_jobs[ii].streamer_pid = 0;
            }
            if (s_jobs[ii].stream_name[0] != '\0')
            {
                char shm_im[PATH_MAX];
                char shm_sem[PATH_MAX];
                snprintf(shm_im, sizeof(shm_im), "/dev/shm/%s.im.shm", s_jobs[ii].stream_name);
                snprintf(shm_sem, sizeof(shm_sem), "/dev/shm/%s.sem.shm", s_jobs[ii].stream_name);
                unlink(shm_im);
                unlink(shm_sem);
                s_jobs[ii].stream_name[0] = '\0';
            }
            kill(s_jobs[ii].pid, SIGTERM);
            usleep(100000);
            kill(s_jobs[ii].pid, SIGKILL);
            int sys_ret = system("tmux kill-session -t gric_cli 2>/dev/null");
            (void)sys_ret;
            s_jobs[ii].active = 0;
            s_jobs[ii].finished = 1;
            s_jobs[ii].exit_code = 137;
            api_send_json(client_fd, 200, "{\"status\":\"ok\",\"killed\":true}");
            return;
        }
    }

    api_send_json(client_fd, 404, "{\"error\":\"Active job not found\"}");
} // handle_api_cli_kill

void server_jobs_cleanup(
    void)
{
    stop_tmux_session();
    for (int ii = 0; ii < MAX_JOBS; ii++)
    {
        if (s_jobs[ii].streamer_pid > 0)
        {
            kill(s_jobs[ii].streamer_pid, SIGTERM);
            usleep(20000);
            kill(s_jobs[ii].streamer_pid, SIGKILL);
            s_jobs[ii].streamer_pid = 0;
        }
        if (s_jobs[ii].stream_name[0] != '\0')
        {
            char shm_im[PATH_MAX];
            char shm_sem[PATH_MAX];
            snprintf(shm_im, sizeof(shm_im), "/dev/shm/%s.im.shm", s_jobs[ii].stream_name);
            snprintf(shm_sem, sizeof(shm_sem), "/dev/shm/%s.sem.shm", s_jobs[ii].stream_name);
            unlink(shm_im);
            unlink(shm_sem);
            s_jobs[ii].stream_name[0] = '\0';
        }
        if (s_jobs[ii].active && s_jobs[ii].pid > 0)
        {
            kill(s_jobs[ii].pid, SIGTERM);
            s_jobs[ii].active = 0;
        }
        if (s_jobs[ii].log_path[0] != '\0')
        {
            unlink(s_jobs[ii].log_path);
        }
        if (s_jobs[ii].shm_path[0] != '\0')
        {
            unlink(s_jobs[ii].shm_path);
        }
    }
} // server_jobs_cleanup

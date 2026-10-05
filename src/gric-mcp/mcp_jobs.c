/**
 * @file mcp_jobs.c
 * @brief Async background job execution, monitoring, and lifecycle management for gric-mcp.
 */

#include "mcp_jobs.h"
#include "shared/cjson/cJSON.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MCP_JOBS_BASE_DIR "/tmp/gric_jobs"
#define LOG_READ_BUFFER_SIZE 8192

static int g_job_counter = 0;

int mcp_jobs_init(void)
{
    if (access(MCP_JOBS_BASE_DIR, F_OK) != 0)
    {
        if (mkdir(MCP_JOBS_BASE_DIR, 0755) != 0 && errno != EEXIST)
        {
            return -1;
        }
    }
    return 0;
} // mcp_jobs_init

const char *mcp_job_status_to_string(
    enum mcp_job_status status)
{
    switch (status)
    {
        case MCP_JOB_STATUS_RUNNING:
            return "RUNNING";
        case MCP_JOB_STATUS_COMPLETED:
            return "COMPLETED";
        case MCP_JOB_STATUS_FAILED:
            return "FAILED";
        case MCP_JOB_STATUS_CANCELLED:
            return "CANCELLED";
        default:
            return "UNKNOWN";
    }
} // mcp_job_status_to_string

static enum mcp_job_status string_to_job_status(
    const char *str)
{
    if (str == NULL)
    {
        return MCP_JOB_STATUS_UNKNOWN;
    }
    if (strcmp(str, "RUNNING") == 0)
    {
        return MCP_JOB_STATUS_RUNNING;
    }
    if (strcmp(str, "COMPLETED") == 0)
    {
        return MCP_JOB_STATUS_COMPLETED;
    }
    if (strcmp(str, "FAILED") == 0)
    {
        return MCP_JOB_STATUS_FAILED;
    }
    if (strcmp(str, "CANCELLED") == 0)
    {
        return MCP_JOB_STATUS_CANCELLED;
    }
    return MCP_JOB_STATUS_UNKNOWN;
} // string_to_job_status

static void write_job_meta(
    const struct mcp_job_info *info)
{
    char meta_path[1060];
    snprintf(meta_path, sizeof(meta_path), "%s/%s/job.meta", MCP_JOBS_BASE_DIR, info->job_id);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "job_id", info->job_id);
    cJSON_AddNumberToObject(root, "pid", (double)info->pid);
    cJSON_AddStringToObject(root, "status", mcp_job_status_to_string(info->status));
    cJSON_AddNumberToObject(root, "exit_code", (double)info->exit_code);
    cJSON_AddStringToObject(root, "program", info->program);
    cJSON_AddStringToObject(root, "log_path", info->log_path);
    cJSON_AddStringToObject(root, "outdir", info->outdir);
    cJSON_AddNumberToObject(root, "start_time_sec", info->start_time_sec);
    cJSON_AddNumberToObject(root, "end_time_sec", info->end_time_sec);

    char *json_str = cJSON_Print(root);
    if (json_str != NULL)
    {
        FILE *fp = fopen(meta_path, "w");
        if (fp != NULL)
        {
            fputs(json_str, fp);
            fclose(fp);
        }
        free(json_str);
    }
    cJSON_Delete(root);
} // write_job_meta

static int read_job_meta(
    const char          *job_id,
    struct mcp_job_info *info)
{
    char meta_path[1060];
    snprintf(meta_path, sizeof(meta_path), "%s/%s/job.meta", MCP_JOBS_BASE_DIR, job_id);

    FILE *fp = fopen(meta_path, "r");
    if (fp == NULL)
    {
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (sz <= 0 || sz > 65536)
    {
        fclose(fp);
        return -1;
    }

    char *buf = malloc((size_t)sz + 1);
    if (buf == NULL)
    {
        fclose(fp);
        return -1;
    }

    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);

    if (root == NULL)
    {
        return -1;
    }

    memset(info, 0, sizeof(*info));
    snprintf(info->job_id, sizeof(info->job_id), "%s", job_id);

    cJSON *p_pid = cJSON_GetObjectItem(root, "pid");
    if (p_pid != NULL)
    {
        info->pid = (pid_t)p_pid->valueint;
    }

    cJSON *p_stat = cJSON_GetObjectItem(root, "status");
    if (p_stat != NULL && cJSON_IsString(p_stat))
    {
        info->status = string_to_job_status(p_stat->valuestring);
    }

    cJSON *p_code = cJSON_GetObjectItem(root, "exit_code");
    if (p_code != NULL)
    {
        info->exit_code = p_code->valueint;
    }

    cJSON *p_prog = cJSON_GetObjectItem(root, "program");
    if (p_prog != NULL && cJSON_IsString(p_prog))
    {
        snprintf(info->program, sizeof(info->program), "%s", p_prog->valuestring);
    }

    cJSON *p_log = cJSON_GetObjectItem(root, "log_path");
    if (p_log != NULL && cJSON_IsString(p_log))
    {
        snprintf(info->log_path, sizeof(info->log_path), "%s", p_log->valuestring);
    }

    cJSON *p_out = cJSON_GetObjectItem(root, "outdir");
    if (p_out != NULL && cJSON_IsString(p_out))
    {
        snprintf(info->outdir, sizeof(info->outdir), "%s", p_out->valuestring);
    }

    cJSON *p_tstart = cJSON_GetObjectItem(root, "start_time_sec");
    if (p_tstart != NULL)
    {
        info->start_time_sec = p_tstart->valuedouble;
    }

    cJSON *p_tend = cJSON_GetObjectItem(root, "end_time_sec");
    if (p_tend != NULL)
    {
        info->end_time_sec = p_tend->valuedouble;
    }

    cJSON_Delete(root);
    return 0;
} // read_job_meta

static void parse_log_progress_tail(
    struct mcp_job_info *info)
{
    FILE *fp = fopen(info->log_path, "r");
    if (fp == NULL)
    {
        return;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);

    long read_size = (file_size < LOG_READ_BUFFER_SIZE) ? file_size : LOG_READ_BUFFER_SIZE;
    fseek(fp, file_size - read_size, SEEK_SET);

    char *buf = malloc((size_t)read_size + 1);
    if (buf == NULL)
    {
        fclose(fp);
        return;
    }

    size_t rd = fread(buf, 1, (size_t)read_size, fp);
    fclose(fp);
    buf[rd] = '\0';

    /* Parse progress patterns: "Processing frame 10 / 500 (Clusters: 9, Dists: 37, ...)" */
    const char *last_proc = NULL;
    const char *p = buf;
    while ((p = strstr(p, "Processing frame ")) != NULL)
    {
        last_proc = p;
        p += 17;
    }

    if (last_proc != NULL)
    {
        long cur = 0;
        long tot = 0;
        int cls = 0;
        long dst = 0;
        long prn = 0;

        if (sscanf(last_proc, "Processing frame %ld / %ld (Clusters: %d, Dists: %ld",
                   &cur, &tot, &cls, &dst) >= 2)
        {
            info->progress.current_frame = cur;
            info->progress.total_frames = tot;
            info->progress.clusters = cls;
            info->progress.dists = dst;

            const char *p_pruned = strstr(last_proc, "Pruned: ");
            if (p_pruned != NULL)
            {
                sscanf(p_pruned + 8, "%ld", &prn);
                info->progress.pruned = prn;
            }

            if (tot > 0)
            {
                info->progress.progress_pct = ((double)cur / (double)tot) * 100.0;
            }
        }
    }

    if (strstr(buf, "All frames clustered.") != NULL ||
        strstr(buf, "Analysis complete.") != NULL)
    {
        info->progress.progress_pct = 100.0;
        if (info->progress.total_frames > 0)
        {
            info->progress.current_frame = info->progress.total_frames;
        }
    }

    /* Extract log tail (last 500 chars / lines) */
    size_t tail_len = (rd < sizeof(info->log_tail) - 1) ? rd : sizeof(info->log_tail) - 1;
    memcpy(info->log_tail, buf + (rd - tail_len), tail_len);
    info->log_tail[tail_len] = '\0';

    free(buf);
} // parse_log_progress_tail

int mcp_job_spawn(
    const char        *program,
    const char *const  argv[],
    const char        *output_dir,
    char              *out_job_id,
    size_t             job_id_size)
{
    if (mcp_jobs_init() != 0)
    {
        return -1;
    }

    g_job_counter++;
    snprintf(
        out_job_id, job_id_size,
        "job_%ld_%d_%d",
        (long)time(NULL), (int)getpid(), g_job_counter);

    char job_dir[256];
    snprintf(job_dir, sizeof(job_dir), "%s/%s", MCP_JOBS_BASE_DIR, out_job_id);
    if (mkdir(job_dir, 0755) != 0)
    {
        return -1;
    }

    char log_path[1024];
    snprintf(log_path, sizeof(log_path), "%s/job.log", job_dir);

    int log_fd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd < 0)
    {
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        close(log_fd);
        return -1;
    }

    if (pid == 0)
    {
        /* Child process */
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            close(devnull);
        }

        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        close(log_fd);

        execvp(argv[0], (char *const *)argv);

        /* execvp failed */
        perror("execvp");
        _exit(127);
    }

    /* Parent process */
    close(log_fd);

    struct mcp_job_info info;
    memset(&info, 0, sizeof(info));
    snprintf(info.job_id, sizeof(info.job_id), "%s", out_job_id);
    info.pid = pid;
    info.status = MCP_JOB_STATUS_RUNNING;
    info.exit_code = 0;
    snprintf(info.program, sizeof(info.program), "%s", (program != NULL) ? program : "unknown");
    snprintf(info.log_path, sizeof(info.log_path), "%s", log_path);
    if (output_dir != NULL)
    {
        snprintf(info.outdir, sizeof(info.outdir), "%s", output_dir);
    }
    info.start_time_sec = (double)time(NULL);
    info.end_time_sec = 0.0;

    write_job_meta(&info);
    return 0;
} // mcp_job_spawn

int mcp_job_get_info(
    const char          *job_id,
    struct mcp_job_info *info)
{
    if (read_job_meta(job_id, info) != 0)
    {
        return -1;
    }

    if (info->status == MCP_JOB_STATUS_RUNNING)
    {
        int status = 0;
        pid_t ret = waitpid(info->pid, &status, WNOHANG);

        if (ret == info->pid)
        {
            info->end_time_sec = (double)time(NULL);
            if (WIFEXITED(status))
            {
                info->exit_code = WEXITSTATUS(status);
                info->status = (info->exit_code == 0) ?
                    MCP_JOB_STATUS_COMPLETED : MCP_JOB_STATUS_FAILED;
            }
            else if (WIFSIGNALED(status))
            {
                info->exit_code = 128 + WTERMSIG(status);
                info->status = MCP_JOB_STATUS_CANCELLED;
            }
            else
            {
                info->status = MCP_JOB_STATUS_FAILED;
            }
            write_job_meta(info);
        }
        else if (ret == 0)
        {
            /* Still running */
            if (kill(info->pid, 0) == -1 && errno == ESRCH)
            {
                info->status = MCP_JOB_STATUS_FAILED;
                info->end_time_sec = (double)time(NULL);
                write_job_meta(info);
            }
        }
    }

    double now = (double)time(NULL);
    double end = (info->end_time_sec > 0.0) ? info->end_time_sec : now;
    info->progress.elapsed_sec = (end >= info->start_time_sec) ? (end - info->start_time_sec) : 0.0;

    parse_log_progress_tail(info);
    return 0;
} // mcp_job_get_info

int mcp_job_cancel(
    const char *job_id,
    int         clean_output)
{
    struct mcp_job_info info;
    if (mcp_job_get_info(job_id, &info) != 0)
    {
        return -1;
    }

    if (info.status == MCP_JOB_STATUS_RUNNING)
    {
        kill(info.pid, SIGTERM);

        int status = 0;
        int stopped = 0;
        for (int ii = 0; ii < 5; ii++)
        {
            usleep(100000); /* 100 ms */
            if (waitpid(info.pid, &status, WNOHANG) == info.pid)
            {
                stopped = 1;
                break;
            }
        }

        if (!stopped)
        {
            kill(info.pid, SIGKILL);
            waitpid(info.pid, &status, 0);
        }

        info.status = MCP_JOB_STATUS_CANCELLED;
        info.end_time_sec = (double)time(NULL);
        info.exit_code = 137; /* SIGKILL exit code */
        write_job_meta(&info);
    }

    if (clean_output && info.outdir[0] != '\0' && access(info.outdir, F_OK) == 0)
    {
        /* Safe check on prefix */
        if (strncmp(info.outdir, "/tmp/", 5) == 0)
        {
            pid_t rm_pid = fork();
            if (rm_pid == 0)
            {
                execlp("rm", "rm", "-rf", info.outdir, (char *)NULL);
                _exit(1);
            }
            if (rm_pid > 0)
            {
                int rm_status = 0;
                waitpid(rm_pid, &rm_status, 0);
            }
        }
    }

    return 0;
} // mcp_job_cancel

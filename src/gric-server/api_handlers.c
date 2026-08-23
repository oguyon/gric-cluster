/**
 * @file api_handlers.c
 * @brief REST API endpoint handlers for the GRIC desktop micro-server.
 */

#define _GNU_SOURCE
#include "api_handlers.h"
#include "cluster_shm.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_JOBS 16

typedef struct
{
    char   id[64];
    pid_t  pid;
    int    active;
    int    finished;
    int    exit_code;
    char   log_path[PATH_MAX];
    char   shm_path[PATH_MAX];
    time_t start_time;
} CliJob;

static CliJob s_jobs[MAX_JOBS];
static int    s_job_count = 0;

static void api_send_response(
    int         client_fd,
    int         status_code,
    const char *content_type,
    const char *data,
    size_t      data_len)
{
    char header[1024];
    const char *status_text = "OK";
    if (status_code == 400)
    {
        status_text = "Bad Request";
    }
    else if (status_code == 404)
    {
        status_text = "Not Found";
    }
    else if (status_code == 500)
    {
        status_text = "Internal Server Error";
    }

    int header_len = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text, content_type, data_len);

    if (header_len > 0)
    {
        ssize_t nw = write(client_fd, header, (size_t)header_len);
        (void)nw;
    }
    if (data && data_len > 0)
    {
        ssize_t nw = write(client_fd, data, data_len);
        (void)nw;
    }
}

static void api_send_json(
    int         client_fd,
    int         status_code,
    const char *json_str)
{
    size_t len = json_str ? strlen(json_str) : 0;
    api_send_response(client_fd, status_code, "application/json; charset=utf-8",
                      json_str ? json_str : "{}", len);
}

static void url_decode(
    const char *src,
    char       *dst,
    size_t      dst_size)
{
    size_t d = 0;
    for (size_t s = 0; src[s] != '\0' && d + 1 < dst_size; s++)
    {
        if (src[s] == '%' && src[s + 1] && src[s + 2] &&
            isxdigit((unsigned char)src[s + 1]) && isxdigit((unsigned char)src[s + 2]))
        {
            char hex[3] = { src[s + 1], src[s + 2], '\0' };
            dst[d++] = (char)strtol(hex, NULL, 16);
            s += 2;
        }
        else if (src[s] == '+')
        {
            dst[d++] = ' ';
        }
        else
        {
            dst[d++] = src[s];
        }
    }
    dst[d] = '\0';
}

static int get_query_param(
    const char *query,
    const char *key,
    char       *out_val,
    size_t      out_size)
{
    if (!query || !key || !out_val || out_size == 0)
    {
        return 0;
    }

    size_t key_len = strlen(key);
    const char *cur = query;

    while (*cur)
    {
        if ((cur == query || *(cur - 1) == '&') &&
            strncmp(cur, key, key_len) == 0 && cur[key_len] == '=')
        {
            const char *val_start = cur + key_len + 1;
            const char *val_end = strchr(val_start, '&');
            size_t val_len = val_end ? (size_t)(val_end - val_start) : strlen(val_start);

            char tmp[1024];
            if (val_len >= sizeof(tmp))
            {
                val_len = sizeof(tmp) - 1;
            }
            memcpy(tmp, val_start, val_len);
            tmp[val_len] = '\0';

            url_decode(tmp, out_val, out_size);
            return 1;
        }
        cur++;
    }
    return 0;
}

static int sanitize_path(
    const char *base_dir,
    const char *rel_path,
    char       *out_full_path,
    size_t      out_size)
{
    if (!base_dir || !rel_path || !out_full_path || out_size == 0)
    {
        return 0;
    }

    /* Reject leading slashes or parent traversal */
    if (rel_path[0] == '/' || strstr(rel_path, "..") != NULL)
    {
        return 0;
    }

    int written = snprintf(out_full_path, out_size, "%s/%s", base_dir, rel_path);
    return (written > 0 && (size_t)written < out_size);
}

static int check_binary_available(
    const ServerConfig *config,
    const char         *bin_name,
    char               *out_path,
    size_t              out_size)
{
    /* 1. Check in server_bin_dir */
    if (config->server_bin_dir[0] != '\0')
    {
        int w = snprintf(out_path, out_size, "%s/%s", config->server_bin_dir, bin_name);
        if (w > 0 && (size_t)w < out_size && access(out_path, X_OK) == 0)
        {
            return 1;
        }
    }

    /* 2. Check in current working directory */
    int w = snprintf(out_path, out_size, "%s/%s", config->workdir, bin_name);
    if (w > 0 && (size_t)w < out_size && access(out_path, X_OK) == 0)
    {
        return 1;
    }

    /* 3. Check in PATH */
    const char *path_env = getenv("PATH");
    if (path_env)
    {
        char path_copy[4096];
        strncpy(path_copy, path_env, sizeof(path_copy) - 1);
        path_copy[sizeof(path_copy) - 1] = '\0';

        char *saveptr = NULL;
        char *dir = strtok_r(path_copy, ":", &saveptr);
        while (dir)
        {
            int pw = snprintf(out_path, out_size, "%s/%s", dir, bin_name);
            if (pw > 0 && (size_t)pw < out_size && access(out_path, X_OK) == 0)
            {
                return 1;
            }
            dir = strtok_r(NULL, ":", &saveptr);
        }
    }

    out_path[0] = '\0';
    return 0;
}

static void handle_api_info(
    int                 client_fd,
    const ServerConfig *config)
{
    char cluster_bin[PATH_MAX];
    char knn_bin[PATH_MAX];
    char status_bin[PATH_MAX];

    int has_cluster = check_binary_available(config, "gric-cluster",
                                             cluster_bin, sizeof(cluster_bin));
    int has_knn = check_binary_available(config, "gric-knn",
                                         knn_bin, sizeof(knn_bin));
    int has_status = check_binary_available(config, "gric-status",
                                            status_bin, sizeof(status_bin));

    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus < 1)
    {
        ncpus = 1;
    }

    char resp[PATH_MAX + 1024];
    snprintf(resp, sizeof(resp),
             "{\n"
             "  \"status\": \"ok\",\n"
             "  \"mode\": \"desktop\",\n"
             "  \"cwd\": \"%s\",\n"
             "  \"cpus\": %ld,\n"
             "  \"binaries\": {\n"
             "    \"gric-cluster\": %s,\n"
             "    \"gric-knn\": %s,\n"
             "    \"gric-status\": %s\n"
             "  }\n"
             "}\n",
             config->workdir, ncpus,
             has_cluster ? "true" : "false",
             has_knn ? "true" : "false",
             has_status ? "true" : "false");

    api_send_json(client_fd, 200, resp);
}

static void handle_api_files(
    int                 client_fd,
    const ServerConfig *config)
{
    DIR *dir = opendir(config->workdir);
    if (!dir)
    {
        api_send_json(client_fd, 500, "{\"error\":\"Failed to open working directory\"}");
        return;
    }

    size_t buf_cap = 65536;
    char *buf = (char *)malloc(buf_cap);
    if (!buf)
    {
        closedir(dir);
        api_send_json(client_fd, 500, "{\"error\":\"Out of memory\"}");
        return;
    }

    size_t buf_len = 0;
    buf_len += (size_t)snprintf(buf + buf_len, buf_cap - buf_len, "{\"files\":[");

    struct dirent *entry = NULL;
    int first = 1;

    while ((entry = readdir(dir)) != NULL)
    {
        /* Skip dot files except clusterdat folders */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }

        char full_path[PATH_MAX + 256];
        int pw = snprintf(full_path, sizeof(full_path), "%s/%s", config->workdir, entry->d_name);
        if (pw <= 0 || (size_t)pw >= sizeof(full_path))
        {
            continue;
        }

        struct stat st;
        if (stat(full_path, &st) != 0)
        {
            continue;
        }

        int is_dir = S_ISDIR(st.st_mode);
        const char *dot = strrchr(entry->d_name, '.');
        const char *ext = dot ? dot + 1 : "";

        /* Filter relevant files: txt, csv, fits, dat, log, clusterdat */
        int is_relevant = 0;
        if (is_dir && (strstr(entry->d_name, "clusterdat") ||
                       strstr(entry->d_name, "cluster.out")))
        {
            is_relevant = 1;
        }
        else if (!is_dir)
        {
            if (strcasecmp(ext, "txt") == 0 || strcasecmp(ext, "csv") == 0 ||
                strcasecmp(ext, "dat") == 0 || strcasecmp(ext, "fits") == 0 ||
                strcasecmp(ext, "log") == 0)
            {
                is_relevant = 1;
            }
        }

        if (!is_relevant)
        {
            continue;
        }

        char item[1024];
        int item_len = snprintf(item, sizeof(item),
                                "%s{\"name\":\"%s\",\"size\":%ld,\"is_dir\":%s,\"ext\":\"%s\"}",
                                first ? "" : ",",
                                entry->d_name,
                                (long)st.st_size,
                                is_dir ? "true" : "false",
                                ext);
        first = 0;

        if (buf_len + (size_t)item_len + 16 >= buf_cap)
        {
            buf_cap *= 2;
            char *new_buf = (char *)realloc(buf, buf_cap);
            if (!new_buf)
            {
                break;
            }
            buf = new_buf;
        }

        memcpy(buf + buf_len, item, (size_t)item_len);
        buf_len += (size_t)item_len;
    } // while (readdir)

    closedir(dir);

    snprintf(buf + buf_len, buf_cap - buf_len, "]}");
    api_send_json(client_fd, 200, buf);
    free(buf);
}

static void handle_api_file_read(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query)
{
    char rel_path[PATH_MAX];
    if (!get_query_param(query, "path", rel_path, sizeof(rel_path)))
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing 'path' query parameter\"}");
        return;
    }

    char full_path[PATH_MAX];
    if (!sanitize_path(config->workdir, rel_path, full_path, sizeof(full_path)))
    {
        api_send_json(client_fd, 400, "{\"error\":\"Invalid file path\"}");
        return;
    }

    FILE *f = fopen(full_path, "rb");
    if (!f)
    {
        api_send_json(client_fd, 404, "{\"error\":\"File not found\"}");
        return;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < 0 || fsize > 100 * 1024 * 1024)
    {
        fclose(f);
        api_send_json(client_fd, 400, "{\"error\":\"File too large or invalid\"}");
        return;
    }

    char *content = (char *)malloc((size_t)fsize + 1);
    if (!content)
    {
        fclose(f);
        api_send_json(client_fd, 500, "{\"error\":\"Memory allocation failed\"}");
        return;
    }

    size_t read_bytes = fread(content, 1, (size_t)fsize, f);
    fclose(f);
    content[read_bytes] = '\0';

    api_send_response(client_fd, 200, "text/plain; charset=utf-8",
                      content, read_bytes);
    free(content);
}

static void handle_api_file_write(
    int                 client_fd,
    const ServerConfig *config,
    const char         *body,
    size_t              body_len)
{
    if (!body || body_len == 0)
    {
        api_send_json(client_fd, 400, "{\"error\":\"Empty write payload\"}");
        return;
    }

    /* Extract path from simple JSON: {"path":"foo","content":"bar"} */
    const char *path_key = "\"path\":";
    const char *p_start = strstr(body, path_key);
    if (!p_start)
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing 'path' field in JSON\"}");
        return;
    }
    p_start += strlen(path_key);
    while (*p_start == ' ' || *p_start == '"')
    {
        p_start++;
    }
    const char *p_end = strchr(p_start, '"');
    if (!p_end)
    {
        api_send_json(client_fd, 400, "{\"error\":\"Malformed 'path' in JSON\"}");
        return;
    }

    char rel_path[PATH_MAX];
    size_t path_len = (size_t)(p_end - p_start);
    if (path_len >= sizeof(rel_path))
    {
        path_len = sizeof(rel_path) - 1;
    }
    memcpy(rel_path, p_start, path_len);
    rel_path[path_len] = '\0';

    const char *content_key = "\"content\":";
    const char *c_start = strstr(p_end, content_key);
    if (!c_start)
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing 'content' field in JSON\"}");
        return;
    }
    c_start += strlen(content_key);
    while (*c_start == ' ')
    {
        c_start++;
    }

    char full_path[PATH_MAX];
    if (!sanitize_path(config->workdir, rel_path, full_path, sizeof(full_path)))
    {
        api_send_json(client_fd, 400, "{\"error\":\"Invalid target file path\"}");
        return;
    }

    /* Ensure parent directories exist */
    char dir_copy[PATH_MAX];
    strncpy(dir_copy, full_path, sizeof(dir_copy) - 1);
    dir_copy[sizeof(dir_copy) - 1] = '\0';
    char *slash = strrchr(dir_copy, '/');
    if (slash)
    {
        *slash = '\0';
        char cmd[PATH_MAX + 32];
        snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", dir_copy);
        int ret = system(cmd);
        (void)ret;
    }

    FILE *f = fopen(full_path, "wb");
    if (!f)
    {
        api_send_json(client_fd, 500, "{\"error\":\"Failed to create file for writing\"}");
        return;
    }

    if (*c_start == '"')
    {
        c_start++;
        const char *c_end = body + body_len - 1;
        while (c_end > c_start && (*c_end == ' ' || *c_end == '}' ||
                                   *c_end == '\n' || *c_end == '\r' || *c_end == '"'))
        {
            if (*c_end == '"')
            {
                break;
            }
            c_end--;
        }
        for (const char *p = c_start; p < c_end; p++)
        {
            if (*p == '\\' && (p + 1 < c_end))
            {
                if (*(p + 1) == 'n') { fputc('\n', f); p++; }
                else if (*(p + 1) == 't') { fputc('\t', f); p++; }
                else if (*(p + 1) == 'r') { fputc('\r', f); p++; }
                else if (*(p + 1) == '"') { fputc('"', f); p++; }
                else if (*(p + 1) == '\\') { fputc('\\', f); p++; }
                else { fputc(*p, f); }
            }
            else
            {
                fputc(*p, f);
            }
        }
    }
    else
    {
        size_t rem = body_len - (size_t)(c_start - body);
        fwrite(c_start, 1, rem, f);
    }
    fclose(f);

    api_send_json(client_fd, 200, "{\"status\":\"ok\",\"written\":true}");
}

/**
 * @brief Ensure that the persistent 'gric_cli' tmux session is active.
 *
 * @param config Server configuration containing workspace directory.
 * @return 1 if session is active or successfully created, 0 otherwise.
 */
static int ensure_tmux_session(
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
}

/**
 * @brief Terminate the persistent 'gric_cli' tmux session.
 */
static void stop_tmux_session(
    void)
{
    int ret = system("tmux kill-session -t gric_cli 2>/dev/null");
    (void)ret;
}

static void handle_api_cli_session_init(
    int                 client_fd,
    const ServerConfig *config)
{
    int ok = ensure_tmux_session(config);
    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"session\":\"gric_cli\",\"active\":%s}",
             ok ? "true" : "false");
    api_send_json(client_fd, 200, resp);
}

static void handle_api_cli_session_stop(
    int client_fd)
{
    stop_tmux_session();
    api_send_json(client_fd, 200,
                  "{\"status\":\"ok\",\"session\":\"gric_cli\",\"active\":false}");
}

static void handle_api_cli_session_status(
    int client_fd)
{
    int exists = (system("tmux has-session -t gric_cli 2>/dev/null") == 0);
    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"session\":\"gric_cli\",\"exists\":%s}",
             exists ? "true" : "false");
    api_send_json(client_fd, 200, resp);
}

static void handle_api_cli_run(
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

    /* Extract command name (default: gric-cluster) */
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

    /* Parse args array from JSON: "args": ["arg1", "arg2", ...] */
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

    /* Allocate slot in jobs table */
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

    /* For gric-cluster, attach a shared-memory telemetry file */
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

    /* Ensure persistent gric_cli tmux session is active */
    ensure_tmux_session(config);

    char done_path[PATH_MAX];
    snprintf(done_path, sizeof(done_path), "/tmp/gric_%s.done", job->id);
    unlink(done_path);

    /* Build command line string */
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
        for (int i = 0; i < argc; i++) free(argv[i]);
        api_send_json(client_fd, 500, "{\"error\":\"fork() failed\"}");
        return;
    }

    if (pid == 0)
    {
        /* Child Process: Dispatch command into persistent gric_cli tmux session */
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

        /* Wait for completion sentinel */
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

    /* Parent Process */
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
}

/**
 * @brief Format a GricClusterShmStatus struct into a JSON telemetry block.
 */
static int format_shm_telemetry_json(
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
}

static void handle_api_cli_status(
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

    /* Check child process status if active */
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
        }
    }

    /* Read new output from log file starting from offset */
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

    /* Read SHM telemetry if available */
    char telemetry_json[4096] = "";
    int has_telemetry = 0;
    if (job->shm_path[0] != '\0')
    {
        has_telemetry = format_shm_telemetry_json(
            job->shm_path, telemetry_json, sizeof(telemetry_json));
    }

    /* Build JSON with escaped output and telemetry */
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
}

static void handle_api_cli_kill(
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
}

/**
 * @brief List all active ImageStreamIO shared memory streams in /dev/shm.
 */
static void handle_api_shm_list(
    int client_fd)
{
    DIR *d = opendir("/dev/shm");
    if (!d)
    {
        api_send_json(client_fd, 200, "{\"streams\":[]}");
        return;
    }

    size_t cap = 8192;
    char *json = (char *)malloc(cap);
    if (!json)
    {
        closedir(d);
        api_send_json(client_fd, 500, "{\"error\":\"Memory allocation failed\"}");
        return;
    }

    size_t len = (size_t)snprintf(json, cap, "{\"streams\":[");
    int first = 1;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (ent->d_name[0] == '.')
        {
            continue;
        }

        const char *dot = strstr(ent->d_name, ".im.shm");
        if (dot != NULL)
        {
            char stream_name[256];
            size_t name_len = (size_t)(dot - ent->d_name);
            if (name_len >= sizeof(stream_name))
            {
                name_len = sizeof(stream_name) - 1;
            }
            memcpy(stream_name, ent->d_name, name_len);
            stream_name[name_len] = '\0';

            char fullpath[PATH_MAX];
            snprintf(fullpath, sizeof(fullpath), "/dev/shm/%s", ent->d_name);
            struct stat st;
            uint64_t fsz = 0;
            if (stat(fullpath, &st) == 0)
            {
                fsz = (uint64_t)st.st_size;
            }

            char entry[512];
            int elen = snprintf(
                entry, sizeof(entry),
                "%s{\"name\":\"%s\",\"type\":\"ImageStreamIO\",\"file\":\"%s\",\"size\":%llu}",
                first ? "" : ",",
                stream_name,
                ent->d_name,
                (unsigned long long)fsz);
            first = 0;

            if (len + (size_t)elen + 16 < cap)
            {
                memcpy(json + len, entry, (size_t)elen);
                len += (size_t)elen;
            }
        }
    }
    closedir(d);

    if (len + 4 < cap)
    {
        json[len++] = ']';
        json[len++] = '}';
        json[len] = '\0';
    }

    api_send_json(client_fd, 200, json);
    free(json);
}

/**
 * @brief Query live telemetry directly from any specified SHM file path.
 */
static void handle_api_shm_telemetry(
    int         client_fd,
    const char *query)
{
    char path[PATH_MAX] = "";
    if (!get_query_param(query, "path", path, sizeof(path)))
    {
        api_send_json(client_fd, 400, "{\"error\":\"Missing path parameter\"}");
        return;
    }

    char telemetry_json[4096] = "";
    if (!format_shm_telemetry_json(path, telemetry_json, sizeof(telemetry_json)))
    {
        api_send_json(client_fd, 404, "{\"error\":\"Invalid or unreadable SHM status file\"}");
        return;
    }

    char resp[5000];
    snprintf(resp, sizeof(resp), "{\n  \"status\": \"ok\",\n%s\n}", telemetry_json);
    api_send_json(client_fd, 200, resp);
}

int handle_api_request(
    int                 client_fd,
    const ServerConfig *config,
    const char         *method,
    const char         *path,
    const char         *query,
    const char         *body,
    size_t              body_len)
{
    /* Handle CORS Pre-flight */
    if (strcmp(method, "OPTIONS") == 0)
    {
        api_send_response(client_fd, 204, "text/plain", NULL, 0);
        return 1;
    }

    if (strcmp(path, "/api/info") == 0 && strcmp(method, "GET") == 0)
    {
        handle_api_info(client_fd, config);
        return 1;
    }
    if ((strcmp(path, "/api/files") == 0 || strcmp(path, "/api/files/list") == 0) &&
        strcmp(method, "GET") == 0)
    {
        handle_api_files(client_fd, config);
        return 1;
    }
    if ((strcmp(path, "/api/file/read") == 0 || strcmp(path, "/api/files/read") == 0) &&
        strcmp(method, "GET") == 0)
    {
        handle_api_file_read(client_fd, config, query);
        return 1;
    }
    if ((strcmp(path, "/api/file/write") == 0 || strcmp(path, "/api/files/write") == 0) &&
        strcmp(method, "POST") == 0)
    {
        handle_api_file_write(client_fd, config, body, body_len);
        return 1;
    }
    if (strcmp(path, "/api/cli/session/init") == 0 &&
        (strcmp(method, "POST") == 0 || strcmp(method, "GET") == 0))
    {
        handle_api_cli_session_init(client_fd, config);
        return 1;
    }
    if (strcmp(path, "/api/cli/session/stop") == 0 &&
        (strcmp(method, "POST") == 0 || strcmp(method, "GET") == 0))
    {
        handle_api_cli_session_stop(client_fd);
        return 1;
    }
    if (strcmp(path, "/api/cli/session/status") == 0 && strcmp(method, "GET") == 0)
    {
        handle_api_cli_session_status(client_fd);
        return 1;
    }
    if (strcmp(path, "/api/cli/run") == 0 && strcmp(method, "POST") == 0)
    {
        handle_api_cli_run(client_fd, config, body, body_len);
        return 1;
    }
    if (strcmp(path, "/api/cli/status") == 0 && strcmp(method, "GET") == 0)
    {
        handle_api_cli_status(client_fd, config, query);
        return 1;
    }
    if (strcmp(path, "/api/cli/kill") == 0 && strcmp(method, "POST") == 0)
    {
        handle_api_cli_kill(client_fd, body);
        return 1;
    }
    if (strcmp(path, "/api/shm/list") == 0 && strcmp(method, "GET") == 0)
    {
        handle_api_shm_list(client_fd);
        return 1;
    }
    if (strcmp(path, "/api/shm/telemetry") == 0 && strcmp(method, "GET") == 0)
    {
        handle_api_shm_telemetry(client_fd, query);
        return 1;
    }

    return 0;
}

void api_handlers_cleanup(
    void)
{
    stop_tmux_session();
    for (int ii = 0; ii < MAX_JOBS; ii++)
    {
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
}

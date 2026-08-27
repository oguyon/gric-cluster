/**
 * @file server_fs_api.c
 * @brief File system and shared memory listing API handlers.
 */

#define _GNU_SOURCE
#include "server_fs_api.h"
#include "server_http_utils.h"
#include "server_job_mgmt.h"
#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void handle_api_info(
    int                 client_fd,
    const ServerConfig *config)
{
    char cluster_bin[PATH_MAX];
    char knn_bin[PATH_MAX];
    char status_bin[PATH_MAX];
    char txt2stream_bin[PATH_MAX];

    int has_cluster = check_binary_available(config, "gric-cluster",
                                             cluster_bin, sizeof(cluster_bin));
    int has_knn = check_binary_available(config, "gric-knn",
                                         knn_bin, sizeof(knn_bin));
    int has_status = check_binary_available(config, "gric-status",
                                            status_bin, sizeof(status_bin));
    int has_txt2stream = check_binary_available(config, "gric-txt2stream",
                                                txt2stream_bin, sizeof(txt2stream_bin));

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
             "    \"gric-status\": %s,\n"
             "    \"gric-txt2stream\": %s\n"
             "  }\n"
             "}\n",
             config->workdir, ncpus,
             has_cluster ? "true" : "false",
             has_knn ? "true" : "false",
             has_status ? "true" : "false",
             has_txt2stream ? "true" : "false");

    api_send_json(client_fd, 200, resp);
} // handle_api_info

void handle_api_files(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query)
{
    char sub_dir[PATH_MAX] = "";
    char target_dir[PATH_MAX];

    if (query && get_query_param(query, "dir", sub_dir, sizeof(sub_dir)) && sub_dir[0] != '\0')
    {
        if (!sanitize_path(config->workdir, sub_dir, target_dir, sizeof(target_dir)))
        {
            api_send_json(client_fd, 400, "{\"error\":\"Invalid directory path\"}");
            return;
        }
    }
    else
    {
        snprintf(target_dir, sizeof(target_dir), "%s", config->workdir);
    }

    DIR *dir = opendir(target_dir);
    if (!dir)
    {
        api_send_json(client_fd, 404, "{\"error\":\"Failed to open directory\"}");
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
    buf_len += (size_t)snprintf(buf + buf_len, buf_cap - buf_len,
                                "{\"dir\":\"%s\",\"files\":[", sub_dir);

    struct dirent *entry = NULL;
    int first = 1;

    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }

        char full_path[PATH_MAX + 256];
        int pw = snprintf(full_path, sizeof(full_path), "%s/%s", target_dir, entry->d_name);
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

        int is_relevant = 0;
        if (is_dir)
        {
            is_relevant = 1;
        }
        else
        {
            if (strcasecmp(ext, "txt") == 0 || strcasecmp(ext, "csv") == 0 ||
                strcasecmp(ext, "dat") == 0 || strcasecmp(ext, "fits") == 0 ||
                strcasecmp(ext, "log") == 0 || strcasecmp(ext, "json") == 0)
            {
                is_relevant = 1;
            }
        }

        if (!is_relevant)
        {
            continue;
        }

        char item[1024];
        int item_len = snprintf(
            item, sizeof(item),
            "%s{\"name\":\"%s\",\"size\":%ld,\"mtime\":%ld,\"is_dir\":%s,\"ext\":\"%s\"}",
            first ? "" : ",",
            entry->d_name,
            (long)st.st_size,
            (long)st.st_mtime,
            is_dir ? "true" : "false",
            ext);
        first = 0;

        if (buf_len + (size_t)item_len + 32 >= buf_cap)
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
    }

    closedir(dir);

    snprintf(buf + buf_len, buf_cap - buf_len, "]}");
    api_send_json(client_fd, 200, buf);
    free(buf);
} // handle_api_files

void handle_api_file_read(
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

    const char *content_type = "text/plain; charset=utf-8";
    const char *ext = strrchr(rel_path, '.');
    if (ext && (strcmp(ext, ".bin") == 0 || strcmp(ext, ".fits") == 0 ||
                strcmp(ext, ".png") == 0 || strcmp(ext, ".mp4") == 0))
    {
        content_type = "application/octet-stream";
    }

    api_send_response(client_fd, 200, content_type,
                      content, read_bytes);
    free(content);
} // handle_api_file_read

void handle_api_file_write(
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
} // handle_api_file_write

void handle_api_shm_list(
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
} // handle_api_shm_list

void handle_api_shm_telemetry(
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
} // handle_api_shm_telemetry

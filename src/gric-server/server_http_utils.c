/**
 * @file server_http_utils.c
 * @brief HTTP helper functions, URL decoding, query parsing, and binary discovery.
 */

#define _GNU_SOURCE
#include "server_http_utils.h"
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void api_send_response(
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
} // api_send_response

void api_send_json(
    int         client_fd,
    int         status_code,
    const char *json_str)
{
    size_t len = (json_str != NULL) ? strlen(json_str) : 0;
    api_send_response(client_fd, status_code, "application/json; charset=utf-8",
                      (json_str != NULL) ? json_str : "{}", len);
} // api_send_json

void url_decode(
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
} // url_decode

int get_query_param(
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
} // get_query_param

int sanitize_path(
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
} // sanitize_path

int check_binary_available(
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
} // check_binary_available

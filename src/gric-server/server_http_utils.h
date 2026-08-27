/**
 * @file server_http_utils.h
 * @brief HTTP helper functions, URL decoding, query parsing, and binary discovery.
 */

#ifndef SERVER_HTTP_UTILS_H
#define SERVER_HTTP_UTILS_H

#include "http_server.h"
#include <stddef.h>

void api_send_response(
    int         client_fd,
    int         status_code,
    const char *content_type,
    const char *data,
    size_t      data_len);

void api_send_json(
    int         client_fd,
    int         status_code,
    const char *json_str);

void url_decode(
    const char *src,
    char       *dst,
    size_t      dst_size);

int get_query_param(
    const char *query,
    const char *key,
    char       *out_val,
    size_t      out_size);

int sanitize_path(
    const char *base_dir,
    const char *rel_path,
    char       *out_full_path,
    size_t      out_size);

int check_binary_available(
    const ServerConfig *config,
    const char         *bin_name,
    char               *out_path,
    size_t              out_size);

#endif /* SERVER_HTTP_UTILS_H */

/**
 * @file server_http_utils.h
 * @brief HTTP helper functions, URL decoding, query parsing, and binary discovery.
 */

#ifndef SERVER_HTTP_UTILS_H
#define SERVER_HTTP_UTILS_H

#include "http_server.h"
#include <stddef.h>

/**
 * api_send_response() - Send raw HTTP response with headers to client socket.
 * @client_fd:    Client socket file descriptor.
 * @status_code:  HTTP status code (e.g., 200, 404, 500).
 * @content_type: MIME content type header string.
 * @data:         Payload buffer.
 * @data_len:     Payload length in bytes.
 */
void api_send_response(
    int         client_fd,
    int         status_code,
    const char *content_type,
    const char *data,
    size_t      data_len);

/**
 * api_send_json() - Send JSON response with application/json header.
 * @client_fd:   Client socket file descriptor.
 * @status_code: HTTP status code.
 * @json_str:    Null-terminated JSON payload string.
 */
void api_send_json(
    int         client_fd,
    int         status_code,
    const char *json_str);

/**
 * url_decode() - Decode percent-encoded URL string in place or to buffer.
 * @src:      Encoded source string.
 * @dst:      Destination buffer for decoded string.
 * @dst_size: Size of destination buffer in bytes.
 */
void url_decode(
    const char *src,
    char       *dst,
    size_t      dst_size);

/**
 * get_query_param() - Extract value of query string parameter by key.
 * @query:    Raw query string (after '?').
 * @key:      Key name to search for.
 * @out_val:  Buffer to store decoded parameter value.
 * @out_size: Destination buffer capacity in bytes.
 *
 * Return: 0 if found and extracted, -1 if not found.
 */
int get_query_param(
    const char *query,
    const char *key,
    char       *out_val,
    size_t      out_size);

/**
 * sanitize_path() - Prevent directory traversal attacks and resolve full path.
 * @base_dir:      Allowed root directory.
 * @rel_path:      Relative path requested by client.
 * @out_full_path: Buffer for sanitized absolute path.
 * @out_size:      Buffer capacity in bytes.
 *
 * Return: 0 if valid and safe within base_dir, -1 if traversal detected or error.
 */
int sanitize_path(
    const char *base_dir,
    const char *rel_path,
    char       *out_full_path,
    size_t      out_size);

/**
 * check_binary_available() - Locate executable binary in known system and build paths.
 * @config:   Server configuration with search paths.
 * @bin_name: Executable binary name (e.g. "gric-cluster").
 * @out_path: Buffer to store resolved absolute path.
 * @out_size: Buffer capacity in bytes.
 *
 * Return: 0 if found and executable, -1 otherwise.
 */
int check_binary_available(
    const ServerConfig *config,
    const char         *bin_name,
    char               *out_path,
    size_t              out_size);

#endif /* SERVER_HTTP_UTILS_H */

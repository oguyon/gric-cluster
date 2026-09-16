/**
 * @file server_fs_api.h
 * @brief File system and shared memory listing API handlers.
 */

#ifndef SERVER_FS_API_H
#define SERVER_FS_API_H

#include "http_server.h"
#include <stddef.h>

/**
 * handle_api_info() - Handle GET /api/info request returning server metadata.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration and paths.
 */
void handle_api_info(
    int                 client_fd,
    const ServerConfig *config);

/**
 * handle_api_files() - Handle GET /api/files request listing directory contents.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration.
 * @query:     HTTP query string containing target directory.
 */
void handle_api_files(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

/**
 * handle_api_file_read() - Handle GET /api/file/read request serving file contents.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration.
 * @query:     HTTP query string specifying target filename.
 */
void handle_api_file_read(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

/**
 * handle_api_file_write() - Handle POST /api/file/write request saving file.
 * @client_fd: Connected client socket descriptor.
 * @config:    Server configuration.
 * @body:      Raw request payload containing path and contents.
 * @body_len:  Payload length in bytes.
 */
void handle_api_file_write(
    int                 client_fd,
    const ServerConfig *config,
    const char         *body,
    size_t              body_len);

/**
 * handle_api_shm_list() - Handle GET /api/shm/list returning active SHM images.
 * @client_fd: Connected client socket descriptor.
 */
void handle_api_shm_list(
    int client_fd);

/**
 * handle_api_shm_telemetry() - Handle GET /api/shm/telemetry returning stream metrics.
 * @client_fd: Connected client socket descriptor.
 * @query:     HTTP query string specifying SHM stream name.
 */
void handle_api_shm_telemetry(
    int         client_fd,
    const char *query);

#endif /* SERVER_FS_API_H */

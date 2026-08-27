/**
 * @file server_fs_api.h
 * @brief File system and shared memory listing API handlers.
 */

#ifndef SERVER_FS_API_H
#define SERVER_FS_API_H

#include "http_server.h"
#include <stddef.h>

void handle_api_info(
    int                 client_fd,
    const ServerConfig *config);

void handle_api_files(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

void handle_api_file_read(
    int                 client_fd,
    const ServerConfig *config,
    const char         *query);

void handle_api_file_write(
    int                 client_fd,
    const ServerConfig *config,
    const char         *body,
    size_t              body_len);

void handle_api_shm_list(
    int client_fd);

void handle_api_shm_telemetry(
    int         client_fd,
    const char *query);

#endif /* SERVER_FS_API_H */

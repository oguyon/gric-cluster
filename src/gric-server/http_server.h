/**
 * @file http_server.h
 * @brief Lightweight POSIX HTTP server for the GRIC Interactive Simulator GUI.
 */

#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include <limits.h>
#include <stdint.h>
#include <sys/types.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct
{
    int  port;
    char bind_ip[64];
    char workdir[PATH_MAX];
    char docs_dir[PATH_MAX];
    char server_bin_dir[PATH_MAX];
    int  verbose;
} ServerConfig;

/**
 * @brief Initialize server configuration with default values.
 *
 * @param config Pointer to ServerConfig struct.
 */
void server_config_init(
    ServerConfig *config);

/**
 * @brief Run the HTTP server main event loop (blocking until stopped).
 *
 * @param config Pointer to configured ServerConfig struct.
 * @return 0 on normal termination, non-zero on error.
 */
int server_run(
    const ServerConfig *config);

/**
 * @brief Request graceful server shutdown.
 */
void server_stop(
    void);

#endif // HTTP_SERVER_H

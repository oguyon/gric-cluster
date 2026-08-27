/**
 * @file server_cli_runner.h
 * @brief Dispatch and execution of asynchronous CLI tasks into tmux sessions.
 */

#ifndef SERVER_CLI_RUNNER_H
#define SERVER_CLI_RUNNER_H

#include "http_server.h"
#include <stddef.h>

void handle_api_cli_run(
    int                 client_fd,
    const ServerConfig *config,
    const char         *body,
    size_t              body_len);

#endif /* SERVER_CLI_RUNNER_H */

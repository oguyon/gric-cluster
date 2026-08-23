/**
 * @file api_handlers.h
 * @brief REST API endpoint handlers for the GRIC desktop micro-server.
 */

#ifndef API_HANDLERS_H
#define API_HANDLERS_H

#include "http_server.h"
#include <stddef.h>

/**
 * @brief Handle an incoming /api/ route request.
 *
 * @param client_fd Socket file descriptor for the connected HTTP client.
 * @param config    Pointer to the active ServerConfig.
 * @param method    HTTP method string ("GET", "POST", "OPTIONS").
 * @param path      Request URL path (starts with "/api/").
 * @param query     Query string (after '?') or NULL.
 * @param body      Request body payload buffer or NULL.
 * @param body_len  Length of request body in bytes.
 * @return 1 if handled, 0 if route not found.
 */
int handle_api_request(
    int                 client_fd,
    const ServerConfig *config,
    const char         *method,
    const char         *path,
    const char         *query,
    const char         *body,
    size_t              body_len);

/**
 * @brief Clean up any background CLI processes and temporary files.
 */
void api_handlers_cleanup(
    void);

#endif // API_HANDLERS_H

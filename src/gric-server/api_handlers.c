/**
 * @file api_handlers.c
 * @brief Top-level REST API request dispatcher for the GRIC desktop micro-server.
 */

#define _GNU_SOURCE
#include "api_handlers.h"
#include "server_http_utils.h"
#include "server_fs_api.h"
#include "server_cli_runner.h"
#include "server_job_mgmt.h"
#include <string.h>

static void handle_api_heartbeat(
    int client_fd)
{
    server_record_heartbeat();
    api_send_json(client_fd, 200, "{\"status\":\"ok\"}");
} // handle_api_heartbeat

static void handle_api_heartbeat_leave(
    int client_fd)
{
    server_client_leave();
    api_send_json(client_fd, 200, "{\"status\":\"bye\"}");
} // handle_api_heartbeat_leave

static void handle_api_shutdown(
    int client_fd)
{
    api_send_json(client_fd, 200, "{\"status\":\"shutting_down\"}");
    server_stop();
} // handle_api_shutdown

int handle_api_request(
    int                 client_fd,
    const ServerConfig *config,
    const char         *method,
    const char         *path,
    const char         *query,
    const char         *body,
    size_t              body_len)
{
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
        handle_api_files(client_fd, config, query);
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
    if (strcmp(path, "/api/heartbeat") == 0 &&
        (strcmp(method, "POST") == 0 || strcmp(method, "GET") == 0))
    {
        handle_api_heartbeat(client_fd);
        return 1;
    }
    if (strcmp(path, "/api/heartbeat/leave") == 0 &&
        (strcmp(method, "POST") == 0 || strcmp(method, "GET") == 0))
    {
        handle_api_heartbeat_leave(client_fd);
        return 1;
    }
    if (strcmp(path, "/api/shutdown") == 0 &&
        (strcmp(method, "POST") == 0 || strcmp(method, "GET") == 0))
    {
        handle_api_shutdown(client_fd);
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
} // handle_api_request

void api_handlers_cleanup(
    void)
{
    server_jobs_cleanup();
} // api_handlers_cleanup

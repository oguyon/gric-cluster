/**
 * @file http_server.c
 * @brief POSIX socket HTTP server implementation for GRIC Simulator desktop mode.
 */

#define _GNU_SOURCE
#include "http_server.h"
#include "api_handlers.h"
#include "shared/cli_colors.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static volatile sig_atomic_t s_server_running = 1;

void server_config_init(
    ServerConfig *config)
{
    if (!config)
    {
        return;
    }

    memset(config, 0, sizeof(ServerConfig));
    config->port = 8080;
    snprintf(config->bind_ip, sizeof(config->bind_ip), "127.0.0.1");

    const char *env_port = getenv("GRIC_SERVER_PORT");
    if (env_port && *env_port)
    {
        int p = atoi(env_port);
        if (p > 0 && p < 65536)
        {
            config->port = p;
        }
    }

    if (!getcwd(config->workdir, sizeof(config->workdir)))
    {
        snprintf(config->workdir, sizeof(config->workdir), ".");
    }

    /* Auto-detect docs directory */
    const char *doc_candidates[] = {
        "docs",
        "../docs",
        "../../docs",
        "/usr/local/share/gric/docs",
        NULL
    };

    config->docs_dir[0] = '\0';
    for (int i = 0; doc_candidates[i] != NULL; i++)
    {
        char test_path[PATH_MAX];
        snprintf(test_path, sizeof(test_path), "%s/visual_simulator.html", doc_candidates[i]);
        if (access(test_path, R_OK) == 0)
        {
            char resolved[PATH_MAX];
            if (realpath(doc_candidates[i], resolved))
            {
                snprintf(config->docs_dir, sizeof(config->docs_dir), "%s", resolved);
                break;
            }
        }
    }

    if (config->docs_dir[0] == '\0')
    {
        snprintf(config->docs_dir, sizeof(config->docs_dir), "docs");
    }
}

void server_stop(
    void)
{
    s_server_running = 0;
}

static const char *get_mime_type(
    const char *path)
{
    const char *dot = strrchr(path, '.');
    if (!dot)
    {
        return "application/octet-stream";
    }

    if (strcasecmp(dot, ".html") == 0 || strcasecmp(dot, ".htm") == 0)
    {
        return "text/html; charset=utf-8";
    }
    if (strcasecmp(dot, ".js") == 0 || strcasecmp(dot, ".mjs") == 0)
    {
        return "application/javascript; charset=utf-8";
    }
    if (strcasecmp(dot, ".css") == 0)
    {
        return "text/css; charset=utf-8";
    }
    if (strcasecmp(dot, ".wasm") == 0)
    {
        return "application/wasm";
    }
    if (strcasecmp(dot, ".json") == 0)
    {
        return "application/json; charset=utf-8";
    }
    if (strcasecmp(dot, ".png") == 0)
    {
        return "image/png";
    }
    if (strcasecmp(dot, ".svg") == 0)
    {
        return "image/svg+xml";
    }
    if (strcasecmp(dot, ".txt") == 0 || strcasecmp(dot, ".csv") == 0 ||
        strcasecmp(dot, ".dat") == 0 || strcasecmp(dot, ".log") == 0)
    {
        return "text/plain; charset=utf-8";
    }

    return "application/octet-stream";
}

static void serve_static_file(
    int                 client_fd,
    const ServerConfig *config,
    const char         *req_path)
{
    const char *clean_path = req_path;
    if (strcmp(clean_path, "/") == 0 || clean_path[0] == '\0')
    {
        clean_path = "/visual_simulator.html";
    }

    /* Prevent parent directory traversal */
    if (strstr(clean_path, "..") != NULL)
    {
        const char *resp = "HTTP/1.1 403 Forbidden\r\nContent-Length: 9\r\n\r\nForbidden";
        ssize_t nw = write(client_fd, resp, strlen(resp));
        (void)nw;
        return;
    }

    /* Strip leading slash for path joining */
    if (clean_path[0] == '/')
    {
        clean_path++;
    }

    char full_path[PATH_MAX + 512];
    int pw = snprintf(full_path, sizeof(full_path), "%s/%s", config->docs_dir, clean_path);
    if (pw <= 0 || (size_t)pw >= sizeof(full_path))
    {
        const char *resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request";
        ssize_t nw = write(client_fd, resp, strlen(resp));
        (void)nw;
        return;
    }

    struct stat st;
    if (stat(full_path, &st) != 0 || S_ISDIR(st.st_mode))
    {
        const char *resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
        ssize_t nw = write(client_fd, resp, strlen(resp));
        (void)nw;
        return;
    }

    FILE *f = fopen(full_path, "rb");
    if (!f)
    {
        const char *resp = "HTTP/1.1 500 Internal Error\r\nContent-Length: 14\r\n\r\nInternal Error";
        ssize_t nw = write(client_fd, resp, strlen(resp));
        (void)nw;
        return;
    }

    const char *mime = get_mime_type(full_path);
    char header[1024];
    int hlen = snprintf(
        header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Expires: 0\r\n"
        "Cross-Origin-Opener-Policy: same-origin\r\n"
        "Cross-Origin-Embedder-Policy: require-corp\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime, (size_t)st.st_size);

    if (hlen > 0)
    {
        ssize_t nw = write(client_fd, header, (size_t)hlen);
        (void)nw;
    }

    char buf[65536];
    size_t nread = 0;
    while ((nread = fread(buf, 1, sizeof(buf), f)) > 0)
    {
        ssize_t nw = write(client_fd, buf, nread);
        (void)nw;
    }
    fclose(f);
}

static void handle_client_connection(
    int                 client_fd,
    const ServerConfig *config)
{
    size_t cap = 65536;
    char *req_buf = (char *)malloc(cap);
    if (!req_buf)
    {
        close(client_fd);
        return;
    }

    ssize_t nread = recv(client_fd, req_buf, cap - 1, 0);
    if (nread <= 0)
    {
        free(req_buf);
        close(client_fd);
        return;
    }
    req_buf[nread] = '\0';

    /* Find header boundary \r\n\r\n */
    char *hdr_end = strstr(req_buf, "\r\n\r\n");
    size_t hdr_len = 0;
    size_t content_len = 0;

    if (hdr_end)
    {
        hdr_len = (size_t)(hdr_end - req_buf) + 4;
        const char *cl_hdr = strcasestr(req_buf, "Content-Length:");
        if (cl_hdr && cl_hdr < hdr_end)
        {
            content_len = (size_t)strtoul(cl_hdr + 15, NULL, 10);
        }
    }

    /* If Content-Length exceeds what was initially read, buffer the rest */
    if (content_len > 0 && (size_t)nread < hdr_len + content_len)
    {
        size_t total_expected = hdr_len + content_len;
        if (total_expected > 128 * 1024 * 1024)
        {
            free(req_buf);
            close(client_fd);
            return;
        }

        char *expanded = (char *)realloc(req_buf, total_expected + 1);
        if (!expanded)
        {
            free(req_buf);
            close(client_fd);
            return;
        }
        req_buf = expanded;

        while ((size_t)nread < total_expected)
        {
            ssize_t r = recv(client_fd, req_buf + nread, total_expected - (size_t)nread, 0);
            if (r <= 0)
            {
                break;
            }
            nread += r;
        }
        req_buf[nread] = '\0';
    }

    /* Parse request line: METHOD PATH HTTP/1.1 */
    char method[16] = {0};
    char uri[2048] = {0};
    char version[16] = {0};

    char *line_end = strstr(req_buf, "\r\n");
    if (!line_end)
    {
        line_end = strchr(req_buf, '\n');
    }
    if (!line_end)
    {
        free(req_buf);
        close(client_fd);
        return;
    }

    if (sscanf(req_buf, "%15s %2047s %15s", method, uri, version) < 2)
    {
        free(req_buf);
        close(client_fd);
        return;
    }

    /* Split URI into path and query */
    char path[2048];
    char query[2048] = {0};
    char *qmark = strchr(uri, '?');
    if (qmark)
    {
        size_t plen = (size_t)(qmark - uri);
        if (plen >= sizeof(path))
        {
            plen = sizeof(path) - 1;
        }
        memcpy(path, uri, plen);
        path[plen] = '\0';
        snprintf(query, sizeof(query), "%s", qmark + 1);
    }
    else
    {
        snprintf(path, sizeof(path), "%s", uri);
    }

    /* Find body start */
    char *body_start = strstr(req_buf, "\r\n\r\n");
    size_t body_len = 0;
    const char *body = NULL;
    if (body_start)
    {
        body_start += 4;
        body = body_start;
        body_len = (size_t)(nread - (body_start - req_buf));
    }

    /* Check if this is an API route */
    if (strncmp(path, "/api/", 5) == 0 || strcmp(method, "OPTIONS") == 0)
    {
        if (handle_api_request(client_fd, config, method, path, query, body, body_len))
        {
            free(req_buf);
            close(client_fd);
            return;
        }
    }

    /* Serve static file */
    serve_static_file(client_fd, config, path);
    free(req_buf);
    close(client_fd);
}

int server_run(
    const ServerConfig *config)
{
    cli_colors_init();

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
        fprintf(stderr, "%sError: socket() failed: %s%s\n",
                ansi_color_red, strerror(errno), ansi_reset);
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in saddr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons((uint16_t)config->port);
    inet_pton(AF_INET, config->bind_ip, &saddr.sin_addr);

    if (bind(server_fd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
    {
        fprintf(stderr, "%sError: bind() failed on %s:%d: %s%s\n",
                ansi_color_red, config->bind_ip, config->port, strerror(errno), ansi_reset);
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 64) < 0)
    {
        fprintf(stderr, "%sError: listen() failed: %s%s\n",
                ansi_color_red, strerror(errno), ansi_reset);
        close(server_fd);
        return 1;
    }

    printf("%s⚡ GRIC Desktop Micro-Server started%s\n", ansi_bold_green, ansi_reset);
    printf("  %sURL:%s          %shttp://%s:%d/visual_simulator.html%s\n",
           ansi_color_cyan, ansi_reset, ansi_bold_cyan,
           config->bind_ip, config->port, ansi_reset);
    printf("  %sWorkspace:%s    %s%s%s\n",
           ansi_color_cyan, ansi_reset, ansi_color_yellow, config->workdir, ansi_reset);
    printf("  %sDocs Dir:%s     %s%s%s\n",
           ansi_color_cyan, ansi_reset, ansi_color_grey, config->docs_dir, ansi_reset);
    printf("  %sEngine Mode:%s  %sDual-Mode (WASM + Native C CLI)%s\n\n",
           ansi_color_cyan, ansi_reset, ansi_color_magenta, ansi_reset);

    struct pollfd pfd;
    pfd.fd = server_fd;
    pfd.events = POLLIN;

    while (s_server_running)
    {
        int ret = poll(&pfd, 1, 250);
        if (ret < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
        if (ret > 0 && (pfd.revents & POLLIN))
        {
            struct sockaddr_in caddr;
            socklen_t clen = sizeof(caddr);
            int client_fd = accept(server_fd, (struct sockaddr *)&caddr, &clen);
            if (client_fd >= 0)
            {
                handle_client_connection(client_fd, config);
            }
        }
    } // while (s_server_running)

    close(server_fd);
    api_handlers_cleanup();
    printf("\n%sGRIC Desktop Micro-Server stopped.%s\n", ansi_color_yellow, ansi_reset);
    return 0;
}

/**
 * @file gric-mcp.c
 * @brief Main entrypoint for GRIC Model Context Protocol (MCP) C17 server.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mcp_dispatch.h"
#include "cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GRIC_MCP_VERSION "1.0.0"

/**
 * print_usage() - Print usage instructions for manual terminal invocation.
 * @prog: Program invocation name.
 */
static void print_usage(
    const char *prog)
{
    printf("%sgric-mcp%s v%s - Native C17 Model Context Protocol Server for GRIC\n\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET, GRIC_MCP_VERSION);
    printf("Usage:\n");
    printf("  %s                Start stdio JSON-RPC 2.0 MCP server loop\n", prog);
    printf("  %s --list         Print JSON schema of all available tools\n", prog);
    printf("  %s -h, --help     Show this help message\n", prog);
    printf("  %s -v, --version  Print version information\n\n", prog);
    printf("Description:\n");
    printf("  Provides high-speed C-native MCP tools to AI coding agents for automated\n");
    printf("  source code style auditing, SIMD vectorization inspection, clustering\n");
    printf("  invariant verification, dataset profiling, and shared memory telemetry.\n");
} // print_usage

int main(
    int   argc,
    char *argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
        {
            printf("gric-mcp version %s (C17 / JSON-RPC 2.0)\n", GRIC_MCP_VERSION);
            return 0;
        }
        if (strcmp(argv[1], "--list") == 0)
        {
            const char *req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}";
            char *resp = mcp_dispatch_message(req);
            if (resp != NULL)
            {
                printf("%s\n", resp);
                free(resp);
            }
            return 0;
        }
    }

    /* Disable buffering on stdout to ensure immediate transmission of responses */
    setvbuf(stdout, NULL, _IONBF, 0);

    char *line = NULL;
    size_t cap = 0;
    ssize_t nread = 0;

    while ((nread = getline(&line, &cap, stdin)) != -1)
    {
        /* Strip trailing newline characters */
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'))
        {
            line[nread - 1] = '\0';
            nread--;
        }

        if (nread == 0)
        {
            continue;
        }

        char *response = mcp_dispatch_message(line);
        if (response != NULL)
        {
            fputs(response, stdout);
            fputc('\n', stdout);
            fflush(stdout);
            free(response);
        }
    } // while getline

    if (line != NULL)
    {
        free(line);
    }

    return 0;
} // main

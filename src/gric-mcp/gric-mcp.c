/**
 * @file gric-mcp.c
 * @brief Main entrypoint for GRIC Model Context Protocol (MCP) C17 server.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mcp_dispatch.h"
#include "mcp_registry.h"
#include "mcp_tools.h"
#include "gric_build_info.h"
#include "shared/cjson/cJSON.h"
#include "shared/cli/cli_colors.h"
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
    printf(
        "%sgric-mcp%s (%s) - Native C17 Model Context Protocol Server for GRIC\n\n",
        ANSI_BOLD_CYAN, ANSI_COLOR_RESET, GRIC_GIT_DESCRIBE);
    printf("Usage:\n");
    printf("  %s [options]                     Start stdio JSON-RPC 2.0 MCP server loop\n", prog);
    printf("  %s --list [options]              Print JSON list of available tools\n", prog);
    printf("  %s --resource <uri>              Print resource content for given URI\n", prog);
    printf("  %s -h, --help                    Show this help message\n", prog);
    printf("  %s -v, --version                 Print version information\n\n", prog);
    printf("Options:\n");
    printf("  --toolsets=a,b,...   Enable specific toolsets (default: knowledge,analysis,run)\n");
    printf("                       Available: knowledge, analysis, run, ops, dev, all\n");
    printf("  --read-only          Hide and reject side-effecting operational tools\n\n");
    printf("Description:\n");
    printf("  Provides high-speed C-native MCP tools to AI coding agents for automated\n");
    printf("  source code style auditing, SIMD vectorization inspection, clustering\n");
    printf("  invariant verification, dataset profiling, and shared memory telemetry.\n");
} // print_usage

/**
 * parse_toolsets_arg() - Parse comma-delimited toolset names into bitmask.
 * @arg:       Comma-delimited string of toolset names.
 * @mask_out:  Pointer to unsigned integer receiving bitmask.
 *
 * Return: 0 on success, -1 on unknown toolset name.
 */
static int parse_toolsets_arg(
    const char *arg,
    unsigned   *mask_out)
{
    char buf[256];
    strncpy(buf, arg, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    unsigned mask = 0;
    char *saveptr = NULL;
    char *token = strtok_r(buf, ",", &saveptr);

    while (token != NULL)
    {
        while (*token == ' ')
        {
            token++;
        }
        if (strcmp(token, "knowledge") == 0)
        {
            mask |= MCP_TS_KNOWLEDGE;
        }
        else if (strcmp(token, "analysis") == 0)
        {
            mask |= MCP_TS_ANALYSIS;
        }
        else if (strcmp(token, "run") == 0)
        {
            mask |= MCP_TS_RUN;
        }
        else if (strcmp(token, "ops") == 0)
        {
            mask |= MCP_TS_OPS;
        }
        else if (strcmp(token, "dev") == 0)
        {
            mask |= MCP_TS_DEV;
        }
        else if (strcmp(token, "all") == 0)
        {
            mask |= MCP_TS_ALL;
        }
        else
        {
            fprintf(stderr, "Error: Unknown toolset '%s'\n", token);
            return -1;
        }
        token = strtok_r(NULL, ",", &saveptr);
    } // while token != NULL

    *mask_out = mask;
    return 0;
} // parse_toolsets_arg

int main(
    int   argc,
    char *argv[])
{
    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_DEFAULT,
        .read_only       = 0,
        .has_source_tree = 1,
    };

    char root[512];
    mcp_get_project_root(root, sizeof(root));
    char check_file[1024];
    snprintf(check_file, sizeof(check_file), "%s/src/gric-mcp/gric-mcp.c", root);
    cfg.has_source_tree = (access(check_file, R_OK) == 0) ? 1 : 0;

    int do_list = 0;
    const char *resource_uri = NULL;

    for (int ii = 1; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--version") == 0)
        {
            printf(
                "gric-mcp version %s (%s, HEAD %s)\n",
                GRIC_MCP_VERSION, GRIC_GIT_DESCRIBE, GRIC_GIT_HEAD);
            return 0;
        }
        if (strcmp(arg, "--read-only") == 0)
        {
            cfg.read_only = 1;
            continue;
        }
        if (strcmp(arg, "--list") == 0)
        {
            do_list = 1;
            continue;
        }
        if (strncmp(arg, "--toolsets=", 11) == 0)
        {
            if (parse_toolsets_arg(arg + 11, &cfg.toolsets) != 0)
            {
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--toolsets") == 0 && ii + 1 < argc)
        {
            ii++;
            if (parse_toolsets_arg(argv[ii], &cfg.toolsets) != 0)
            {
                return 2;
            }
            continue;
        }
        if (strncmp(arg, "--resource=", 11) == 0)
        {
            resource_uri = arg + 11;
            continue;
        }
        if (strcmp(arg, "--resource") == 0 && ii + 1 < argc)
        {
            ii++;
            resource_uri = argv[ii];
            continue;
        }

        fprintf(stderr, "Error: Unknown argument '%s'\n", arg);
        return 2;
    } // for ii

    mcp_registry_configure(&cfg);

    if (do_list)
    {
        const char *req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}";
        char *resp = mcp_dispatch_message(req);
        if (resp != NULL)
        {
            cJSON *parsed = cJSON_Parse(resp);
            if (parsed != NULL)
            {
                char *formatted = cJSON_Print(parsed);
                if (formatted != NULL)
                {
                    printf("%s\n", formatted);
                    free(formatted);
                }
                else
                {
                    printf("%s\n", resp);
                }
                cJSON_Delete(parsed);
            }
            else
            {
                printf("%s\n", resp);
            }
            free(resp);
        }
        return 0;
    }

    if (resource_uri != NULL)
    {
        char req[1024];
        snprintf(
            req, sizeof(req),
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"resources/read\","
            "\"params\":{\"uri\":\"%s\"}}",
            resource_uri);
        char *resp = mcp_dispatch_message(req);
        if (resp != NULL)
        {
            printf("%s\n", resp);
            free(resp);
        }
        return 0;
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

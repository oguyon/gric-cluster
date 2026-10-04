/**
 * @file mcp_tools.c
 * @brief Path resolution and tools list delegation for GRIC MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

cJSON *mcp_tools_get_list(void)
{
    return mcp_registry_tools_list();
} // mcp_tools_get_list

void mcp_get_project_root(
    char   *buf,
    size_t  size)
{
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0)
    {
        exe_path[len] = '\0';
        char *build_pos = strstr(exe_path, "/build/");
        if (build_pos != NULL)
        {
            *build_pos = '\0';
            strncpy(buf, exe_path, size - 1);
            buf[size - 1] = '\0';
            return;
        }
    }
    strncpy(buf, ".", size - 1);
    buf[size - 1] = '\0';
} // mcp_get_project_root

void mcp_resolve_path(
    const char *input_path,
    char       *out_buf,
    size_t      out_size)
{
    if (input_path == NULL || input_path[0] == '\0')
    {
        out_buf[0] = '\0';
        return;
    }

    /* Check if already accessible directly */
    if (access(input_path, R_OK) == 0)
    {
        strncpy(out_buf, input_path, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* If relative, try prepending project root */
    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char candidate[2048];
    snprintf(candidate, sizeof(candidate), "%s/%s", root, input_path);
    if (access(candidate, R_OK) == 0)
    {
        strncpy(out_buf, candidate, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* Fallback to original */
    strncpy(out_buf, input_path, out_size - 1);
    out_buf[out_size - 1] = '\0';
} // mcp_resolve_path

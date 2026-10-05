/**
 * @file mcp_registry.h
 * @brief Declarative tool registry and filtering for gric-mcp.
 */

#ifndef MCP_REGISTRY_H
#define MCP_REGISTRY_H

#include "shared/cjson/cJSON.h"

typedef int (*mcp_tool_fn)(const cJSON *args, cJSON *res);

enum mcp_toolset
{
    MCP_TS_KNOWLEDGE = 1u << 0,
    MCP_TS_ANALYSIS  = 1u << 1,
    MCP_TS_RUN       = 1u << 2,
    MCP_TS_OPS       = 1u << 3,
    MCP_TS_DEV       = 1u << 4,
};

#define MCP_TS_ALL (MCP_TS_KNOWLEDGE | MCP_TS_ANALYSIS | MCP_TS_RUN | \
                    MCP_TS_OPS | MCP_TS_DEV)
#define MCP_TS_DEFAULT (MCP_TS_KNOWLEDGE | MCP_TS_ANALYSIS | MCP_TS_RUN)

struct mcp_tool_def
{
    const char  *name;
    unsigned     toolset;
    int          side_effects;   /* 1 = hidden and rejected under --read-only */
    mcp_tool_fn  fn;
    const char  *description;
    const char  *input_schema;   /* JSON text */
};

struct mcp_server_config
{
    unsigned toolsets;
    int      read_only;
    int      has_source_tree;
};

/**
 * mcp_registry_configure() - Set global server configuration (toolsets, read-only, source tree).
 * @cfg: Configuration values to apply.
 */
void mcp_registry_configure(
    const struct mcp_server_config *cfg);

/**
 * mcp_registry_get_config() - Get current global server configuration.
 *
 * Return: Pointer to global server configuration struct.
 */
const struct mcp_server_config *mcp_registry_get_config(void);

/**
 * mcp_registry_find() - Look up an active, unhidden tool by name.
 * @name: Name of tool to look up.
 *
 * Return: Pointer to tool definition if active and permitted, NULL if unknown or hidden.
 */
const struct mcp_tool_def *mcp_registry_find(
    const char *name);

/**
 * mcp_registry_tools_list() - Generate cJSON array of active tools for tools/list.
 *
 * Return: Newly allocated cJSON array containing tool descriptors.
 */
cJSON *mcp_registry_tools_list(void);

#endif // MCP_REGISTRY_H

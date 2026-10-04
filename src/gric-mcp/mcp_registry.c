/**
 * @file mcp_registry.c
 * @brief Declarative tool registry implementation and filtering for gric-mcp.
 */

#include "mcp_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const struct mcp_tool_def mcp_tooldef_audit_code_style;
extern const struct mcp_tool_def mcp_tooldef_align_parameters;
extern const struct mcp_tool_def mcp_tooldef_inspect_simd;
extern const struct mcp_tool_def mcp_tooldef_verify_invariants;
extern const struct mcp_tool_def mcp_tooldef_inspect_run;
extern const struct mcp_tool_def mcp_tooldef_probe_dataset;
extern const struct mcp_tool_def mcp_tooldef_probe_shm;
extern const struct mcp_tool_def mcp_tooldef_help;
extern const struct mcp_tool_def mcp_tooldef_list_suite;
extern const struct mcp_tool_def mcp_tooldef_get_recipe;
extern const struct mcp_tool_def mcp_tooldef_server_info;
extern const struct mcp_tool_def mcp_tooldef_fps_status;
extern const struct mcp_tool_def mcp_tooldef_fps_run;
extern const struct mcp_tool_def mcp_tooldef_fps_set;
extern const struct mcp_tool_def mcp_tooldef_fps_stop;
extern const struct mcp_tool_def mcp_tooldef_probe_fps_streams;
extern const struct mcp_tool_def mcp_tooldef_dev_build_test;
extern const struct mcp_tool_def mcp_tooldef_dev_golden_compare;

static const struct mcp_tool_def *const g_all_tools[] = {
    &mcp_tooldef_help,
    &mcp_tooldef_list_suite,
    &mcp_tooldef_get_recipe,
    &mcp_tooldef_server_info,
    &mcp_tooldef_inspect_run,
    &mcp_tooldef_probe_dataset,
    &mcp_tooldef_verify_invariants,
    &mcp_tooldef_probe_shm,
    &mcp_tooldef_fps_status,
    &mcp_tooldef_probe_fps_streams,
    &mcp_tooldef_fps_run,
    &mcp_tooldef_fps_set,
    &mcp_tooldef_fps_stop,
    &mcp_tooldef_audit_code_style,
    &mcp_tooldef_align_parameters,
    &mcp_tooldef_inspect_simd,
    &mcp_tooldef_dev_build_test,
    &mcp_tooldef_dev_golden_compare,
};

static const size_t g_num_tools = sizeof(g_all_tools) / sizeof(g_all_tools[0]);

static struct mcp_server_config g_server_config = {
    .toolsets        = MCP_TS_DEFAULT,
    .read_only       = 0,
    .has_source_tree = 1,
};

void mcp_registry_configure(
    const struct mcp_server_config *cfg)
{
    if (cfg != NULL)
    {
        g_server_config = *cfg;
    }
} // mcp_registry_configure

const struct mcp_server_config *mcp_registry_get_config(void)
{
    return &g_server_config;
} // mcp_registry_get_config

const struct mcp_tool_def *mcp_registry_find(
    const char *name)
{
    if (name == NULL)
    {
        return NULL;
    }

    for (size_t ii = 0; ii < g_num_tools; ii++)
    {
        const struct mcp_tool_def *t = g_all_tools[ii];
        if (strcmp(t->name, name) == 0)
        {
            if ((t->toolset & g_server_config.toolsets) == 0)
            {
                return NULL;
            }
            if (g_server_config.read_only && t->side_effects)
            {
                return NULL;
            }
            if ((t->toolset & MCP_TS_DEV) && !g_server_config.has_source_tree)
            {
                return NULL;
            }
            return t;
        }
    }

    return NULL;
} // mcp_registry_find

cJSON *mcp_registry_tools_list(void)
{
    cJSON *list = cJSON_CreateArray();
    if (list == NULL)
    {
        return NULL;
    }

    for (size_t ii = 0; ii < g_num_tools; ii++)
    {
        const struct mcp_tool_def *t = g_all_tools[ii];
        if (mcp_registry_find(t->name) == NULL)
        {
            continue;
        }

        cJSON *tool = cJSON_CreateObject();
        if (tool == NULL)
        {
            continue;
        }

        cJSON_AddStringToObject(tool, "name", t->name);
        cJSON_AddStringToObject(tool, "description", t->description);

        cJSON *schema = cJSON_Parse(t->input_schema);
        if (schema != NULL)
        {
            cJSON_AddItemToObject(tool, "inputSchema", schema);
        }
        else
        {
            cJSON *empty = cJSON_CreateObject();
            cJSON_AddStringToObject(empty, "type", "object");
            cJSON_AddItemToObject(tool, "inputSchema", empty);
        }

        cJSON_AddItemToArray(list, tool);
    } // for ii

    return list;
} // mcp_registry_tools_list

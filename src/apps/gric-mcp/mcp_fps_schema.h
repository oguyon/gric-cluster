/**
 * @file mcp_fps_schema.h
 * @brief Schema declarations for Milk Function Parameter Structure (FPS) modules.
 */

#ifndef MCP_FPS_SCHEMA_H
#define MCP_FPS_SCHEMA_H

#include <stddef.h>

struct mcp_fps_param
{
    const char *key;       /* ".rlim" */
    const char *fptype;    /* "FLOAT64" (FPTYPE_ prefix stripped) */
    const char *access;    /* "input" | "output" | "trigger_stream" */
    const char *descr;
};

struct mcp_fps_module
{
    const char                 *module;            /* "cluster" | "knn" | "recon" */
    const char                 *binary;            /* "milk-fpsexec-gric-cluster" */
    const char                 *default_fps_name;
    const struct mcp_fps_param *params;
    size_t                      nparams;
};

const struct mcp_fps_module *mcp_fps_module_find(const char *module);

const struct mcp_fps_module *mcp_fps_module_table(size_t *count);

const struct mcp_fps_param  *mcp_fps_param_find(
    const struct mcp_fps_module *mod,
    const char                  *key);

#endif /* MCP_FPS_SCHEMA_H */

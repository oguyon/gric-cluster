/**
 * @file mcp_fps_schema.c
 * @brief Implementation of FPS module schema registry from X-macro definitions.
 */

#include "mcp_fps_schema.h"
#include "gric_fps_params.h"
#include "gric_knn_fps_params.h"
#include "gric_recon_fps_params.h"
#include <string.h>

struct mcp_fps_param_raw
{
    const char *key;
    const char *raw_fptype;
    const char *raw_cliflag;
    const char *descr;
};

#define MCP_FPS_X(KEY, PTR, FPTYPE, FPFLAG, CLIFLAG, DESCR) \
    { KEY, #FPTYPE, #CLIFLAG, DESCR },

static const struct mcp_fps_param_raw cluster_raw[] = {
    GRIC_FPS_PARAMS(MCP_FPS_X)
};

static const struct mcp_fps_param_raw knn_raw[] = {
    GRIC_KNN_FPS_PARAMS(MCP_FPS_X)
};

static const struct mcp_fps_param_raw recon_raw[] = {
    GRIC_RECON_FPS_PARAMS(MCP_FPS_X)
};

#undef MCP_FPS_X

#define CLUSTER_NPARAMS (sizeof(cluster_raw) / sizeof(cluster_raw[0]))
#define KNN_NPARAMS     (sizeof(knn_raw) / sizeof(knn_raw[0]))
#define RECON_NPARAMS   (sizeof(recon_raw) / sizeof(recon_raw[0]))

static struct mcp_fps_param cluster_params[CLUSTER_NPARAMS];
static struct mcp_fps_param knn_params[KNN_NPARAMS];
static struct mcp_fps_param recon_params[RECON_NPARAMS];

static struct mcp_fps_module modules[] = {
    { "cluster", "milk-fpsexec-gric-cluster", "gric_cluster", cluster_params, CLUSTER_NPARAMS },
    { "knn", "milk-fpsexec-gric-knn", "gric_knn", knn_params, KNN_NPARAMS },
    { "recon", "milk-fpsexec-gric-reconstruct", "gric_reconstruct", recon_params, RECON_NPARAMS }
};

static int initialized = 0;

static const char *clean_fptype(
    const char *raw)
{
    if (strncmp(raw, "FPTYPE_", 7) == 0)
    {
        return raw + 7;
    }
    return raw;
}

static const char *clean_access(
    const char *raw)
{
    if (strcmp(raw, "FPFLAG_DEFAULT_OUTPUT") == 0)
    {
        return "output";
    }
    if (strcmp(raw, "FPFLAG_DEFAULT_TRIGGER_STREAM") == 0)
    {
        return "trigger_stream";
    }
    return "input";
}

static void init_params_table(
    const struct mcp_fps_param_raw *raw,
    struct mcp_fps_param           *cooked,
    size_t                          n)
{
    for (size_t ii = 0; ii < n; ii++)
    {
        cooked[ii].key    = raw[ii].key;
        cooked[ii].fptype = clean_fptype(raw[ii].raw_fptype);
        cooked[ii].access = clean_access(raw[ii].raw_cliflag);
        cooked[ii].descr  = raw[ii].descr;
    }
}

static void ensure_initialized(void)
{
    if (initialized)
    {
        return;
    }
    init_params_table(cluster_raw, cluster_params, CLUSTER_NPARAMS);
    init_params_table(knn_raw, knn_params, KNN_NPARAMS);
    init_params_table(recon_raw, recon_params, RECON_NPARAMS);
    initialized = 1;
}

/**
 * mcp_fps_module_table() - Return the table of all available FPS modules.
 * @count: Optional pointer to store module count.
 *
 * Return: Pointer to array of modules.
 */
const struct mcp_fps_module *mcp_fps_module_table(
    size_t *count)
{
    ensure_initialized();
    if (count != NULL)
    {
        *count = sizeof(modules) / sizeof(modules[0]);
    }
    return modules;
}

/**
 * mcp_fps_module_find() - Find an FPS module by name.
 * @module: Module name ("cluster", "knn", "recon").
 *
 * Return: Pointer to module definition or NULL if not found.
 */
const struct mcp_fps_module *mcp_fps_module_find(
    const char *module)
{
    if (module == NULL)
    {
        return NULL;
    }
    ensure_initialized();
    size_t count = sizeof(modules) / sizeof(modules[0]);
    for (size_t ii = 0; ii < count; ii++)
    {
        if (strcmp(modules[ii].module, module) == 0)
        {
            return &modules[ii];
        }
    }
    return NULL;
}

/**
 * mcp_fps_param_find() - Find a parameter in an FPS module schema.
 * @mod: Module definition.
 * @key: Parameter key string (e.g. ".rlim").
 *
 * Return: Pointer to param definition or NULL if not found.
 */
const struct mcp_fps_param *mcp_fps_param_find(
    const struct mcp_fps_module *mod,
    const char                  *key)
{
    if (mod == NULL || key == NULL)
    {
        return NULL;
    }
    ensure_initialized();
    for (size_t ii = 0; ii < mod->nparams; ii++)
    {
        if (strcmp(mod->params[ii].key, key) == 0)
        {
            return &mod->params[ii];
        }
    }
    return NULL;
}

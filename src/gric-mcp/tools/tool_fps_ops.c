/**
 * @file tool_fps_ops.c
 * @brief Shell-free Milk Function Parameter Structure (FPS) operational control tools.
 */

#include "mcp_tools.h"
#include "mcp_exec.h"
#include "mcp_validate.h"
#include "mcp_registry.h"
#include "mcp_fps_schema.h"
#include "shared/gric_stream_layout.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

/**
 * get_fps_binary_path() - Locate the standalone FPS binary for a module.
 * @binary_name: Name of the binary (e.g. "milk-fpsexec-gric-cluster").
 * @buf:         Buffer to receive binary path.
 * @size:        Buffer capacity.
 */
static void get_fps_binary_path(
    const char *binary_name,
    char       *buf,
    size_t      size)
{
    char root[512];
    mcp_get_project_root(root, sizeof(root));

    /* 1. In-tree build binary */
    snprintf(buf, size, "%s/build/src/gric-fps/%s", root, binary_name);
    if (access(buf, X_OK) == 0)
    {
        return;
    }

    /* 2. System Milk installation */
    snprintf(buf, size, "/usr/local/milk/bin/%s", binary_name);
    if (access(buf, X_OK) == 0)
    {
        return;
    }

    /* 3. Fallback to command name in PATH */
    strncpy(buf, binary_name, size - 1);
    buf[size - 1] = '\0';
} // get_fps_binary_path

int mcp_tool_fps_status(
    const cJSON *args,
    cJSON       *res)
{
    const char *module = "cluster";
    if (args != NULL)
    {
        cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "module");
        if (m_item != NULL && cJSON_IsString(m_item) && strlen(m_item->valuestring) > 0)
        {
            module = m_item->valuestring;
        }
    }

    const struct mcp_fps_module *mod = mcp_fps_module_find(module);
    if (mod == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Invalid or unknown FPS module");
        return -1;
    }

    const char *fps_name = mod->default_fps_name;
    if (args != NULL)
    {
        cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
        if (n_item != NULL && cJSON_IsString(n_item) && strlen(n_item->valuestring) > 0)
        {
            fps_name = n_item->valuestring;
        }
    }

    if (!mcp_valid_identifier(fps_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid fps_name identifier");
        return -1;
    }

    char bin_path[1024];
    get_fps_binary_path(mod->binary, bin_path, sizeof(bin_path));

    const char *const procinfo_argv[] = {
        bin_path, "-procinfo", "fps", fps_name, NULL
    };

    char output[8192] = "";
    int exit_status = 0;
    int exec_ret = mcp_exec_capture(procinfo_argv, output, sizeof(output), 5000, &exit_status);
    if (exec_ret != 0)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "Failed to query FPS status");
        return -1;
    }

    cJSON *params_obj = cJSON_CreateObject();
    int in_params_table = 0;
    char *saveptr = NULL;
    char *line = strtok_r(output, "\r\n", &saveptr);

    while (line != NULL)
    {
        if (strstr(line, "CLI Keyword") != NULL)
        {
            in_params_table = 1;
            line = strtok_r(NULL, "\r\n", &saveptr);
            continue;
        }
        if (in_params_table && strncmp(line, "---", 3) == 0)
        {
            line = strtok_r(NULL, "\r\n", &saveptr);
            continue;
        }

        if (in_params_table)
        {
            int idx = -1;
            char key[64] = "";
            char type[32] = "";
            char val[128] = "";

            /* Parse line format: idx keyword type value count description */
            if (sscanf(line, "%d %63s %31s %127s", &idx, key, type, val) >= 3)
            {
                cJSON *p_entry = cJSON_CreateObject();
                cJSON_AddStringToObject(p_entry, "type", type);
                cJSON_AddStringToObject(p_entry, "value", val);
                cJSON_AddItemToObject(params_obj, key, p_entry);
            }
        }
        line = strtok_r(NULL, "\r\n", &saveptr);
    } // while line != NULL

    /* Check if process is actively running via non-shell /proc scanner */
    pid_t pid = -1;
    mcp_find_process(mod->binary, fps_name, &pid);

    cJSON_AddStringToObject(res, "status", (pid > 0) ? "RUNNING" : "CONFIGURED");
    cJSON_AddStringToObject(res, "module", mod->module);
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    cJSON_AddBoolToObject(res, "is_active", (pid > 0));
    if (pid > 0)
    {
        cJSON_AddNumberToObject(res, "pid", (double)pid);
    }
    cJSON_AddItemToObject(res, "parameters", params_obj);
    return 0;
} // mcp_tool_fps_status

int mcp_tool_fps_run(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    const char *module = "cluster";
    cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "module");
    if (m_item != NULL && cJSON_IsString(m_item) && strlen(m_item->valuestring) > 0)
    {
        module = m_item->valuestring;
    }

    const struct mcp_fps_module *mod = mcp_fps_module_find(module);
    if (mod == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Invalid or unknown FPS module");
        return -1;
    }

    cJSON *in_item = cJSON_GetObjectItemCaseSensitive(args, "in_name");
    if (in_item == NULL || !cJSON_IsString(in_item))
    {
        cJSON_AddStringToObject(res, "error", "in_name is required");
        return -1;
    }
    const char *in_name = in_item->valuestring;
    if (!mcp_valid_identifier(in_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid in_name identifier");
        return -1;
    }

    const char *fps_name = mod->default_fps_name;
    cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
    if (n_item != NULL && cJSON_IsString(n_item))
    {
        fps_name = n_item->valuestring;
    }
    if (!mcp_valid_identifier(fps_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid fps_name identifier");
        return -1;
    }

    char out_assign_name[128];
    cJSON *out_item = cJSON_GetObjectItemCaseSensitive(args, "out_name");
    if (out_item != NULL && cJSON_IsString(out_item))
    {
        snprintf(out_assign_name, sizeof(out_assign_name), "%s", out_item->valuestring);
    }
    else
    {
        snprintf(out_assign_name, sizeof(out_assign_name), "%s_assign", in_name);
    }
    if (!mcp_valid_identifier(out_assign_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid out_name identifier");
        return -1;
    }

    int use_tmux = 1;
    cJSON *tmux_item = cJSON_GetObjectItemCaseSensitive(args, "use_tmux");
    if (tmux_item != NULL && cJSON_IsBool(tmux_item))
    {
        use_tmux = cJSON_IsTrue(tmux_item);
    }

    char bin_path[1024];
    get_fps_binary_path(mod->binary, bin_path, sizeof(bin_path));

    /* Step 1: Initialize FPS instance */
    const char *const init_argv[] = {
        bin_path, "-procinfo", "fpsinit", fps_name, NULL
    };
    int init_status = 0;
    int err = mcp_exec_capture(init_argv, NULL, 0, 5000, &init_status);
    if (err != 0 || init_status != 0)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "fpsinit failed");
        return -1;
    }

    /* Step 2: Configure essential parameters via milk-fps-set */
    char set_in_param[128];
    snprintf(set_in_param, sizeof(set_in_param), "%s.in_name", fps_name);
    const char *const set_in_argv[] = { "milk-fps-set", set_in_param, in_name, NULL };
    int set_in_status = 0;
    mcp_exec_capture(set_in_argv, NULL, 0, 5000, &set_in_status);

    char set_out_param[128];
    snprintf(set_out_param, sizeof(set_out_param), "%s.out_name", fps_name);
    const char *const set_out_argv[] = { "milk-fps-set", set_out_param, out_assign_name, NULL };
    int set_out_status = 0;
    mcp_exec_capture(set_out_argv, NULL, 0, 5000, &set_out_status);

    /* Apply optional overrides */
    cJSON *rlim_item = cJSON_GetObjectItemCaseSensitive(args, "rlim");
    if (rlim_item != NULL && cJSON_IsNumber(rlim_item))
    {
        char rlim_buf[64];
        snprintf(rlim_buf, sizeof(rlim_buf), "%.6f", rlim_item->valuedouble);
        char set_rlim_param[128];
        snprintf(set_rlim_param, sizeof(set_rlim_param), "%s.rlim", fps_name);
        const char *const set_rlim_argv[] = { "milk-fps-set", set_rlim_param, rlim_buf, NULL };
        int set_rlim_status = 0;
        mcp_exec_capture(set_rlim_argv, NULL, 0, 5000, &set_rlim_status);
    }

    cJSON *drop_item = cJSON_GetObjectItemCaseSensitive(args, "allow_frame_drop");
    if (drop_item != NULL)
    {
        int drop = cJSON_IsTrue(drop_item);
        char set_drop_param[128];
        snprintf(set_drop_param, sizeof(set_drop_param), "%s.allow_frame_drop", fps_name);
        const char *const set_drop_argv[] = {
            "milk-fps-set", set_drop_param, drop ? "ON" : "OFF", NULL
        };
        int set_drop_status = 0;
        mcp_exec_capture(set_drop_argv, NULL, 0, 5000, &set_drop_status);
    }

    /* Step 3: Launch runstart */
    int launch_status = 0;
    if (use_tmux)
    {
        const char *const launch_argv[] = {
            bin_path, "-tmux", "-procinfo", "-loops", "-n", fps_name, "runstart", NULL
        };
        err = mcp_exec_capture(launch_argv, NULL, 0, 10000, &launch_status);
    }
    else
    {
        const char *const launch_argv[] = {
            bin_path, "-procinfo", "-loops", "-n", fps_name, "runstart", NULL
        };
        pid_t child_pid = 0;
        err = mcp_exec_spawn_detached(launch_argv, NULL, &child_pid);
    }

    if (err != 0 || launch_status != 0)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "runstart failed");
        return -1;
    }

    cJSON_AddStringToObject(res, "status", "LAUNCHED");
    cJSON_AddStringToObject(res, "module", mod->module);
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    cJSON_AddStringToObject(res, "in_name", in_name);
    cJSON_AddStringToObject(res, "out_name", out_assign_name);
    cJSON_AddBoolToObject(res, "tmux_enabled", use_tmux);
    return 0;
} // mcp_tool_fps_run

int mcp_tool_fps_set(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    const char *module = "cluster";
    cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "module");
    if (m_item != NULL && cJSON_IsString(m_item) && strlen(m_item->valuestring) > 0)
    {
        module = m_item->valuestring;
    }

    const struct mcp_fps_module *mod = mcp_fps_module_find(module);
    if (mod == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Invalid or unknown FPS module");
        return -1;
    }

    cJSON *name_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
    cJSON *param_item = cJSON_GetObjectItemCaseSensitive(args, "param");
    cJSON *val_item = cJSON_GetObjectItemCaseSensitive(args, "value");

    if (name_item == NULL || !cJSON_IsString(name_item) ||
        param_item == NULL || !cJSON_IsString(param_item) ||
        val_item == NULL)
    {
        cJSON_AddStringToObject(res, "error", "fps_name, param, and value are required");
        return -1;
    }

    const char *fps_name = name_item->valuestring;
    const char *raw_param = param_item->valuestring;

    if (!mcp_valid_identifier(fps_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid fps_name identifier");
        return -1;
    }

    /* Normalize parameter: strip leading dot if present */
    const char *clean_param = raw_param;
    while (*clean_param == '.')
    {
        clean_param++;
    }

    if (!mcp_valid_identifier(clean_param, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid param identifier");
        return -1;
    }

    char full_key[128];
    snprintf(full_key, sizeof(full_key), ".%s", clean_param);
    const struct mcp_fps_param *pdef = mcp_fps_param_find(mod, full_key);
    if (pdef == NULL)
    {
        pdef = mcp_fps_param_find(mod, raw_param);
    }
    if (pdef == NULL)
    {
        /* Check common parameter aliases */
        if (strcmp(clean_param, "dprob") == 0)
        {
            pdef = mcp_fps_param_find(mod, ".deltaprob");
        }
        else if (strcmp(clean_param, "maxcl") == 0)
        {
            pdef = mcp_fps_param_find(mod, ".maxnbclust");
        }
        else if (strcmp(clean_param, "maxim") == 0)
        {
            pdef = mcp_fps_param_find(mod, ".max_frames");
        }
    }
    if (pdef == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Unknown parameter key for module");
        return -1;
    }
    if (pdef->key[0] == '.')
    {
        clean_param = pdef->key + 1;
    }

    if (strcmp(pdef->access, "output") == 0)
    {
        cJSON_AddStringToObject(res, "error", "Parameter is read-only output in FPS");
        return -1;
    }

    char val_str[128];
    if (strcmp(pdef->fptype, "ONOFF") == 0)
    {
        int onoff_val = 0;
        if (mcp_parse_onoff(val_item, &onoff_val) != 0)
        {
            cJSON_AddStringToObject(
                res, "error", "Invalid ONOFF value (must be ON/OFF or boolean)");
            return -1;
        }
        snprintf(val_str, sizeof(val_str), "%s", onoff_val ? "ON" : "OFF");
    }
    else if (strcmp(pdef->fptype, "FLOAT64") == 0)
    {
        double d = 0.0;
        if (cJSON_IsNumber(val_item))
        {
            d = val_item->valuedouble;
        }
        else if (cJSON_IsString(val_item))
        {
            if (mcp_parse_double_strict(val_item->valuestring, &d) != 0)
            {
                cJSON_AddStringToObject(res, "error", "Invalid value string format");
                return -1;
            }
        }
        else
        {
            cJSON_AddStringToObject(res, "error", "Invalid FLOAT64 value type");
            return -1;
        }
        if (strcmp(clean_param, "rlim") == 0 && d <= 0.0)
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter rlim must be strictly positive (> 0.0)");
            return -1;
        }
        if ((strcmp(clean_param, "dprob") == 0 ||
             strcmp(clean_param, "deltaprob") == 0) && (d < 0.0 || d > 1.0))
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter deltaprob/dprob must be within range [0.0, 1.0]");
            return -1;
        }
        if ((strcmp(clean_param, "fmatcha") == 0 || strcmp(clean_param, "fmatchb") == 0) &&
            d <= 0.0)
        {
            cJSON_AddStringToObject(
                res, "error", "Match parameter must be strictly positive (> 0.0)");
            return -1;
        }
        if (strcmp(clean_param, "soft_bayesian_sigma") == 0 && d <= 0.0)
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter soft_bayesian_sigma must be strictly positive (> 0.0)");
            return -1;
        }
        if (strcmp(clean_param, "entropy_gate") == 0 && d <= 0.0)
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter entropy_gate must be strictly positive (> 0.0)");
            return -1;
        }
        snprintf(val_str, sizeof(val_str), "%g", d);
    }
    else if (strcmp(pdef->fptype, "UINT32") == 0 || strcmp(pdef->fptype, "UINT64") == 0 ||
             strcmp(pdef->fptype, "INT64") == 0)
    {
        int64_t i64 = 0;
        if (cJSON_IsNumber(val_item))
        {
            i64 = (int64_t)val_item->valuedouble;
        }
        else if (cJSON_IsString(val_item))
        {
            if (mcp_parse_int64_strict(val_item->valuestring, &i64) != 0)
            {
                cJSON_AddStringToObject(res, "error", "Invalid value string format");
                return -1;
            }
        }
        else
        {
            cJSON_AddStringToObject(res, "error", "Invalid integer value type");
            return -1;
        }

        if ((strcmp(pdef->fptype, "UINT32") == 0 || strcmp(pdef->fptype, "UINT64") == 0) &&
            i64 < 0)
        {
            char uerr[128];
            snprintf(uerr, sizeof(uerr), "Unsigned parameter %s cannot be negative", clean_param);
            cJSON_AddStringToObject(res, "error", uerr);
            return -1;
        }
        if ((strcmp(clean_param, "maxcl") == 0 ||
             strcmp(clean_param, "maxnbclust") == 0) && i64 < 1)
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter maxnbclust/maxcl must be at least 1");
            return -1;
        }
        if ((strcmp(clean_param, "maxim") == 0 ||
             strcmp(clean_param, "max_frames") == 0) && i64 < 0)
        {
            cJSON_AddStringToObject(
                res, "error", "Parameter max_frames/maxim cannot be negative");
            return -1;
        }
        if (strcmp(clean_param, "k") == 0 && i64 < 1)
        {
            cJSON_AddStringToObject(res, "error", "Parameter k must be at least 1");
            return -1;
        }
        if (strcmp(clean_param, "dtmin") == 0 && i64 < 0)
        {
            cJSON_AddStringToObject(res, "error", "Parameter dtmin cannot be negative");
            return -1;
        }
        snprintf(val_str, sizeof(val_str), "%lld", (long long)i64);
    }
    else
    {
        /* String / path / streamname */
        if (!cJSON_IsString(val_item))
        {
            cJSON_AddStringToObject(res, "error", "String value expected for this parameter");
            return -1;
        }
        const char *s = val_item->valuestring;
        if (!mcp_valid_identifier(s, 120) && !mcp_valid_path(s, 120))
        {
            cJSON_AddStringToObject(res, "error", "Invalid value string format");
            return -1;
        }
        strncpy(val_str, s, sizeof(val_str) - 1);
        val_str[sizeof(val_str) - 1] = '\0';
    }

    char target[256];
    snprintf(target, sizeof(target), "%s.%s", fps_name, clean_param);

    const char *const set_argv[] = {
        "milk-fps-set", target, val_str, NULL
    };

    char output[512] = "";
    int exit_status = 0;
    int err = mcp_exec_capture(set_argv, output, sizeof(output), 5000, &exit_status);

    cJSON_AddStringToObject(res, "status", (err == 0 && exit_status == 0) ? "SUCCESS" : "ERROR");
    cJSON_AddStringToObject(res, "module", mod->module);
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    cJSON_AddStringToObject(res, "param", clean_param);
    cJSON_AddStringToObject(res, "value", val_str);
    cJSON_AddStringToObject(res, "output", output);
    return (err == 0 && exit_status == 0) ? 0 : -1;
} // mcp_tool_fps_set

int mcp_tool_fps_stop(
    const cJSON *args,
    cJSON       *res)
{
    const char *module = "cluster";
    if (args != NULL)
    {
        cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "module");
        if (m_item != NULL && cJSON_IsString(m_item) && strlen(m_item->valuestring) > 0)
        {
            module = m_item->valuestring;
        }
    }

    const struct mcp_fps_module *mod = mcp_fps_module_find(module);
    if (mod == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Invalid or unknown FPS module");
        return -1;
    }

    const char *fps_name = mod->default_fps_name;
    if (args != NULL)
    {
        cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
        if (n_item != NULL && cJSON_IsString(n_item) && strlen(n_item->valuestring) > 0)
        {
            fps_name = n_item->valuestring;
        }
    }

    if (!mcp_valid_identifier(fps_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid fps_name identifier");
        return -1;
    }

    char bin_path[1024];
    get_fps_binary_path(mod->binary, bin_path, sizeof(bin_path));

    /* 1. Dispatch stop to tmux ctrl window */
    char target_ctrl[128];
    snprintf(target_ctrl, sizeof(target_ctrl), "%s:ctrl", fps_name);

    const char *const tmux_conf[] = {
        "tmux", "send-keys", "-t", target_ctrl, "confstop", "C-m", NULL
    };
    int tmux_status = 0;
    mcp_exec_capture(tmux_conf, NULL, 0, 2000, &tmux_status);

    const char *const tmux_run[] = {
        "tmux", "send-keys", "-t", target_ctrl, "runstop", "C-m", NULL
    };
    mcp_exec_capture(tmux_run, NULL, 0, 2000, &tmux_status);

    /* 2. Direct standalone invocation fallback */
    char runstop_arg[128], confstop_arg[128];
    snprintf(runstop_arg, sizeof(runstop_arg), "%s:runstop", fps_name);
    snprintf(confstop_arg, sizeof(confstop_arg), "%s:confstop", fps_name);

    const char *const direct_run[] = { bin_path, "-procinfo", runstop_arg, NULL };
    int direct_status = 0;
    mcp_exec_capture(direct_run, NULL, 0, 3000, &direct_status);

    const char *const direct_conf[] = { bin_path, "-procinfo", confstop_arg, NULL };
    mcp_exec_capture(direct_conf, NULL, 0, 3000, &direct_status);

    cJSON_AddStringToObject(res, "status", "STOPPED");
    cJSON_AddStringToObject(res, "module", mod->module);
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    return 0;
} // mcp_tool_fps_stop

/**
 * mcp_decode_assign() - Decode assignment telemetry stream vector into cJSON object.
 * @vec: Input float array of at least GRIC_ASSIGN_NFIELDS elements.
 * @n:   Number of elements in vec.
 * @out: Target cJSON object to populate with telemetry fields.
 *
 * Return: 0 on success, -1 on invalid arguments or insufficient length.
 */
int mcp_decode_assign(
    const float *vec,
    size_t       n,
    cJSON       *out)
{
    if (vec == NULL || out == NULL || n < GRIC_ASSIGN_NFIELDS)
    {
        return -1;
    }

#define MCP_DEC_X(NAME, IDX, KEY, DESCR) \
    if (IDX == GRIC_ASSIGN_IS_NEW) \
    { \
        cJSON_AddBoolToObject(out, KEY, (vec[IDX] > 0.5f)); \
    } \
    else \
    { \
        cJSON_AddNumberToObject(out, KEY, (double)vec[IDX]); \
    }

    GRIC_ASSIGN_FIELDS(MCP_DEC_X)
#undef MCP_DEC_X
    return 0;
} // mcp_decode_assign

int mcp_tool_probe_fps_streams(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *s_item = cJSON_GetObjectItemCaseSensitive(args, "out_name");
    if (s_item == NULL || !cJSON_IsString(s_item))
    {
        s_item = cJSON_GetObjectItemCaseSensitive(args, "stream_name");
    }

    if (s_item == NULL || !cJSON_IsString(s_item))
    {
        cJSON_AddStringToObject(res, "error", "out_name is required");
        return -1;
    }

    const char *stream_name = s_item->valuestring;
    if (!mcp_valid_identifier(stream_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid stream_name identifier");
        return -1;
    }

#ifdef USE_IMAGESTREAMIO
    IMAGE image;
    memset(&image, 0, sizeof(IMAGE));

    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &image) != 0)
    {
        cJSON_AddStringToObject(res, "stream_name", stream_name);
        cJSON_AddStringToObject(res, "status", "NOT_FOUND");
        cJSON_AddStringToObject(res, "error", "Could not connect to shared memory stream");
        return -1;
    }

    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "status", "CONNECTED");
    cJSON_AddNumberToObject(res, "naxis", (double)image.md[0].naxis);
    cJSON_AddNumberToObject(res, "size0", (double)image.md[0].size[0]);
    cJSON_AddNumberToObject(res, "size1", (double)image.md[0].size[1]);
    cJSON_AddNumberToObject(res, "datatype", (double)image.md[0].datatype);
    cJSON_AddNumberToObject(res, "write_cnt0", (double)image.md[0].cnt0);

    /* Decode canonical assignment telemetry packet if dimensions match */
    if (image.md[0].size[0] == GRIC_ASSIGN_NFIELDS &&
        (image.md[0].naxis == 1 || image.md[0].size[1] == 1) &&
        image.md[0].datatype == _DATATYPE_FLOAT && image.array.raw != NULL)
    {
        const float *vec = (const float *)image.array.raw;
        cJSON *telemetry = cJSON_CreateObject();
        if (mcp_decode_assign(vec, (size_t)image.md[0].size[0], telemetry) == 0)
        {
            cJSON_AddItemToObject(res, "telemetry_vector", telemetry);
        }
        else
        {
            cJSON_Delete(telemetry);
        }
    }

    ImageStreamIO_closeIm(&image);
    return 0;
#else
    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "status", "DISABLED");
    cJSON_AddStringToObject(res, "error", "ImageStreamIO support not compiled into this build");
    return -1;
#endif
} // mcp_tool_probe_fps_streams

const struct mcp_tool_def mcp_tooldef_fps_status = {
    .name         = "gric_fps_status",
    .toolset      = MCP_TS_OPS,
    .side_effects = 0,
    .fn           = mcp_tool_fps_status,
    .description  = "Inspect live status, loop rate, PID, and current parameters of a Milk "
                    "FPS streaming instance (e.g. gric_cluster, gric_knn, gric_reconstruct).",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"module\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS module: cluster (default), knn, or recon.\"\n"
        "    },\n"
        "    \"fps_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS instance name (default depends on module).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_fps_run = {
    .name         = "gric_fps_run",
    .toolset      = MCP_TS_OPS,
    .side_effects = 1,
    .fn           = mcp_tool_fps_run,
    .description  = "Launch standalone Milk streaming daemon (cluster, knn, recon) "
                    "with input/output streams and initial parameters.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"module\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS module: cluster (default), knn, or recon.\"\n"
        "    },\n"
        "    \"in_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Input ImageStreamIO stream name (TRIGGER).\"\n"
        "    },\n"
        "    \"out_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Output assignment stream name (<out>_assign).\"\n"
        "    },\n"
        "    \"fps_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS instance name (default depends on module).\"\n"
        "    },\n"
        "    \"rlim\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Cluster radius threshold (default: 0.5).\"\n"
        "    },\n"
        "    \"allow_frame_drop\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Lag policy: true=jump to latest frame, false=sequential.\"\n"
        "    },\n"
        "    \"use_tmux\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Run inside an isolated tmux session (default: true).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"in_name\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_fps_set = {
    .name         = "gric_fps_set",
    .toolset      = MCP_TS_OPS,
    .side_effects = 1,
    .fn           = mcp_tool_fps_set,
    .description  = "Dynamically update an FPS parameter (e.g. .rlim, .allow_frame_drop, "
                    ".reset_state) on a live streaming instance using milk-fps-set.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"module\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS module: cluster (default), knn, or recon.\"\n"
        "    },\n"
        "    \"fps_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS instance name (e.g. gric_cluster).\"\n"
        "    },\n"
        "    \"param\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Parameter key (e.g. 'rlim', 'allow_frame_drop').\"\n"
        "    },\n"
        "    \"value\": {\n"
        "      \"description\": \"New parameter value (number, boolean ON/OFF, or string).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"fps_name\", \"param\", \"value\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_fps_stop = {
    .name         = "gric_fps_stop",
    .toolset      = MCP_TS_OPS,
    .side_effects = 1,
    .fn           = mcp_tool_fps_stop,
    .description  = "Gracefully stop an active Milk FPS instance (dispatching runstop and "
                    "confstop to tmux ctrl window).",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"module\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS module: cluster (default), knn, or recon.\"\n"
        "    },\n"
        "    \"fps_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS instance name (default depends on module).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_probe_fps_streams = {
    .name         = "gric_probe_fps_streams",
    .toolset      = MCP_TS_OPS,
    .side_effects = 0,
    .fn           = mcp_tool_probe_fps_streams,
    .description  = "Inspect Milk live streams non-blockingly and decode the canonical "
                    "assignment telemetry vector (cluster ID, distance, latency, flags).",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"module\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"FPS module: cluster (default), knn, or recon.\"\n"
        "    },\n"
        "    \"out_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Name of output stream (e.g. gric_cluster_assign).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"out_name\"]\n"
        "}",
};

static int float_compare(
    const void *a,
    const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb)
    {
        return -1;
    }
    if (fa > fb)
    {
        return 1;
    }
    return 0;
} // float_compare

int mcp_tool_stream_window(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *s_item = cJSON_GetObjectItemCaseSensitive(args, "stream_name");
    if (s_item == NULL || !cJSON_IsString(s_item))
    {
        s_item = cJSON_GetObjectItemCaseSensitive(args, "out_name");
    }
    if (s_item == NULL || !cJSON_IsString(s_item) || s_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "stream_name is required");
        return -1;
    }

    const char *stream_name = s_item->valuestring;
    if (!mcp_valid_identifier(stream_name, 64))
    {
        cJSON_AddStringToObject(res, "error", "Invalid stream_name identifier");
        return -1;
    }

    int num_frames = 100;
    cJSON *nf_item = cJSON_GetObjectItemCaseSensitive(args, "num_frames");
    if (nf_item != NULL && cJSON_IsNumber(nf_item))
    {
        num_frames = nf_item->valueint;
        if (num_frames < 1)
        {
            num_frames = 1;
        }
        if (num_frames > 5000)
        {
            num_frames = 5000;
        }
    }

    int timeout_ms = 3000;
    cJSON *to_item = cJSON_GetObjectItemCaseSensitive(args, "timeout_ms");
    if (to_item != NULL && cJSON_IsNumber(to_item))
    {
        timeout_ms = to_item->valueint;
        if (timeout_ms < 10)
        {
            timeout_ms = 10;
        }
        if (timeout_ms > 30000)
        {
            timeout_ms = 30000;
        }
    }

#ifndef USE_IMAGESTREAMIO
    (void)num_frames;
    (void)timeout_ms;
    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "error", "ImageStreamIO support not compiled into this build");
    return -1;
#else
    IMAGE image;
    memset(&image, 0, sizeof(IMAGE));

    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &image) != 0)
    {
        cJSON_AddStringToObject(res, "stream_name", stream_name);
        cJSON_AddStringToObject(res, "status", "STREAM_NOT_FOUND");
        cJSON_AddStringToObject(res, "error", "Could not connect to shared memory stream");
        return -1;
    }

    if (image.md[0].size[0] < GRIC_ASSIGN_NFIELDS ||
        image.md[0].datatype != _DATATYPE_FLOAT || image.array.raw == NULL)
    {
        ImageStreamIO_closeIm(&image);
        cJSON_AddStringToObject(
            res, "error", "Stream is not a valid GRIC assignment telemetry stream");
        return -1;
    }

    float *latencies = (float *)malloc((size_t)num_frames * sizeof(float));
    float *distances = (float *)malloc((size_t)num_frames * sizeof(float));
    if (latencies == NULL || distances == NULL)
    {
        free(latencies);
        free(distances);
        ImageStreamIO_closeIm(&image);
        cJSON_AddStringToObject(res, "error", "Memory allocation failed");
        return -1;
    }

    /* Track rolling statistics */
    uint64_t last_cnt0 = image.md[0].cnt0;
    int sampled = 0;
    int dropped_frames = 0;
    int new_anchors = 0;
    int within_rlim_count = 0;
    float start_clusters = 0.0f;
    float end_clusters = 0.0f;

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    /* Read initial frame snapshot */
    const float *vec = (const float *)image.array.raw;
    start_clusters = vec[GRIC_ASSIGN_TOTAL_CLUSTERS];

    while (sampled < num_frames)
    {
        uint64_t cur_cnt0 = image.md[0].cnt0;
        if (cur_cnt0 != last_cnt0)
        {
            if (last_cnt0 != 0 && cur_cnt0 > last_cnt0 + 1)
            {
                dropped_frames += (int)(cur_cnt0 - last_cnt0 - 1);
            }
            last_cnt0 = cur_cnt0;

            vec = (const float *)image.array.raw;
            latencies[sampled] = vec[GRIC_ASSIGN_LATENCY_US];
            distances[sampled] = vec[GRIC_ASSIGN_DIST];
            if (vec[GRIC_ASSIGN_IS_NEW] > 0.5f)
            {
                new_anchors++;
            }
            if (vec[GRIC_ASSIGN_WITHIN_RLIM] > 0.5f)
            {
                within_rlim_count++;
            }
            end_clusters = vec[GRIC_ASSIGN_TOTAL_CLUSTERS];
            sampled++;
        }
        else
        {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            long elapsed_ms = (ts_now.tv_sec - ts_start.tv_sec) * 1000 +
                              (ts_now.tv_nsec - ts_start.tv_nsec) / 1000000;
            if (elapsed_ms >= timeout_ms)
            {
                break;
            }
            usleep(200);
        }
    }

    /* If stream was idle and no new frames arrived during sampling, take 1 snapshot */
    int is_idle = 0;
    if (sampled == 0)
    {
        is_idle = 1;
        vec = (const float *)image.array.raw;
        latencies[0] = vec[GRIC_ASSIGN_LATENCY_US];
        distances[0] = vec[GRIC_ASSIGN_DIST];
        if (vec[GRIC_ASSIGN_IS_NEW] > 0.5f)
        {
            new_anchors++;
        }
        if (vec[GRIC_ASSIGN_WITHIN_RLIM] > 0.5f)
        {
            within_rlim_count++;
        }
        end_clusters = vec[GRIC_ASSIGN_TOTAL_CLUSTERS];
        sampled = 1;
    }

    ImageStreamIO_closeIm(&image);

    /* Compute percentiles on latencies and distances */
    qsort(latencies, (size_t)sampled, sizeof(float), float_compare);

    double sum_lat = 0.0;
    double sum_dist = 0.0;
    float min_dist = distances[0];
    float max_dist = distances[0];

    for (int ii = 0; ii < sampled; ii++)
    {
        sum_lat += (double)latencies[ii];
        sum_dist += (double)distances[ii];
        if (distances[ii] < min_dist)
        {
            min_dist = distances[ii];
        }
        if (distances[ii] > max_dist)
        {
            max_dist = distances[ii];
        }
    }

    double mean_lat = sum_lat / (double)sampled;
    double mean_dist = sum_dist / (double)sampled;
    float p50_lat = latencies[sampled / 2];
    float p90_lat = latencies[(size_t)((double)sampled * 0.90)];
    float p99_lat = latencies[(size_t)((double)sampled * 0.99)];
    float min_lat = latencies[0];
    float max_lat = latencies[sampled - 1];
    double anchor_rate_1k = ((double)new_anchors * 1000.0) / (double)sampled;
    double within_rlim_pct = ((double)within_rlim_count * 100.0) / (double)sampled;

    free(latencies);
    free(distances);

    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "status", is_idle ? "STREAM_IDLE" : "STREAMING_ACTIVE");
    cJSON_AddNumberToObject(res, "frames_sampled", (double)sampled);
    cJSON_AddNumberToObject(res, "dropped_frames", (double)dropped_frames);

    cJSON *lat_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(lat_obj, "min_us", (double)min_lat);
    cJSON_AddNumberToObject(lat_obj, "mean_us", mean_lat);
    cJSON_AddNumberToObject(lat_obj, "p50_us", (double)p50_lat);
    cJSON_AddNumberToObject(lat_obj, "p90_us", (double)p90_lat);
    cJSON_AddNumberToObject(lat_obj, "p99_us", (double)p99_lat);
    cJSON_AddNumberToObject(lat_obj, "max_us", (double)max_lat);
    cJSON_AddItemToObject(res, "latency_us", lat_obj);

    cJSON *dist_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(dist_obj, "min", (double)min_dist);
    cJSON_AddNumberToObject(dist_obj, "mean", mean_dist);
    cJSON_AddNumberToObject(dist_obj, "max", (double)max_dist);
    cJSON_AddNumberToObject(dist_obj, "within_rlim_pct", within_rlim_pct);
    cJSON_AddItemToObject(res, "distances", dist_obj);

    cJSON *cl_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(cl_obj, "new_anchors_in_window", (double)new_anchors);
    cJSON_AddNumberToObject(cl_obj, "anchor_rate_per_1000", anchor_rate_1k);
    cJSON_AddNumberToObject(cl_obj, "start_clusters", (double)start_clusters);
    cJSON_AddNumberToObject(cl_obj, "end_clusters", (double)end_clusters);
    cJSON_AddItemToObject(res, "clustering_dynamics", cl_obj);

    char summary[512];
    if (is_idle)
    {
        snprintf(summary, sizeof(summary),
                 "Stream '%s' is connected but IDLE (0 new frames during %d ms window). "
                 "Current snapshot: latency=%.1f us, total_clusters=%.0f.",
                 stream_name, timeout_ms, (double)min_lat, (double)end_clusters);
    }
    else
    {
        snprintf(summary, sizeof(summary),
                 "Sampled %d frames from '%s': latency p50=%.1f us, p99=%.1f us; "
                 "%d new anchors (%.1f/1k frames); %d dropped frames; within_rlim=%.1f%%.",
                 sampled, stream_name, (double)p50_lat, (double)p99_lat,
                 new_anchors, anchor_rate_1k, dropped_frames, within_rlim_pct);
    }
    cJSON_AddStringToObject(res, "summary", summary);

    return 0;
#endif
} // mcp_tool_stream_window

const struct mcp_tool_def mcp_tooldef_stream_window = {
    .name         = "gric_stream_window",
    .toolset      = MCP_TS_OPS,
    .side_effects = 0,
    .fn           = mcp_tool_stream_window,
    .description  = "Sample a rolling window of frames from an active ImageStreamIO telemetry "
                    "stream and compute latency (p50/p90/p99), anchor rates, and drops.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"stream_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Name of output stream (e.g. 'cluster_assign').\"\n"
        "    },\n"
        "    \"num_frames\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of frames to sample (default: 100, max: 5000).\"\n"
        "    },\n"
        "    \"timeout_ms\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Maximum sampling timeout in milliseconds (default: 3000).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"stream_name\"]\n"
        "}",
};


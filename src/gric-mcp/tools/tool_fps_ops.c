/**
 * @file tool_fps_ops.c
 * @brief Milk Function Parameter Structure (FPS) operational control tools.
 */

#include "mcp_tools.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

/**
 * get_fps_binary_path() - Locate the standalone FPS binary.
 * @buf:  Buffer to receive binary path.
 * @size: Buffer capacity.
 */
static void get_fps_binary_path(
    char   *buf,
    size_t  size)
{
    char root[512];
    mcp_get_project_root(root, sizeof(root));

    /* 1. In-tree build binary */
    snprintf(buf, size, "%s/build/src/gric-fps/milk-fpsexec-gric-cluster", root);
    if (access(buf, X_OK) == 0)
    {
        return;
    }

    /* 2. System Milk installation */
    snprintf(buf, size, "/usr/local/milk/bin/milk-fpsexec-gric-cluster");
    if (access(buf, X_OK) == 0)
    {
        return;
    }

    /* 3. Fallback to command name in PATH */
    strncpy(buf, "milk-fpsexec-gric-cluster", size - 1);
    buf[size - 1] = '\0';
} // get_fps_binary_path

/**
 * run_sys_cmd() - Execute a shell command and return its exit code.
 * @cmd: Command line to execute.
 *
 * Return: Command return status.
 */
static int run_sys_cmd(
    const char *cmd)
{
    return system(cmd);
} // run_sys_cmd

int mcp_tool_fps_status(
    const cJSON *args,
    cJSON       *res)
{
    const char *fps_name = "gric_cluster";
    if (args != NULL)
    {
        cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
        if (n_item != NULL && cJSON_IsString(n_item) && strlen(n_item->valuestring) > 0)
        {
            fps_name = n_item->valuestring;
        }
    }

    char bin_path[1024];
    get_fps_binary_path(bin_path, sizeof(bin_path));

    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "%s fps %s 2>&1", bin_path, fps_name);

    FILE *pipe = popen(cmd, "r");
    if (pipe == NULL)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "Failed to query FPS status");
        return -1;
    }

    cJSON *params_obj = cJSON_CreateObject();
    char line[1024];
    int in_params_table = 0;

    while (fgets(line, sizeof(line), pipe) != NULL)
    {
        if (strstr(line, "CLI Keyword") != NULL)
        {
            in_params_table = 1;
            continue;
        }
        if (in_params_table && strncmp(line, "---", 3) == 0)
        {
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
    } // while fgets

    pclose(pipe);

    /* Check if process is actively running via pgrep */
    char pgrep_cmd[512];
    snprintf(pgrep_cmd, sizeof(pgrep_cmd),
             "pgrep -f 'milk-fpsexec-gric-cluster.*%s' | head -n 1", fps_name);
    FILE *pp = popen(pgrep_cmd, "r");
    long pid = -1;
    if (pp != NULL)
    {
        char pid_buf[32] = "";
        if (fgets(pid_buf, sizeof(pid_buf), pp) != NULL)
        {
            pid = atol(pid_buf);
        }
        pclose(pp);
    }

    cJSON_AddStringToObject(res, "status", (pid > 0) ? "RUNNING" : "CONFIGURED");
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

    cJSON *in_item = cJSON_GetObjectItemCaseSensitive(args, "in_name");
    if (in_item == NULL || !cJSON_IsString(in_item))
    {
        cJSON_AddStringToObject(res, "error", "in_name is required");
        return -1;
    }
    const char *in_name = in_item->valuestring;

    const char *fps_name = "gric_cluster";
    cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
    if (n_item != NULL && cJSON_IsString(n_item))
    {
        fps_name = n_item->valuestring;
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

    int use_tmux = 1;
    cJSON *tmux_item = cJSON_GetObjectItemCaseSensitive(args, "use_tmux");
    if (tmux_item != NULL && cJSON_IsBool(tmux_item))
    {
        use_tmux = cJSON_IsTrue(tmux_item);
    }

    char bin_path[1024];
    get_fps_binary_path(bin_path, sizeof(bin_path));

    /* Step 1: Initialize FPS instance */
    char init_cmd[2048];
    snprintf(init_cmd, sizeof(init_cmd), "%s fpsinit %s > /dev/null 2>&1", bin_path, fps_name);
    (void)run_sys_cmd(init_cmd);

    /* Step 2: Configure essential parameters via milk-fps-set */
    char set_cmd[2048];
    snprintf(set_cmd, sizeof(set_cmd), "milk-fps-set %s.in_name %s > /dev/null 2>&1",
             fps_name, in_name);
    (void)run_sys_cmd(set_cmd);

    snprintf(set_cmd, sizeof(set_cmd), "milk-fps-set %s.out_name %s > /dev/null 2>&1",
             fps_name, out_assign_name);
    (void)run_sys_cmd(set_cmd);

    /* Apply optional overrides */
    cJSON *rlim_item = cJSON_GetObjectItemCaseSensitive(args, "rlim");
    if (rlim_item != NULL && cJSON_IsNumber(rlim_item))
    {
        snprintf(set_cmd, sizeof(set_cmd), "milk-fps-set %s.rlim %.6f > /dev/null 2>&1",
                 fps_name, rlim_item->valuedouble);
        (void)run_sys_cmd(set_cmd);
    }

    cJSON *drop_item = cJSON_GetObjectItemCaseSensitive(args, "allow_frame_drop");
    if (drop_item != NULL)
    {
        int drop = cJSON_IsTrue(drop_item);
        snprintf(set_cmd, sizeof(set_cmd), "milk-fps-set %s.allow_frame_drop %s > /dev/null 2>&1",
                 fps_name, drop ? "ON" : "OFF");
        (void)run_sys_cmd(set_cmd);
    }

    /* Step 3: Launch runstart */
    char launch_cmd[2048];
    if (use_tmux)
    {
        snprintf(launch_cmd, sizeof(launch_cmd),
                 "%s -tmux -procinfo -loops -n %s runstart > /dev/null 2>&1",
                 bin_path, fps_name);
    }
    else
    {
        snprintf(launch_cmd, sizeof(launch_cmd),
                 "%s -procinfo -loops -n %s runstart > /dev/null 2>&1 &",
                 bin_path, fps_name);
    }
    (void)run_sys_cmd(launch_cmd);

    cJSON_AddStringToObject(res, "status", "LAUNCHED");
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

    /* Normalize parameter: strip leading dot if present */
    const char *clean_param = raw_param;
    while (*clean_param == '.')
    {
        clean_param++;
    }

    char val_str[128];
    if (cJSON_IsString(val_item))
    {
        strncpy(val_str, val_item->valuestring, sizeof(val_str) - 1);
    }
    else if (cJSON_IsNumber(val_item))
    {
        snprintf(val_str, sizeof(val_str), "%.6f", val_item->valuedouble);
    }
    else if (cJSON_IsBool(val_item))
    {
        snprintf(val_str, sizeof(val_str), "%s", cJSON_IsTrue(val_item) ? "ON" : "OFF");
    }
    else
    {
        snprintf(val_str, sizeof(val_str), "0");
    }
    val_str[sizeof(val_str) - 1] = '\0';

    char set_cmd[1024];
    snprintf(set_cmd, sizeof(set_cmd), "milk-fps-set %s.%s %s 2>&1",
             fps_name, clean_param, val_str);

    FILE *pipe = popen(set_cmd, "r");
    char output[512] = "";
    if (pipe != NULL)
    {
        size_t n = fread(output, 1, sizeof(output) - 1, pipe);
        output[n] = '\0';
        pclose(pipe);
    }

    cJSON_AddStringToObject(res, "status", "SUCCESS");
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    cJSON_AddStringToObject(res, "param", clean_param);
    cJSON_AddStringToObject(res, "value", val_str);
    cJSON_AddStringToObject(res, "output", output);
    return 0;
} // mcp_tool_fps_set

int mcp_tool_fps_stop(
    const cJSON *args,
    cJSON       *res)
{
    const char *fps_name = "gric_cluster";
    if (args != NULL)
    {
        cJSON *n_item = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
        if (n_item != NULL && cJSON_IsString(n_item) && strlen(n_item->valuestring) > 0)
        {
            fps_name = n_item->valuestring;
        }
    }

    char bin_path[1024];
    get_fps_binary_path(bin_path, sizeof(bin_path));

    /* 1. Dispatch stop to tmux ctrl window (per tmux dispatch rules) */
    char tmux_cmd[512];
    snprintf(tmux_cmd, sizeof(tmux_cmd),
             "tmux send-keys -t %s:ctrl 'confstop' C-m 2>/dev/null", fps_name);
    (void)run_sys_cmd(tmux_cmd);

    snprintf(tmux_cmd, sizeof(tmux_cmd),
             "tmux send-keys -t %s:ctrl 'runstop' C-m 2>/dev/null", fps_name);
    (void)run_sys_cmd(tmux_cmd);

    /* 2. Direct standalone invocation fallback */
    char stop_cmd[2048];
    snprintf(stop_cmd, sizeof(stop_cmd), "%s %s:runstop > /dev/null 2>&1", bin_path, fps_name);
    (void)run_sys_cmd(stop_cmd);

    snprintf(stop_cmd, sizeof(stop_cmd), "%s %s:confstop > /dev/null 2>&1", bin_path, fps_name);
    (void)run_sys_cmd(stop_cmd);

    cJSON_AddStringToObject(res, "status", "STOPPED");
    cJSON_AddStringToObject(res, "fps_name", fps_name);
    return 0;
} // mcp_tool_fps_stop

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

    /* Decode 8-element telemetry vector if this is an _assign stream */
    if (image.md[0].size[0] == 8 && (image.md[0].naxis == 1 || image.md[0].size[1] == 1) &&
        image.md[0].datatype == _DATATYPE_FLOAT && image.array.raw != NULL)
    {
        const float *vec = (const float *)image.array.raw;
        cJSON *telemetry = cJSON_CreateObject();
        cJSON_AddNumberToObject(telemetry, "input_cnt0", (double)vec[0]);
        cJSON_AddNumberToObject(telemetry, "assigned_cluster_id", (double)vec[1]);
        cJSON_AddNumberToObject(telemetry, "distance_to_anchor", (double)vec[2]);
        cJSON_AddBoolToObject(telemetry, "is_new_anchor", (vec[3] > 0.5f));
        cJSON_AddNumberToObject(telemetry, "total_clusters", (double)vec[4]);
        cJSON_AddNumberToObject(telemetry, "processing_latency_us", (double)vec[5]);
        cJSON_AddNumberToObject(telemetry, "distance_evaluations", (double)vec[6]);
        cJSON_AddNumberToObject(telemetry, "quality_score", (double)vec[7]);
        cJSON_AddItemToObject(res, "telemetry_vector", telemetry);
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

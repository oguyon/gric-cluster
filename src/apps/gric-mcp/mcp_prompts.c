/**
 * @file mcp_prompts.c
 * @brief MCP Prompts and Resource Templates protocol handlers for gric-mcp.
 */

#include "mcp_prompts.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * create_error_response() - Format a standard JSON-RPC 2.0 error response object.
 * @id:      Request identifier (may be NULL, number, or string).
 * @code:    JSON-RPC error code.
 * @message: Descriptive error message.
 *
 * Return: Newly allocated cJSON object representing the error response.
 */
static cJSON *create_error_response(
    const cJSON *id,
    int          code,
    const char  *message)
{
    cJSON *resp = cJSON_CreateObject();
    if (resp == NULL)
    {
        return NULL;
    }

    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id != NULL)
    {
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    }
    else
    {
        cJSON_AddNullToObject(resp, "id");
    }

    cJSON *err = cJSON_CreateObject();
    cJSON_AddNumberToObject(err, "code", (double)code);
    cJSON_AddStringToObject(err, "message", message);
    cJSON_AddItemToObject(resp, "error", err);

    return resp;
} // create_error_response

/**
 * add_prompt_arg() - Helper to append an argument descriptor to a prompt definition.
 * @args_arr: Target cJSON array.
 * @name:     Argument name.
 * @desc:     Argument description.
 * @required: 1 if mandatory, 0 if optional.
 */
static void add_prompt_arg(
    cJSON      *args_arr,
    const char *name,
    const char *desc,
    int         required)
{
    cJSON *arg = cJSON_CreateObject();
    cJSON_AddStringToObject(arg, "name", name);
    cJSON_AddStringToObject(arg, "description", desc);
    cJSON_AddBoolToObject(arg, "required", required);
    cJSON_AddItemToArray(args_arr, arg);
} // add_prompt_arg

cJSON *mcp_prompts_list(
    const cJSON *id)
{
    cJSON *resp = cJSON_CreateObject();
    if (resp == NULL)
    {
        return NULL;
    }

    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id != NULL)
    {
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    }
    else
    {
        cJSON_AddNullToObject(resp, "id");
    }

    cJSON *result = cJSON_CreateObject();
    cJSON *prompts = cJSON_CreateArray();

    /* 1. calibrate-and-cluster */
    {
        cJSON *p = cJSON_CreateObject();
        cJSON_AddStringToObject(p, "name", "calibrate-and-cluster");
        cJSON_AddStringToObject(
            p, "description",
            "Guided workflow to inspect dataset density, calibrate radius, recommend "
            "parameters, and launch clustering");
        cJSON *args_arr = cJSON_CreateArray();
        add_prompt_arg(
            args_arr, "input_path", "Path to input dataset (.txt or .fits)", 1);
        add_prompt_arg(
            args_arr, "target_clusters", "Desired number of clusters (default: 100)", 0);
        cJSON_AddItemToObject(p, "arguments", args_arr);
        cJSON_AddItemToArray(prompts, p);
    }

    /* 2. setup-milk-stream */
    {
        cJSON *p = cJSON_CreateObject();
        cJSON_AddStringToObject(p, "name", "setup-milk-stream");
        cJSON_AddStringToObject(
            p, "description",
            "Step-by-step workflow to verify ImageStreamIO streams, launch standalone "
            "Milk FPS daemon, and monitor rolling telemetry");
        cJSON *args_arr = cJSON_CreateArray();
        add_prompt_arg(
            args_arr, "in_stream", "Input ImageStreamIO stream name", 1);
        add_prompt_arg(
            args_arr, "fps_name", "Unique FPS instance name", 0);
        cJSON_AddItemToObject(p, "arguments", args_arr);
        cJSON_AddItemToArray(prompts, p);
    }

    /* 3. compare-quantization */
    {
        cJSON *p = cJSON_CreateObject();
        cJSON_AddStringToObject(p, "name", "compare-quantization");
        cJSON_AddStringToObject(
            p, "description",
            "Workflow to benchmark clustering and k-NN accuracy across FP32, SQ8, SQ16, "
            "EQ16, RQ8, and RaBitQ quantization modes");
        cJSON *args_arr = cJSON_CreateArray();
        add_prompt_arg(
            args_arr, "input_path", "Path to input dataset (.txt, .fits, or .clusterdat)", 1);
        add_prompt_arg(
            args_arr, "quant_modes",
            "Comma-separated quantization modes to evaluate (e.g. 'fp32,sq8,eq16,rq8')", 0);
        cJSON_AddItemToObject(p, "arguments", args_arr);
        cJSON_AddItemToArray(prompts, p);
    }

    cJSON_AddItemToObject(result, "prompts", prompts);
    cJSON_AddItemToObject(resp, "result", result);
    return resp;
} // mcp_prompts_list

cJSON *mcp_prompts_get(
    const cJSON *id,
    const cJSON *params)
{
    if (params == NULL || !cJSON_IsObject(params))
    {
        return create_error_response(id, -32602, "Invalid params for prompts/get");
    }

    cJSON *name_item = cJSON_GetObjectItemCaseSensitive(params, "name");
    if (name_item == NULL || !cJSON_IsString(name_item))
    {
        return create_error_response(id, -32602, "Missing prompt name in prompts/get");
    }

    const char *name = name_item->valuestring;
    cJSON *args = cJSON_GetObjectItemCaseSensitive(params, "arguments");

    char prompt_text[2048];
    const char *prompt_desc = "";

    if (strcmp(name, "calibrate-and-cluster") == 0)
    {
        prompt_desc = "Guided dataset calibration and clustering workflow";
        const char *input_path = "dataset.fits";
        const char *target_cl = "100";

        if (args != NULL)
        {
            cJSON *ip = cJSON_GetObjectItemCaseSensitive(args, "input_path");
            if (ip != NULL && cJSON_IsString(ip) && ip->valuestring[0] != '\0')
            {
                input_path = ip->valuestring;
            }
            cJSON *tc = cJSON_GetObjectItemCaseSensitive(args, "target_clusters");
            if (tc != NULL && cJSON_IsString(tc) && tc->valuestring[0] != '\0')
            {
                target_cl = tc->valuestring;
            }
        }

        snprintf(
            prompt_text, sizeof(prompt_text),
            "Execute a complete GRIC clustering calibration and run pipeline:\n"
            "1. Dataset Inspection: Call `gric_probe_dataset` with path='%s' to verify "
            "geometry, frame count, and sparsity.\n"
            "2. Radius Calibration: Call `gric_calibrate_radius` with input_path='%s' and "
            "target_clusters=%s to determine optimal rlim.\n"
            "3. Parameter Advisory: Call `gric_recommend_params` using the dataset geometry and "
            "desired accuracy.\n"
            "4. Validation: Verify CLI flags with `gric_validate_command` before launching.\n"
            "5. Execution: Launch clustering job via `gric_cluster_start` and track convergence "
            "using `gric_job_status`.",
            input_path, input_path, target_cl);
    }
    else if (strcmp(name, "setup-milk-stream") == 0)
    {
        prompt_desc = "Milk ImageStreamIO real-time streaming setup and monitoring";
        const char *in_stream = "shm_in";
        const char *fps_name = "gric_stream";

        if (args != NULL)
        {
            cJSON *is = cJSON_GetObjectItemCaseSensitive(args, "in_stream");
            if (is != NULL && cJSON_IsString(is) && is->valuestring[0] != '\0')
            {
                in_stream = is->valuestring;
            }
            cJSON *fn = cJSON_GetObjectItemCaseSensitive(args, "fps_name");
            if (fn != NULL && cJSON_IsString(fn) && fn->valuestring[0] != '\0')
            {
                fps_name = fn->valuestring;
            }
        }

        snprintf(
            prompt_text, sizeof(prompt_text),
            "Set up and monitor a real-time Milk ImageStreamIO clustering daemon:\n"
            "1. Stream Verification: Call `gric_probe_shm` with stream_name='%s' to confirm SHM "
            "allocation and dimensions.\n"
            "2. Launch Daemon: Call `gric_fps_run` with in_name='%s', fps_name='%s', "
            "and calibrated rlim.\n"
            "3. Status & Parameters: Inspect running state via `gric_fps_status` with "
            "fps_name='%s'.\n"
            "4. Dynamic Configuration: Fine-tune parameters via `gric_fps_set` (e.g. deltaprob, "
            "maxnbclust).\n"
            "5. Telemetry & Latency: Monitor rolling window latency percentiles with "
            "`gric_stream_window` on '%s_clust'.",
            in_stream, in_stream, fps_name, fps_name, in_stream);
    }
    else if (strcmp(name, "compare-quantization") == 0)
    {
        prompt_desc = "Quantization accuracy and performance benchmark workflow";
        const char *input_path = "dataset.fits";
        const char *quant_modes = "fp32,sq8,eq16,rq8";

        if (args != NULL)
        {
            cJSON *ip = cJSON_GetObjectItemCaseSensitive(args, "input_path");
            if (ip != NULL && cJSON_IsString(ip) && ip->valuestring[0] != '\0')
            {
                input_path = ip->valuestring;
            }
            cJSON *qm = cJSON_GetObjectItemCaseSensitive(args, "quant_modes");
            if (qm != NULL && cJSON_IsString(qm) && qm->valuestring[0] != '\0')
            {
                quant_modes = qm->valuestring;
            }
        }

        snprintf(
            prompt_text, sizeof(prompt_text),
            "Benchmark quantization efficiency and accuracy across modes (%s) for '%s':\n"
            "1. Baseline Run: Launch unquantized baseline clustering job using "
            "`gric_cluster_start` with outdir='/tmp/gric_run_fp32'.\n"
            "2. Quantized Runs: Execute clustering runs for each requested quantization mode "
            "(e.g. -sq8, -eq16, -rq8).\n"
            "3. Metric Comparison: Compare cluster topology and centroids using "
            "`gric_compare_runs` between baseline and quantized outputs.\n"
            "4. k-NN Quality Audit: Evaluate query recall and relative distance error using "
            "`gric_knn_quality`.",
            quant_modes, input_path);
    }
    else
    {
        return create_error_response(id, -32602, "Unknown prompt requested");
    }

    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id != NULL)
    {
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    }
    else
    {
        cJSON_AddNullToObject(resp, "id");
    }

    cJSON *result = cJSON_CreateObject();
    cJSON_AddStringToObject(result, "description", prompt_desc);

    cJSON *messages = cJSON_CreateArray();
    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "role", "user");

    cJSON *content = cJSON_CreateObject();
    cJSON_AddStringToObject(content, "type", "text");
    cJSON_AddStringToObject(content, "text", prompt_text);
    cJSON_AddItemToObject(msg, "content", content);

    cJSON_AddItemToArray(messages, msg);
    cJSON_AddItemToObject(result, "messages", messages);

    cJSON_AddItemToObject(resp, "result", result);
    return resp;
} // mcp_prompts_get

cJSON *mcp_resource_templates_list(
    const cJSON *id)
{
    cJSON *resp = cJSON_CreateObject();
    if (resp == NULL)
    {
        return NULL;
    }

    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id != NULL)
    {
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    }
    else
    {
        cJSON_AddNullToObject(resp, "id");
    }

    cJSON *result = cJSON_CreateObject();
    cJSON *templates = cJSON_CreateArray();

    /* Template 1: Run summary */
    {
        cJSON *t = cJSON_CreateObject();
        cJSON_AddStringToObject(t, "uriTemplate", "gric://run/{path}/summary");
        cJSON_AddStringToObject(t, "name", "Clustering Run Summary");
        cJSON_AddStringToObject(
            t, "description",
            "Detailed execution summary, cluster counts, and compression metrics for "
            "run directory");
        cJSON_AddStringToObject(t, "mimeType", "application/json");
        cJSON_AddItemToArray(templates, t);
    }

    /* Template 2: Stream stats */
    {
        cJSON *t = cJSON_CreateObject();
        cJSON_AddStringToObject(t, "uriTemplate", "gric://stream/{name}/stats");
        cJSON_AddStringToObject(t, "name", "ImageStreamIO Stream Telemetry");
        cJSON_AddStringToObject(
            t, "description",
            "Real-time frame rates, dropped frames, and ring buffer statistics for "
            "shared memory stream");
        cJSON_AddStringToObject(t, "mimeType", "application/json");
        cJSON_AddItemToArray(templates, t);
    }

    /* Template 3: Recipe instructions */
    {
        cJSON *t = cJSON_CreateObject();
        cJSON_AddStringToObject(t, "uriTemplate", "gric://recipe/{name}");
        cJSON_AddStringToObject(t, "name", "GRIC Procedural Recipe");
        cJSON_AddStringToObject(
            t, "description",
            "Step-by-step cookbook instructions and CLI workflows");
        cJSON_AddStringToObject(t, "mimeType", "text/markdown");
        cJSON_AddItemToArray(templates, t);
    }

    cJSON_AddItemToObject(result, "resourceTemplates", templates);
    cJSON_AddItemToObject(resp, "result", result);
    return resp;
} // mcp_resource_templates_list

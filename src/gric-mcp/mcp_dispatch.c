/**
 * @file mcp_dispatch.c
 * @brief Top-level JSON-RPC 2.0 protocol router for gric-mcp server.
 */

#include "mcp_dispatch.h"
#include "mcp_tools.h"
#include "shared/cjson/cJSON.h"
#include "shared/help_topics.h"
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
    cJSON_AddNumberToObject(err, "code", code);
    cJSON_AddStringToObject(err, "message", message);
    cJSON_AddItemToObject(resp, "error", err);

    return resp;
} // create_error_response

/**
 * handle_initialize() - Handle MCP initialize handshake.
 * @id:     Request ID.
 * @params: Request parameters object.
 *
 * Return: JSON-RPC response object.
 */
static cJSON *handle_initialize(
    const cJSON *id,
    const cJSON *params)
{
    (void)params;
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
    cJSON_AddStringToObject(result, "protocolVersion", "2024-11-05");

    cJSON *capabilities = cJSON_CreateObject();
    cJSON_AddItemToObject(capabilities, "tools", cJSON_CreateObject());
    cJSON_AddItemToObject(capabilities, "resources", cJSON_CreateObject());
    cJSON_AddItemToObject(result, "capabilities", capabilities);

    cJSON *server_info = cJSON_CreateObject();
    cJSON_AddStringToObject(server_info, "name", "gric-mcp");
    cJSON_AddStringToObject(server_info, "version", "1.0.0");
    cJSON_AddItemToObject(result, "serverInfo", server_info);

    cJSON_AddItemToObject(resp, "result", result);
    return resp;
} // handle_initialize

/**
 * handle_tools_list() - Handle tools/list request.
 * @id: Request ID.
 *
 * Return: JSON-RPC response object.
 */
static cJSON *handle_tools_list(
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
    cJSON_AddItemToObject(result, "tools", mcp_tools_get_list());
    cJSON_AddItemToObject(resp, "result", result);

    return resp;
} // handle_tools_list

/**
 * handle_tools_call() - Execute an MCP tool and format response.
 * @id:     Request ID.
 * @params: Request parameters containing name and arguments.
 *
 * Return: JSON-RPC response object.
 */
static cJSON *handle_tools_call(
    const cJSON *id,
    const cJSON *params)
{
    if (params == NULL || !cJSON_IsObject(params))
    {
        return create_error_response(id, -32602, "Invalid params for tools/call");
    }

    cJSON *name_item = cJSON_GetObjectItemCaseSensitive(params, "name");
    if (name_item == NULL || !cJSON_IsString(name_item))
    {
        return create_error_response(id, -32602, "Missing tool name in tools/call");
    }
    const char *tool_name = name_item->valuestring;

    cJSON *args = cJSON_GetObjectItemCaseSensitive(params, "arguments");
    cJSON *tool_result = cJSON_CreateObject();

    int status = -1;
    if (strcmp(tool_name, "gric_audit_code_style") == 0)
    {
        status = mcp_tool_audit_code_style(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_align_parameters") == 0)
    {
        status = mcp_tool_align_parameters(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_inspect_simd") == 0)
    {
        status = mcp_tool_inspect_simd(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_verify_invariants") == 0)
    {
        status = mcp_tool_verify_invariants(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_inspect_run") == 0)
    {
        status = mcp_tool_inspect_run(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_probe_dataset") == 0)
    {
        status = mcp_tool_probe_dataset(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_probe_shm") == 0)
    {
        status = mcp_tool_probe_shm(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_help") == 0)
    {
        status = mcp_tool_help(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_list_suite") == 0)
    {
        status = mcp_tool_list_suite(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_get_recipe") == 0)
    {
        status = mcp_tool_get_recipe(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_fps_status") == 0)
    {
        status = mcp_tool_fps_status(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_fps_run") == 0)
    {
        status = mcp_tool_fps_run(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_fps_set") == 0)
    {
        status = mcp_tool_fps_set(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_fps_stop") == 0)
    {
        status = mcp_tool_fps_stop(args, tool_result);
    }
    else if (strcmp(tool_name, "gric_probe_fps_streams") == 0)
    {
        status = mcp_tool_probe_fps_streams(args, tool_result);
    }
    else
    {
        cJSON_Delete(tool_result);
        return create_error_response(id, -32601, "Unknown tool requested");
    }

    char *result_text = cJSON_PrintUnformatted(tool_result);
    cJSON_Delete(tool_result);

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

    cJSON *result_obj = cJSON_CreateObject();
    cJSON *content_arr = cJSON_CreateArray();
    cJSON *content_item = cJSON_CreateObject();

    cJSON_AddStringToObject(content_item, "type", "text");
    cJSON_AddStringToObject(content_item, "text", result_text ? result_text : "{}");
    cJSON_AddItemToArray(content_arr, content_item);

    cJSON_AddItemToObject(result_obj, "content", content_arr);
    cJSON_AddBoolToObject(result_obj, "isError", (status != 0));
    cJSON_AddItemToObject(resp, "result", result_obj);

    if (result_text != NULL)
    {
        free(result_text);
    }

    return resp;
} // handle_tools_call

/**
 * handle_resources_list() - Handle resources/list request.
 * @id: Request ID.
 *
 * Return: JSON-RPC response object.
 */
static cJSON *handle_resources_list(
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
    cJSON *resources = cJSON_CreateArray();

    /* 1. Core overview and cheatsheet resources */
    {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddStringToObject(r, "uri", "gric://overview");
        cJSON_AddStringToObject(r, "name", "GRIC Architecture & Overview");
        cJSON_AddStringToObject(r, "description", "High-level overview of GRIC clustering engine");
        cJSON_AddStringToObject(r, "mimeType", "text/markdown");
        cJSON_AddItemToArray(resources, r);
    }
    {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddStringToObject(r, "uri", "gric://cheatsheet/cli");
        cJSON_AddStringToObject(r, "name", "GRIC CLI Quick Reference");
        cJSON_AddStringToObject(r, "description",
                                "Command-line syntax, flags, and parameter rules");
        cJSON_AddStringToObject(r, "mimeType", "text/markdown");
        cJSON_AddItemToArray(resources, r);
    }
    {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddStringToObject(r, "uri", "gric://milk/overview");
        cJSON_AddStringToObject(r, "name", "Milk Framework Integration");
        cJSON_AddStringToObject(r, "description",
                                "Guide to Milk CLI, standalone FPS daemon, and SHM");
        cJSON_AddStringToObject(r, "mimeType", "text/markdown");
        cJSON_AddItemToArray(resources, r);
    }
    {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddStringToObject(r, "uri", "gric://milk/fps_params");
        cJSON_AddStringToObject(r, "name", "Milk FPS Parameters Reference");
        cJSON_AddStringToObject(r, "description",
                                "Dictionary of all 16 parameters in gric_cluster FPS");
        cJSON_AddStringToObject(r, "mimeType", "application/json");
        cJSON_AddItemToArray(resources, r);
    }

    /* 2. Embedded Help Topics */
    const struct HelpTopicEntry *table = help_topic_get_table();
    if (table != NULL)
    {
        for (size_t ii = 0; table[ii].keyword != NULL; ii++)
        {
            char uri[256];
            snprintf(uri, sizeof(uri), "gric://help/%s", table[ii].keyword);

            cJSON *r = cJSON_CreateObject();
            cJSON_AddStringToObject(r, "uri", uri);
            cJSON_AddStringToObject(r, "name", table[ii].keyword);
            cJSON_AddStringToObject(r, "description",
                                    "Documentation topic from embedded help database");
            cJSON_AddStringToObject(r, "mimeType", "text/markdown");
            cJSON_AddItemToArray(resources, r);
        } // for ii
    }

    cJSON_AddItemToObject(result, "resources", resources);
    cJSON_AddItemToObject(resp, "result", result);
    return resp;
} // handle_resources_list

/**
 * handle_resources_read() - Handle resources/read request.
 * @id:     Request ID.
 * @params: Request parameters containing uri.
 *
 * Return: JSON-RPC response object.
 */
static cJSON *handle_resources_read(
    const cJSON *id,
    const cJSON *params)
{
    if (params == NULL || !cJSON_IsObject(params))
    {
        return create_error_response(id, -32602, "Invalid params for resources/read");
    }

    cJSON *uri_item = cJSON_GetObjectItemCaseSensitive(params, "uri");
    if (uri_item == NULL || !cJSON_IsString(uri_item))
    {
        return create_error_response(id, -32602, "Missing uri in resources/read");
    }
    const char *uri = uri_item->valuestring;

    const char *content_text = NULL;
    char *allocated_content = NULL;
    const char *mime_type = "text/markdown";

    if (strcmp(uri, "gric://overview") == 0)
    {
        content_text =
            "# GRIC: Geometric Real-Time Image Clustering\n\n"
            "GRIC is an ultra-high-speed, distance-based clustering engine designed for\n"
            "sequential image streams and high-dimensional vector data. It leverages metric space\n"
            "triangle inequality bounds (TE3, TE4, TE5), E8 lattice quantization (EQ16), and\n"
            "out-of-core nearest neighbors to achieve strictly bounded memory and low latency.\n";
    }
    else if (strcmp(uri, "gric://cheatsheet/cli") == 0)
    {
        content_text =
            "# GRIC CLI Cheatsheet\n\n"
            "## Core Clustering:\n"
            "  gric-cluster <rlim> <input> [options]\n"
            "  - a1.0: Auto-scale radius to median distance\n"
            "  - -maxcl <N>: Cluster allocation budget (e.g. 1000)\n"
            "  - -tiles 2x2: Spatial tiling for localized image features\n"
            "  - -te4: 4-point metric pruning\n"
            "  - -eq16: 16-bit E8 lattice quantization acceleration\n"
            "  - -entropy: Maximize Shannon information gain\n"
            "  - -outdir <dir>: Result directory (.clusterdat)\n\n"
            "## k-Nearest Neighbors:\n"
            "  gric-knn <input> <cluster_dir> -k 10 [options]\n"
            "  - -multipivot: AESA multi-anchor distance bounding\n"
            "  - -angular: Cosine directional bounding\n";
    }
    else if (strcmp(uri, "gric://milk/overview") == 0)
    {
        content_text = help_topic_lookup("milk");
    }
    else if (strcmp(uri, "gric://milk/fps_params") == 0)
    {
        mime_type = "application/json";
        content_text =
            "{\n"
            "  \"parameters\": [\n"
            "    {\"name\": \".in_name\", \"type\": \"STREAMNAME\", "
            "\"description\": \"Input ImageStreamIO stream\"},\n"
            "    {\"name\": \".out_name\", \"type\": \"STRING\", "
            "\"description\": \"Output assignment stream (<out>_assign)\"},\n"
            "    {\"name\": \".out_anchors\", \"type\": \"STRING\", "
            "\"description\": \"Output centroids stream\"},\n"
            "    {\"name\": \".out_counts\", \"type\": \"STRING\", "
            "\"description\": \"Output cluster counts stream\"},\n"
            "    {\"name\": \".stream_anchors\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"Publish anchors live\"},\n"
            "    {\"name\": \".stream_counts\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"Publish counts live\"},\n"
            "    {\"name\": \".allow_frame_drop\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"1=jump to latest frame\"},\n"
            "    {\"name\": \".rlim\", \"type\": \"FLOAT64\", \"default\": 0.5, "
            "\"description\": \"Cluster radius threshold\"},\n"
            "    {\"name\": \".deltaprob\", \"type\": \"FLOAT64\", \"default\": 0.01, "
            "\"description\": \"Neighbor search cutoff\"},\n"
            "    {\"name\": \".maxnbclust\", \"type\": \"UINT32\", \"default\": 256, "
            "\"description\": \"Maximum cluster capacity\"},\n"
            "    {\"name\": \".maxcl_strategy\", \"type\": \"INT64\", \"default\": 0, "
            "\"description\": \"0=stop, 1=discard\"},\n"
            "    {\"name\": \".ncpu\", \"type\": \"UINT32\", \"default\": 0, "
            "\"description\": \"Thread count (0=auto)\"},\n"
            "    {\"name\": \".use_double\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"64-bit float compute\"},\n"
            "    {\"name\": \".use_sq16\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"16-bit scalar quantization\"},\n"
            "    {\"name\": \".entropy_mode\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"Shannon entropy search\"},\n"
            "    {\"name\": \".reset_state\", \"type\": \"ONOFF\", \"default\": \"OFF\", "
            "\"description\": \"Dynamic reset trigger\"}\n"
            "  ]\n"
            "}";
    }
    else if (strncmp(uri, "gric://help/", 12) == 0)
    {
        const char *topic_name = uri + 12;
        content_text = help_topic_lookup(topic_name);
    }
    else if (strncmp(uri, "gric://recipes/", 15) == 0)
    {
        const char *recipe_name = uri + 15;
        cJSON *r_args = cJSON_CreateObject();
        cJSON *r_res = cJSON_CreateObject();
        cJSON_AddStringToObject(r_args, "recipe", recipe_name);
        mcp_tool_get_recipe(r_args, r_res);
        cJSON_Delete(r_args);

        cJSON *inst = cJSON_GetObjectItemCaseSensitive(r_res, "instructions");
        if (inst != NULL && cJSON_IsString(inst))
        {
            allocated_content = strdup(inst->valuestring);
            content_text = allocated_content;
        }
        cJSON_Delete(r_res);
    }

    if (content_text == NULL)
    {
        if (allocated_content != NULL)
        {
            free(allocated_content);
        }
        return create_error_response(id, -32602, "Resource not found");
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
    cJSON *contents = cJSON_CreateArray();
    cJSON *item = cJSON_CreateObject();
    cJSON_AddStringToObject(item, "uri", uri);
    cJSON_AddStringToObject(item, "mimeType", mime_type);
    cJSON_AddStringToObject(item, "text", content_text);
    cJSON_AddItemToArray(contents, item);

    cJSON_AddItemToObject(result, "contents", contents);
    cJSON_AddItemToObject(resp, "result", result);

    if (allocated_content != NULL)
    {
        free(allocated_content);
    }

    return resp;
} // handle_resources_read

char *mcp_dispatch_message(
    const char *input_json)
{
    if (input_json == NULL || strlen(input_json) == 0)
    {
        return NULL;
    }

    cJSON *req = cJSON_Parse(input_json);
    if (req == NULL || !cJSON_IsObject(req))
    {
        cJSON *err = create_error_response(NULL, -32700, "Parse error");
        char *err_str = cJSON_PrintUnformatted(err);
        cJSON_Delete(err);
        if (req != NULL)
        {
            cJSON_Delete(req);
        }
        return err_str;
    }

    cJSON *method_item = cJSON_GetObjectItemCaseSensitive(req, "method");
    cJSON *id_item = cJSON_GetObjectItemCaseSensitive(req, "id");
    cJSON *params_item = cJSON_GetObjectItemCaseSensitive(req, "params");

    /* Notification check: notifications do not contain an id */
    int is_notification = (id_item == NULL);

    if (method_item == NULL || !cJSON_IsString(method_item))
    {
        cJSON_Delete(req);
        if (is_notification)
        {
            return NULL;
        }
        cJSON *err = create_error_response(id_item, -32600, "Invalid Request");
        char *err_str = cJSON_PrintUnformatted(err);
        cJSON_Delete(err);
        return err_str;
    }

    const char *method = method_item->valuestring;
    cJSON *resp_obj = NULL;

    if (strcmp(method, "initialize") == 0)
    {
        resp_obj = handle_initialize(id_item, params_item);
    }
    else if (strcmp(method, "notifications/initialized") == 0)
    {
        /* MCP notification: no response required */
        cJSON_Delete(req);
        return NULL;
    }
    else if (strcmp(method, "tools/list") == 0)
    {
        resp_obj = handle_tools_list(id_item);
    }
    else if (strcmp(method, "tools/call") == 0)
    {
        resp_obj = handle_tools_call(id_item, params_item);
    }
    else if (strcmp(method, "resources/list") == 0)
    {
        resp_obj = handle_resources_list(id_item);
    }
    else if (strcmp(method, "resources/read") == 0)
    {
        resp_obj = handle_resources_read(id_item, params_item);
    }
    else if (strcmp(method, "ping") == 0)
    {
        resp_obj = cJSON_CreateObject();
        cJSON_AddStringToObject(resp_obj, "jsonrpc", "2.0");
        if (id_item != NULL)
        {
            cJSON_AddItemToObject(resp_obj, "id", cJSON_Duplicate(id_item, 1));
        }
        cJSON_AddItemToObject(resp_obj, "result", cJSON_CreateObject());
    }
    else
    {
        if (!is_notification)
        {
            resp_obj = create_error_response(id_item, -32601, "Method not found");
        }
    }

    cJSON_Delete(req);

    if (resp_obj == NULL)
    {
        return NULL;
    }

    char *out_str = cJSON_PrintUnformatted(resp_obj);
    cJSON_Delete(resp_obj);
    return out_str;
} // mcp_dispatch_message

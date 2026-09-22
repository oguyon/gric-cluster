/**
 * @file mcp_dispatch.c
 * @brief Top-level JSON-RPC 2.0 protocol router for gric-mcp server.
 */

#include "mcp_dispatch.h"
#include "mcp_tools.h"
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

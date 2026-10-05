/**
 * @file mcp_prompts.h
 * @brief MCP Prompts and Resource Templates protocol handlers for gric-mcp.
 */

#ifndef MCP_PROMPTS_H
#define MCP_PROMPTS_H

#include "shared/cjson/cJSON.h"

/**
 * mcp_prompts_list() - Handle prompts/list request.
 * @id: Request ID.
 *
 * Return: Newly allocated JSON-RPC response object.
 */
cJSON *mcp_prompts_list(
    const cJSON *id);

/**
 * mcp_prompts_get() - Handle prompts/get request.
 * @id:     Request ID.
 * @params: Request parameters containing prompt name and arguments.
 *
 * Return: Newly allocated JSON-RPC response object.
 */
cJSON *mcp_prompts_get(
    const cJSON *id,
    const cJSON *params);

/**
 * mcp_resource_templates_list() - Handle resources/templates/list request.
 * @id: Request ID.
 *
 * Return: Newly allocated JSON-RPC response object.
 */
cJSON *mcp_resource_templates_list(
    const cJSON *id);

#endif // MCP_PROMPTS_H

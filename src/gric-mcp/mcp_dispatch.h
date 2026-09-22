/**
 * @file mcp_dispatch.h
 * @brief Dispatcher and protocol router for JSON-RPC 2.0 MCP requests.
 */

#ifndef MCP_DISPATCH_H
#define MCP_DISPATCH_H

/**
 * mcp_dispatch_message() - Process a raw JSON-RPC 2.0 message and generate response.
 * @input_json: Raw JSON message string received from stdin.
 *
 * Return: Heap-allocated JSON response string (caller must free), or NULL if notification.
 */
char *mcp_dispatch_message(
    const char *input_json);

#endif // MCP_DISPATCH_H

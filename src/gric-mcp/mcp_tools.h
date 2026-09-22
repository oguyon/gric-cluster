/**
 * @file mcp_tools.h
 * @brief Declarations and schemas for GRIC Model Context Protocol tools.
 */

#ifndef MCP_TOOLS_H
#define MCP_TOOLS_H

#include "shared/cjson/cJSON.h"

/**
 * mcp_tool_audit_code_style() - Audit C source against style and architectural rules.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_audit_code_style(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_align_parameters() - Check or generate column-aligned parameter prototypes.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_align_parameters(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_inspect_simd() - Compile C code to assembly and audit vectorization instructions.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_inspect_simd(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_verify_invariants() - Audit cluster boundary and radius invariants with SIMD.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_verify_invariants(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_inspect_run() - Read cluster run directory and return structured telemetry.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_inspect_run(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_probe_dataset() - Probe raw dataset geometry and recommend clustering parameters.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_probe_dataset(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_probe_shm() - Query ImageStreamIO shared-memory stream metadata non-blockingly.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_probe_shm(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tools_get_list() - Build the full cJSON array of all tool definitions and schemas.
 *
 * Return: Newly allocated cJSON array containing tool descriptors.
 */
cJSON *mcp_tools_get_list(void);

/**
 * mcp_get_project_root() - Retrieve the repository root directory.
 * @buf:  Target buffer to store the root directory path.
 * @size: Buffer capacity.
 */
void mcp_get_project_root(
    char   *buf,
    size_t  size);

/**
 * mcp_resolve_path() - Resolve a relative or absolute file path within the project.
 * @input_path: Input path string (may be relative to project root or CWD).
 * @out_buf:    Target buffer for resolved path.
 * @out_size:   Buffer capacity.
 */
void mcp_resolve_path(
    const char *input_path,
    char       *out_buf,
    size_t      out_size);

#endif // MCP_TOOLS_H

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
 * mcp_tool_help() - Query compiled embedded help topics database by keyword or search.
 * @args:   JSON object containing optional topic or query.
 * @res:    Output JSON object containing matched topic markdown or catalog.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_help(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_list_suite() - Retrieve structured catalog of all executables in the GRIC suite.
 * @args:   Unused / NULL.
 * @res:    Output JSON object containing array of programs with summaries and usages.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_list_suite(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_get_recipe() - Retrieve step-by-step procedural cookbook recipes.
 * @args:   JSON object containing optional recipe name.
 * @res:    Output JSON object containing formatted instructions or available recipes.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_get_recipe(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_fps_status() - Inspect running state and parameter values of a Milk FPS instance.
 * @args:   JSON object containing optional fps_name.
 * @res:    Output JSON object with active status, PID, and parameter dictionary.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_fps_status(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_fps_run() - Launch a standalone milk-fpsexec-gric-cluster streaming daemon.
 * @args:   JSON object containing in_name, out_name, rlim, and options.
 * @res:    Output JSON object confirming launch parameters.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_fps_run(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_fps_set() - Dynamically update a parameter on a running Milk FPS instance.
 * @args:   JSON object containing fps_name, param, and value.
 * @res:    Output JSON object confirming update.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_fps_set(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_fps_stop() - Gracefully stop an active Milk FPS instance (runstop and confstop).
 * @args:   JSON object containing fps_name.
 * @res:    Output JSON object confirming termination.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_fps_stop(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_probe_fps_streams() - Non-blocking inspection of Milk output streams and telemetry.
 * @args:   JSON object containing out_name / stream_name.
 * @res:    Output JSON object decoding the 8-element telemetry vector.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_probe_fps_streams(
    const cJSON *args,
    cJSON       *res);

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
    cJSON       *out);

/**
 * mcp_tool_server_info() - Retrieve server runtime status, active toolsets, and flags.
 * @args:   JSON object containing optional arguments.
 * @res:    Output JSON object with version, toolsets, and read_only status.
 *
 * Return: 0 on success, -1 on error.
 */
int mcp_tool_server_info(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_dev_build_test() - Incremental compilation and CTest runner with diagnostic parsing.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing the result content or error details.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_dev_build_test(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_parse_compiler_diagnostics() - Parse compiler output into warning and error lists.
 * @output:       Compiler stdout/stderr string.
 * @warnings_arr: cJSON array to append parsed warnings to.
 * @errors_arr:   cJSON array to append parsed errors to.
 */
void mcp_parse_compiler_diagnostics(
    const char *output,
    cJSON      *warnings_arr,
    cJSON      *errors_arr);

/**
 * mcp_parse_ctest_output() - Parse ctest output into summary and failed test list.
 * @output:       CTest stdout/stderr string.
 * @summary_obj:  cJSON object to populate with test run counts and duration.
 * @failures_arr: cJSON array to append failed test details to.
 */
void mcp_parse_ctest_output(
    const char *output,
    cJSON      *summary_obj,
    cJSON      *failures_arr);


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

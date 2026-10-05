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
 * mcp_tool_dev_golden_compare() - Rapid regression check against golden baseline outputs.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing comparison metrics and status.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_dev_golden_compare(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_dev_bench() - Deterministic micro-benchmark runner with statistical aggregation.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing benchmark results and comparison.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_dev_bench(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_cluster_start() - Launch asynchronous gric-cluster background job.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing job_id and initial status.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_cluster_start(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_knn_start() - Launch asynchronous gric-knn background job.
 * @args:   JSON object containing tool arguments.
 * @res:    Output JSON object containing job_id and initial status.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_knn_start(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_job_status() - Check asynchronous job state, runtime progress %, and frame counts.
 * @args:   JSON object containing job_id.
 * @res:    Output JSON object containing job status and telemetry.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_job_status(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_job_result() - Retrieve structured execution metrics and telemetry once completed.
 * @args:   JSON object containing job_id.
 * @res:    Output JSON object containing run metrics.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_job_result(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_job_cancel() - Cancel a running asynchronous job and optionally clean outputs.
 * @args:   JSON object containing job_id and optional clean_output flag.
 * @res:    Output JSON object containing cancellation confirmation.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_job_cancel(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_recommend_params() - Recommend clustering parameters based on dataset and constraints.
 * @args:   JSON object containing dataset_path and optional parameters.
 * @res:    Output JSON object containing recommendations and rationale.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_recommend_params(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_calibrate_radius() - Calibrate radius thresholds from distance percentiles.
 * @args:   JSON object containing dataset_path and optional sample_size.
 * @res:    Output JSON object containing percentiles and radius presets.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_calibrate_radius(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_validate_command() - Validate CLI commands and flag compatibility.
 * @args:   JSON object containing command string or binary and args.
 * @res:    Output JSON object containing validation results, errors, and warnings.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_validate_command(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_compare_runs() - Compare performance and stability across two clustering runs.
 * @args:   JSON object containing run_dir_a and run_dir_b.
 * @res:    Output JSON object containing ARI, agreement %, cluster delta, and report.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_compare_runs(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_cluster_detail() - Deep inspection of a specific cluster's geometry and dynamics.
 * @args:   JSON object containing run_dir and cluster_id.
 * @res:    Output JSON object containing member counts, lifetime, and dynamics.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_cluster_detail(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_knn_quality() - Evaluate recall and distance approximation of kNN search results.
 * @args:   JSON object containing dataset_path and knn_output.
 * @res:    Output JSON object containing recall, MRR, distance ratios, and report.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_knn_quality(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_plot() - Generate diagnostic visual plots using gric-plot.
 * @args:   JSON object containing run_dir, points_path, log_path, output_path, format.
 * @res:    Output JSON object containing generated image paths and clustering summary.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_plot(
    const cJSON *args,
    cJSON       *res);

/**
 * mcp_tool_stream_window() - Sample and compute rolling statistics over an ImageStreamIO stream.
 * @args:   JSON object containing stream_name, num_frames, timeout_ms.
 * @res:    Output JSON object containing latency percentiles, anchor rates, and diagnostics.
 *
 * Return: 0 on success, -1 on fatal failure.
 */
int mcp_tool_stream_window(
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

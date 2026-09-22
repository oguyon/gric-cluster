/**
 * @file test_mcp_protocol.c
 * @brief Integration tests for gric-mcp JSON-RPC 2.0 protocol and tools.
 */

#include "gric-mcp/mcp_dispatch.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_initialize() - Verify MCP initialize handshake response.
 */
static void test_initialize(void)
{
    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "\"name\":\"gric-mcp\"") != NULL);
    assert(strstr(resp, "\"protocolVersion\":\"2024-11-05\"") != NULL);
    assert(strstr(resp, "\"tools\"") != NULL);
    free(resp);
    printf("PASS: test_initialize\n");
} // test_initialize

/**
 * test_ping() - Verify JSON-RPC ping method.
 */
static void test_ping(void)
{
    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "\"result\":{}") != NULL);
    free(resp);
    printf("PASS: test_ping\n");
} // test_ping

/**
 * test_tools_list() - Verify tools/list exposes all expected tools and schemas.
 */
static void test_tools_list(void)
{
    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\"}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "gric_audit_code_style") != NULL);
    assert(strstr(resp, "gric_align_parameters") != NULL);
    assert(strstr(resp, "gric_inspect_simd") != NULL);
    assert(strstr(resp, "gric_verify_invariants") != NULL);
    assert(strstr(resp, "gric_inspect_run") != NULL);
    assert(strstr(resp, "gric_probe_dataset") != NULL);
    assert(strstr(resp, "gric_probe_shm") != NULL);
    free(resp);
    printf("PASS: test_tools_list\n");
} // test_tools_list

/**
 * test_align_parameters() - Test parameter column-alignment tool.
 */
static void test_align_parameters(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_align_parameters\","
        "\"arguments\":{\"prototype\":\"int test_func(\\n    int a,\\n"
        "    const char *long_name\\n)\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "FORMATTED") != NULL);
    assert(strstr(resp, "alignment_column") != NULL);
    free(resp);
    printf("PASS: test_align_parameters\n");
} // test_align_parameters

/**
 * test_audit_code_style() - Test code style auditor on existing C source file.
 */
static void test_audit_code_style(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_audit_code_style\","
        "\"arguments\":{\"file_path\":\"src/gric-mcp/gric-mcp.c\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "PASS") != NULL);
    assert(strstr(resp, "total_lines") != NULL);
    free(resp);
    printf("PASS: test_audit_code_style\n");
} // test_audit_code_style

/**
 * test_inspect_run() - Test inspect_run tool on 3Dspiral.clusterdat benchmark run.
 */
static void test_inspect_run(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_inspect_run\","
        "\"arguments\":{\"run_dir\":\"3Dspiral.clusterdat\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "num_clusters") != NULL);
    assert(strstr(resp, "num_frames") != NULL);
    free(resp);
    printf("PASS: test_inspect_run\n");
} // test_inspect_run

/**
 * test_verify_invariants() - Test invariant verification on 3Dspiral.clusterdat.
 */
static void test_verify_invariants(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_verify_invariants\","
        "\"arguments\":{\"run_dir\":\"3Dspiral.clusterdat\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "PASS") != NULL);
    assert(strstr(resp, "violations_count") != NULL);
    free(resp);
    printf("PASS: test_verify_invariants\n");
} // test_verify_invariants

/**
 * test_inspect_simd() - Test SIMD vectorization inspection on framedistance.c.
 */
static void test_inspect_simd(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_inspect_simd\","
        "\"arguments\":{\"source_file\":\"src/gric-cluster/math/framedistance.c\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "is_vectorized") != NULL);
    assert(strstr(resp, "dominant_isa") != NULL);
    free(resp);
    printf("PASS: test_inspect_simd\n");
} // test_inspect_simd

/**
 * test_probe_dataset() - Test probe dataset tool on benchmarks/3Dspiral.txt.
 */
static void test_probe_dataset(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_probe_dataset\","
        "\"arguments\":{\"dataset_path\":\"benchmarks/3Dspiral.txt\",\"sample_limit\":500}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "recommended_rlim") != NULL);
    assert(strstr(resp, "recommended_cli_args") != NULL);
    free(resp);
    printf("PASS: test_probe_dataset\n");
} // test_probe_dataset

/**
 * test_notifications() - Test MCP initialized notification returns no response.
 */
static void test_notifications(void)
{
    const char *req = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}";
    char *resp = mcp_dispatch_message(req);
    assert(resp == NULL);
    printf("PASS: test_notifications\n");
} // test_notifications

int main(void)
{
    printf("=== Running gric-mcp Protocol Tests ===\n");
    test_initialize();
    test_ping();
    test_tools_list();
    test_align_parameters();
    test_audit_code_style();
    test_inspect_run();
    test_verify_invariants();
    test_inspect_simd();
    test_probe_dataset();
    test_notifications();
    printf("=== All gric-mcp Protocol Tests PASSED ===\n");
    return 0;
} // main

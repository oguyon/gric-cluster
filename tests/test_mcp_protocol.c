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
    assert(strstr(resp, "\"resources\"") != NULL);
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
    assert(strstr(resp, "gric_help") != NULL);
    assert(strstr(resp, "gric_list_suite") != NULL);
    assert(strstr(resp, "gric_get_recipe") != NULL);
    assert(strstr(resp, "gric_fps_status") != NULL);
    assert(strstr(resp, "gric_fps_run") != NULL);
    assert(strstr(resp, "gric_fps_set") != NULL);
    assert(strstr(resp, "gric_fps_stop") != NULL);
    assert(strstr(resp, "gric_probe_fps_streams") != NULL);
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

#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_test_run_dir[256] = "3Dspiral.clusterdat";
static char g_test_dataset[256] = "benchmarks/3Dspiral.txt";
static int  g_cleanup_fixture = 0;

static void setup_test_fixtures(void)
{
    if (access("3Dspiral.clusterdat/cluster_run.log", R_OK) != 0)
    {
        snprintf(g_test_run_dir, sizeof(g_test_run_dir),
                 "/tmp/test_mcp_fixture_%d.clusterdat", (int)getpid());
        mkdir(g_test_run_dir, 0755);

        char log_path[512];
        snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", g_test_run_dir);
        FILE *lfp = fopen(log_path, "w");
        assert(lfp != NULL);
        fprintf(lfp, "CMD: gric-cluster 0.10 test\n");
        fprintf(lfp, "PARAM_RLIM: 0.100000\n");
        fprintf(lfp, "STATS_CLUSTERS: 5\n");
        fprintf(lfp, "STATS_FRAMES: 100\n");
        fprintf(lfp, "STATS_DISTS: 200\n");
        fprintf(lfp, "STATS_PRUNED: 50\n");
        fprintf(lfp, "TIME_CLUSTERING_MS: 15.0\n");
        fclose(lfp);

        char mem_path[512];
        snprintf(mem_path, sizeof(mem_path), "%s/frame_membership.txt", g_test_run_dir);
        FILE *mfp = fopen(mem_path, "w");
        assert(mfp != NULL);
        for (int i = 0; i < 100; i++)
        {
            fprintf(mfp, "%d %d 0.050000\n", i, i % 5);
        }
        fclose(mfp);
        g_cleanup_fixture = 1;
    }

    if (access("benchmarks/3Dspiral.txt", R_OK) != 0)
    {
        snprintf(g_test_dataset, sizeof(g_test_dataset),
                 "/tmp/test_mcp_3Dspiral_%d.txt", (int)getpid());
        FILE *dfp = fopen(g_test_dataset, "w");
        assert(dfp != NULL);
        for (long ii = 0; ii < 500; ii++)
        {
            double t = (double)ii / 500.0;
            double theta = 4.0 * 3.14159265358979323846 * t;
            double x = 0.15 * t * cos(theta);
            double y = 0.15 * t * sin(theta);
            double z = 2.0 * t - 1.0;
            fprintf(dfp, "%.6f %.6f %.6f\n", x, y, z);
        }
        fclose(dfp);
    }
}

static void cleanup_test_fixtures(void)
{
    if (g_cleanup_fixture)
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/cluster_run.log", g_test_run_dir);
        unlink(path);
        snprintf(path, sizeof(path), "%s/frame_membership.txt", g_test_run_dir);
        unlink(path);
        rmdir(g_test_run_dir);
    }
    if (strncmp(g_test_dataset, "/tmp/", 5) == 0)
    {
        unlink(g_test_dataset);
    }
}

/**
 * test_inspect_run() - Test inspect_run tool on benchmark run directory.
 */
static void test_inspect_run(void)
{
    char req[512];
    snprintf(req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_inspect_run\","
        "\"arguments\":{\"run_dir\":\"%s\"}"
        "}}", g_test_run_dir);
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "num_clusters") != NULL);
    assert(strstr(resp, "num_frames") != NULL);
    free(resp);
    printf("PASS: test_inspect_run\n");
} // test_inspect_run

/**
 * test_verify_invariants() - Test invariant verification on benchmark run directory.
 */
static void test_verify_invariants(void)
{
    char req[512];
    snprintf(req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_verify_invariants\","
        "\"arguments\":{\"run_dir\":\"%s\"}"
        "}}", g_test_run_dir);
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
 * test_probe_dataset() - Test probe dataset tool.
 */
static void test_probe_dataset(void)
{
    char req[512];
    snprintf(req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_probe_dataset\","
        "\"arguments\":{\"dataset_path\":\"%s\",\"sample_limit\":500}"
        "}}", g_test_dataset);
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

/**
 * test_help_topic() - Verify gric_help tool with exact topic and query search.
 */
static void test_help_topic(void)
{
    /* 1. Test topic lookup for 'milk' */
    const char *req1 =
        "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_help\","
        "\"arguments\":{\"topic\":\"milk\"}"
        "}}";
    char *resp1 = mcp_dispatch_message(req1);
    assert(resp1 != NULL);
    assert(strstr(resp1, "FOUND") != NULL);
    assert(strstr(resp1, "Milk") != NULL);
    free(resp1);

    /* 2. Test topic lookup for 'milk_fpsexec' */
    const char *req2 =
        "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_help\","
        "\"arguments\":{\"topic\":\"milk_fpsexec\"}"
        "}}";
    char *resp2 = mcp_dispatch_message(req2);
    assert(resp2 != NULL);
    assert(strstr(resp2, "FOUND") != NULL);
    assert(strstr(resp2, "milk-fpsexec-gric-cluster") != NULL);
    free(resp2);

    /* 3. Test topic search for 'clustering' */
    const char *req3 =
        "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_help\","
        "\"arguments\":{\"query\":\"clustering\"}"
        "}}";
    char *resp3 = mcp_dispatch_message(req3);
    assert(resp3 != NULL);
    assert(strstr(resp3, "MATCHES_FOUND") != NULL);
    free(resp3);

    printf("PASS: test_help_topic\n");
} // test_help_topic

/**
 * test_list_suite() - Verify gric_list_suite tool.
 */
static void test_list_suite(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_list_suite\","
        "\"arguments\":{}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "SUCCESS") != NULL);
    assert(strstr(resp, "gric-cluster") != NULL);
    assert(strstr(resp, "gric-knn") != NULL);
    assert(strstr(resp, "milk-fpsexec-gric-cluster") != NULL);
    free(resp);
    printf("PASS: test_list_suite\n");
} // test_list_suite

/**
 * test_get_recipe() - Verify gric_get_recipe tool.
 */
static void test_get_recipe(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":14,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_get_recipe\","
        "\"arguments\":{\"recipe\":\"milk_realtime_streaming\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "Real-Time Milk Stream Clustering") != NULL);
    assert(strstr(resp, "gric_fps_run") != NULL);
    free(resp);
    printf("PASS: test_get_recipe\n");
} // test_get_recipe

/**
 * test_resources() - Verify MCP resources/list and resources/read protocol methods.
 */
static void test_resources(void)
{
    /* 1. Test resources/list */
    const char *req1 = "{\"jsonrpc\":\"2.0\",\"id\":15,\"method\":\"resources/list\"}";
    char *resp1 = mcp_dispatch_message(req1);
    assert(resp1 != NULL);
    assert(strstr(resp1, "gric://overview") != NULL);
    assert(strstr(resp1, "gric://cheatsheet/cli") != NULL);
    assert(strstr(resp1, "gric://milk/overview") != NULL);
    assert(strstr(resp1, "gric://milk/fps_params") != NULL);
    assert(strstr(resp1, "gric://help/milk") != NULL);
    free(resp1);

    /* 2. Test resources/read for gric://help/milk */
    const char *req2 =
        "{\"jsonrpc\":\"2.0\",\"id\":16,\"method\":\"resources/read\",\"params\":{"
        "\"uri\":\"gric://help/milk\""
        "}}";
    char *resp2 = mcp_dispatch_message(req2);
    assert(resp2 != NULL);
    assert(strstr(resp2, "Milk Framework Integration") != NULL);
    free(resp2);

    /* 3. Test resources/read for gric://milk/fps_params */
    const char *req3 =
        "{\"jsonrpc\":\"2.0\",\"id\":17,\"method\":\"resources/read\",\"params\":{"
        "\"uri\":\"gric://milk/fps_params\""
        "}}";
    char *resp3 = mcp_dispatch_message(req3);
    assert(resp3 != NULL);
    assert(strstr(resp3, ".in_name") != NULL);
    assert(strstr(resp3, ".rlim") != NULL);
    free(resp3);

    printf("PASS: test_resources\n");
} // test_resources

/**
 * test_fps_ops() - Verify gric_fps_status and gric_fps_stop tools.
 */
static void test_fps_ops(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":18,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_status\","
        "\"arguments\":{\"fps_name\":\"gric_cluster\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    assert(resp != NULL);
    assert(strstr(resp, "fps_name") != NULL);
    free(resp);

    const char *stop_req =
        "{\"jsonrpc\":\"2.0\",\"id\":19,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_stop\","
        "\"arguments\":{\"fps_name\":\"gric_cluster\"}"
        "}}";
    char *stop_resp = mcp_dispatch_message(stop_req);
    assert(stop_resp != NULL);
    assert(strstr(stop_resp, "STOPPED") != NULL);
    free(stop_resp);

    printf("PASS: test_fps_ops\n");
} // test_fps_ops

int main(void)
{
    printf("=== Running gric-mcp Protocol Tests ===\n");
    setup_test_fixtures();
    test_initialize();
    test_ping();
    test_tools_list();
    test_align_parameters();
    test_audit_code_style();
    test_inspect_run();
    test_verify_invariants();
    test_inspect_simd();
    test_probe_dataset();
    test_help_topic();
    test_list_suite();
    test_get_recipe();
    test_resources();
    test_fps_ops();
    test_notifications();
    cleanup_test_fixtures();
    printf("=== All gric-mcp Protocol Tests PASSED ===\n");
    return 0;
} // main

/**
 * @file test_mcp_protocol.c
 * @brief Integration tests for gric-mcp JSON-RPC 2.0 protocol and tools.
 */

#include "gric-mcp/mcp_dispatch.h"
#include "gric-mcp/mcp_registry.h"
#include "gric-mcp/mcp_exec.h"
#include "tests/mcp/mcp_test_util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_test_run_dir[256] = "3Dspiral.clusterdat";
static char g_test_dataset[256] = "benchmarks/3Dspiral.txt";
static int  g_cleanup_fixture = 0;

static void setup_test_fixtures(void)
{
    if (access("3Dspiral.clusterdat/cluster_run.log", R_OK) != 0)
    {
        snprintf(
            g_test_run_dir, sizeof(g_test_run_dir),
            "/tmp/test_mcp_fixture_%d.clusterdat", (int)getpid());
        mkdir(g_test_run_dir, 0755);

        char log_path[512];
        snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", g_test_run_dir);
        FILE *lfp = fopen(log_path, "w");
        CHECK(lfp != NULL);
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
        CHECK(mfp != NULL);
        for (int ii = 0; ii < 100; ii++)
        {
            fprintf(mfp, "%d %d 0.050000\n", ii, ii % 5);
        }
        fclose(mfp);
        g_cleanup_fixture = 1;
    }

    if (access("benchmarks/3Dspiral.txt", R_OK) != 0)
    {
        snprintf(
            g_test_dataset, sizeof(g_test_dataset),
            "/tmp/test_mcp_3Dspiral_%d.txt", (int)getpid());
        FILE *dfp = fopen(g_test_dataset, "w");
        CHECK(dfp != NULL);
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
} // setup_test_fixtures

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
} // cleanup_test_fixtures

/**
 * test_initialize() - Verify MCP initialize handshake response.
 */
static void test_initialize(void)
{
    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "\"name\":\"gric-mcp\"") != NULL);
    CHECK(strstr(resp, "\"protocolVersion\":\"2024-11-05\"") != NULL);
    CHECK(strstr(resp, "\"tools\"") != NULL);
    CHECK(strstr(resp, "\"resources\"") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "\"result\":{}") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "gric_audit_code_style") != NULL);
    CHECK(strstr(resp, "gric_align_parameters") != NULL);
    CHECK(strstr(resp, "gric_inspect_simd") != NULL);
    CHECK(strstr(resp, "gric_verify_invariants") != NULL);
    CHECK(strstr(resp, "gric_inspect_run") != NULL);
    CHECK(strstr(resp, "gric_probe_dataset") != NULL);
    CHECK(strstr(resp, "gric_probe_shm") != NULL);
    CHECK(strstr(resp, "gric_help") != NULL);
    CHECK(strstr(resp, "gric_list_suite") != NULL);
    CHECK(strstr(resp, "gric_get_recipe") != NULL);
    CHECK(strstr(resp, "gric_server_info") != NULL);
    CHECK(strstr(resp, "gric_fps_status") != NULL);
    CHECK(strstr(resp, "gric_fps_run") != NULL);
    CHECK(strstr(resp, "gric_fps_set") != NULL);
    CHECK(strstr(resp, "gric_fps_stop") != NULL);
    CHECK(strstr(resp, "gric_probe_fps_streams") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "FORMATTED") != NULL);
    CHECK(strstr(resp, "alignment_column") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "PASS") != NULL);
    CHECK(strstr(resp, "total_lines") != NULL);
    free(resp);
    printf("PASS: test_audit_code_style\n");
} // test_audit_code_style

/**
 * test_inspect_run() - Test inspect_run tool on benchmark run directory.
 */
static void test_inspect_run(void)
{
    char req[512];
    snprintf(
        req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_inspect_run\","
        "\"arguments\":{\"run_dir\":\"%s\"}"
        "}}", g_test_run_dir);
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "num_clusters") != NULL);
    CHECK(strstr(resp, "num_frames") != NULL);
    free(resp);
    printf("PASS: test_inspect_run\n");
} // test_inspect_run

/**
 * test_verify_invariants() - Test invariant verification on benchmark run directory.
 */
static void test_verify_invariants(void)
{
    char req[512];
    snprintf(
        req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_verify_invariants\","
        "\"arguments\":{\"run_dir\":\"%s\"}"
        "}}", g_test_run_dir);
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "PASS") != NULL);
    CHECK(strstr(resp, "violations_count") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "is_vectorized") != NULL);
    CHECK(strstr(resp, "dominant_isa") != NULL);
    free(resp);
    printf("PASS: test_inspect_simd\n");
} // test_inspect_simd

/**
 * test_probe_dataset() - Test probe dataset tool.
 */
static void test_probe_dataset(void)
{
    char req[512];
    snprintf(
        req, sizeof(req),
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_probe_dataset\","
        "\"arguments\":{\"dataset_path\":\"%s\",\"sample_limit\":500}"
        "}}", g_test_dataset);
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "recommended_rlim") != NULL);
    CHECK(strstr(resp, "recommended_cli_args") != NULL);
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
    CHECK(resp == NULL);
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
    CHECK(resp1 != NULL);
    CHECK(strstr(resp1, "FOUND") != NULL);
    CHECK(strstr(resp1, "Milk") != NULL);
    free(resp1);

    /* 2. Test topic lookup for 'milk_fpsexec' */
    const char *req2 =
        "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_help\","
        "\"arguments\":{\"topic\":\"milk_fpsexec\"}"
        "}}";
    char *resp2 = mcp_dispatch_message(req2);
    CHECK(resp2 != NULL);
    CHECK(strstr(resp2, "FOUND") != NULL);
    CHECK(strstr(resp2, "milk-fpsexec-gric-cluster") != NULL);
    free(resp2);

    /* 3. Test topic search for 'clustering' */
    const char *req3 =
        "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_help\","
        "\"arguments\":{\"query\":\"clustering\"}"
        "}}";
    char *resp3 = mcp_dispatch_message(req3);
    CHECK(resp3 != NULL);
    CHECK(strstr(resp3, "MATCHES_FOUND") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "SUCCESS") != NULL);
    CHECK(strstr(resp, "gric-cluster") != NULL);
    CHECK(strstr(resp, "gric-knn") != NULL);
    CHECK(strstr(resp, "milk-fpsexec-gric-cluster") != NULL);
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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "Real-Time Milk Stream Clustering") != NULL);
    CHECK(strstr(resp, "gric_fps_run") != NULL);
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
    CHECK(resp1 != NULL);
    CHECK(strstr(resp1, "gric://overview") != NULL);
    CHECK(strstr(resp1, "gric://cheatsheet/cli") != NULL);
    CHECK(strstr(resp1, "gric://milk/overview") != NULL);
    CHECK(strstr(resp1, "gric://milk/fps_params") != NULL);
    CHECK(strstr(resp1, "gric://help/milk") != NULL);
    free(resp1);

    /* 2. Test resources/read for gric://help/milk */
    const char *req2 =
        "{\"jsonrpc\":\"2.0\",\"id\":16,\"method\":\"resources/read\",\"params\":{"
        "\"uri\":\"gric://help/milk\""
        "}}";
    char *resp2 = mcp_dispatch_message(req2);
    CHECK(resp2 != NULL);
    CHECK(strstr(resp2, "Milk Framework Integration") != NULL);
    free(resp2);

    /* 3. Test resources/read for gric://milk/fps_params (module index) */
    const char *req3 =
        "{\"jsonrpc\":\"2.0\",\"id\":17,\"method\":\"resources/read\",\"params\":{"
        "\"uri\":\"gric://milk/fps_params\""
        "}}";
    char *resp3 = mcp_dispatch_message(req3);
    CHECK(resp3 != NULL);
    CHECK(strstr(resp3, "cluster") != NULL);
    CHECK(strstr(resp3, "gric://milk/fps_params/cluster") != NULL);
    free(resp3);

    /* 4. Test resources/read for gric://milk/fps_params/cluster */
    const char *req4 =
        "{\"jsonrpc\":\"2.0\",\"id\":18,\"method\":\"resources/read\",\"params\":{"
        "\"uri\":\"gric://milk/fps_params/cluster\""
        "}}";
    char *resp4 = mcp_dispatch_message(req4);
    CHECK(resp4 != NULL);
    CHECK(strstr(resp4, ".in_name") != NULL);
    CHECK(strstr(resp4, ".rlim") != NULL);
    free(resp4);

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
    CHECK(resp != NULL);
    CHECK(strstr(resp, "fps_name") != NULL);
    free(resp);

    const char *stop_req =
        "{\"jsonrpc\":\"2.0\",\"id\":19,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_stop\","
        "\"arguments\":{\"fps_name\":\"gric_cluster\"}"
        "}}";
    char *stop_resp = mcp_dispatch_message(stop_req);
    CHECK(stop_resp != NULL);
    CHECK(strstr(stop_resp, "STOPPED") != NULL);
    free(stop_resp);

    printf("PASS: test_fps_ops\n");
} // test_fps_ops

/**
 * test_server_info() - Verify gric_server_info tool returns runtime metadata.
 */
static void test_server_info(void)
{
    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":20,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_server_info\","
        "\"arguments\":{}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "gric-mcp") != NULL);
    CHECK(strstr(resp, "toolsets") != NULL);
    CHECK(strstr(resp, "read_only") != NULL);
    free(resp);
    printf("PASS: test_server_info\n");
} // test_server_info

/**
 * test_tools_call_hidden() - Verify hidden/disabled toolsets return error -32601.
 */
static void test_tools_call_hidden(void)
{
    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_KNOWLEDGE,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    const char *req =
        "{\"jsonrpc\":\"2.0\",\"id\":21,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_inspect_run\","
        "\"arguments\":{\"run_dir\":\"nonexistent\"}"
        "}}";
    char *resp = mcp_dispatch_message(req);
    CHECK(resp != NULL);
    CHECK(strstr(resp, "-32601") != NULL);
    free(resp);
    printf("PASS: test_tools_call_hidden\n");
} // test_tools_call_hidden

/**
 * test_readonly_mode() - Verify --read-only hides and rejects side-effecting tools.
 */
static void test_readonly_mode(void)
{
    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 1,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    /* tools/list must not include side-effecting tools */
    const char *list_req = "{\"jsonrpc\":\"2.0\",\"id\":22,\"method\":\"tools/list\"}";
    char *list_resp = mcp_dispatch_message(list_req);
    CHECK(list_resp != NULL);
    CHECK(strstr(list_resp, "gric_fps_run") == NULL);
    CHECK(strstr(list_resp, "gric_fps_set") == NULL);
    CHECK(strstr(list_resp, "gric_fps_stop") == NULL);
    CHECK(strstr(list_resp, "gric_fps_status") != NULL);
    free(list_resp);

    /* Direct call to gric_fps_set must be rejected with -32601 */
    const char *call_req =
        "{\"jsonrpc\":\"2.0\",\"id\":23,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_set\","
        "\"arguments\":{\"fps_name\":\"gric_cluster\",\"param\":\"rlim\",\"value\":\"0.5\"}"
        "}}";
    char *call_resp = mcp_dispatch_message(call_req);
    CHECK(call_resp != NULL);
    CHECK(strstr(call_resp, "-32601") != NULL);
    free(call_resp);

    printf("PASS: test_readonly_mode\n");
} // test_readonly_mode

/**
 * test_fps_validation() - Verify invalid identifiers and injection attempts fail.
 */
static void test_fps_validation(void)
{
    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    /* Invalid fps_name with space */
    const char *req1 =
        "{\"jsonrpc\":\"2.0\",\"id\":24,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_status\","
        "\"arguments\":{\"fps_name\":\"a b\"}"
        "}}";
    char *resp1 = mcp_dispatch_message(req1);
    CHECK(resp1 != NULL);
    CHECK(strstr(resp1, "Invalid fps_name identifier") != NULL);
    free(resp1);

    /* Invalid fps_name with directory traversal */
    const char *req2 =
        "{\"jsonrpc\":\"2.0\",\"id\":25,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_status\","
        "\"arguments\":{\"fps_name\":\"../x\"}"
        "}}";
    char *resp2 = mcp_dispatch_message(req2);
    CHECK(resp2 != NULL);
    CHECK(strstr(resp2, "Invalid fps_name identifier") != NULL);
    free(resp2);

    /* Invalid value attempting command injection */
    const char *req3 =
        "{\"jsonrpc\":\"2.0\",\"id\":26,\"method\":\"tools/call\",\"params\":{"
        "\"name\":\"gric_fps_set\","
        "\"arguments\":{\"fps_name\":\"gric_cluster\",\"param\":\"rlim\","
        "\"value\":\"0.4; touch /tmp/pwned\"}"
        "}}";
    char *resp3 = mcp_dispatch_message(req3);
    CHECK(resp3 != NULL);
    CHECK(strstr(resp3, "Invalid value string format") != NULL);
    free(resp3);
    CHECK(access("/tmp/pwned", F_OK) != 0);

    printf("PASS: test_fps_validation\n");
} // test_fps_validation

/**
 * test_exec_timeout() - Verify mcp_exec_capture terminates timed out commands.
 */
static void test_exec_timeout(void)
{
    const char *const sleep_argv[] = { "sh", "-c", "sleep 5", NULL };
    char out_buf[128] = "";
    int exit_status = 0;
    int ret = mcp_exec_capture(sleep_argv, out_buf, sizeof(out_buf), 100, &exit_status);
    CHECK(ret == -2);
    printf("PASS: test_exec_timeout\n");
} // test_exec_timeout

int main(void)
{
    printf("=== Running gric-mcp Protocol Tests ===\n");
    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

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
    test_server_info();
    test_tools_call_hidden();
    test_readonly_mode();
    test_fps_validation();
    test_exec_timeout();
    cleanup_test_fixtures();
    printf("=== All gric-mcp Protocol Tests PASSED ===\n");
    return 0;
} // main

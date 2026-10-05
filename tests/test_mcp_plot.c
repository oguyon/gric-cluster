/**
 * @file test_mcp_plot.c
 * @brief Unit tests for gric_plot diagnostic visualization tool.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TEST_FIXTURE_DIR "/tmp/test_mcp_plot_fixture"

/**
 * setup_fixture() - Create synthetic clustering run and dataset files for testing.
 */
static void setup_fixture(void)
{
    mkdir(TEST_FIXTURE_DIR, 0755);

    FILE *fp_pts = fopen(TEST_FIXTURE_DIR "/points.txt", "w");
    assert(fp_pts != NULL);
    fprintf(fp_pts, "0.100 0.200\n");
    fprintf(fp_pts, "0.150 0.250\n");
    fprintf(fp_pts, "0.800 0.900\n");
    fprintf(fp_pts, "0.850 0.950\n");
    fclose(fp_pts);

    FILE *fp_memb = fopen(TEST_FIXTURE_DIR "/frame_membership.txt", "w");
    assert(fp_memb != NULL);
    fprintf(fp_memb, "0 0\n");
    fprintf(fp_memb, "1 0\n");
    fprintf(fp_memb, "2 1\n");
    fprintf(fp_memb, "3 1\n");
    fclose(fp_memb);

    FILE *fp_log = fopen(TEST_FIXTURE_DIR "/cluster_run.log", "w");
    assert(fp_log != NULL);
    fprintf(fp_log, "CMD: gric-cluster 0.200 points.txt -maxim 4\n");
    fprintf(fp_log, "OUTPUT_DIR: %s\n", TEST_FIXTURE_DIR);
    fprintf(fp_log, "PARAM_RLIM: 0.200000\n");
    fprintf(fp_log, "PARAM_DPROB: 0.010000\n");
    fprintf(fp_log, "STATS_FRAMES: 4\n");
    fprintf(fp_log, "STATS_CLUSTERS: 2\n");
    fprintf(fp_log, "STATS_DISTS: 6\n");
    fprintf(fp_log, "STATS_PRUNED: 0\n");
    fprintf(fp_log, "STATS_DIST_HIST_START\n");
    fprintf(fp_log, "0 2 0\n");
    fprintf(fp_log, "1 2 0\n");
    fprintf(fp_log, "STATS_DIST_HIST_END\n");
    fclose(fp_log);
} // setup_fixture

/**
 * cleanup_fixture() - Remove temporary test fixtures and generated images.
 */
static void cleanup_fixture(void)
{
    unlink(TEST_FIXTURE_DIR "/points.txt");
    unlink(TEST_FIXTURE_DIR "/frame_membership.txt");
    unlink(TEST_FIXTURE_DIR "/cluster_run.log");
    unlink(TEST_FIXTURE_DIR "/plot.png");
    unlink(TEST_FIXTURE_DIR "/plot.queries.png");
    unlink(TEST_FIXTURE_DIR "/plot.svg");
    unlink(TEST_FIXTURE_DIR "/plot.queries.svg");
    unlink(TEST_FIXTURE_DIR "/cluster_plot.png");
    unlink(TEST_FIXTURE_DIR "/cluster_plot.queries.png");
    rmdir(TEST_FIXTURE_DIR);
} // cleanup_fixture

/**
 * test_registry_lookup() - Verify tool is registered and inspectable in the registry.
 */
static void test_registry_lookup(void)
{
    printf("[test_mcp_plot] Running test_registry_lookup...\n");

    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    const struct mcp_tool_def *tool = mcp_registry_find("gric_plot");
    assert(tool != NULL);
    assert(strcmp(tool->name, "gric_plot") == 0);
    assert(tool->toolset == MCP_TS_ANALYSIS);
    assert(tool->side_effects == 0);
    assert(tool->fn != NULL);
    assert(tool->input_schema != NULL);
} // test_registry_lookup

/**
 * test_error_cases() - Verify proper error handling for invalid or missing inputs.
 */
static void test_error_cases(void)
{
    printf("[test_mcp_plot] Running test_error_cases...\n");

    /* 1. Missing / NULL arguments */
    {
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_plot(NULL, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));
        cJSON_Delete(res);
    }

    /* 2. Nonexistent log path */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "points_path", TEST_FIXTURE_DIR "/points.txt");
        cJSON_AddStringToObject(args, "log_path", "/tmp/nonexistent_path/nonexistent.log");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_plot(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 3. Nonexistent points path */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "points_path", "/tmp/nonexistent_points.txt");
        cJSON_AddStringToObject(args, "log_path", TEST_FIXTURE_DIR "/cluster_run.log");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_plot(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 4. Invalid format */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "points_path", TEST_FIXTURE_DIR "/points.txt");
        cJSON_AddStringToObject(args, "log_path", TEST_FIXTURE_DIR "/cluster_run.log");
        cJSON_AddStringToObject(args, "format", "bmp");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_plot(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }
} // test_error_cases

/**
 * test_plot_png() - Verify PNG plot generation with explicit output path.
 */
static void test_plot_png(void)
{
    printf("[test_mcp_plot] Running test_plot_png...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "points_path", TEST_FIXTURE_DIR "/points.txt");
    cJSON_AddStringToObject(args, "log_path", TEST_FIXTURE_DIR "/cluster_run.log");
    cJSON_AddStringToObject(args, "output_path", TEST_FIXTURE_DIR "/plot.png");
    cJSON_AddStringToObject(args, "format", "png");
    cJSON_AddNumberToObject(args, "font_size", 16.0);

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_plot(args, res);
    assert(rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && strcmp(status->valuestring, "success") == 0);

    cJSON *out_path = cJSON_GetObjectItemCaseSensitive(res, "output_path");
    assert(out_path != NULL && strcmp(out_path->valuestring, TEST_FIXTURE_DIR "/plot.png") == 0);

    cJSON *fsize = cJSON_GetObjectItemCaseSensitive(res, "file_size_bytes");
    assert(fsize != NULL && fsize->valuedouble > 0.0);

    cJSON *fmt = cJSON_GetObjectItemCaseSensitive(res, "format");
    assert(fmt != NULL && strcmp(fmt->valuestring, "png") == 0);

    cJSON *stats = cJSON_GetObjectItemCaseSensitive(res, "stats");
    assert(stats != NULL);
    assert(cJSON_GetObjectItemCaseSensitive(stats, "total_frames")->valuedouble == 4.0);
    assert(cJSON_GetObjectItemCaseSensitive(stats, "total_clusters")->valuedouble == 2.0);

    struct stat st;
    assert(stat(TEST_FIXTURE_DIR "/plot.png", &st) == 0 && st.st_size > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_plot_png

/**
 * test_plot_svg() - Verify SVG plot generation.
 */
static void test_plot_svg(void)
{
    printf("[test_mcp_plot] Running test_plot_svg...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "points_path", TEST_FIXTURE_DIR "/points.txt");
    cJSON_AddStringToObject(args, "log_path", TEST_FIXTURE_DIR "/cluster_run.log");
    cJSON_AddStringToObject(args, "output_path", TEST_FIXTURE_DIR "/plot.svg");
    cJSON_AddStringToObject(args, "format", "svg");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_plot(args, res);
    assert(rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && strcmp(status->valuestring, "success") == 0);

    cJSON *fmt = cJSON_GetObjectItemCaseSensitive(res, "format");
    assert(fmt != NULL && strcmp(fmt->valuestring, "svg") == 0);

    struct stat st;
    assert(stat(TEST_FIXTURE_DIR "/plot.svg", &st) == 0 && st.st_size > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_plot_svg

/**
 * test_plot_run_dir() - Verify plot generation using run_dir parameter and auto-naming.
 */
static void test_plot_run_dir(void)
{
    printf("[test_mcp_plot] Running test_plot_run_dir...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "run_dir", TEST_FIXTURE_DIR);
    cJSON_AddStringToObject(args, "format", "png");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_plot(args, res);
    assert(rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && strcmp(status->valuestring, "success") == 0);

    cJSON *out_path = cJSON_GetObjectItemCaseSensitive(res, "output_path");
    assert(out_path != NULL && strstr(out_path->valuestring, "cluster_plot.png") != NULL);

    struct stat st;
    assert(stat(out_path->valuestring, &st) == 0 && st.st_size > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_plot_run_dir

int main(void)
{
    printf("[test_mcp_plot] Setting up fixture...\n");
    setup_fixture();

    test_registry_lookup();
    test_error_cases();
    test_plot_png();
    test_plot_svg();
    test_plot_run_dir();

    printf("[test_mcp_plot] Cleaning up fixture...\n");
    cleanup_fixture();

    printf("[test_mcp_plot] All tests passed!\n");
    return 0;
} // main

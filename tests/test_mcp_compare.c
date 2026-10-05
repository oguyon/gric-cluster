/**
 * @file test_mcp_compare.c
 * @brief Unit tests for clustering run comparison and cluster detail profiling tools.
 */

#include "mcp_tools.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void create_test_dir(const char *dir)
{
    mkdir(dir, 0777);
}

static void write_file(
    const char *path,
    const char *content)
{
    FILE *f = fopen(path, "w");
    assert(f != NULL);
    fputs(content, f);
    fclose(f);
}

static void setup_test_fixtures(void)
{
    create_test_dir("/tmp/test_mcp_run_a");
    create_test_dir("/tmp/test_mcp_run_b");
    create_test_dir("/tmp/test_mcp_run_c");

    /* Run A */
    write_file(
        "/tmp/test_mcp_run_a/job.log",
        "CMD: gric-cluster 0.1 data.txt -maxcl 10\n"
        "START_TIME: 2026-10-04T12:00:00\n"
        "TIME_CLUSTERING_MS: 100.0\n"
        "PARAM_RLIM: 0.1000\n"
        "PARAM_MAXCL: 10\n"
        "STATS_CLUSTERS: 3\n"
        "STATS_FRAMES: 6\n"
        "STATS_DISTS: 20\n"
        "STATS_PRUNED: 80\n"
        "STATS_MAX_RSS_KB: 4096\n");

    write_file(
        "/tmp/test_mcp_run_a/membership.txt",
        "0 0\n"
        "1 0\n"
        "2 1\n"
        "3 1\n"
        "4 2\n"
        "5 2\n");

    write_file(
        "/tmp/test_mcp_run_a/clusters.txt",
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
        "7.0 8.0 9.0\n");

    /* Run B (Identical to Run A but faster) */
    write_file(
        "/tmp/test_mcp_run_b/job.log",
        "CMD: gric-cluster 0.1 data.txt -maxcl 10 -fast\n"
        "START_TIME: 2026-10-04T12:05:00\n"
        "TIME_CLUSTERING_MS: 50.0\n"
        "PARAM_RLIM: 0.1000\n"
        "PARAM_MAXCL: 10\n"
        "STATS_CLUSTERS: 3\n"
        "STATS_FRAMES: 6\n"
        "STATS_DISTS: 10\n"
        "STATS_PRUNED: 90\n"
        "STATS_MAX_RSS_KB: 4096\n");

    write_file(
        "/tmp/test_mcp_run_b/membership.txt",
        "0 0\n"
        "1 0\n"
        "2 1\n"
        "3 1\n"
        "4 2\n"
        "5 2\n");

    /* Run C (Different cluster assignments) */
    write_file(
        "/tmp/test_mcp_run_c/job.log",
        "CMD: gric-cluster 0.2 data.txt -maxcl 10\n"
        "TIME_CLUSTERING_MS: 80.0\n"
        "STATS_CLUSTERS: 2\n"
        "STATS_FRAMES: 6\n");

    write_file(
        "/tmp/test_mcp_run_c/membership.txt",
        "0 0\n"
        "1 0\n"
        "2 0\n"
        "3 1\n"
        "4 1\n"
        "5 1\n");
}

static void cleanup_test_fixtures(void)
{
    unlink("/tmp/test_mcp_run_a/job.log");
    unlink("/tmp/test_mcp_run_a/membership.txt");
    unlink("/tmp/test_mcp_run_a/clusters.txt");
    rmdir("/tmp/test_mcp_run_a");

    unlink("/tmp/test_mcp_run_b/job.log");
    unlink("/tmp/test_mcp_run_b/membership.txt");
    rmdir("/tmp/test_mcp_run_b");

    unlink("/tmp/test_mcp_run_c/job.log");
    unlink("/tmp/test_mcp_run_c/membership.txt");
    rmdir("/tmp/test_mcp_run_c");
}

static void test_compare_runs_errors(void)
{
    /* Test 1: NULL args */
    cJSON *res1 = cJSON_CreateObject();
    int rc1 = mcp_tool_compare_runs(NULL, res1);
    assert(rc1 == -1);
    cJSON_Delete(res1);

    /* Test 2: Missing arguments */
    cJSON *args2 = cJSON_CreateObject();
    cJSON *res2 = cJSON_CreateObject();
    int rc2 = mcp_tool_compare_runs(args2, res2);
    assert(rc2 == -1);
    cJSON_Delete(args2);
    cJSON_Delete(res2);
}

static void test_compare_runs_identical(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "run_dir_a", "/tmp/test_mcp_run_a");
    cJSON_AddStringToObject(args, "run_dir_b", "/tmp/test_mcp_run_b");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_compare_runs(args, res);
    assert(rc == 0);

    cJSON *comp = cJSON_GetObjectItem(res, "comparison");
    assert(comp != NULL);

    double ari = cJSON_GetObjectItem(comp, "adjusted_rand_index")->valuedouble;
    double agree = cJSON_GetObjectItem(comp, "agreement_percent")->valuedouble;
    double speedup = cJSON_GetObjectItem(comp, "speedup_ratio")->valuedouble;

    assert(fabs(ari - 1.0) < 1e-4);
    assert(fabs(agree - 100.0) < 1e-4);
    assert(fabs(speedup - 2.0) < 1e-2);

    cJSON *rep = cJSON_GetObjectItem(res, "report");
    assert(rep != NULL && strstr(rep->valuestring, "Identical") != NULL);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

static void test_compare_runs_different(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "run_dir_a", "/tmp/test_mcp_run_a");
    cJSON_AddStringToObject(args, "run_dir_b", "/tmp/test_mcp_run_c");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_compare_runs(args, res);
    assert(rc == 0);

    cJSON *comp = cJSON_GetObjectItem(res, "comparison");
    assert(comp != NULL);

    double ari = cJSON_GetObjectItem(comp, "adjusted_rand_index")->valuedouble;
    double agree = cJSON_GetObjectItem(comp, "agreement_percent")->valuedouble;
    int delta = cJSON_GetObjectItem(comp, "delta_clusters")->valueint;

    assert(ari < 1.0);
    assert(agree < 100.0);
    assert(delta == -1);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

static void test_cluster_detail(void)
{
    /* Test 1: Invalid cluster ID */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "run_dir", "/tmp/test_mcp_run_a");
        cJSON_AddNumberToObject(args, "cluster_id", -1);
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_cluster_detail(args, res);
        assert(rc == -1);
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 2: Non-existent cluster ID */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "run_dir", "/tmp/test_mcp_run_a");
        cJSON_AddNumberToObject(args, "cluster_id", 99);
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_cluster_detail(args, res);
        assert(rc == -1);
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 3: Valid cluster ID 0 */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "run_dir", "/tmp/test_mcp_run_a");
        cJSON_AddNumberToObject(args, "cluster_id", 0);
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_cluster_detail(args, res);
        assert(rc == 0);

        assert(cJSON_GetObjectItem(res, "member_frames")->valueint == 2);
        assert(cJSON_GetObjectItem(res, "total_run_frames")->valueint == 6);

        cJSON *dyn = cJSON_GetObjectItem(res, "temporal_dynamics");
        assert(dyn != NULL);
        assert(cJSON_GetObjectItem(dyn, "birth_frame")->valueint == 0);
        assert(cJSON_GetObjectItem(dyn, "death_frame")->valueint == 1);
        assert(cJSON_GetObjectItem(dyn, "active_span_frames")->valueint == 2);
        assert(cJSON_GetObjectItem(dyn, "primary_next_cluster")->valueint == 1);

        cJSON *geom = cJSON_GetObjectItem(res, "centroid_geometry");
        assert(geom != NULL);
        assert(cJSON_GetObjectItem(geom, "dimension")->valueint == 3);

        cJSON *summary = cJSON_GetObjectItem(res, "summary");
        assert(summary != NULL && strlen(summary->valuestring) > 0);

        cJSON_Delete(args);
        cJSON_Delete(res);
    }
}

int main(void)
{
    printf("[test_mcp_compare] Setting up fixtures...\n");
    setup_test_fixtures();

    printf("[test_mcp_compare] Running compare_runs error tests...\n");
    test_compare_runs_errors();

    printf("[test_mcp_compare] Running compare_runs identical tests...\n");
    test_compare_runs_identical();

    printf("[test_mcp_compare] Running compare_runs different tests...\n");
    test_compare_runs_different();

    printf("[test_mcp_compare] Running cluster_detail tests...\n");
    test_cluster_detail();

    cleanup_test_fixtures();
    printf("[test_mcp_compare] All run analysis tests passed!\n");
    return 0;
}

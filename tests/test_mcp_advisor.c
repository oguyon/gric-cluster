/**
 * @file test_mcp_advisor.c
 * @brief Unit and integration tests for parameter advisor and radius calibration tools.
 */

#include "mcp_tools.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_recommend_params_errors(void)
{
    /* Test 1: NULL arguments */
    cJSON *res1 = cJSON_CreateObject();
    int rc1 = mcp_tool_recommend_params(NULL, res1);
    assert(rc1 == -1);
    assert(cJSON_GetObjectItem(res1, "error") != NULL);
    cJSON_Delete(res1);

    /* Test 2: Missing dataset_path */
    cJSON *args2 = cJSON_CreateObject();
    cJSON *res2 = cJSON_CreateObject();
    int rc2 = mcp_tool_recommend_params(args2, res2);
    assert(rc2 == -1);
    assert(cJSON_GetObjectItem(res2, "error") != NULL);
    cJSON_Delete(args2);
    cJSON_Delete(res2);

    /* Test 3: Non-existent dataset_path */
    cJSON *args3 = cJSON_CreateObject();
    cJSON_AddStringToObject(args3, "dataset_path", "/tmp/nonexistent_dataset_12345.bin");
    cJSON *res3 = cJSON_CreateObject();
    int rc3 = mcp_tool_recommend_params(args3, res3);
    assert(rc3 == -1);
    assert(cJSON_GetObjectItem(res3, "error") != NULL);
    cJSON_Delete(args3);
    cJSON_Delete(res3);
}

static void test_recommend_params_valid(void)
{
    const char *ds_path = "tests/test_strat.txt";

    /* Test with balanced speed preference */
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset_path", ds_path);
    cJSON_AddNumberToObject(args, "target_clusters", 30);
    cJSON_AddNumberToObject(args, "memory_budget_gb", 4.0);
    cJSON_AddStringToObject(args, "speed_preference", "balanced");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_recommend_params(args, res);
    assert(rc == 0);

    cJSON *ds_info = cJSON_GetObjectItem(res, "dataset_info");
    assert(ds_info != NULL);
    assert(cJSON_GetObjectItem(ds_info, "num_frames") != NULL);
    assert(cJSON_GetObjectItem(ds_info, "dim") != NULL);

    cJSON *recs = cJSON_GetObjectItem(res, "recommendations");
    assert(recs != NULL);
    cJSON *rlim_item = cJSON_GetObjectItem(recs, "rlim");
    assert(rlim_item != NULL && rlim_item->valuedouble > 0.0);
    cJSON *maxcl_item = cJSON_GetObjectItem(recs, "maxcl");
    assert(maxcl_item != NULL && maxcl_item->valueint >= 30);
    cJSON *cli_cmd = cJSON_GetObjectItem(recs, "cli_command");
    assert(cli_cmd != NULL && strstr(cli_cmd->valuestring, "gric-cluster") != NULL);

    cJSON *res_est = cJSON_GetObjectItem(res, "resource_estimates");
    assert(res_est != NULL);
    assert(cJSON_GetObjectItem(res_est, "estimated_rss_mb") != NULL);

    cJSON *reasoning = cJSON_GetObjectItem(res, "reasoning");
    assert(reasoning != NULL && strlen(reasoning->valuestring) > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);

    /* Test with fast speed preference */
    cJSON *args_fast = cJSON_CreateObject();
    cJSON_AddStringToObject(args_fast, "dataset_path", ds_path);
    cJSON_AddStringToObject(args_fast, "speed_preference", "fast");

    cJSON *res_fast = cJSON_CreateObject();
    rc = mcp_tool_recommend_params(args_fast, res_fast);
    assert(rc == 0);
    cJSON_Delete(args_fast);
    cJSON_Delete(res_fast);

    /* Test with accurate speed preference */
    cJSON *args_acc = cJSON_CreateObject();
    cJSON_AddStringToObject(args_acc, "dataset_path", ds_path);
    cJSON_AddStringToObject(args_acc, "speed_preference", "accurate");

    cJSON *res_acc = cJSON_CreateObject();
    rc = mcp_tool_recommend_params(args_acc, res_acc);
    assert(rc == 0);
    cJSON_Delete(args_acc);
    cJSON_Delete(res_acc);
}

static void test_calibrate_radius_errors(void)
{
    /* Test 1: NULL arguments */
    cJSON *res1 = cJSON_CreateObject();
    int rc1 = mcp_tool_calibrate_radius(NULL, res1);
    assert(rc1 == -1);
    assert(cJSON_GetObjectItem(res1, "error") != NULL);
    cJSON_Delete(res1);

    /* Test 2: Missing dataset_path */
    cJSON *args2 = cJSON_CreateObject();
    cJSON *res2 = cJSON_CreateObject();
    int rc2 = mcp_tool_calibrate_radius(args2, res2);
    assert(rc2 == -1);
    assert(cJSON_GetObjectItem(res2, "error") != NULL);
    cJSON_Delete(args2);
    cJSON_Delete(res2);

    /* Test 3: Non-existent dataset_path */
    cJSON *args3 = cJSON_CreateObject();
    cJSON_AddStringToObject(args3, "dataset_path", "/tmp/nonexistent_dataset_12345.bin");
    cJSON *res3 = cJSON_CreateObject();
    int rc3 = mcp_tool_calibrate_radius(args3, res3);
    assert(rc3 == -1);
    assert(cJSON_GetObjectItem(res3, "error") != NULL);
    cJSON_Delete(args3);
    cJSON_Delete(res3);
}

static void test_calibrate_radius_valid(void)
{
    const char *ds_path = "tests/test_strat.txt";

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset_path", ds_path);
    cJSON_AddNumberToObject(args, "sample_size", 100);

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_calibrate_radius(args, res);
    assert(rc == 0);

    cJSON *ds_info = cJSON_GetObjectItem(res, "dataset_info");
    assert(ds_info != NULL);
    assert(cJSON_GetObjectItem(ds_info, "num_frames") != NULL);
    assert(cJSON_GetObjectItem(ds_info, "dim") != NULL);

    cJSON *pcts = cJSON_GetObjectItem(res, "percentiles");
    assert(pcts != NULL);

    double min_v = cJSON_GetObjectItem(pcts, "min")->valuedouble;
    double p01_v = cJSON_GetObjectItem(pcts, "p01")->valuedouble;
    double p05_v = cJSON_GetObjectItem(pcts, "p05")->valuedouble;
    double p10_v = cJSON_GetObjectItem(pcts, "p10")->valuedouble;
    double p15_v = cJSON_GetObjectItem(pcts, "p15")->valuedouble;
    double p25_v = cJSON_GetObjectItem(pcts, "p25")->valuedouble;
    double p35_v = cJSON_GetObjectItem(pcts, "p35")->valuedouble;
    double p50_v = cJSON_GetObjectItem(pcts, "p50")->valuedouble;
    double p75_v = cJSON_GetObjectItem(pcts, "p75")->valuedouble;
    double p90_v = cJSON_GetObjectItem(pcts, "p90")->valuedouble;
    double max_v = cJSON_GetObjectItem(pcts, "max")->valuedouble;

    /* Verify non-decreasing percentiles */
    assert(min_v <= p01_v);
    assert(p01_v <= p05_v);
    assert(p05_v <= p10_v);
    assert(p10_v <= p15_v);
    assert(p15_v <= p25_v);
    assert(p25_v <= p35_v);
    assert(p35_v <= p50_v);
    assert(p50_v <= p75_v);
    assert(p75_v <= p90_v);
    assert(p90_v <= max_v);

    cJSON *recs = cJSON_GetObjectItem(res, "recommendations");
    assert(recs != NULL);
    double fine_v = cJSON_GetObjectItem(recs, "fine")->valuedouble;
    double bal_v = cJSON_GetObjectItem(recs, "balanced")->valuedouble;
    double coarse_v = cJSON_GetObjectItem(recs, "coarse")->valuedouble;

    assert(fine_v <= bal_v);
    assert(bal_v <= coarse_v);

    cJSON *analysis = cJSON_GetObjectItem(res, "analysis");
    assert(analysis != NULL && strlen(analysis->valuestring) > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

int main(void)
{
    printf("[test_mcp_advisor] Starting parameter advisor tool tests...\n");

    test_recommend_params_errors();
    test_recommend_params_valid();
    test_calibrate_radius_errors();
    test_calibrate_radius_valid();

    printf("[test_mcp_advisor] All parameter advisor tests passed!\n");
    return 0;
}

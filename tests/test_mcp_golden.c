/**
 * @file test_mcp_golden.c
 * @brief Unit and integration tests for developer golden baseline comparison tool.
 */

#include "mcp_tools.h"
#include "tests/mcp/mcp_test_util.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_golden_preset_spiral(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "preset", "spiral");

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_golden_compare(args, res);

    CHECK(ret == 0);

    cJSON *status = cJSON_GetObjectItem(res, "status");
    CHECK(status != NULL);
    CHECK_MSG(strcmp(status->valuestring, "PASS") == 0, status->valuestring);

    cJSON *metrics = cJSON_GetObjectItem(res, "metrics");
    CHECK(metrics != NULL);

    cJSON *clusters = cJSON_GetObjectItem(metrics, "cluster_count");
    CHECK(clusters != NULL);
    cJSON *curr_cl = cJSON_GetObjectItem(clusters, "current");
    CHECK(curr_cl != NULL && curr_cl->valueint == 62);
    cJSON *delta_cl = cJSON_GetObjectItem(clusters, "delta");
    CHECK(delta_cl != NULL && delta_cl->valueint == 0);

    cJSON *dists = cJSON_GetObjectItem(metrics, "dist_evals");
    CHECK(dists != NULL);
    cJSON *curr_d = cJSON_GetObjectItem(dists, "current");
    CHECK(curr_d != NULL && curr_d->valuedouble >= 2828.0 && curr_d->valuedouble <= 2830.0);
    cJSON *ratio_d = cJSON_GetObjectItem(dists, "ratio");
    CHECK(ratio_d != NULL && ratio_d->valuedouble >= 0.99 && ratio_d->valuedouble <= 1.01);

    cJSON *regs = cJSON_GetObjectItem(res, "regressions");
    CHECK(regs != NULL && cJSON_GetArraySize(regs) == 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_golden_preset_spiral

static void test_golden_custom_baseline_agreement(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "preset", "spiral");
    cJSON_AddStringToObject(args, "baseline_dir", "/tmp/ctest_spiral_out");

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_golden_compare(args, res);

    CHECK(ret == 0);

    cJSON *status = cJSON_GetObjectItem(res, "status");
    CHECK(status != NULL);
    CHECK_MSG(strcmp(status->valuestring, "PASS") == 0, status->valuestring);

    cJSON *metrics = cJSON_GetObjectItem(res, "metrics");
    CHECK(metrics != NULL);

    cJSON *agreement = cJSON_GetObjectItem(metrics, "assignment_agreement_pct");
    CHECK(agreement != NULL && agreement->valuedouble >= 99.9);

    cJSON *drift = cJSON_GetObjectItem(metrics, "centroid_max_drift");
    CHECK(drift != NULL && drift->valuedouble < 1e-4);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_golden_custom_baseline_agreement

static void test_golden_regression_detection(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "preset", "spiral");
    /* Force regression by setting an impossible agreement threshold */
    cJSON_AddNumberToObject(args, "min_agreement_pct", 100.1);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_golden_compare(args, res);

    CHECK(ret == 0);

    cJSON *status = cJSON_GetObjectItem(res, "status");
    CHECK(status != NULL);
    CHECK(strcmp(status->valuestring, "REGRESSION") == 0);

    cJSON *regs = cJSON_GetObjectItem(res, "regressions");
    CHECK(regs != NULL && cJSON_GetArraySize(regs) > 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_golden_regression_detection

int main(void)
{
    printf("Running test_golden_preset_spiral...\n");
    test_golden_preset_spiral();

    printf("Running test_golden_custom_baseline_agreement...\n");
    test_golden_custom_baseline_agreement();

    printf("Running test_golden_regression_detection...\n");
    test_golden_regression_detection();

    printf("All test_mcp_golden tests PASSED!\n");
    return 0;
}

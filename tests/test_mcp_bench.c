/**
 * @file test_mcp_bench.c
 * @brief Unit and integration tests for developer micro-benchmark runner tool.
 */

#include "mcp_tools.h"
#include "tests/mcp/mcp_test_util.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void test_bench_clustering_loop(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "target", "clustering_loop");
    cJSON_AddNumberToObject(args, "trials", 3);
    cJSON_AddNumberToObject(args, "warmup", 1);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_bench(args, res);

    CHECK(ret == 0);

    cJSON *target = cJSON_GetObjectItem(res, "target");
    CHECK(target != NULL && strcmp(target->valuestring, "clustering_loop") == 0);

    cJSON *trials = cJSON_GetObjectItem(res, "trials");
    CHECK(trials != NULL && trials->valueint == 3);

    cJSON *env = cJSON_GetObjectItem(res, "environment");
    CHECK(env != NULL);
    cJSON *gov = cJSON_GetObjectItem(env, "cpu_governor");
    CHECK(gov != NULL && cJSON_IsString(gov));

    cJSON *results = cJSON_GetObjectItem(res, "results");
    CHECK(results != NULL);

    cJSON *med = cJSON_GetObjectItem(results, "median_ms");
    cJSON *min_val = cJSON_GetObjectItem(results, "min_ms");
    cJSON *max_val = cJSON_GetObjectItem(results, "max_ms");
    cJSON *iqr = cJSON_GetObjectItem(results, "iqr_ms");

    CHECK(med != NULL && med->valuedouble > 0.0);
    CHECK(min_val != NULL && min_val->valuedouble > 0.0);
    CHECK(max_val != NULL && max_val->valuedouble >= min_val->valuedouble);
    CHECK(med->valuedouble >= min_val->valuedouble);
    CHECK(med->valuedouble <= max_val->valuedouble);
    CHECK(iqr != NULL && iqr->valuedouble >= 0.0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_bench_clustering_loop

static void test_bench_simd_kernels(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "target", "simd_kernels");
    cJSON_AddNumberToObject(args, "trials", 2);
    cJSON_AddNumberToObject(args, "warmup", 0);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_bench(args, res);

    CHECK(ret == 0);

    cJSON *kernels = cJSON_GetObjectItem(res, "kernels");
    CHECK(kernels != NULL && cJSON_IsArray(kernels));
    CHECK(cJSON_GetArraySize(kernels) > 0);

    cJSON *k0 = cJSON_GetArrayItem(kernels, 0);
    CHECK(k0 != NULL);
    CHECK(cJSON_GetObjectItem(k0, "kernel") != NULL);
    CHECK(cJSON_GetObjectItem(k0, "dimension") != NULL);
    CHECK(cJSON_GetObjectItem(k0, "scalar_sec") != NULL);
    CHECK(cJSON_GetObjectItem(k0, "avx2_sec") != NULL);
    CHECK(cJSON_GetObjectItem(k0, "avx2_speedup") != NULL);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_bench_simd_kernels

static void test_bench_baseline_save_and_compare(void)
{
    const char *tmp_base = "/tmp/test_mcp_bench_base.json";
    unlink(tmp_base);

    /* 1. First run: save baseline */
    cJSON *args1 = cJSON_CreateObject();
    cJSON_AddStringToObject(args1, "target", "clustering_loop");
    cJSON_AddNumberToObject(args1, "trials", 3);
    cJSON_AddStringToObject(args1, "baseline_file", tmp_base);
    cJSON_AddBoolToObject(args1, "save_baseline", 1);

    cJSON *res1 = cJSON_CreateObject();
    int ret1 = mcp_tool_dev_bench(args1, res1);

    CHECK(ret1 == 0);
    cJSON *comp1 = cJSON_GetObjectItem(res1, "comparison");
    CHECK(comp1 != NULL);
    cJSON *has_b1 = cJSON_GetObjectItem(comp1, "has_baseline");
    CHECK(has_b1 != NULL && !cJSON_IsTrue(has_b1));

    CHECK(access(tmp_base, R_OK) == 0);

    /* 2. Second run: compare against stored baseline */
    cJSON *args2 = cJSON_CreateObject();
    cJSON_AddStringToObject(args2, "target", "clustering_loop");
    cJSON_AddNumberToObject(args2, "trials", 3);
    cJSON_AddStringToObject(args2, "baseline_file", tmp_base);
    cJSON_AddBoolToObject(args2, "save_baseline", 0);
    cJSON_AddNumberToObject(args2, "regression_threshold_pct", 100.0);

    cJSON *res2 = cJSON_CreateObject();
    int ret2 = mcp_tool_dev_bench(args2, res2);

    CHECK(ret2 == 0);
    cJSON *comp2 = cJSON_GetObjectItem(res2, "comparison");
    CHECK(comp2 != NULL);
    cJSON *has_b2 = cJSON_GetObjectItem(comp2, "has_baseline");
    CHECK(has_b2 != NULL && cJSON_IsTrue(has_b2));

    cJSON *bm = cJSON_GetObjectItem(comp2, "baseline_median_ms");
    CHECK(bm != NULL && bm->valuedouble > 0.0);

    cJSON *status = cJSON_GetObjectItem(comp2, "status");
    CHECK(status != NULL && strcmp(status->valuestring, "PASS") == 0);

    cJSON_Delete(args1);
    cJSON_Delete(res1);
    cJSON_Delete(args2);
    cJSON_Delete(res2);
    unlink(tmp_base);
} // test_bench_baseline_save_and_compare

int main(void)
{
    printf("Running test_bench_clustering_loop...\n");
    test_bench_clustering_loop();

    printf("Running test_bench_simd_kernels...\n");
    test_bench_simd_kernels();

    printf("Running test_bench_baseline_save_and_compare...\n");
    test_bench_baseline_save_and_compare();

    printf("All test_mcp_bench tests PASSED!\n");
    return 0;
}

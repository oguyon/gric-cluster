/**
 * @file test_mcp_knn_quality.c
 * @brief Unit tests for kNN search recall and approximation quality evaluation tool.
 */

#include "mcp_tools.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_file(
    const char *path,
    const char *content)
{
    FILE *f = fopen(path, "w");
    assert(f != NULL);
    fputs(content, f);
    fclose(f);
}

static void setup_knn_fixtures(void)
{
    /* Simple 1D line of 10 points along the X axis */
    write_file(
        "/tmp/test_knn_q_dataset.txt",
        "0.0 0.0\n"
        "1.0 0.0\n"
        "2.0 0.0\n"
        "3.0 0.0\n"
        "4.0 0.0\n"
        "5.0 0.0\n"
        "6.0 0.0\n"
        "7.0 0.0\n"
        "8.0 0.0\n"
        "9.0 0.0\n");

    /* Exact k=2 results (excluding self) */
    write_file(
        "/tmp/test_knn_q_exact.txt",
        "# gric-knn results: k = 2, total_queries = 10\n"
        "0  1 1.000000  2 2.000000\n"
        "1  0 1.000000  2 1.000000\n"
        "2  1 1.000000  3 1.000000\n"
        "3  2 1.000000  4 1.000000\n"
        "4  3 1.000000  5 1.000000\n"
        "5  4 1.000000  6 1.000000\n"
        "6  5 1.000000  7 1.000000\n"
        "7  6 1.000000  8 1.000000\n"
        "8  7 1.000000  9 1.000000\n"
        "9  8 1.000000  7 2.000000\n");

    /* Degraded k=2 results (wrong neighbors) */
    write_file(
        "/tmp/test_knn_q_degraded.txt",
        "# gric-knn results: k = 2, total_queries = 10\n"
        "0  8 8.000000  9 9.000000\n"
        "1  8 7.000000  9 8.000000\n"
        "2  8 6.000000  9 7.000000\n"
        "3  8 5.000000  9 6.000000\n"
        "4  8 4.000000  9 5.000000\n"
        "5  0 5.000000  1 4.000000\n"
        "6  0 6.000000  1 5.000000\n"
        "7  0 7.000000  1 6.000000\n"
        "8  0 8.000000  1 7.000000\n"
        "9  0 9.000000  1 8.000000\n");
}

static void cleanup_knn_fixtures(void)
{
    unlink("/tmp/test_knn_q_dataset.txt");
    unlink("/tmp/test_knn_q_exact.txt");
    unlink("/tmp/test_knn_q_degraded.txt");
}

static void test_knn_quality_errors(void)
{
    /* Test 1: NULL args */
    cJSON *res1 = cJSON_CreateObject();
    int rc1 = mcp_tool_knn_quality(NULL, res1);
    assert(rc1 == -1);
    cJSON_Delete(res1);

    /* Test 2: Missing arguments */
    cJSON *args2 = cJSON_CreateObject();
    cJSON *res2 = cJSON_CreateObject();
    int rc2 = mcp_tool_knn_quality(args2, res2);
    assert(rc2 == -1);
    cJSON_Delete(args2);
    cJSON_Delete(res2);

    /* Test 3: Non-existent dataset */
    cJSON *args3 = cJSON_CreateObject();
    cJSON_AddStringToObject(args3, "dataset_path", "/tmp/nonexistent_dataset_9876.txt");
    cJSON_AddStringToObject(args3, "knn_output", "/tmp/test_knn_q_exact.txt");
    cJSON *res3 = cJSON_CreateObject();
    int rc3 = mcp_tool_knn_quality(args3, res3);
    assert(rc3 == -1);
    cJSON_Delete(args3);
    cJSON_Delete(res3);
}

static void test_knn_quality_exact(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset_path", "/tmp/test_knn_q_dataset.txt");
    cJSON_AddStringToObject(args, "knn_output", "/tmp/test_knn_q_exact.txt");
    cJSON_AddNumberToObject(args, "k", 2);
    cJSON_AddNumberToObject(args, "sample_queries", 10);
    cJSON_AddNumberToObject(args, "dtmin", 0);

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_knn_quality(args, res);
    assert(rc == 0);

    cJSON *metrics = cJSON_GetObjectItem(res, "accuracy_metrics");
    assert(metrics != NULL);

    double r1 = cJSON_GetObjectItem(metrics, "recall_at_1")->valuedouble;
    double rk = cJSON_GetObjectItem(metrics, "recall_at_k")->valuedouble;
    double mrr = cJSON_GetObjectItem(metrics, "mean_reciprocal_rank")->valuedouble;
    double ratio = cJSON_GetObjectItem(metrics, "mean_distance_ratio")->valuedouble;

    assert(fabs(r1 - 1.0) < 1e-4);
    assert(fabs(rk - 1.0) < 1e-4);
    assert(fabs(mrr - 1.0) < 1e-4);
    assert(fabs(ratio - 1.0) < 1e-4);

    cJSON *rep = cJSON_GetObjectItem(res, "report");
    assert(rep != NULL && strstr(rep->valuestring, "Exact") != NULL);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

static void test_knn_quality_degraded(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset_path", "/tmp/test_knn_q_dataset.txt");
    cJSON_AddStringToObject(args, "knn_output", "/tmp/test_knn_q_degraded.txt");
    cJSON_AddNumberToObject(args, "k", 2);
    cJSON_AddNumberToObject(args, "sample_queries", 10);
    cJSON_AddNumberToObject(args, "dtmin", 0);

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_knn_quality(args, res);
    assert(rc == 0);

    cJSON *metrics = cJSON_GetObjectItem(res, "accuracy_metrics");
    assert(metrics != NULL);

    double r1 = cJSON_GetObjectItem(metrics, "recall_at_1")->valuedouble;
    double rk = cJSON_GetObjectItem(metrics, "recall_at_k")->valuedouble;
    double ratio = cJSON_GetObjectItem(metrics, "mean_distance_ratio")->valuedouble;

    assert(r1 < 0.5);
    assert(rk < 0.5);
    assert(ratio > 1.5);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

int main(void)
{
    printf("[test_mcp_knn_quality] Setting up fixtures...\n");
    setup_knn_fixtures();

    printf("[test_mcp_knn_quality] Running error tests...\n");
    test_knn_quality_errors();

    printf("[test_mcp_knn_quality] Running exact quality tests...\n");
    test_knn_quality_exact();

    printf("[test_mcp_knn_quality] Running degraded quality tests...\n");
    test_knn_quality_degraded();

    cleanup_knn_fixtures();
    printf("[test_mcp_knn_quality] All kNN quality tests passed!\n");
    return 0;
}

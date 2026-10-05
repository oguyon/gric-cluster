/**
 * @file test_mcp_linter.c
 * @brief Unit tests for command validation and linting tool.
 */

#include "mcp_tools.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has_diagnostic_code(
    const cJSON *arr,
    const char  *code)
{
    if (arr == NULL || !cJSON_IsArray(arr))
    {
        return 0;
    }
    int sz = cJSON_GetArraySize(arr);
    for (int ii = 0; ii < sz; ii++)
    {
        cJSON *item = cJSON_GetArrayItem(arr, ii);
        cJSON *c = cJSON_GetObjectItem(item, "code");
        if (c != NULL && cJSON_IsString(c) && strcmp(c->valuestring, code) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static void test_validate_command_inputs(void)
{
    /* Test 1: NULL args */
    cJSON *res1 = cJSON_CreateObject();
    int rc1 = mcp_tool_validate_command(NULL, res1);
    assert(rc1 == -1);
    cJSON_Delete(res1);

    /* Test 2: Empty object */
    cJSON *args2 = cJSON_CreateObject();
    cJSON *res2 = cJSON_CreateObject();
    int rc2 = mcp_tool_validate_command(args2, res2);
    assert(rc2 == -1);
    cJSON_Delete(args2);
    cJSON_Delete(res2);

    /* Test 3: Unknown binary */
    cJSON *args3 = cJSON_CreateObject();
    cJSON_AddStringToObject(args3, "command", "gric-unknown-binary 123");
    cJSON *res3 = cJSON_CreateObject();
    int rc3 = mcp_tool_validate_command(args3, res3);
    assert(rc3 == 0);
    cJSON *valid3 = cJSON_GetObjectItem(res3, "valid");
    assert(valid3 != NULL && !cJSON_IsTrue(valid3));
    cJSON *errs3 = cJSON_GetObjectItem(res3, "errors");
    assert(has_diagnostic_code(errs3, "UNKNOWN_BINARY"));
    cJSON_Delete(args3);
    cJSON_Delete(res3);
}

static void test_validate_cluster_valid(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(
        args, "command",
        "gric-cluster 0.1 tests/test_strat.txt -maxcl 500 -sq8 -ncpu 4 -tiles 2x2");
    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_validate_command(args, res);
    assert(rc == 0);

    cJSON *valid = cJSON_GetObjectItem(res, "valid");
    assert(valid != NULL && cJSON_IsTrue(valid));

    cJSON *errs = cJSON_GetObjectItem(res, "errors");
    assert(errs != NULL && cJSON_GetArraySize(errs) == 0);

    cJSON *opts = cJSON_GetObjectItem(res, "parsed_options");
    assert(opts != NULL);
    assert(cJSON_GetObjectItem(opts, "rlim") != NULL);
    assert(cJSON_GetObjectItem(opts, "maxcl") != NULL);
    assert(cJSON_GetObjectItem(opts, "quantization") != NULL);
    assert(cJSON_GetObjectItem(opts, "ncpu") != NULL);
    assert(cJSON_GetObjectItem(opts, "tiles") != NULL);

    cJSON_Delete(args);
    cJSON_Delete(res);
}

static void test_validate_cluster_conflicts(void)
{
    /* Test 1: Multiple quantization flags */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(
            args, "command",
            "gric-cluster 0.1 data.fits -sq8 -sq16");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *valid = cJSON_GetObjectItem(res, "valid");
        assert(valid != NULL && !cJSON_IsTrue(valid));
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "CONFLICTING_QUANTIZATION"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 2: Double precision with quantization */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(
            args, "command",
            "gric-cluster 0.1 data.fits -double -eq16");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *valid = cJSON_GetObjectItem(res, "valid");
        assert(valid != NULL && !cJSON_IsTrue(valid));
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "CONFLICTING_PRECISION"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 3: Conflicting pruning flags */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(
            args, "command",
            "gric-cluster 0.1 data.fits -te4 -te5");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *valid = cJSON_GetObjectItem(res, "valid");
        assert(valid != NULL && !cJSON_IsTrue(valid));
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "CONFLICTING_PRUNING"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }
}

static void test_validate_cluster_ranges(void)
{
    /* Test 1: Missing radius */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-cluster -maxcl 500");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "MISSING_ARG"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 2: Negative radius */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-cluster -0.05 data.fits");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "INVALID_RADIUS"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 3: Invalid maxcl value */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-cluster 0.1 data.fits -maxcl -10");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "INVALID_VALUE"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 4: Invalid dprob */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-cluster 0.1 data.fits -dprob 1.5");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "INVALID_VALUE"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 5: Invalid tiles format */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-cluster 0.1 data.fits -tiles badformat");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "INVALID_FORMAT"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }
}

static void test_validate_knn(void)
{
    /* Test 1: Valid kNN command with array of args */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "binary", "gric-knn");
        cJSON *arr = cJSON_CreateArray();
        cJSON_AddItemToArray(arr, cJSON_CreateString("coords.txt"));
        cJSON_AddItemToArray(arr, cJSON_CreateString("clusters/"));
        cJSON_AddItemToArray(arr, cJSON_CreateString("-k"));
        cJSON_AddItemToArray(arr, cJSON_CreateString("10"));
        cJSON_AddItemToArray(arr, cJSON_CreateString("-dtmin"));
        cJSON_AddItemToArray(arr, cJSON_CreateString("5"));
        cJSON_AddItemToObject(args, "args", arr);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);

        cJSON *valid = cJSON_GetObjectItem(res, "valid");
        assert(valid != NULL && cJSON_IsTrue(valid));

        cJSON *opts = cJSON_GetObjectItem(res, "parsed_options");
        assert(opts != NULL);
        assert(cJSON_GetObjectItem(opts, "k") != NULL);
        assert(cJSON_GetObjectItem(opts, "dtmin") != NULL);

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 2: Missing positional arguments */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-knn -k 10");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "MISSING_ARG"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* Test 3: Invalid k value */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "command", "gric-knn coords.txt clusters/ -k -5");
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_validate_command(args, res);
        assert(rc == 0);
        cJSON *errs = cJSON_GetObjectItem(res, "errors");
        assert(has_diagnostic_code(errs, "INVALID_VALUE"));
        cJSON_Delete(args);
        cJSON_Delete(res);
    }
}

int main(void)
{
    printf("[test_mcp_linter] Starting command linter tests...\n");

    test_validate_command_inputs();
    test_validate_cluster_valid();
    test_validate_cluster_conflicts();
    test_validate_cluster_ranges();
    test_validate_knn();

    printf("[test_mcp_linter] All command linter tests passed!\n");
    return 0;
}

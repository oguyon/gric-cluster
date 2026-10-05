/**
 * @file test_mcp_sanitize.c
 * @brief Unit tests for gric_dev_sanitize tool and ASan/UBSan report parsing.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_registry_lookup() - Verify gric_dev_sanitize registration in MCP registry.
 */
static void test_registry_lookup(void)
{
    printf("[test_mcp_sanitize] Running test_registry_lookup...\n");

    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    const struct mcp_tool_def *tool = mcp_registry_find("gric_dev_sanitize");
    assert(tool != NULL);
    assert(strcmp(tool->name, "gric_dev_sanitize") == 0);
    assert(tool->toolset == MCP_TS_DEV);
    assert(tool->side_effects == 0);
    assert(tool->fn != NULL);
    assert(tool->input_schema != NULL);
} // test_registry_lookup

/**
 * test_parse_asan_report() - Verify parsing of AddressSanitizer crash dumps and stack traces.
 */
static void test_parse_asan_report(void)
{
    printf("[test_mcp_sanitize] Running test_parse_asan_report...\n");

    const char *asan_sample =
        "=================================================================\n"
        "==45678==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014\n"
        "READ of size 4 at 0x602000000014 thread T0\n"
        "    #0 0x401233 in test_overflow /home/oguyon/src/test_demo.c:25\n"
        "    #1 0x401456 in run_pipeline /home/oguyon/src/pipeline.c:88\n"
        "    #2 0x401678 in main /home/oguyon/src/main.c:15\n"
        "SUMMARY: AddressSanitizer: heap-buffer-overflow "
        "/home/oguyon/src/test_demo.c:25 in test_overflow\n"
        "=================================================================\n";

    cJSON *errors = cJSON_CreateArray();
    mcp_parse_sanitizer_output(asan_sample, errors);

    assert(cJSON_GetArraySize(errors) == 1);
    cJSON *err = cJSON_GetArrayItem(errors, 0);

    cJSON *tool = cJSON_GetObjectItemCaseSensitive(err, "tool");
    assert(tool != NULL && strcmp(tool->valuestring, "AddressSanitizer") == 0);

    cJSON *etype = cJSON_GetObjectItemCaseSensitive(err, "error_type");
    assert(etype != NULL && strcmp(etype->valuestring, "heap-buffer-overflow") == 0);

    cJSON *file = cJSON_GetObjectItemCaseSensitive(err, "file");
    assert(file != NULL && strcmp(file->valuestring, "/home/oguyon/src/test_demo.c") == 0);

    cJSON *line = cJSON_GetObjectItemCaseSensitive(err, "line");
    assert(line != NULL && line->valueint == 25);

    cJSON *fn = cJSON_GetObjectItemCaseSensitive(err, "function");
    assert(fn != NULL && strcmp(fn->valuestring, "test_overflow") == 0);

    cJSON *st = cJSON_GetObjectItemCaseSensitive(err, "stack_trace");
    assert(st != NULL && cJSON_GetArraySize(st) >= 3);

    cJSON_Delete(errors);
} // test_parse_asan_report

/**
 * test_parse_ubsan_report() - Verify parsing of UBSan undefined behavior diagnostics.
 */
static void test_parse_ubsan_report(void)
{
    printf("[test_mcp_sanitize] Running test_parse_ubsan_report...\n");

    const char *ubsan_sample =
        "/home/oguyon/src/math_ops.c:42:15: runtime error: signed integer overflow: "
        "2147483647 + 1 cannot be represented in type 'int'\n"
        "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior "
        "/home/oguyon/src/math_ops.c:42:15 in \n"
        "/home/oguyon/src/ptr_ops.c:105:8: runtime error: null pointer dereference\n";

    cJSON *errors = cJSON_CreateArray();
    mcp_parse_sanitizer_output(ubsan_sample, errors);

    assert(cJSON_GetArraySize(errors) == 2);

    /* First error: signed integer overflow */
    cJSON *e1 = cJSON_GetArrayItem(errors, 0);
    cJSON *t1 = cJSON_GetObjectItemCaseSensitive(e1, "tool");
    assert(t1 != NULL && strcmp(t1->valuestring, "UndefinedBehaviorSanitizer") == 0);
    cJSON *ty1 = cJSON_GetObjectItemCaseSensitive(e1, "error_type");
    assert(ty1 != NULL && strcmp(ty1->valuestring, "signed-integer-overflow") == 0);
    cJSON *f1 = cJSON_GetObjectItemCaseSensitive(e1, "file");
    assert(f1 != NULL && strcmp(f1->valuestring, "/home/oguyon/src/math_ops.c") == 0);
    cJSON *l1 = cJSON_GetObjectItemCaseSensitive(e1, "line");
    assert(l1 != NULL && l1->valueint == 42);

    /* Second error: null pointer dereference */
    cJSON *e2 = cJSON_GetArrayItem(errors, 1);
    cJSON *ty2 = cJSON_GetObjectItemCaseSensitive(e2, "error_type");
    assert(ty2 != NULL && strcmp(ty2->valuestring, "null-pointer-dereference") == 0);
    cJSON *f2 = cJSON_GetObjectItemCaseSensitive(e2, "file");
    assert(f2 != NULL && strcmp(f2->valuestring, "/home/oguyon/src/ptr_ops.c") == 0);
    cJSON *l2 = cJSON_GetObjectItemCaseSensitive(e2, "line");
    assert(l2 != NULL && l2->valueint == 105);

    cJSON_Delete(errors);
} // test_parse_ubsan_report

/**
 * test_parse_only_tool_invocation() - Test gric_dev_sanitize in parse_only mode.
 */
static void test_parse_only_tool_invocation(void)
{
    printf("[test_mcp_sanitize] Running test_parse_only_tool_invocation...\n");

    /* 1. Detection path */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddBoolToObject(args, "parse_only", 1);
        cJSON_AddStringToObject(
            args, "raw_log",
            "/home/user/src/calc.c:12:3: runtime error: division by zero\n");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_dev_sanitize(args, res);
        assert(rc == 0);

        cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
        assert(status != NULL && strcmp(status->valuestring, "SANITIZER_ERRORS_DETECTED") == 0);

        cJSON *ec = cJSON_GetObjectItemCaseSensitive(res, "errors_count");
        assert(ec != NULL && ec->valueint == 1);

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 2. Clean path */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddBoolToObject(args, "parse_only", 1);
        cJSON_AddStringToObject(args, "raw_log", "100% tests passed, 0 tests failed\n");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_dev_sanitize(args, res);
        assert(rc == 0);

        cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
        assert(status != NULL && strcmp(status->valuestring, "CLEAN") == 0);

        cJSON *ec = cJSON_GetObjectItemCaseSensitive(res, "errors_count");
        assert(ec != NULL && ec->valueint == 0);

        cJSON_Delete(args);
        cJSON_Delete(res);
    }
} // test_parse_only_tool_invocation

/**
 * test_error_handling() - Verify rejection of dangerous characters in configuration parameters.
 */
static void test_error_handling(void)
{
    printf("[test_mcp_sanitize] Running test_error_handling...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "build_dir", "build-asan; rm -rf /");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_dev_sanitize(args, res);
    assert(rc != 0);

    cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
    assert(err != NULL && cJSON_IsString(err));

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_error_handling

int main(void)
{
    test_registry_lookup();
    test_parse_asan_report();
    test_parse_ubsan_report();
    test_parse_only_tool_invocation();
    test_error_handling();

    printf("[test_mcp_sanitize] All sanitizer runner and parser tests PASSED.\n");
    return 0;
}

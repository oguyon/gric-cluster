/**
 * @file test_mcp_dev_build.c
 * @brief Unit and integration tests for developer build and test runner tool.
 */

#include "mcp_tools.h"
#include "tests/mcp/mcp_test_util.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_compiler_diagnostics_parser(void)
{
    const char *sample_log =
        "[ 10%] Building C object CMakeFiles/foo.dir/src/foo.c.o\n"
        "src/foo.c:42:10: warning: unused variable 'x' [-Wunused-variable]\n"
        "   42 |     int x = 0;\n"
        "      |          ^\n"
        "src/bar.c:105:5: error: expected ';' before 'return'\n"
        "src/baz.c:1:10: fatal error: nonexistent.h: No such file or directory\n"
        "[ 50%] Linking C shared library libfoo.so\n";

    cJSON *warnings = cJSON_CreateArray();
    cJSON *errors = cJSON_CreateArray();

    mcp_parse_compiler_diagnostics(sample_log, warnings, errors);

    CHECK(cJSON_GetArraySize(warnings) == 1);
    cJSON *w0 = cJSON_GetArrayItem(warnings, 0);
    CHECK(w0 != NULL);
    cJSON *w_file = cJSON_GetObjectItem(w0, "file");
    CHECK(w_file != NULL && strcmp(w_file->valuestring, "src/foo.c") == 0);
    cJSON *w_line = cJSON_GetObjectItem(w0, "line");
    CHECK(w_line != NULL && w_line->valueint == 42);
    cJSON *w_col = cJSON_GetObjectItem(w0, "column");
    CHECK(w_col != NULL && w_col->valueint == 10);
    cJSON *w_lvl = cJSON_GetObjectItem(w0, "level");
    CHECK(w_lvl != NULL && strcmp(w_lvl->valuestring, "warning") == 0);

    CHECK(cJSON_GetArraySize(errors) == 2);
    cJSON *e0 = cJSON_GetArrayItem(errors, 0);
    CHECK(e0 != NULL);
    cJSON *e0_file = cJSON_GetObjectItem(e0, "file");
    CHECK(e0_file != NULL && strcmp(e0_file->valuestring, "src/bar.c") == 0);
    cJSON *e0_line = cJSON_GetObjectItem(e0, "line");
    CHECK(e0_line != NULL && e0_line->valueint == 105);

    cJSON *e1 = cJSON_GetArrayItem(errors, 1);
    CHECK(e1 != NULL);
    cJSON *e1_file = cJSON_GetObjectItem(e1, "file");
    CHECK(e1_file != NULL && strcmp(e1_file->valuestring, "src/baz.c") == 0);
    cJSON *e1_line = cJSON_GetObjectItem(e1, "line");
    CHECK(e1_line != NULL && e1_line->valueint == 1);

    cJSON_Delete(warnings);
    cJSON_Delete(errors);
} // test_compiler_diagnostics_parser

static void test_ctest_parser(void)
{
    const char *sample_ctest =
        "Test project /home/oguyon/src/gric-cluster/build\n"
        "    Start 1: test_sample_pass\n"
        "1/2 Test #1: test_sample_pass ................   Passed    0.05 sec\n"
        "    Start 2: test_sample_fail\n"
        "2/2 Test #2: test_sample_fail ................***Failed    0.12 sec\n"
        "\n"
        "50% tests passed, 1 tests failed out of 2\n"
        "\n"
        "Total Test time (real) =   0.20 sec\n"
        "\n"
        "The following tests FAILED:\n"
        "\t  2 - test_sample_fail (Failed)\n"
        "Errors were encountered while processing:\n"
        " test_sample_fail\n";

    cJSON *summary = cJSON_CreateObject();
    cJSON *failures = cJSON_CreateArray();

    mcp_parse_ctest_output(sample_ctest, summary, failures);

    cJSON *tot = cJSON_GetObjectItem(summary, "total");
    CHECK(tot != NULL && tot->valueint == 2);

    cJSON *pass = cJSON_GetObjectItem(summary, "passed");
    CHECK(pass != NULL && pass->valueint == 1);

    cJSON *fail = cJSON_GetObjectItem(summary, "failed");
    CHECK(fail != NULL && fail->valueint == 1);

    cJSON *rate = cJSON_GetObjectItem(summary, "pass_rate");
    CHECK(rate != NULL && rate->valuedouble >= 49.9 && rate->valuedouble <= 50.1);

    CHECK(cJSON_GetArraySize(failures) == 1);
    cJSON *f0 = cJSON_GetArrayItem(failures, 0);
    CHECK(f0 != NULL);
    cJSON *f0_name = cJSON_GetObjectItem(f0, "test_name");
    CHECK(f0_name != NULL && strcmp(f0_name->valuestring, "test_sample_fail") == 0);
    cJSON *f0_idx = cJSON_GetObjectItem(f0, "test_index");
    CHECK(f0_idx != NULL && f0_idx->valueint == 2);

    cJSON_Delete(summary);
    cJSON_Delete(failures);
} // test_ctest_parser

static void test_dev_build_runner_e2e(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "target", "test_cli_opt");
    cJSON_AddBoolToObject(args, "run_tests", 1);
    cJSON_AddStringToObject(args, "test_filter", "test_cli_opt");

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_dev_build_test(args, res);

    CHECK(ret == 0);

    cJSON *status = cJSON_GetObjectItem(res, "build_status");
    CHECK(status != NULL);
    CHECK_MSG(strcmp(status->valuestring, "SUCCESS") == 0, status->valuestring);

    cJSON *exit_code = cJSON_GetObjectItem(res, "exit_code");
    CHECK(exit_code != NULL && exit_code->valueint == 0);

    cJSON *test_summary = cJSON_GetObjectItem(res, "test_summary");
    CHECK(test_summary != NULL);
    cJSON *failed_cnt = cJSON_GetObjectItem(test_summary, "failed");
    CHECK(failed_cnt != NULL && failed_cnt->valueint == 0);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_dev_build_runner_e2e

int main(void)
{
    printf("Running test_compiler_diagnostics_parser...\n");
    test_compiler_diagnostics_parser();

    printf("Running test_ctest_parser...\n");
    test_ctest_parser();

    printf("Running test_dev_build_runner_e2e...\n");
    test_dev_build_runner_e2e();

    printf("All test_mcp_dev_build tests PASSED!\n");
    return 0;
}

/**
 * @file tool_dev_build_test.c
 * @brief High-speed developer build runner and CTest failure/warning diagnostic parser.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_exec.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUILD_OUTPUT_BYTES 524288 /* 512 KB */

/**
 * is_safe_filter() - Check if a regex filter string contains only safe characters.
 * @str: Input filter string.
 *
 * Return: 1 if safe, 0 if contains dangerous or control characters.
 */
static int is_safe_filter(
    const char *str)
{
    if (str == NULL || *str == '\0')
    {
        return 0;
    }
    for (size_t ii = 0; str[ii] != '\0'; ii++)
    {
        unsigned char c = (unsigned char)str[ii];
        if (c < 32 || c >= 127 || strchr(";\"'`$|&><\n\r", c) != NULL)
        {
            return 0;
        }
    }
    return 1;
}

/**
 * parse_compiler_line() - Parse a single compiler warning or error line.
 * @line:         Raw compiler output line.
 * @warnings_arr: cJSON array to append parsed warning to.
 * @errors_arr:   cJSON array to append parsed error to.
 */
static void parse_compiler_line(
    const char *line,
    cJSON      *warnings_arr,
    cJSON      *errors_arr)
{
    if (line == NULL || *line == '\0')
    {
        return;
    }

    const char *p_warn = strstr(line, ": warning: ");
    const char *p_err = strstr(line, ": error: ");
    const char *p_fatal = strstr(line, ": fatal error: ");

    const char *tag = NULL;
    int is_error = 0;
    size_t tag_len = 0;

    if (p_err != NULL)
    {
        tag = p_err;
        is_error = 1;
        tag_len = 9;
    }
    else if (p_fatal != NULL)
    {
        tag = p_fatal;
        is_error = 1;
        tag_len = 15;
    }
    else if (p_warn != NULL)
    {
        tag = p_warn;
        is_error = 0;
        tag_len = 11;
    }
    else
    {
        return;
    }

    /* Extract prefix before tag: <filepath>:<line>[:<col>] */
    size_t prefix_len = (size_t)(tag - line);
    if (prefix_len >= 1024 || prefix_len < 3)
    {
        return;
    }

    char prefix[1024];
    memcpy(prefix, line, prefix_len);
    prefix[prefix_len] = '\0';

    /* Find last colon (column or line) */
    char *colon2 = strrchr(prefix, ':');
    if (colon2 == NULL)
    {
        return;
    }

    int line_num = 0;
    int col_num = 0;
    char *colon1 = NULL;

    *colon2 = '\0';
    colon1 = strrchr(prefix, ':');
    if (colon1 != NULL)
    {
        *colon1 = '\0';
        line_num = atoi(colon1 + 1);
        col_num = atoi(colon2 + 1);
    }
    else
    {
        line_num = atoi(colon2 + 1);
        col_num = 0;
    }

    if (line_num <= 0)
    {
        return;
    }

    cJSON *item = cJSON_CreateObject();
    cJSON_AddStringToObject(item, "file", prefix);
    cJSON_AddNumberToObject(item, "line", line_num);
    if (col_num > 0)
    {
        cJSON_AddNumberToObject(item, "column", col_num);
    }
    cJSON_AddStringToObject(item, "level", is_error ? "error" : "warning");
    cJSON_AddStringToObject(item, "message", tag + tag_len);

    if (is_error)
    {
        cJSON_AddItemToArray(errors_arr, item);
    }
    else
    {
        cJSON_AddItemToArray(warnings_arr, item);
    }
} // parse_compiler_line

/**
 * parse_compiler_diagnostics() - Parse entire build log into warnings and errors.
 * @output:       Merged compiler stdout/stderr string.
 * @warnings_arr: cJSON array receiving warnings.
 * @errors_arr:   cJSON array receiving errors.
 */
void mcp_parse_compiler_diagnostics(
    const char *output,
    cJSON      *warnings_arr,
    cJSON      *errors_arr)
{
    if (output == NULL)
    {
        return;
    }

    const char *cur = output;
    char line[2048];

    while (*cur != '\0')
    {
        const char *next = strchr(cur, '\n');
        size_t len = (next != NULL) ? (size_t)(next - cur) : strlen(cur);

        if (len < sizeof(line))
        {
            memcpy(line, cur, len);
            line[len] = '\0';
            parse_compiler_line(line, warnings_arr, errors_arr);
        }

        if (next == NULL)
        {
            break;
        }
        cur = next + 1;
    } // while cur
} // mcp_parse_compiler_diagnostics

/**
 * mcp_parse_ctest_output() - Parse ctest output into summary and failed test list.
 * @output:       Raw ctest output string.
 * @summary_obj:  cJSON object to populate with test counts and durations.
 * @failures_arr: cJSON array to populate with failing test objects.
 */
void mcp_parse_ctest_output(
    const char *output,
    cJSON      *summary_obj,
    cJSON      *failures_arr)
{
    if (output == NULL)
    {
        return;
    }

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    double duration_sec = 0.0;

    const char *cur = output;
    char line[2048];

    while (*cur != '\0')
    {
        const char *next = strchr(cur, '\n');
        size_t len = (next != NULL) ? (size_t)(next - cur) : strlen(cur);

        if (len < sizeof(line))
        {
            memcpy(line, cur, len);
            line[len] = '\0';

            /* Summary percentage: "100% tests passed, 0 tests failed out of 75" */
            const char *p_pct = strstr(line, "% tests passed, ");
            if (p_pct != NULL)
            {
                int f = 0;
                int t = 0;
                if (sscanf(p_pct + 16, "%d tests failed out of %d", &f, &t) == 2)
                {
                    failed_tests = f;
                    total_tests = t;
                    passed_tests = t - f;
                }
            }

            /* Duration: "Total Test time (real) =  18.31 sec" */
            const char *p_time = strstr(line, "Total Test time (real) =");
            if (p_time != NULL)
            {
                sscanf(p_time + 24, "%lf", &duration_sec);
            }

            /* Failure line: " 64/74 Test  #64: test_mcp_contract ...***Failed  0.00 sec" */
            const char *p_failed = strstr(line, "***Failed");
            if (p_failed != NULL)
            {
                int test_idx = 0;
                char test_name[128];
                test_name[0] = '\0';

                const char *p_test = strstr(line, "Test  #");
                if (p_test == NULL)
                {
                    p_test = strstr(line, "Test #");
                }
                if (p_test != NULL)
                {
                    const char *colon = strchr(p_test, ':');
                    if (colon != NULL)
                    {
                        sscanf(colon + 1, "%127s", test_name);
                        const char *hash = strchr(p_test, '#');
                        if (hash != NULL)
                        {
                            test_idx = atoi(hash + 1);
                        }
                    }
                }

                cJSON *f_item = cJSON_CreateObject();
                cJSON_AddStringToObject(f_item, "test_name", test_name);
                cJSON_AddNumberToObject(f_item, "test_index", test_idx);
                cJSON_AddItemToArray(failures_arr, f_item);
            }
        }

        if (next == NULL)
        {
            break;
        }
        cur = next + 1;
    } // while cur

    cJSON_AddNumberToObject(summary_obj, "total", total_tests);
    cJSON_AddNumberToObject(summary_obj, "passed", passed_tests);
    cJSON_AddNumberToObject(summary_obj, "failed", failed_tests);
    double pass_rate = (total_tests > 0) ? ((double)passed_tests / total_tests * 100.0) : 100.0;
    cJSON_AddNumberToObject(summary_obj, "pass_rate", pass_rate);
    cJSON_AddNumberToObject(summary_obj, "duration_sec", duration_sec);
} // mcp_parse_ctest_output

int mcp_tool_dev_build_test(
    const cJSON *args,
    cJSON       *res)
{
    const struct mcp_server_config *cfg = mcp_registry_get_config();
    if (cfg != NULL && !cfg->has_source_tree)
    {
        cJSON_AddStringToObject(res, "error", "Tool requires local source tree");
        return -1;
    }

    char root[1024];
    mcp_get_project_root(root, sizeof(root));
    if (root[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "Could not locate project root directory");
        return -1;
    }

    char build_dir[1040];
    snprintf(build_dir, sizeof(build_dir), "%s/build", root);

    /* 1. Parse arguments */
    const char *target = NULL;
    int run_tests = 1;
    const char *test_filter = NULL;
    int do_clean = 0;

    if (args != NULL)
    {
        cJSON *t_item = cJSON_GetObjectItemCaseSensitive(args, "target");
        if (t_item != NULL && cJSON_IsString(t_item) && t_item->valuestring[0] != '\0')
        {
            target = t_item->valuestring;
        }

        cJSON *r_item = cJSON_GetObjectItemCaseSensitive(args, "run_tests");
        if (r_item != NULL && cJSON_IsBool(r_item))
        {
            run_tests = cJSON_IsTrue(r_item) ? 1 : 0;
        }

        cJSON *f_item = cJSON_GetObjectItemCaseSensitive(args, "test_filter");
        if (f_item != NULL && cJSON_IsString(f_item) && f_item->valuestring[0] != '\0')
        {
            if (!is_safe_filter(f_item->valuestring))
            {
                cJSON_AddStringToObject(res, "error", "Invalid characters in test_filter");
                return -1;
            }
            test_filter = f_item->valuestring;
        }

        cJSON *c_item = cJSON_GetObjectItemCaseSensitive(args, "clean");
        if (c_item != NULL && cJSON_IsBool(c_item))
        {
            do_clean = cJSON_IsTrue(c_item) ? 1 : 0;
        }
    }

    /* 2. Optional clean */
    if (do_clean)
    {
        const char *clean_argv[] = {
            "cmake", "--build", build_dir, "--target", "clean", NULL
        };
        int clean_exit = 0;
        mcp_exec_capture(clean_argv, NULL, 0, 30000, &clean_exit);
    }

    /* 3. Build step */
    char *build_output = malloc(MAX_BUILD_OUTPUT_BYTES);
    if (build_output == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Memory allocation failed for build output buffer");
        return -1;
    }
    build_output[0] = '\0';

    const char *build_argv[8];
    int arg_idx = 0;
    build_argv[arg_idx++] = "cmake";
    build_argv[arg_idx++] = "--build";
    build_argv[arg_idx++] = build_dir;
    if (target != NULL && strcmp(target, "all") != 0)
    {
        build_argv[arg_idx++] = "--target";
        build_argv[arg_idx++] = target;
    }
    build_argv[arg_idx++] = "-j";
    build_argv[arg_idx] = NULL;

    int build_exit = 0;
    int exec_ret = mcp_exec_capture(
        build_argv, build_output, MAX_BUILD_OUTPUT_BYTES, 300000, &build_exit);

    if (exec_ret != 0 && exec_ret != -2)
    {
        free(build_output);
        cJSON_AddStringToObject(res, "error", "Failed to launch cmake --build");
        return -1;
    }

    cJSON *warnings_arr = cJSON_CreateArray();
    cJSON *errors_arr = cJSON_CreateArray();
    mcp_parse_compiler_diagnostics(build_output, warnings_arr, errors_arr);

    cJSON_AddNumberToObject(
        res, "compiler_warnings_count", (double)cJSON_GetArraySize(warnings_arr));
    cJSON_AddItemToObject(res, "compiler_warnings", warnings_arr);
    cJSON_AddNumberToObject(
        res, "compiler_errors_count", (double)cJSON_GetArraySize(errors_arr));
    cJSON_AddItemToObject(res, "compiler_errors", errors_arr);

    free(build_output);

    if (build_exit != 0 || exec_ret == -2)
    {
        cJSON_AddStringToObject(res, "build_status", "BUILD_FAILED");
        cJSON_AddNumberToObject(res, "exit_code", build_exit);
        return 0;
    }

    /* 4. Optional CTest execution */
    if (!run_tests)
    {
        cJSON_AddStringToObject(res, "build_status", "SUCCESS");
        cJSON_AddNumberToObject(res, "exit_code", build_exit);
        return 0;
    }

    char *ctest_output = malloc(MAX_BUILD_OUTPUT_BYTES);
    if (ctest_output == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Memory allocation failed for ctest output buffer");
        return -1;
    }
    ctest_output[0] = '\0';

    const char *ctest_argv[8];
    int ctest_idx = 0;
    ctest_argv[ctest_idx++] = "ctest";
    ctest_argv[ctest_idx++] = "--test-dir";
    ctest_argv[ctest_idx++] = build_dir;
    ctest_argv[ctest_idx++] = "--output-on-failure";
    if (test_filter != NULL)
    {
        ctest_argv[ctest_idx++] = "-R";
        ctest_argv[ctest_idx++] = test_filter;
    }
    ctest_argv[ctest_idx] = NULL;

    int ctest_exit = 0;
    int ctest_exec = mcp_exec_capture(
        ctest_argv, ctest_output, MAX_BUILD_OUTPUT_BYTES, 300000, &ctest_exit);

    if (ctest_exec != 0 && ctest_exec != -2)
    {
        free(ctest_output);
        cJSON_AddStringToObject(res, "error", "Failed to launch ctest");
        return -1;
    }

    cJSON *summary_obj = cJSON_CreateObject();
    cJSON *failures_arr = cJSON_CreateArray();
    mcp_parse_ctest_output(ctest_output, summary_obj, failures_arr);

    cJSON_AddItemToObject(res, "test_summary", summary_obj);
    cJSON_AddItemToObject(res, "test_failures", failures_arr);

    free(ctest_output);

    cJSON_AddNumberToObject(res, "exit_code", ctest_exit);
    if (ctest_exit != 0 || cJSON_GetArraySize(failures_arr) > 0)
    {
        cJSON_AddStringToObject(res, "build_status", "TESTS_FAILED");
    }
    else
    {
        cJSON_AddStringToObject(res, "build_status", "SUCCESS");
    }

    return 0;
} // mcp_tool_dev_build_test

const struct mcp_tool_def mcp_tooldef_dev_build_test = {
    .name         = "gric_dev_build_test",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_dev_build_test,
    .description  = "Incremental compilation and CTest runner: compresses output into "
                    "structured compiler warnings [file, line, message] and failing test list.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"target\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Build target name (default: all).\"\n"
        "    },\n"
        "    \"run_tests\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Run test suite after build (default: true).\"\n"
        "    },\n"
        "    \"test_filter\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Regex filter passed to ctest -R.\"\n"
        "    },\n"
        "    \"clean\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Clean build targets first (default: false).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

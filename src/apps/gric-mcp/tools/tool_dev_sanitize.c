/**
 * @file tool_dev_sanitize.c
 * @brief AddressSanitizer and UndefinedBehaviorSanitizer runner and diagnostic parser.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_exec.h"
#include "shared/cjson/cJSON.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SANITIZE_OUTPUT_BYTES 1048576 /* 1 MB */

/**
 * is_safe_param() - Check if a parameter string contains only safe path/flag characters.
 * @str: Input string to validate.
 *
 * Return: 1 if safe, 0 if contains dangerous characters.
 */
static int is_safe_param(
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
} // is_safe_param

/**
 * trim_whitespace() - Strip leading and trailing whitespace from string in-place.
 * @str: String buffer.
 *
 * Return: Pointer to first non-whitespace character.
 */
static char *trim_whitespace(
    char *str)
{
    if (str == NULL)
    {
        return NULL;
    }
    while (isspace((unsigned char)*str))
    {
        str++;
    }
    if (*str == '\0')
    {
        return str;
    }
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end))
    {
        *end = '\0';
        end--;
    }
    return str;
} // trim_whitespace

/**
 * parse_ubsan_type() - Map UBSan runtime error description to canonical error type.
 * @desc: Runtime error description text.
 *
 * Return: Static string with canonical error type.
 */
static const char *parse_ubsan_type(
    const char *desc)
{
    if (strstr(desc, "signed integer overflow") != NULL)
    {
        return "signed-integer-overflow";
    }
    if (strstr(desc, "null pointer") != NULL)
    {
        return "null-pointer-dereference";
    }
    if (strstr(desc, "out of bounds") != NULL)
    {
        return "out-of-bounds";
    }
    if (strstr(desc, "division by zero") != NULL)
    {
        return "division-by-zero";
    }
    if (strstr(desc, "misaligned") != NULL)
    {
        return "misaligned-pointer-use";
    }
    if (strstr(desc, "shift exponent") != NULL)
    {
        return "invalid-shift-exponent";
    }
    if (strstr(desc, "unreachable") != NULL)
    {
        return "unreachable-code";
    }
    return "undefined-behavior";
} // parse_ubsan_type

void mcp_parse_sanitizer_output(
    const char *output,
    cJSON      *errors_arr)
{
    if (output == NULL || errors_arr == NULL)
    {
        return;
    }

    const char *p = output;
    cJSON *cur_error = NULL;
    cJSON *cur_stack = NULL;

    while (*p != '\0')
    {
        const char *line_end = strchr(p, '\n');
        size_t line_len = (line_end != NULL) ? (size_t)(line_end - p) : strlen(p);

        char line[2048];
        if (line_len >= sizeof(line))
        {
            line_len = sizeof(line) - 1;
        }
        memcpy(line, p, line_len);
        line[line_len] = '\0';

        p = (line_end != NULL) ? line_end + 1 : p + line_len;

        char *trimmed = trim_whitespace(line);
        if (*trimmed == '\0')
        {
            continue;
        }

        /* 1. AddressSanitizer ERROR line: ==123==ERROR: AddressSanitizer: <type> on address ... */
        const char *asan_err = strstr(trimmed, "ERROR: AddressSanitizer: ");
        if (asan_err != NULL)
        {
            const char *type_start = asan_err + 25;
            char type_buf[128];
            size_t ti = 0;
            while (type_start[ti] != '\0' && !isspace((unsigned char)type_start[ti]) &&
                   type_start[ti] != ':' && ti < sizeof(type_buf) - 1)
            {
                type_buf[ti] = type_start[ti];
                ti++;
            }
            type_buf[ti] = '\0';

            cur_error = cJSON_CreateObject();
            cJSON_AddStringToObject(cur_error, "tool", "AddressSanitizer");
            cJSON_AddStringToObject(cur_error, "error_type", type_buf);
            cJSON_AddStringToObject(cur_error, "description", trimmed);
            cJSON_AddStringToObject(cur_error, "file", "");
            cJSON_AddNumberToObject(cur_error, "line", 0.0);
            cJSON_AddStringToObject(cur_error, "function", "");

            cur_stack = cJSON_CreateArray();
            cJSON_AddItemToObject(cur_error, "stack_trace", cur_stack);
            cJSON_AddItemToArray(errors_arr, cur_error);
            continue;
        }

        /* 2. AddressSanitizer or UBSan SUMMARY line */
        const char *summary_asan = strstr(trimmed, "SUMMARY: AddressSanitizer: ");
        const char *summary_ubsan = strstr(trimmed, "SUMMARY: UndefinedBehaviorSanitizer: ");

        if (summary_asan != NULL || summary_ubsan != NULL)
        {
            int is_asan = (summary_asan != NULL);
            const char *hdr = is_asan ? (summary_asan + 27) : (summary_ubsan + 37);

            /* Format: <error_type> <file>:<line>[:<col>] in <func> */
            char err_type[128] = {0};
            char location[512] = {0};
            char func_name[128] = {0};

            sscanf(hdr, "%127s %511s in %127s", err_type, location, func_name);

            char file_buf[512] = {0};
            int line_no = 0;
            char *colon = strchr(location, ':');
            if (colon != NULL)
            {
                size_t flen = (size_t)(colon - location);
                if (flen < sizeof(file_buf))
                {
                    memcpy(file_buf, location, flen);
                    file_buf[flen] = '\0';
                }
                line_no = atoi(colon + 1);
            }
            else
            {
                snprintf(file_buf, sizeof(file_buf), "%s", location);
            }

            if (cur_error != NULL)
            {
                /* Update existing record with location details from summary */
                cJSON_ReplaceItemInObject(cur_error, "file", cJSON_CreateString(file_buf));
                cJSON_ReplaceItemInObject(
                    cur_error, "line", cJSON_CreateNumber((double)line_no));
                if (func_name[0] != '\0')
                {
                    cJSON_ReplaceItemInObject(
                        cur_error, "function", cJSON_CreateString(func_name));
                }
                cur_error = NULL;
                cur_stack = NULL;
            }
            else
            {
                /* Standalone summary without previous ERROR or runtime error line */
                cur_error = cJSON_CreateObject();
                cJSON_AddStringToObject(
                    cur_error, "tool", is_asan ? "AddressSanitizer" : "UndefinedBehaviorSanitizer");
                cJSON_AddStringToObject(cur_error, "error_type", err_type);
                cJSON_AddStringToObject(cur_error, "description", trimmed);
                cJSON_AddStringToObject(cur_error, "file", file_buf);
                cJSON_AddNumberToObject(cur_error, "line", (double)line_no);
                cJSON_AddStringToObject(cur_error, "function", func_name);

                cur_stack = cJSON_CreateArray();
                cJSON_AddItemToObject(cur_error, "stack_trace", cur_stack);
                cJSON_AddItemToArray(errors_arr, cur_error);
                cur_error = NULL;
                cur_stack = NULL;
            }
            continue;
        }

        /* 3. UBSan runtime error line: <filepath>:<line>:<col>: runtime error: <desc> */
        const char *rt_err = strstr(trimmed, "runtime error: ");
        if (rt_err != NULL)
        {
            const char *desc = rt_err + 15;
            const char *u_type = parse_ubsan_type(desc);

            char file_buf[512] = {0};
            int line_no = 0;

            if (rt_err > trimmed)
            {
                size_t pfx_len = (size_t)(rt_err - trimmed);
                if (pfx_len >= 3)
                {
                    char pfx[512];
                    if (pfx_len >= sizeof(pfx))
                    {
                        pfx_len = sizeof(pfx) - 1;
                    }
                    memcpy(pfx, trimmed, pfx_len);
                    pfx[pfx_len] = '\0';

                    char *c1 = strchr(pfx, ':');
                    if (c1 != NULL)
                    {
                        size_t flen = (size_t)(c1 - pfx);
                        if (flen < sizeof(file_buf))
                        {
                            memcpy(file_buf, pfx, flen);
                            file_buf[flen] = '\0';
                        }
                        line_no = atoi(c1 + 1);
                    }
                }
            }

            cur_error = cJSON_CreateObject();
            cJSON_AddStringToObject(cur_error, "tool", "UndefinedBehaviorSanitizer");
            cJSON_AddStringToObject(cur_error, "error_type", u_type);
            cJSON_AddStringToObject(cur_error, "description", desc);
            cJSON_AddStringToObject(cur_error, "file", file_buf);
            cJSON_AddNumberToObject(cur_error, "line", (double)line_no);
            cJSON_AddStringToObject(cur_error, "function", "");

            cur_stack = cJSON_CreateArray();
            cJSON_AddItemToObject(cur_error, "stack_trace", cur_stack);
            cJSON_AddItemToArray(errors_arr, cur_error);
            continue;
        }

        /* 4. Stack frame line: #0 0x... in <func> <file>:<line> */
        if (cur_stack != NULL && trimmed[0] == '#' && isdigit((unsigned char)trimmed[1]))
        {
            cJSON_AddItemToArray(cur_stack, cJSON_CreateString(trimmed));
        }
    } // while reading lines
} // mcp_parse_sanitizer_output

int mcp_tool_dev_sanitize(
    const cJSON *args,
    cJSON       *res)
{
    /* Internal testing fast-path: parse raw log string without spawning subprocesses */
    if (args != NULL)
    {
        cJSON *po = cJSON_GetObjectItemCaseSensitive(args, "parse_only");
        if (po != NULL && cJSON_IsTrue(po))
        {
            cJSON *rl = cJSON_GetObjectItemCaseSensitive(args, "raw_log");
            const char *raw = (rl != NULL && cJSON_IsString(rl)) ? rl->valuestring : "";
            cJSON *errs = cJSON_CreateArray();
            mcp_parse_sanitizer_output(raw, errs);
            int count = cJSON_GetArraySize(errs);
            cJSON_AddStringToObject(
                res, "status", (count == 0) ? "CLEAN" : "SANITIZER_ERRORS_DETECTED");
            cJSON_AddNumberToObject(res, "errors_count", (double)count);
            cJSON_AddItemToObject(res, "errors", errs);
            return 0;
        }
    }

    char root[1024];
    mcp_get_project_root(root, sizeof(root));
    if (root[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "Could not locate project root directory");
        return -1;
    }

    const char *build_dir_name = "build-asan";
    const char *target = NULL;
    const char *test_filter = NULL;
    const char *c_flags = "-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer";
    int run_tests = 1;
    int do_clean = 0;
    int timeout_ms = 300000;

    if (args != NULL)
    {
        cJSON *bd = cJSON_GetObjectItemCaseSensitive(args, "build_dir");
        if (bd != NULL && cJSON_IsString(bd) && bd->valuestring[0] != '\0')
        {
            if (!is_safe_param(bd->valuestring))
            {
                cJSON_AddStringToObject(res, "error", "Invalid characters in build_dir");
                return -1;
            }
            build_dir_name = bd->valuestring;
        }

        cJSON *tg = cJSON_GetObjectItemCaseSensitive(args, "target");
        if (tg != NULL && cJSON_IsString(tg) && tg->valuestring[0] != '\0')
        {
            if (!is_safe_param(tg->valuestring))
            {
                cJSON_AddStringToObject(res, "error", "Invalid characters in target");
                return -1;
            }
            target = tg->valuestring;
        }

        cJSON *tf = cJSON_GetObjectItemCaseSensitive(args, "test_filter");
        if (tf != NULL && cJSON_IsString(tf) && tf->valuestring[0] != '\0')
        {
            if (!is_safe_param(tf->valuestring))
            {
                cJSON_AddStringToObject(res, "error", "Invalid characters in test_filter");
                return -1;
            }
            test_filter = tf->valuestring;
        }

        cJSON *cf = cJSON_GetObjectItemCaseSensitive(args, "c_flags");
        if (cf != NULL && cJSON_IsString(cf) && cf->valuestring[0] != '\0')
        {
            if (!is_safe_param(cf->valuestring))
            {
                cJSON_AddStringToObject(res, "error", "Invalid characters in c_flags");
                return -1;
            }
            c_flags = cf->valuestring;
        }

        cJSON *cl = cJSON_GetObjectItemCaseSensitive(args, "clean");
        if (cl != NULL && cJSON_IsBool(cl))
        {
            do_clean = cJSON_IsTrue(cl) ? 1 : 0;
        }

        cJSON *rt = cJSON_GetObjectItemCaseSensitive(args, "run_tests");
        if (rt != NULL && cJSON_IsBool(rt))
        {
            run_tests = cJSON_IsTrue(rt) ? 1 : 0;
        }

        cJSON *to = cJSON_GetObjectItemCaseSensitive(args, "timeout_ms");
        if (to != NULL && cJSON_IsNumber(to) && to->valueint > 0)
        {
            timeout_ms = to->valueint;
        }
    }

    char build_dir[1040];
    if (build_dir_name[0] == '/')
    {
        strncpy(build_dir, build_dir_name, sizeof(build_dir) - 1);
        build_dir[sizeof(build_dir) - 1] = '\0';
    }
    else
    {
        snprintf(build_dir, sizeof(build_dir), "%s/%s", root, build_dir_name);
    }

    /* 1. Optional clean */
    if (do_clean)
    {
        const char *clean_argv[] = {
            "cmake", "--build", build_dir, "--target", "clean", NULL
        };
        int clean_exit = 0;
        mcp_exec_capture(clean_argv, NULL, 0, 30000, &clean_exit);
    }

    /* 2. Configure step: cmake -B <build_dir> -DCMAKE_C_FLAGS="..." */
    char cflags_arg[512];
    char ldflags_arg[512];
    snprintf(cflags_arg, sizeof(cflags_arg), "-DCMAKE_C_FLAGS=%s", c_flags);
    snprintf(ldflags_arg, sizeof(ldflags_arg),
             "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined");

    const char *cfg_argv[] = {
        "cmake", "-B", build_dir, "-S", root, cflags_arg, ldflags_arg, NULL
    };
    int cfg_exit = 0;
    int cfg_ret = mcp_exec_capture(cfg_argv, NULL, 0, 60000, &cfg_exit);
    if (cfg_ret != 0 || cfg_exit != 0)
    {
        cJSON_AddStringToObject(res, "status", "CONFIGURE_FAILED");
        cJSON_AddNumberToObject(res, "exit_code", (double)cfg_exit);
        cJSON_AddStringToObject(res, "error", "CMake configuration with sanitizers failed");
        return 0;
    }

    /* 3. Build step */
    char *build_output = malloc(MAX_SANITIZE_OUTPUT_BYTES);
    if (build_output == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Memory allocation failed for build buffer");
        return -1;
    }
    build_output[0] = '\0';

    const char *build_argv[8];
    int b_idx = 0;
    build_argv[b_idx++] = "cmake";
    build_argv[b_idx++] = "--build";
    build_argv[b_idx++] = build_dir;
    if (target != NULL && strcmp(target, "all") != 0)
    {
        build_argv[b_idx++] = "--target";
        build_argv[b_idx++] = target;
    }
    build_argv[b_idx++] = "-j";
    build_argv[b_idx] = NULL;

    int build_exit = 0;
    int b_ret = mcp_exec_capture(
        build_argv, build_output, MAX_SANITIZE_OUTPUT_BYTES, timeout_ms, &build_exit);

    cJSON *warnings_arr = cJSON_CreateArray();
    cJSON *compile_errs_arr = cJSON_CreateArray();
    mcp_parse_compiler_diagnostics(build_output, warnings_arr, compile_errs_arr);

    cJSON_AddNumberToObject(
        res, "compiler_warnings_count", (double)cJSON_GetArraySize(warnings_arr));
    cJSON_AddItemToObject(res, "compiler_warnings", warnings_arr);
    cJSON_AddNumberToObject(
        res, "compiler_errors_count", (double)cJSON_GetArraySize(compile_errs_arr));
    cJSON_AddItemToObject(res, "compiler_errors", compile_errs_arr);

    free(build_output);

    if (b_ret != 0 || build_exit != 0)
    {
        cJSON_AddStringToObject(res, "status", "BUILD_FAILED");
        cJSON_AddNumberToObject(res, "exit_code", (double)build_exit);
        return 0;
    }

    if (!run_tests)
    {
        cJSON_AddStringToObject(res, "status", "BUILD_SUCCESS");
        cJSON_AddNumberToObject(res, "errors_count", 0.0);
        cJSON_AddItemToObject(res, "errors", cJSON_CreateArray());
        return 0;
    }

    /* 4. Test step with CTest under sanitizers */
    char *ctest_output = malloc(MAX_SANITIZE_OUTPUT_BYTES);
    if (ctest_output == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Memory allocation failed for ctest buffer");
        return -1;
    }
    ctest_output[0] = '\0';

    const char *ctest_argv[8];
    int ct_idx = 0;
    ctest_argv[ct_idx++] = "ctest";
    ctest_argv[ct_idx++] = "--test-dir";
    ctest_argv[ct_idx++] = build_dir;
    ctest_argv[ct_idx++] = "--output-on-failure";
    if (test_filter != NULL)
    {
        ctest_argv[ct_idx++] = "-R";
        ctest_argv[ct_idx++] = test_filter;
    }
    ctest_argv[ct_idx] = NULL;

    int ctest_exit = 0;
    mcp_exec_capture(
        ctest_argv, ctest_output, MAX_SANITIZE_OUTPUT_BYTES, timeout_ms, &ctest_exit);

    cJSON *sanitizer_errors = cJSON_CreateArray();
    mcp_parse_sanitizer_output(ctest_output, sanitizer_errors);
    int san_count = cJSON_GetArraySize(sanitizer_errors);

    cJSON_AddStringToObject(
        res, "status", (san_count == 0 && ctest_exit == 0) ? "CLEAN" :
                       (san_count > 0) ? "SANITIZER_ERRORS_DETECTED" : "TEST_FAILED");
    cJSON_AddNumberToObject(res, "errors_count", (double)san_count);
    cJSON_AddItemToObject(res, "errors", sanitizer_errors);
    cJSON_AddNumberToObject(res, "ctest_exit_code", (double)ctest_exit);

    char summary[512];
    if (san_count == 0 && ctest_exit == 0)
    {
        snprintf(summary, sizeof(summary),
                 "Sanitizer audit PASSED: Zero memory or undefined behavior errors detected.");
    }
    else
    {
        snprintf(summary, sizeof(summary),
                 "Sanitizer audit DETECTED %d issue(s) under %s (exit code %d).",
                 san_count, build_dir_name, ctest_exit);
    }
    cJSON_AddStringToObject(res, "summary", summary);

    free(ctest_output);
    return 0;
} // mcp_tool_dev_sanitize

const struct mcp_tool_def mcp_tooldef_dev_sanitize = {
    .name         = "gric_dev_sanitize",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_dev_sanitize,
    .description  = "Compile and execute test suites under AddressSanitizer and UndefinedBehavior"
                    "Sanitizer, parsing stack traces, bounds errors, and memory leaks.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"build_dir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Isolated build directory (default: 'build-asan').\"\n"
        "    },\n"
        "    \"target\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"CMake build target (default: 'all').\"\n"
        "    },\n"
        "    \"test_filter\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"CTest regex filter pattern (-R).\"\n"
        "    },\n"
        "    \"c_flags\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Compiler sanitization flags.\"\n"
        "    },\n"
        "    \"clean\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Perform clean before compilation (default: false).\"\n"
        "    },\n"
        "    \"run_tests\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Whether to execute CTest after build (default: true).\"\n"
        "    },\n"
        "    \"timeout_ms\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Process timeout in milliseconds (default: 300000).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

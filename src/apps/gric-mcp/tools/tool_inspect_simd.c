/**
 * @file tool_inspect_simd.c
 * @brief Machine code and SIMD vectorization auditor compiling C kernels to assembly.
 */

#include "mcp_tools.h"
#include "mcp_exec.h"
#include "mcp_registry.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int mcp_tool_inspect_simd(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *src_item = cJSON_GetObjectItemCaseSensitive(args, "source_file");
    if (src_item == NULL || !cJSON_IsString(src_item))
    {
        cJSON_AddStringToObject(res, "error", "source_file is required");
        return -1;
    }
    const char *source_file = src_item->valuestring;
    char resolved_src[1024];
    mcp_resolve_path(source_file, resolved_src, sizeof(resolved_src));

    if (access(resolved_src, R_OK) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Cannot access source_file");
        cJSON_AddStringToObject(res, "file_path", src_item->valuestring);
        return -1;
    }

    char root[256];
    mcp_get_project_root(root, sizeof(root));

    const char *march = "-march=native";
    cJSON *march_item = cJSON_GetObjectItemCaseSensitive(args, "march");
    if (march_item != NULL && cJSON_IsString(march_item))
    {
        march = march_item->valuestring;
    }

    char asm_output_path[256];
    snprintf(asm_output_path, sizeof(asm_output_path), "/tmp/gric_simd_%d.s", (int)getpid());

    char inc_src[512], inc_shared[512], inc_quant[512], inc_format[512];
    char inc_sys[512], inc_cli[512], inc_core[512], inc_math[512];
    char inc_io[512], inc_knn[512], inc_knn_core[512], inc_knn_search[512], inc_knn_cache[512];

    snprintf(inc_src, sizeof(inc_src), "-I%s/src", root);
    snprintf(inc_shared, sizeof(inc_shared), "-I%s/src/shared", root);
    snprintf(inc_quant, sizeof(inc_quant), "-I%s/src/shared/quant", root);
    snprintf(inc_format, sizeof(inc_format), "-I%s/src/shared/format", root);
    snprintf(inc_sys, sizeof(inc_sys), "-I%s/src/shared/sys", root);
    snprintf(inc_cli, sizeof(inc_cli), "-I%s/src/shared/cli", root);
    snprintf(inc_core, sizeof(inc_core), "-I%s/src/gric-cluster/core", root);
    snprintf(inc_math, sizeof(inc_math), "-I%s/src/gric-cluster/math", root);
    snprintf(inc_io, sizeof(inc_io), "-I%s/src/gric-cluster/io", root);
    snprintf(inc_knn, sizeof(inc_knn), "-I%s/src/gric-knn", root);
    snprintf(inc_knn_core, sizeof(inc_knn_core), "-I%s/src/gric-knn/core", root);
    snprintf(inc_knn_search, sizeof(inc_knn_search), "-I%s/src/gric-knn/search", root);
    snprintf(inc_knn_cache, sizeof(inc_knn_cache), "-I%s/src/gric-knn/cache", root);

    const char *const gcc_argv[] = {
        "gcc", "-S", "-O3", march, "-funroll-loops", "-fverbose-asm",
        inc_src, inc_shared, inc_quant, inc_format, inc_sys, inc_cli,
        inc_core, inc_math, inc_io, inc_knn, inc_knn_core, inc_knn_search, inc_knn_cache,
        resolved_src, "-o", asm_output_path, NULL
    };

    char err_buf[4096] = "";
    int exit_status = 0;
    int exec_ret = mcp_exec_capture(gcc_argv, err_buf, sizeof(err_buf), 15000, &exit_status);
    if (exec_ret != 0 || exit_status != 0)
    {
        cJSON_AddStringToObject(res, "error", "Compilation to assembly failed");
        cJSON_AddStringToObject(res, "compiler_output", err_buf);
        return -1;
    }

    FILE *afp = fopen(asm_output_path, "r");
    if (afp == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Unable to read generated assembly output");
        return -1;
    }

    int zmm_count = 0;
    int ymm_count = 0;
    int xmm_count = 0;
    int fma_count = 0;
    int vec_arith_count = 0;
    int vec_mem_count = 0;
    int stack_access_count = 0;

    char line[1024];
    while (fgets(line, sizeof(line), afp) != NULL)
    {
        /* Skip assembly labels and comments */
        char *p = line;
        while (*p && isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '#' || *p == '.' || *p == '\0')
        {
            continue;
        }

        if (strstr(p, "%zmm") != NULL)
        {
            zmm_count++;
        }
        if (strstr(p, "%ymm") != NULL)
        {
            ymm_count++;
        }
        if (strstr(p, "%xmm") != NULL)
        {
            xmm_count++;
        }

        if (strstr(p, "vfmadd") != NULL || strstr(p, "vfmsub") != NULL ||
            strstr(p, "vfnmadd") != NULL)
        {
            fma_count++;
        }

        if (strstr(p, "vaddp") != NULL || strstr(p, "vmulp") != NULL ||
            strstr(p, "vsubp") != NULL || strstr(p, "vsqrtp") != NULL)
        {
            vec_arith_count++;
        }

        if (strstr(p, "vmovup") != NULL || strstr(p, "vmovap") != NULL ||
            strstr(p, "vbroadcast") != NULL)
        {
            vec_mem_count++;
        }

        if (strstr(p, "(%rsp)") != NULL || strstr(p, "(%rbp)") != NULL)
        {
            stack_access_count++;
        }
    } // while fgets

    fclose(afp);
    unlink(asm_output_path);

    int is_vectorized = (zmm_count > 0 || ymm_count > 0);
    const char *isa = "SCALAR";
    if (zmm_count > 0)
    {
        isa = "AVX-512";
    }
    else if (ymm_count > 0)
    {
        isa = "AVX2";
    }
    else if (xmm_count > 0)
    {
        isa = "SSE";
    }

    cJSON_AddStringToObject(res, "source_file", source_file);
    cJSON_AddStringToObject(res, "target_architecture", march);
    cJSON_AddBoolToObject(res, "is_vectorized", is_vectorized);
    cJSON_AddStringToObject(res, "dominant_isa", isa);
    cJSON_AddNumberToObject(res, "avx512_zmm_registers", zmm_count);
    cJSON_AddNumberToObject(res, "avx2_ymm_registers", ymm_count);
    cJSON_AddNumberToObject(res, "sse_xmm_registers", xmm_count);
    cJSON_AddNumberToObject(res, "fma_instructions", fma_count);
    cJSON_AddNumberToObject(res, "vector_arithmetic_instructions", vec_arith_count);
    cJSON_AddNumberToObject(res, "vector_memory_instructions", vec_mem_count);
    cJSON_AddNumberToObject(res, "stack_access_instructions", stack_access_count);

    char summary[512];
    if (zmm_count > 0)
    {
        snprintf(
            summary, sizeof(summary),
            "AVX-512 auto-vectorized successfully (%d zmm ops, %d FMAs, %d stack spills)",
            zmm_count, fma_count, stack_access_count);
    }
    else if (ymm_count > 0)
    {
        snprintf(
            summary, sizeof(summary),
            "AVX2 auto-vectorized successfully (%d ymm ops, %d FMAs, %d stack spills)",
            ymm_count, fma_count, stack_access_count);
    }
    else
    {
        snprintf(
            summary, sizeof(summary),
            "Warning: Code did not vectorize into AVX/AVX-512 (scalar or SSE fallback)");
    }
    cJSON_AddStringToObject(res, "summary", summary);

    return 0;
} // mcp_tool_inspect_simd

const struct mcp_tool_def mcp_tooldef_inspect_simd = {
    .name         = "gric_inspect_simd",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_inspect_simd,
    .description  = "Compile a C source file or math kernel to assembly and inspect "
                    "AVX2/AVX-512 SIMD vectorization, FMA ops, and stack spills.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"source_file\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to C source file containing the kernel.\"\n"
        "    },\n"
        "    \"function_name\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Optional target function name to isolate in assembly.\"\n"
        "    },\n"
        "    \"march\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Target architecture flag (default: -march=native).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"source_file\"]\n"
        "}",
};

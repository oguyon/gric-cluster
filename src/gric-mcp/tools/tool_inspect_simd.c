/**
 * @file tool_inspect_simd.c
 * @brief Machine code and SIMD vectorization auditor compiling C kernels to assembly.
 */

#include "mcp_tools.h"
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

    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    const char *march = "-march=native";
    cJSON *march_item = cJSON_GetObjectItemCaseSensitive(args, "march");
    if (march_item != NULL && cJSON_IsString(march_item))
    {
        march = march_item->valuestring;
    }

    char asm_output_path[256];
    snprintf(asm_output_path, sizeof(asm_output_path), "/tmp/gric_simd_%d.s", getpid());

    char compile_cmd[8192];
    snprintf(
        compile_cmd, sizeof(compile_cmd),
        "gcc -S -O3 %s -funroll-loops -fverbose-asm -I%s/src -I%s/src/shared "
        "-I%s/src/shared/quant -I%s/src/shared/format -I%s/src/shared/sys -I%s/src/shared/cli "
        "-I%s/src/gric-cluster/core -I%s/src/gric-cluster/math -I%s/src/gric-cluster/io "
        "-I%s/src/gric-knn -I%s/src/gric-knn/core -I%s/src/gric-knn/search -I%s/src/gric-knn/cache "
        "%s -o %s 2>&1",
        march, root, root, root, root, root, root, root, root, root, root, root, root, root,
        resolved_src, asm_output_path);

    FILE *pipe = popen(compile_cmd, "r");
    if (pipe == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Failed to spawn gcc for assembly inspection");
        return -1;
    }

    char err_buf[2048] = "";
    size_t err_read = fread(err_buf, 1, sizeof(err_buf) - 1, pipe);
    err_buf[err_read] = '\0';
    int ret = pclose(pipe);

    if (ret != 0)
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

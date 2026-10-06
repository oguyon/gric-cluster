/**
 * @file tool_dev_layering.c
 * @brief Architectural layering and dependency hierarchy linter (gric_dev_check_layering).
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "shared/cjson/cJSON.h"
#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_LAYERING_VIOLATIONS 256

struct layering_violation
{
    char file_path[1024];
    int  line_num;
    char include_str[256];
    int  source_level;
    int  target_level;
    char reason[256];
};

struct layering_context
{
    int                       files_scanned;
    int                       violation_count;
    struct layering_violation violations[MAX_LAYERING_VIOLATIONS];
};

/**
 * get_repo_relpath() - Extract path relative to src/ or repository root.
 * @path: Absolute or relative source file path.
 *
 * Return: Pointer to subpath within path.
 */
static const char *get_repo_relpath(
    const char *path)
{
    const char *last_src = NULL;
    const char *curr = path;
    while ((curr = strstr(curr, "/src/")) != NULL)
    {
        last_src = curr + 5;
        curr = curr + 5;
    }
    if (last_src != NULL)
    {
        return last_src;
    }
    if (strncmp(path, "src/", 4) == 0)
    {
        return path + 4;
    }
    return path;
} // get_repo_relpath

/**
 * determine_source_level() - Classify source file path into architectural tier (0..3).
 * @path: Absolute or relative source file path.
 *
 * Return: Tier level (0=shared, 1=gpu, 2=engines, 3=apps/tools).
 */
static int determine_source_level(
    const char *path)
{
    const char *rel = get_repo_relpath(path);

    if (strncmp(rel, "base/", 5) == 0 ||
        strncmp(rel, "quant/", 6) == 0 ||
        strncmp(rel, "third_party/", 12) == 0 ||
        strncmp(rel, "shared/", 7) == 0 ||
        strstr(rel, "/base/") != NULL ||
        strstr(rel, "/quant/") != NULL ||
        strstr(rel, "/third_party/") != NULL ||
        strstr(rel, "/shared/") != NULL)
    {
        return 0;
    }
    if (strncmp(rel, "engine/", 7) == 0 ||
        strncmp(rel, "gric-cluster/", 13) == 0 ||
        strncmp(rel, "gric-knn/", 9) == 0 ||
        strstr(rel, "/engine/") != NULL ||
        strstr(rel, "/gric-cluster/") != NULL ||
        strstr(rel, "/gric-knn/") != NULL)
    {
        return 2;
    }
    if (strncmp(rel, "accel/", 6) == 0 ||
        strncmp(rel, "gpu/", 4) == 0 ||
        strstr(rel, "/accel/") != NULL ||
        strstr(rel, "/gpu/") != NULL)
    {
        return 1;
    }
    return 3;
} // determine_source_level

/**
 * determine_target_level() - Classify an included header into architectural tier.
 * @inc: Header filename or relative path inside #include.
 *
 * Return: Tier level (0..3) if recognized, or -1 if standard library/external.
 */
static int determine_target_level(
    const char *inc)
{
    if (strncmp(inc, "base/", 5) == 0 ||
        strncmp(inc, "quant/", 6) == 0 ||
        strncmp(inc, "third_party/", 12) == 0 ||
        strncmp(inc, "shared/", 7) == 0 ||
        strcmp(inc, "gric_simd.h") == 0 ||
        strcmp(inc, "scalar_quant.h") == 0 ||
        strcmp(inc, "residual_quant.h") == 0 ||
        strcmp(inc, "eq16_quant.h") == 0 ||
        strcmp(inc, "rabit_quant.h") == 0 ||
        strcmp(inc, "product_quant.h") == 0 ||
        strcmp(inc, "e8_lattice.h") == 0 ||
        strcmp(inc, "quant_memo.h") == 0 ||
        strcmp(inc, "gric_bin_header.h") == 0 ||
        strcmp(inc, "gric_bin_io.h") == 0 ||
        strcmp(inc, "cluster_locator.h") == 0 ||
        strcmp(inc, "cli_colors.h") == 0 ||
        strcmp(inc, "cli_table.h") == 0 ||
        strcmp(inc, "gric_mem.h") == 0 ||
        strcmp(inc, "gric_omp.h") == 0 ||
        strcmp(inc, "gric_stream_layout.h") == 0 ||
        strcmp(inc, "cJSON.h") == 0)
    {
        return 0;
    }

    if (strncmp(inc, "accel/", 6) == 0 ||
        strncmp(inc, "gpu/", 4) == 0 ||
        strcmp(inc, "cuda_common.h") == 0 ||
        strcmp(inc, "cuda_anchor_store.h") == 0 ||
        strcmp(inc, "cuda_ivf_index.h") == 0)
    {
        return 1;
    }

    if (strncmp(inc, "engine/", 7) == 0 ||
        strncmp(inc, "gric-cluster/", 13) == 0 ||
        strncmp(inc, "gric-knn/", 9) == 0 ||
        strcmp(inc, "run_clustering.h") == 0 ||
        strcmp(inc, "knn_engine.h") == 0 ||
        strcmp(inc, "knn_cache.h") == 0 ||
        strcmp(inc, "knn_heap.h") == 0 ||
        strcmp(inc, "knn_pruning.h") == 0 ||
        strcmp(inc, "knn_types.h") == 0 ||
        strcmp(inc, "cluster_types.h") == 0 ||
        strcmp(inc, "framedist.h") == 0)
    {
        return 2;
    }

    if (strncmp(inc, "ui/", 3) == 0 ||
        strncmp(inc, "apps/", 5) == 0 ||
        strncmp(inc, "adapters/", 9) == 0 ||
        strncmp(inc, "gric-mcp/", 9) == 0 ||
        strncmp(inc, "gric-fps/", 9) == 0 ||
        strncmp(inc, "gric-server/", 12) == 0 ||
        strncmp(inc, "tools/", 6) == 0 ||
        strncmp(inc, "mcp_", 4) == 0 ||
        strcmp(inc, "gric_fps_params.h") == 0)
    {
        return 3;
    }

    return -1;
} // determine_target_level

/**
 * check_file_layering() - Scan a single C or header file for architectural violations.
 * @file_path: Path to source file.
 * @ctx:       Linter context accumulating findings.
 */
static void check_file_layering(
    const char              *file_path,
    struct layering_context *ctx)
{
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL)
    {
        return;
    }

    ctx->files_scanned++;
    int source_level = determine_source_level(file_path);
    int is_l2_core = (source_level == 2 &&
                      strstr(file_path, "/io/") == NULL &&
                      strstr(file_path, "/help/") == NULL);

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        line_num++;
        char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p != '#')
        {
            continue;
        }
        p++;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (strncmp(p, "include", 7) != 0)
        {
            continue;
        }
        p += 7;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }

        char delim_close = '\0';
        if (*p == '"')
        {
            delim_close = '"';
        }
        else if (*p == '<')
        {
            delim_close = '>';
        }
        else
        {
            continue;
        }
        p++;

        char inc_buf[256];
        size_t inc_len = 0;
        while (*p != '\0' && *p != delim_close && inc_len < sizeof(inc_buf) - 1)
        {
            inc_buf[inc_len++] = *p++;
        }
        inc_buf[inc_len] = '\0';

        int target_level = determine_target_level(inc_buf);
        int is_milk_header = (strcmp(inc_buf, "fps.h") == 0 ||
                              strstr(inc_buf, "ImageStreamIO") != NULL);

        int has_violation = 0;
        char reason[256] = {0};

        /* Rule 1: Level 0 cannot include Level 1, 2, or 3 */
        if (source_level == 0 && target_level > 0)
        {
            has_violation = 1;
            snprintf(reason, sizeof(reason),
                     "Level 0 (src/shared/) cannot depend on higher Tier %d", target_level);
        }
        /* Rule 2: Level 1 cannot include Level 2 or 3 */
        else if (source_level == 1 && target_level > 1)
        {
            has_violation = 1;
            snprintf(reason, sizeof(reason),
                     "Level 1 (src/gpu/) cannot depend on higher Tier %d", target_level);
        }
        /* Rule 3: Level 2 cannot include Level 3 */
        else if (source_level == 2 && target_level == 3)
        {
            has_violation = 1;
            snprintf(reason, sizeof(reason),
                     "Level 2 domain engines cannot depend on Level 3 tools or server");
        }
        /* Rule 4: Level 2 core compute cannot include Milk/ImageStreamIO directly */
        else if (is_l2_core && is_milk_header)
        {
            has_violation = 1;
            snprintf(reason, sizeof(reason),
                     "Level 2 core compute cannot include external Milk/ImageStreamIO headers");
        }

        if (has_violation && ctx->violation_count < MAX_LAYERING_VIOLATIONS)
        {
            struct layering_violation *v = &ctx->violations[ctx->violation_count++];
            snprintf(v->file_path, sizeof(v->file_path), "%s", file_path);
            v->line_num = line_num;
            snprintf(v->include_str, sizeof(v->include_str), "%s", inc_buf);
            v->source_level = source_level;
            v->target_level = target_level;
            snprintf(v->reason, sizeof(v->reason), "%s", reason);
        }
    } // while reading lines

    fclose(fp);
} // check_file_layering

/**
 * scan_directory_recursive() - Walk directory tree and lint all C/H files.
 * @dir_path: Directory path to scan.
 * @ctx:      Linter context accumulating findings.
 */
static void scan_directory_recursive(
    const char              *dir_path,
    struct layering_context *ctx)
{
    DIR *d = opendir(dir_path);
    if (d == NULL)
    {
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }

        char sub_path[1024];
        snprintf(sub_path, sizeof(sub_path), "%s/%s", dir_path, entry->d_name);

        struct stat st;
        if (stat(sub_path, &st) != 0)
        {
            continue;
        }

        if (S_ISDIR(st.st_mode))
        {
            /* Skip build artifacts, git repository, and documentation folders */
            if (strcmp(entry->d_name, "build") != 0 &&
                strcmp(entry->d_name, ".git") != 0 &&
                strcmp(entry->d_name, "docs") != 0 &&
                strcmp(entry->d_name, ".agents") != 0)
            {
                scan_directory_recursive(sub_path, ctx);
            }
        }
        else if (S_ISREG(st.st_mode))
        {
            size_t len = strlen(entry->d_name);
            if ((len > 2 && strcmp(entry->d_name + len - 2, ".c") == 0) ||
                (len > 2 && strcmp(entry->d_name + len - 2, ".h") == 0) ||
                (len > 3 && strcmp(entry->d_name + len - 3, ".cu") == 0))
            {
                check_file_layering(sub_path, ctx);
            }
        }
    } // while readdir

    closedir(d);
} // scan_directory_recursive

int mcp_tool_dev_check_layering(
    const cJSON *args,
    cJSON       *res)
{
    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    const char *sub_dir = "src";
    if (args != NULL)
    {
        cJSON *d_item = cJSON_GetObjectItemCaseSensitive(args, "src_dir");
        if (d_item != NULL && cJSON_IsString(d_item) && d_item->valuestring[0] != '\0')
        {
            sub_dir = d_item->valuestring;
        }
    }

    char target_path[2048];
    if (sub_dir[0] == '/')
    {
        snprintf(target_path, sizeof(target_path), "%s", sub_dir);
    }
    else
    {
        snprintf(target_path, sizeof(target_path), "%s/%s", root, sub_dir);
    }

    struct stat st;
    if (stat(target_path, &st) != 0 || !S_ISDIR(st.st_mode))
    {
        cJSON_AddStringToObject(
            res, "error", "Target directory does not exist or is not a directory");
        return -1;
    }

    struct layering_context ctx;
    memset(&ctx, 0, sizeof(ctx));

    scan_directory_recursive(target_path, &ctx);

    cJSON_AddStringToObject(
        res, "status", (ctx.violation_count == 0) ? "PASS" : "VIOLATIONS_FOUND");
    cJSON_AddNumberToObject(res, "files_scanned", (double)ctx.files_scanned);
    cJSON_AddNumberToObject(res, "violations_count", (double)ctx.violation_count);

    cJSON *viols_arr = cJSON_CreateArray();
    for (int ii = 0; ii < ctx.violation_count; ii++)
    {
        const struct layering_violation *v = &ctx.violations[ii];
        cJSON *v_obj = cJSON_CreateObject();
        cJSON_AddStringToObject(v_obj, "source_file", v->file_path);
        cJSON_AddNumberToObject(v_obj, "line_number", (double)v->line_num);
        cJSON_AddStringToObject(v_obj, "include_directive", v->include_str);
        cJSON_AddNumberToObject(v_obj, "source_layer", (double)v->source_level);
        cJSON_AddNumberToObject(v_obj, "target_layer", (double)v->target_level);
        cJSON_AddStringToObject(v_obj, "reason", v->reason);
        cJSON_AddItemToArray(viols_arr, v_obj);
    }
    cJSON_AddItemToObject(res, "violations", viols_arr);

    char summary[512];
    if (ctx.violation_count == 0)
    {
        snprintf(summary, sizeof(summary),
                 "Layering check PASSED: %d files scanned in '%s' "
                 "with zero architectural violations.",
                 ctx.files_scanned, sub_dir);
    }
    else
    {
        snprintf(summary, sizeof(summary),
                 "Layering check FAILED: %d architectural include violation(s) detected across "
                 "%d files scanned in '%s'.",
                 ctx.violation_count, ctx.files_scanned, sub_dir);
    }
    cJSON_AddStringToObject(res, "summary", summary);

    return 0;
} // mcp_tool_dev_check_layering

const struct mcp_tool_def mcp_tooldef_dev_check_layering = {
    .name         = "gric_dev_check_layering",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_dev_check_layering,
    .description  = "Verify architectural module hierarchy and detect forbidden cross-tier "
                    "includes (Level 0 shared, Level 1 GPU, Level 2 engines, Level 3 tools).",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"src_dir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Source directory path to scan (default: 'src').\"\n"
        "    },\n"
        "    \"strict\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Whether warnings are treated as errors (default: true).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

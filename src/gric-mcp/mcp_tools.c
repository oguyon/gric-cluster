/**
 * @file mcp_tools.c
 * @brief Schema declarations and registry for GRIC Model Context Protocol tools.
 */

#include "mcp_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * create_tool_schema() - Helper to construct an MCP tool descriptor with JSON schema.
 * @name:        Name of the tool.
 * @description: Human-readable description for the agent.
 * @schema_json: JSON string defining inputSchema object.
 *
 * Return: Newly allocated cJSON object representing the tool.
 */
static cJSON *create_tool_schema(
    const char *name,
    const char *description,
    const char *schema_json)
{
    cJSON *tool = cJSON_CreateObject();
    if (tool == NULL)
    {
        return NULL;
    }

    cJSON_AddStringToObject(tool, "name", name);
    cJSON_AddStringToObject(tool, "description", description);

    cJSON *schema = cJSON_Parse(schema_json);
    if (schema != NULL)
    {
        cJSON_AddItemToObject(tool, "inputSchema", schema);
    }
    else
    {
        cJSON *empty = cJSON_CreateObject();
        cJSON_AddStringToObject(empty, "type", "object");
        cJSON_AddItemToObject(tool, "inputSchema", empty);
    }

    return tool;
} // create_tool_schema

cJSON *mcp_tools_get_list(void)
{
    cJSON *list = cJSON_CreateArray();
    if (list == NULL)
    {
        return NULL;
    }

    /* 1. gric_audit_code_style */
    {
        const char *desc = "Audit C source files against GRIC coding standards (100-char line "
                           "limit, Allman braces, Kernel-Doc, loop types, and malloc checks).";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"file_path\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Path to C source or header file to audit.\""
            "    }"
            "  },"
            "  \"required\": [\"file_path\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_audit_code_style", desc, schema));
    }

    /* 2. gric_align_parameters */
    {
        const char *desc = "Check or compute column-aligned parameter names for multi-line "
                           "function prototypes per parameter-alignment.md.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"prototype\": {"
            "      \"type\": \"string\","
            "      \"description\": \"C function prototype or definition snippet to align.\""
            "    }"
            "  },"
            "  \"required\": [\"prototype\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_align_parameters", desc, schema));
    }

    /* 3. gric_inspect_simd */
    {
        const char *desc = "Compile a C source file or math kernel to assembly and inspect "
                           "AVX2/AVX-512 SIMD vectorization, FMA ops, and stack spills.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"source_file\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Path to C source file containing the kernel.\""
            "    },"
            "    \"function_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Optional target function name to isolate in assembly.\""
            "    },"
            "    \"march\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Target architecture flag (default: -march=native).\""
            "    }"
            "  },"
            "  \"required\": [\"source_file\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_inspect_simd", desc, schema));
    }

    /* 4. gric_verify_invariants */
    {
        const char *desc = "Verify mathematical clustering guarantees: asserts d(frame, anchor) <= "
                           "rlim across all frames with SIMD. Reports violations and max error.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"run_dir\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Clustering output directory (e.g., 3Dspiral.clusterdat).\""
            "    },"
            "    \"dataset\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Path to original dataset frames used during clustering.\""
            "    },"
            "    \"rlim\": {"
            "      \"type\": \"number\","
            "      \"description\": \"Clustering radius threshold. If omitted, parsed from log.\""
            "    }"
            "  },"
            "  \"required\": [\"run_dir\", \"dataset\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_verify_invariants", desc, schema));
    }

    /* 5. gric_inspect_run */
    {
        const char *desc = "Inspect clustering run output (.clusterdat): extracts cluster counts, "
                           "dists/frame, pruning efficiency, entropy, and execution time.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"run_dir\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Path to clustering results directory.\""
            "    }"
            "  },"
            "  \"required\": [\"run_dir\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_inspect_run", desc, schema));
    }

    /* 6. gric_probe_dataset */
    {
        const char *desc = "Analyze dataset geometry: estimates local intrinsic dimensionality, "
                           "noise floor, recommended radii (fine/balanced), and pruning flags.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"dataset_path\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Path to dataset coordinates or image cube.\""
            "    },"
            "    \"sample_limit\": {"
            "      \"type\": \"integer\","
            "      \"description\": \"Maximum sample count for adaptive probing (0 = default).\""
            "    }"
            "  },"
            "  \"required\": [\"dataset_path\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_probe_dataset", desc, schema));
    }

    /* 7. gric_probe_shm */
    {
        const char *desc = "Non-blocking probe of ImageStreamIO shared-memory stream: checks "
                           "dimensions, write index, frame count, and semaphore status.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"stream_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Name of shared memory stream (e.g. imrec1).\""
            "    }"
            "  },"
            "  \"required\": [\"stream_name\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_probe_shm", desc, schema));
    }

    /* 8. gric_help */
    {
        const char *desc = "Query GRIC embedded help database (95+ topics) by keyword or fuzzy "
                           "search: algorithms, parameters (rlim, entropy, tiling, milk), flags.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"topic\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Topic keyword or flag name (e.g. 'entropy', 'rlim').\""
            "    },"
            "    \"query\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Optional search query to match in documentation.\""
            "    }"
            "  }"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_help", desc, schema));
    }

    /* 9. gric_list_suite */
    {
        const char *desc = "List all 18+ programs in the GRIC suite with categories, summaries, "
                           "and CLI usage prototypes.";
        const char *schema = "{\"type\": \"object\"}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_list_suite", desc, schema));
    }

    /* 10. gric_get_recipe */
    {
        const char *desc = "Retrieve step-by-step cookbook instructions for common tasks (e.g. "
                           "'milk_realtime_streaming', 'image_cube_clustering').";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"recipe\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Name of recipe, or 'list' to see available recipes.\""
            "    }"
            "  }"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_get_recipe", desc, schema));
    }

    /* 11. gric_fps_status */
    {
        const char *desc = "Inspect live status, loop rate, PID, and current parameters of a Milk "
                           "FPS streaming instance (e.g. gric_cluster).";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"fps_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"FPS instance name (default: gric_cluster).\""
            "    }"
            "  }"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_fps_status", desc, schema));
    }

    /* 12. gric_fps_run */
    {
        const char *desc = "Launch standalone milk-fpsexec-gric-cluster streaming daemon "
                           "with input/output streams and initial parameters.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"in_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Input ImageStreamIO stream name (TRIGGER).\""
            "    },"
            "    \"out_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Output assignment stream name (<out>_assign).\""
            "    },"
            "    \"fps_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"FPS instance name (default: gric_cluster).\""
            "    },"
            "    \"rlim\": {"
            "      \"type\": \"number\","
            "      \"description\": \"Cluster radius threshold (default: 0.5).\""
            "    },"
            "    \"allow_frame_drop\": {"
            "      \"type\": \"boolean\","
            "      \"description\": \"Lag policy: true=jump to latest frame, false=sequential.\""
            "    },"
            "    \"use_tmux\": {"
            "      \"type\": \"boolean\","
            "      \"description\": \"Run inside an isolated tmux session (default: true).\""
            "    }"
            "  },"
            "  \"required\": [\"in_name\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_fps_run", desc, schema));
    }

    /* 13. gric_fps_set */
    {
        const char *desc = "Dynamically update an FPS parameter (e.g. .rlim, .allow_frame_drop, "
                           ".reset_state) on a live streaming instance using milk-fps-set.";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"fps_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"FPS instance name (e.g. gric_cluster).\""
            "    },"
            "    \"param\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Parameter key (e.g. 'rlim', 'allow_frame_drop').\""
            "    },"
            "    \"value\": {"
            "      \"description\": \"New parameter value (number, boolean ON/OFF, or string).\""
            "    }"
            "  },"
            "  \"required\": [\"fps_name\", \"param\", \"value\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_fps_set", desc, schema));
    }

    /* 14. gric_fps_stop */
    {
        const char *desc = "Gracefully stop an active Milk FPS instance (dispatching runstop and "
                           "confstop to tmux ctrl window).";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"fps_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"FPS instance name (default: gric_cluster).\""
            "    }"
            "  }"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_fps_stop", desc, schema));
    }

    /* 15. gric_probe_fps_streams */
    {
        const char *desc = "Inspect Milk live streams non-blockingly and decode the 8-element "
                           "assignment telemetry vector (cluster ID, distance, latency, evals).";
        const char *schema =
            "{"
            "  \"type\": \"object\","
            "  \"properties\": {"
            "    \"out_name\": {"
            "      \"type\": \"string\","
            "      \"description\": \"Name of output stream (e.g. gric_cluster_assign).\""
            "    }"
            "  },"
            "  \"required\": [\"out_name\"]"
            "}";
        cJSON_AddItemToArray(list, create_tool_schema("gric_probe_fps_streams", desc, schema));
    }

    return list;
} // mcp_tools_get_list

void mcp_get_project_root(
    char   *buf,
    size_t  size)
{
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0)
    {
        exe_path[len] = '\0';
        char *build_pos = strstr(exe_path, "/build/");
        if (build_pos != NULL)
        {
            *build_pos = '\0';
            strncpy(buf, exe_path, size - 1);
            buf[size - 1] = '\0';
            return;
        }
    }
    strncpy(buf, ".", size - 1);
    buf[size - 1] = '\0';
} // mcp_get_project_root

void mcp_resolve_path(
    const char *input_path,
    char       *out_buf,
    size_t      out_size)
{
    if (input_path == NULL || input_path[0] == '\0')
    {
        out_buf[0] = '\0';
        return;
    }

    /* Check if already accessible directly */
    if (access(input_path, R_OK) == 0)
    {
        strncpy(out_buf, input_path, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* If relative, try prepending project root */
    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char candidate[2048];
    snprintf(candidate, sizeof(candidate), "%s/%s", root, input_path);
    if (access(candidate, R_OK) == 0)
    {
        strncpy(out_buf, candidate, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* Fallback to original */
    strncpy(out_buf, input_path, out_size - 1);
    out_buf[out_size - 1] = '\0';
} // mcp_resolve_path

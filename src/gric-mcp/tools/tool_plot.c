/**
 * @file tool_plot.c
 * @brief Diagnostic visualization tool (gric_plot) for clustering results.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_exec.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * find_log_file() - Locate clustering log file in run directory.
 * @dir_path: Run directory path.
 * @out_path: Buffer to store resolved log file path.
 * @out_size: Buffer capacity.
 *
 * Return: 0 if found, -1 otherwise.
 */
static int find_log_file(
    const char *dir_path,
    char       *out_path,
    size_t      out_size)
{
    const char *candidates[] = {
        "cluster_run.log",
        "gric_cluster.log",
        "cluster.log",
        NULL
    };

    for (int idx = 0; candidates[idx] != NULL; idx++)
    {
        size_t dlen = strlen(dir_path);
        size_t clen = strlen(candidates[idx]);
        if (dlen + clen + 2 >= out_size)
        {
            continue;
        }
        memcpy(out_path, dir_path, dlen);
        out_path[dlen] = '/';
        memcpy(out_path + dlen + 1, candidates[idx], clen + 1);
        if (access(out_path, R_OK) == 0)
        {
            return 0;
        }
    }
    return -1;
} // find_log_file

/**
 * extract_points_from_log() - Extract dataset points file from clustering CMD line in log.
 * @log_path: Path to log file.
 * @out_pts:  Buffer to store extracted points file path.
 * @out_size: Buffer capacity.
 *
 * Return: 0 if extracted, -1 otherwise.
 */
static int extract_points_from_log(
    const char *log_path,
    char       *out_pts,
    size_t      out_size)
{
    FILE *fp = fopen(log_path, "r");
    if (fp == NULL)
    {
        return -1;
    }

    char line[4096];
    int found = -1;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strncmp(line, "CMD: ", 5) == 0)
        {
            /* Format: CMD: <exe> <rlim> <points_file> ... */
            char *p = line + 5;
            char exe[1024];
            char rlim_str[64];
            char pts[1024];
            if (sscanf(p, "%1023s %63s %1023s", exe, rlim_str, pts) == 3)
            {
                strncpy(out_pts, pts, out_size - 1);
                out_pts[out_size - 1] = '\0';
                found = 0;
            }
            break;
        }
    }
    fclose(fp);
    return found;
} // extract_points_from_log

/**
 * locate_plot_binary() - Find gric-plot executable path.
 * @out_exe: Buffer to store executable path.
 * @out_size: Buffer capacity.
 */
static void locate_plot_binary(
    char   *out_exe,
    size_t  out_size)
{
    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    if (root[0] != '\0')
    {
        size_t rlen = strlen(root);
        const char *suffix = "/build/gric-plot";
        size_t slen = strlen(suffix);
        if (rlen + slen < out_size)
        {
            memcpy(out_exe, root, rlen);
            memcpy(out_exe + rlen, suffix, slen + 1);
            if (access(out_exe, X_OK) == 0)
            {
                return;
            }
        }
    }

    if (access("./build/gric-plot", X_OK) == 0)
    {
        strncpy(out_exe, "./build/gric-plot", out_size - 1);
        out_exe[out_size - 1] = '\0';
        return;
    }

    strncpy(out_exe, "gric-plot", out_size - 1);
    out_exe[out_size - 1] = '\0';
} // locate_plot_binary

/**
 * parse_log_summary() - Extract key statistics from clustering log file.
 * @log_path: Path to log file.
 * @frames:   Pointer to store frame count.
 * @clusters: Pointer to store cluster count.
 * @dists:    Pointer to store distance count.
 * @pruned:   Pointer to store pruned count.
 * @rlim:     Pointer to store radius limit.
 */
static void parse_log_summary(
    const char *log_path,
    long       *frames,
    long       *clusters,
    long       *dists,
    long       *pruned,
    double     *rlim)
{
    *frames = 0;
    *clusters = 0;
    *dists = 0;
    *pruned = 0;
    *rlim = 0.0;

    FILE *fp = fopen(log_path, "r");
    if (fp == NULL)
    {
        return;
    }

    char line[4096];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (strncmp(line, "PARAM_RLIM: ", 12) == 0)
        {
            sscanf(line + 12, "%lf", rlim);
        }
        else if (strncmp(line, "STATS_FRAMES: ", 14) == 0)
        {
            *frames = atol(line + 14);
        }
        else if (strncmp(line, "STATS_CLUSTERS: ", 16) == 0)
        {
            *clusters = atol(line + 16);
        }
        else if (strncmp(line, "STATS_DISTS: ", 13) == 0)
        {
            *dists = atol(line + 13);
        }
        else if (strncmp(line, "STATS_PRUNED: ", 14) == 0)
        {
            *pruned = atol(line + 14);
        }
    }
    fclose(fp);
} // parse_log_summary

int mcp_tool_plot(
    const cJSON *args,
    cJSON       *res)
{
    const char *run_dir_arg = NULL;
    const char *points_arg = NULL;
    const char *log_arg = NULL;
    const char *output_arg = NULL;
    const char *format_arg = "png";
    const char *plot_type_arg = "all";
    double font_size = 18.0;

    if (args != NULL)
    {
        const cJSON *rd = cJSON_GetObjectItemCaseSensitive(args, "run_dir");
        if (rd != NULL && cJSON_IsString(rd) && rd->valuestring[0] != '\0')
        {
            run_dir_arg = rd->valuestring;
        }

        const cJSON *pt = cJSON_GetObjectItemCaseSensitive(args, "points_path");
        if (pt != NULL && cJSON_IsString(pt) && pt->valuestring[0] != '\0')
        {
            points_arg = pt->valuestring;
        }

        const cJSON *lg = cJSON_GetObjectItemCaseSensitive(args, "log_path");
        if (lg != NULL && cJSON_IsString(lg) && lg->valuestring[0] != '\0')
        {
            log_arg = lg->valuestring;
        }

        const cJSON *op = cJSON_GetObjectItemCaseSensitive(args, "output_path");
        if (op != NULL && cJSON_IsString(op) && op->valuestring[0] != '\0')
        {
            output_arg = op->valuestring;
        }

        const cJSON *fmt = cJSON_GetObjectItemCaseSensitive(args, "format");
        if (fmt != NULL && cJSON_IsString(fmt) && fmt->valuestring[0] != '\0')
        {
            format_arg = fmt->valuestring;
        }

        const cJSON *ptyp = cJSON_GetObjectItemCaseSensitive(args, "plot_type");
        if (ptyp != NULL && cJSON_IsString(ptyp) && ptyp->valuestring[0] != '\0')
        {
            plot_type_arg = ptyp->valuestring;
        }

        const cJSON *fs = cJSON_GetObjectItemCaseSensitive(args, "font_size");
        if (fs != NULL && cJSON_IsNumber(fs) && fs->valuedouble > 0.0)
        {
            font_size = fs->valuedouble;
        }
    }

    int is_svg = (strcmp(format_arg, "svg") == 0);
    if (!is_svg && strcmp(format_arg, "png") != 0)
    {
        cJSON_AddStringToObject(res, "error", "Unsupported format (must be 'png' or 'svg')");
        return -1;
    }

    char resolved_log[2048] = {0};
    char resolved_run_dir[2048] = {0};

    if (run_dir_arg != NULL)
    {
        mcp_resolve_path(run_dir_arg, resolved_run_dir, sizeof(resolved_run_dir));
        if (log_arg == NULL)
        {
            if (find_log_file(resolved_run_dir, resolved_log, sizeof(resolved_log)) != 0)
            {
                cJSON_AddStringToObject(
                    res, "error",
                    "Could not find cluster_run.log or gric_cluster.log in run_dir");
                return -1;
            }
        }
    }

    if (log_arg != NULL)
    {
        char temp_log[2048];
        mcp_resolve_path(log_arg, temp_log, sizeof(temp_log));
        struct stat st_log;
        if (stat(temp_log, &st_log) == 0 && S_ISDIR(st_log.st_mode))
        {
            strncpy(resolved_run_dir, temp_log, sizeof(resolved_run_dir) - 1);
            if (find_log_file(temp_log, resolved_log, sizeof(resolved_log)) != 0)
            {
                cJSON_AddStringToObject(
                    res, "error",
                    "Provided log_path is a directory but no cluster log was found inside");
                return -1;
            }
        }
        else
        {
            strncpy(resolved_log, temp_log, sizeof(resolved_log) - 1);
            if (resolved_run_dir[0] == '\0')
            {
                char log_dir[2048];
                strncpy(log_dir, temp_log, sizeof(log_dir) - 1);
                char *slash = strrchr(log_dir, '/');
                if (slash != NULL)
                {
                    *slash = '\0';
                    strncpy(resolved_run_dir, log_dir, sizeof(resolved_run_dir) - 1);
                }
            }
        }
    }

    if (resolved_log[0] == '\0' || access(resolved_log, R_OK) != 0)
    {
        cJSON_AddStringToObject(
            res, "error", "Log file not found or not readable; specify valid log_path or run_dir");
        return -1;
    }

    /* Locate points file */
    char resolved_points[2048] = {0};
    if (points_arg != NULL)
    {
        mcp_resolve_path(points_arg, resolved_points, sizeof(resolved_points));
    }
    else
    {
        /* Try extracting points path from log CMD */
        char extracted_pts[1024];
        if (extract_points_from_log(resolved_log, extracted_pts, sizeof(extracted_pts)) == 0)
        {
            mcp_resolve_path(extracted_pts, resolved_points, sizeof(resolved_points));
            if (access(resolved_points, R_OK) != 0 && resolved_run_dir[0] != '\0')
            {
                /* Try relative to run_dir itself */
                char alt_pts[4096];
                snprintf(alt_pts, sizeof(alt_pts), "%s/%s", resolved_run_dir, extracted_pts);
                if (access(alt_pts, R_OK) == 0)
                {
                    strncpy(resolved_points, alt_pts, sizeof(resolved_points) - 1);
                }
                else
                {
                    /* Try relative to parent directory of run_dir */
                    char run_parent[2048];
                    strncpy(run_parent, resolved_run_dir, sizeof(run_parent) - 1);
                    char *slash = strrchr(run_parent, '/');
                    if (slash != NULL)
                    {
                        *slash = '\0';
                        snprintf(alt_pts, sizeof(alt_pts), "%s/%s", run_parent, extracted_pts);
                        if (access(alt_pts, R_OK) == 0)
                        {
                            strncpy(resolved_points, alt_pts, sizeof(resolved_points) - 1);
                        }
                    }
                }
            }
        }

        /* Fallback: look for points.txt or input_points.txt in run_dir */
        if ((resolved_points[0] == '\0' || access(resolved_points, R_OK) != 0) &&
            resolved_run_dir[0] != '\0')
        {
            char def_pts[4096];
            snprintf(def_pts, sizeof(def_pts), "%s/points.txt", resolved_run_dir);
            if (access(def_pts, R_OK) == 0)
            {
                strncpy(resolved_points, def_pts, sizeof(resolved_points) - 1);
            }
            else
            {
                snprintf(def_pts, sizeof(def_pts), "%s/input_points.txt", resolved_run_dir);
                if (access(def_pts, R_OK) == 0)
                {
                    strncpy(resolved_points, def_pts, sizeof(resolved_points) - 1);
                }
            }
        }
    }

    if (resolved_points[0] == '\0' || access(resolved_points, R_OK) != 0)
    {
        cJSON_AddStringToObject(
            res, "error",
            "Points dataset file not found or not readable; specify valid points_path");
        return -1;
    }

    /* Determine output image path */
    char resolved_output[2048] = {0};
    if (output_arg != NULL)
    {
        strncpy(resolved_output, output_arg, sizeof(resolved_output) - 1);
    }
    else
    {
        const char *ext = is_svg ? "svg" : "png";
        if (resolved_run_dir[0] != '\0')
        {
            snprintf(resolved_output, sizeof(resolved_output),
                     "%s/cluster_plot.%s", resolved_run_dir, ext);
        }
        else
        {
            snprintf(resolved_output, sizeof(resolved_output),
                     "%s_plot.%s", resolved_points, ext);
        }
    }

    /* Locate gric-plot binary */
    char plot_exe[1024];
    locate_plot_binary(plot_exe, sizeof(plot_exe));

    char fs_str[32];
    snprintf(fs_str, sizeof(fs_str), "%.1f", font_size);

    /* Build command line arguments */
    const char *argv[10];
    int argc = 0;
    argv[argc++] = plot_exe;
    if (is_svg)
    {
        argv[argc++] = "-svg";
    }
    argv[argc++] = "-fs";
    argv[argc++] = fs_str;
    argv[argc++] = resolved_points;
    argv[argc++] = resolved_log;
    argv[argc++] = resolved_output;
    argv[argc++] = NULL;

    char err_buf[8192] = {0};
    int exit_status = 0;
    int exec_ret = mcp_exec_capture(
        argv, err_buf, sizeof(err_buf), 30000, &exit_status);

    if (exec_ret != 0 || exit_status != 0)
    {
        cJSON_AddStringToObject(res, "error", "gric-plot execution failed");
        cJSON_AddNumberToObject(res, "exit_code", exit_status);
        cJSON_AddStringToObject(res, "output", err_buf);
        return -1;
    }

    /* Verify generated output file */
    struct stat st;
    if (stat(resolved_output, &st) != 0)
    {
        cJSON_AddStringToObject(
            res, "error", "gric-plot succeeded but output image was not found on disk");
        return -1;
    }

    /* Determine queries output filename */
    char queries_output[2048];
    strncpy(queries_output, resolved_output, sizeof(queries_output) - 1);
    queries_output[sizeof(queries_output) - 1] = '\0';
    char *ext_dot = strrchr(queries_output, '.');
    if (ext_dot != NULL)
    {
        char ext_saved[32];
        strncpy(ext_saved, ext_dot, sizeof(ext_saved) - 1);
        ext_saved[sizeof(ext_saved) - 1] = '\0';
        strcpy(ext_dot, ".queries");
        strncat(queries_output, ext_saved,
                sizeof(queries_output) - strlen(queries_output) - 1);
    }
    else
    {
        strncat(queries_output, ".queries",
                sizeof(queries_output) - strlen(queries_output) - 1);
    }

    struct stat st_q;
    int has_queries = (stat(queries_output, &st_q) == 0);

    /* Extract summary statistics from log */
    long total_frames = 0;
    long total_clusters = 0;
    long total_dists = 0;
    long total_pruned = 0;
    double rlim = 0.0;
    parse_log_summary(
        resolved_log, &total_frames, &total_clusters, &total_dists, &total_pruned, &rlim);

    /* Construct JSON response */
    cJSON_AddStringToObject(res, "status", "success");
    cJSON_AddStringToObject(res, "output_path", resolved_output);
    cJSON_AddNumberToObject(res, "file_size_bytes", (double)st.st_size);
    cJSON_AddStringToObject(res, "format", is_svg ? "svg" : "png");
    cJSON_AddStringToObject(res, "plot_type", plot_type_arg);
    cJSON_AddNumberToObject(res, "font_size", font_size);

    if (has_queries)
    {
        cJSON_AddStringToObject(res, "queries_output_path", queries_output);
        cJSON_AddNumberToObject(res, "queries_file_size_bytes", (double)st_q.st_size);
    }
    else
    {
        cJSON_AddNullToObject(res, "queries_output_path");
    }

    cJSON *stats_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(stats_obj, "total_frames", (double)total_frames);
    cJSON_AddNumberToObject(stats_obj, "total_clusters", (double)total_clusters);
    cJSON_AddNumberToObject(stats_obj, "total_distances", (double)total_dists);
    cJSON_AddNumberToObject(stats_obj, "total_pruned", (double)total_pruned);
    cJSON_AddNumberToObject(stats_obj, "radius_limit", rlim);
    cJSON_AddItemToObject(res, "stats", stats_obj);

    char summary_buf[512];
    snprintf(summary_buf, sizeof(summary_buf),
             "Rendered clustering diagnostic visualization: %ld frames clustered into %ld "
             "clusters (rlim=%.3f, %ld distance calls, %ld pruned). Primary image: %s (%ld KB).",
             total_frames, total_clusters, rlim, total_dists, total_pruned,
             resolved_output, (long)(st.st_size / 1024));
    cJSON_AddStringToObject(res, "summary", summary_buf);

    return 0;
} // mcp_tool_plot

const struct mcp_tool_def mcp_tooldef_plot = {
    .name         = "gric_plot",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_plot,
    .description  = "Generate diagnostic visual scatter plots, distance histograms, "
                    "and cluster size distributions using gric-plot.",
    .input_schema = "{\n"
                    "  \"type\": \"object\",\n"
                    "  \"properties\": {\n"
                    "    \"run_dir\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"description\": \"Run directory containing clustering results.\"\n"
                    "    },\n"
                    "    \"points_path\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"description\": \"Path to input points/dataset coordinate file.\"\n"
                    "    },\n"
                    "    \"log_path\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"description\": \"Path to clustering run log file.\"\n"
                    "    },\n"
                    "    \"output_path\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"description\": \"Target output image file (.png or .svg).\"\n"
                    "    },\n"
                    "    \"format\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"enum\": [\"png\", \"svg\"],\n"
                    "      \"description\": \"Output image format (default: 'png').\"\n"
                    "    },\n"
                    "    \"plot_type\": {\n"
                    "      \"type\": \"string\",\n"
                    "      \"enum\": [\"all\", \"distances\", \"sizes\", \"pruning\"],\n"
                    "      \"description\": \"Diagnostic plot type (default: 'all').\"\n"
                    "    },\n"
                    "    \"font_size\": {\n"
                    "      \"type\": \"number\",\n"
                    "      \"description\": \"Font size for plot labels (default: 18.0).\"\n"
                    "    }\n"
                    "  }\n"
                    "}\n",
};

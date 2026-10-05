/**
 * @file tool_jobs.c
 * @brief MCP tools for asynchronous job lifecycle execution, monitoring, and analysis.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_jobs.h"
#include "gric-cluster-analysis/analysis_state.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_ARGV_ITEMS 64

/**
 * is_safe_identifier() - Check if a parameter string contains only safe identifier characters.
 * @str: Input string.
 *
 * Return: 1 if safe, 0 otherwise.
 */
static int is_safe_identifier(
    const char *str)
{
    if (str == NULL || *str == '\0')
    {
        return 0;
    }
    for (size_t ii = 0; str[ii] != '\0'; ii++)
    {
        unsigned char c = (unsigned char)str[ii];
        if (!isalnum(c) && c != '_' && c != '-' && c != '.' && c != '/')
        {
            return 0;
        }
    }
    return 1;
} // is_safe_identifier

int mcp_tool_cluster_start(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *d_item = cJSON_GetObjectItemCaseSensitive(args, "dataset");
    if (d_item == NULL || !cJSON_IsString(d_item) || d_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "dataset parameter is required");
        return -1;
    }

    cJSON *r_item = cJSON_GetObjectItemCaseSensitive(args, "rlim");
    if (r_item == NULL || !cJSON_IsNumber(r_item) || r_item->valuedouble <= 0.0)
    {
        cJSON_AddStringToObject(res, "error", "rlim parameter must be positive number");
        return -1;
    }

    char dataset_path[1024];
    mcp_resolve_path(d_item->valuestring, dataset_path, sizeof(dataset_path));
    if (access(dataset_path, R_OK) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Dataset file not found or unreadable");
        cJSON_AddStringToObject(res, "resolved_path", dataset_path);
        return -1;
    }

    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char cluster_bin[1060];
    mcp_find_executable("gric-cluster", cluster_bin, sizeof(cluster_bin));

    char rlim_str[32];
    snprintf(rlim_str, sizeof(rlim_str), "%.6f", r_item->valuedouble);

    char outdir_buf[1024];
    cJSON *o_item = cJSON_GetObjectItemCaseSensitive(args, "outdir");
    if (o_item != NULL && cJSON_IsString(o_item) && o_item->valuestring[0] != '\0')
    {
        mcp_resolve_path(o_item->valuestring, outdir_buf, sizeof(outdir_buf));
    }
    else
    {
        snprintf(
            outdir_buf, sizeof(outdir_buf),
            "/tmp/gric_cluster_out_%ld_%d.clusterdat",
            (long)time(NULL), (int)getpid());
    }

    const char *argv[MAX_ARGV_ITEMS];
    int idx = 0;
    argv[idx++] = cluster_bin;
    argv[idx++] = rlim_str;
    argv[idx++] = dataset_path;

    char maxim_str[32];
    cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "maxim");
    if (m_item != NULL && cJSON_IsNumber(m_item) && m_item->valueint > 0)
    {
        snprintf(maxim_str, sizeof(maxim_str), "%d", m_item->valueint);
        argv[idx++] = "-maxim";
        argv[idx++] = maxim_str;
    }

    char maxcl_str[32];
    cJSON *mc_item = cJSON_GetObjectItemCaseSensitive(args, "maxcl");
    if (mc_item != NULL && cJSON_IsNumber(mc_item) && mc_item->valueint > 0)
    {
        snprintf(maxcl_str, sizeof(maxcl_str), "%d", mc_item->valueint);
        argv[idx++] = "-maxcl";
        argv[idx++] = maxcl_str;
    }

    argv[idx++] = "-outdir";
    argv[idx++] = outdir_buf;
    argv[idx++] = "-txt";

    cJSON *q_item = cJSON_GetObjectItemCaseSensitive(args, "quant");
    if (q_item != NULL && cJSON_IsString(q_item))
    {
        const char *q = q_item->valuestring;
        if (strcmp(q, "sq8") == 0)
        {
            argv[idx++] = "-sq8";
        }
        else if (strcmp(q, "sq16") == 0)
        {
            argv[idx++] = "-sq16";
        }
        else if (strcmp(q, "eq16") == 0)
        {
            argv[idx++] = "-eq16";
        }
        else if (strcmp(q, "rq8") == 0)
        {
            argv[idx++] = "-rq8";
        }
        else if (strcmp(q, "rabitq") == 0)
        {
            argv[idx++] = "-rabitq";
        }
    }

    cJSON *dbl_item = cJSON_GetObjectItemCaseSensitive(args, "double_precision");
    if (dbl_item != NULL && cJSON_IsTrue(dbl_item))
    {
        argv[idx++] = "-double";
    }

    char th_str[32];
    cJSON *th_item = cJSON_GetObjectItemCaseSensitive(args, "threads");
    if (th_item != NULL && cJSON_IsNumber(th_item) && th_item->valueint > 0)
    {
        snprintf(th_str, sizeof(th_str), "%d", th_item->valueint);
        argv[idx++] = "-ncpu";
        argv[idx++] = th_str;
    }

    cJSON *sp_item = cJSON_GetObjectItemCaseSensitive(args, "sparse_dcc");
    if (sp_item != NULL && cJSON_IsTrue(sp_item))
    {
        argv[idx++] = "-sparse_dcc";
    }

    cJSON *ent_item = cJSON_GetObjectItemCaseSensitive(args, "entropy");
    if (ent_item != NULL && cJSON_IsTrue(ent_item))
    {
        argv[idx++] = "-entropy";
    }

    argv[idx] = NULL;

    char job_id[64];
    if (mcp_job_spawn("gric-cluster", argv, outdir_buf, job_id, sizeof(job_id)) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Failed to spawn gric-cluster background job");
        return -1;
    }

    struct mcp_job_info info;
    mcp_job_get_info(job_id, &info);

    cJSON_AddStringToObject(res, "job_id", job_id);
    cJSON_AddStringToObject(res, "status", "RUNNING");
    cJSON_AddNumberToObject(res, "pid", (double)info.pid);
    cJSON_AddStringToObject(res, "log_path", info.log_path);
    cJSON_AddStringToObject(res, "outdir", outdir_buf);
    return 0;
} // mcp_tool_cluster_start

int mcp_tool_knn_start(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *d_item = cJSON_GetObjectItemCaseSensitive(args, "dataset");
    if (d_item == NULL || !cJSON_IsString(d_item) || d_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "dataset parameter is required");
        return -1;
    }

    cJSON *cl_item = cJSON_GetObjectItemCaseSensitive(args, "cluster_dir");
    if (cl_item == NULL || !cJSON_IsString(cl_item) || cl_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "cluster_dir parameter is required");
        return -1;
    }

    cJSON *out_item = cJSON_GetObjectItemCaseSensitive(args, "output");
    if (out_item == NULL || !cJSON_IsString(out_item) || out_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "output parameter is required");
        return -1;
    }

    char dataset_path[1024];
    char cluster_dir[1024];
    char out_path[1024];
    mcp_resolve_path(d_item->valuestring, dataset_path, sizeof(dataset_path));
    mcp_resolve_path(cl_item->valuestring, cluster_dir, sizeof(cluster_dir));
    mcp_resolve_path(out_item->valuestring, out_path, sizeof(out_path));

    if (access(dataset_path, R_OK) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Dataset file not found or unreadable");
        return -1;
    }
    if (access(cluster_dir, R_OK) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Cluster directory not found or unreadable");
        return -1;
    }

    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char knn_bin[1060];
    mcp_find_executable("gric-knn", knn_bin, sizeof(knn_bin));

    const char *argv[MAX_ARGV_ITEMS];
    int idx = 0;
    argv[idx++] = knn_bin;
    argv[idx++] = dataset_path;
    argv[idx++] = cluster_dir;

    char k_str[32];
    int k_val = 10;
    cJSON *k_item = cJSON_GetObjectItemCaseSensitive(args, "k");
    if (k_item != NULL && cJSON_IsNumber(k_item) && k_item->valueint > 0)
    {
        k_val = k_item->valueint;
    }
    snprintf(k_str, sizeof(k_str), "%d", k_val);
    argv[idx++] = "-k";
    argv[idx++] = k_str;

    char dt_str[32];
    int dt_val = 5;
    cJSON *dt_item = cJSON_GetObjectItemCaseSensitive(args, "dtmin");
    if (dt_item != NULL && cJSON_IsNumber(dt_item) && dt_item->valueint >= 0)
    {
        dt_val = dt_item->valueint;
    }
    snprintf(dt_str, sizeof(dt_str), "%d", dt_val);
    argv[idx++] = "-dtmin";
    argv[idx++] = dt_str;

    argv[idx++] = "-o";
    argv[idx++] = out_path;

    cJSON *q_item = cJSON_GetObjectItemCaseSensitive(args, "quant");
    if (q_item != NULL && cJSON_IsString(q_item))
    {
        const char *q = q_item->valuestring;
        if (strcmp(q, "sq8") == 0)
        {
            argv[idx++] = "-sq8";
        }
        else if (strcmp(q, "sq16") == 0)
        {
            argv[idx++] = "-sq16";
        }
        else if (strcmp(q, "eq16") == 0)
        {
            argv[idx++] = "-eq16";
        }
        else if (strcmp(q, "rq8") == 0)
        {
            argv[idx++] = "-rq8";
        }
        else if (strcmp(q, "rabitq") == 0)
        {
            argv[idx++] = "-rabitq";
        }
    }

    argv[idx] = NULL;

    char job_id[64];
    if (mcp_job_spawn("gric-knn", argv, out_path, job_id, sizeof(job_id)) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Failed to spawn gric-knn background job");
        return -1;
    }

    struct mcp_job_info info;
    mcp_job_get_info(job_id, &info);

    cJSON_AddStringToObject(res, "job_id", job_id);
    cJSON_AddStringToObject(res, "status", "RUNNING");
    cJSON_AddNumberToObject(res, "pid", (double)info.pid);
    cJSON_AddStringToObject(res, "log_path", info.log_path);
    cJSON_AddStringToObject(res, "output", out_path);
    return 0;
} // mcp_tool_knn_start

int mcp_tool_job_status(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *j_item = cJSON_GetObjectItemCaseSensitive(args, "job_id");
    if (j_item == NULL || !cJSON_IsString(j_item) || j_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "job_id parameter is required");
        return -1;
    }

    const char *job_id = j_item->valuestring;
    if (!is_safe_identifier(job_id))
    {
        cJSON_AddStringToObject(res, "error", "Invalid characters in job_id");
        return -1;
    }

    struct mcp_job_info info;
    if (mcp_job_get_info(job_id, &info) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Job not found or inaccessible");
        return -1;
    }

    cJSON_AddStringToObject(res, "job_id", info.job_id);
    cJSON_AddStringToObject(res, "status", mcp_job_status_to_string(info.status));
    cJSON_AddStringToObject(res, "program", info.program);
    cJSON_AddNumberToObject(res, "pid", (double)info.pid);
    cJSON_AddNumberToObject(res, "exit_code", (double)info.exit_code);
    cJSON_AddNumberToObject(res, "elapsed_sec", info.progress.elapsed_sec);
    cJSON_AddNumberToObject(res, "progress_pct", info.progress.progress_pct);
    cJSON_AddNumberToObject(res, "current_frame", (double)info.progress.current_frame);
    cJSON_AddNumberToObject(res, "total_frames", (double)info.progress.total_frames);
    cJSON_AddNumberToObject(res, "clusters_found", (double)info.progress.clusters);
    cJSON_AddStringToObject(res, "log_path", info.log_path);
    if (info.outdir[0] != '\0')
    {
        cJSON_AddStringToObject(res, "outdir", info.outdir);
    }
    if (info.log_tail[0] != '\0')
    {
        cJSON_AddStringToObject(res, "log_tail", info.log_tail);
    }

    return 0;
} // mcp_tool_job_status

int mcp_tool_job_result(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *j_item = cJSON_GetObjectItemCaseSensitive(args, "job_id");
    if (j_item == NULL || !cJSON_IsString(j_item) || j_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "job_id parameter is required");
        return -1;
    }

    const char *job_id = j_item->valuestring;
    if (!is_safe_identifier(job_id))
    {
        cJSON_AddStringToObject(res, "error", "Invalid characters in job_id");
        return -1;
    }

    struct mcp_job_info info;
    if (mcp_job_get_info(job_id, &info) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Job not found or inaccessible");
        return -1;
    }

    cJSON_AddStringToObject(res, "job_id", info.job_id);
    cJSON_AddStringToObject(res, "status", mcp_job_status_to_string(info.status));
    cJSON_AddStringToObject(res, "program", info.program);

    if (info.status == MCP_JOB_STATUS_RUNNING)
    {
        cJSON_AddStringToObject(res, "message", "Job is still running");
        cJSON_AddNumberToObject(res, "progress_pct", info.progress.progress_pct);
        cJSON_AddNumberToObject(res, "elapsed_sec", info.progress.elapsed_sec);
        return 0;
    }

    if (info.status != MCP_JOB_STATUS_COMPLETED)
    {
        cJSON_AddNumberToObject(res, "exit_code", (double)info.exit_code);
        cJSON_AddStringToObject(res, "error", "Job did not complete successfully");
        if (info.log_tail[0] != '\0')
        {
            cJSON_AddStringToObject(res, "log_tail", info.log_tail);
        }
        return 0;
    }

    if (strcmp(info.program, "gric-cluster") == 0 && info.outdir[0] != '\0')
    {
        char log_path[2048];
        char mem_path[2048];
        snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", info.outdir);
        snprintf(mem_path, sizeof(mem_path), "%s/frame_membership.txt", info.outdir);

        AnalysisState state;
        init_state(&state);

        if (parse_log_file(log_path, &state) == 0)
        {
            parse_membership_file(mem_path, &state);
            compute_derived_stats(&state);

            double avg_dists = (state.num_frames > 0) ?
                ((double)state.num_dists / (double)state.num_frames) : 0.0;
            double total_evals = (double)state.num_dists + (double)state.num_pruned;
            double prune_pct = (total_evals > 0.0) ?
                (100.0 * (double)state.num_pruned / total_evals) : 0.0;
            double fps = (state.time_clustering_ms > 0.0) ?
                ((double)state.num_frames / (state.time_clustering_ms / 1000.0)) : 0.0;

            cJSON_AddStringToObject(res, "outdir", info.outdir);
            cJSON_AddNumberToObject(res, "num_clusters", (double)state.num_clusters);
            cJSON_AddNumberToObject(res, "num_frames", (double)state.num_frames);
            cJSON_AddNumberToObject(res, "num_distances_evaluated", (double)state.num_dists);
            cJSON_AddNumberToObject(res, "num_pruned_evaluations", (double)state.num_pruned);
            cJSON_AddNumberToObject(res, "avg_dists_per_frame", avg_dists);
            cJSON_AddNumberToObject(res, "pruning_efficiency_pct", prune_pct);
            cJSON_AddNumberToObject(res, "time_clustering_ms", state.time_clustering_ms);
            cJSON_AddNumberToObject(res, "throughput_fps", fps);
            cJSON_AddNumberToObject(res, "shannon_entropy", state.shannon_entropy);
            cJSON_AddNumberToObject(res, "max_rss_kb", (double)state.max_rss);

            free_state(&state);
            return 0;
        }
        free_state(&state);
    }

    cJSON_AddStringToObject(res, "output", info.outdir);
    cJSON_AddNumberToObject(res, "elapsed_sec", info.progress.elapsed_sec);
    return 0;
} // mcp_tool_job_result

int mcp_tool_job_cancel(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *j_item = cJSON_GetObjectItemCaseSensitive(args, "job_id");
    if (j_item == NULL || !cJSON_IsString(j_item) || j_item->valuestring[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "job_id parameter is required");
        return -1;
    }

    const char *job_id = j_item->valuestring;
    if (!is_safe_identifier(job_id))
    {
        cJSON_AddStringToObject(res, "error", "Invalid characters in job_id");
        return -1;
    }

    int clean_output = 0;
    cJSON *c_item = cJSON_GetObjectItemCaseSensitive(args, "clean_output");
    if (c_item != NULL && cJSON_IsTrue(c_item))
    {
        clean_output = 1;
    }

    if (mcp_job_cancel(job_id, clean_output) != 0)
    {
        cJSON_AddStringToObject(res, "error", "Failed to cancel job");
        return -1;
    }

    cJSON_AddStringToObject(res, "job_id", job_id);
    cJSON_AddStringToObject(res, "status", "CANCELLED");
    cJSON_AddBoolToObject(res, "output_cleaned", clean_output);
    return 0;
} // mcp_tool_job_cancel

const struct mcp_tool_def mcp_tooldef_cluster_start = {
    .name         = "gric_cluster_start",
    .toolset      = MCP_TS_RUN,
    .side_effects = 1,
    .fn           = mcp_tool_cluster_start,
    .description  = "Launch asynchronous gric-cluster background job. Returns job_id immediately.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"dataset\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to input dataset file.\"\n"
        "    },\n"
        "    \"rlim\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Clustering radius limit.\"\n"
        "    },\n"
        "    \"maxim\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Maximum frames to process.\"\n"
        "    },\n"
        "    \"maxcl\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Maximum clusters limit.\"\n"
        "    },\n"
        "    \"outdir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Custom output directory path.\"\n"
        "    },\n"
        "    \"quant\": {\n"
        "      \"type\": \"string\",\n"
        "      \"enum\": [\"sq8\", \"sq16\", \"eq16\", \"rq8\", \"rabitq\"],\n"
        "      \"description\": \"Quantization mode.\"\n"
        "    },\n"
        "    \"double_precision\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Enable double-precision mode.\"\n"
        "    },\n"
        "    \"threads\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of OpenMP worker threads.\"\n"
        "    },\n"
        "    \"sparse_dcc\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Enable sparse inter-cluster distance matrix.\"\n"
        "    },\n"
        "    \"entropy\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Enable dynamic entropy gating.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"dataset\", \"rlim\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_knn_start = {
    .name         = "gric_knn_start",
    .toolset      = MCP_TS_RUN,
    .side_effects = 1,
    .fn           = mcp_tool_knn_start,
    .description  = "Launch asynchronous gric-knn background job. Returns job_id immediately.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"dataset\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to input dataset file.\"\n"
        "    },\n"
        "    \"cluster_dir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to clustered results directory.\"\n"
        "    },\n"
        "    \"output\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Output path for k-NN result file.\"\n"
        "    },\n"
        "    \"k\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of nearest neighbors to find (default: 10).\"\n"
        "    },\n"
        "    \"dtmin\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Minimum temporal sample exclusion delta (default: 5).\"\n"
        "    },\n"
        "    \"quant\": {\n"
        "      \"type\": \"string\",\n"
        "      \"enum\": [\"sq8\", \"sq16\", \"eq16\", \"rq8\", \"rabitq\"],\n"
        "      \"description\": \"Quantization mode.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"dataset\", \"cluster_dir\", \"output\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_job_status = {
    .name         = "gric_job_status",
    .toolset      = MCP_TS_RUN,
    .side_effects = 0,
    .fn           = mcp_tool_job_status,
    .description  = "Check asynchronous job state, runtime progress %, frame counts, and log tail.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"job_id\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Identifier of the background job.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"job_id\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_job_result = {
    .name         = "gric_job_result",
    .toolset      = MCP_TS_RUN,
    .side_effects = 0,
    .fn           = mcp_tool_job_result,
    .description  = "Retrieve structured execution metrics and telemetry once job has completed.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"job_id\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Identifier of the completed job.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"job_id\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_job_cancel = {
    .name         = "gric_job_cancel",
    .toolset      = MCP_TS_RUN,
    .side_effects = 1,
    .fn           = mcp_tool_job_cancel,
    .description  = "Cancel a running asynchronous job and optionally clean up partial outputs.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"job_id\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Identifier of the job to cancel.\"\n"
        "    },\n"
        "    \"clean_output\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Clean up partial output directory if true (default: false).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"job_id\"]\n"
        "}",
};

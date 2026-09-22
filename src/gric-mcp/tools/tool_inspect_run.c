/**
 * @file tool_inspect_run.c
 * @brief High-speed telemetry parser and performance summarizer for clustering runs.
 */

#include "mcp_tools.h"
#include "gric-cluster-analysis/analysis_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mcp_tool_inspect_run(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *dir_item = cJSON_GetObjectItemCaseSensitive(args, "run_dir");
    if (dir_item == NULL || !cJSON_IsString(dir_item))
    {
        cJSON_AddStringToObject(res, "error", "run_dir parameter is required");
        return -1;
    }

    const char *run_dir = dir_item->valuestring;
    char resolved_dir[1024];
    mcp_resolve_path(run_dir, resolved_dir, sizeof(resolved_dir));

    char log_path[2048];
    char mem_path[2048];

    snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", resolved_dir);
    snprintf(mem_path, sizeof(mem_path), "%s/frame_membership.txt", resolved_dir);

    AnalysisState state;
    init_state(&state);

    if (parse_log_file(log_path, &state) != 0)
    {
        free_state(&state);
        cJSON_AddStringToObject(res, "error", "Failed to parse cluster_run.log in run_dir");
        cJSON_AddStringToObject(res, "log_path", log_path);
        return -1;
    }

    /* Try parsing membership file if available */
    parse_membership_file(mem_path, &state);
    compute_derived_stats(&state);

    double avg_dists = (state.num_frames > 0) ?
        ((double)state.num_dists / (double)state.num_frames) : 0.0;

    double total_evals = (double)state.num_dists + (double)state.num_pruned;
    double prune_pct = (total_evals > 0.0) ?
        (100.0 * (double)state.num_pruned / total_evals) : 0.0;

    double fps = (state.time_clustering_ms > 0.0) ?
        ((double)state.num_frames / (state.time_clustering_ms / 1000.0)) : 0.0;

    cJSON_AddStringToObject(res, "run_directory", run_dir);
    cJSON_AddStringToObject(res, "cmdline", state.cmdline);
    cJSON_AddNumberToObject(res, "num_clusters", state.num_clusters);
    cJSON_AddNumberToObject(res, "num_frames", (double)state.num_frames);
    cJSON_AddNumberToObject(res, "num_distances_evaluated", (double)state.num_dists);
    cJSON_AddNumberToObject(res, "num_pruned_evaluations", (double)state.num_pruned);
    cJSON_AddNumberToObject(res, "avg_dists_per_frame", avg_dists);
    cJSON_AddNumberToObject(res, "pruning_efficiency_pct", prune_pct);
    cJSON_AddNumberToObject(res, "time_clustering_ms", state.time_clustering_ms);
    cJSON_AddNumberToObject(res, "throughput_fps", fps);
    cJSON_AddNumberToObject(res, "shannon_entropy", state.shannon_entropy);
    cJSON_AddNumberToObject(res, "normalized_entropy", state.normalized_entropy);
    cJSON_AddNumberToObject(res, "max_rss_kb", (double)state.max_rss);

    free_state(&state);
    return 0;
} // mcp_tool_inspect_run

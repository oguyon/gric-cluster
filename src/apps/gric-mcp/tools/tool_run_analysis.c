/**
 * @file tool_run_analysis.c
 * @brief Run comparison, Adjusted Rand Index (ARI), and cluster profiling tools for MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "gric-cluster-analysis/analysis_state.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * find_file_variant() - Locate an analysis file with common naming conventions.
 */
static int find_file_variant(
    const char *base_dir,
    const char *primary_name,
    const char *alt_name,
    char       *out_path,
    size_t      out_size)
{
    snprintf(out_path, out_size, "%s/%s", base_dir, primary_name);
    if (access(out_path, R_OK) == 0)
    {
        return 0;
    }

    if (alt_name != NULL)
    {
        snprintf(out_path, out_size, "%s/%s", base_dir, alt_name);
        if (access(out_path, R_OK) == 0)
        {
            return 0;
        }
    }

    /* Try base_dir directly as the file if it matches extension */
    if (access(base_dir, R_OK) == 0)
    {
        snprintf(out_path, out_size, "%s", base_dir);
        return 0;
    }

    return -1;
}

/**
 * compute_ari() - Calculate the Adjusted Rand Index between two cluster assignments.
 */
static double compute_ari(
    const int *labels_a,
    long       n_a,
    const int *labels_b,
    long       n_b)
{
    long n = (n_a < n_b) ? n_a : n_b;
    if (n <= 1 || labels_a == NULL || labels_b == NULL)
    {
        return 1.0;
    }

    int max_a = 0;
    int max_b = 0;
    for (long ii = 0; ii < n; ii++)
    {
        if (labels_a[ii] > max_a)
        {
            max_a = labels_a[ii];
        }
        if (labels_b[ii] > max_b)
        {
            max_b = labels_b[ii];
        }
    }

    int k_a = max_a + 1;
    int k_b = max_b + 1;

    /* Guard against excessive contingency table memory */
    if ((double)k_a * (double)k_b > 10000000.0)
    {
        /* Fallback: compute agreement fraction */
        long match = 0;
        for (long ii = 0; ii < n; ii++)
        {
            if (labels_a[ii] == labels_b[ii])
            {
                match++;
            }
        }
        return (double)match / (double)n;
    }

    int *table = (int *)calloc((size_t)k_a * (size_t)k_b, sizeof(int));
    int *row_sum = (int *)calloc((size_t)k_a, sizeof(int));
    int *col_sum = (int *)calloc((size_t)k_b, sizeof(int));

    if (table == NULL || row_sum == NULL || col_sum == NULL)
    {
        free(table);
        free(row_sum);
        free(col_sum);
        return 1.0;
    }

    for (long ii = 0; ii < n; ii++)
    {
        int ca = labels_a[ii];
        int cb = labels_b[ii];
        if (ca >= 0 && ca < k_a && cb >= 0 && cb < k_b)
        {
            table[ca * k_b + cb]++;
            row_sum[ca]++;
            col_sum[cb]++;
        }
    }

    double sum_comb = 0.0;
    for (int i = 0; i < k_a; i++)
    {
        for (int j = 0; j < k_b; j++)
        {
            long count = table[i * k_b + j];
            if (count > 1)
            {
                sum_comb += (double)(count * (count - 1)) / 2.0;
            }
        }
    }

    double sum_a = 0.0;
    for (int i = 0; i < k_a; i++)
    {
        long count = row_sum[i];
        if (count > 1)
        {
            sum_a += (double)(count * (count - 1)) / 2.0;
        }
    }

    double sum_b = 0.0;
    for (int j = 0; j < k_b; j++)
    {
        long count = col_sum[j];
        if (count > 1)
        {
            sum_b += (double)(count * (count - 1)) / 2.0;
        }
    }

    free(table);
    free(row_sum);
    free(col_sum);

    double total_comb = (double)(n * (n - 1)) / 2.0;
    if (total_comb <= 0.0)
    {
        return 1.0;
    }

    double expected_idx = (sum_a * sum_b) / total_comb;
    double max_idx = 0.5 * (sum_a + sum_b);
    double denom = max_idx - expected_idx;

    if (fabs(denom) < 1e-12)
    {
        return 1.0;
    }

    double ari = (sum_comb - expected_idx) / denom;
    if (ari < -1.0)
    {
        ari = -1.0;
    }
    if (ari > 1.0)
    {
        ari = 1.0;
    }
    return ari;
} // compute_ari

/**
 * mcp_tool_compare_runs() - Compare performance and stability across two clustering runs.
 */
int mcp_tool_compare_runs(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *item_a = cJSON_GetObjectItemCaseSensitive(args, "run_dir_a");
    cJSON *item_b = cJSON_GetObjectItemCaseSensitive(args, "run_dir_b");

    if (item_a == NULL || !cJSON_IsString(item_a) ||
        item_b == NULL || !cJSON_IsString(item_b))
    {
        cJSON_AddStringToObject(res, "error", "run_dir_a and run_dir_b are required");
        return -1;
    }

    char resolved_a[512];
    char resolved_b[512];
    mcp_resolve_path(item_a->valuestring, resolved_a, sizeof(resolved_a));
    mcp_resolve_path(item_b->valuestring, resolved_b, sizeof(resolved_b));

    AnalysisState state_a;
    AnalysisState state_b;
    init_state(&state_a);
    init_state(&state_b);

    /* Ingest Run A */
    char path_log_a[512];
    char path_mem_a[512];
    if (find_file_variant(
            resolved_a, "job.log", "clustering.log",
            path_log_a, sizeof(path_log_a)) == 0)
    {
        parse_log_file(path_log_a, &state_a);
    }
    if (find_file_variant(
            resolved_a, "membership.txt", "membership.dat",
            path_mem_a, sizeof(path_mem_a)) == 0)
    {
        parse_membership_file(path_mem_a, &state_a);
    }

    /* Ingest Run B */
    char path_log_b[512];
    char path_mem_b[512];
    if (find_file_variant(
            resolved_b, "job.log", "clustering.log",
            path_log_b, sizeof(path_log_b)) == 0)
    {
        parse_log_file(path_log_b, &state_b);
    }
    if (find_file_variant(
            resolved_b, "membership.txt", "membership.dat",
            path_mem_b, sizeof(path_mem_b)) == 0)
    {
        parse_membership_file(path_mem_b, &state_b);
    }

    /* Compute derived statistics if assignments are present */
    compute_derived_stats(&state_a);
    compute_derived_stats(&state_b);

    /* Assignment comparison */
    double agreement_pct = 100.0;
    double ari = 1.0;
    long total_evaluated = 0;

    if (state_a.assignments != NULL && state_b.assignments != NULL)
    {
        total_evaluated = (state_a.assignments_count < state_b.assignments_count) ?
                          state_a.assignments_count : state_b.assignments_count;
        if (total_evaluated > 0)
        {
            long matching = 0;
            for (long ii = 0; ii < total_evaluated; ii++)
            {
                if (state_a.assignments[ii] == state_b.assignments[ii])
                {
                    matching++;
                }
            }
            agreement_pct = (double)matching / (double)total_evaluated * 100.0;
            ari = compute_ari(
                state_a.assignments, state_a.assignments_count,
                state_b.assignments, state_b.assignments_count);
        }
    }

    /* Performance and speedup calculation */
    double speedup = 1.0;
    if (state_a.time_clustering_ms > 0.0 && state_b.time_clustering_ms > 0.0)
    {
        speedup = state_a.time_clustering_ms / state_b.time_clustering_ms;
    }

    double pruning_a = 0.0;
    double pruning_b = 0.0;
    if (state_a.num_dists + state_a.num_pruned > 0)
    {
        pruning_a = (double)state_a.num_pruned /
                    (double)(state_a.num_dists + state_a.num_pruned) * 100.0;
    }
    if (state_b.num_dists + state_b.num_pruned > 0)
    {
        pruning_b = (double)state_b.num_pruned /
                    (double)(state_b.num_dists + state_b.num_pruned) * 100.0;
    }

    /* Construct JSON result */
    cJSON *conf_a = cJSON_CreateObject();
    cJSON_AddStringToObject(conf_a, "path", resolved_a);
    cJSON_AddNumberToObject(conf_a, "clusters", state_a.num_clusters);
    cJSON_AddNumberToObject(conf_a, "frames", (double)state_a.num_frames);
    cJSON_AddNumberToObject(conf_a, "time_ms", state_a.time_clustering_ms);
    cJSON_AddNumberToObject(conf_a, "pruning_pct", pruning_a);
    cJSON_AddNumberToObject(conf_a, "max_rss_kb", (double)state_a.max_rss);
    cJSON_AddItemToObject(res, "run_a", conf_a);

    cJSON *conf_b = cJSON_CreateObject();
    cJSON_AddStringToObject(conf_b, "path", resolved_b);
    cJSON_AddNumberToObject(conf_b, "clusters", state_b.num_clusters);
    cJSON_AddNumberToObject(conf_b, "frames", (double)state_b.num_frames);
    cJSON_AddNumberToObject(conf_b, "time_ms", state_b.time_clustering_ms);
    cJSON_AddNumberToObject(conf_b, "pruning_pct", pruning_b);
    cJSON_AddNumberToObject(conf_b, "max_rss_kb", (double)state_b.max_rss);
    cJSON_AddItemToObject(res, "run_b", conf_b);

    cJSON *comp = cJSON_CreateObject();
    cJSON_AddNumberToObject(comp, "adjusted_rand_index", ari);
    cJSON_AddNumberToObject(comp, "agreement_percent", agreement_pct);
    cJSON_AddNumberToObject(comp, "delta_clusters", state_b.num_clusters - state_a.num_clusters);
    cJSON_AddNumberToObject(comp, "speedup_ratio", speedup);
    cJSON_AddNumberToObject(comp, "frames_compared", (double)total_evaluated);
    cJSON_AddItemToObject(res, "comparison", comp);

    /* Formatted comparison report */
    char report[1024];
    const char *verdict = "Distinct Clusterings";
    if (ari >= 0.999 && fabs(agreement_pct - 100.0) < 1e-4)
    {
        verdict = "Identical / Bit-Exact Equivalent";
    }
    else if (ari >= 0.95)
    {
        verdict = "High Cluster Consistency (ARI >= 0.95)";
    }
    else if (ari >= 0.80)
    {
        verdict = "Moderate Cluster Alignment (ARI >= 0.80)";
    }

    snprintf(
        report, sizeof(report),
        "### Clustering Run Comparison\n"
        "- **Status**: %s\n"
        "- **Adjusted Rand Index (ARI)**: %.4f\n"
        "- **Frame Agreement**: %.2f%% (%ld frames evaluated)\n"
        "- **Cluster Count**: Run A: %d | Run B: %d (Delta: %+d)\n"
        "- **Clustering Time**: Run A: %.2f ms | Run B: %.2f ms (Speedup: %.2fx)\n"
        "- **Pruning Rate**: Run A: %.2f%% | Run B: %.2f%%",
        verdict, ari, agreement_pct, total_evaluated,
        state_a.num_clusters, state_b.num_clusters,
        state_b.num_clusters - state_a.num_clusters,
        state_a.time_clustering_ms, state_b.time_clustering_ms, speedup,
        pruning_a, pruning_b);
    cJSON_AddStringToObject(res, "report", report);

    free_state(&state_a);
    free_state(&state_b);
    return 0;
} // mcp_tool_compare_runs

/**
 * mcp_tool_cluster_detail() - Deep inspection of a specific cluster's geometry and dynamics.
 */
int mcp_tool_cluster_detail(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *dir_item = cJSON_GetObjectItemCaseSensitive(args, "run_dir");
    cJSON *cid_item = cJSON_GetObjectItemCaseSensitive(args, "cluster_id");

    if (dir_item == NULL || !cJSON_IsString(dir_item) ||
        cid_item == NULL || !cJSON_IsNumber(cid_item))
    {
        cJSON_AddStringToObject(res, "error", "run_dir and cluster_id are required");
        return -1;
    }

    const char *run_dir = dir_item->valuestring;
    int target_cid = cid_item->valueint;

    if (target_cid < 0)
    {
        cJSON_AddStringToObject(res, "error", "cluster_id must be non-negative (>= 0)");
        return -1;
    }

    char resolved_dir[512];
    mcp_resolve_path(run_dir, resolved_dir, sizeof(resolved_dir));

    AnalysisState state;
    init_state(&state);

    char mem_path[512];
    if (find_file_variant(
            resolved_dir, "membership.txt", "membership.dat",
            mem_path, sizeof(mem_path)) != 0)
    {
        free_state(&state);
        cJSON_AddStringToObject(res, "error", "membership file not found in run_dir");
        return -1;
    }

    if (parse_membership_file(mem_path, &state) != 0 || state.assignments_count == 0)
    {
        free_state(&state);
        cJSON_AddStringToObject(res, "error", "failed to parse membership file");
        return -1;
    }

    /* Compute member stats, lifetimes, and transitions */
    long member_count = 0;
    long birth_frame = -1;
    long death_frame = -1;

    /* Transition frequency tracking (top outgoing / incoming) */
    int max_seen_cid = 0;
    for (long t = 0; t < state.assignments_count; t++)
    {
        int c = state.assignments[t];
        if (c > max_seen_cid)
        {
            max_seen_cid = c;
        }
        if (c == target_cid)
        {
            member_count++;
            if (birth_frame == -1)
            {
                birth_frame = t;
            }
            death_frame = t;
        }
    }

    if (member_count == 0)
    {
        free_state(&state);
        cJSON_AddStringToObject(res, "error", "cluster_id not found in membership assignments");
        return -1;
    }

    int trans_capacity = max_seen_cid + 1;
    long *out_trans = (long *)calloc((size_t)trans_capacity, sizeof(long));
    long *in_trans = (long *)calloc((size_t)trans_capacity, sizeof(long));

    for (long t = 0; t < state.assignments_count; t++)
    {
        int c = state.assignments[t];
        if (t > 0)
        {
            int prev_c = state.assignments[t - 1];
            if (prev_c == target_cid && c != target_cid && c < trans_capacity)
            {
                out_trans[c]++;
            }
            if (c == target_cid && prev_c != target_cid && prev_c < trans_capacity)
            {
                in_trans[prev_c]++;
            }
        }
    }

    /* Find top transition targets */
    int top_out_cid = -1;
    long top_out_count = 0;
    int top_in_cid = -1;
    long top_in_count = 0;

    for (int c = 0; c < trans_capacity; c++)
    {
        if (out_trans[c] > top_out_count)
        {
            top_out_count = out_trans[c];
            top_out_cid = c;
        }
        if (in_trans[c] > top_in_count)
        {
            top_in_count = in_trans[c];
            top_in_cid = c;
        }
    }

    free(out_trans);
    free(in_trans);

    long temporal_span = death_frame - birth_frame + 1;
    double duty_cycle_pct = (temporal_span > 0) ?
                            ((double)member_count / (double)temporal_span * 100.0) : 100.0;
    double pop_pct = (double)member_count / (double)state.assignments_count * 100.0;

    /* Check centroid file */
    char cent_path[512];
    double centroid_norm = 0.0;
    int cent_dim = 0;
    cJSON *coords_sample = cJSON_CreateArray();

    if (find_file_variant(
            resolved_dir, "clusters.txt", "clusters.dat",
            cent_path, sizeof(cent_path)) == 0)
    {
        FILE *fc = fopen(cent_path, "r");
        if (fc != NULL)
        {
            char line[65536];
            int curr_line = 0;
            while (fgets(line, sizeof(line), fc) != NULL)
            {
                if (curr_line == target_cid)
                {
                    char *p = line;
                    while (*p != '\0')
                    {
                        char *end = NULL;
                        double val = strtod(p, &end);
                        if (end == p)
                        {
                            break;
                        }
                        centroid_norm += val * val;
                        cent_dim++;
                        if (cent_dim <= 8)
                        {
                            cJSON_AddItemToArray(coords_sample, cJSON_CreateNumber(val));
                        }
                        p = end;
                    }
                    centroid_norm = sqrt(centroid_norm);
                    break;
                }
                curr_line++;
            }
            fclose(fc);
        }
    }

    cJSON_AddNumberToObject(res, "cluster_id", target_cid);
    cJSON_AddNumberToObject(res, "member_frames", (double)member_count);
    cJSON_AddNumberToObject(res, "total_run_frames", (double)state.assignments_count);
    cJSON_AddNumberToObject(res, "population_percentage", pop_pct);

    cJSON *dyn = cJSON_CreateObject();
    cJSON_AddNumberToObject(dyn, "birth_frame", (double)birth_frame);
    cJSON_AddNumberToObject(dyn, "death_frame", (double)death_frame);
    cJSON_AddNumberToObject(dyn, "active_span_frames", (double)temporal_span);
    cJSON_AddNumberToObject(dyn, "duty_cycle_percent", duty_cycle_pct);
    if (top_out_cid >= 0)
    {
        cJSON_AddNumberToObject(dyn, "primary_next_cluster", top_out_cid);
        cJSON_AddNumberToObject(dyn, "primary_next_transitions", (double)top_out_count);
    }
    if (top_in_cid >= 0)
    {
        cJSON_AddNumberToObject(dyn, "primary_prev_cluster", top_in_cid);
        cJSON_AddNumberToObject(dyn, "primary_prev_transitions", (double)top_in_count);
    }
    cJSON_AddItemToObject(res, "temporal_dynamics", dyn);

    cJSON *geom = cJSON_CreateObject();
    cJSON_AddNumberToObject(geom, "dimension", cent_dim);
    cJSON_AddNumberToObject(geom, "centroid_l2_norm", centroid_norm);
    cJSON_AddItemToObject(geom, "coordinates_sample", coords_sample);
    cJSON_AddItemToObject(res, "centroid_geometry", geom);

    char summary[1024];
    snprintf(
        summary, sizeof(summary),
        "### Cluster %d Profile\n"
        "- **Member Frames**: %ld (%.2f%% of run)\n"
        "- **Active Lifetime**: Frame %ld to %ld (Span: %ld frames, Duty cycle: %.1f%%)\n"
        "- **Centroid**: %d-dimensional (L2 Norm: %.4f)\n"
        "- **Transitions**: Primary successor -> Cluster %d (%ld transitions)",
        target_cid, member_count, pop_pct,
        birth_frame, death_frame, temporal_span, duty_cycle_pct,
        cent_dim, centroid_norm,
        top_out_cid, top_out_count);
    cJSON_AddStringToObject(res, "summary", summary);

    free_state(&state);
    return 0;
} // mcp_tool_cluster_detail

const struct mcp_tool_def mcp_tooldef_compare_runs = {
    .name         = "gric_compare_runs",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_compare_runs,
    .description  = "Compare two clustering runs: evaluates Adjusted Rand Index (ARI), "
                    "frame assignment agreement, cluster count deltas, and runtime speedup.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"run_dir_a\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Baseline clustering run directory or log file.\"\n"
        "    },\n"
        "    \"run_dir_b\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Candidate clustering run directory or log file.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"run_dir_a\", \"run_dir_b\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_cluster_detail = {
    .name         = "gric_cluster_detail",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_cluster_detail,
    .description  = "Inspect a specific cluster in detail: member frames, temporal lifetime "
                    "(birth/death), Markov transition dynamics, and centroid geometry.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"run_dir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Clustering run output directory.\"\n"
        "    },\n"
        "    \"cluster_id\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"0-indexed identifier of the cluster to inspect.\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"run_dir\", \"cluster_id\"]\n"
        "}",
};

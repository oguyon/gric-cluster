/**
 * @file tool_dev_golden_compare.c
 * @brief Rapid regression testing tool comparing clustering outputs against golden baselines.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_exec.h"
#include "gric-cluster-analysis/analysis_state.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_EXEC_OUTPUT_BYTES 131072 /* 128 KB */

/**
 * struct golden_preset - Reference metrics and settings for standard golden presets.
 */
struct golden_preset
{
    const char *name;
    const char *dataset_rel;
    double      rlim;
    int         maxim;
    int         baseline_clusters;
    long        baseline_frames;
    long        baseline_dists;
    long        baseline_pruned;
    double      baseline_time_ms;
};

static const struct golden_preset g_presets[] = {
    {
        .name              = "spiral",
        .dataset_rel       = "/tmp/ctest_spiral.txt",
        .rlim              = 0.10,
        .maxim             = 1000,
        .baseline_clusters = 62,
        .baseline_frames   = 1000,
        .baseline_dists    = 2829,
        .baseline_pruned   = 4984,
        .baseline_time_ms  = 1.5,
    },
    {
        .name              = "balls",
        .dataset_rel       = "/tmp/ctest_balls_1.fits",
        .rlim              = 3.0,
        .maxim             = 500,
        .baseline_clusters = 153,
        .baseline_frames   = 500,
        .baseline_dists    = 11975,
        .baseline_pruned   = 52630,
        .baseline_time_ms  = 8.0,
    },
    {
        .name              = "gaussian64",
        .dataset_rel       = "/tmp/ctest_gaussian64.txt",
        .rlim              = 2.5,
        .maxim             = 500,
        .baseline_clusters = 500,
        .baseline_frames   = 500,
        .baseline_dists    = 124750,
        .baseline_pruned   = 124750,
        .baseline_time_ms  = 4.0,
    },
};

static const size_t g_num_presets = sizeof(g_presets) / sizeof(g_presets[0]);

/**
 * cleanup_directory() - Safely delete a temporary golden test directory.
 * @dir_path: Path to directory to remove (must have /tmp/gric_golden_ prefix).
 */
static void cleanup_directory(
    const char *dir_path)
{
    if (dir_path == NULL || strncmp(dir_path, "/tmp/gric_golden_", 17) != 0)
    {
        return;
    }
    const char *const rm_argv[] = {"rm", "-rf", dir_path, NULL};
    char dummy_out[64];
    int status = 0;
    mcp_exec_capture(rm_argv, dummy_out, sizeof(dummy_out), 5000, &status);
} // cleanup_directory

/**
 * ensure_preset_dataset() - Ensure input dataset file exists or generate it.
 * @preset_name: Name of the preset ("spiral", "balls", "gaussian64").
 * @project_root: Repository root path containing build binaries.
 * @dataset_path: Target path for the dataset.
 *
 * Return: 0 on success, -1 on failure.
 */
static int ensure_preset_dataset(
    const char *preset_name,
    const char *project_root,
    const char *dataset_path)
{
    if (access(dataset_path, R_OK) == 0)
    {
        return 0;
    }

    char exe_buf[1060];
    char dummy_out[1024];
    int exit_status = 0;

    if (strcmp(preset_name, "spiral") == 0)
    {
        snprintf(exe_buf, sizeof(exe_buf), "%s/build/gric-mktxtseq", project_root);
        const char *const argv[] = {
            exe_buf, "1000", dataset_path, "2Dspiral", NULL
        };
        int ret = mcp_exec_capture(argv, dummy_out, sizeof(dummy_out), 15000, &exit_status);
        return (ret == 0 && exit_status == 0) ? 0 : -1;
    }
    if (strcmp(preset_name, "balls") == 0)
    {
        snprintf(exe_buf, sizeof(exe_buf), "%s/build/gric-gen-balls", project_root);
        const char *const argv[] = {
            exe_buf, "-n", "1", "-r", "5.0", "-W", "32", "-H", "32",
            "-f", "500", "-s", "42", dataset_path, NULL
        };
        int ret = mcp_exec_capture(argv, dummy_out, sizeof(dummy_out), 15000, &exit_status);
        return (ret == 0 && exit_status == 0) ? 0 : -1;
    }
    if (strcmp(preset_name, "gaussian64") == 0)
    {
        snprintf(exe_buf, sizeof(exe_buf), "%s/build/gric-mktxtseq", project_root);
        const char *const argv[] = {
            exe_buf, "500", dataset_path, "64Drandom", NULL
        };
        int ret = mcp_exec_capture(argv, dummy_out, sizeof(dummy_out), 15000, &exit_status);
        return (ret == 0 && exit_status == 0) ? 0 : -1;
    }

    return -1;
} // ensure_preset_dataset

/**
 * compute_centroid_max_drift() - Calculate maximum coordinate distance drift across anchors.
 * @baseline_dir: Path to baseline results directory containing anchors.txt.
 * @curr_dir:     Path to current results directory containing anchors.txt.
 *
 * Return: Maximum Euclidean distance shift between matching clusters, or 0.0 on error.
 */
static double compute_centroid_max_drift(
    const char *baseline_dir,
    const char *curr_dir)
{
    char base_anchors[1060];
    char curr_anchors[1060];
    snprintf(base_anchors, sizeof(base_anchors), "%s/anchors.txt", baseline_dir);
    snprintf(curr_anchors, sizeof(curr_anchors), "%s/anchors.txt", curr_dir);

    FILE *f_base = fopen(base_anchors, "r");
    FILE *f_curr = fopen(curr_anchors, "r");
    if (f_base == NULL || f_curr == NULL)
    {
        if (f_base != NULL)
        {
            fclose(f_base);
        }
        if (f_curr != NULL)
        {
            fclose(f_curr);
        }
        return 0.0;
    }

    char line_b[4096];
    char line_c[4096];
    double max_drift = 0.0;

    while (fgets(line_b, sizeof(line_b), f_base) != NULL &&
           fgets(line_c, sizeof(line_c), f_curr) != NULL)
    {
        char *end_b = line_b;
        char *end_c = line_c;
        double sum_sq = 0.0;

        while (1)
        {
            double val_b = strtod(end_b, &end_b);
            double val_c = strtod(end_c, &end_c);
            double diff = val_b - val_c;
            sum_sq += diff * diff;

            while (isspace((unsigned char)*end_b))
            {
                end_b++;
            }
            while (isspace((unsigned char)*end_c))
            {
                end_c++;
            }
            if (*end_b == '\0' || *end_c == '\0')
            {
                break;
            }
        } // while coordinates

        double dist = sqrt(sum_sq);
        if (dist > max_drift)
        {
            max_drift = dist;
        }
    } // while lines

    fclose(f_base);
    fclose(f_curr);
    return max_drift;
} // compute_centroid_max_drift

/**
 * compute_assignment_agreement() - Calculate percentage of identical frame cluster assignments.
 * @base_state: Baseline parsed analysis state.
 * @curr_state: Current parsed analysis state.
 *
 * Return: Agreement percentage from 0.0 to 100.0.
 */
static double compute_assignment_agreement(
    const AnalysisState *base_state,
    const AnalysisState *curr_state)
{
    if (base_state->assignments == NULL || curr_state->assignments == NULL)
    {
        return 100.0;
    }
    long total = base_state->assignments_count;
    if (curr_state->assignments_count < total)
    {
        total = curr_state->assignments_count;
    }
    if (total <= 0)
    {
        return 100.0;
    }

    long matches = 0;
    for (long ii = 0; ii < total; ii++)
    {
        if (base_state->assignments[ii] == curr_state->assignments[ii])
        {
            matches++;
        }
    }

    return (100.0 * (double)matches) / (double)total;
} // compute_assignment_agreement

int mcp_tool_dev_golden_compare(
    const cJSON *args,
    cJSON       *res)
{
    char root[1024];
    mcp_get_project_root(root, sizeof(root));
    if (root[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "Could not locate project root directory");
        return -1;
    }

    const char *preset_name = "spiral";
    const char *custom_dataset = NULL;
    const char *custom_baseline_dir = NULL;
    double custom_rlim = -1.0;
    double max_dist_drift_pct = 5.0;
    double min_agreement_pct = 99.0;

    if (args != NULL)
    {
        cJSON *p_item = cJSON_GetObjectItemCaseSensitive(args, "preset");
        if (p_item != NULL && cJSON_IsString(p_item) && p_item->valuestring[0] != '\0')
        {
            preset_name = p_item->valuestring;
        }

        cJSON *d_item = cJSON_GetObjectItemCaseSensitive(args, "dataset");
        if (d_item != NULL && cJSON_IsString(d_item) && d_item->valuestring[0] != '\0')
        {
            custom_dataset = d_item->valuestring;
        }

        cJSON *b_item = cJSON_GetObjectItemCaseSensitive(args, "baseline_dir");
        if (b_item != NULL && cJSON_IsString(b_item) && b_item->valuestring[0] != '\0')
        {
            custom_baseline_dir = b_item->valuestring;
        }

        cJSON *r_item = cJSON_GetObjectItemCaseSensitive(args, "rlim");
        if (r_item != NULL && cJSON_IsNumber(r_item) && r_item->valuedouble > 0.0)
        {
            custom_rlim = r_item->valuedouble;
        }

        cJSON *m_item = cJSON_GetObjectItemCaseSensitive(args, "max_dist_drift_pct");
        if (m_item != NULL && cJSON_IsNumber(m_item))
        {
            max_dist_drift_pct = m_item->valuedouble;
        }

        cJSON *a_item = cJSON_GetObjectItemCaseSensitive(args, "min_agreement_pct");
        if (a_item != NULL && cJSON_IsNumber(a_item))
        {
            min_agreement_pct = a_item->valuedouble;
        }
    }

    const struct golden_preset *preset = NULL;
    for (size_t ii = 0; ii < g_num_presets; ii++)
    {
        if (strcmp(g_presets[ii].name, preset_name) == 0)
        {
            preset = &g_presets[ii];
            break;
        }
    }

    if (preset == NULL && custom_dataset == NULL)
    {
        cJSON_AddStringToObject(
            res, "error", "Unknown preset name and no custom dataset specified");
        return -1;
    }

    char dataset_path[1024];
    double rlim = (preset != NULL) ? preset->rlim : 0.10;
    int maxim = (preset != NULL) ? preset->maxim : 1000;

    if (custom_dataset != NULL)
    {
        mcp_resolve_path(custom_dataset, dataset_path, sizeof(dataset_path));
    }
    else
    {
        snprintf(dataset_path, sizeof(dataset_path), "%s", preset->dataset_rel);
        if (ensure_preset_dataset(preset->name, root, dataset_path) != 0)
        {
            cJSON_AddStringToObject(res, "error", "Failed to generate or access preset dataset");
            return -1;
        }
    }

    if (custom_rlim > 0.0)
    {
        rlim = custom_rlim;
    }

    /* 1. Parse baseline if provided */
    AnalysisState base_state;
    init_state(&base_state);
    int has_baseline_state = 0;

    if (custom_baseline_dir != NULL)
    {
        char resolved_base[1024];
        mcp_resolve_path(custom_baseline_dir, resolved_base, sizeof(resolved_base));

        char base_log[2048];
        char base_mem[2048];
        snprintf(base_log, sizeof(base_log), "%s/cluster_run.log", resolved_base);
        snprintf(base_mem, sizeof(base_mem), "%s/frame_membership.txt", resolved_base);

        if (parse_log_file(base_log, &base_state) == 0)
        {
            parse_membership_file(base_mem, &base_state);
            has_baseline_state = 1;
        }
    }

    /* 2. Execute gric-cluster in temporary output directory */
    char tmp_outdir[256];
    snprintf(
        tmp_outdir, sizeof(tmp_outdir),
        "/tmp/gric_golden_%s_%d.clusterdat",
        preset_name, (int)getpid());

    char cluster_bin[1060];
    snprintf(cluster_bin, sizeof(cluster_bin), "%s/build/gric-cluster", root);

    char rlim_str[32];
    snprintf(rlim_str, sizeof(rlim_str), "%.6f", rlim);

    char maxim_str[32];
    snprintf(maxim_str, sizeof(maxim_str), "%d", maxim);

    const char *const cluster_argv[] = {
        cluster_bin,
        rlim_str,
        dataset_path,
        "-maxim", maxim_str,
        "-outdir", tmp_outdir,
        "-txt",
        NULL
    };

    char *exec_out = malloc(MAX_EXEC_OUTPUT_BYTES);
    if (exec_out == NULL)
    {
        free_state(&base_state);
        cJSON_AddStringToObject(res, "error", "Out of memory allocating exec buffer");
        return -1;
    }
    exec_out[0] = '\0';

    int exit_status = 0;
    int exec_ret = mcp_exec_capture(
        cluster_argv, exec_out, MAX_EXEC_OUTPUT_BYTES, 60000, &exit_status);
    free(exec_out);

    if (exec_ret != 0 || exit_status != 0)
    {
        cleanup_directory(tmp_outdir);
        free_state(&base_state);
        cJSON_AddStringToObject(res, "error", "gric-cluster run failed during golden compare");
        cJSON_AddNumberToObject(res, "exit_code", exit_status);
        return -1;
    }

    /* 3. Parse current run results */
    AnalysisState curr_state;
    init_state(&curr_state);

    char curr_log[2048];
    char curr_mem[2048];
    snprintf(curr_log, sizeof(curr_log), "%s/cluster_run.log", tmp_outdir);
    snprintf(curr_mem, sizeof(curr_mem), "%s/frame_membership.txt", tmp_outdir);

    if (parse_log_file(curr_log, &curr_state) != 0)
    {
        cleanup_directory(tmp_outdir);
        free_state(&base_state);
        free_state(&curr_state);
        cJSON_AddStringToObject(res, "error", "Failed to parse current cluster_run.log");
        return -1;
    }
    parse_membership_file(curr_mem, &curr_state);

    /* 4. Compare metrics */
    int base_clusters = has_baseline_state ? base_state.num_clusters :
                        (preset ? preset->baseline_clusters : curr_state.num_clusters);
    long base_dists = has_baseline_state ? base_state.num_dists :
                      (preset ? preset->baseline_dists : curr_state.num_dists);
    double base_time = has_baseline_state ? base_state.time_clustering_ms :
                       (preset ? preset->baseline_time_ms : curr_state.time_clustering_ms);

    int cluster_delta = curr_state.num_clusters - base_clusters;
    double dist_ratio = (base_dists > 0) ?
        ((double)curr_state.num_dists / (double)base_dists) : 1.0;
    double speedup = (curr_state.time_clustering_ms > 0.0) ?
        (base_time / curr_state.time_clustering_ms) : 1.0;

    double agreement_pct = 100.0;
    if (has_baseline_state)
    {
        agreement_pct = compute_assignment_agreement(&base_state, &curr_state);
    }

    double centroid_drift = 0.0;
    if (has_baseline_state && custom_baseline_dir != NULL)
    {
        char resolved_base[1024];
        mcp_resolve_path(custom_baseline_dir, resolved_base, sizeof(resolved_base));
        centroid_drift = compute_centroid_max_drift(resolved_base, tmp_outdir);
    }

    /* 5. Determine regression status and list issues */
    cJSON *regressions_arr = cJSON_CreateArray();

    if (cluster_delta != 0)
    {
        char msg[256];
        snprintf(
            msg, sizeof(msg),
            "Cluster count changed from %d to %d (delta: %d)",
            base_clusters, curr_state.num_clusters, cluster_delta);
        cJSON_AddItemToArray(regressions_arr, cJSON_CreateString(msg));
    }

    double dist_increase_pct = (dist_ratio - 1.0) * 100.0;
    if (dist_increase_pct > max_dist_drift_pct)
    {
        char msg[256];
        snprintf(
            msg, sizeof(msg),
            "Distance evaluations increased by %.2f%% (ratio: %.3f)",
            dist_increase_pct, dist_ratio);
        cJSON_AddItemToArray(regressions_arr, cJSON_CreateString(msg));
    }

    if (agreement_pct < min_agreement_pct)
    {
        char msg[256];
        snprintf(
            msg, sizeof(msg),
            "Assignment agreement %.2f%% below minimum threshold %.2f%%",
            agreement_pct, min_agreement_pct);
        cJSON_AddItemToArray(regressions_arr, cJSON_CreateString(msg));
    }

    if (centroid_drift > 1e-4)
    {
        char msg[256];
        snprintf(
            msg, sizeof(msg),
            "Centroid max drift %.6f exceeds tolerance 1e-4",
            centroid_drift);
        cJSON_AddItemToArray(regressions_arr, cJSON_CreateString(msg));
    }

    const char *status_str = "PASS";
    if (cJSON_GetArraySize(regressions_arr) > 0)
    {
        status_str = "REGRESSION";
    }
    else if (dist_increase_pct > 0.01)
    {
        status_str = "WARN";
    }

    /* 6. Populate result JSON */
    cJSON_AddStringToObject(res, "preset", preset_name);
    cJSON_AddStringToObject(res, "status", status_str);

    cJSON *metrics_obj = cJSON_CreateObject();

    cJSON *c_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(c_obj, "baseline", base_clusters);
    cJSON_AddNumberToObject(c_obj, "current", curr_state.num_clusters);
    cJSON_AddNumberToObject(c_obj, "delta", cluster_delta);
    cJSON_AddItemToObject(metrics_obj, "cluster_count", c_obj);

    cJSON *d_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(d_obj, "baseline", (double)base_dists);
    cJSON_AddNumberToObject(d_obj, "current", (double)curr_state.num_dists);
    cJSON_AddNumberToObject(d_obj, "ratio", dist_ratio);
    cJSON_AddItemToObject(metrics_obj, "dist_evals", d_obj);

    cJSON *t_obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(t_obj, "baseline_ms", base_time);
    cJSON_AddNumberToObject(t_obj, "current_ms", curr_state.time_clustering_ms);
    cJSON_AddNumberToObject(t_obj, "speedup", speedup);
    cJSON_AddItemToObject(metrics_obj, "elapsed_ms", t_obj);

    cJSON_AddNumberToObject(metrics_obj, "assignment_agreement_pct", agreement_pct);
    cJSON_AddNumberToObject(metrics_obj, "centroid_max_drift", centroid_drift);

    cJSON_AddItemToObject(res, "metrics", metrics_obj);
    cJSON_AddItemToObject(res, "regressions", regressions_arr);

    cleanup_directory(tmp_outdir);
    free_state(&base_state);
    free_state(&curr_state);

    return 0;
} // mcp_tool_dev_golden_compare

const struct mcp_tool_def mcp_tooldef_dev_golden_compare = {
    .name         = "gric_dev_golden_compare",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_dev_golden_compare,
    .description  = "Run rapid regression check comparing clustering outputs against golden "
                    "benchmarks (spiral, balls, gaussian64) or a baseline directory.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"preset\": {\n"
        "      \"type\": \"string\",\n"
        "      \"enum\": [\"spiral\", \"balls\", \"gaussian64\"],\n"
        "      \"description\": \"Standard golden regression preset (default: spiral).\"\n"
        "    },\n"
        "    \"dataset\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Custom dataset path (overrides preset dataset).\"\n"
        "    },\n"
        "    \"baseline_dir\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Directory with baseline cluster_run.log to compare against.\"\n"
        "    },\n"
        "    \"rlim\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Clustering radius limit (overrides preset rlim).\"\n"
        "    },\n"
        "    \"max_dist_drift_pct\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Allowed dist evals increase % (default: 5.0).\"\n"
        "    },\n"
        "    \"min_agreement_pct\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Minimum assignment agreement % (default: 99.0).\"\n"
        "    }\n"
        "  }\n"
        "}",
};

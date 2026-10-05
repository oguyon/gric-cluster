/**
 * @file tool_param_advisor.c
 * @brief Intelligent parameter advisor and radius calibration tools for MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "gric-probe/probe_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

/**
 * mcp_tool_recommend_params() - Recommends optimal clustering parameters.
 */
int mcp_tool_recommend_params(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *path_item = cJSON_GetObjectItemCaseSensitive(args, "dataset_path");
    if (path_item == NULL || !cJSON_IsString(path_item))
    {
        cJSON_AddStringToObject(res, "error", "dataset_path is required");
        return -1;
    }

    const char *dataset_path = path_item->valuestring;
    char resolved_path[512];
    mcp_resolve_path(dataset_path, resolved_path, sizeof(resolved_path));

    int target_clusters = 0;
    cJSON *tc_item = cJSON_GetObjectItemCaseSensitive(args, "target_clusters");
    if (tc_item != NULL && cJSON_IsNumber(tc_item))
    {
        target_clusters = tc_item->valueint;
    }

    double memory_budget_gb = 0.0;
    cJSON *mem_item = cJSON_GetObjectItemCaseSensitive(args, "memory_budget_gb");
    if (mem_item != NULL && cJSON_IsNumber(mem_item))
    {
        memory_budget_gb = mem_item->valuedouble;
    }

    const char *speed_pref = "balanced";
    cJSON *speed_item = cJSON_GetObjectItemCaseSensitive(args, "speed_preference");
    if (speed_item != NULL && cJSON_IsString(speed_item))
    {
        speed_pref = speed_item->valuestring;
    }

    /* Probe dataset structure and metrics */
    ProbeConfig config;
    memset(&config, 0, sizeof(ProbeConfig));
    snprintf(config.dataset_path, sizeof(config.dataset_path), "%s", resolved_path);
    config.sample_limit = 500;
    config.verbose_level = 0;

    ProbeResults results;
    memset(&results, 0, sizeof(ProbeResults));

    if (probe_run(&config, &results) != 0)
    {
        probe_results_free(&results);
        cJSON_AddStringToObject(res, "error", "probe_run failed on specified dataset");
        cJSON_AddStringToObject(res, "dataset_path", dataset_path);
        return -1;
    }

    const GricProfile *p = &results.profile;
    long num_frames = p->num_frames;
    long dim = p->dim;

    /* Detect system memory if not specified */
    if (memory_budget_gb <= 0.0)
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0)
        {
            double total_ram_gb = ((double)pages * (double)page_size) / (1024.0 * 1024.0 * 1024.0);
            memory_budget_gb = total_ram_gb * 0.75;
        }
        else
        {
            memory_budget_gb = 8.0;
        }
    }

    /* Calculate uncompressed dataset footprint */
    double elem_size = p->is_double ? 8.0 : 4.0;
    double uncomp_bytes = (double)num_frames * (double)dim * elem_size;
    double mem_budget_bytes = memory_budget_gb * 1024.0 * 1024.0 * 1024.0;

    /* Quantization selection */
    const char *quant = "none";
    const char *quant_flag = "";
    double quant_elem_size = elem_size;

    if (uncomp_bytes > mem_budget_bytes * 0.6)
    {
        /* High memory pressure */
        if (dim >= 64 && (uncomp_bytes > mem_budget_bytes * 2.0))
        {
            quant = "rabitq";
            quant_flag = "-rabitq";
            quant_elem_size = 0.125;
        }
        else if (dim >= 32)
        {
            quant = "sq8";
            quant_flag = "-sq8";
            quant_elem_size = 1.0;
        }
        else
        {
            quant = "sq16";
            quant_flag = "-sq16";
            quant_elem_size = 2.0;
        }
    }
    else
    {
        /* Fits in memory budget: select based on speed_preference */
        if (strcmp(speed_pref, "fast") == 0)
        {
            if (dim >= 64)
            {
                quant = "rabitq";
                quant_flag = "-rabitq";
                quant_elem_size = 0.125;
            }
            else if (dim >= 32)
            {
                quant = "sq8";
                quant_flag = "-sq8";
                quant_elem_size = 1.0;
            }
            else
            {
                quant = "none";
                quant_flag = "";
                quant_elem_size = elem_size;
            }
        }
        else if (strcmp(speed_pref, "accurate") == 0)
        {
            if (dim >= 32)
            {
                quant = "eq16";
                quant_flag = "-eq16";
                quant_elem_size = 2.0;
            }
            else
            {
                quant = "none";
                quant_flag = "";
                quant_elem_size = elem_size;
            }
        }
        else
        {
            /* Balanced */
            if (dim < 32)
            {
                quant = "none";
                quant_flag = "";
                quant_elem_size = elem_size;
            }
            else if (dim <= 128)
            {
                quant = "sq16";
                quant_flag = "-sq16";
                quant_elem_size = 2.0;
            }
            else
            {
                quant = "sq8";
                quant_flag = "-sq8";
                quant_elem_size = 1.0;
            }
        }
    } // if (uncomp_bytes > mem_budget_bytes * 0.6)

    /* Recommended radius and cluster count */
    double rec_rlim = p->rlim_recommended;
    int rec_maxcl = p->recommended_maxcl;

    if (target_clusters > 0)
    {
        rec_maxcl = (int)((double)target_clusters * 1.25);
        if (rec_maxcl < 50)
        {
            rec_maxcl = 50;
        }

        double ratio = (double)target_clusters / ((double)num_frames + 1.0);
        if (ratio >= 0.15)
        {
            rec_rlim = p->rlim_fine;
        }
        else if (ratio <= 0.02)
        {
            rec_rlim = p->rlim_coarse;
        }
        else
        {
            rec_rlim = p->rlim_balanced;
        }
    }

    /* CPU thread recommendation */
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    int rec_ncpu = 1;
    if (nprocs > 0)
    {
        if (num_frames < 500)
        {
            rec_ncpu = (nprocs > 2) ? 2 : (int)nprocs;
        }
        else if (num_frames < 5000)
        {
            rec_ncpu = (nprocs > 8) ? 8 : (int)nprocs;
        }
        else
        {
            rec_ncpu = (nprocs > 16) ? 16 : (int)nprocs;
        }
    }

    /* Tiling recommendation */
    int tiles_x = 1;
    int tiles_y = 1;
    if (p->is_image && p->width >= 64 && p->height >= 64 && dim >= 1024)
    {
        tiles_x = 2;
        tiles_y = 2;
    }

    /* Memory and disk estimates */
    double quant_bytes = (double)num_frames * (double)dim * quant_elem_size;
    double centers_bytes = (double)rec_maxcl * (double)dim * 4.0;
    double dcc_bytes = (double)rec_maxcl * (double)rec_maxcl * 4.0 * 0.2;
    double est_rss_mb = (quant_bytes + centers_bytes + dcc_bytes) * 1.2 / (1024.0 * 1024.0);
    double disk_mb = (centers_bytes + (double)num_frames * 4.0) / (1024.0 * 1024.0);

    /* Construct JSON response */
    cJSON *ds_info = cJSON_CreateObject();
    cJSON_AddStringToObject(ds_info, "path", dataset_path);
    cJSON_AddNumberToObject(ds_info, "num_frames", (double)num_frames);
    cJSON_AddNumberToObject(ds_info, "dim", (double)dim);
    cJSON_AddNumberToObject(ds_info, "width", (double)p->width);
    cJSON_AddNumberToObject(ds_info, "height", (double)p->height);
    cJSON_AddBoolToObject(ds_info, "is_image", p->is_image);
    cJSON_AddItemToObject(res, "dataset_info", ds_info);

    cJSON *recs = cJSON_CreateObject();
    cJSON_AddNumberToObject(recs, "rlim", rec_rlim);
    cJSON_AddStringToObject(recs, "quantization", quant);
    cJSON_AddStringToObject(recs, "quantization_flag", quant_flag);
    cJSON_AddNumberToObject(recs, "maxcl", rec_maxcl);
    cJSON_AddNumberToObject(recs, "ncpu", rec_ncpu);

    cJSON *tiling = cJSON_CreateObject();
    cJSON_AddNumberToObject(tiling, "tiles_x", tiles_x);
    cJSON_AddNumberToObject(tiling, "tiles_y", tiles_y);
    cJSON_AddItemToObject(recs, "tiling", tiling);

    /* Recommended command line */
    char cmd[512];
    char tile_opt[64] = "";
    if (tiles_x > 1 || tiles_y > 1)
    {
        snprintf(tile_opt, sizeof(tile_opt), " -tiles %dx%d", tiles_x, tiles_y);
    }
    snprintf(
        cmd, sizeof(cmd),
        "gric-cluster %.4f %s -maxcl %d -ncpu %d%s%s%s%s",
        rec_rlim, dataset_path, rec_maxcl, rec_ncpu,
        (strlen(quant_flag) > 0) ? " " : "",
        quant_flag,
        tile_opt,
        p->pred_enabled ? " -entropy" : "");
    cJSON_AddStringToObject(recs, "cli_command", cmd);
    cJSON_AddItemToObject(res, "recommendations", recs);

    cJSON *res_est = cJSON_CreateObject();
    cJSON_AddNumberToObject(res_est, "uncompressed_mb", uncomp_bytes / (1024.0 * 1024.0));
    cJSON_AddNumberToObject(res_est, "quantized_mb", quant_bytes / (1024.0 * 1024.0));
    cJSON_AddNumberToObject(res_est, "estimated_rss_mb", est_rss_mb);
    cJSON_AddNumberToObject(res_est, "disk_footprint_mb", disk_mb);
    cJSON_AddItemToObject(res, "resource_estimates", res_est);

    /* Rationale explanation */
    char reasoning[1024];
    snprintf(
        reasoning, sizeof(reasoning),
        "### Parameter Advisor Rationale\n"
        "- **Radius (rlim = %.4f)**: Calibrated based on pairwise distance percentiles "
        "for %ld frames in %ld dimensions%s.\n"
        "- **Quantization (%s)**: Selected for speed preference '%s' with %.1f GB memory budget. "
        "Reduces memory from %.1f MB to %.1f MB.\n"
        "- **Concurrency (-ncpu %d)**: Optimized for available CPU cores without "
        "excessive thread scheduling overhead.\n"
        "- **Max Clusters (-maxcl %d)**: Sized to comfortably bound cluster allocation.\n"
        "- **Prediction / Entropy**: %s based on temporal autocorrelation ratio (%.3f).",
        rec_rlim, num_frames, dim,
        (target_clusters > 0) ? " to target cluster count" : "",
        quant, speed_pref, memory_budget_gb,
        uncomp_bytes / (1024.0 * 1024.0), quant_bytes / (1024.0 * 1024.0),
        rec_ncpu, rec_maxcl,
        p->pred_enabled ? "Enabled (-entropy)" : "Disabled",
        p->continuity_ratio);
    cJSON_AddStringToObject(res, "reasoning", reasoning);

    probe_results_free(&results);
    return 0;
} // mcp_tool_recommend_params

/**
 * mcp_tool_calibrate_radius() - Calibrates radius thresholds from distance percentiles.
 */
int mcp_tool_calibrate_radius(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *path_item = cJSON_GetObjectItemCaseSensitive(args, "dataset_path");
    if (path_item == NULL || !cJSON_IsString(path_item))
    {
        cJSON_AddStringToObject(res, "error", "dataset_path is required");
        return -1;
    }

    const char *dataset_path = path_item->valuestring;
    char resolved_path[512];
    mcp_resolve_path(dataset_path, resolved_path, sizeof(resolved_path));

    int sample_size = 200;
    cJSON *sz_item = cJSON_GetObjectItemCaseSensitive(args, "sample_size");
    if (sz_item != NULL && cJSON_IsNumber(sz_item))
    {
        sample_size = sz_item->valueint;
        if (sample_size < 10)
        {
            sample_size = 10;
        }
        if (sample_size > 1000)
        {
            sample_size = 1000;
        }
    }

    ProbeConfig config;
    memset(&config, 0, sizeof(ProbeConfig));
    snprintf(config.dataset_path, sizeof(config.dataset_path), "%s", resolved_path);
    config.sample_limit = sample_size;
    config.verbose_level = 0;

    ProbeResults results;
    memset(&results, 0, sizeof(ProbeResults));

    if (probe_run(&config, &results) != 0)
    {
        probe_results_free(&results);
        cJSON_AddStringToObject(res, "error", "probe_run failed on specified dataset");
        cJSON_AddStringToObject(res, "dataset_path", dataset_path);
        return -1;
    }

    const GricProfile *p = &results.profile;

    /* Interpolate extra percentiles (p15 and p35) */
    double p15 = p->dist_p10 + (5.0 / 15.0) * (p->dist_p25 - p->dist_p10);
    double p35 = p->dist_p25 + (10.0 / 25.0) * (p->dist_p50 - p->dist_p25);

    cJSON *ds_info = cJSON_CreateObject();
    cJSON_AddStringToObject(ds_info, "path", dataset_path);
    cJSON_AddNumberToObject(ds_info, "num_frames", (double)p->num_frames);
    cJSON_AddNumberToObject(ds_info, "dim", (double)p->dim);
    cJSON_AddNumberToObject(ds_info, "sample_frames", (double)sample_size);
    cJSON_AddItemToObject(res, "dataset_info", ds_info);

    cJSON *pcts = cJSON_CreateObject();
    cJSON_AddNumberToObject(pcts, "min", p->dist_min);
    cJSON_AddNumberToObject(pcts, "p01", p->dist_p01);
    cJSON_AddNumberToObject(pcts, "p05", p->dist_p05);
    cJSON_AddNumberToObject(pcts, "p10", p->dist_p10);
    cJSON_AddNumberToObject(pcts, "p15", p15);
    cJSON_AddNumberToObject(pcts, "p25", p->dist_p25);
    cJSON_AddNumberToObject(pcts, "p35", p35);
    cJSON_AddNumberToObject(pcts, "p50", p->dist_p50);
    cJSON_AddNumberToObject(pcts, "p75", p->dist_p75);
    cJSON_AddNumberToObject(pcts, "p90", p->dist_p90);
    cJSON_AddNumberToObject(pcts, "max", p->dist_max);
    cJSON_AddItemToObject(res, "percentiles", pcts);

    cJSON *recs = cJSON_CreateObject();
    cJSON_AddNumberToObject(recs, "fine", p->rlim_fine);
    cJSON_AddNumberToObject(recs, "balanced", p->rlim_balanced);
    cJSON_AddNumberToObject(recs, "coarse", p->rlim_coarse);
    cJSON_AddItemToObject(res, "recommendations", recs);

    char analysis[1024];
    snprintf(
        analysis, sizeof(analysis),
        "### Pairwise Distance Spectrum Calibration\n"
        "- **Measured Distance Range**: [%.4f, %.4f] (Median: %.4f)\n"
        "- **Fine Radius (p05 = %.4f)**: Use for fine-grained sub-structure segmentation.\n"
        "- **Balanced Radius (p10 = %.4f)**: Recommended starting point for general clustering.\n"
        "- **Coarse Radius (p25 = %.4f)**: Use for coarse grouping and broad phase-space basins.\n"
        "- **Noise Floor Estimate**: %.4f",
        p->dist_min, p->dist_max, p->dist_p50,
        p->rlim_fine, p->rlim_balanced, p->rlim_coarse,
        results.noise_floor_est);
    cJSON_AddStringToObject(res, "analysis", analysis);

    probe_results_free(&results);
    return 0;
} // mcp_tool_calibrate_radius

const struct mcp_tool_def mcp_tooldef_recommend_params = {
    .name         = "gric_recommend_params",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_recommend_params,
    .description  = "Recommend optimal clustering parameters (rlim, quantization, tiling, "
                    "ncpu, maxcl) based on dataset geometry, memory constraints, and speed goals.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"dataset_path\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to dataset coordinates or image cube (FITS, BIN, ASCII).\"\n"
        "    },\n"
        "    \"target_clusters\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Desired approximate number of clusters (optional).\"\n"
        "    },\n"
        "    \"memory_budget_gb\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Memory budget in GB (optional, defaults to system RAM).\"\n"
        "    },\n"
        "    \"speed_preference\": {\n"
        "      \"type\": \"string\",\n"
        "      \"enum\": [\"fast\", \"balanced\", \"accurate\"],\n"
        "      \"description\": \"Performance tradeoff (default: balanced).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"dataset_path\"]\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_calibrate_radius = {
    .name         = "gric_calibrate_radius",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_calibrate_radius,
    .description  = "Subsample dataset frames, compute pairwise distance percentiles, "
                    "and recommend fine, balanced, and coarse clustering radii.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"dataset_path\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to dataset coordinates or image cube (FITS, BIN, ASCII).\"\n"
        "    },\n"
        "    \"sample_size\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Sample size for pairwise distance estimation (10-1000).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"dataset_path\"]\n"
        "}",
};

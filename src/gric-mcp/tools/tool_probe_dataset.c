/**
 * @file tool_probe_dataset.c
 * @brief Manifold geometry probe and hyperparameter tuning recommendation tool.
 */

#include "mcp_tools.h"
#include "gric-probe/probe_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mcp_tool_probe_dataset(
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

    int sample_limit = 0;
    cJSON *sample_item = cJSON_GetObjectItemCaseSensitive(args, "sample_limit");
    if (sample_item != NULL && cJSON_IsNumber(sample_item))
    {
        sample_limit = sample_item->valueint;
    }

    ProbeConfig config;
    memset(&config, 0, sizeof(ProbeConfig));
    snprintf(config.dataset_path, sizeof(config.dataset_path), "%s", resolved_path);
    config.sample_limit = sample_limit;
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

    cJSON_AddStringToObject(res, "dataset_path", p->dataset_path);
    cJSON_AddNumberToObject(res, "num_frames", (double)p->num_frames);
    cJSON_AddNumberToObject(res, "dimension", (double)p->dim);
    cJSON_AddNumberToObject(res, "width", (double)p->width);
    cJSON_AddNumberToObject(res, "height", (double)p->height);
    cJSON_AddBoolToObject(res, "is_image", p->is_image);

    /* Radii calibration presets */
    cJSON *radii = cJSON_CreateObject();
    cJSON_AddNumberToObject(radii, "fine_rlim", p->rlim_fine);
    cJSON_AddNumberToObject(radii, "balanced_rlim", p->rlim_balanced);
    cJSON_AddNumberToObject(radii, "coarse_rlim", p->rlim_coarse);
    cJSON_AddNumberToObject(radii, "recommended_rlim", p->rlim_recommended);
    cJSON_AddItemToObject(res, "radii_presets", radii);

    cJSON_AddNumberToObject(res, "recommended_maxcl", p->recommended_maxcl);
    cJSON_AddNumberToObject(res, "noise_floor_estimate", results.noise_floor_est);
    cJSON_AddNumberToObject(res, "dead_dims_count", results.dead_dims_count);

    /* Recommended CLI execution flags */
    char flags[256];
    snprintf(
        flags, sizeof(flags),
        "%.4f %s -maxcl %d%s%s",
        p->rlim_recommended,
        dataset_path,
        p->recommended_maxcl,
        (p->dim <= 32) ? " -te4" : " -sq16",
        (p->pred_enabled) ? " -entropy" : "");
    cJSON_AddStringToObject(res, "recommended_cli_args", flags);

    probe_results_free(&results);
    return 0;
} // mcp_tool_probe_dataset

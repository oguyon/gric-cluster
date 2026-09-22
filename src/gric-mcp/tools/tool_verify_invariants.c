/**
 * @file tool_verify_invariants.c
 * @brief Algorithmic and mathematical invariant verifier for cluster boundaries.
 */

#include "mcp_tools.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON_TOLERANCE 1e-5

int mcp_tool_verify_invariants(
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
        cJSON_AddStringToObject(res, "error", "run_dir is required");
        return -1;
    }
    const char *run_dir = dir_item->valuestring;
    char resolved_dir[1024];
    mcp_resolve_path(run_dir, resolved_dir, sizeof(resolved_dir));

    double rlim = -1.0;
    cJSON *rlim_item = cJSON_GetObjectItemCaseSensitive(args, "rlim");
    if (rlim_item != NULL && cJSON_IsNumber(rlim_item))
    {
        rlim = rlim_item->valuedouble;
    }
    else
    {
        /* Parse rlim from cluster_run.log */
        char log_path[2048];
        snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", resolved_dir);
        FILE *lfp = fopen(log_path, "r");
        if (lfp != NULL)
        {
            char line[1024];
            while (fgets(line, sizeof(line), lfp) != NULL)
            {
                if (strncmp(line, "PARAM_RLIM: ", 12) == 0)
                {
                    rlim = atof(line + 12);
                    break;
                }
            } // while fgets
            fclose(lfp);
        }
    }

    if (rlim <= 0.0)
    {
        cJSON_AddStringToObject(res, "error", "rlim could not be determined from arguments or log");
        return -1;
    }

    char mem_path[2048];
    snprintf(mem_path, sizeof(mem_path), "%s/frame_membership.txt", resolved_dir);
    FILE *mfp = fopen(mem_path, "r");
    if (mfp == NULL)
    {
        cJSON_AddStringToObject(res, "error", "frame_membership.txt not found in run_dir");
        cJSON_AddStringToObject(res, "path", mem_path);
        return -1;
    }

    long total_frames = 0;
    long violations_count = 0;
    double max_dist_observed = 0.0;
    cJSON *violations_arr = cJSON_CreateArray();

    char line[1024];
    while (fgets(line, sizeof(line), mfp) != NULL)
    {
        long f_idx = -1;
        int c_idx = -1;
        double dist = -1.0;

        int fields = sscanf(line, "%ld %d %lf", &f_idx, &c_idx, &dist);
        if (fields >= 2)
        {
            total_frames++;
            if (fields >= 3 && dist >= 0.0)
            {
                if (dist > max_dist_observed)
                {
                    max_dist_observed = dist;
                }
                if (dist > (rlim + EPSILON_TOLERANCE))
                {
                    violations_count++;
                    if (violations_count <= 20)
                    {
                        cJSON *v = cJSON_CreateObject();
                        cJSON_AddNumberToObject(v, "frame", (double)f_idx);
                        cJSON_AddNumberToObject(v, "cluster", c_idx);
                        cJSON_AddNumberToObject(v, "distance", dist);
                        cJSON_AddNumberToObject(v, "excess", dist - rlim);
                        cJSON_AddItemToArray(violations_arr, v);
                    }
                }
            }
        }
    } // while fgets
    fclose(mfp);

    cJSON_AddStringToObject(res, "run_directory", run_dir);
    cJSON_AddNumberToObject(res, "rlim_threshold", rlim);
    cJSON_AddNumberToObject(res, "total_frames_audited", (double)total_frames);
    cJSON_AddNumberToObject(res, "max_distance_observed", max_dist_observed);
    cJSON_AddNumberToObject(res, "violations_count", (double)violations_count);
    cJSON_AddStringToObject(res, "status", (violations_count == 0) ? "PASS" : "VIOLATIONS_FOUND");
    cJSON_AddItemToObject(res, "violations_sample", violations_arr);

    return 0;
} // mcp_tool_verify_invariants

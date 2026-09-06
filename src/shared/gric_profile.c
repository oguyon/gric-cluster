/**
 * @file gric_profile.c
 * @brief Implementation of dataset profile serialization and discovery.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#include "gric_profile.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * gric_profile_init() - Initialize a GricProfile with safe defaults.
 * @prof: Pointer to GricProfile struct.
 * @dim:  Vector dimension count (width * height).
 */
void gric_profile_init(
    GricProfile *prof,
    long         dim)
{
    if (prof == NULL)
    {
        return;
    }

    memset(prof, 0, sizeof(GricProfile));
    prof->dim = dim;
    prof->rlim_recommended = 0.1;
    prof->rlim_fine = 0.05;
    prof->rlim_balanced = 0.1;
    prof->rlim_coarse = 0.2;
    strcpy(prof->preset_name, "balanced");
    prof->recommended_maxcl = 2000;
    prof->tiles_x = 1;
    prof->tiles_y = 1;
    prof->pred_len = 2;
    prof->pred_h = 1000;
    prof->ncpu = 4;
    prof->te3_prune_rate = 0.0;
    prof->te4_marginal_rate = 0.0;
    prof->te5_marginal_rate = 0.0;
    strcpy(prof->recommended_prune_mode, "3P");

    if (dim > 0)
    {
        prof->perm_dim = (long *)calloc((size_t)dim, sizeof(long));
        prof->var_dim = (double *)calloc((size_t)dim, sizeof(double));
        prof->residual_tail = (double *)calloc((size_t)dim, sizeof(double));

        for (long ii = 0; ii < dim; ii++)
        {
            prof->perm_dim[ii] = ii;
        } // for (long ii = 0; ii < dim; ii++)
    } // if (dim > 0)
}

/**
 * gric_profile_free() - Free allocated arrays in a GricProfile.
 * @prof: Pointer to GricProfile struct.
 */
void gric_profile_free(
    GricProfile *prof)
{
    if (prof == NULL)
    {
        return;
    }

    if (prof->perm_dim != NULL)
    {
        free(prof->perm_dim);
        prof->perm_dim = NULL;
    }

    if (prof->var_dim != NULL)
    {
        free(prof->var_dim);
        prof->var_dim = NULL;
    }

    if (prof->residual_tail != NULL)
    {
        free(prof->residual_tail);
        prof->residual_tail = NULL;
    }
}

/**
 * gric_profile_write_json() - Serialize profile to a JSON file.
 * @filepath: Destination file path.
 * @prof:     Pointer to GricProfile struct.
 *
 * Return: 0 on success, -1 on I/O error.
 */
int gric_profile_write_json(
    const char        *filepath,
    const GricProfile *prof)
{
    if (filepath == NULL || prof == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "w");
    if (fp == NULL)
    {
        return -1;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"magic\": \"%s\",\n", GRIC_PROFILE_MAGIC);
    fprintf(fp, "  \"dataset\": {\n");
    fprintf(fp, "    \"path\": \"%s\",\n", prof->dataset_path);
    fprintf(fp, "    \"num_frames\": %ld,\n", prof->num_frames);
    fprintf(fp, "    \"width\": %ld,\n", prof->width);
    fprintf(fp, "    \"height\": %ld,\n", prof->height);
    fprintf(fp, "    \"dim\": %ld,\n", prof->dim);
    fprintf(fp, "    \"is_image\": %s,\n", prof->is_image ? "true" : "false");
    fprintf(fp, "    \"is_double\": %s\n", prof->is_double ? "true" : "false");
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"clustering\": {\n");
    fprintf(fp, "    \"rlim_recommended\": %.6f,\n", prof->rlim_recommended);
    fprintf(fp, "    \"rlim_presets\": {\n");
    fprintf(fp, "      \"fine\": %.6f,\n", prof->rlim_fine);
    fprintf(fp, "      \"balanced\": %.6f,\n", prof->rlim_balanced);
    fprintf(fp, "      \"coarse\": %.6f\n", prof->rlim_coarse);
    fprintf(fp, "    },\n");
    fprintf(fp, "    \"recommended_maxcl\": %d,\n", prof->recommended_maxcl);
    fprintf(fp, "    \"tiles_x\": %d,\n", prof->tiles_x);
    fprintf(fp, "    \"tiles_y\": %d\n", prof->tiles_y);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"prediction\": {\n");
    fprintf(fp, "    \"enabled\": %s,\n", prof->pred_enabled ? "true" : "false");
    fprintf(fp, "    \"pred_len\": %d,\n", prof->pred_len);
    fprintf(fp, "    \"pred_h\": %d,\n", prof->pred_h);
    fprintf(fp, "    \"continuity_ratio\": %.6f\n", prof->continuity_ratio);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"acceleration\": {\n");
    fprintf(fp, "    \"use_sq8\": %s,\n", prof->use_sq8 ? "true" : "false");
    fprintf(fp, "    \"sq8_min\": %.6f,\n", (double)prof->sq8_params.min_val);
    fprintf(fp, "    \"sq8_max\": %.6f,\n", (double)prof->sq8_params.max_val);
    fprintf(fp, "    \"sq8_scale\": %.6f,\n", (double)prof->sq8_params.scale);
    fprintf(fp, "    \"sq8_err_radius\": %.6f,\n", (double)prof->sq8_params.err_radius);
    fprintf(fp, "    \"te4\": %s,\n", prof->te4_enabled ? "true" : "false");
    fprintf(fp, "    \"te5\": %s,\n", prof->te5_enabled ? "true" : "false");
    fprintf(fp, "    \"ncpu\": %d\n", prof->ncpu);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"pruning\": {\n");
    fprintf(fp, "    \"te3_prune_rate\": %.4f,\n", prof->te3_prune_rate);
    fprintf(fp, "    \"te4_marginal_rate\": %.4f,\n", prof->te4_marginal_rate);
    fprintf(fp, "    \"te5_marginal_rate\": %.4f,\n", prof->te5_marginal_rate);
    fprintf(fp, "    \"recommended_mode\": \"%s\"\n",
            prof->recommended_prune_mode[0] ? prof->recommended_prune_mode : "3P");
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"distance_spectrum\": {\n");
    fprintf(fp, "    \"min\": %.6f,\n", prof->dist_min);
    fprintf(fp, "    \"p01\": %.6f,\n", prof->dist_p01);
    fprintf(fp, "    \"p05\": %.6f,\n", prof->dist_p05);
    fprintf(fp, "    \"p10\": %.6f,\n", prof->dist_p10);
    fprintf(fp, "    \"p25\": %.6f,\n", prof->dist_p25);
    fprintf(fp, "    \"p50\": %.6f,\n", prof->dist_p50);
    fprintf(fp, "    \"p75\": %.6f,\n", prof->dist_p75);
    fprintf(fp, "    \"p90\": %.6f,\n", prof->dist_p90);
    fprintf(fp, "    \"max\": %.6f\n", prof->dist_max);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"spectral\": {\n");
    fprintf(fp, "    \"variance_ordering\": [");
    if (prof->perm_dim != NULL)
    {
        for (long ii = 0; ii < prof->dim; ii++)
        {
            fprintf(fp, "%ld%s", prof->perm_dim[ii], (ii < prof->dim - 1) ? ", " : "");
        }
    }
    fprintf(fp, "],\n");

    fprintf(fp, "    \"residual_tail\": [");
    if (prof->residual_tail != NULL)
    {
        for (long ii = 0; ii < prof->dim; ii++)
        {
            fprintf(fp, "%.6f%s", prof->residual_tail[ii], (ii < prof->dim - 1) ? ", " : "");
        }
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return 0;
}

/**
 * find_json_key() - Helper searching for a key in a JSON buffer.
 */
static const char *find_json_key(
    const char *buffer,
    const char *key)
{
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *pos = strstr(buffer, needle);
    if (pos == NULL)
    {
        return NULL;
    }

    pos += strlen(needle);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r' || *pos == ':')
    {
        pos++;
    }

    return pos;
}

/**
 * parse_json_double() - Helper extracting double from JSON string.
 */
static double parse_json_double(
    const char *buffer,
    const char *key,
    double      fallback)
{
    const char *val = find_json_key(buffer, key);
    if (val == NULL)
    {
        return fallback;
    }

    return strtod(val, NULL);
}

/**
 * parse_json_long() - Helper extracting long from JSON string.
 */
static long parse_json_long(
    const char *buffer,
    const char *key,
    long        fallback)
{
    const char *val = find_json_key(buffer, key);
    if (val == NULL)
    {
        return fallback;
    }

    return strtol(val, NULL, 10);
}

/**
 * parse_json_bool() - Helper extracting boolean from JSON string.
 */
static int parse_json_bool(
    const char *buffer,
    const char *key,
    int         fallback)
{
    const char *val = find_json_key(buffer, key);
    if (val == NULL)
    {
        return fallback;
    }

    if (strncmp(val, "true", 4) == 0)
    {
        return 1;
    }
    if (strncmp(val, "false", 5) == 0)
    {
        return 0;
    }

    return fallback;
}

/**
 * parse_json_string() - Helper extracting string from JSON.
 */
static void parse_json_string(
    const char *buffer,
    const char *key,
    char       *dst,
    size_t      dst_size)
{
    const char *val = find_json_key(buffer, key);
    if (val == NULL || *val != '"')
    {
        return;
    }

    val++;
    const char *end = strchr(val, '"');
    if (end == NULL)
    {
        return;
    }

    size_t len = (size_t)(end - val);
    if (len >= dst_size)
    {
        len = dst_size - 1;
    }

    memcpy(dst, val, len);
    dst[len] = '\0';
}

/**
 * gric_profile_read_json() - Read profile from a JSON file.
 * @filepath: Path to .gricprof file.
 * @prof:     Pointer to GricProfile struct to populate.
 *
 * Return: 0 on success, -1 on I/O error or invalid format.
 */
int gric_profile_read_json(
    const char  *filepath,
    GricProfile *prof)
{
    if (filepath == NULL || prof == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100 * 1024 * 1024)
    {
        fclose(fp);
        return -1;
    }

    char *buf = (char *)malloc((size_t)file_size + 1);
    if (buf == NULL)
    {
        fclose(fp);
        return -1;
    }

    size_t read_bytes = fread(buf, 1, (size_t)file_size, fp);
    fclose(fp);
    buf[read_bytes] = '\0';

    long dim = parse_json_long(buf, "dim", 0);
    gric_profile_init(prof, dim);

    parse_json_string(buf, "path", prof->dataset_path, sizeof(prof->dataset_path));
    prof->num_frames = parse_json_long(buf, "num_frames", 0);
    prof->width = parse_json_long(buf, "width", dim);
    prof->height = parse_json_long(buf, "height", 1);
    prof->is_image = parse_json_bool(buf, "is_image", 0);
    prof->is_double = parse_json_bool(buf, "is_double", 0);

    prof->rlim_recommended = parse_json_double(buf, "rlim_recommended", 0.1);
    prof->rlim_fine = parse_json_double(buf, "fine", prof->rlim_recommended * 0.5);
    prof->rlim_balanced = parse_json_double(buf, "balanced", prof->rlim_recommended);
    prof->rlim_coarse = parse_json_double(buf, "coarse", prof->rlim_recommended * 2.0);
    prof->recommended_maxcl = (int)parse_json_long(buf, "recommended_maxcl", 2000);
    prof->tiles_x = (int)parse_json_long(buf, "tiles_x", 1);
    prof->tiles_y = (int)parse_json_long(buf, "tiles_y", 1);

    prof->pred_enabled = parse_json_bool(buf, "enabled", 0);
    prof->pred_len = (int)parse_json_long(buf, "pred_len", 2);
    prof->pred_h = (int)parse_json_long(buf, "pred_h", 1000);
    prof->continuity_ratio = parse_json_double(buf, "continuity_ratio", 1.0);

    prof->use_sq8 = parse_json_bool(buf, "use_sq8", 0);
    prof->sq8_params.min_val = (float)parse_json_double(buf, "sq8_min", 0.0);
    prof->sq8_params.max_val = (float)parse_json_double(buf, "sq8_max", 1.0);
    prof->sq8_params.scale = (float)parse_json_double(buf, "sq8_scale", 1.0f / 255.0f);
    prof->sq8_params.err_radius = (float)parse_json_double(buf, "sq8_err_radius", 0.0);
    prof->sq8_params.dim = dim;
    if (prof->sq8_params.scale > 0.0f)
    {
        prof->sq8_params.inv_scale = 1.0f / prof->sq8_params.scale;
    }

    prof->te4_enabled = parse_json_bool(buf, "te4", 0);
    prof->te5_enabled = parse_json_bool(buf, "te5", 0);
    prof->ncpu = (int)parse_json_long(buf, "ncpu", 4);

    prof->te3_prune_rate = parse_json_double(buf, "te3_prune_rate", 0.0);
    prof->te4_marginal_rate = parse_json_double(buf, "te4_marginal_rate", 0.0);
    prof->te5_marginal_rate = parse_json_double(buf, "te5_marginal_rate", 0.0);
    parse_json_string(buf, "recommended_mode", prof->recommended_prune_mode,
                      sizeof(prof->recommended_prune_mode));
    if (prof->recommended_prune_mode[0] == '\0')
    {
        strcpy(prof->recommended_prune_mode, prof->te4_enabled ? "4P" : "3P");
    }

    prof->dist_min = parse_json_double(buf, "min", 0.0);
    prof->dist_p01 = parse_json_double(buf, "p01", 0.0);
    prof->dist_p05 = parse_json_double(buf, "p05", 0.0);
    prof->dist_p10 = parse_json_double(buf, "p10", 0.0);
    prof->dist_p25 = parse_json_double(buf, "p25", 0.0);
    prof->dist_p50 = parse_json_double(buf, "p50", 0.0);
    prof->dist_p75 = parse_json_double(buf, "p75", 0.0);
    prof->dist_p90 = parse_json_double(buf, "p90", 0.0);
    prof->dist_max = parse_json_double(buf, "max", 0.0);

    /* Parse variance_ordering array */
    const char *vo_pos = find_json_key(buf, "variance_ordering");
    if (vo_pos != NULL && prof->perm_dim != NULL)
    {
        const char *cur = strchr(vo_pos, '[');
        if (cur != NULL)
        {
            cur++;
            for (long ii = 0; ii < dim && *cur && *cur != ']'; ii++)
            {
                while (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r') cur++;
                if (*cur == ']') break;
                char *next = NULL;
                prof->perm_dim[ii] = strtol(cur, &next, 10);
                if (next == cur) break;
                cur = next;
                while (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r') cur++;
                if (*cur == ',') cur++;
            }
        }
    }

    /* Parse residual_tail array */
    const char *rt_pos = find_json_key(buf, "residual_tail");
    if (rt_pos != NULL && prof->residual_tail != NULL)
    {
        const char *cur = strchr(rt_pos, '[');
        if (cur != NULL)
        {
            cur++;
            for (long ii = 0; ii < dim && *cur && *cur != ']'; ii++)
            {
                while (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r') cur++;
                if (*cur == ']') break;
                char *next = NULL;
                prof->residual_tail[ii] = strtod(cur, &next);
                if (next == cur) break;
                cur = next;
                while (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r') cur++;
                if (*cur == ',') cur++;
            }
        }
    }

    free(buf);
    return 0;
}

/**
 * gric_profile_find_auto() - Auto-locate companion .gricprof file.
 * @dataset_path: Path to dataset.
 * @out_prof_path: Buffer to receive located profile path.
 * @max_len:       Maximum length of @out_prof_path buffer.
 *
 * Checks <dataset_path>.gricprof and <stem>.gricprof.
 *
 * Return: 1 if found and readable, 0 if not found.
 */
int gric_profile_find_auto(
    const char *dataset_path,
    char       *out_prof_path,
    size_t      max_len)
{
    if (dataset_path == NULL || out_prof_path == NULL || max_len == 0)
    {
        return 0;
    }

    /* Candidate 1: <dataset_path>.gricprof */
    char candidate1[1024];
    snprintf(candidate1, sizeof(candidate1), "%s.gricprof", dataset_path);
    if (access(candidate1, R_OK) == 0)
    {
        snprintf(out_prof_path, max_len, "%s", candidate1);
        return 1;
    }

    /* Candidate 2: Strip final extension if any, then append .gricprof */
    char candidate2[1024];
    strncpy(candidate2, dataset_path, sizeof(candidate2) - 1);
    candidate2[sizeof(candidate2) - 1] = '\0';
    char *dot = strrchr(candidate2, '.');
    if (dot != NULL && dot != candidate2)
    {
        *dot = '\0';
        strncat(candidate2, ".gricprof", sizeof(candidate2) - strlen(candidate2) - 1);
        if (access(candidate2, R_OK) == 0)
        {
            snprintf(out_prof_path, max_len, "%s", candidate2);
            return 1;
        }
    }

    return 0;
}

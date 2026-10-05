/**
 * @file tool_knn_quality.c
 * @brief kNN search recall and distance approximation quality evaluation tool for MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "format/gric_bin_io.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * Neighbor candidate record.
 */
typedef struct
{
    long   index;
    double dist;
} KnnNeighbor;

/**
 * Parsed kNN query result from output file.
 */
typedef struct
{
    long         query_id;
    int          num_neighbors;
    KnnNeighbor *neighbors;
} KnnQueryResult;

/**
 * load_dataset_vectors() - Load dataset coordinates from ASCII or binary file.
 */
static float *load_dataset_vectors(
    const char *path,
    long       *out_num_frames,
    int        *out_dim)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    char magic[4];
    if (fread(magic, 1, 4, fp) == 4 && memcmp(magic, GRIC_BIN_MAGIC, 4) == 0)
    {
        rewind(fp);
        gric_bin_header_t hdr;
        if (gric_bin_read_header(fp, &hdr, NULL) != 0)
        {
            fclose(fp);
            return NULL;
        }

        long nframes = (long)hdr.dims[0];
        int dim = (hdr.ndim > 1) ? (int)hdr.dims[1] : 1;
        if (hdr.ndim > 2)
        {
            dim *= (int)hdr.dims[2];
        }
        size_t total = (size_t)hdr.num_elements;

        float *data = (float *)malloc(total * sizeof(float));
        if (data == NULL)
        {
            fclose(fp);
            return NULL;
        }

        if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
        {
            if (fread(data, sizeof(float), total, fp) != total)
            {
                free(data);
                fclose(fp);
                return NULL;
            }
        }
        else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
        {
            double *d_data = (double *)malloc(total * sizeof(double));
            if (d_data == NULL || fread(d_data, sizeof(double), total, fp) != total)
            {
                free(d_data);
                free(data);
                fclose(fp);
                return NULL;
            }
            for (size_t ii = 0; ii < total; ii++)
            {
                data[ii] = (float)d_data[ii];
            }
            free(d_data);
        }

        fclose(fp);
        *out_num_frames = nframes;
        *out_dim = dim;
        return data;
    } // if binary

    /* Fallback: ASCII coordinate parser */
    rewind(fp);
    char line[65536];
    int dim = 0;
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0')
        {
            continue;
        }
        while (*p != '\0')
        {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\0' || *p == '\n' || *p == '\r')
            {
                break;
            }
            dim++;
            while (*p != ' ' && *p != '\t' && *p != '\0' && *p != '\n' && *p != '\r')
            {
                p++;
            }
        }
        break;
    } // while

    if (dim <= 0)
    {
        fclose(fp);
        return NULL;
    }

    rewind(fp);
    size_t cap = 1000;
    long nframes = 0;
    float *data = (float *)malloc(cap * (size_t)dim * sizeof(float));
    if (data == NULL)
    {
        fclose(fp);
        return NULL;
    }

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0')
        {
            continue;
        }

        if ((size_t)nframes >= cap)
        {
            cap *= 2;
            float *new_data = (float *)realloc(data, cap * (size_t)dim * sizeof(float));
            if (new_data == NULL)
            {
                free(data);
                fclose(fp);
                return NULL;
            }
            data = new_data;
        }

        float *dest = data + nframes * dim;
        int d = 0;
        while (*p != '\0' && d < dim)
        {
            char *end = NULL;
            dest[d] = strtof(p, &end);
            if (end == p)
            {
                break;
            }
            p = end;
            d++;
        }
        if (d == dim)
        {
            nframes++;
        }
    } // while

    fclose(fp);
    *out_num_frames = nframes;
    *out_dim = dim;
    return data;
} // load_dataset_vectors

/**
 * insert_sorted_nn() - Maintain sorted top-k nearest neighbors list.
 */
static void insert_sorted_nn(
    KnnNeighbor *list,
    int          k,
    long         idx,
    double       d)
{
    if (d >= list[k - 1].dist && list[k - 1].index != -1)
    {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && (list[pos - 1].index == -1 || d < list[pos - 1].dist))
    {
        list[pos] = list[pos - 1];
        pos--;
    }
    list[pos].index = idx;
    list[pos].dist = d;
}

/**
 * mcp_tool_knn_quality() - Evaluate recall and distance approximation of kNN search results.
 */
int mcp_tool_knn_quality(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *ds_item = cJSON_GetObjectItemCaseSensitive(args, "dataset_path");
    cJSON *out_item = cJSON_GetObjectItemCaseSensitive(args, "knn_output");

    if (ds_item == NULL || !cJSON_IsString(ds_item) ||
        out_item == NULL || !cJSON_IsString(out_item))
    {
        cJSON_AddStringToObject(res, "error", "dataset_path and knn_output are required");
        return -1;
    }

    char resolved_ds[512];
    char resolved_out[512];
    mcp_resolve_path(ds_item->valuestring, resolved_ds, sizeof(resolved_ds));
    mcp_resolve_path(out_item->valuestring, resolved_out, sizeof(resolved_out));

    int sample_queries = 50;
    cJSON *sq_item = cJSON_GetObjectItemCaseSensitive(args, "sample_queries");
    if (sq_item != NULL && cJSON_IsNumber(sq_item))
    {
        sample_queries = sq_item->valueint;
        if (sample_queries < 5) sample_queries = 5;
        if (sample_queries > 500) sample_queries = 500;
    }

    int target_k = 10;
    cJSON *k_item = cJSON_GetObjectItemCaseSensitive(args, "k");
    if (k_item != NULL && cJSON_IsNumber(k_item))
    {
        target_k = k_item->valueint;
        if (target_k < 1) target_k = 1;
        if (target_k > 100) target_k = 100;
    }

    int user_dtmin = -1;
    cJSON *dt_item = cJSON_GetObjectItemCaseSensitive(args, "dtmin");
    if (dt_item != NULL && cJSON_IsNumber(dt_item))
    {
        user_dtmin = dt_item->valueint;
    }

    /* Step 1: Load dataset vectors */
    long num_frames = 0;
    int dim = 0;
    float *vectors = load_dataset_vectors(resolved_ds, &num_frames, &dim);
    if (vectors == NULL || num_frames <= 0 || dim <= 0)
    {
        cJSON_AddStringToObject(res, "error", "Failed to load dataset vectors");
        return -1;
    }

    /* Step 2: Open and parse kNN results */
    FILE *fk = fopen(resolved_out, "r");
    if (fk == NULL)
    {
        free(vectors);
        cJSON_AddStringToObject(res, "error", "Failed to open knn_output file");
        return -1;
    }

    /* Read all reported lines into an array */
    size_t cap_res = 1000;
    size_t count_res = 0;
    KnnQueryResult *res_list = (KnnQueryResult *)malloc(cap_res * sizeof(KnnQueryResult));
    if (res_list == NULL)
    {
        free(vectors);
        fclose(fk);
        cJSON_AddStringToObject(res, "error", "Memory allocation failed");
        return -1;
    }

    char line[65536];
    long min_dt_seen = 999999;

    while (fgets(line, sizeof(line), fk) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
        {
            continue;
        }

        char *p = line;
        char *end = NULL;
        long q_id = strtol(p, &end, 10);
        if (end == p)
        {
            continue;
        }
        p = end;

        if (count_res >= cap_res)
        {
            cap_res *= 2;
            KnnQueryResult *new_list =
                (KnnQueryResult *)realloc(res_list, cap_res * sizeof(KnnQueryResult));
            if (new_list == NULL)
            {
                break;
            }
            res_list = new_list;
        }

        KnnNeighbor *nbrs = (KnnNeighbor *)malloc((size_t)target_k * sizeof(KnnNeighbor));
        int n_count = 0;

        while (*p != '\0' && n_count < target_k)
        {
            long n_idx = strtol(p, &end, 10);
            if (end == p)
            {
                break;
            }
            p = end;

            double dist = strtod(p, &end);
            if (end == p)
            {
                break;
            }
            p = end;

            nbrs[n_count].index = n_idx;
            nbrs[n_count].dist = dist;
            n_count++;

            long diff = labs(n_idx - q_id);
            if (diff > 0 && diff < min_dt_seen)
            {
                min_dt_seen = diff;
            }
        } // while

        res_list[count_res].query_id = q_id;
        res_list[count_res].num_neighbors = n_count;
        res_list[count_res].neighbors = nbrs;
        count_res++;
    } // while

    fclose(fk);

    if (count_res == 0)
    {
        free(vectors);
        free(res_list);
        cJSON_AddStringToObject(res, "error", "No valid kNN query rows found in output");
        return -1;
    }

    long eff_dtmin = (user_dtmin >= 0) ? user_dtmin : min_dt_seen;
    if (eff_dtmin > 100000)
    {
        eff_dtmin = 0;
    }

    /* Step 3: Sample query evaluations */
    long step = (long)count_res / sample_queries;
    if (step < 1) step = 1;

    double sum_recall_at_1 = 0.0;
    double sum_recall_at_k = 0.0;
    double sum_mrr = 0.0;
    double sum_dist_ratio = 0.0;
    double max_dist_ratio = 1.0;
    long evaluated_queries = 0;

    KnnNeighbor *exact_nn = (KnnNeighbor *)malloc((size_t)target_k * sizeof(KnnNeighbor));
    if (exact_nn == NULL)
    {
        for (size_t ii = 0; ii < count_res; ii++)
        {
            free(res_list[ii].neighbors);
        }
        free(res_list);
        free(vectors);
        cJSON_AddStringToObject(res, "error", "Allocation failed");
        return -1;
    }

    for (size_t s = 0; s < count_res && evaluated_queries < sample_queries; s += (size_t)step)
    {
        long q_id = res_list[s].query_id;
        if (q_id < 0 || q_id >= num_frames)
        {
            continue;
        }

        const float *qv = vectors + q_id * dim;

        for (int k = 0; k < target_k; k++)
        {
            exact_nn[k].index = -1;
            exact_nn[k].dist = 1e30;
        }

        /* Brute-force ground truth search */
        for (long f = 0; f < num_frames; f++)
        {
            if (f == q_id)
            {
                continue;
            }
            if (labs(f - q_id) < eff_dtmin)
            {
                continue;
            }

            const float *fv = vectors + f * dim;
            double sum_sq = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double diff = (double)qv[d] - (double)fv[d];
                sum_sq += diff * diff;
            }
            double d = sqrt(sum_sq);
            insert_sorted_nn(exact_nn, target_k, f, d);
        } // for f

        if (exact_nn[0].index == -1)
        {
            continue;
        }

        /* Compare reported vs exact */
        int rep_count = res_list[s].num_neighbors;
        const KnnNeighbor *rep_nbrs = res_list[s].neighbors;

        /* Recall@1 */
        int found_1 = 0;
        for (int i = 0; i < rep_count; i++)
        {
            if (rep_nbrs[i].index == exact_nn[0].index)
            {
                found_1 = 1;
                break;
            }
        }
        if (found_1) sum_recall_at_1 += 1.0;

        /* Recall@k */
        int matches_k = 0;
        int active_k = (target_k < rep_count) ? target_k : rep_count;
        for (int i = 0; i < active_k; i++)
        {
            long t_idx = exact_nn[i].index;
            if (t_idx == -1) break;
            for (int j = 0; j < rep_count; j++)
            {
                if (rep_nbrs[j].index == t_idx)
                {
                    matches_k++;
                    break;
                }
            }
        }
        double rec_k = (active_k > 0) ? ((double)matches_k / (double)active_k) : 1.0;
        sum_recall_at_k += rec_k;

        /* MRR */
        double mrr_sample = 0.0;
        for (int i = 0; i < rep_count; i++)
        {
            if (rep_nbrs[i].index == exact_nn[0].index)
            {
                mrr_sample = 1.0 / (double)(i + 1);
                break;
            }
        }
        sum_mrr += mrr_sample;

        /* Distance ratio */
        if (rep_count > 0 && exact_nn[0].dist > 1e-12)
        {
            double ratio = rep_nbrs[0].dist / exact_nn[0].dist;
            if (ratio < 1.0) ratio = 1.0;
            sum_dist_ratio += ratio;
            if (ratio > max_dist_ratio)
            {
                max_dist_ratio = ratio;
            }
        }
        else
        {
            sum_dist_ratio += 1.0;
        }

        evaluated_queries++;
    } // for s

    free(exact_nn);

    double mean_recall_1 = (evaluated_queries > 0) ?
                           (sum_recall_at_1 / (double)evaluated_queries) : 1.0;
    double mean_recall_k = (evaluated_queries > 0) ?
                           (sum_recall_at_k / (double)evaluated_queries) : 1.0;
    double mrr = (evaluated_queries > 0) ?
                 (sum_mrr / (double)evaluated_queries) : 1.0;
    double mean_ratio = (evaluated_queries > 0) ?
                        (sum_dist_ratio / (double)evaluated_queries) : 1.0;

    const char *verdict = "Sub-optimal Approximation";
    if (mean_recall_1 >= 0.999 && mean_ratio <= 1.0001)
    {
        verdict = "Exact / Bit-Exact Equivalent";
    }
    else if (mean_recall_k >= 0.95)
    {
        verdict = "High Fidelity (Recall@k >= 95%)";
    }
    else if (mean_recall_k >= 0.80)
    {
        verdict = "Good Approximation (Recall@k >= 80%)";
    }

    cJSON *qinfo = cJSON_CreateObject();
    cJSON_AddNumberToObject(qinfo, "total_queries", (double)count_res);
    cJSON_AddNumberToObject(qinfo, "evaluated_sample_queries", (double)evaluated_queries);
    cJSON_AddNumberToObject(qinfo, "k", target_k);
    cJSON_AddNumberToObject(qinfo, "inferred_dtmin", (double)eff_dtmin);
    cJSON_AddItemToObject(res, "query_info", qinfo);

    cJSON *metrics = cJSON_CreateObject();
    cJSON_AddNumberToObject(metrics, "recall_at_1", mean_recall_1);
    cJSON_AddNumberToObject(metrics, "recall_at_k", mean_recall_k);
    cJSON_AddNumberToObject(metrics, "mean_reciprocal_rank", mrr);
    cJSON_AddNumberToObject(metrics, "mean_distance_ratio", mean_ratio);
    cJSON_AddNumberToObject(metrics, "max_distance_ratio", max_dist_ratio);
    cJSON_AddItemToObject(res, "accuracy_metrics", metrics);

    char report[1024];
    snprintf(
        report, sizeof(report),
        "### kNN Quality Verification\n"
        "- **Overall Verdict**: %s\n"
        "- **Recall@1**: %.2f%% (Queries with true 1-NN in candidates)\n"
        "- **Recall@%d**: %.2f%% (Mean true top-%d neighbor coverage)\n"
        "- **Mean Reciprocal Rank (MRR)**: %.4f\n"
        "- **Distance Ratio (d_approx / d_exact)**: Mean: %.4f | Max: %.4f\n"
        "- **Parameters Verified**: Evaluated %ld sampled queries (dtmin = %ld).",
        verdict, mean_recall_1 * 100.0,
        target_k, mean_recall_k * 100.0, target_k,
        mrr, mean_ratio, max_dist_ratio,
        evaluated_queries, eff_dtmin);
    cJSON_AddStringToObject(res, "report", report);

    for (size_t ii = 0; ii < count_res; ii++)
    {
        free(res_list[ii].neighbors);
    }
    free(res_list);
    free(vectors);
    return 0;
} // mcp_tool_knn_quality

const struct mcp_tool_def mcp_tooldef_knn_quality = {
    .name         = "gric_knn_quality",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_knn_quality,
    .description  = "Evaluate recall, Mean Reciprocal Rank (MRR), and distance approximation "
                    "accuracy of kNN results against exact brute-force ground truth.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"dataset_path\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to dataset coordinates file (ASCII or BIN).\"\n"
        "    },\n"
        "    \"knn_output\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to gric-knn output results file.\"\n"
        "    },\n"
        "    \"sample_queries\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of sample queries to evaluate (default: 50).\"\n"
        "    },\n"
        "    \"k\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of top nearest neighbors to verify (default: 10).\"\n"
        "    },\n"
        "    \"dtmin\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Minimum temporal frame exclusion distance (optional).\"\n"
        "    }\n"
        "  },\n"
        "  \"required\": [\"dataset_path\", \"knn_output\"]\n"
        "}",
};

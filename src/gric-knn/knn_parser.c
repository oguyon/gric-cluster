/**
 * @file knn_parser.c
 * @brief Parsers for Pass 1 clustering artifacts and metadata.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_parser.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/**
 * knn_compare_member_meta_radii() - Sort member records by ascending anchor distance.
 * @a: Pointer to first MemberMeta.
 * @b: Pointer to second MemberMeta.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
int knn_compare_member_meta_radii(
    const void *a,
    const void *b)
{
    const MemberMeta *ma = (const MemberMeta *)a;
    const MemberMeta *mb = (const MemberMeta *)b;
    if (ma->r_anchor < mb->r_anchor)
    {
        return -1;
    }
    if (ma->r_anchor > mb->r_anchor)
    {
        return 1;
    }
    return 0;
}

/**
 * parse_membership_file() - Load frame membership and anchor distance metadata.
 * @path:  Path to frame_membership.txt.
 * @model: Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_parse_membership_file(
    const char *path,
    KnnModel   *model)
{
    // Try binary format first
    FILE *f_bin = fopen(path, "rb");
    if (f_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(f_bin, &hdr, &comment) == 0 &&
            hdr.data_type == GRIC_BIN_DTYPE_UINT32)
        {
            long nframes = (long)hdr.dims[0];
            uint32_t *ubuf = (uint32_t *)malloc((size_t)nframes * sizeof(uint32_t));
            if (ubuf != NULL && fread(ubuf, sizeof(uint32_t), nframes, f_bin) == (size_t)nframes)
            {
                int max_c = -1;
                for (long i = 0; i < nframes; i++)
                {
                    if (ubuf[i] > (uint32_t)INT_MAX)
                    {
                        fprintf(stderr,
                                "Error: membership file '%s' contains out-of-range cluster IDs\n",
                                path);
                        free(ubuf);
                        if (comment != NULL)
                        {
                            free(comment);
                        }
                        fclose(f_bin);
                        return -1;
                    }
                    if ((int)ubuf[i] > max_c)
                    {
                        max_c = (int)ubuf[i];
                    }
                }
                model->total_dataset_frames = nframes;
                model->num_clusters = max_c + 1;
                model->clusters =
                    (KnnCluster *)calloc((size_t)model->num_clusters, sizeof(KnnCluster));
                model->frame_cluster_map =
                    (int *)malloc((size_t)nframes * sizeof(int));
                model->frame_r_anchor =
                    (float *)calloc((size_t)nframes, sizeof(float));

                for (long i = 0; i < nframes; i++)
                {
                    int c = (int)ubuf[i];
                    model->frame_cluster_map[i] = c;
                    model->clusters[c].num_members++;
                }
                for (int c = 0; c < model->num_clusters; c++)
                {
                    if (model->clusters[c].num_members > 0)
                    {
                        model->clusters[c].members =
                            (MemberMeta *)malloc((size_t)model->clusters[c].num_members *
                                                 sizeof(MemberMeta));
                    }
                }
                int *curr_m = (int *)calloc((size_t)model->num_clusters, sizeof(int));
                for (long i = 0; i < nframes; i++)
                {
                    int c = model->frame_cluster_map[i];
                    int slot = curr_m[c]++;
                    model->clusters[c].members[slot].frame_id = i;
                    model->clusters[c].members[slot].r_anchor = 0.0;
                }
                free(curr_m);
                free(ubuf);
                if (comment != NULL) free(comment);
                fclose(f_bin);
                return 0;
            }
            if (ubuf != NULL) free(ubuf);
        }
        if (comment != NULL) free(comment);
        fclose(f_bin);
    }

    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Error: Could not open membership file '%s'\n", path);
        return -1;
    }

    char line[2048];
    long count = 0;
    int max_cluster_id = -1;

    // First pass: count entries and find maximum cluster ID
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
        {
            continue;
        }

        long f_id = -1;
        int  c_id = -1;
        if (sscanf(line, "%ld %d", &f_id, &c_id) >= 2)
        {
            count++;
            if (c_id > max_cluster_id)
            {
                max_cluster_id = c_id;
            }
        }
    } // while counting entries

    if (count == 0 || max_cluster_id < 0)
    {
        fprintf(stderr, "Error: Membership file '%s' contains no valid records\n", path);
        fclose(f);
        return -1;
    }

    model->total_dataset_frames = count;
    model->num_clusters = max_cluster_id + 1;

    model->frame_cluster_map = (int *)malloc((size_t)count * sizeof(int));
    model->frame_r_anchor = (float *)malloc((size_t)count * sizeof(float));
    model->clusters = (KnnCluster *)calloc((size_t)model->num_clusters, sizeof(KnnCluster));

    if (model->frame_cluster_map == NULL || model->frame_r_anchor == NULL
        || model->clusters == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for membership metadata\n");
        fclose(f);
        return -1;
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        model->clusters[c].cluster_id = c;
        model->clusters[c].radius = 0.0;
        model->clusters[c].num_members = 0;
        model->clusters[c].capacity = 16;
        model->clusters[c].members =
            (MemberMeta *)malloc((size_t)model->clusters[c].capacity * sizeof(MemberMeta));
        if (model->clusters[c].members == NULL)
        {
            fclose(f);
            return -1;
        }
    } // for (int c = 0; ...)

    rewind(f);

    long read_idx = 0;
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
        {
            continue;
        }

        long   f_id = -1;
        int    c_id = -1;
        double r_anchor = 0.0;
        int    n_tokens = sscanf(line, "%ld %d %lf", &f_id, &c_id, &r_anchor);

        if (n_tokens >= 2 && read_idx < count)
        {
            if (n_tokens < 3)
            {
                r_anchor = 0.0; // fallback if legacy 2-column file
            }

            model->frame_cluster_map[read_idx] = c_id;
            model->frame_r_anchor[read_idx] = (float)r_anchor;

            if (c_id >= 0 && c_id < model->num_clusters)
            {
                KnnCluster *cl = &model->clusters[c_id];
                if (cl->num_members >= cl->capacity)
                {
                    cl->capacity *= 2;
                    cl->members = (MemberMeta *)realloc(
                        cl->members,
                        (size_t)cl->capacity * sizeof(MemberMeta));
                    if (cl->members == NULL)
                    {
                        fclose(f);
                        return -1;
                    }
                }
                cl->members[cl->num_members].frame_id = (uint32_t)read_idx;
                cl->members[cl->num_members].r_anchor = (float)r_anchor;
                cl->num_members++;

                if (r_anchor > cl->radius)
                {
                    cl->radius = r_anchor;
                }
            }

            read_idx++;
        }
    } // while reading entries

    fclose(f);

    /* Sort each cluster's members array by ascending r_anchor for O(log N) binary search */
    for (int c = 0; c < model->num_clusters; c++)
    {
        if (model->clusters[c].num_members > 1)
        {
            qsort(model->clusters[c].members,
                  (size_t)model->clusters[c].num_members,
                  sizeof(MemberMeta),
                  knn_compare_member_meta_radii);
        }
    }

    return 0;
}

/**
 * parse_radii_file() - Parse cluster_radii.bin or cluster_radii.txt if available.
 * @cluster_dir: Path to directory containing cluster output files.
 * @model:       Pointer to resident KnnModel.
 */
void knn_parse_radii_file(
    const char *cluster_dir,
    KnnModel   *model)
{
    char bin_path[2048];
    snprintf(bin_path, sizeof(bin_path), "%s/cluster_radii.bin", cluster_dir);
    FILE *f_bin = fopen(bin_path, "rb");
    if (f_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(f_bin, &hdr, &comment) == 0 &&
            hdr.num_elements == (size_t)model->num_clusters &&
            hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
        {
            float *buf = (float *)malloc(hdr.data_bytes);
            if (buf != NULL && fread(buf, 1, hdr.data_bytes, f_bin) == hdr.data_bytes)
            {
                for (int c = 0; c < model->num_clusters; c++)
                {
                    model->clusters[c].radius = (double)buf[c];
                }
                free(buf);
                if (comment != NULL)
                {
                    free(comment);
                }
                fclose(f_bin);
                return;
            }
            if (buf != NULL)
            {
                free(buf);
            }
        }
        if (comment != NULL)
        {
            free(comment);
        }
        fclose(f_bin);
    }

    char txt_path[2048];
    snprintf(txt_path, sizeof(txt_path), "%s/cluster_radii.txt", cluster_dir);
    FILE *f = fopen(txt_path, "r");
    if (f != NULL)
    {
        char line[1024];
        while (fgets(line, sizeof(line), f) != NULL)
        {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
            {
                continue;
            }

            int    c_id = -1;
            int    count = 0;
            double radius = 0.0;
            if (sscanf(line, "%d %d %lf", &c_id, &count, &radius) == 3)
            {
                if (c_id >= 0 && c_id < model->num_clusters)
                {
                    if (radius > model->clusters[c_id].radius)
                    {
                        model->clusters[c_id].radius = radius;
                    }
                }
            }
        } // while parsing radii
        fclose(f);
        return;
    }

    /* Fallback: if radii file not found, use member metadata if populated */
    for (int c = 0; c < model->num_clusters; c++)
    {
        if (model->clusters[c].radius <= 0.0)
        {
            double max_r = 0.0;
            for (int m = 0; m < model->clusters[c].num_members; m++)
            {
                double r = (double)model->clusters[c].members[m].r_anchor;
                if (r > max_r)
                {
                    max_r = r;
                }
            }
            model->clusters[c].radius = (max_r > 0.0) ? max_r : model->model_rlim;
        }
    }
}

/**
 * parse_cluster_log() - Parse parameters from cluster_run.log if available.
 * @path:  Path to cluster_run.log.
 * @model: Pointer to KnnModel.
 */
void knn_parse_cluster_log(
    const char *path,
    KnnModel   *model)
{
    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        return;
    }

    char line[512];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        double rlim_val = 0.0;
        if (sscanf(line, "PARAM_RLIM: %lf", &rlim_val) == 1)
        {
            model->model_rlim = rlim_val;
            break;
        }
    } // while reading cluster log

    fclose(f);
}

/**
 * propagate_triangle_lower_bounds() - Use triangle inequality to compute
 *                                     tight lower bounds for missing pairs.
 * @model: Pointer to KnnModel with partially populated dcc_matrix.
 */
static void propagate_triangle_lower_bounds(
    KnnModel *model)
{
    int M = model->num_clusters;
    if (M <= 1)
    {
        return;
    }

    double *d_max = (double *)malloc((size_t)M * (size_t)M * sizeof(double));
    if (d_max == NULL)
    {
        return;
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (i == j)
            {
                d_max[i * M + j] = 0.0;
            }
            else
            {
                double measured = model->dcc_matrix[i * M + j];
                if (measured > 0.0)
                {
                    d_max[i * M + j] = measured;
                }
                else
                {
                    d_max[i * M + j] = 1e19;
                }
            }
        }
    }

    // Step 1: Upper Bound Relaxation (All-Pairs Shortest Path)
    for (int k = 0; k < M; k++)
    {
        #pragma omp parallel for schedule(static) if(M >= 64)
        for (int i = 0; i < M; i++)
        {
            double d_ik = d_max[i * M + k];
            if (d_ik >= 1e18)
            {
                continue;
            }
            for (int j = 0; j < M; j++)
            {
                double d_kj = d_max[k * M + j];
                if (d_kj >= 1e18)
                {
                    continue;
                }
                double sum = d_ik + d_kj;
                if (sum < d_max[i * M + j])
                {
                    d_max[i * M + j] = sum;
                }
            }
        }
    }

    // Step 2: Lower Bound Propagation via Triangle Inequality
    #pragma omp parallel for schedule(dynamic) if(M >= 64)
    for (int i = 0; i < M; i++)
    {
        for (int j = i + 1; j < M; j++)
        {
            if (model->dcc_matrix[i * M + j] > 0.0)
            {
                continue;
            }

            double max_lb = 0.0;
            for (int k = 0; k < M; k++)
            {
                double d_min_ik = model->dcc_matrix[i * M + k];
                double d_max_kj = d_max[k * M + j];
                if (d_min_ik > 0.0 && d_max_kj < 1e18)
                {
                    double lb1 = d_min_ik - d_max_kj;
                    if (lb1 > max_lb)
                    {
                        max_lb = lb1;
                    }
                }

                double d_min_jk = model->dcc_matrix[j * M + k];
                double d_max_ki = d_max[k * M + i];
                if (d_min_jk > 0.0 && d_max_ki < 1e18)
                {
                    double lb2 = d_min_jk - d_max_ki;
                    if (lb2 > max_lb)
                    {
                        max_lb = lb2;
                    }
                }
            }

            model->dcc_matrix[i * M + j] = max_lb;
            model->dcc_matrix[j * M + i] = max_lb;
        }
    }

    free(d_max);
}

/**
 * parse_dcc_file() - Load M x M inter-cluster distance matrix or lower bounds.
 * @cluster_dir: Directory containing cluster results.
 * @model:       Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_parse_dcc_file(
    const char *cluster_dir,
    KnnModel   *model)
{
    int M = model->num_clusters;
    model->dcc_matrix = (double *)malloc((size_t)M * (size_t)M * sizeof(double));
    if (model->dcc_matrix == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for DCC matrix\n");
        return -1;
    }

    for (int i = 0; i < M * M; i++)
    {
        model->dcc_matrix[i] = 0.0;
    }

    char path[2048];
    snprintf(path, sizeof(path), "%s/dcc.bin", cluster_dir);
    FILE *f_bin = fopen(path, "rb");
    if (f_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(f_bin, &hdr, &comment) == 0)
        {
            if (hdr.data_type == GRIC_BIN_DTYPE_UINT16)
            {
                model->dcc_sq16 = (uint16_t *)malloc((size_t)M * (size_t)M * sizeof(uint16_t));
                if (model->dcc_sq16 != NULL)
                {
                    if (fread(model->dcc_sq16, sizeof(uint16_t), (size_t)M * (size_t)M, f_bin) ==
                        (size_t)M * (size_t)M)
                    {
                        double scale = 0.0;
                        if (comment != NULL)
                        {
                            const char *sp = strstr(comment, "scale=");
                            if (sp != NULL)
                            {
                                sscanf(sp, "scale=%lf", &scale);
                            }
                        }
                        if (scale <= 0.0)
                        {
                            scale = (model->model_rlim > 0.0)
                                ? (16384.0 / model->model_rlim) : 1000.0;
                        }
                        model->dcc_sq16_scale = scale;
                        model->dcc_sq16_inv_scale = 1.0 / scale;
                        for (int i = 0; i < M * M; i++)
                        {
                            model->dcc_matrix[i] =
                                (double)model->dcc_sq16[i] * model->dcc_sq16_inv_scale;
                        }
                        if (comment != NULL) free(comment);
                        fclose(f_bin);
                        return 0;
                    }
                    free(model->dcc_sq16);
                    model->dcc_sq16 = NULL;
                }
            }
            else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
            {
                if (fread(model->dcc_matrix, sizeof(double), (size_t)M * (size_t)M, f_bin) ==
                    (size_t)M * (size_t)M)
                {
                    double scale = (model->model_rlim > 0.0)
                        ? (16384.0 / model->model_rlim) : 1000.0;
                    model->dcc_sq16_scale = scale;
                    model->dcc_sq16_inv_scale = 1.0 / scale;
                    model->dcc_sq16 = (uint16_t *)malloc((size_t)M * (size_t)M * sizeof(uint16_t));
                    if (model->dcc_sq16 != NULL)
                    {
                        for (int i = 0; i < M * M; i++)
                        {
                            double d = model->dcc_matrix[i];
                            model->dcc_sq16[i] = (d <= 0.0) ? 0 :
                                ((d * scale >= 65534.0) ? 65534 : (uint16_t)(d * scale + 0.5));
                        }
                    }
                    if (comment != NULL) free(comment);
                    fclose(f_bin);
                    return 0;
                }
            }
            else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                float *fbuf = (float *)malloc((size_t)M * (size_t)M * sizeof(float));
                if (fbuf != NULL)
                {
                    if (fread(fbuf, sizeof(float), (size_t)M * (size_t)M, f_bin) ==
                        (size_t)M * (size_t)M)
                    {
                        double scale = (model->model_rlim > 0.0)
                            ? (16384.0 / model->model_rlim) : 1000.0;
                        model->dcc_sq16_scale = scale;
                        model->dcc_sq16_inv_scale = 1.0 / scale;
                        model->dcc_sq16 =
                            (uint16_t *)malloc((size_t)M * (size_t)M * sizeof(uint16_t));

                        for (int i = 0; i < M * M; i++)
                        {
                            double d = (double)fbuf[i];
                            model->dcc_matrix[i] = d;
                            if (model->dcc_sq16 != NULL)
                            {
                                model->dcc_sq16[i] = (d <= 0.0) ? 0 :
                                    ((d * scale >= 65534.0) ? 65534 :
                                     (uint16_t)(d * scale + 0.5));
                            }
                        }
                        free(fbuf);
                        if (comment != NULL) free(comment);
                        fclose(f_bin);
                        return 0;
                    }
                    free(fbuf);
                }
            }
        }
        if (comment != NULL) free(comment);
        fclose(f_bin);
    }

    snprintf(path, sizeof(path), "%s/dccmin.txt", cluster_dir);
    FILE *f = fopen(path, "r");
    int   using_dccmin = 0;
    if (f != NULL)
    {
        using_dccmin = 1;
    }
    else
    {
        snprintf(path, sizeof(path), "%s/dcc.txt", cluster_dir);
        f = fopen(path, "r");
    }

    if (f == NULL)
    {
        return 0;
    }

    long entries_read = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
        {
            continue;
        }

        int    c1 = -1;
        int    c2 = -1;
        double dist = 0.0;
        if (sscanf(line, "%d %d %lf", &c1, &c2, &dist) == 3)
        {
            if (c1 >= 0 && c1 < M && c2 >= 0 && c2 < M)
            {
                model->dcc_matrix[c1 * M + c2] = dist;
                model->dcc_matrix[c2 * M + c1] = dist;
                entries_read++;
            }
        }
    } // while reading DCC

    fclose(f);

    long total_pairs = (long)M * (M - 1) / 2;
    if (!using_dccmin && entries_read < total_pairs && entries_read > 0)
    {
        propagate_triangle_lower_bounds(model);
    }

    if (model->dcc_sq16 == NULL)
    {
        double scale = (model->model_rlim > 0.0)
            ? (16384.0 / model->model_rlim) : 1000.0;
        model->dcc_sq16_scale = scale;
        model->dcc_sq16_inv_scale = 1.0 / scale;
        model->dcc_sq16 = (uint16_t *)malloc((size_t)M * (size_t)M * sizeof(uint16_t));
        if (model->dcc_sq16 != NULL)
        {
            for (int i = 0; i < M * M; i++)
            {
                double d = model->dcc_matrix[i];
                model->dcc_sq16[i] = (d <= 0.0) ? 0 :
                    ((d * scale >= 65534.0) ? 65534 : (uint16_t)(d * scale + 0.5));
            }
        }
    }

    return 0;
}

/**
 * reconstruct_anchors_from_input() - Extract anchors directly from input dataset.
 * @input_data_path: Path to original input dataset.
 * @model:           Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */

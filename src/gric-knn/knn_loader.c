/**
 * @file knn_loader.c
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_loader.h"
#include "knn_tree.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

/**
 * check_is_fits() - Check if filename has a FITS extension.
 * @filename: Path to the input dataset.
 *
 * Return: 1 if FITS, 0 otherwise.
 */
static int check_is_fits(
    const char *filename)
{
    if (filename == NULL)
    {
        return 0;
    }

    size_t len = strlen(filename);
    if (len >= 5 && strcasecmp(filename + len - 5, ".fits") == 0)
    {
        return 1;
    }
    if (len >= 8 && strcasecmp(filename + len - 8, ".fits.gz") == 0)
    {
        return 1;
    }

    return 0;
}

/**
 * compare_member_meta_radii() - Sort member records by ascending anchor distance.
 * @a: Pointer to first MemberMeta.
 * @b: Pointer to second MemberMeta.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_member_meta_radii(
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
static int parse_membership_file(
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
                  compare_member_meta_radii);
        }
    }

    return 0;
}

/**
 * parse_radii_file() - Parse cluster_radii.txt if available.
 * @path:  Path to cluster_radii.txt.
 * @model: Pointer to KnnModel.
 */
static void parse_radii_file(
    const char *path,
    KnnModel   *model)
{
    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        return; // Optional file
    }

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
                model->clusters[c_id].radius = radius;
            }
        }
    } // while parsing radii

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
static int parse_dcc_file(
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
        if (gric_bin_read_header(f_bin, &hdr, &comment) == 0 &&
            hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
        {
            float *fbuf = (float *)malloc((size_t)M * (size_t)M * sizeof(float));
            if (fbuf != NULL)
            {
                if (fread(fbuf, sizeof(float), (size_t)M * (size_t)M, f_bin) ==
                    (size_t)M * (size_t)M)
                {
                    for (int i = 0; i < M * M; i++)
                    {
                        model->dcc_matrix[i] = (double)fbuf[i];
                    }
                    free(fbuf);
                    if (comment != NULL) free(comment);
                    fclose(f_bin);
                    return 0;
                }
                free(fbuf);
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

    return 0;
}

/**
 * reconstruct_anchors_from_input() - Extract anchors directly from input dataset.
 * @input_data_path: Path to original input dataset.
 * @model:           Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
static int reconstruct_anchors_from_input(
    const char *input_data_path,
    KnnModel   *model)
{
#ifdef USE_CFITSIO
    if (model->is_fits_input)
    {
        int status = 0;
        fitsfile *fptr = NULL;
        fits_open_file(&fptr, input_data_path, READONLY, &status);
        if (status == 0 && fptr != NULL)
        {
            int bitpix = 0;
            int naxis = 0;
            long naxes[3] = {0, 0, 0};
            fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
            if (status == 0 && naxis >= 2)
            {
                model->frame_width = naxes[0];
                model->frame_height = naxes[1];
                model->frame_elements = model->frame_width * model->frame_height;

                for (int c = 0; c < model->num_clusters; c++)
                {
                    model->clusters[c].anchor_data =
                        (double *)malloc((size_t)model->frame_elements * sizeof(double));
                    if (model->clusters[c].anchor_data == NULL)
                    {
                        fits_close_file(fptr, &status);
                        return -1;
                    }
                    long f_anchor = (model->clusters[c].num_members > 0) ?
                                    (long)model->clusters[c].members[0].frame_id : 0;
                    long fpixel[3] = {1, 1, f_anchor + 1};
                    fits_read_pix(fptr, TDOUBLE, fpixel, model->frame_elements, NULL,
                                  model->clusters[c].anchor_data, NULL, &status);
                }
                fits_close_file(fptr, &status);
                return (status == 0) ? 0 : -1;
            }
            fits_close_file(fptr, &status);
        }
    }
#endif // USE_CFITSIO

    FILE *f = fopen(input_data_path, "r");
    if (f == NULL)
    {
        return -1;
    }

    char line_buf[65536];
    long elements_detected = 0;
    while (fgets(line_buf, sizeof(line_buf), f) != NULL)
    {
        if (line_buf[0] == '#' || line_buf[0] == '\n' || line_buf[0] == '\0')
        {
            continue;
        }
        char *ptr = line_buf;
        while (*ptr != '\0')
        {
            while (isspace((unsigned char)*ptr))
            {
                ptr++;
            }
            if (*ptr == '\0')
            {
                break;
            }
            elements_detected++;
            while (*ptr != '\0' && !isspace((unsigned char)*ptr))
            {
                ptr++;
            }
        } // while (*ptr != '\0')
        break;
    } // while detecting dimension

    if (elements_detected <= 0)
    {
        fclose(f);
        return -1;
    }

    model->frame_width = elements_detected;
    model->frame_height = 1;
    model->frame_elements = elements_detected;

    rewind(f);
    uint64_t *offsets =
        (uint64_t *)malloc((size_t)model->total_dataset_frames * sizeof(uint64_t));
    if (offsets == NULL)
    {
        fclose(f);
        return -1;
    }

    long frame_idx = 0;
    off_t cur_offset = ftello(f);
    while (fgets(line_buf, sizeof(line_buf), f) != NULL
           && frame_idx < model->total_dataset_frames)
    {
        if (line_buf[0] != '#' && line_buf[0] != '\n' && line_buf[0] != '\0')
        {
            offsets[frame_idx++] = (uint64_t)cur_offset;
        }
        cur_offset = ftello(f);
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        model->clusters[c].anchor_data =
            (double *)malloc((size_t)model->frame_elements * sizeof(double));
        if (model->clusters[c].anchor_data == NULL)
        {
            free(offsets);
            fclose(f);
            return -1;
        }
        long f_anchor = (model->clusters[c].num_members > 0) ?
                        (long)model->clusters[c].members[0].frame_id : 0;
        fseeko(f, (off_t)offsets[f_anchor], SEEK_SET);
        for (long k = 0; k < model->frame_elements; k++)
        {
            if (fscanf(f, "%lf", &model->clusters[c].anchor_data[k]) != 1)
            {
                model->clusters[c].anchor_data[k] = 0.0;
            }
        }
    } // for (int c = 0; ...)

    free(offsets);
    fclose(f);
    return 0;
}

/**
 * load_anchors() - Load anchor frames from cluster dir or fallback to input dataset.
 * @cluster_dir:      Directory containing Pass 1 outputs.
 * @input_data_path:  Original dataset path.
 * @model:            Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
static int load_anchors(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model)
{
    int M = model->num_clusters;
    char path[2048];

    // Try binary anchors.bin first
    snprintf(path, sizeof(path), "%s/anchors.bin", cluster_dir);
    FILE *a_bin = fopen(path, "rb");
    if (a_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(a_bin, &hdr, &comment) == 0)
        {
            model->frame_width = (hdr.ndim > 1) ? (long)hdr.dims[1] : 1;
            model->frame_height = 1;
            model->frame_elements = model->frame_width * model->frame_height;

            float *fbuf = (float *)malloc((size_t)model->frame_elements * sizeof(float));
            if (fbuf != NULL)
            {
                int ok = 1;
                for (int c = 0; c < M; c++)
                {
                    model->clusters[c].anchor_data =
                        (double *)malloc((size_t)model->frame_elements * sizeof(double));
                    if (model->clusters[c].anchor_data == NULL ||
                        fread(fbuf, sizeof(float), (size_t)model->frame_elements, a_bin) !=
                        (size_t)model->frame_elements)
                    {
                        ok = 0;
                        break;
                    }
                    for (long k = 0; k < model->frame_elements; k++)
                    {
                        model->clusters[c].anchor_data[k] = (double)fbuf[k];
                    }
                }
                free(fbuf);
                if (comment != NULL) free(comment);
                fclose(a_bin);
                if (ok) return 0;
            }
        }
        if (comment != NULL) free(comment);
        fclose(a_bin);
    }

    // Try FITS second
    snprintf(path, sizeof(path), "%s/anchors.fits", cluster_dir);

#ifdef USE_CFITSIO
    int status = 0;
    fitsfile *fptr = NULL;
    fits_open_file(&fptr, path, READONLY, &status);
    if (status == 0 && fptr != NULL)
    {
        int bitpix = 0;
        int naxis = 0;
        long naxes[3] = {0, 0, 0};
        fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);

        if (status == 0 && naxis >= 2)
        {
            model->frame_width = naxes[0];
            model->frame_height = naxes[1];
            model->frame_elements = model->frame_width * model->frame_height;

            for (int c = 0; c < M; c++)
            {
                model->clusters[c].anchor_data =
                    (double *)malloc((size_t)model->frame_elements * sizeof(double));
                if (model->clusters[c].anchor_data == NULL)
                {
                    fits_close_file(fptr, &status);
                    return -1;
                }

                long fpixel[3] = {1, 1, c + 1};
                fits_read_pix(fptr, TDOUBLE, fpixel, model->frame_elements, NULL,
                              model->clusters[c].anchor_data, NULL, &status);
            } // for (int c = 0; ...)

            fits_close_file(fptr, &status);
            return 0;
        } // if (status == 0)

        fits_close_file(fptr, &status);
    }
#endif // USE_CFITSIO

    // Try anchors.txt
    snprintf(path, sizeof(path), "%s/anchors.txt", cluster_dir);
    FILE *f = fopen(path, "r");
    if (f != NULL)
    {
        char line_buf[65536];
        long elements_detected = 0;

        if (fgets(line_buf, sizeof(line_buf), f) != NULL)
        {
            char *ptr = line_buf;
            while (*ptr != '\0')
            {
                while (isspace((unsigned char)*ptr))
                {
                    ptr++;
                }
                if (*ptr == '\0')
                {
                    break;
                }
                elements_detected++;
                while (*ptr != '\0' && !isspace((unsigned char)*ptr))
                {
                    ptr++;
                }
            }
        }

        if (elements_detected > 0)
        {
            model->frame_width = elements_detected;
            model->frame_height = 1;
            model->frame_elements = elements_detected;
            rewind(f);

            for (int c = 0; c < M; c++)
            {
                model->clusters[c].anchor_data =
                    (double *)malloc((size_t)model->frame_elements * sizeof(double));
                if (model->clusters[c].anchor_data == NULL)
                {
                    fclose(f);
                    return -1;
                }

                for (long k = 0; k < model->frame_elements; k++)
                {
                    if (fscanf(f, "%lf", &model->clusters[c].anchor_data[k]) != 1)
                    {
                        model->clusters[c].anchor_data[k] = 0.0;
                    }
                }
            } // for (int c = 0; ...)

            fclose(f);
            return 0;
        }
        fclose(f);
    } // if (f != NULL)

    // Fallback: Reconstruct anchors directly from original dataset
    return reconstruct_anchors_from_input(input_data_path, model);
}

/**
 * knn_model_load() - Load all Pass 1 artifacts into resident KnnModel.
 * @cluster_dir:      Directory with Pass 1 outputs.
 * @input_data_path:  Path to original input dataset.
 * @model:            Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_model_load(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model)
{
    if (cluster_dir == NULL || input_data_path == NULL || model == NULL)
    {
        return -1;
    }

    memset(model, 0, sizeof(KnnModel));
    model->is_fits_input = check_is_fits(input_data_path);

    char memb_path[2048];
    snprintf(memb_path, sizeof(memb_path), "%s/frame_membership.bin", cluster_dir);
    if (parse_membership_file(memb_path, model) != 0)
    {
        snprintf(memb_path, sizeof(memb_path), "%s/frame_membership.txt", cluster_dir);
        if (parse_membership_file(memb_path, model) != 0)
        {
            return -1;
        }
    }

    char radii_path[2048];
    snprintf(radii_path, sizeof(radii_path), "%s/cluster_radii.txt", cluster_dir);
    parse_radii_file(radii_path, model);

    if (parse_dcc_file(cluster_dir, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (load_anchors(cluster_dir, input_data_path, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (knn_build_super_clusters(model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    return 0;
}

/**
 * knn_model_free() - Clean up resident KnnModel allocations.
 * @model: Pointer to KnnModel.
 */
void knn_model_free(
    KnnModel *model)
{
    if (model == NULL)
    {
        return;
    }

    knn_free_super_clusters(model);

    if (model->clusters != NULL)
    {
        for (int c = 0; c < model->num_clusters; c++)
        {
            if (model->clusters[c].anchor_data != NULL)
            {
                free(model->clusters[c].anchor_data);
                model->clusters[c].anchor_data = NULL;
            }
            if (model->clusters[c].members != NULL)
            {
                free(model->clusters[c].members);
                model->clusters[c].members = NULL;
            }
        } // for (int c = 0; ...)
        free(model->clusters);
        model->clusters = NULL;
    }

    if (model->dcc_matrix != NULL)
    {
        free(model->dcc_matrix);
        model->dcc_matrix = NULL;
    }

    if (model->frame_cluster_map != NULL)
    {
        free(model->frame_cluster_map);
        model->frame_cluster_map = NULL;
    }

    if (model->frame_r_anchor != NULL)
    {
        free(model->frame_r_anchor);
        model->frame_r_anchor = NULL;
    }
}

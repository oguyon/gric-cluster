/**
 * @file knn_loader.c
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_loader.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
 * calc_euclidean_dist() - Euclidean distance between two coordinate vectors.
 * @a: First vector.
 * @b: Second vector.
 * @n: Vector length.
 *
 * Return: Euclidean distance.
 */
static inline double calc_euclidean_dist(
    const void *restrict a,
    const void *restrict b,
    long                 n,
    int                  is_double)
{
    if (is_double)
    {
        const double *restrict da = (const double *)a;
        const double *restrict db = (const double *)b;
        double sum = 0.0;
        for (long i = 0; i < n; i++)
        {
            double diff = da[i] - db[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    else
    {
        const float *restrict fa = (const float *)a;
        const float *restrict fb = (const float *)b;
        float sum = 0.0f;
        for (long i = 0; i < n; i++)
        {
            float diff = fa[i] - fb[i];
            sum += diff * diff;
        }
        return (double)sqrtf(sum);
    }
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
                if (radius > model->clusters[c_id].radius)
                {
                    model->clusters[c_id].radius = radius;
                }
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
        if (gric_bin_read_header(f_bin, &hdr, &comment) == 0)
        {
            if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
            {
                if (fread(model->dcc_matrix, sizeof(double), (size_t)M * (size_t)M, f_bin) ==
                    (size_t)M * (size_t)M)
                {
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

                size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
                int dtype = model->is_double ? TDOUBLE : TFLOAT;
                for (int c = 0; c < model->num_clusters; c++)
                {
                    model->clusters[c].anchor_data =
                        malloc((size_t)model->frame_elements * elem_size);
                    if (model->clusters[c].anchor_data == NULL)
                    {
                        fits_close_file(fptr, &status);
                        return -1;
                    }
                    long f_anchor = (model->clusters[c].num_members > 0) ?
                                    (long)model->clusters[c].members[0].frame_id : 0;
                    long fpixel[3] = {1, 1, f_anchor + 1};
                    fits_read_pix(fptr, dtype, fpixel, model->frame_elements, NULL,
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

    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    for (int c = 0; c < model->num_clusters; c++)
    {
        model->clusters[c].anchor_data =
            malloc((size_t)model->frame_elements * elem_size);
        if (model->clusters[c].anchor_data == NULL)
        {
            free(offsets);
            fclose(f);
            return -1;
        }
        long f_anchor = (model->clusters[c].num_members > 0) ?
                        (long)model->clusters[c].members[0].frame_id : 0;
        fseeko(f, (off_t)offsets[f_anchor], SEEK_SET);
        if (model->is_double)
        {
            double *dptr = (double *)model->clusters[c].anchor_data;
            for (long k = 0; k < model->frame_elements; k++)
            {
                if (fscanf(f, "%lf", &dptr[k]) != 1)
                {
                    dptr[k] = 0.0;
                }
            }
        }
        else
        {
            float *fptr = (float *)model->clusters[c].anchor_data;
            for (long k = 0; k < model->frame_elements; k++)
            {
                if (fscanf(f, "%f", &fptr[k]) != 1)
                {
                    fptr[k] = 0.0f;
                }
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

            size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
            int ok = 1;

            if (!model->is_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                for (int c = 0; c < M; c++)
                {
                    model->clusters[c].anchor_data =
                        malloc((size_t)model->frame_elements * elem_size);
                    if (model->clusters[c].anchor_data == NULL ||
                        fread(model->clusters[c].anchor_data, sizeof(float),
                              (size_t)model->frame_elements, a_bin) !=
                              (size_t)model->frame_elements)
                    {
                        ok = 0;
                        break;
                    }
                }
            }
            else if (model->is_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
            {
                for (int c = 0; c < M; c++)
                {
                    model->clusters[c].anchor_data =
                        malloc((size_t)model->frame_elements * elem_size);
                    if (model->clusters[c].anchor_data == NULL ||
                        fread(model->clusters[c].anchor_data, sizeof(double),
                              (size_t)model->frame_elements, a_bin) !=
                              (size_t)model->frame_elements)
                    {
                        ok = 0;
                        break;
                    }
                }
            }
            else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                float *fbuf = (float *)malloc((size_t)model->frame_elements * sizeof(float));
                if (fbuf != NULL)
                {
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            malloc((size_t)model->frame_elements * elem_size);
                        if (model->clusters[c].anchor_data == NULL ||
                            fread(fbuf, sizeof(float), (size_t)model->frame_elements, a_bin) !=
                            (size_t)model->frame_elements)
                        {
                            ok = 0;
                            break;
                        }
                        double *dptr = (double *)model->clusters[c].anchor_data;
                        for (long k = 0; k < model->frame_elements; k++)
                        {
                            dptr[k] = (double)fbuf[k];
                        }
                    }
                    free(fbuf);
                }
                else
                {
                    ok = 0;
                }
            }
            else
            {
                double *dbuf = (double *)malloc((size_t)model->frame_elements * sizeof(double));
                if (dbuf != NULL)
                {
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            malloc((size_t)model->frame_elements * elem_size);
                        if (model->clusters[c].anchor_data == NULL ||
                            fread(dbuf, sizeof(double), (size_t)model->frame_elements, a_bin) !=
                            (size_t)model->frame_elements)
                        {
                            ok = 0;
                            break;
                        }
                        float *fptr = (float *)model->clusters[c].anchor_data;
                        for (long k = 0; k < model->frame_elements; k++)
                        {
                            fptr[k] = (float)dbuf[k];
                        }
                    }
                    free(dbuf);
                }
                else
                {
                    ok = 0;
                }
            }

            if (comment != NULL) free(comment);
            fclose(a_bin);
            if (ok) return 0;
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

            size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
            int dtype = model->is_double ? TDOUBLE : TFLOAT;
            for (int c = 0; c < M; c++)
            {
                model->clusters[c].anchor_data =
                    malloc((size_t)model->frame_elements * elem_size);
                if (model->clusters[c].anchor_data == NULL)
                {
                    fits_close_file(fptr, &status);
                    return -1;
                }

                long fpixel[3] = {1, 1, c + 1};
                fits_read_pix(fptr, dtype, fpixel, model->frame_elements, NULL,
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

            size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
            for (int c = 0; c < M; c++)
            {
                model->clusters[c].anchor_data =
                    malloc((size_t)model->frame_elements * elem_size);
                if (model->clusters[c].anchor_data == NULL)
                {
                    fclose(f);
                    return -1;
                }

                if (model->is_double)
                {
                    double *dptr = (double *)model->clusters[c].anchor_data;
                    for (long k = 0; k < model->frame_elements; k++)
                    {
                        if (fscanf(f, "%lf", &dptr[k]) != 1)
                        {
                            dptr[k] = 0.0;
                        }
                    }
                }
                else
                {
                    float *fptr = (float *)model->clusters[c].anchor_data;
                    for (long k = 0; k < model->frame_elements; k++)
                    {
                        if (fscanf(f, "%f", &fptr[k]) != 1)
                        {
                            fptr[k] = 0.0f;
                        }
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
 * compute_exact_frame_anchor_radii() - Compute distance to cluster anchor for each frame.
 * @input_data_path: Path to dataset file.
 * @model:           Pointer to KnnModel with populated anchors and assignments.
 *
 * Return: 0 on success, -1 on error.
 */
static int compute_exact_frame_anchor_radii(
    const char *input_data_path,
    KnnModel   *model)
{
    KnnFrameReader reader;
    if (knn_reader_open(&reader, input_data_path, model->total_dataset_frames,
                        model->frame_width, model->frame_height, model->is_double) != 0)
    {
        return -1;
    }

    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    void *fbuf = malloc((size_t)model->frame_elements * elem_size);
    if (fbuf == NULL)
    {
        knn_reader_close(&reader);
        return -1;
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        model->clusters[c].radius = 0.0;
        model->clusters[c].num_members = 0;
    }

    for (long i = 0; i < model->total_dataset_frames; i++)
    {
        if (knn_reader_read_frame(&reader, i, fbuf) == 0)
        {
            int c = model->frame_cluster_map[i];
            if (c >= 0 && c < model->num_clusters)
            {
                double r = calc_euclidean_dist(
                    fbuf,
                    model->clusters[c].anchor_data,
                    model->frame_elements,
                    model->is_double);
                model->frame_r_anchor[i] = (float)r;

                int slot = model->clusters[c].num_members++;
                model->clusters[c].members[slot].frame_id = (uint32_t)i;
                model->clusters[c].members[slot].r_anchor = (float)r;

                if (r > model->clusters[c].radius)
                {
                    model->clusters[c].radius = r;
                }
            }
        }
    } // for (long i = 0; ...)

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

    free(fbuf);
    knn_reader_close(&reader);
    return 0;
}

/**
 * load_knn_graph() - Opportunistically load pre-computed k-NN graph of dataset A.
 * @cluster_dir:     Path to the cluster directory.
 * @input_data_path: Path to dataset input file.
 * @model:           Pointer to KnnModel.
 *
 * Return: 0 if loaded or not present (non-fatal), -1 on critical parse error.
 */
static int load_knn_graph(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model)
{
    model->has_knn_graph = 0;
    model->graph_k = 0;
    model->graph_indices = NULL;
    model->graph_distances = NULL;
    model->graph_mutual_dists = NULL;

    if (cluster_dir == NULL || model == NULL || model->total_dataset_frames <= 0)
    {
        return 0;
    }

    char idx_path[2048];
    char dst_path[2048];
    snprintf(idx_path, sizeof(idx_path), "%s/knn_indices.bin", cluster_dir);
    snprintf(dst_path, sizeof(dst_path), "%s/knn_distances.bin", cluster_dir);

    FILE *fp_idx = fopen(idx_path, "rb");
    FILE *fp_dst = fopen(dst_path, "rb");
    if (fp_idx == NULL || fp_dst == NULL)
    {
        if (fp_idx != NULL)
        {
            fclose(fp_idx);
        }
        if (fp_dst != NULL)
        {
            fclose(fp_dst);
        }
        return 0;
    }

    gric_bin_header_t hdr_idx;
    gric_bin_header_t hdr_dst;
    if (gric_bin_read_header(fp_idx, &hdr_idx, NULL) != 0 ||
        gric_bin_read_header(fp_dst, &hdr_dst, NULL) != 0)
    {
        fclose(fp_idx);
        fclose(fp_dst);
        return 0;
    }

    if (hdr_idx.ndim < 2 || hdr_dst.ndim < 2 ||
        hdr_idx.dims[0] != (uint64_t)model->total_dataset_frames ||
        hdr_dst.dims[0] != (uint64_t)model->total_dataset_frames ||
        hdr_idx.dims[1] != hdr_dst.dims[1] ||
        hdr_idx.dims[1] == 0)
    {
        fclose(fp_idx);
        fclose(fp_dst);
        return 0;
    }

    uint64_t n_frames = hdr_idx.dims[0];
    uint64_t graph_k = hdr_idx.dims[1];
    uint64_t total_elems = n_frames * graph_k;

    uint32_t *indices = (uint32_t *)malloc(total_elems * sizeof(uint32_t));
    float    *distances = (float *)malloc(total_elems * sizeof(float));
    if (indices == NULL || distances == NULL)
    {
        if (indices != NULL)
        {
            free(indices);
        }
        if (distances != NULL)
        {
            free(distances);
        }
        fclose(fp_idx);
        fclose(fp_dst);
        return 0;
    }

    size_t r_idx = fread(indices, sizeof(uint32_t), total_elems, fp_idx);
    size_t r_dst = fread(distances, sizeof(float), total_elems, fp_dst);
    fclose(fp_idx);
    fclose(fp_dst);

    if (r_idx != total_elems || r_dst != total_elems)
    {
        free(indices);
        free(distances);
        return 0;
    }

    model->has_knn_graph = 1;
    model->graph_k = (int)graph_k;
    model->graph_indices = indices;
    model->graph_distances = distances;

    printf("  k-NN Metric Graph:   Loaded %lu frames x %lu neighbors from cluster directory\n",
           n_frames, graph_k);

    /* Opportunistically load precomputed mutual distances */
    char mut_path[2048];
    snprintf(mut_path, sizeof(mut_path), "%s/knn_mutual_dists.bin", cluster_dir);
    FILE *fp_mut = fopen(mut_path, "rb");
    if (fp_mut != NULL)
    {
        gric_bin_header_t hdr_mut;
        if (gric_bin_read_header(fp_mut, &hdr_mut, NULL) == 0)
        {
            uint64_t m_pairs = (graph_k * (graph_k - 1)) / 2;
            if (hdr_mut.ndim >= 2 &&
                hdr_mut.dims[0] == n_frames &&
                hdr_mut.dims[1] == m_pairs)
            {
                uint64_t total_mut = n_frames * m_pairs;
                float *mut_dists = (float *)malloc(total_mut * sizeof(float));
                if (mut_dists != NULL)
                {
                    if (fread(mut_dists, sizeof(float), total_mut, fp_mut) == total_mut)
                    {
                        model->graph_mutual_dists = mut_dists;
                        printf("  k-NN Mutual Dists:   Loaded %lu frames x %lu pairs from %s\n",
                               n_frames, m_pairs, mut_path);
                    }
                    else
                    {
                        free(mut_dists);
                    }
                }
            }
        }
        fclose(fp_mut);
    }

    /* If mutual distances missing on disk, compute and cache them */
    if (model->graph_mutual_dists == NULL && input_data_path != NULL && graph_k >= 2)
    {
        KnnFrameReader rdr;
        if (knn_reader_open(&rdr, input_data_path, (long)n_frames,
                            model->frame_width, model->frame_height, model->is_double) == 0)
        {
            long elem = model->frame_elements;
            size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
            size_t total_bytes = (size_t)n_frames * (size_t)elem * elem_size;
            if (total_bytes <= 1024ULL * 1024ULL * 1024ULL) // 1 GB allocation threshold
            {
                void *frames = malloc(total_bytes);
                if (frames != NULL)
                {
                    for (long f = 0; f < (long)n_frames; f++)
                    {
                        knn_reader_read_frame(&rdr, f,
                            (char *)frames + (size_t)f * (size_t)elem * elem_size);
                    }

                    uint64_t m_pairs = (graph_k * (graph_k - 1)) / 2;
                    uint64_t total_mut = n_frames * m_pairs;
                    float *mut_dists = (float *)calloc(total_mut, sizeof(float));
                    if (mut_dists != NULL)
                    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                        for (long u = 0; u < (long)n_frames; u++)
                        {
                            for (int i = 0; i < (int)graph_k; i++)
                            {
                                long id_i = (long)indices[u * (long)graph_k + i];
                                if (id_i < 0 || id_i >= (long)n_frames)
                                {
                                    continue;
                                }
                                const void *f_i =
                                    (const char *)frames + (size_t)id_i * (size_t)elem * elem_size;

                                for (int j = i + 1; j < (int)graph_k; j++)
                                {
                                    long id_j = (long)indices[u * (long)graph_k + j];
                                    if (id_j < 0 || id_j >= (long)n_frames)
                                    {
                                        continue;
                                    }
                                    const void *f_j = (const char *)frames +
                                        (size_t)id_j * (size_t)elem * elem_size;

                                    double d = calc_euclidean_dist(
                                        f_i, f_j, elem, model->is_double
                                    );

                                    long pair_idx = (long)i * (long)graph_k -
                                        ((long)i * (long)(i + 1)) / 2 + (long)(j - i - 1);
                                    mut_dists[(uint64_t)u * m_pairs + (uint64_t)pair_idx] =
                                        (float)d;
                                }
                            }
                        } // for (long u = 0; ...)

                        model->graph_mutual_dists = mut_dists;
                        printf("  k-NN Mutual Dists:   Computed %lu frames x %lu pairs\n",
                               n_frames, m_pairs);

                        FILE *fp_w = fopen(mut_path, "wb");
                        if (fp_w != NULL)
                        {
                            gric_bin_header_t hdr_w;
                            memset(&hdr_w, 0, sizeof(hdr_w));
                            hdr_w.file_type = GRIC_BIN_TYPE_GENERIC;
                            hdr_w.data_type = GRIC_BIN_DTYPE_FLOAT32;
                            hdr_w.flags = GRIC_BIN_FLAG_ROW_MAJOR;
                            hdr_w.ndim = 2;
                            hdr_w.dims[0] = n_frames;
                            hdr_w.dims[1] = m_pairs;
                            hdr_w.num_elements = total_mut;
                            hdr_w.data_bytes = total_mut * sizeof(float);
                            if (gric_bin_write_header(fp_w, &hdr_w,
                                                      "k-NN mutual distances [N x k*(k-1)/2]") == 0)
                            {
                                fwrite(mut_dists, sizeof(float), total_mut, fp_w);
                            }
                            fclose(fp_w);
                        }
                    }
                    free(frames);
                }
            }
            knn_reader_close(&rdr);
        }
    }

    return 0;
}

/**
 * knn_model_load() - Load Pass 1 clustering artifacts and prepare resident model.
 * @cluster_dir:      Directory containing Pass 1 artifacts.
 * @input_data_path:  Path to the input dataset.
 * @model:            Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_model_load(
    const char *cluster_dir,
    const char *input_data_path,
    KnnModel   *model,
    int         use_double)
{
    if (cluster_dir == NULL || input_data_path == NULL || model == NULL)
    {
        return -1;
    }

    memset(model, 0, sizeof(KnnModel));
    model->is_double = use_double;
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

    /* Compute exact pairwise inter-cluster anchor distances */
    for (int i = 0; i < model->num_clusters; i++)
    {
        model->dcc_matrix[i * model->num_clusters + i] = 0.0;
        for (int j = i + 1; j < model->num_clusters; j++)
        {
            double d = calc_euclidean_dist(
                model->clusters[i].anchor_data,
                model->clusters[j].anchor_data,
                model->frame_elements,
                model->is_double);
            model->dcc_matrix[i * model->num_clusters + j] = d;
            model->dcc_matrix[j * model->num_clusters + i] = d;
        }
    }

    /* Compute exact frame-to-anchor distances and exact cluster enclosing radii */
    if (compute_exact_frame_anchor_radii(input_data_path, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (knn_build_super_clusters(model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    /* Populate fast lookup pointer arrays for shared cluster locator */
    model->anchor_ptrs =
        (const void **)malloc((size_t)model->num_clusters * sizeof(const void *));
    model->cluster_radii =
        (double *)malloc((size_t)model->num_clusters * sizeof(double));
    if (model->anchor_ptrs != NULL && model->cluster_radii != NULL)
    {
        for (int c = 0; c < model->num_clusters; c++)
        {
            model->anchor_ptrs[c] = model->clusters[c].anchor_data;
            model->cluster_radii[c] = model->clusters[c].radius;
        }
    }

    /* Opportunistically load precomputed k-NN graph of dataset A */
    load_knn_graph(cluster_dir, input_data_path, model);

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

    if (model->graph_indices != NULL)
    {
        free(model->graph_indices);
        model->graph_indices = NULL;
    }

    if (model->graph_distances != NULL)
    {
        free(model->graph_distances);
        model->graph_distances = NULL;
    }

    if (model->graph_mutual_dists != NULL)
    {
        free(model->graph_mutual_dists);
        model->graph_mutual_dists = NULL;
    }

    if (model->anchor_ptrs != NULL)
    {
        free((void *)model->anchor_ptrs);
        model->anchor_ptrs = NULL;
    }

    if (model->cluster_radii != NULL)
    {
        free(model->cluster_radii);
        model->cluster_radii = NULL;
    }
}

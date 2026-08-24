/**
 * @file knn_loader.c
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_loader.h"
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
 * parse_dcc_file() - Load M x M inter-cluster distance matrix.
 * @path:  Path to dcc.txt.
 * @model: Pointer to KnnModel.
 *
 * Return: 0 on success, -1 on error.
 */
static int parse_dcc_file(
    const char *path,
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

    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        return 0;
    }

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
            }
        }
    } // while reading DCC

    fclose(f);
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

    // Try FITS first
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
    snprintf(memb_path, sizeof(memb_path), "%s/frame_membership.txt", cluster_dir);
    if (parse_membership_file(memb_path, model) != 0)
    {
        return -1;
    }

    char radii_path[2048];
    snprintf(radii_path, sizeof(radii_path), "%s/cluster_radii.txt", cluster_dir);
    parse_radii_file(radii_path, model);

    char dcc_path[2048];
    snprintf(dcc_path, sizeof(dcc_path), "%s/dcc.txt", cluster_dir);
    if (parse_dcc_file(dcc_path, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (load_anchors(cluster_dir, input_data_path, model) != 0)
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

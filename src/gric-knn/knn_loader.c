/**
 * @file knn_loader.c
 * @brief Loader for Pass 1 clustering artifacts into KnnModel resident structure.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_loader.h"
#include "knn_parser.h"
#include "knn_cache.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "gric_bin_io.h"
#include "gric_hash.h"
#include "framedistance.h"
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
        return framedist_double((const double *)a, (const double *)b, n);
    }
    return framedist_float((const float *)a, (const float *)b, n);
}


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

            size_t total_elements = (size_t)M * (size_t)model->frame_elements;
            int ok = 1;

            if (!model->is_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                model->anchor_matrix = malloc(total_elements * sizeof(float));
                if (model->anchor_matrix != NULL &&
                    fread(model->anchor_matrix, sizeof(float), total_elements, a_bin) ==
                    total_elements)
                {
                    float *base = (float *)model->anchor_matrix;
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            base + (size_t)c * (size_t)model->frame_elements;
                    }
                }
                else
                {
                    ok = 0;
                }
            }
            else if (model->is_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
            {
                model->anchor_matrix = malloc(total_elements * sizeof(double));
                if (model->anchor_matrix != NULL &&
                    fread(model->anchor_matrix, sizeof(double), total_elements, a_bin) ==
                    total_elements)
                {
                    double *base = (double *)model->anchor_matrix;
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            base + (size_t)c * (size_t)model->frame_elements;
                    }
                }
                else
                {
                    ok = 0;
                }
            }
            else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                /* File is float32, model requested double */
                model->anchor_matrix = malloc(total_elements * sizeof(double));
                float *fbuf = (float *)malloc(total_elements * sizeof(float));
                if (model->anchor_matrix != NULL && fbuf != NULL &&
                    fread(fbuf, sizeof(float), total_elements, a_bin) == total_elements)
                {
                    double *base = (double *)model->anchor_matrix;
                    for (size_t k = 0; k < total_elements; k++)
                    {
                        base[k] = (double)fbuf[k];
                    }
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            base + (size_t)c * (size_t)model->frame_elements;
                    }
                }
                else
                {
                    ok = 0;
                }
                if (fbuf != NULL)
                {
                    free(fbuf);
                }
            }
            else
            {
                /* File is float64, model requested float32 */
                model->anchor_matrix = malloc(total_elements * sizeof(float));
                double *dbuf = (double *)malloc(total_elements * sizeof(double));
                if (model->anchor_matrix != NULL && dbuf != NULL &&
                    fread(dbuf, sizeof(double), total_elements, a_bin) == total_elements)
                {
                    float *base = (float *)model->anchor_matrix;
                    for (size_t k = 0; k < total_elements; k++)
                    {
                        base[k] = (float)dbuf[k];
                    }
                    for (int c = 0; c < M; c++)
                    {
                        model->clusters[c].anchor_data =
                            base + (size_t)c * (size_t)model->frame_elements;
                    }
                }
                else
                {
                    ok = 0;
                }
                if (dbuf != NULL)
                {
                    free(dbuf);
                }
            }

            if (!ok && model->anchor_matrix != NULL)
            {
                free(model->anchor_matrix);
                model->anchor_matrix = NULL;
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
                  knn_compare_member_meta_radii);
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
    if (knn_parse_membership_file(memb_path, model) != 0)
    {
        snprintf(memb_path, sizeof(memb_path), "%s/frame_membership.txt", cluster_dir);
        if (knn_parse_membership_file(memb_path, model) != 0)
        {
            return -1;
        }
    }

    knn_parse_radii_file(cluster_dir, model);

    char log_path[2048];
    snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", cluster_dir);
    knn_parse_cluster_log(log_path, model);

    long total_members = 0;
    for (int c = 0; c < model->num_clusters; c++)
    {
        total_members += (long)model->clusters[c].num_members;
        if (model->clusters[c].radius <= 0.0 && model->model_rlim > 0.0)
        {
            model->clusters[c].radius = model->model_rlim;
        }
    }
    model->avg_cluster_size = (model->num_clusters > 0)
                              ? (double)total_members / (double)model->num_clusters
                              : 1.0;

    if (knn_parse_dcc_file(cluster_dir, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (load_anchors(cluster_dir, input_data_path, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    /* Compute exact pairwise inter-cluster anchor distances if not already loaded from DCC */
    if (model->dcc_matrix == NULL || model->dcc_sq16 == NULL)
    {
        if (model->dcc_matrix == NULL)
        {
            model->dcc_matrix = (double *)malloc((size_t)model->num_clusters *
                                                 (size_t)model->num_clusters * sizeof(double));
        }
        if (model->dcc_matrix != NULL)
        {
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
        }

        double s = (model->model_rlim > 0.0) ? (16384.0 / model->model_rlim) : 1000.0;
        model->dcc_sq16_scale = s;
        model->dcc_sq16_inv_scale = 1.0 / s;
        if (model->dcc_sq16 == NULL)
        {
            model->dcc_sq16 = (uint16_t *)malloc((size_t)model->num_clusters *
                                                 (size_t)model->num_clusters * sizeof(uint16_t));
        }
        if (model->dcc_sq16 != NULL && model->dcc_matrix != NULL)
        {
            for (int i = 0; i < model->num_clusters; i++)
            {
                model->dcc_sq16[i * model->num_clusters + i] = 0;
                for (int j = i + 1; j < model->num_clusters; j++)
                {
                    double d = model->dcc_matrix[i * model->num_clusters + j];
                    uint16_t q = (d <= 0.0) ? 0 :
                        ((d * s >= 65534.0) ? 65534 : (uint16_t)(d * s + 0.5));
                    model->dcc_sq16[i * model->num_clusters + j] = q;
                    model->dcc_sq16[j * model->num_clusters + i] = q;
                }
            }
        }
    }

    /* Compute exact frame-to-anchor distances and exact cluster enclosing radii */
    if (compute_exact_frame_anchor_radii(input_data_path, model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    if (knn_build_cluster_graph(model) != 0)
    {
        knn_model_free(model);
        return -1;
    }

    /* Populate fast lookup pointer arrays for shared cluster locator */
    model->anchor_ptrs =
        (const void **)malloc((size_t)model->num_clusters * sizeof(const void *));
    model->cluster_radii =
        (double *)malloc((size_t)model->num_clusters * sizeof(double));
    model->cluster_radii_sq16 =
        (uint16_t *)malloc((size_t)model->num_clusters * sizeof(uint16_t));
    if (model->anchor_ptrs != NULL && model->cluster_radii != NULL)
    {
        double scale = (model->dcc_sq16_scale > 0.0) ? model->dcc_sq16_scale :
                       ((model->model_rlim > 0.0) ? (16384.0 / model->model_rlim) : 1000.0);
        for (int c = 0; c < model->num_clusters; c++)
        {
            model->anchor_ptrs[c] = model->clusters[c].anchor_data;
            model->cluster_radii[c] = model->clusters[c].radius;
            if (model->cluster_radii_sq16 != NULL)
            {
                model->cluster_radii_sq16[c] =
                    (uint16_t)(model->clusters[c].radius * scale + 0.5);
            }
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

    knn_free_cluster_graph(model);

    if (model->anchor_matrix != NULL)
    {
        free(model->anchor_matrix);
        model->anchor_matrix = NULL;
    }
    else if (model->clusters != NULL)
    {
        for (int c = 0; c < model->num_clusters; c++)
        {
            if (model->clusters[c].anchor_data != NULL)
            {
                free(model->clusters[c].anchor_data);
                model->clusters[c].anchor_data = NULL;
            }
        }
    }

    if (model->clusters != NULL)
    {
        for (int c = 0; c < model->num_clusters; c++)
        {
            if (model->clusters[c].members != NULL)
            {
                free(model->clusters[c].members);
                model->clusters[c].members = NULL;
            }
            if (model->clusters[c].rabitq_meta != NULL)
            {
                free(model->clusters[c].rabitq_meta);
                model->clusters[c].rabitq_meta = NULL;
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

    if (model->dcc_sq16 != NULL)
    {
        free(model->dcc_sq16);
        model->dcc_sq16 = NULL;
    }

    if (model->cluster_radii_sq16 != NULL)
    {
        free(model->cluster_radii_sq16);
        model->cluster_radii_sq16 = NULL;
    }

    if (model->sq8_dataset_buffer != NULL)
    {
        free(model->sq8_dataset_buffer);
        model->sq8_dataset_buffer = NULL;
    }

    if (model->anchor_sq8_buffer != NULL)
    {
        free(model->anchor_sq8_buffer);
        model->anchor_sq8_buffer = NULL;
    }

    if (model->sq16_dataset_buffer != NULL)
    {
        free(model->sq16_dataset_buffer);
        model->sq16_dataset_buffer = NULL;
    }

    if (model->sq16_transposed_buffer != NULL)
    {
        free(model->sq16_transposed_buffer);
        model->sq16_transposed_buffer = NULL;
    }

    if (model->anchor_sq16_buffer != NULL)
    {
        free(model->anchor_sq16_buffer);
        model->anchor_sq16_buffer = NULL;
    }

    if (model->rq8_dataset_buffer != NULL)
    {
        free(model->rq8_dataset_buffer);
        model->rq8_dataset_buffer = NULL;
    }

    if (model->rq8_transposed_buffer != NULL)
    {
        free(model->rq8_transposed_buffer);
        model->rq8_transposed_buffer = NULL;
    }

    if (model->pq_codebook != NULL)
    {
        pq_codebook_free(model->pq_codebook);
        model->pq_codebook = NULL;
    }

    if (model->pq_dataset_buffer != NULL)
    {
        free(model->pq_dataset_buffer);
        model->pq_dataset_buffer = NULL;
    }

    if (model->pq_transposed_buffer != NULL)
    {
        free(model->pq_transposed_buffer);
        model->pq_transposed_buffer = NULL;
    }

    rabitq_free_params(&model->rabitq_params);

    if (model->rabitq_meta_buffer != NULL)
    {
        free(model->rabitq_meta_buffer);
        model->rabitq_meta_buffer = NULL;
    }

    if (model->rabitq_dataset_buffer != NULL)
    {
        free(model->rabitq_dataset_buffer);
        model->rabitq_dataset_buffer = NULL;
    }

    if (model->rabitq_transposed_buffer != NULL)
    {
        free(model->rabitq_transposed_buffer);
        model->rabitq_transposed_buffer = NULL;
    }

    if (model->dataset_mmap_addr != NULL)
    {
        munmap(model->dataset_mmap_addr, model->dataset_mmap_size);
        model->dataset_mmap_addr = NULL;
        model->dataset_buffer = NULL;
    }
    else if (model->dataset_buffer != NULL)
    {
        free(model->dataset_buffer);
        model->dataset_buffer = NULL;
    }

    if (model->ivf_dataset_buffer != NULL)
    {
        free(model->ivf_dataset_buffer);
        model->ivf_dataset_buffer = NULL;
    }

    if (model->has_profile)
    {
        gric_profile_free(&model->profile);
        model->has_profile = 0;
    }

    if (model->frame_to_unique_map != NULL)
    {
        free(model->frame_to_unique_map);
        model->frame_to_unique_map = NULL;
    }
}

/**
 * knn_model_build_or_load_sq8() - Build or load quantized SQ8 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */

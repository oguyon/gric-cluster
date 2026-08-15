/**
 * @file knn_writer.c
 * @brief Output serialization for gric-knn results into FITS or ASCII formats.
 */

#include "knn_writer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

/**
 * write_ascii_results() - Write results as formatted ASCII table.
 * @path:    Output filename.
 * @config:  Active KnnConfig.
 * @model:   Active KnnModel.
 * @results: Computed KnnResults.
 *
 * Return: 0 on success, -1 on error.
 */
static int write_ascii_results(
    const char       *path,
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results)
{
    FILE *f = fopen(path, "w");
    if (f == NULL)
    {
        fprintf(stderr, "Error: Could not open output file '%s' for writing\n", path);
        return -1;
    }

    fprintf(f, "# gric-knn results: k = %d, total_queries = %ld\n",
            config->k, model->total_dataset_frames);
    fprintf(f, "# Columns: query_frame_id  [neighbor_1 dist_1  neighbor_2 dist_2 ...]\n");

    long N = model->total_dataset_frames;
    int k = config->k;

    for (long i = 0; i < N; i++)
    {
        fprintf(f, "%-8ld", i);
        for (int j = 0; j < k; j++)
        {
            int n_id = results->indices[i * k + j];
            double d = results->distances[i * k + j];
            if (n_id < 0 || isnan(d))
            {
                fprintf(f, "  %-8d %12.6f", -1, -1.0);
            }
            else
            {
                fprintf(f, "  %-8d %12.6f", n_id, d);
            }
        }
        fprintf(f, "\n");
    } // for (long i = 0; ...)

    fclose(f);
    return 0;
}

#ifdef USE_CFITSIO
/**
 * write_fits_results() - Write results as dual FITS cubes (indices and distances).
 * @out_indices_path:   Output path for knn_indices.fits.
 * @out_distances_path: Output path for knn_distances.fits.
 * @config:             Active KnnConfig.
 * @model:              Active KnnModel.
 * @results:            Computed KnnResults.
 *
 * Return: 0 on success, -1 on error.
 */
static int write_fits_results(
    const char       *out_indices_path,
    const char       *out_distances_path,
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results)
{
    int status = 0;
    fitsfile *f_idx = NULL;
    fitsfile *f_dst = NULL;

    long N = model->total_dataset_frames;
    long k = config->k;
    long naxes[2] = {k, N};
    long n_total_elements = k * N;

    // Remove existing files if any by prepending '!'
    char path_idx_clobber[4100];
    char path_dst_clobber[4100];
    snprintf(path_idx_clobber, sizeof(path_idx_clobber), "!%s", out_indices_path);
    snprintf(path_dst_clobber, sizeof(path_dst_clobber), "!%s", out_distances_path);

    fits_create_file(&f_idx, path_idx_clobber, &status);
    if (status != 0)
    {
        fprintf(stderr, "Error: Could not create FITS file '%s' (CFITSIO error %d)\n",
                out_indices_path, status);
        return -1;
    }

    fits_create_img(f_idx, LONG_IMG, 2, naxes, &status);
    long fpixel[2] = {1, 1};

    fits_write_pix(f_idx, TINT, fpixel, n_total_elements, results->indices, &status);
    fits_close_file(f_idx, &status);

    status = 0;
    fits_create_file(&f_dst, path_dst_clobber, &status);
    if (status != 0)
    {
        fprintf(stderr, "Error: Could not create FITS file '%s' (CFITSIO error %d)\n",
                out_distances_path, status);
        return -1;
    }

    fits_create_img(f_dst, DOUBLE_IMG, 2, naxes, &status);
    fits_write_pix(f_dst, TDOUBLE, fpixel, n_total_elements, results->distances, &status);
    fits_close_file(f_dst, &status);

    return (status == 0) ? 0 : -1;
}
#endif // USE_CFITSIO

/**
 * knn_write_results() - Save k-NN results into configured format.
 * @config:  Active KnnConfig.
 * @model:   Active KnnModel.
 * @results: Computed KnnResults.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_write_results(
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results)
{
    if (config == NULL || model == NULL || results == NULL)
    {
        return -1;
    }

    int use_fits = 0;
    if (config->output_format == KNN_FORMAT_FITS)
    {
        use_fits = 1;
    }
    else if (config->output_format == KNN_FORMAT_AUTO)
    {
        use_fits = model->is_fits_input;
    }

    char final_out_path[2048];
    if (config->output_path != NULL)
    {
        snprintf(final_out_path, sizeof(final_out_path), "%s", config->output_path);
    }
    else
    {
        if (use_fits)
        {
            snprintf(final_out_path, sizeof(final_out_path), "%s/knn_k%d.fits",
                     config->cluster_dir, config->k);
        }
        else
        {
            snprintf(final_out_path, sizeof(final_out_path), "%s/knn_results.txt",
                     config->cluster_dir);
        }
    }

    if (use_fits)
    {
#ifdef USE_CFITSIO
        char idx_path[4096];
        char dst_path[4096];

        size_t len = strlen(final_out_path);
        if (len >= 5 && strcasecmp(final_out_path + len - 5, ".fits") == 0)
        {
            char base[2048];
            size_t copy_len = len - 5;
            if (copy_len >= sizeof(base))
            {
                copy_len = sizeof(base) - 1;
            }
            memcpy(base, final_out_path, copy_len);
            base[copy_len] = '\0';
            snprintf(idx_path, sizeof(idx_path), "%s_indices.fits", base);
            snprintf(dst_path, sizeof(dst_path), "%s_distances.fits", base);
        }
        else
        {
            snprintf(idx_path, sizeof(idx_path), "%s/knn_indices.fits", final_out_path);
            snprintf(dst_path, sizeof(dst_path), "%s/knn_distances.fits", final_out_path);
        }

        printf("Writing FITS outputs:\n  - %s\n  - %s\n", idx_path, dst_path);
        return write_fits_results(idx_path, dst_path, config, model, results);
#else
        fprintf(stderr, "Warning: CFITSIO not enabled. Falling back to ASCII output.\n");
        return write_ascii_results(final_out_path, config, model, results);
#endif
    }
    else
    {
        printf("Writing ASCII output: %s\n", final_out_path);
        return write_ascii_results(final_out_path, config, model, results);
    }
}

/**
 * @file dimdensity_loader.c
 * @brief Out-of-core and memory loader for k-NN distance datasets.
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_loader.h"
#include "shared/gric_bin_io.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

/**
 * compare_doubles() - Qsort comparator for sorting neighbor distances.
 * @a: Pointer to first double.
 * @b: Pointer to second double.
 *
 * Return: -1 if *a < *b, 1 if *a > *b, 0 if equal.
 */
static int compare_doubles(
    const void *a,
    const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
    {
        return -1;
    }
    if (da > db)
    {
        return 1;
    }
    return 0;
}

/**
 * sort_distance_rows() - Ensure each sample row has ascending distances.
 * @data: Pointer to KnnDistanceData.
 */
static void sort_distance_rows(
    KnnDistanceData *data)
{
    if (data == NULL || data->distances == NULL || data->k_available <= 1)
    {
        return;
    }

    uint64_t n = data->num_samples;
    int k = data->k_available;

    for (uint64_t i = 0; i < n; i++)
    {
        double *row = &data->distances[i * (uint64_t)k];
        qsort(row, (size_t)k, sizeof(double), compare_doubles);
    }
}

/**
 * is_directory_path() - Check if a given path is a directory.
 * @path: Filesystem path to inspect.
 *
 * Return: 1 if directory, 0 otherwise.
 */
static int is_directory_path(
    const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
    {
        return 1;
    }
    return 0;
}

/**
 * resolve_input_file() - Resolve actual distance file path from file or dir.
 * @input_path: User-provided input path.
 * @resolved:   Buffer to store resolved path.
 * @max_len:    Maximum buffer size.
 *
 * Return: 0 on success, -1 on error.
 */
static int resolve_input_file(
    const char *input_path,
    char       *resolved,
    size_t      max_len)
{
    if (input_path == NULL || resolved == NULL || max_len == 0)
    {
        return -1;
    }

    if (is_directory_path(input_path))
    {
        // Try candidate files in directory
        const char *candidates[] = {
            "knn_distances.bin",
            "knn_distances.fits",
            "knn_k10_distances.fits",
            "knn_results.txt",
            NULL
        };

        for (int i = 0; candidates[i] != NULL; i++)
        {
            snprintf(resolved, max_len, "%s/%s", input_path, candidates[i]);
            struct stat st;
            if (stat(resolved, &st) == 0 && S_ISREG(st.st_mode))
            {
                return 0;
            }
        }

        fprintf(stderr, "Error: No k-NN distance file found in directory '%s'\n", input_path);
        return -1;
    }

    struct stat st;
    if (stat(input_path, &st) == 0)
    {
        snprintf(resolved, max_len, "%s", input_path);
        return 0;
    }

    fprintf(stderr, "Error: Cannot access input path '%s'\n", input_path);
    return -1;
}

/**
 * load_bin_distances() - Load distances from GRIC binary file.
 * @path:    Path to binary file.
 * @data:    Pointer to KnnDistanceData structure.
 * @verbose: Verbosity level.
 *
 * Return: 0 on success, -1 on error.
 */
static int load_bin_distances(
    const char      *path,
    KnnDistanceData *data,
    int              verbose)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open binary file '%s'\n", path);
        return -1;
    }

    gric_bin_header_t hdr;
    char *comment = NULL;
    if (gric_bin_read_header(fp, &hdr, &comment) != 0)
    {
        fprintf(stderr, "Error: Invalid GRIC binary header in '%s'\n", path);
        fclose(fp);
        return -1;
    }

    if (comment != NULL)
    {
        if (verbose >= 2)
        {
            printf("  Binary Header Comment: %s\n", comment);
        }
        free(comment);
    }

    if (hdr.ndim < 2 || hdr.dims[0] == 0 || hdr.dims[1] == 0)
    {
        fprintf(stderr, "Error: Binary file must be at least 2D [N x k]\n");
        fclose(fp);
        return -1;
    }

    uint64_t n = hdr.dims[0];
    int k = (int)hdr.dims[1];
    uint64_t total = n * (uint64_t)k;

    data->distances = (double *)malloc(total * sizeof(double));
    if (data->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failure for %lu distances\n", total);
        fclose(fp);
        return -1;
    }

    if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
    {
        float *f32_buf = (float *)malloc(total * sizeof(float));
        if (f32_buf == NULL)
        {
            free(data->distances);
            data->distances = NULL;
            fclose(fp);
            return -1;
        }

        size_t nread = fread(f32_buf, sizeof(float), total, fp);
        if (nread != total)
        {
            fprintf(stderr, "Error: Truncated binary read (expected %lu, got %zu)\n",
                    total, nread);
            free(f32_buf);
            free(data->distances);
            data->distances = NULL;
            fclose(fp);
            return -1;
        }

        for (uint64_t i = 0; i < total; i++)
        {
            data->distances[i] = (double)f32_buf[i];
        }
        free(f32_buf);
    }
    else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
    {
        size_t nread = fread(data->distances, sizeof(double), total, fp);
        if (nread != total)
        {
            fprintf(stderr, "Error: Truncated binary read (expected %lu, got %zu)\n",
                    total, nread);
            free(data->distances);
            data->distances = NULL;
            fclose(fp);
            return -1;
        }
    }
    else
    {
        fprintf(stderr, "Error: Unsupported binary data type %d\n", hdr.data_type);
        free(data->distances);
        data->distances = NULL;
        fclose(fp);
        return -1;
    }

    fclose(fp);
    data->num_samples = n;
    data->k_available = k;
    data->is_fits = 0;
    data->resolved_path = strdup(path);

    sort_distance_rows(data);
    return 0;
}

#ifdef USE_CFITSIO
/**
 * load_fits_distances() - Load distances from FITS file.
 * @path:    Path to FITS file.
 * @data:    Pointer to KnnDistanceData structure.
 * @verbose: Verbosity level.
 *
 * Return: 0 on success, -1 on error.
 */
static int load_fits_distances(
    const char      *path,
    KnnDistanceData *data,
    int              verbose)
{
    int status = 0;
    fitsfile *fptr = NULL;

    fits_open_file(&fptr, path, READONLY, &status);
    if (status != 0 || fptr == NULL)
    {
        fprintf(stderr, "Error: Could not open FITS file '%s' (status %d)\n", path, status);
        return -1;
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 2, naxes, &status);

    if (status != 0 || naxis < 2 || naxes[0] <= 0 || naxes[1] <= 0)
    {
        fprintf(stderr, "Error: Invalid 2D image dimensions in FITS file '%s'\n", path);
        fits_close_file(fptr, &status);
        return -1;
    }

    // FITS convention: naxes[0] is width (k), naxes[1] is height (N)
    int k = (int)naxes[0];
    uint64_t n = (uint64_t)naxes[1];
    uint64_t total = n * (uint64_t)k;

    data->distances = (double *)malloc(total * sizeof(double));
    if (data->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failure for FITS distances\n");
        fits_close_file(fptr, &status);
        return -1;
    }

    long fpixel[2] = {1, 1};
    fits_read_pix(fptr, TDOUBLE, fpixel, (long)total, NULL, data->distances, NULL, &status);
    fits_close_file(fptr, &status);

    if (status != 0)
    {
        fprintf(stderr, "Error: Failed to read FITS pixel data (status %d)\n", status);
        free(data->distances);
        data->distances = NULL;
        return -1;
    }

    data->num_samples = n;
    data->k_available = k;
    data->is_fits = 1;
    data->resolved_path = strdup(path);

    if (verbose >= 2)
    {
        printf("  Loaded FITS: %lu samples, %d neighbors per sample\n", n, k);
    }

    sort_distance_rows(data);
    return 0;
}
#endif // USE_CFITSIO

/**
 * load_ascii_distances() - Load distances from formatted ASCII text table.
 * @path:    Path to ASCII file.
 * @data:    Pointer to KnnDistanceData structure.
 * @verbose: Verbosity level.
 *
 * Return: 0 on success, -1 on error.
 */
static int load_ascii_distances(
    const char      *path,
    KnnDistanceData *data,
    int              verbose)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open ASCII file '%s'\n", path);
        return -1;
    }

    char line[65536];
    uint64_t count = 0;
    int k_detected = 0;

    // First pass: detect k and count sample rows
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        char *p = line;
        while (isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '#' || *p == '\0')
        {
            continue;
        }

        if (k_detected == 0)
        {
            // Tokenize first data line to determine column structure
            char line_copy[65536];
            strncpy(line_copy, p, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';

            int token_count = 0;
            char *tok = strtok(line_copy, " \t\r\n");
            while (tok != NULL)
            {
                token_count++;
                tok = strtok(NULL, " \t\r\n");
            }

            // If format is [id, n1, d1, n2, d2, ...], token_count is 1 + 2*k
            if (token_count > 1 && (token_count - 1) % 2 == 0)
            {
                k_detected = (token_count - 1) / 2;
            }
            else if (token_count > 1)
            {
                // Format is [id, d1, d2, ...] or [d1, d2, ...]
                k_detected = token_count - 1;
            }
            else
            {
                k_detected = 1;
            }
        } // if (k_detected == 0)

        count++;
    } // while (fgets)

    if (count == 0 || k_detected <= 0)
    {
        fprintf(stderr, "Error: No valid sample rows found in ASCII file '%s'\n", path);
        fclose(fp);
        return -1;
    }

    rewind(fp);
    uint64_t total = count * (uint64_t)k_detected;
    data->distances = (double *)malloc(total * sizeof(double));
    if (data->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failure for ASCII distances\n");
        fclose(fp);
        return -1;
    }

    uint64_t row_idx = 0;
    while (fgets(line, sizeof(line), fp) != NULL && row_idx < count)
    {
        char *p = line;
        while (isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '#' || *p == '\0')
        {
            continue;
        }

        char *tok = strtok(p, " \t\r\n");
        if (tok == NULL)
        {
            continue;
        }

        // Check if line format is [id, n1, d1, n2, d2, ...]
        // By reading tokens sequentially
        double *row_dist = &data->distances[row_idx * (uint64_t)k_detected];

        // Skip sample id token
        for (int j = 0; j < k_detected; j++)
        {
            char *tok_neighbor = strtok(NULL, " \t\r\n");
            char *tok_dist = strtok(NULL, " \t\r\n");

            if (tok_neighbor != NULL && tok_dist != NULL)
            {
                row_dist[j] = atof(tok_dist);
            }
            else if (tok_neighbor != NULL)
            {
                row_dist[j] = atof(tok_neighbor);
            }
            else
            {
                row_dist[j] = 0.0;
            }
        }

        row_idx++;
    } // while (fgets)

    fclose(fp);
    data->num_samples = row_idx;
    data->k_available = k_detected;
    data->is_fits = 0;
    data->resolved_path = strdup(path);

    if (verbose >= 2)
    {
        printf("  Loaded ASCII: %lu samples, %d neighbors per sample\n",
               data->num_samples, data->k_available);
    }

    sort_distance_rows(data);
    return 0;
}

/**
 * dimdensity_load_distances() - Load k-NN distance matrix from file or dir.
 * @input_path: File or directory path.
 * @data:       Destination KnnDistanceData structure.
 * @verbose:    Verbosity level.
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_load_distances(
    const char      *input_path,
    KnnDistanceData *data,
    int              verbose)
{
    if (input_path == NULL || data == NULL)
    {
        return -1;
    }

    memset(data, 0, sizeof(KnnDistanceData));
    char resolved[4096];
    if (resolve_input_file(input_path, resolved, sizeof(resolved)) != 0)
    {
        return -1;
    }

    if (verbose >= 1)
    {
        printf("Loading k-NN distances from: %s\n", resolved);
    }

    size_t len = strlen(resolved);

    // Check .fits
    if ((len >= 5 && strcasecmp(resolved + len - 5, ".fits") == 0) ||
        (len >= 8 && strcasecmp(resolved + len - 8, ".fits.gz") == 0))
    {
#ifdef USE_CFITSIO
        return load_fits_distances(resolved, data, verbose);
#else
        fprintf(stderr, "Error: CFITSIO support not compiled in for '%s'\n", resolved);
        return -1;
#endif
    }

    // Check .bin
    if (len >= 4 && strcasecmp(resolved + len - 4, ".bin") == 0)
    {
        return load_bin_distances(resolved, data, verbose);
    }

    // Default to ASCII table
    return load_ascii_distances(resolved, data, verbose);
}

/**
 * dimdensity_free_distances() - Free KnnDistanceData allocations.
 * @data: Pointer to KnnDistanceData structure.
 */
void dimdensity_free_distances(
    KnnDistanceData *data)
{
    if (data == NULL)
    {
        return;
    }

    if (data->distances != NULL)
    {
        free(data->distances);
        data->distances = NULL;
    }
    if (data->resolved_path != NULL)
    {
        free(data->resolved_path);
        data->resolved_path = NULL;
    }
    data->num_samples = 0;
    data->k_available = 0;
}

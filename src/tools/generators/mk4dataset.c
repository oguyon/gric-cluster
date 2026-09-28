/**
 * @file mk4dataset.c
 * @brief Native C generator for pre-synchronized 4-dataset reconstruction benchmarks.
 *
 * Generates four coupled benchmark datasets (A, B, C, D_true) without any external dependencies:
 *   - Dataset A: Training input vectors [N x D_A] (noisy observation trajectory)
 *   - Dataset B: Training target vectors [N x D_B] (coupled state space trajectory)
 *                * Strictly synchronized sample-by-sample with A at t_i = 4*pi*i/(N-1)
 *   - Dataset C: Query input vectors [M x D_A] (query trajectory in observation space)
 *   - Dataset D_true: Ground truth target vectors [M x D_B] for quality validation
 */

#define _POSIX_C_SOURCE 200809L
#include "cli_colors.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * rand_normal() - Generate standard Gaussian random variable via Box-Muller transform.
 * @mean:   Desired mean.
 * @stddev: Desired standard deviation.
 *
 * Return: Normally distributed random float.
 */
static float rand_normal(
    float mean,
    float stddev)
{
    static int   have_spare = 0;
    static float spare      = 0.0f;

    if (have_spare)
    {
        have_spare = 0;
        return mean + stddev * spare;
    }

    have_spare = 1;
    float u = 0.0f;
    float v = 0.0f;
    float s = 0.0f;

    do
    {
        u = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + stddev * (u * s);
}

/**
 * ensure_dir_exists() - Create directory hierarchy if it does not already exist.
 * @path: Directory path.
 *
 * Return: 0 on success, non-zero on failure.
 */
static int ensure_dir_exists(
    const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0)
    {
        if (S_ISDIR(st.st_mode))
        {
            return 0;
        }
        return -1;
    }

    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0)
    {
        return 0;
    }

    for (char *p = tmp + 1; *p != '\0'; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
}

/**
 * write_ascii_dataset() - Write 2D float array to ASCII space-delimited file.
 * @path: Output file path.
 * @data: Float array buffer.
 * @rows: Sample count.
 * @cols: Coordinate dimension.
 *
 * Return: 0 on success, non-zero on error.
 */
static int write_ascii_dataset(
    const char  *path,
    const float *data,
    uint32_t     rows,
    uint32_t     cols)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: Unable to open '%s' for writing: %s\n",
                path, strerror(errno));
        return -1;
    }

    for (uint32_t r = 0; r < rows; r++)
    {
        const float *row_ptr = data + ((size_t)r * cols);
        for (uint32_t c = 0; c < cols; c++)
        {
            fprintf(fp, (c == cols - 1) ? "%.6f\n" : "%.6f ", (double)row_ptr[c]);
        }
    }

    fclose(fp);
    return 0;
}

/**
 * load_ascii_dataset() - Load 2D float matrix from ASCII space-delimited text file.
 * @path:     Input file path.
 * @out_data: Output dynamically allocated array.
 * @out_rows: Output sample count.
 * @out_cols: Output coordinate dimension.
 *
 * Return: 0 on success, non-zero on error.
 */
static int load_ascii_dataset(
    const char  *path,
    float      **out_data,
    uint32_t    *out_rows,
    uint32_t    *out_cols)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Unable to open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    size_t   capacity   = 1000;
    uint32_t cols       = 0;
    uint32_t rows       = 0;
    float   *data       = NULL;
    char    *line       = NULL;
    size_t   len        = 0;
    ssize_t  read_bytes = 0;

    while ((read_bytes = getline(&line, &len, fp)) != -1)
    {
        /* Skip comments and empty lines */
        char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }

        if (cols == 0)
        {
            /* Determine column count on first valid row */
            char *scan_p = p;
            char *endptr = NULL;
            while (*scan_p != '\0' && *scan_p != '\n' && *scan_p != '\r')
            {
                (void)strtod(scan_p, &endptr);
                if (scan_p == endptr)
                {
                    break;
                }
                cols++;
                scan_p = endptr;
            }
            if (cols == 0)
            {
                continue;
            }
            data = (float *)malloc(capacity * (size_t)cols * sizeof(float));
            if (!data)
            {
                free(line);
                fclose(fp);
                return -1;
            }
        }

        if (rows >= capacity)
        {
            capacity *= 2;
            float *new_data = (float *)realloc(data, capacity * (size_t)cols * sizeof(float));
            if (!new_data)
            {
                free(data);
                free(line);
                fclose(fp);
                return -1;
            }
            data = new_data;
        }

        float *row_dst = data + ((size_t)rows * cols);
        char  *scan_p  = p;
        char  *endptr  = NULL;
        uint32_t c     = 0;
        for (; c < cols; c++)
        {
            row_dst[c] = (float)strtod(scan_p, &endptr);
            if (scan_p == endptr)
            {
                break;
            }
            scan_p = endptr;
        }
        if (c == cols)
        {
            rows++;
        }
    } // while (getline)

    free(line);
    fclose(fp);

    if (rows == 0 || cols == 0)
    {
        if (data)
        {
            free(data);
        }
        return -1;
    }

    *out_data = data;
    *out_rows = rows;
    *out_cols = cols;
    return 0;
}

/**
 * evaluate_reconstruction() - Assess fidelity of reconstructed target D vs ground truth D_true.
 * @path_recon:   Path to reconstructed dataset D.
 * @path_true:    Path to ground-truth dataset D_true.
 * @target_noise: Injected Gaussian noise sigma on target dataset B (or 0 if unknown).
 *
 * Return: 0 on success, non-zero on error.
 */
static int evaluate_reconstruction(
    const char *path_recon,
    const char *path_true,
    double      target_noise)
{
    float   *data_recon = NULL;
    float   *data_true  = NULL;
    uint32_t rows_recon = 0;
    uint32_t cols_recon = 0;
    uint32_t rows_true  = 0;
    uint32_t cols_true  = 0;

    if (load_ascii_dataset(path_recon, &data_recon, &rows_recon, &cols_recon) != 0)
    {
        fprintf(stderr, "Error loading reconstructed dataset '%s'\n", path_recon);
        return 1;
    }
    if (load_ascii_dataset(path_true, &data_true, &rows_true, &cols_true) != 0)
    {
        fprintf(stderr, "Error loading ground truth dataset '%s'\n", path_true);
        free(data_recon);
        return 1;
    }

    if (cols_recon != cols_true)
    {
        fprintf(stderr, "Error: Coordinate dimension mismatch (recon: %uD, true: %uD)\n",
                cols_recon, cols_true);
        free(data_recon);
        free(data_true);
        return 1;
    }

    uint32_t eval_rows = (rows_recon < rows_true) ? rows_recon : rows_true;
    if (eval_rows == 0)
    {
        fprintf(stderr, "Error: Zero valid samples to evaluate.\n");
        free(data_recon);
        free(data_true);
        return 1;
    }

    double total_sq_err  = 0.0;
    double total_abs_err = 0.0;
    double max_abs_err   = 0.0;
    double true_min      = data_true[0];
    double true_max      = data_true[0];
    double sum_true      = 0.0;

    uint64_t total_elements = (uint64_t)eval_rows * cols_recon;

    for (uint32_t r = 0; r < eval_rows; r++)
    {
        for (uint32_t c = 0; c < cols_recon; c++)
        {
            size_t idx = (size_t)r * cols_recon + c;
            double yt  = (double)data_true[idx];
            double yr  = (double)data_recon[idx];

            if (yt < true_min)
            {
                true_min = yt;
            }
            if (yt > true_max)
            {
                true_max = yt;
            }

            sum_true += yt;

            double diff     = yr - yt;
            double abs_diff = fabs(diff);
            total_sq_err  += diff * diff;
            total_abs_err += abs_diff;
            if (abs_diff > max_abs_err)
            {
                max_abs_err = abs_diff;
            }
        }
    }

    double mean_true  = sum_true / (double)total_elements;

    double ss_tot = 0.0;

    for (uint32_t r = 0; r < eval_rows; r++)
    {
        for (uint32_t c = 0; c < cols_recon; c++)
        {
            size_t idx = (size_t)r * cols_recon + c;
            double yt  = (double)data_true[idx];
            double dt  = yt - mean_true;

            ss_tot += dt * dt;
        }
    }

    double rmse      = sqrt(total_sq_err / (double)total_elements);
    double mae       = total_abs_err / (double)total_elements;
    double range     = (true_max > true_min) ? (true_max - true_min) : 1.0;
    double nrmse_pct = (rmse / range) * 100.0;
    double r2        = (ss_tot > 1e-12) ? (1.0 - (total_sq_err / ss_tot)) : 1.0;

    const char *rating = (r2 >= 0.95) ? "EXCELLENT" :
                         (r2 >= 0.85) ? "GOOD" :
                         (r2 >= 0.70) ? "FAIR" : "POOR";
    const char *rating_col = (r2 >= 0.85) ? ansi_bold_green :
                             (r2 >= 0.70) ? ansi_color_yellow : ansi_color_red;

    printf("\n%s======================================================================%s\n",
           ansi_bold_cyan, ansi_reset);
    printf("    %s%sRECONSTRUCTION ACCURACY & FIDELITY EVALUATION (D vs D_true)%s\n",
           ansi_bold, ansi_bold_cyan, ansi_reset);
    printf("%s======================================================================%s\n",
           ansi_bold_cyan, ansi_reset);
    printf("  Evaluated Samples:            %s%u%s\n", ansi_bold_green, eval_rows, ansi_reset);
    printf("  Coordinate Dimensions:        %s%u%s\n", ansi_bold_green, cols_recon, ansi_reset);
    printf("  Root Mean Squared Err (RMSE): %s%.6f%s\n", ansi_color_yellow, rmse, ansi_reset);
    printf("  Mean Absolute Error (MAE):    %s%.6f%s\n", ansi_color_yellow, mae, ansi_reset);
    printf("  Max Absolute Error (L_inf):   %s%.6f%s\n",
           ansi_color_yellow, max_abs_err, ansi_reset);
    printf("  Normalized RMSE (NRMSE):     %s%.2f %%%s (relative to signal range)\n",
           ansi_color_yellow, nrmse_pct, ansi_reset);
    if (target_noise > 0.0)
    {
        double noise_ratio = rmse / target_noise;
        const char *ratio_str =
            (noise_ratio < 1.0)  ? "DENOISED (sub-noise floor via k-averaging)" :
            (noise_ratio <= 1.5) ? "OPTIMAL (near raw noise floor)" :
            (noise_ratio <= 2.5) ? "GOOD (near noise floor)" :
                                   "ELEVATED";
        const char *ratio_col =
            (noise_ratio < 1.0)  ? ansi_bold_green :
            (noise_ratio <= 1.5) ? ansi_color_green :
            (noise_ratio <= 2.5) ? ansi_color_yellow :
                                   ansi_color_red;
        printf("  Target Noise Floor (sigma_B): %s%.6f%s\n",
               ansi_color_yellow, target_noise, ansi_reset);
        printf("  RMSE / Noise Floor Ratio:     %s%.2fx%s [%s%s%s]\n",
               ratio_col, noise_ratio, ansi_reset, ratio_col, ratio_str, ansi_reset);
    }
    printf("  Coefficient of Det (R²):      %s%.4f%s [%s%s%s] (variance explained)\n",
           rating_col, r2, ansi_reset, rating_col, rating, ansi_reset);
    if (ss_tot > 1e-12 && total_sq_err > 1e-12)
    {
        double snr_db = 10.0 * log10(ss_tot / total_sq_err);
        printf("  Reconstruction SNR:           %s%.2f dB%s\n",
               ansi_bold_green, snr_db, ansi_reset);
    }

    printf("\n  %sPer-Dimension Breakdown:%s\n", ansi_bold_cyan, ansi_reset);
    for (uint32_t c = 0; c < cols_recon; c++)
    {
        double dim_sq_err  = 0.0;
        double dim_abs_err = 0.0;
        double dim_sum_t   = 0.0;
        for (uint32_t r = 0; r < eval_rows; r++)
        {
            size_t idx = (size_t)r * cols_recon + c;
            double yt  = (double)data_true[idx];
            double yr  = (double)data_recon[idx];
            double diff = yr - yt;
            dim_sq_err  += diff * diff;
            dim_abs_err += fabs(diff);
            dim_sum_t   += yt;
        }
        double dim_mean_t = dim_sum_t / (double)eval_rows;
        double dim_ss_tot = 0.0;
        for (uint32_t r = 0; r < eval_rows; r++)
        {
            size_t idx = (size_t)r * cols_recon + c;
            double yt  = (double)data_true[idx];
            double dt  = yt - dim_mean_t;
            dim_ss_tot += dt * dt;
        }
        double dim_rmse = sqrt(dim_sq_err / (double)eval_rows);
        double dim_mae  = dim_abs_err / (double)eval_rows;
        double dim_r2   = (dim_ss_tot > 1e-12) ? (1.0 - (dim_sq_err / dim_ss_tot)) : 1.0;

        if (target_noise > 0.0)
        {
            printf("    Dim %u: RMSE = %.6f (%.2fx noise), MAE = %.6f, R² = %.4f\n",
                   c, dim_rmse, dim_rmse / target_noise, dim_mae, dim_r2);
        }
        else
        {
            printf("    Dim %u: RMSE = %.6f, MAE = %.6f, R² = %.4f\n",
                   c, dim_rmse, dim_mae, dim_r2);
        }
    }
    printf("%s======================================================================%s\n",
           ansi_bold_cyan, ansi_reset);

    free(data_recon);
    free(data_true);
    return (r2 >= 0.70) ? 0 : 1;
}

/**
 * run_parametric_sweep() - Evaluate reconstruction error over a sweep of k_eff.
 * @path_knn:      Path to k-NN ASCII results file.
 * @path_b:        Path to target dataset B file.
 * @path_true:     Path to ground truth dataset D_true file.
 * @target_noise:  Target noise sigma on B (e.g. 0.1).
 *
 * Return: 0 on success, non-zero on error.
 */
static int run_parametric_sweep(
    const char *path_knn,
    const char *path_b,
    const char *path_true,
    double      target_noise)
{
    float   *data_b     = NULL;
    float   *data_true  = NULL;
    uint32_t rows_b     = 0;
    uint32_t cols_b     = 0;
    uint32_t rows_true  = 0;
    uint32_t cols_true  = 0;

    if (load_ascii_dataset(path_b, &data_b, &rows_b, &cols_b) != 0)
    {
        fprintf(stderr, "Error loading target dataset '%s'\n", path_b);
        return 1;
    }
    if (load_ascii_dataset(path_true, &data_true, &rows_true, &cols_true) != 0)
    {
        fprintf(stderr, "Error loading ground truth dataset '%s'\n", path_true);
        free(data_b);
        return 1;
    }
    if (cols_b != cols_true)
    {
        fprintf(stderr, "Error: Dimension mismatch between B (%uD) and True (%uD)\n",
                cols_b, cols_true);
        free(data_b);
        free(data_true);
        return 1;
    }

    FILE *fp_knn = fopen(path_knn, "r");
    if (!fp_knn)
    {
        fprintf(stderr, "Error opening k-NN results file '%s'\n", path_knn);
        free(data_b);
        free(data_true);
        return 1;
    }

    /* First pass: count queries and max K in k-NN results file */
    char    *line = NULL;
    size_t   len = 0;
    ssize_t  nread;
    uint32_t num_queries = 0;
    uint32_t max_k = 0;

    while ((nread = getline(&line, &len, fp_knn)) != -1)
    {
        char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }
        num_queries++;
        /* Count neighbor tokens */
        char *endptr = NULL;
        (void)strtol(p, &endptr, 10); /* skip query_id */
        p = endptr;
        uint32_t k_count = 0;
        while (*p != '\0' && *p != '\n' && *p != '\r')
        {
            (void)strtoul(p, &endptr, 10); /* neighbor idx */
            if (p == endptr)
            {
                break;
            }
            p = endptr;
            (void)strtod(p, &endptr); /* neighbor dist */
            if (p == endptr)
            {
                break;
            }
            p = endptr;
            k_count++;
        }
        if (k_count > max_k)
        {
            max_k = k_count;
        }
    }

    if (num_queries == 0 || max_k == 0)
    {
        fprintf(stderr, "Error: No valid queries parsed from '%s'\n", path_knn);
        free(line);
        fclose(fp_knn);
        free(data_b);
        free(data_true);
        return 1;
    }

    if (num_queries > rows_true)
    {
        num_queries = rows_true;
    }

    /* Allocate k-NN indices and distances cache */
    uint32_t *knn_idx  = (uint32_t *)malloc((size_t)num_queries * max_k * sizeof(uint32_t));
    float    *knn_dist = (float *)malloc((size_t)num_queries * max_k * sizeof(float));
    uint32_t *query_k  = (uint32_t *)calloc(num_queries, sizeof(uint32_t));

    if (!knn_idx || !knn_dist || !query_k)
    {
        fprintf(stderr, "Error: Out of memory allocating k-NN cache buffers\n");
        free(line);
        fclose(fp_knn);
        free(data_b);
        free(data_true);
        free(knn_idx);
        free(knn_dist);
        free(query_k);
        return 1;
    }

    /* Second pass: load neighbor indices and distances */
    rewind(fp_knn);
    uint32_t q = 0;
    while (q < num_queries && (nread = getline(&line, &len, fp_knn)) != -1)
    {
        char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }
        char *endptr = NULL;
        (void)strtol(p, &endptr, 10); /* query_id */
        p = endptr;
        uint32_t k = 0;
        while (*p != '\0' && *p != '\n' && *p != '\r' && k < max_k)
        {
            unsigned long idx = strtoul(p, &endptr, 10);
            if (p == endptr)
            {
                break;
            }
            p = endptr;
            double d = strtod(p, &endptr);
            if (p == endptr)
            {
                break;
            }
            p = endptr;

            knn_idx[(size_t)q * max_k + k]  = (uint32_t)idx;
            knn_dist[(size_t)q * max_k + k] = (float)d;
            k++;
        }
        query_k[q] = k;
        q++;
    }

    free(line);
    fclose(fp_knn);

    /* Compute ss_tot of ground truth */
    uint64_t total_elements = (uint64_t)num_queries * cols_b;

    /* Allocate scratch buffers outside the loop */
    float *scratch_d = (float *)malloc(cols_b * sizeof(float));
    float *weights   = (float *)malloc(max_k * sizeof(float));

    if (!scratch_d || !weights)
    {
        fprintf(stderr, "Error: Out of memory allocating scratch buffers\n");
        free(scratch_d);
        free(weights);
        free(knn_idx);
        free(knn_dist);
        free(query_k);
        free(data_b);
        free(data_true);
        return 1;
    }

    static const char sep_double[] =
        "========================================================================================";
    static const char sep_single[] =
        "----------------------------------------------------------------------------------------";

    /* Print Table Header */
    printf("\n%s%s%s\n", ansi_bold_cyan, sep_double, ansi_reset);
    printf("    %s%sPARAMETRIC SWEEP: EFFECT OF k_eff ON NOISE SUPPRESSION "
           "(target sigma_B = %.4f)%s\n",
           ansi_bold, ansi_bold_cyan, target_noise, ansi_reset);
    printf("%s%s%s\n", ansi_bold_cyan, sep_double, ansi_reset);
    printf("  %s%-3s  %-15s  %-6s  %-15s  %-12s  %-14s  %-16s%s\n",
           ansi_bold, "k", "Weight Mode", "k_eff", "Floor (sigma/sqrt)",
           "Meas RMSE", "RMSE / sigma", "Status", ansi_reset);
    printf("%s\n", sep_single);

    static const uint32_t sweep_k[] = { 1, 2, 3, 5, 8, 10, 15, 20, 30 };
    size_t num_k = sizeof(sweep_k) / sizeof(sweep_k[0]);

    /* Part 1: IDW Weighting */
    for (size_t ki = 0; ki < num_k; ki++)
    {
        uint32_t k = sweep_k[ki];
        if (k > max_k)
        {
            continue;
        }

        double total_sq_err = 0.0;
        double sum_k_eff    = 0.0;

        for (uint32_t i = 0; i < num_queries; i++)
        {
            uint32_t k_used = (k < query_k[i]) ? k : query_k[i];
            if (k_used == 0)
            {
                continue;
            }

            const uint32_t *indices = knn_idx + ((size_t)i * max_k);
            const float    *dists   = knn_dist + ((size_t)i * max_k);

            /* Calculate IDW weights */
            int exact = -1;
            for (uint32_t j = 0; j < k_used; j++)
            {
                if (dists[j] < 1e-9f)
                {
                    exact = (int)j;
                    break;
                }
            }

            if (exact >= 0)
            {
                for (uint32_t j = 0; j < k_used; j++)
                {
                    weights[j] = (j == (uint32_t)exact) ? 1.0f : 0.0f;
                }
            }
            else
            {
                float sum_w = 0.0f;
                for (uint32_t j = 0; j < k_used; j++)
                {
                    float d = fmaxf(dists[j], 1e-7f);
                    weights[j] = 1.0f / d;
                    sum_w += weights[j];
                }
                float inv_s = (sum_w > 0.0f) ? (1.0f / sum_w) : 0.0f;
                for (uint32_t j = 0; j < k_used; j++)
                {
                    weights[j] *= inv_s;
                }
            }

            /* Compute sample k_eff */
            float sum_w_sq = 0.0f;
            for (uint32_t j = 0; j < k_used; j++)
            {
                sum_w_sq += weights[j] * weights[j];
            }
            double k_eff = (sum_w_sq > 1e-12f) ? (1.0 / (double)sum_w_sq) : 1.0;
            sum_k_eff += k_eff;

            /* Accumulate reconstructed frame */
            memset(scratch_d, 0, cols_b * sizeof(float));
            for (uint32_t j = 0; j < k_used; j++)
            {
                uint32_t b_idx = indices[j];
                if (b_idx < rows_b)
                {
                    const float *row_b = data_b + ((size_t)b_idx * cols_b);
                    for (uint32_t c = 0; c < cols_b; c++)
                    {
                        scratch_d[c] += weights[j] * row_b[c];
                    }
                }
            }

            /* Error comparison */
            const float *row_true = data_true + ((size_t)i * cols_b);
            for (uint32_t c = 0; c < cols_b; c++)
            {
                double diff = (double)scratch_d[c] - (double)row_true[c];
                total_sq_err += diff * diff;
            }
        } // for (num_queries)

        double mean_k_eff = sum_k_eff / (double)num_queries;
        double rmse       = sqrt(total_sq_err / (double)total_elements);
        double floor_lim  = (target_noise > 0.0 && mean_k_eff > 0.0)
                                ? (target_noise / sqrt(mean_k_eff))
                                : 0.0;
        double ratio      = (target_noise > 0.0) ? (rmse / target_noise) : 0.0;

        const char *status = (ratio < 1.0)  ? "DENOISED" :
                             (ratio <= 1.5) ? "OPTIMAL" : "ELEVATED";
        const char *stat_col = (ratio < 1.0)  ? ansi_bold_green :
                               (ratio <= 1.5) ? ansi_color_green : ansi_color_yellow;

        printf("  %-3u  %-15s  %6.2f  %15.6f  %12.6f  %s%7.2fx%s         [%s%s%s]\n",
               k, "idw (alpha=1)", mean_k_eff, floor_lim, rmse,
               stat_col, ratio, ansi_reset, stat_col, status, ansi_reset);
    } // for (IDW)

    printf("%s\n", sep_single);

    /* Part 2: Uniform Weighting */
    static const uint32_t uniform_k[] = { 1, 3, 5, 10, 20, 30 };
    size_t num_unif = sizeof(uniform_k) / sizeof(uniform_k[0]);

    for (size_t ki = 0; ki < num_unif; ki++)
    {
        uint32_t k = uniform_k[ki];
        if (k > max_k)
        {
            continue;
        }

        double total_sq_err = 0.0;
        double sum_k_eff    = 0.0;

        for (uint32_t i = 0; i < num_queries; i++)
        {
            uint32_t k_used = (k < query_k[i]) ? k : query_k[i];
            if (k_used == 0)
            {
                continue;
            }

            const uint32_t *indices = knn_idx + ((size_t)i * max_k);
            float uniform_w = 1.0f / (float)k_used;

            sum_k_eff += (double)k_used;

            /* Accumulate reconstructed frame */
            memset(scratch_d, 0, cols_b * sizeof(float));
            for (uint32_t j = 0; j < k_used; j++)
            {
                uint32_t b_idx = indices[j];
                if (b_idx < rows_b)
                {
                    const float *row_b = data_b + ((size_t)b_idx * cols_b);
                    for (uint32_t c = 0; c < cols_b; c++)
                    {
                        scratch_d[c] += uniform_w * row_b[c];
                    }
                }
            }

            /* Error comparison */
            const float *row_true = data_true + ((size_t)i * cols_b);
            for (uint32_t c = 0; c < cols_b; c++)
            {
                double diff = (double)scratch_d[c] - (double)row_true[c];
                total_sq_err += diff * diff;
            }
        } // for (num_queries)

        double mean_k_eff = sum_k_eff / (double)num_queries;
        double rmse       = sqrt(total_sq_err / (double)total_elements);
        double floor_lim  = (target_noise > 0.0 && mean_k_eff > 0.0)
                                ? (target_noise / sqrt(mean_k_eff))
                                : 0.0;
        double ratio      = (target_noise > 0.0) ? (rmse / target_noise) : 0.0;

        const char *status = (ratio < 1.0)  ? "DENOISED" :
                             (ratio <= 1.5) ? "OPTIMAL" : "ELEVATED";
        const char *stat_col = (ratio < 1.0)  ? ansi_bold_green :
                               (ratio <= 1.5) ? ansi_color_green : ansi_color_yellow;

        printf("  %-3u  %-15s  %6.2f  %15.6f  %12.6f  %s%7.2fx%s         [%s%s%s]\n",
               k, "uniform", mean_k_eff, floor_lim, rmse,
               stat_col, ratio, ansi_reset, stat_col, status, ansi_reset);
    } // for (Uniform)

    printf("%s%s%s\n", ansi_bold_cyan, sep_double, ansi_reset);

    free(scratch_d);
    free(weights);
    free(knn_idx);
    free(knn_dist);
    free(query_k);
    free(data_b);
    free(data_true);

    return 0;
}

/**
 * print_usage() - Display command-line usage summary.
 * @progname: Binary name.
 */
static void print_usage(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-mk4dataset%s - Native C 4-dataset coupled benchmark generator\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s [outdir] [-ntrain N] [-nquery M] [-dima D] [-dimb D] "
           "[-noise-a sigma] [-noise-b sigma] [-seed S]\n\n",
           ansi_bold_green, progname, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Generates pre-synchronized datasets A, B, C, D_true directly on filesystem\n"
           "  with zero external dependencies (pure C math library).\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-ntrain%s %s<N>%s        Training sample count (default: 1000)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-nquery%s %s<M>%s        Query sample count (default: 200)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-dima%s %s<D>%s          Dimension of A and C (default: 3)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-dimb%s %s<D>%s          Dimension of B and D (default: 3)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-noise-a%s %s<S>%s       Gaussian noise sigma on A (default: 0.02)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-noise-b%s %s<S>%s       Gaussian noise sigma on B (default: 0.01)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-seed%s %s<S>%s          Random number generator seed (default: 42)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-eval%s %s<rec> <true> [noise]%s Evaluate reconstructed D vs ground truth\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-sweep%s %s<knn> <b> <true> [noise]%s Parametric sweep over k_eff\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-h, --help%s         Display this help message\n\n",
           ansi_color_green, ansi_reset);
}

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    if (argc >= 4 && strcmp(argv[1], "-eval") == 0)
    {
        double target_noise = (argc >= 5) ? atof(argv[4]) : 0.0;
        return evaluate_reconstruction(argv[2], argv[3], target_noise);
    }
    if (argc >= 5 && strcmp(argv[1], "-sweep") == 0)
    {
        double target_noise = (argc >= 6) ? atof(argv[5]) : 0.1;
        return run_parametric_sweep(argv[2], argv[3], argv[4], target_noise);
    }

    const char *outdir   = "data_4dataset";
    uint32_t    n_train  = 1000;
    uint32_t    n_query  = 200;
    uint32_t    dim_a    = 3;
    uint32_t    dim_b    = 3;
    float       noise_a  = 0.02f;
    float       noise_b  = 0.01f;
    unsigned int seed    = 42;

    int positional_idx = 0;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-ntrain") == 0 && i + 1 < argc)
        {
            n_train = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nquery") == 0 && i + 1 < argc)
        {
            n_query = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-dima") == 0 && i + 1 < argc)
        {
            dim_a = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-dimb") == 0 && i + 1 < argc)
        {
            dim_b = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-noise-a") == 0 && i + 1 < argc)
        {
            noise_a = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-noise-b") == 0 && i + 1 < argc)
        {
            noise_b = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
        {
            seed = (unsigned int)atoi(argv[++i]);
        }
        else if (argv[i][0] != '-')
        {
            if (positional_idx == 0)
            {
                outdir = argv[i];
                positional_idx++;
            }
        }
    } // for (argv)

    if (dim_a < 3 || dim_b < 3 || n_train == 0 || n_query == 0)
    {
        fprintf(stderr, "Error: Dimensions must be >= 3 and sample counts > 0.\n");
        return 1;
    }

    srand(seed);

    if (ensure_dir_exists(outdir) != 0)
    {
        fprintf(stderr, "Error: Cannot create output directory '%s'\n", outdir);
        return 1;
    }

    size_t size_a      = (size_t)n_train * dim_a * sizeof(float);
    size_t size_b      = (size_t)n_train * dim_b * sizeof(float);
    size_t size_c      = (size_t)n_query * dim_a * sizeof(float);
    size_t size_d_true = (size_t)n_query * dim_b * sizeof(float);

    float *data_a      = (float *)malloc(size_a);
    float *data_b      = (float *)malloc(size_b);
    float *data_c      = (float *)malloc(size_c);
    float *data_d_true = (float *)malloc(size_d_true);

    if (!data_a || !data_b || !data_c || !data_d_true)
    {
        fprintf(stderr, "Error: Out of memory allocating dataset arrays.\n");
        free(data_a);
        free(data_b);
        free(data_c);
        free(data_d_true);
        return 1;
    }

    /* 1. Generate Training Sets A and B (Strictly synchronized at t_i) */
    float t_step = (n_train > 1) ? (4.0f * (float)M_PI / (float)(n_train - 1)) : 0.0f;
    for (uint32_t i = 0; i < n_train; i++)
    {
        float t = (float)i * t_step;

        /* Dataset A (Observation space) */
        float *row_a = data_a + ((size_t)i * dim_a);
        row_a[0] = cosf(t) + rand_normal(0.0f, noise_a);
        row_a[1] = sinf(t) + rand_normal(0.0f, noise_a);
        row_a[2] = 0.25f * t + rand_normal(0.0f, noise_a);
        for (uint32_t d = 3; d < dim_a; d++)
        {
            row_a[d] = rand_normal(0.0f, 0.05f) + rand_normal(0.0f, noise_a);
        }

        /* Dataset B (Coupled state space, synchronized at exact same t) */
        float *row_b = data_b + ((size_t)i * dim_b);
        row_b[0] = sinf(2.0f * t) + rand_normal(0.0f, noise_b);
        row_b[1] = cosf(2.0f * t) + rand_normal(0.0f, noise_b);
        row_b[2] = sinf(0.5f * t) + rand_normal(0.0f, noise_b);
        for (uint32_t d = 3; d < dim_b; d++)
        {
            row_b[d] = cosf(3.0f * t) + rand_normal(0.0f, noise_b);
        }
    } // for (n_train)

    /* 2. Generate Query C and Ground Truth D_true */
    float t_q_start = 0.1f;
    float t_q_end   = 4.0f * (float)M_PI - 0.1f;
    float t_q_step  = (n_query > 1) ? ((t_q_end - t_q_start) / (float)(n_query - 1)) : 0.0f;

    for (uint32_t j = 0; j < n_query; j++)
    {
        float t_q = t_q_start + (float)j * t_q_step;

        /* Dataset C (Observation space query) */
        float *row_c = data_c + ((size_t)j * dim_a);
        row_c[0] = cosf(t_q) + rand_normal(0.0f, noise_a * 0.5f);
        row_c[1] = sinf(t_q) + rand_normal(0.0f, noise_a * 0.5f);
        row_c[2] = 0.25f * t_q + rand_normal(0.0f, noise_a * 0.5f);
        for (uint32_t d = 3; d < dim_a; d++)
        {
            row_c[d] = rand_normal(0.0f, 0.05f) + rand_normal(0.0f, noise_a * 0.5f);
        }

        /* Ground Truth D_true (True state on manifold B without noise) */
        float *row_d = data_d_true + ((size_t)j * dim_b);
        row_d[0] = sinf(2.0f * t_q);
        row_d[1] = cosf(2.0f * t_q);
        row_d[2] = sinf(0.5f * t_q);
        for (uint32_t d = 3; d < dim_b; d++)
        {
            row_d[d] = cosf(3.0f * t_q);
        }
    } // for (n_query)

    char path_a[1024];
    char path_b[1024];
    char path_c[1024];
    char path_d[1024];

    snprintf(path_a, sizeof(path_a), "%s/dataset_A.txt", outdir);
    snprintf(path_b, sizeof(path_b), "%s/dataset_B.txt", outdir);
    snprintf(path_c, sizeof(path_c), "%s/dataset_C.txt", outdir);
    snprintf(path_d, sizeof(path_d), "%s/dataset_D_true.txt", outdir);

    write_ascii_dataset(path_a, data_a, n_train, dim_a);
    write_ascii_dataset(path_b, data_b, n_train, dim_b);
    write_ascii_dataset(path_c, data_c, n_query, dim_a);
    write_ascii_dataset(path_d, data_d_true, n_query, dim_b);

    printf("%s[gric-mk4dataset]%s Generated synchronized benchmark datasets in '%s':\n",
           ansi_bold_green, ansi_reset, outdir);
    printf("  Dataset A:      %u samples x %uD -> %s\n", n_train, dim_a, path_a);
    printf("  Dataset B:      %u samples x %uD -> %s (pre-synchronized with A)\n",
           n_train, dim_b, path_b);
    printf("  Dataset C:      %u samples x %uD -> %s\n", n_query, dim_a, path_c);
    printf("  Dataset D_true: %u samples x %uD -> %s\n", n_query, dim_b, path_d);

    free(data_a);
    free(data_b);
    free(data_c);
    free(data_d_true);

    return 0;
}

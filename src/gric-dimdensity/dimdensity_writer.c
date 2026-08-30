/**
 * @file dimdensity_writer.c
 * @brief Output serialization and terminal dashboard reporting implementation.
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_writer.h"
#include "shared/cli_colors.h"
#include "shared/gric_bin_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

/**
 * kernel_name() - Get human-readable name for kernel type.
 * @kernel: DimDensityKernel enum.
 *
 * Return: String name of kernel.
 */
static const char *kernel_name(
    DimDensityKernel kernel)
{
    switch (kernel)
    {
        case DIMDENSITY_KERNEL_EPANECHNIKOV:
            return "Epanechnikov";
        case DIMDENSITY_KERNEL_GAUSSIAN:
            return "Gaussian";
        case DIMDENSITY_KERNEL_UNIFORM:
        default:
            return "Uniform (Mack k-NN)";
    }
}

/**
 * print_ascii_bar_chart() - Print formatted terminal histogram bars.
 * @hist:     Array of counts per bin.
 * @bins:     Number of bins.
 * @min_v:    Minimum data bound.
 * @max_v:    Maximum data bound.
 * @total:    Total sample count.
 * @max_cols: Maximum terminal column width for largest bar.
 */
static void print_ascii_bar_chart(
    const long *hist,
    int         bins,
    double      min_v,
    double      max_v,
    uint64_t    total,
    int         max_cols)
{
    if (hist == NULL || bins <= 0 || total == 0)
    {
        return;
    }

    long max_count = 0;
    for (int b = 0; b < bins; b++)
    {
        if (hist[b] > max_count)
        {
            max_count = hist[b];
        }
    }
    if (max_count == 0)
    {
        max_count = 1;
    }

    double range = max_v - min_v;
    double bin_w = (bins > 0) ? (range / (double)bins) : 0.0;

    for (int b = 0; b < bins; b++)
    {
        double b_start = min_v + (double)b * bin_w;
        double b_end = b_start + bin_w;
        int bar_len = (int)((double)hist[b] / (double)max_count * (double)max_cols);
        double pct = 100.0 * (double)hist[b] / (double)total;

        printf("  %s[%6.2f - %6.2f]%s %4ld (%5.1f%%) | %s",
               ansi_color_grey, b_start, b_end, ansi_reset,
               hist[b], pct, ansi_color_cyan);

        for (int c = 0; c < bar_len; c++)
        {
            printf("■");
        }
        printf("%s\n", ansi_reset);
    } // for (int b = 0; ...)
}

/**
 * write_ascii_file() - Write formatted ASCII results table.
 * @path:      Output file path.
 * @config:    Configuration parameters.
 * @results:   Evaluation results.
 * @stats:     Summary statistics.
 *
 * Return: 0 on success, -1 on error.
 */
static int write_ascii_file(
    const char              *path,
    const DimDensityConfig  *config,
    const DimDensityResults *results,
    const DimDensityStats   *stats)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open output file '%s' for writing\n", path);
        return -1;
    }

    fprintf(fp, "# gric-dimdensity results\n");
    fprintf(fp, "# Samples: %lu, Target k: %d, Kernel: %s, Estimator: %s\n",
            results->num_samples, results->k_used, kernel_name(config->kernel_type),
            config->unbiased_mle ? "Unbiased MLE (k-2)" : "Classic MLE (k-1)");
    fprintf(fp, "# Intrinsic Dimension: Mean = %.4f +/- %.4f [Median = %.4f, "
            "Min = %.4f, Max = %.4f]\n",
            stats->dim_mean, stats->dim_std, stats->dim_pct.p50, stats->dim_min, stats->dim_max);
    fprintf(fp, "# Log-Density:         Mean = %.4f +/- %.4f [Median = %.4f, "
            "Min = %.4f, Max = %.4f]\n",
            stats->log_dens_mean, stats->log_dens_std, stats->log_dens_pct.p50,
            stats->log_dens_min, stats->log_dens_max);
    fprintf(fp, "# Columns: sample_id  local_dim  density  log_density  r_k\n");

    uint64_t n = results->num_samples;
    for (uint64_t i = 0; i < n; i++)
    {
        fprintf(fp, "%-8lu  %10.4f  %14.6e  %12.4f  %12.6f\n",
                (unsigned long)i, results->local_dim[i], results->density[i],
                results->log_density[i], results->rk_dist[i]);
    }

    fclose(fp);
    return 0;
}

/**
 * write_bin_file() - Write GRIC binary [N x 4] array.
 * @path:    Output file path.
 * @results: Evaluation results.
 *
 * Return: 0 on success, -1 on error.
 */
static int write_bin_file(
    const char              *path,
    const DimDensityResults *results)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open binary output file '%s'\n", path);
        return -1;
    }

    uint64_t n = results->num_samples;
    uint64_t num_cols = 4;
    uint64_t total_elems = n * num_cols;

    gric_bin_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr.ndim = 2;
    hdr.dims[0] = n;
    hdr.dims[1] = num_cols;
    hdr.num_elements = total_elems;
    hdr.data_bytes = total_elems * sizeof(float);

    const char *comment = "gric-dimdensity output matrix [N x 4]: local_dim, dens, log_dens, rk";
    if (gric_bin_write_header(fp, &hdr, comment) != 0)
    {
        fclose(fp);
        return -1;
    }

    float *buf = (float *)malloc(total_elems * sizeof(float));
    if (buf == NULL)
    {
        fclose(fp);
        return -1;
    }

    for (uint64_t i = 0; i < n; i++)
    {
        buf[i * 4 + 0] = (float)results->local_dim[i];
        buf[i * 4 + 1] = (float)results->density[i];
        buf[i * 4 + 2] = (float)results->log_density[i];
        buf[i * 4 + 3] = (float)results->rk_dist[i];
    }

    fwrite(buf, sizeof(float), total_elems, fp);
    free(buf);
    fclose(fp);
    return 0;
}

#ifdef USE_CFITSIO
/**
 * write_fits_file() - Write results as 2D FITS image HDU [4 x N].
 * @path:    Output FITS file path.
 * @results: Evaluation results.
 *
 * Return: 0 on success, -1 on error.
 */
static int write_fits_file(
    const char              *path,
    const DimDensityResults *results)
{
    int status = 0;
    fitsfile *fptr = NULL;

    char clobber_path[4100];
    snprintf(clobber_path, sizeof(clobber_path), "!%s", path);

    fits_create_file(&fptr, clobber_path, &status);
    if (status != 0 || fptr == NULL)
    {
        fprintf(stderr, "Error: Could not create FITS file '%s' (status %d)\n", path, status);
        return -1;
    }

    long n = (long)results->num_samples;
    long naxes[2] = {4, n};
    fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status);

    double *buf = (double *)malloc((size_t)(n * 4) * sizeof(double));
    if (buf == NULL)
    {
        fits_close_file(fptr, &status);
        return -1;
    }

    for (long i = 0; i < n; i++)
    {
        buf[i * 4 + 0] = results->local_dim[i];
        buf[i * 4 + 1] = results->density[i];
        buf[i * 4 + 2] = results->log_density[i];
        buf[i * 4 + 3] = results->rk_dist[i];
    }

    long fpixel[2] = {1, 1};
    fits_write_pix(fptr, TDOUBLE, fpixel, n * 4, buf, &status);
    free(buf);

    fits_write_key(fptr, TSTRING, "COL1", "local_dim", "Estimated Local Intrinsic Dimension",
                   &status);
    fits_write_key(fptr, TSTRING, "COL2", "density", "Mack-Rosenblatt Local Density", &status);
    fits_write_key(fptr, TSTRING, "COL3", "log_dens", "Log-Density ln(f)", &status);
    fits_write_key(fptr, TSTRING, "COL4", "rk_dist", "Distance to k-th neighbor", &status);

    fits_close_file(fptr, &status);
    return (status == 0) ? 0 : -1;
}
#endif // USE_CFITSIO

/**
 * dimdensity_write_json_report() - Output structured JSON summary.
 * @stream:     Output destination stream.
 * @config:     Configuration parameters.
 * @dist_data:  Input distance data.
 * @results:    Evaluation results.
 * @stats:      Summary statistics.
 * @elapsed_ms: Execution duration in milliseconds.
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_write_json_report(
    FILE                    *stream,
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats,
    double                   elapsed_ms)
{
    if (stream == NULL || config == NULL || results == NULL || stats == NULL)
    {
        return -1;
    }

    fprintf(stream, "{\n");
    fprintf(stream, "  \"tool\": \"gric-dimdensity\",\n");
    fprintf(stream, "  \"input_file\": \"%s\",\n",
            (dist_data && dist_data->resolved_path) ? dist_data->resolved_path : "unknown");
    fprintf(stream, "  \"num_samples\": %lu,\n", results->num_samples);
    fprintf(stream, "  \"target_k\": %d,\n", results->k_used);
    fprintf(stream, "  \"kernel\": \"%s\",\n", kernel_name(config->kernel_type));
    fprintf(stream, "  \"unbiased_mle\": %s,\n", config->unbiased_mle ? "true" : "false");
    fprintf(stream, "  \"execution_time_ms\": %.2f,\n", elapsed_ms);

    // Intrinsic Dimension section
    fprintf(stream, "  \"intrinsic_dimension\": {\n");
    fprintf(stream, "    \"mean\": %.4f,\n", stats->dim_mean);
    fprintf(stream, "    \"std_dev\": %.4f,\n", stats->dim_std);
    fprintf(stream, "    \"min\": %.4f,\n", stats->dim_min);
    fprintf(stream, "    \"max\": %.4f,\n", stats->dim_max);
    fprintf(stream, "    \"percentiles\": {\n");
    fprintf(stream, "      \"p10\": %.4f,\n", stats->dim_pct.p10);
    fprintf(stream, "      \"p25\": %.4f,\n", stats->dim_pct.p25);
    fprintf(stream, "      \"p50\": %.4f,\n", stats->dim_pct.p50);
    fprintf(stream, "      \"p75\": %.4f,\n", stats->dim_pct.p75);
    fprintf(stream, "      \"p90\": %.4f\n", stats->dim_pct.p90);
    fprintf(stream, "    }\n");
    fprintf(stream, "  },\n");

    // Density section
    fprintf(stream, "  \"local_density\": {\n");
    fprintf(stream, "    \"mean\": %.6e,\n", stats->dens_mean);
    fprintf(stream, "    \"std_dev\": %.6e,\n", stats->dens_std);
    fprintf(stream, "    \"min\": %.6e,\n", stats->dens_min);
    fprintf(stream, "    \"max\": %.6e,\n", stats->dens_max);
    fprintf(stream, "    \"percentiles\": {\n");
    fprintf(stream, "      \"p10\": %.6e,\n", stats->dens_pct.p10);
    fprintf(stream, "      \"p25\": %.6e,\n", stats->dens_pct.p25);
    fprintf(stream, "      \"p50\": %.6e,\n", stats->dens_pct.p50);
    fprintf(stream, "      \"p75\": %.6e,\n", stats->dens_pct.p75);
    fprintf(stream, "      \"p90\": %.6e\n", stats->dens_pct.p90);
    fprintf(stream, "    }\n");
    fprintf(stream, "  },\n");

    // Log-Density section
    fprintf(stream, "  \"log_density\": {\n");
    fprintf(stream, "    \"mean\": %.4f,\n", stats->log_dens_mean);
    fprintf(stream, "    \"std_dev\": %.4f,\n", stats->log_dens_std);
    fprintf(stream, "    \"min\": %.4f,\n", stats->log_dens_min);
    fprintf(stream, "    \"max\": %.4f,\n", stats->log_dens_max);
    fprintf(stream, "    \"percentiles\": {\n");
    fprintf(stream, "      \"p10\": %.4f,\n", stats->log_dens_pct.p10);
    fprintf(stream, "      \"p25\": %.4f,\n", stats->log_dens_pct.p25);
    fprintf(stream, "      \"p50\": %.4f,\n", stats->log_dens_pct.p50);
    fprintf(stream, "      \"p75\": %.4f,\n", stats->log_dens_pct.p75);
    fprintf(stream, "      \"p90\": %.4f\n", stats->log_dens_pct.p90);
    fprintf(stream, "    }\n");
    fprintf(stream, "  }\n");
    fprintf(stream, "}\n");

    return 0;
}

/**
 * dimdensity_print_dashboard() - Print colorful terminal dashboard.
 * @config:     Active configuration.
 * @dist_data:  Input distance metadata.
 * @results:    Evaluation results.
 * @stats:      Summary statistics.
 * @elapsed_ms: Execution duration in milliseconds.
 */
void dimdensity_print_dashboard(
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats,
    double                   elapsed_ms)
{
    if (config == NULL || results == NULL || stats == NULL)
    {
        return;
    }

    printf("\n%s╔═══════════════════════════════════════════════════════════════════════╗%s\n",
           ansi_bold_cyan, ansi_reset);
    printf("%s║             gric-dimdensity Analysis Summary Dashboard                ║%s\n",
           ansi_bold_cyan, ansi_reset);
    printf("%s╚═══════════════════════════════════════════════════════════════════════╝%s\n\n",
           ansi_bold_cyan, ansi_reset);

    printf("  %sInput Source:%s        %s\n", ansi_color_grey, ansi_reset,
           dist_data->resolved_path ? dist_data->resolved_path : "in-memory");
    printf("  %sTotal Samples:%s       %lu query points\n", ansi_color_grey, ansi_reset,
           results->num_samples);
    printf("  %sTarget k Neighbors:%s  %d (available: %d)\n", ansi_color_grey, ansi_reset,
           results->k_used, dist_data->k_available);
    printf("  %sLID Estimator:%s       %s%s%s\n", ansi_color_grey, ansi_reset,
           ansi_bold_green, config->unbiased_mle ? "Unbiased MLE (k-2)" : "Classic MLE (k-1)",
           ansi_reset);
    printf("  %sDensity Kernel:%s      %s%s%s\n", ansi_color_grey, ansi_reset,
           ansi_bold_green, kernel_name(config->kernel_type), ansi_reset);
    printf("  %sCompute Time:%s        %.2f ms (%.1f samples/sec)\n\n",
           ansi_color_grey, ansi_reset, elapsed_ms,
           (elapsed_ms > 0.0) ? (double)results->num_samples / (elapsed_ms / 1000.0) : 0.0);

    // Distribution Table
    printf("%s--- Distribution Summary ---%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%-20s %-20s %7s %7s %12s %7s %7s%s\n",
           ansi_color_grey, "Metric", "Mean ± Std Dev", "P10", "P25", "Median (P50)",
           "P75", "P90", ansi_reset);

    printf("  %s%-20s%s  %6.3f ± %-6.3f   %7.3f %7.3f    %s%7.3f%s   %7.3f %7.3f\n",
           ansi_color_magenta, "Intrinsic Dimension", ansi_reset,
           stats->dim_mean, stats->dim_std,
           stats->dim_pct.p10, stats->dim_pct.p25,
           ansi_bold_green, stats->dim_pct.p50, ansi_reset,
           stats->dim_pct.p75, stats->dim_pct.p90);

    printf("  %s%-20s%s  %6.3f ± %-6.3f   %7.3f %7.3f    %s%7.3f%s   %7.3f %7.3f\n",
           ansi_color_magenta, "Log-Density ln(f)", ansi_reset,
           stats->log_dens_mean, stats->log_dens_std,
           stats->log_dens_pct.p10, stats->log_dens_pct.p25,
           ansi_bold_green, stats->log_dens_pct.p50, ansi_reset,
           stats->log_dens_pct.p75, stats->log_dens_pct.p90);

    printf("  %s%-20s%s %9.2e ± %-9.2e %9.2e %9.2e  %s%9.2e%s %9.2e %9.2e\n\n",
           ansi_color_magenta, "Density f(x)", ansi_reset,
           stats->dens_mean, stats->dens_std,
           stats->dens_pct.p10, stats->dens_pct.p25,
           ansi_bold_green, stats->dens_pct.p50, ansi_reset,
           stats->dens_pct.p75, stats->dens_pct.p90);

    // Histograms
    printf("%s--- Local Intrinsic Dimension Distribution ---%s\n", ansi_bold_cyan, ansi_reset);
    print_ascii_bar_chart(stats->dim_hist, stats->hist_bins, stats->dim_min, stats->dim_max,
                          results->num_samples, 35);
    printf("\n");

    printf("%s--- Log-Density ln(f(x)) Distribution ---%s\n", ansi_bold_cyan, ansi_reset);
    print_ascii_bar_chart(stats->log_dens_hist, stats->hist_bins, stats->log_dens_min,
                          stats->log_dens_max, results->num_samples, 35);
    printf("\n");
}

/**
 * dimdensity_write_results() - Save evaluation results to file.
 * @config:    Active configuration.
 * @dist_data: Input distance data.
 * @results:   Evaluation results.
 * @stats:     Summary statistics.
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_write_results(
    const DimDensityConfig  *config,
    const KnnDistanceData   *dist_data,
    const DimDensityResults *results,
    const DimDensityStats   *stats)
{
    (void)dist_data;
    if (config == NULL || results == NULL || stats == NULL)
    {
        return -1;
    }

    char base_path[4096];
    if (config->output_path != NULL)
    {
        snprintf(base_path, sizeof(base_path), "%s", config->output_path);
    }
    else
    {
        snprintf(base_path, sizeof(base_path), "dimdensity.txt");
    }

    size_t len = strlen(base_path);

    // If output path is explicitly .bin
    if (len >= 4 && strcasecmp(base_path + len - 4, ".bin") == 0)
    {
        if (!config->json_mode)
        {
            printf("Writing binary matrix: %s\n", base_path);
        }
        return write_bin_file(base_path, results);
    }

#ifdef USE_CFITSIO
    // If output path is explicitly .fits
    if ((len >= 5 && strcasecmp(base_path + len - 5, ".fits") == 0) ||
        (len >= 8 && strcasecmp(base_path + len - 8, ".fits.gz") == 0))
    {
        if (!config->json_mode)
        {
            printf("Writing FITS image: %s\n", base_path);
        }
        return write_fits_file(base_path, results);
    }
#endif

    // Default: write ASCII table and binary array alongside
    char bin_path[4096];
    if (len >= 4 && strcasecmp(base_path + len - 4, ".txt") == 0)
    {
        snprintf(bin_path, sizeof(bin_path), "%.*s.bin", (int)(len - 4), base_path);
    }
    else
    {
        snprintf(bin_path, sizeof(bin_path), "%.4090s.bin", base_path);
    }

    if (!config->json_mode)
    {
        printf("Writing ASCII table:   %s\n", base_path);
    }
    int ret_txt = write_ascii_file(base_path, config, results, stats);

    if (!config->json_mode)
    {
        printf("Writing binary array:  %s\n", bin_path);
    }
    int ret_bin = write_bin_file(bin_path, results);

    return (ret_txt == 0 && ret_bin == 0) ? 0 : -1;
}

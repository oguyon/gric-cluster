/**
 * @file dimdensity_stats.c
 * @brief Distribution metrics, percentiles, and histogram calculations.
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_stats.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * compare_doubles() - Qsort comparator for sorting doubles.
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
 * compute_vector_percentiles() - Compute key percentiles from sorted array.
 * @sorted: Pre-sorted array of double values.
 * @n:      Number of elements.
 * @pct:    Output percentiles structure.
 */
static void compute_vector_percentiles(
    const double          *sorted,
    uint64_t               n,
    DimDensityPercentiles *pct)
{
    if (sorted == NULL || n == 0 || pct == NULL)
    {
        return;
    }

    uint64_t i10 = (uint64_t)(0.10 * (double)(n - 1));
    uint64_t i25 = (uint64_t)(0.25 * (double)(n - 1));
    uint64_t i50 = (uint64_t)(0.50 * (double)(n - 1));
    uint64_t i75 = (uint64_t)(0.75 * (double)(n - 1));
    uint64_t i90 = (uint64_t)(0.90 * (double)(n - 1));

    pct->p10 = sorted[i10];
    pct->p25 = sorted[i25];
    pct->p50 = sorted[i50];
    pct->p75 = sorted[i75];
    pct->p90 = sorted[i90];
}

/**
 * compute_histogram() - Populate histogram bins for a given dataset.
 * @values: Array of values.
 * @n:      Number of values.
 * @min_v:  Minimum value bound.
 * @max_v:  Maximum value bound.
 * @bins:   Number of bins.
 * @hist:   Output histogram count array (size bins).
 */
static void compute_histogram(
    const double *values,
    uint64_t      n,
    double        min_v,
    double        max_v,
    int           bins,
    long         *hist)
{
    if (values == NULL || n == 0 || bins <= 0 || hist == NULL)
    {
        return;
    }

    memset(hist, 0, (size_t)bins * sizeof(long));
    double range = max_v - min_v;
    if (range <= 0.0)
    {
        hist[0] = (long)n;
        return;
    }

    double bin_width = range / (double)bins;

    for (uint64_t i = 0; i < n; i++)
    {
        double v = values[i];
        if (isnan(v))
        {
            continue;
        }

        int b = (int)((v - min_v) / bin_width);
        if (b < 0)
        {
            b = 0;
        }
        else if (b >= bins)
        {
            b = bins - 1;
        }
        hist[b]++;
    }
}

/**
 * dimdensity_compute_stats() - Compute distribution statistics and histograms.
 * @results:       Input evaluation results.
 * @stats:         Output statistics structure.
 * @num_hist_bins: Number of histogram bins to allocate (default 15).
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_compute_stats(
    const DimDensityResults *results,
    DimDensityStats         *stats,
    int                      num_hist_bins)
{
    if (results == NULL || stats == NULL || results->num_samples == 0)
    {
        return -1;
    }

    uint64_t n = results->num_samples;
    int bins = (num_hist_bins > 0) ? num_hist_bins : 15;

    memset(stats, 0, sizeof(DimDensityStats));
    stats->hist_bins = bins;
    stats->dim_hist = (long *)calloc((size_t)bins, sizeof(long));
    stats->log_dens_hist = (long *)calloc((size_t)bins, sizeof(long));

    if (stats->dim_hist == NULL || stats->log_dens_hist == NULL)
    {
        dimdensity_free_stats(stats);
        return -1;
    }

    // 1. Compute summary stats for local dimension
    double dim_sum = 0.0;
    double dim_min = DBL_MAX;
    double dim_max = -DBL_MAX;

    for (uint64_t i = 0; i < n; i++)
    {
        double d = results->local_dim[i];
        dim_sum += d;
        if (d < dim_min)
        {
            dim_min = d;
        }
        if (d > dim_max)
        {
            dim_max = d;
        }
    }

    stats->dim_mean = dim_sum / (double)n;
    stats->dim_min = dim_min;
    stats->dim_max = dim_max;

    double dim_sq_sum = 0.0;
    for (uint64_t i = 0; i < n; i++)
    {
        double diff = results->local_dim[i] - stats->dim_mean;
        dim_sq_sum += (diff * diff);
    }
    stats->dim_std = (n > 1) ? sqrt(dim_sq_sum / (double)(n - 1)) : 0.0;

    // 2. Compute summary stats for density and log-density
    double dens_sum = 0.0;
    double dens_min = DBL_MAX;
    double dens_max = -DBL_MAX;

    double log_dens_sum = 0.0;
    double log_dens_min = DBL_MAX;
    double log_dens_max = -DBL_MAX;

    for (uint64_t i = 0; i < n; i++)
    {
        double f = results->density[i];
        double lf = results->log_density[i];

        dens_sum += f;
        if (f < dens_min)
        {
            dens_min = f;
        }
        if (f > dens_max)
        {
            dens_max = f;
        }

        log_dens_sum += lf;
        if (lf < log_dens_min)
        {
            log_dens_min = lf;
        }
        if (lf > log_dens_max)
        {
            log_dens_max = lf;
        }
    }

    stats->dens_mean = dens_sum / (double)n;
    stats->dens_min = dens_min;
    stats->dens_max = dens_max;

    stats->log_dens_mean = log_dens_sum / (double)n;
    stats->log_dens_min = log_dens_min;
    stats->log_dens_max = log_dens_max;

    double dens_sq_sum = 0.0;
    double log_dens_sq_sum = 0.0;
    for (uint64_t i = 0; i < n; i++)
    {
        double diff_d = results->density[i] - stats->dens_mean;
        dens_sq_sum += (diff_d * diff_d);

        double diff_ld = results->log_density[i] - stats->log_dens_mean;
        log_dens_sq_sum += (diff_ld * diff_ld);
    }
    stats->dens_std = (n > 1) ? sqrt(dens_sq_sum / (double)(n - 1)) : 0.0;
    stats->log_dens_std = (n > 1) ? sqrt(log_dens_sq_sum / (double)(n - 1)) : 0.0;

    // 3. Compute Percentiles via temporary sorted arrays
    double *tmp_sort = (double *)malloc(n * sizeof(double));
    if (tmp_sort != NULL)
    {
        memcpy(tmp_sort, results->local_dim, n * sizeof(double));
        qsort(tmp_sort, (size_t)n, sizeof(double), compare_doubles);
        compute_vector_percentiles(tmp_sort, n, &stats->dim_pct);

        memcpy(tmp_sort, results->density, n * sizeof(double));
        qsort(tmp_sort, (size_t)n, sizeof(double), compare_doubles);
        compute_vector_percentiles(tmp_sort, n, &stats->dens_pct);

        memcpy(tmp_sort, results->log_density, n * sizeof(double));
        qsort(tmp_sort, (size_t)n, sizeof(double), compare_doubles);
        compute_vector_percentiles(tmp_sort, n, &stats->log_dens_pct);

        free(tmp_sort);
    }

    // 4. Compute Histograms
    compute_histogram(results->local_dim, n, stats->dim_min, stats->dim_max,
                      bins, stats->dim_hist);
    compute_histogram(results->log_density, n, stats->log_dens_min, stats->log_dens_max,
                      bins, stats->log_dens_hist);

    return 0;
}

/**
 * dimdensity_free_stats() - Free memory allocations in DimDensityStats.
 * @stats: Pointer to DimDensityStats.
 */
void dimdensity_free_stats(
    DimDensityStats *stats)
{
    if (stats == NULL)
    {
        return;
    }

    if (stats->dim_hist != NULL)
    {
        free(stats->dim_hist);
        stats->dim_hist = NULL;
    }
    if (stats->log_dens_hist != NULL)
    {
        free(stats->log_dens_hist);
        stats->log_dens_hist = NULL;
    }
}

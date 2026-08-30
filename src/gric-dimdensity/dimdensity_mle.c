/**
 * @file dimdensity_mle.c
 * @brief Maximum Likelihood Estimation of Local Intrinsic Dimension (Levina & Bickel).
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_mle.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPS_REG 1e-12
#define MIN_DIM_VAL 0.01
#define MAX_DIM_VAL 1000.0

/**
 * dimdensity_unit_ball_log_volume() - Log volume of d-dimensional unit ball.
 * @d: Effective intrinsic dimension.
 *
 * Return: ln(V_d) = (d/2) * ln(pi) - ln(Gamma(d/2 + 1)).
 */
double dimdensity_unit_ball_log_volume(
    double d)
{
    if (d <= 0.0)
    {
        return 0.0;
    }
    double half_d = 0.5 * d;
    return half_d * log(M_PI) - lgamma(half_d + 1.0);
}

/**
 * dimdensity_estimate_single_mle() - Local dimension for a single sample.
 * @distances: Array of sorted nearest neighbor distances [r_1, r_2, ..., r_k].
 * @k:         Target neighbor count (must be >= 3).
 * @mode:      0 for (k-1) classic, 1 for (k-2) mean-unbiased, 2 for (k-4/3) median-unbiased.
 *
 * Return: Estimated local intrinsic dimension.
 */
double dimdensity_estimate_single_mle(
    const double *restrict distances,
    int                    k,
    int                    mode)
{
    if (distances == NULL || k < 3)
    {
        return 1.0;
    }

    double r_k = distances[k - 1];
    if (r_k < EPS_REG)
    {
        r_k = EPS_REG;
    }
    double log_rk = log(r_k);

    double sum_log_diff = 0.0;
    int valid_neighbors = 0;

    for (int j = 0; j < k - 1; j++)
    {
        double r_j = distances[j];
        if (r_j < EPS_REG)
        {
            r_j = EPS_REG;
        }

        if (r_j < r_k)
        {
            sum_log_diff += (log_rk - log(r_j));
            valid_neighbors++;
        }
    } // for (int j = 0; ...)

    if (valid_neighbors == 0 || sum_log_diff < EPS_REG)
    {
        return 1.0;
    }

    double num;
    if (mode == 2)
    {
        num = (double)k - 4.0 / 3.0; // Median-unbiased: k - 4/3
    }
    else if (mode == 1)
    {
        num = (double)(k - 2);       // Mean-unbiased: k - 2
    }
    else
    {
        num = (double)(k - 1);       // Classic MLE: k - 1
    }

    double d = num / sum_log_diff;

    if (isnan(d) || isinf(d))
    {
        return 1.0;
    }

    if (d < MIN_DIM_VAL)
    {
        d = MIN_DIM_VAL;
    }
    else if (d > MAX_DIM_VAL)
    {
        d = MAX_DIM_VAL;
    }

    return d;
}

/**
 * dimdensity_compute_mle_dimensions() - Estimate dimension for all samples.
 * @dist_data: Input k-NN distance data structure.
 * @config:    Active configuration options.
 * @results:   Results structure to populate.
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_compute_mle_dimensions(
    const KnnDistanceData  *dist_data,
    const DimDensityConfig *config,
    DimDensityResults      *results)
{
    if (dist_data == NULL || config == NULL || results == NULL)
    {
        return -1;
    }

    uint64_t n = dist_data->num_samples;
    int k_avail = dist_data->k_available;

    int target_k = config->k;
    if (target_k <= 0 || target_k > k_avail)
    {
        target_k = k_avail;
    }
    if (target_k < 3)
    {
        fprintf(stderr, "Error: Target k must be >= 3 for MLE (k = %d)\n", target_k);
        return -1;
    }

    results->num_samples = n;
    results->k_used = target_k;

    if (results->local_dim == NULL)
    {
        results->local_dim = (double *)malloc(n * sizeof(double));
    }
    if (results->local_dim_med == NULL)
    {
        results->local_dim_med = (double *)malloc(n * sizeof(double));
    }
    if (results->local_dim_mean == NULL)
    {
        results->local_dim_mean = (double *)malloc(n * sizeof(double));
    }
    if (results->rk_dist == NULL)
    {
        results->rk_dist = (double *)malloc(n * sizeof(double));
    }

    if (results->local_dim == NULL || results->local_dim_med == NULL ||
        results->local_dim_mean == NULL || results->rk_dist == NULL)
    {
        fprintf(stderr, "Error: Failed to allocate memory for results\n");
        return -1;
    }

    int k_min = config->k_min;
    int k_max = config->k_max;
    int use_range = config->use_range_avg;

    if (use_range)
    {
        if (k_min < 3)
        {
            k_min = 3;
        }
        if (k_max > target_k)
        {
            k_max = target_k;
        }
        if (k_min > k_max)
        {
            k_min = k_max;
        }
    }

#ifdef _OPENMP
    if (config->nthreads > 0)
    {
        omp_set_num_threads(config->nthreads);
    }
#endif

#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n; i++)
    {
        const double *row = &dist_data->distances[i * (uint64_t)k_avail];
        results->rk_dist[i] = row[target_k - 1];

        if (use_range && k_max > k_min)
        {
            double sum_med = 0.0, sum_mean = 0.0;
            int count_k = 0;
            for (int k_curr = k_min; k_curr <= k_max; k_curr++)
            {
                sum_med += dimdensity_estimate_single_mle(row, k_curr, 2);
                sum_mean += dimdensity_estimate_single_mle(row, k_curr, 1);
                count_k++;
            }
            results->local_dim_med[i] = (count_k > 0) ? (sum_med / (double)count_k) : 1.0;
            results->local_dim_mean[i] = (count_k > 0) ? (sum_mean / (double)count_k) : 1.0;
        }
        else
        {
            results->local_dim_med[i] = dimdensity_estimate_single_mle(row, target_k, 2);
            results->local_dim_mean[i] = dimdensity_estimate_single_mle(row, target_k, 1);
        }

        // Primary alias matches median-unbiased default
        results->local_dim[i] = results->local_dim_med[i];
    } // for (uint64_t i = 0; ...)

    return 0;
}

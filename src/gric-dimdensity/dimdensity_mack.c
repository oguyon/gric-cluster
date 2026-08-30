/**
 * @file dimdensity_mack.c
 * @brief Mack-Rosenblatt k-NN density estimation with manifold dimension correction.
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_mack.h"
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

/**
 * dimdensity_compute_mack_density() - Compute density for all samples.
 * @dist_data: Input k-NN distance data structure.
 * @config:    Active configuration options.
 * @results:   Results structure containing local_dim and output arrays.
 *
 * Return: 0 on success, -1 on error.
 */
int dimdensity_compute_mack_density(
    const KnnDistanceData  *dist_data,
    const DimDensityConfig *config,
    DimDensityResults      *results)
{
    if (dist_data == NULL || config == NULL || results == NULL)
    {
        return -1;
    }

    uint64_t n = results->num_samples;
    int target_k = results->k_used;
    int k_avail = dist_data->k_available;

    if (results->density == NULL)
    {
        results->density = (double *)malloc(n * sizeof(double));
    }
    if (results->log_density == NULL)
    {
        results->log_density = (double *)malloc(n * sizeof(double));
    }

    if (results->density == NULL || results->log_density == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failure for density arrays\n");
        return -1;
    }

    double log_n = log((double)n);
    double log_k_minus_1 = log((target_k > 1) ? (double)(target_k - 1) : 1.0);
    DimDensityKernel kernel = config->kernel_type;

#ifdef _OPENMP
    if (config->nthreads > 0)
    {
        omp_set_num_threads(config->nthreads);
    }
#endif

#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n; i++)
    {
        double d_i = results->local_dim[i];
        if (d_i < 0.01)
        {
            d_i = 0.01;
        }

        double r_k = results->rk_dist[i];
        if (r_k < EPS_REG)
        {
            r_k = EPS_REG;
        }

        double log_rk = log(r_k);
        double log_vd = dimdensity_unit_ball_log_volume(d_i);

        const double *row = &dist_data->distances[i * (uint64_t)k_avail];

        double log_dens = 0.0;

        if (kernel == DIMDENSITY_KERNEL_UNIFORM)
        {
            // Standard uniform volume-based Mack k-NN density
            log_dens = log_k_minus_1 - log_n - log_vd - d_i * log_rk;
        }
        else if (kernel == DIMDENSITY_KERNEL_EPANECHNIKOV)
        {
            // Epanechnikov kernel weighting
            double sum_k = 0.0;
            double inv_rk = 1.0 / r_k;

            for (int j = 0; j < target_k - 1; j++)
            {
                double u = row[j] * inv_rk;
                if (u < 1.0)
                {
                    sum_k += (1.0 - u * u);
                }
            }

            if (sum_k < EPS_REG)
            {
                sum_k = EPS_REG;
            }

            // Normalization: (d_i + 2) / (2 * N * V_d * r_k^d_i) * sum_k
            log_dens = log(d_i + 2.0) - log(2.0) - log_n - log_vd - d_i * log_rk + log(sum_k);
        }
        else if (kernel == DIMDENSITY_KERNEL_GAUSSIAN)
        {
            // Gaussian kernel weighting
            double sum_g = 0.0;
            double inv_rk = 1.0 / r_k;

            for (int j = 0; j < target_k; j++)
            {
                double u = row[j] * inv_rk;
                sum_g += exp(-0.5 * u * u);
            }

            if (sum_g < EPS_REG)
            {
                sum_g = EPS_REG;
            }

            // Normalization: (2pi)^(-d/2) / (N * r_k^d) * sum_g
            double log_norm = -0.5 * d_i * log(2.0 * M_PI);
            log_dens = log_norm - log_n - d_i * log_rk + log(sum_g);
        }

        results->log_density[i] = log_dens;

        // Linear density with safe exponential bounds
        if (log_dens > 700.0)
        {
            results->density[i] = DBL_MAX;
        }
        else if (log_dens < -700.0)
        {
            results->density[i] = 0.0;
        }
        else
        {
            results->density[i] = exp(log_dens);
        }
    } // for (uint64_t i = 0; ...)

    return 0;
}

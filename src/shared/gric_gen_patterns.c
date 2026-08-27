/**
 * @file gric_gen_patterns.c
 * @brief Implementation of reusable synthetic coordinate and geometric pattern generators.
 */

#include "shared/gric_gen_patterns.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

double rand_double(
    void)
{
    return (double)rand() / (double)RAND_MAX;
} // rand_double

void gen_random_point(
    double *out,
    int     dim)
{
    if (dim == 3)
    {
        double r = cbrt(rand_double());
        double costheta = 1.0 - 2.0 * rand_double();
        double phi = 2.0 * M_PI * rand_double();
        double sintheta = sqrt(1.0 - costheta * costheta);
        out[0] = r * sintheta * cos(phi);
        out[1] = r * sintheta * sin(phi);
        out[2] = r * costheta;
    }
    else if (dim == 2)
    {
        double r = sqrt(rand_double());
        double theta = 2.0 * M_PI * rand_double();
        out[0] = r * cos(theta);
        out[1] = r * sin(theta);
    }
    else
    {
        for (int d = 0; d < dim; d++)
        {
            out[d] = 2.0 * rand_double() - 1.0;
        }
    }
} // gen_random_point

void gen_randexp_point(
    double *out,
    int     dim)
{
    if (dim == 3)
    {
        double sigma = 1.0 / sqrt(-2.0 * log(0.01));
        double r;
        do
        {
            double u1;
            double u2;
            double u3;
            double u4;
            do
            {
                u1 = rand_double();
            } while (u1 <= 1e-15);
            u2 = rand_double();
            do
            {
                u3 = rand_double();
            } while (u3 <= 1e-15);
            u4 = rand_double();

            double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
            double z2 = sqrt(-2.0 * log(u3)) * cos(2.0 * M_PI * u4);

            out[0] = z0 * sigma;
            out[1] = z1 * sigma;
            out[2] = z2 * sigma;

            r = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        } while (r > 1.0);
    }
    else
    {
        gen_random_point(out, dim);
    }
} // gen_randexp_point

void gen_sphere_point(
    double *out,
    int     dim)
{
    if (dim == 3)
    {
        double costheta = 1.0 - 2.0 * rand_double();
        double phi = 2.0 * M_PI * rand_double();
        double sintheta = sqrt(1.0 - costheta * costheta);
        out[0] = sintheta * cos(phi);
        out[1] = sintheta * sin(phi);
        out[2] = costheta;
    }
    else if (dim == 2)
    {
        double theta = 2.0 * M_PI * rand_double();
        out[0] = cos(theta);
        out[1] = sin(theta);
    }
    else
    {
        double sum_sq = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double u1 = rand_double();
            double u2 = rand_double();
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            out[d] = z;
            sum_sq += z * z;
        }
        double norm = sqrt(sum_sq);
        for (int d = 0; d < dim; d++)
        {
            out[d] /= norm;
        }
    }
} // gen_sphere_point

void gen_circle_point(
    double *out,
    long    index,
    double  period,
    int     dim)
{
    if (period <= 0.0)
    {
        period = 1.0;
    }
    double theta = 2.0 * M_PI * (double)index / period;
    out[0] = cos(theta);
    out[1] = sin(theta);
    for (int d = 2; d < dim; d++)
    {
        out[d] = 0.0;
    }
} // gen_circle_point

void gen_spiral_point(
    double *out,
    long    index,
    long    total_points,
    double  loops,
    int     dim)
{
    double t = (double)index / (double)total_points;
    if (dim == 3)
    {
        double raw_x = 0.15 * t * cos(2.0 * M_PI * loops * t);
        double raw_y = 0.15 * t * sin(2.0 * M_PI * loops * t);
        double raw_z = 2.0 * t - 1.0;

        double cos60 = 0.5000000000000000;
        double sin60 = 0.8660254037844386;
        double cos30 = 0.8660254037844386;
        double sin30 = 0.5000000000000000;

        double x1 = raw_x * cos60 + raw_z * sin60;
        double z1 = -raw_x * sin60 + raw_z * cos60;
        double y1 = raw_y;

        out[0] = x1;
        out[1] = y1 * cos30 - z1 * sin30;
        out[2] = y1 * sin30 + z1 * cos30;

        for (int d = 3; d < dim; d++)
        {
            out[d] = 0.0;
        }
    }
    else
    {
        double r = t;
        double theta = 2.0 * M_PI * loops * t;
        out[0] = r * cos(theta);
        out[1] = r * sin(theta);
        for (int d = 2; d < dim; d++)
        {
            out[d] = 0.0;
        }
    }
} // gen_spiral_point

void gen_star_point(
    double *out,
    long    index,
    long    total_points,
    double  spokes,
    int     dim)
{
    (void)total_points;
    int num_spokes = (spokes > 0.0) ? (int)spokes : 20;
    int spoke_idx = (int)(index % num_spokes);

    if (spoke_idx == 0)
    {
        for (int d = 0; d < dim; d++)
        {
            out[d] = 0.0;
        }
        return;
    }

    double phi = acos(1.0 - 2.0 * (double)spoke_idx / (double)num_spokes);
    double theta = M_PI * (1.0 + sqrt(5.0)) * (double)spoke_idx;

    double ux = sin(phi) * cos(theta);
    double uy = sin(phi) * sin(theta);
    double uz = cos(phi);

    double R = 1.0 + 0.5 * (double)spoke_idx;

    out[0] = R * ux;
    out[1] = R * uy;
    if (dim >= 3)
    {
        out[2] = R * uz;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
} // gen_star_point

void gen_concentric_point(
    double *out,
    long    index,
    long    total_points,
    double  shells,
    int     dim)
{
    (void)total_points;
    int num_shells = (shells > 0.0) ? (int)shells : 5;
    int clusters_per_shell = 10;
    int total_clusters = num_shells * clusters_per_shell + 1;

    int cluster_idx = (int)(index % total_clusters);

    if (cluster_idx == 0)
    {
        for (int d = 0; d < dim; d++)
        {
            out[d] = 0.0;
        }
        return;
    }

    int shell_idx = (cluster_idx - 1) / clusters_per_shell;
    int in_shell_idx = (cluster_idx - 1) % clusters_per_shell;

    double R = 1.5 + 1.0 * (double)shell_idx;
    double theta = 2.0 * M_PI * (double)in_shell_idx / (double)clusters_per_shell;

    out[0] = R * cos(theta);
    out[1] = R * sin(theta);
    if (dim >= 3)
    {
        out[2] = 0.0;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
} // gen_concentric_point

void gen_tree_point(
    double *out,
    long    index,
    long    total_points,
    double  unused_param,
    int     dim)
{
    (void)total_points;
    (void)unused_param;
    if (index % 100 == 0)
    {
        int parent_idx = (int)((index / 100) % 31);
        int L = 0;
        int level_start = 0;
        while (L < 4 && parent_idx >= level_start + (1 << L))
        {
            level_start += (1 << L);
            L++;
        }
        int path_val = parent_idx - level_start;

        for (int d = 0; d < 5; d++)
        {
            if (d < L)
            {
                int bit = (path_val >> (L - 1 - d)) & 1;
                out[d] = (bit == 0 ? -2.0 : 2.0);
            }
            else if (d == L)
            {
                out[d] = -1.0;
            }
            else
            {
                out[d] = 0.0;
            }
        }
    }
    else
    {
        int leaf_idx = (int)(index % 32);
        for (int d = 0; d < 5; d++)
        {
            int bit = (leaf_idx >> (4 - d)) & 1;
            out[d] = (bit == 0 ? -2.0 : 2.0);
        }
    }

    for (int d = 5; d < dim; d++)
    {
        out[d] = 0.0;
    }
} // gen_tree_point

void gen_concentric_dense_point(
    double *out,
    long    index,
    long    total_points,
    double  shells,
    int     dim)
{
    (void)total_points;
    int num_shells = (shells > 0.0) ? (int)shells : 10;
    int clusters_per_shell = 30;
    int total_clusters = num_shells * clusters_per_shell + 1;

    int cluster_idx = (int)(index % total_clusters);

    if (cluster_idx == 0)
    {
        for (int d = 0; d < dim; d++)
        {
            out[d] = 0.0;
        }
        return;
    }

    int shell_idx = (cluster_idx - 1) / clusters_per_shell;
    int in_shell_idx = (cluster_idx - 1) % clusters_per_shell;

    double R = 1.5 + 1.0 * (double)shell_idx;
    double theta = 2.0 * M_PI * (double)in_shell_idx / (double)clusters_per_shell;

    out[0] = R * cos(theta);
    out[1] = R * sin(theta);
    if (dim >= 3)
    {
        out[2] = 0.0;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
} // gen_concentric_dense_point

void gen_walk_point(
    double *current,
    double  step_size,
    int     dim)
{
    double *next = (double *)malloc((size_t)dim * sizeof(double));
    if (next == NULL)
    {
        return;
    }
    int attempts = 0;

    while (1)
    {
        if (dim == 3)
        {
            double costheta = 1.0 - 2.0 * rand_double();
            double phi = 2.0 * M_PI * rand_double();
            double sintheta = sqrt(1.0 - costheta * costheta);
            double dx = step_size * sintheta * cos(phi);
            double dy = step_size * sintheta * sin(phi);
            double dz = step_size * costheta;
            next[0] = current[0] + dx;
            next[1] = current[1] + dy;
            next[2] = current[2] + dz;
        }
        else if (dim == 2)
        {
            double angle = 2.0 * M_PI * rand_double();
            next[0] = current[0] + step_size * cos(angle);
            next[1] = current[1] + step_size * sin(angle);
        }
        else
        {
            double sum_sq = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double u1 = rand_double();
                double u2 = rand_double();
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                next[d] = z;
                sum_sq += z * z;
            }
            double norm = sqrt(sum_sq);
            for (int d = 0; d < dim; d++)
            {
                next[d] = current[d] + (next[d] / norm) * step_size;
            }
        }

        double r2 = 0.0;
        for (int d = 0; d < dim; d++)
        {
            r2 += next[d] * next[d];
        }

        if (r2 <= 1.0)
        {
            break;
        }

        attempts++;
        if (attempts > 100)
        {
            memcpy(next, current, (size_t)dim * sizeof(double));
            break;
        }
    }

    memcpy(current, next, (size_t)dim * sizeof(double));
    free(next);
} // gen_walk_point

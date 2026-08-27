/**
 * @file analysis_stats.c
 * @brief Statistical metrics, entropy, lifetimes, and spatial spread computations.
 */

#include "analysis_state.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void compute_derived_stats(
    AnalysisState *state)
{
    if (state->num_clusters <= 0)
    {
        return;
    }

    int n = state->num_clusters;
    state->cluster_sizes = calloc((size_t)n, sizeof(long));
    state->birth_frames = malloc((size_t)n * sizeof(long));
    state->death_frames = malloc((size_t)n * sizeof(long));
    state->transition_matrix = calloc((size_t)(n * n), sizeof(long));

    for (int i = 0; i < n; i++)
    {
        state->birth_frames[i] = -1;
        state->death_frames[i] = -1;
    }

    for (long t = 0; t < state->assignments_count; t++)
    {
        int c = state->assignments[t];
        if (c >= 0 && c < n)
        {
            state->cluster_sizes[c]++;
            if (state->birth_frames[c] == -1)
            {
                state->birth_frames[c] = t;
            }
            state->death_frames[c] = t;

            if (t > 0)
            {
                int prev_c = state->assignments[t - 1];
                if (prev_c >= 0 && prev_c < n)
                {
                    state->transition_matrix[prev_c * n + c]++;
                }
            }
        }
    }

    double entropy = 0.0;
    double log_2 = log(2.0);
    for (int i = 0; i < n; i++)
    {
        if (state->cluster_sizes[i] > 0 && state->assignments_count > 0)
        {
            double p = (double)state->cluster_sizes[i] / (double)state->assignments_count;
            entropy -= p * (log(p) / log_2);
        }
    }

    state->shannon_entropy = entropy;
    if (n > 1)
    {
        state->normalized_entropy = entropy / (log((double)n) / log_2);
    }
    else
    {
        state->normalized_entropy = 0.0;
    }
} // compute_derived_stats

double compute_euclidean_distance(
    const double *restrict p1,
    const double *restrict p2,
    int                    dim)
{
    double sum = 0.0;
    for (int k = 0; k < dim; k++)
    {
        double diff = p1[k] - p2[k];
        sum += diff * diff;
    }
    return sqrt(sum);
} // compute_euclidean_distance

int analyze_spatial_spread(
    const char    *points_file,
    const char    *anchors_file,
    AnalysisState *state)
{
    FILE *f_pts = fopen(points_file, "r");
    FILE *f_anc = fopen(anchors_file, "r");
    if (f_pts == NULL || f_anc == NULL)
    {
        if (f_pts != NULL)
        {
            fclose(f_pts);
        }
        if (f_anc != NULL)
        {
            fclose(f_anc);
        }
        return -1;
    }

    int n = state->num_clusters;

    int    dim = 0;
    char   first_line[65536];
    if (fgets(first_line, sizeof(first_line), f_anc) == NULL)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }
    rewind(f_anc);

    {
        char *p = first_line;
        int   in_num = 0;
        while (*p != '\0')
        {
            if (!isspace((unsigned char)*p))
            {
                if (!in_num)
                {
                    dim++;
                    in_num = 1;
                }
            }
            else
            {
                in_num = 0;
            }
            p++;
        }
    }

    if (dim == 0)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }

    double *anchors = malloc((size_t)(n * dim) * sizeof(double));
    if (anchors == NULL)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        char anchor_line[65536];
        if (fgets(anchor_line, sizeof(anchor_line), f_anc) == NULL)
        {
            break;
        }
        char *token = strtok(anchor_line, " \t\r\n");
        for (int k = 0; k < dim && token != NULL; k++)
        {
            anchors[i * dim + k] = atof(token);
            token = strtok(NULL, " \t\r\n");
        }
    }
    fclose(f_anc);

    double *cluster_dist_sum = calloc((size_t)n, sizeof(double));
    double *cluster_dist_sq_sum = calloc((size_t)n, sizeof(double));
    double *cluster_dist_max = calloc((size_t)n, sizeof(double));
    long   *cluster_counts_computed = calloc((size_t)n, sizeof(long));

    double *point_coords = malloc((size_t)dim * sizeof(double));
    if (cluster_dist_sum == NULL || cluster_dist_sq_sum == NULL ||
        cluster_dist_max == NULL || cluster_counts_computed == NULL ||
        point_coords == NULL)
    {
        free(anchors);
        free(cluster_dist_sum);
        free(cluster_dist_sq_sum);
        free(cluster_dist_max);
        free(cluster_counts_computed);
        free(point_coords);
        fclose(f_pts);
        return -1;
    }

    long frame_idx = 0;
    char point_line[65536];
    while (fgets(point_line, sizeof(point_line), f_pts) != NULL)
    {
        if (point_line[0] == '#')
        {
            continue;
        }

        if (frame_idx >= state->assignments_count)
        {
            break;
        }

        char *token = strtok(point_line, " \t\r\n");
        for (int k = 0; k < dim && token != NULL; k++)
        {
            point_coords[k] = atof(token);
            token = strtok(NULL, " \t\r\n");
        }

        int assigned_c = state->assignments[frame_idx];
        if (assigned_c >= 0 && assigned_c < n)
        {
            double d = compute_euclidean_distance(point_coords,
                                                  &anchors[assigned_c * dim],
                                                  dim);
            cluster_dist_sum[assigned_c] += d;
            cluster_dist_sq_sum[assigned_c] += (d * d);
            if (d > cluster_dist_max[assigned_c])
            {
                cluster_dist_max[assigned_c] = d;
            }
            cluster_counts_computed[assigned_c]++;
        }
        frame_idx++;
    }
    fclose(f_pts);

    printf("\n%s--- Cluster Spatial Spread Analysis ---%s\n", ansi_bold_cyan, ansi_reset);
    printf("Dimensions: %d\n", dim);
    printf("  %5s | %10s | %10s | %10s | %10s\n",
           "ID", "Count", "Mean Dist", "Max Dist", "Std Dev");
    for (int i = 0; i < n; i++)
    {
        long cnt = cluster_counts_computed[i];
        if (cnt > 0)
        {
            double mean = cluster_dist_sum[i] / (double)cnt;
            double variance = (cluster_dist_sq_sum[i] / (double)cnt) - (mean * mean);
            if (variance < 0.0)
            {
                variance = 0.0;
            }
            double std_dev = sqrt(variance);
            printf("  %5d | %10ld | %10.5f | %10.5f | %10.5f\n",
                   i, cnt, mean, cluster_dist_max[i], std_dev);
        }
    }

    free(anchors);
    free(cluster_dist_sum);
    free(cluster_dist_sq_sum);
    free(cluster_dist_max);
    free(cluster_counts_computed);
    free(point_coords);
    return 0;
} // analyze_spatial_spread

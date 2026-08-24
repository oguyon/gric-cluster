/**
 * @file knn_tree.c
 * @brief Super-Cluster (Meta-Cluster) hierarchy builder for gric-knn.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_tree.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * knn_build_super_clusters() - Group M clusters into K super-clusters using D_cc.
 * @model: Pointer to KnnModel with populated clusters, radii, and dcc_matrix.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_build_super_clusters(
    KnnModel *model)
{
    if (model == NULL || model->dcc_matrix == NULL || model->num_clusters <= 0)
    {
        return -1;
    }

    int M = model->num_clusters;

    /* For small cluster counts, 1-level search is optimal */
    if (M <= 16)
    {
        model->num_super_clusters = 0;
        model->super_clusters = NULL;
        model->dss_matrix = NULL;
        model->cluster_super_map = NULL;
        return 0;
    }

    /* Target K = ceil(sqrt(M)), clamped between 2 and 64 */
    int K = (int)ceil(sqrt((double)M));
    if (K < 2)
    {
        K = 2;
    }
    if (K > 64)
    {
        K = 64;
    }
    if (K >= M)
    {
        K = M - 1;
    }

    int *seeds = (int *)malloc((size_t)K * sizeof(int));
    double *min_dist = (double *)malloc((size_t)M * sizeof(double));
    if (seeds == NULL || min_dist == NULL)
    {
        free(seeds);
        free(min_dist);
        return -1;
    }

    /* 1. Farthest-point sampling (k-medoids seeding on D_cc) */
    seeds[0] = 0;
    for (int c = 0; c < M; c++)
    {
        min_dist[c] = model->dcc_matrix[c * M + seeds[0]];
    }

    for (int k = 1; k < K; k++)
    {
        int best_c = 0;
        double max_d = -1.0;
        for (int c = 0; c < M; c++)
        {
            if (min_dist[c] > max_d)
            {
                max_d = min_dist[c];
                best_c = c;
            }
        }
        seeds[k] = best_c;

        for (int c = 0; c < M; c++)
        {
            double d = model->dcc_matrix[c * M + best_c];
            if (d < min_dist[c])
            {
                min_dist[c] = d;
            }
        }
    } // for (int k = 1; ...)

    free(min_dist);

    /* 2. Allocate Super-Cluster structures */
    model->num_super_clusters = K;
    model->super_clusters = (KnnSuperCluster *)calloc((size_t)K, sizeof(KnnSuperCluster));
    model->cluster_super_map = (int *)malloc((size_t)M * sizeof(int));
    model->dss_matrix = (double *)malloc((size_t)K * (size_t)K * sizeof(double));

    if (model->super_clusters == NULL || model->cluster_super_map == NULL
        || model->dss_matrix == NULL)
    {
        free(seeds);
        knn_free_super_clusters(model);
        return -1;
    }

    for (int s = 0; s < K; s++)
    {
        model->super_clusters[s].super_id = s;
        model->super_clusters[s].medoid_cluster_id = seeds[s];
        model->super_clusters[s].radius = 0.0;
        model->super_clusters[s].num_clusters = 0;
        model->super_clusters[s].cluster_ids = NULL;
    }

    /* 3. Assign each cluster to its nearest seed */
    for (int c = 0; c < M; c++)
    {
        int best_s = 0;
        double best_d = model->dcc_matrix[c * M + seeds[0]];
        for (int s = 1; s < K; s++)
        {
            double d = model->dcc_matrix[c * M + seeds[s]];
            if (d < best_d)
            {
                best_d = d;
                best_s = s;
            }
        }
        model->cluster_super_map[c] = best_s;
        model->super_clusters[best_s].num_clusters++;
    } // for (int c = 0; ...)

    /* 4. Allocate child cluster ID arrays and compute super-radii */
    for (int s = 0; s < K; s++)
    {
        int count = model->super_clusters[s].num_clusters;
        if (count > 0)
        {
            model->super_clusters[s].cluster_ids = (int *)malloc((size_t)count * sizeof(int));
            if (model->super_clusters[s].cluster_ids == NULL)
            {
                free(seeds);
                knn_free_super_clusters(model);
                return -1;
            }
        }
        model->super_clusters[s].num_clusters = 0; // Reset for insertion indexing
    }

    for (int c = 0; c < M; c++)
    {
        int s = model->cluster_super_map[c];
        int idx = model->super_clusters[s].num_clusters;
        model->super_clusters[s].cluster_ids[idx] = c;
        model->super_clusters[s].num_clusters++;

        int medoid_c = model->super_clusters[s].medoid_cluster_id;
        double d_medoid_c = model->dcc_matrix[medoid_c * M + c];
        double total_r = d_medoid_c + model->clusters[c].radius;
        if (total_r > model->super_clusters[s].radius)
        {
            model->super_clusters[s].radius = total_r;
        }
    } // for (int c = 0; ...)

    /* 5. Build dense K x K inter-super-anchor distance matrix */
    for (int i = 0; i < K; i++)
    {
        int c_i = seeds[i];
        for (int j = 0; j < K; j++)
        {
            int c_j = seeds[j];
            model->dss_matrix[i * K + j] = model->dcc_matrix[c_i * M + c_j];
        }
    } // for (int i = 0; ...)

    free(seeds);
    return 0;
}

/**
 * knn_free_super_clusters() - Free memory allocated for super-cluster structures.
 * @model: Pointer to KnnModel.
 */
void knn_free_super_clusters(
    KnnModel *model)
{
    if (model == NULL)
    {
        return;
    }

    if (model->super_clusters != NULL)
    {
        for (int s = 0; s < model->num_super_clusters; s++)
        {
            if (model->super_clusters[s].cluster_ids != NULL)
            {
                free(model->super_clusters[s].cluster_ids);
                model->super_clusters[s].cluster_ids = NULL;
            }
        }
        free(model->super_clusters);
        model->super_clusters = NULL;
    }

    if (model->cluster_super_map != NULL)
    {
        free(model->cluster_super_map);
        model->cluster_super_map = NULL;
    }

    if (model->dss_matrix != NULL)
    {
        free(model->dss_matrix);
        model->dss_matrix = NULL;
    }

    model->num_super_clusters = 0;
}

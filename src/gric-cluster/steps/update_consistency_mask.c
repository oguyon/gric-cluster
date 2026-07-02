#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include <stdlib.h>
#include <string.h>

/**
 * recompute_consistency_mask - Rebuild the entire geometric consistency mask from scratch.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 *
 * Scans all active cluster triplets (i, j, k) to determine geometric consistency,
 * clearing and rebuilding the entire bitmask. Called during cluster eviction/merges.
 */
void recompute_consistency_mask(
    ClusterConfig *config,
    ClusterState  *state)
{
    int N = config->algo.maxnbclust;
    int words = (N + 63) / 64;
    double rc = config->algo.rlim;

    size_t clear_bytes = (size_t)N * (size_t)N * (size_t)words * sizeof(uint64_t);
    memset(state->scratch.consistency_mask, 0, clear_bytes);

    #pragma omp parallel for if(state->num_clusters >= OMP_MIN_CLUSTERS)
    for (int i = 0; i < state->num_clusters; i++)
    {
        for (int j = 0; j < state->num_clusters; j++)
        {
            uint64_t *mask = &state->scratch.consistency_mask[(i * N + j) * words];

            if (config->optim.sparse_dcc_mode)
            {
                double d_min_ij = state->scratch.dcc_min[i * N + j];
                double d_max_ij = state->scratch.dcc_max[i * N + j];

                for (int k = 0; k < state->num_clusters; k++)
                {
                    double d_min_ik = state->scratch.dcc_min[i * N + k];
                    double d_max_ik = state->scratch.dcc_max[i * N + k];

                    double diff1 = d_min_ik - d_max_ij;
                    double diff2 = d_min_ij - d_max_ik;
                    double delta_min = 0.0;
                    if (diff1 > delta_min) delta_min = diff1;
                    if (diff2 > delta_min) delta_min = diff2;

                    if (delta_min <= 2.0 * rc)
                    {
                        mask[k / 64] |= (1ULL << (k % 64));
                    }
                }
            }
            else
            {
                double measured_dist = state->scratch.dcc_min[i * N + j];
                if (measured_dist < 0.0)
                {
                    continue;
                }

                for (int k = 0; k < state->num_clusters; k++)
                {
                    double dist_ti_k = state->scratch.dcc_min[i * N + k];
                    if (dist_ti_k < 0.0)
                    {
                        continue;
                    }

                    if (dist_ti_k >= (measured_dist - 2.0 * rc) &&
                        dist_ti_k <= (measured_dist + 2.0 * rc))
                    {
                        mask[k / 64] |= (1ULL << (k % 64));
                    }
                }
            }
        }
    }
}

/**
 * update_consistency_mask_for_new_cluster - Compute consistency relations for a new cluster.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @new_cl: Index of the newly added cluster.
 *
 * Computes and sets consistency bitmask flags involving the new cluster, avoiding full recalculations.
 * Handles cases where the new cluster is the target target_ci, hypothesis hypothesis_cj, or candidate k.
 */
void update_consistency_mask_for_new_cluster(
    ClusterConfig *config,
    ClusterState  *state,
    int            new_cl)
{
    int N = config->algo.maxnbclust;
    int words = (N + 63) / 64;
    double rc = config->algo.rlim;

    // 1. Clear new rows/cols of the consistency mask
    memset(&state->scratch.consistency_mask[(size_t)new_cl * (size_t)N * (size_t)words], 0,
           (size_t)(new_cl + 1) * (size_t)words * sizeof(uint64_t));

    #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
    for (int j = 0; j <= new_cl; j++)
    {
        memset(&state->scratch.consistency_mask[(j * N + new_cl) * words], 0, words * sizeof(uint64_t));
    }

    // Pre-load the row of new_cl into heap-allocated arrays to avoid stack overflow
    double *d_min_new_k = (double *)malloc((new_cl + 1) * sizeof(double));
    double *d_max_new_k = (double *)malloc((new_cl + 1) * sizeof(double));
    if (d_min_new_k == NULL || d_max_new_k == NULL)
    {
        free(d_min_new_k);
        free(d_max_new_k);
        return;
    }
    for (int k = 0; k <= new_cl; k++)
    {
        d_min_new_k[k] = state->scratch.dcc_min[new_cl * N + k];
        d_max_new_k[k] = state->scratch.dcc_max[new_cl * N + k];
    }

    // 2. Compute for target_ci = new_cl
    #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
    for (int j = 0; j <= new_cl; j++)
    {
        uint64_t *mask = &state->scratch.consistency_mask[(new_cl * N + j) * words];
        if (config->optim.sparse_dcc_mode)
        {
            double d_min_ij = state->scratch.dcc_min[new_cl * N + j];
            double d_max_ij = state->scratch.dcc_max[new_cl * N + j];

            for (int k = 0; k <= new_cl; k++)
            {
                double d_min_ik = d_min_new_k[k];
                double d_max_ik = d_max_new_k[k];

                double diff1 = d_min_ik - d_max_ij;
                double diff2 = d_min_ij - d_max_ik;
                double delta_min = 0.0;
                if (diff1 > delta_min) delta_min = diff1;
                if (diff2 > delta_min) delta_min = diff2;

                if (delta_min <= 2.0 * rc)
                {
                    mask[k / 64] |= (1ULL << (k % 64));
                }
            }
        }
        else
        {
            double measured_dist = state->scratch.dcc_min[new_cl * N + j];
            if (measured_dist >= 0.0)
            {
                for (int k = 0; k <= new_cl; k++)
                {
                    double dist_ti_k = d_min_new_k[k];
                    if (dist_ti_k >= 0.0 &&
                        dist_ti_k >= (measured_dist - 2.0 * rc) &&
                        dist_ti_k <= (measured_dist + 2.0 * rc))
                    {
                        mask[k / 64] |= (1ULL << (k % 64));
                    }
                }
            }
        }
    }

    // 3. Compute for hypothesis_cj = new_cl (with target_ci < new_cl)
    #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
    for (int i = 0; i < new_cl; i++)
    {
        uint64_t *mask = &state->scratch.consistency_mask[(i * N + new_cl) * words];
        double *d_min_i = &state->scratch.dcc_min[i * N];
        double *d_max_i = &state->scratch.dcc_max[i * N];

        if (config->optim.sparse_dcc_mode)
        {
            double d_min_inew = d_min_i[new_cl];
            double d_max_inew = d_max_i[new_cl];

            for (int k = 0; k <= new_cl; k++)
            {
                double d_min_ik = d_min_i[k];
                double d_max_ik = d_max_i[k];

                double diff1 = d_min_ik - d_max_inew;
                double diff2 = d_min_inew - d_max_ik;
                double delta_min = 0.0;
                if (diff1 > delta_min) delta_min = diff1;
                if (diff2 > delta_min) delta_min = diff2;

                if (delta_min <= 2.0 * rc)
                {
                    mask[k / 64] |= (1ULL << (k % 64));
                }
            }
        }
        else
        {
            double measured_dist = d_min_i[new_cl];
            if (measured_dist >= 0.0)
            {
                for (int k = 0; k <= new_cl; k++)
                {
                    double dist_ti_k = d_min_i[k];
                    if (dist_ti_k >= 0.0 &&
                        dist_ti_k >= (measured_dist - 2.0 * rc) &&
                        dist_ti_k <= (measured_dist + 2.0 * rc))
                    {
                        mask[k / 64] |= (1ULL << (k % 64));
                    }
                }
            }
        }
    }

    // 4. Compute for candidate k = new_cl (with target_ci < new_cl and hypothesis_cj < new_cl)
    int word_offset = new_cl / 64;
    uint64_t bit_mask = 1ULL << (new_cl % 64);

    #pragma omp parallel for if(new_cl >= OMP_MIN_CLUSTERS)
    for (int i = 0; i < new_cl; i++)
    {
        double d_min_inew = state->scratch.dcc_min[i * N + new_cl];
        double d_max_inew = state->scratch.dcc_max[i * N + new_cl];
        double *dcc_min_i = &state->scratch.dcc_min[i * N];
        double *dcc_max_i = &state->scratch.dcc_max[i * N];
        uint64_t *mask_row = &state->scratch.consistency_mask[i * N * words];

        for (int j = 0; j < new_cl; j++)
        {
            if (config->optim.sparse_dcc_mode)
            {
                double d_min_ij = dcc_min_i[j];
                double d_max_ij = dcc_max_i[j];

                double diff1 = d_min_inew - d_max_ij;
                double diff2 = d_min_ij - d_max_inew;
                double delta_min = 0.0;
                if (diff1 > delta_min) delta_min = diff1;
                if (diff2 > delta_min) delta_min = diff2;

                if (delta_min <= 2.0 * rc)
                {
                    mask_row[j * words + word_offset] |= bit_mask;
                }
            }
            else
            {
                double measured_dist = dcc_min_i[j];
                if (measured_dist >= 0.0)
                {
                    double dist_ti_new = d_min_inew;
                    if (dist_ti_new >= 0.0 &&
                        dist_ti_new >= (measured_dist - 2.0 * rc) &&
                        dist_ti_new <= (measured_dist + 2.0 * rc))
                    {
                        mask_row[j * words + word_offset] |= bit_mask;
                    }
                }
            }
        }
    }

    free(d_min_new_k);
    free(d_max_new_k);
}

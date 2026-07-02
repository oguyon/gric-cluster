#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
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
    for (int j = 0; j <= new_cl; j++)
    {
        memset(&state->scratch.consistency_mask[(new_cl * N + j) * words], 0, words * sizeof(uint64_t));
        memset(&state->scratch.consistency_mask[(j * N + new_cl) * words], 0, words * sizeof(uint64_t));
    }

    // 2. Compute for target_ci = new_cl
    for (int j = 0; j <= new_cl; j++)
    {
        uint64_t *mask = &state->scratch.consistency_mask[(new_cl * N + j) * words];
        if (config->optim.sparse_dcc_mode)
        {
            double d_min_ij = state->scratch.dcc_min[new_cl * N + j];
            double d_max_ij = state->scratch.dcc_max[new_cl * N + j];

            for (int k = 0; k <= new_cl; k++)
            {
                double d_min_ik = state->scratch.dcc_min[new_cl * N + k];
                double d_max_ik = state->scratch.dcc_max[new_cl * N + k];

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
                    double dist_ti_k = state->scratch.dcc_min[new_cl * N + k];
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
    for (int i = 0; i < new_cl; i++)
    {
        uint64_t *mask = &state->scratch.consistency_mask[(i * N + new_cl) * words];
        if (config->optim.sparse_dcc_mode)
        {
            double d_min_inew = state->scratch.dcc_min[i * N + new_cl];
            double d_max_inew = state->scratch.dcc_max[i * N + new_cl];

            for (int k = 0; k <= new_cl; k++)
            {
                double d_min_ik = state->scratch.dcc_min[i * N + k];
                double d_max_ik = state->scratch.dcc_max[i * N + k];

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
            double measured_dist = state->scratch.dcc_min[i * N + new_cl];
            if (measured_dist >= 0.0)
            {
                for (int k = 0; k <= new_cl; k++)
                {
                    double dist_ti_k = state->scratch.dcc_min[i * N + k];
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
    for (int i = 0; i < new_cl; i++)
    {
        for (int j = 0; j < new_cl; j++)
        {
            if (config->optim.sparse_dcc_mode)
            {
                double d_min_ij = state->scratch.dcc_min[i * N + j];
                double d_max_ij = state->scratch.dcc_max[i * N + j];
                double d_min_inew = state->scratch.dcc_min[i * N + new_cl];
                double d_max_inew = state->scratch.dcc_max[i * N + new_cl];

                double diff1 = d_min_inew - d_max_ij;
                double diff2 = d_min_ij - d_max_inew;
                double delta_min = 0.0;
                if (diff1 > delta_min) delta_min = diff1;
                if (diff2 > delta_min) delta_min = diff2;

                if (delta_min <= 2.0 * rc)
                {
                    uint64_t *mask = &state->scratch.consistency_mask[(i * N + j) * words];
                    mask[new_cl / 64] |= (1ULL << (new_cl % 64));
                }
            }
            else
            {
                double measured_dist = state->scratch.dcc_min[i * N + j];
                if (measured_dist >= 0.0)
                {
                    double dist_ti_new = state->scratch.dcc_min[i * N + new_cl];
                    if (dist_ti_new >= 0.0 &&
                        dist_ti_new >= (measured_dist - 2.0 * rc) &&
                        dist_ti_new <= (measured_dist + 2.0 * rc))
                    {
                        uint64_t *mask = &state->scratch.consistency_mask[(i * N + j) * words];
                        mask[new_cl / 64] |= (1ULL << (new_cl % 64));
                    }
                }
            }
        }
    }
}

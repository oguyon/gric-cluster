/**
 * @file update_probabilities_and_pruning.c
 * @brief Bayesian probability updates and candidate
 *        pruning after distance measurements.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include "cluster_prune.h"
#include "cluster_math.h"
#include <math.h>

/* OMP_MIN_CLUSTERS — defined in cluster_defs.h */

/**
 * update_probabilities_and_pruning - Prune search space and update geometric priorities.
 * @cj: Cluster index measured in the last step.
 * @dfc: Computed distance to cluster index cj.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @temp_indices: Array of cluster indices measured in this frame.
 * @temp_dists: Array of computed distances in this frame.
 * @temp_count: Total count of measurements recorded in this frame.
 *
 * Employs Multi-Point Triangle Inequality heuristics (TE4/TE5) to prune distant
 * cluster candidates (setting clmembflag[cl] = 0). Updates geometric probabilities.
 */
void update_probabilities_and_pruning(
    int            cj,
    double         dfc,
    ClusterConfig *config,
    ClusterState  *state,
    int           *temp_indices,
    double        *temp_dists,
    int            temp_count)
{
    long local_pruned = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
    for (int cl = 0; cl < state->num_clusters; cl++)
    {
        if (state->scratch.clmembflag[cl] == 0)
        {
            continue;
        }

        if (config->optim.sparse_dcc_mode)
        {
            double d_min = state->scratch.dcc_min[cj * config->algo.maxnbclust + cl];
            double d_max = state->scratch.dcc_max[cj * config->algo.maxnbclust + cl];

            if (d_min - dfc > config->algo.rlim)
            {
                state->scratch.clmembflag[cl] = 0;
                local_pruned++;
            }
            else if (d_max < 1e18 && dfc - d_max > config->algo.rlim)
            {
                state->scratch.clmembflag[cl] = 0;
                local_pruned++;
            }
        }
        else
        {
            double dcc = state->scratch.dcc_min[cj * config->algo.maxnbclust + cl];
            if (dcc < 0.0)
            {
                dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[cl].anchor, -1,
                               -1.0, -1.0, config, state);
                state->scratch.dcc_min[cj * config->algo.maxnbclust + cl] = dcc;
                state->scratch.dcc_min[cl * config->algo.maxnbclust + cj] = dcc;
                state->scratch.dcc_max[cj * config->algo.maxnbclust + cl] = dcc;
                state->scratch.dcc_max[cl * config->algo.maxnbclust + cj] = dcc;
                state->scratch.dcc_measured[cj * config->algo.maxnbclust + cl] = 1;
                state->scratch.dcc_measured[cl * config->algo.maxnbclust + cj] = 1;
            }

            if (dcc - dfc > config->algo.rlim)
            {
                state->scratch.clmembflag[cl] = 0;
                local_pruned++;
            }
            else if (dfc - dcc > config->algo.rlim)
            {
                state->scratch.clmembflag[cl] = 0;
                local_pruned++;
            }
        }
    }
    state->telemetry.clusters_pruned += local_pruned;

    if (config->optim.te4_mode && temp_count > 1)
    {
        for (int p = 0; p < temp_count - 1; p++)
        {
            int    cprev = temp_indices[p];
            double d_m_cprev = temp_dists[p];
            double d_ci_cprev = 0.0;

            if (config->optim.sparse_dcc_mode)
            {
                if (!state->scratch.dcc_measured[cj * config->algo.maxnbclust + cprev])
                {
                    continue;
                }
                d_ci_cprev = state->scratch.dcc_min[cj * config->algo.maxnbclust + cprev];
            }
            else
            {
                d_ci_cprev = state->scratch.dcc_min[cj * config->algo.maxnbclust + cprev];
                if (d_ci_cprev < 0.0)
                {
                    d_ci_cprev = get_dist(&state->clusters[cj].anchor,
                                          &state->clusters[cprev].anchor, -1, -1.0, -1.0,
                                          config, state);
                    state->scratch.dcc_min[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                    state->scratch.dcc_min[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
                    state->scratch.dcc_max[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                    state->scratch.dcc_max[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
                    state->scratch.dcc_measured[cj * config->algo.maxnbclust + cprev] = 1;
                    state->scratch.dcc_measured[cprev * config->algo.maxnbclust + cj] = 1;
                }
            }

            long local_pruned_te4 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : local_pruned_te4) if(state->num_clusters >= OMP_MIN_CLUSTERS)
#endif
            for (int k = 0; k < state->num_clusters; k++)
            {
                if (!state->scratch.clmembflag[k])
                {
                    continue;
                }
                if (k == cj || k == cprev)
                {
                    continue;
                }

                double d_ci_ck = 0.0;
                double d_cprev_ck = 0.0;

                if (config->optim.sparse_dcc_mode)
                {
                    if (!state->scratch.dcc_measured[cj * config->algo.maxnbclust + k] ||
                        !state->scratch.dcc_measured[cprev * config->algo.maxnbclust + k])
                    {
                        continue;
                    }
                    d_ci_ck = state->scratch.dcc_min[cj * config->algo.maxnbclust + k];
                    d_cprev_ck = state->scratch.dcc_min[cprev * config->algo.maxnbclust + k];
                }
                else
                {
                    d_ci_ck = state->scratch.dcc_min[cj * config->algo.maxnbclust + k];
                    if (d_ci_ck < 0.0)
                    {
                        d_ci_ck = get_dist(&state->clusters[cj].anchor,
                                           &state->clusters[k].anchor, -1, -1.0, -1.0,
                                           config, state);
                        state->scratch.dcc_min[cj * config->algo.maxnbclust + k] = d_ci_ck;
                        state->scratch.dcc_min[k * config->algo.maxnbclust + cj] = d_ci_ck;
                        state->scratch.dcc_max[cj * config->algo.maxnbclust + k] = d_ci_ck;
                        state->scratch.dcc_max[k * config->algo.maxnbclust + cj] = d_ci_ck;
                        state->scratch.dcc_measured[cj * config->algo.maxnbclust + k] = 1;
                        state->scratch.dcc_measured[k * config->algo.maxnbclust + cj] = 1;
                    }

                    d_cprev_ck = state->scratch.dcc_min[cprev * config->algo.maxnbclust + k];
                    if (d_cprev_ck < 0.0)
                    {
                        d_cprev_ck = get_dist(
                            &state->clusters[cprev].anchor, &state->clusters[k].anchor,
                            -1, -1.0, -1.0, config, state);
                        state->scratch.dcc_min[cprev * config->algo.maxnbclust + k] = d_cprev_ck;
                        state->scratch.dcc_min[k * config->algo.maxnbclust + cprev] = d_cprev_ck;
                        state->scratch.dcc_max[cprev * config->algo.maxnbclust + k] = d_cprev_ck;
                        state->scratch.dcc_max[k * config->algo.maxnbclust + cprev] = d_cprev_ck;
                        state->scratch.dcc_measured[cprev * config->algo.maxnbclust + k] = 1;
                        state->scratch.dcc_measured[k * config->algo.maxnbclust + cprev] = 1;
                    }
                }

                double min_d = calc_min_dist_4pt(dfc, d_m_cprev, d_ci_cprev, d_ci_ck, d_cprev_ck);
                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[k] = 0;
                    local_pruned_te4++;
                }
            }
            state->telemetry.clusters_pruned += local_pruned_te4;
        }
    }

    if (config->optim.te5_mode)
    {
        prune_candidates_te5(config, state, temp_indices, temp_dists, temp_count);
    }

    state->scratch.clmembflag[cj] = 0;

    int active_cluster_count = 0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i])
        {
            active_cluster_count++;
        }
    }

    if ((config->optim.gprob_mode || (config->output.distall_mode && state->distall_out) ||
         config->output.verbose_level >= 2) &&
        active_cluster_count > 1)
    {
        update_geometric_probabilities(config, state, cj, dfc);
    }

    /* Apply pruning and soft Bayesian updates to the posterior */
    double sum_p = 0.0;

    if (config->optim.soft_bayesian_mode)
    {
        double sigma = config->optim.soft_bayesian_sigma_coeff * config->algo.rlim;
        double two_sigma_sq = 2.0 * sigma * sigma;
        int N = config->algo.maxnbclust;

        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i] == 0)
            {
                state->scratch.entropy_p_current[i] = 0.0;
            }
            else
            {
                double dcc = state->scratch.dcc_min[cj * N + i];
                if (dcc < 0.0)
                {
                    dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[i].anchor, -1,
                                   -1.0, -1.0, config, state);
                    state->scratch.dcc_min[cj * N + i] = dcc;
                    state->scratch.dcc_min[i * N + cj] = dcc;
                    state->scratch.dcc_max[cj * N + i] = dcc;
                    state->scratch.dcc_max[i * N + cj] = dcc;
                    state->scratch.dcc_measured[cj * N + i] = 1;
                    state->scratch.dcc_measured[i * N + cj] = 1;
                }
                double diff = dfc - dcc;
                double x = (diff * diff) / two_sigma_sq;
                double likelihood = 0.0;
                /*
                 * Minimax polynomial approximation of
                 * exp(-x) on [0, 2], accurate to ~1e-4.
                 * Avoids expensive exp() in the inner
                 * loop.
                 */
                if (x <= 2.0)
                {
                    likelihood = 1.0 - x * (0.978371 - x * (0.419481 - x * 0.073231));
                }
                state->scratch.entropy_p_current[i] *= likelihood;
                sum_p += state->scratch.entropy_p_current[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i] == 0)
            {
                state->scratch.entropy_p_current[i] = 0.0;
            }
            else
            {
                sum_p += state->scratch.entropy_p_current[i];
            }
        }
    }

    /* Renormalize posterior */
    if (sum_p > 0.0)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->scratch.entropy_p_current[i] /= sum_p;
        }
    }
    else
    {
        /* Fallback: flat distribution over remaining active clusters */
        int active_cnt = 0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i])
            {
                active_cnt++;
            }
        }
        if (active_cnt > 0)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                state->scratch.entropy_p_current[i] =
                    state->scratch.clmembflag[i] ? (1.0 / active_cnt) : 0.0;
            }
        }
    }
}

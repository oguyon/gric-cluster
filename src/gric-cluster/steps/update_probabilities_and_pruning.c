#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include "cluster_prune.h"
#include "cluster_math.h"

#define OMP_MIN_CLUSTERS 256

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

        double dcc = state->scratch.dccarray[cj * config->algo.maxnbclust + cl];
        if (dcc < 0)
        {
            dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[cl].anchor, -1,
                           -1.0, -1.0, config, state);
            state->scratch.dccarray[cj * config->algo.maxnbclust + cl] = dcc;
            state->scratch.dccarray[cl * config->algo.maxnbclust + cj] = dcc;
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
    state->telemetry.clusters_pruned += local_pruned;

    if (config->optim.te4_mode && temp_count > 1)
    {
        for (int p = 0; p < temp_count - 1; p++)
        {
            int    cprev = temp_indices[p];
            double d_m_cprev = temp_dists[p];
            double d_ci_cprev = state->scratch.dccarray[cj * config->algo.maxnbclust + cprev];

            if (d_ci_cprev < 0)
            {
                d_ci_cprev = get_dist(&state->clusters[cj].anchor, &state->clusters[cprev].anchor,
                                      -1, -1.0, -1.0, config, state);
                state->scratch.dccarray[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                state->scratch.dccarray[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
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

                double d_ci_ck = state->scratch.dccarray[cj * config->algo.maxnbclust + k];
                if (d_ci_ck < 0)
                {
                    d_ci_ck = get_dist(&state->clusters[cj].anchor, &state->clusters[k].anchor,
                                       -1, -1.0, -1.0, config, state);
                    state->scratch.dccarray[cj * config->algo.maxnbclust + k] = d_ci_ck;
                    state->scratch.dccarray[k * config->algo.maxnbclust + cj] = d_ci_ck;
                }

                double d_cprev_ck = state->scratch.dccarray[cprev * config->algo.maxnbclust + k];
                if (d_cprev_ck < 0)
                {
                    d_cprev_ck = get_dist(
                        &state->clusters[cprev].anchor, &state->clusters[k].anchor,
                        -1, -1.0, -1.0, config, state);
                    state->scratch.dccarray[cprev * config->algo.maxnbclust + k] = d_cprev_ck;
                    state->scratch.dccarray[k * config->algo.maxnbclust + cprev] = d_cprev_ck;
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
}

/**
 * @file update_probabilities_and_pruning.c
 * @brief Bayesian probability updates and candidate
 *        pruning after distance measurements.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include "cluster_prune.h"
#include "cluster_bounds.h"
#include "cluster_math.h"
#include "cluster_locator.h"
#include "gric_simd.h"
#include <math.h>
#include "cluster_trace.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

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
    int maxnbc = config->algo.maxnbclust;
    double rlim = config->algo.rlim;

    if (config->optim.sparse_dcc_mode)
    {
        int active_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int idx = 0;

        while (idx < active_cnt)
        {
            int cl = act[idx];
            double d_min = state->scratch.dcc_min[cj * maxnbc + cl];
            double d_max = state->scratch.dcc_max[cj * maxnbc + cl];

            if (d_min - dfc > rlim || (d_max < 1e18 && dfc - d_max > rlim))
            {
                state->scratch.clmembflag[cl] = 0;
                state->scratch.entropy_p_current[cl] = 0.0;
                local_pruned++;
                active_cnt--;
                act[idx] = act[active_cnt];
            }
            else
            {
                idx++;
            }
        }
        state->scratch.num_active_clusters = active_cnt;
    }
    else
    {
        const uint16_t *row_sq16 = (state->scratch.dcc_sq16 != NULL)
            ? &state->scratch.dcc_sq16[cj * maxnbc]
            : NULL;
        const double *row_dcc = &state->scratch.dcc_min[cj * maxnbc];
        int active_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int idx = 0;

        if (row_sq16 != NULL)
        {
            double scale = state->scratch.dcc_sq16_scale;
            int q_dfc = (int)(dfc * scale + 0.5);
            int q_rlim = (int)(rlim * scale);

            while (idx < active_cnt)
            {
                int cl = act[idx];
                uint16_t q_dcc = row_sq16[cl];

                if (q_dcc == DCC_SQ16_UNMEASURED)
                {
                    double dcc = get_dist(
                        &state->clusters[cj].anchor,
                        &state->clusters[cl].anchor,
                        -1, -1.0, -1.0, config, state);
                    set_dcc_pair(state, maxnbc, cj, cl, dcc);
                    q_dcc = row_sq16[cl];
                }

                int diff = abs((int)q_dcc - q_dfc);
                int prune = 0;

                if (diff > q_rlim + 1)
                {
                    prune = 1;
                }
                else if (diff >= q_rlim - 1)
                {
                    /* Boundary verification against exact double */
                    double dcc = row_dcc[cl];
                    if (fabs(dcc - dfc) > rlim)
                    {
                        prune = 1;
                    }
                }

                if (prune)
                {
                    state->scratch.clmembflag[cl] = 0;
                    state->scratch.entropy_p_current[cl] = 0.0;
                    local_pruned++;
                    active_cnt--;
                    act[idx] = act[active_cnt];
                }
                else
                {
                    idx++;
                }
            }
        }
        else
        {
            while (idx < active_cnt)
            {
                int cl = act[idx];
                double dcc = row_dcc[cl];
                if (dcc < 0.0)
                {
                    dcc = get_dist(
                        &state->clusters[cj].anchor,
                        &state->clusters[cl].anchor,
                        -1, -1.0, -1.0, config, state);
                    set_dcc_pair(state, maxnbc, cj, cl, dcc);
                }

                if (fabs(dcc - dfc) > rlim)
                {
                    state->scratch.clmembflag[cl] = 0;
                    state->scratch.entropy_p_current[cl] = 0.0;
                    local_pruned++;
                    active_cnt--;
                    act[idx] = act[active_cnt];
                }
                else
                {
                    idx++;
                }
            }
        }
        state->scratch.num_active_clusters = active_cnt;
    }
    state->telemetry.clusters_pruned += local_pruned;

    if (state->trace)
    {
        TraceEvent *ev = trace_emit(state->trace, TRACE_PRUNE_3P);
        if (ev)
        {
            ev->pruned_count = local_pruned;
            ev->cluster_id = cj;
            ev->active_remaining = state->scratch.num_active_clusters;
        }
    }

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
                    set_dcc_pair(state, config->algo.maxnbclust, cj, cprev, d_ci_cprev);
                }
            }

            TE4Ref te4_ref;
            calc_te4_ref_init(&te4_ref, dfc, d_m_cprev, d_ci_cprev);

            const double *row_dcc_cj = &state->scratch.dcc_min[cj * config->algo.maxnbclust];
            const double *row_dcc_cprev = &state->scratch.dcc_min[cprev * config->algo.maxnbclust];
            const char   *row_meas_cj = &state->scratch.dcc_measured[cj * config->algo.maxnbclust];
            const char   *row_meas_cprev =
                &state->scratch.dcc_measured[cprev * config->algo.maxnbclust];

            long local_pruned_te4 = 0;
            int k = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
            if (!config->optim.sparse_dcc_mode && gric_get_simd_level() >= GRIC_SIMD_AVX512)
            {
                for (; k <= state->num_clusters - 8; k += 8)
                {
                    int any_alive = 0;
                    for (int sub = 0; sub < 8; sub++)
                    {
                        if (state->scratch.clmembflag[k + sub])
                        {
                            any_alive = 1;
                            break;
                        }
                    }
                    if (!any_alive)
                    {
                        continue;
                    }

                    int any_missing = 0;
                    for (int sub = 0; sub < 8; sub++)
                    {
                        int kk = k + sub;
                        if (row_dcc_cj[kk] < 0.0 || row_dcc_cprev[kk] < 0.0)
                        {
                            any_missing = 1;
                            break;
                        }
                    }

                    if (any_missing)
                    {
                        for (int sub = 0; sub < 8; sub++)
                        {
                            int kk = k + sub;
                            if (!state->scratch.clmembflag[kk] || kk == cj || kk == cprev)
                            {
                                continue;
                            }
                            double d_ci_ck = row_dcc_cj[kk];
                            if (d_ci_ck < 0.0)
                            {
                                d_ci_ck = get_dist(&state->clusters[cj].anchor,
                                                   &state->clusters[kk].anchor, -1, -1.0, -1.0,
                                                   config, state);
                                set_dcc_pair(state, config->algo.maxnbclust, cj, kk, d_ci_ck);
                            }
                            double d_cprev_ck = row_dcc_cprev[kk];
                            if (d_cprev_ck < 0.0)
                            {
                                d_cprev_ck = get_dist(&state->clusters[cprev].anchor,
                                                      &state->clusters[kk].anchor,
                                                      -1, -1.0, -1.0, config, state);
                                set_dcc_pair(state, config->algo.maxnbclust, cprev, kk, d_cprev_ck);
                            }
                            double min_d = calc_min_dist_4pt_ref(&te4_ref, d_ci_ck, d_cprev_ck);
                            if (min_d > config->algo.rlim)
                            {
                                state->scratch.clmembflag[kk] = 0;
                                local_pruned_te4++;
                            }
                        }
                        continue;
                    }

                    double b_out[8];
                    calc_min_dist_4pt_batch8_avx512(&te4_ref, &row_dcc_cj[k],
                                                    &row_dcc_cprev[k], b_out);
                    for (int sub = 0; sub < 8; sub++)
                    {
                        int kk = k + sub;
                        if (kk == cj || kk == cprev)
                        {
                            continue;
                        }
                        if (state->scratch.clmembflag[kk] && b_out[sub] > config->algo.rlim)
                        {
                            state->scratch.clmembflag[kk] = 0;
                            local_pruned_te4++;
                        }
                    }
                }
            }

            if (!config->optim.sparse_dcc_mode && gric_get_simd_level() >= GRIC_SIMD_AVX2)
            {
                for (; k <= state->num_clusters - 4; k += 4)
                {
                    if (!state->scratch.clmembflag[k] &&
                        !state->scratch.clmembflag[k + 1] &&
                        !state->scratch.clmembflag[k + 2] &&
                        !state->scratch.clmembflag[k + 3])
                    {
                        continue;
                    }
                    if (row_dcc_cj[k] < 0.0 || row_dcc_cj[k + 1] < 0.0 ||
                        row_dcc_cj[k + 2] < 0.0 || row_dcc_cj[k + 3] < 0.0 ||
                        row_dcc_cprev[k] < 0.0 || row_dcc_cprev[k + 1] < 0.0 ||
                        row_dcc_cprev[k + 2] < 0.0 || row_dcc_cprev[k + 3] < 0.0)
                    {
                        for (int sub = 0; sub < 4; sub++)
                        {
                            int kk = k + sub;
                            if (!state->scratch.clmembflag[kk] || kk == cj || kk == cprev)
                            {
                                continue;
                            }
                            double d_ci_ck = row_dcc_cj[kk];
                            if (d_ci_ck < 0.0)
                            {
                                d_ci_ck = get_dist(&state->clusters[cj].anchor,
                                                   &state->clusters[kk].anchor, -1, -1.0, -1.0,
                                                   config, state);
                                set_dcc_pair(state, config->algo.maxnbclust, cj, kk, d_ci_ck);
                            }
                            double d_cprev_ck = row_dcc_cprev[kk];
                            if (d_cprev_ck < 0.0)
                            {
                                d_cprev_ck = get_dist(&state->clusters[cprev].anchor,
                                                      &state->clusters[kk].anchor,
                                                      -1, -1.0, -1.0, config, state);
                                set_dcc_pair(state, config->algo.maxnbclust, cprev, kk, d_cprev_ck);
                            }
                            double min_d = calc_min_dist_4pt_ref(&te4_ref, d_ci_ck, d_cprev_ck);
                            if (min_d > config->algo.rlim)
                            {
                                state->scratch.clmembflag[kk] = 0;
                                local_pruned_te4++;
                            }
                        }
                        continue;
                    }

                    double b_out[4];
                    calc_min_dist_4pt_batch4_avx2(&te4_ref, &row_dcc_cj[k],
                                                  &row_dcc_cprev[k], b_out);
                    for (int sub = 0; sub < 4; sub++)
                    {
                        int kk = k + sub;
                        if (kk == cj || kk == cprev)
                        {
                            continue;
                        }
                        if (state->scratch.clmembflag[kk] && b_out[sub] > config->algo.rlim)
                        {
                            state->scratch.clmembflag[kk] = 0;
                            local_pruned_te4++;
                        }
                    }
                }
            }
#endif

            for (; k < state->num_clusters; k++)
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
                    if (!row_meas_cj[k] || !row_meas_cprev[k])
                    {
                        continue;
                    }
                    d_ci_ck = row_dcc_cj[k];
                    d_cprev_ck = row_dcc_cprev[k];
                }
                else
                {
                    d_ci_ck = row_dcc_cj[k];
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

                    d_cprev_ck = row_dcc_cprev[k];
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

                double min_d = calc_min_dist_4pt_ref(&te4_ref, d_ci_ck, d_cprev_ck);
                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[k] = 0;
                    local_pruned_te4++;
                }
            } // for (int k = 0; ...)
            state->telemetry.clusters_pruned += local_pruned_te4;

            if (state->trace)
            {
                TraceEvent *ev = trace_emit(state->trace, TRACE_PRUNE_4P);
                if (ev)
                {
                    ev->pruned_count = local_pruned_te4;
                    ev->cluster_id = cj;
                    int active_cnt = 0;
                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        if (state->scratch.clmembflag[i]) active_cnt++;
                    }
                    ev->active_remaining = active_cnt;
                }
            }
        }
    }

    if (config->optim.te5_mode)
    {
        int active_before = 0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i]) active_before++;
        }

        prune_candidates_te5(config, state, temp_indices, temp_dists, temp_count);

        if (state->trace)
        {
            int active_after = 0;
            for (int i = 0; i < state->num_clusters; i++)
            {
                if (state->scratch.clmembflag[i]) active_after++;
            }

            TraceEvent *ev = trace_emit(state->trace, TRACE_PRUNE_5P);
            if (ev)
            {
                ev->pruned_count = active_before - active_after;
                ev->active_remaining = active_after;
            }
        }
    }

    state->scratch.clmembflag[cj] = 0;
    state->scratch.entropy_p_current[cj] = 0.0;

    /* Ensure cj itself is removed from active list */
    for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
    {
        if (state->scratch.active_clusters[idx] == cj)
        {
            state->scratch.num_active_clusters--;
            state->scratch.active_clusters[idx] =
                state->scratch.active_clusters[state->scratch.num_active_clusters];
            break;
        }
    }

    int active_cluster_count = state->scratch.num_active_clusters;

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
        int act_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        int idx = 0;

        while (idx < act_cnt)
        {
            int i = act[idx];
            double dcc = state->scratch.dcc_min[cj * N + i];
            if (dcc < 0.0)
            {
                dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[i].anchor, -1,
                               -1.0, -1.0, config, state);
                set_dcc_pair(state, N, cj, i, dcc);
            }
            double diff = dfc - dcc;
            double x = (diff * diff) / two_sigma_sq;
            double likelihood = 0.0;
            if (x <= 2.0)
            {
                likelihood = 1.0 - x * (0.978371 - x * (0.419481 - x * 0.073231));
            }
            state->scratch.entropy_p_current[i] *= likelihood;
            if (state->scratch.entropy_p_current[i] < 1e-15)
            {
                state->scratch.clmembflag[i] = 0;
                state->scratch.entropy_p_current[i] = 0.0;
                act_cnt--;
                act[idx] = act[act_cnt];
            }
            else
            {
                sum_p += state->scratch.entropy_p_current[i];
                idx++;
            }
        }
        state->scratch.num_active_clusters = act_cnt;
    }
    else
    {
        for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
        {
            int i = state->scratch.active_clusters[idx];
            sum_p += state->scratch.entropy_p_current[i];
        }
    }

    /* Renormalize posterior */
    if (sum_p > 0.0)
    {
        double inv_sum_p = 1.0 / sum_p;
        for (int idx = 0; idx < state->scratch.num_active_clusters; idx++)
        {
            int i = state->scratch.active_clusters[idx];
            state->scratch.entropy_p_current[i] *= inv_sum_p;
        }
    }
    else
    {
        /* Fallback: flat distribution over remaining active clusters */
        int active_cnt = state->scratch.num_active_clusters;
        if (active_cnt > 0)
        {
            double flat_p = 1.0 / active_cnt;
            for (int idx = 0; idx < active_cnt; idx++)
            {
                int i = state->scratch.active_clusters[idx];
                state->scratch.entropy_p_current[i] = flat_p;
            }
        }
    }
}

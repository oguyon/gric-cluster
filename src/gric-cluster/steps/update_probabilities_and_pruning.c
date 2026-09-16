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
#include "cluster_locator.h"
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
    if (config->optim.sparse_dcc_mode)
    {
        for (int cl = 0; cl < state->num_clusters; cl++)
        {
            if (state->scratch.clmembflag[cl] == 0)
            {
                continue;
            }

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
    }
    else
    {
        const double *row_dcc = &state->scratch.dcc_min[cj * config->algo.maxnbclust];
        int cl = 0;

#if defined(__AVX2__)
        __m256d v_dfc = _mm256_set1_pd(dfc);
        __m256d v_rlim = _mm256_set1_pd(config->algo.rlim);
        __m256d v_sign_mask = _mm256_set1_pd(-0.0);
        __m256d v_zero = _mm256_setzero_pd();

        for (; cl <= state->num_clusters - 4; cl += 4)
        {
            if (!state->scratch.clmembflag[cl] &&
                !state->scratch.clmembflag[cl + 1] &&
                !state->scratch.clmembflag[cl + 2] &&
                !state->scratch.clmembflag[cl + 3])
            {
                continue;
            }

            __m256d vdcc = _mm256_loadu_pd(&row_dcc[cl]);
            if (_mm256_movemask_pd(_mm256_cmp_pd(vdcc, v_zero, _CMP_LT_OQ)) != 0)
            {
                for (int sub = 0; sub < 4; sub++)
                {
                    int c = cl + sub;
                    if (!state->scratch.clmembflag[c])
                    {
                        continue;
                    }
                    double dcc = row_dcc[c];
                    if (dcc < 0.0)
                    {
                        dcc = get_dist(&state->clusters[cj].anchor, &state->clusters[c].anchor,
                                       -1, -1.0, -1.0, config, state);
                        state->scratch.dcc_min[cj * config->algo.maxnbclust + c] = dcc;
                        state->scratch.dcc_min[c * config->algo.maxnbclust + cj] = dcc;
                        state->scratch.dcc_max[cj * config->algo.maxnbclust + c] = dcc;
                        state->scratch.dcc_max[c * config->algo.maxnbclust + cj] = dcc;
                        state->scratch.dcc_measured[cj * config->algo.maxnbclust + c] = 1;
                        state->scratch.dcc_measured[c * config->algo.maxnbclust + cj] = 1;
                    }
                    if (fabs(dcc - dfc) > config->algo.rlim)
                    {
                        state->scratch.clmembflag[c] = 0;
                        local_pruned++;
                    }
                }
                continue;
            }

            __m256d vdiff = _mm256_sub_pd(vdcc, v_dfc);
            __m256d vabs = _mm256_andnot_pd(v_sign_mask, vdiff);
            __m256d vprune = _mm256_cmp_pd(vabs, v_rlim, _CMP_GT_OQ);
            int pmask = _mm256_movemask_pd(vprune);
            if (pmask != 0)
            {
                for (int sub = 0; sub < 4; sub++)
                {
                    if (pmask & (1 << sub))
                    {
                        int c = cl + sub;
                        if (state->scratch.clmembflag[c])
                        {
                            state->scratch.clmembflag[c] = 0;
                            local_pruned++;
                        }
                    }
                }
            }
        }
#endif

        for (; cl < state->num_clusters; cl++)
        {
            if (state->scratch.clmembflag[cl] == 0)
            {
                continue;
            }
            double dcc = row_dcc[cl];
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

            if (fabs(dcc - dfc) > config->algo.rlim)
            {
                state->scratch.clmembflag[cl] = 0;
                local_pruned++;
            }
        }
    }
    state->telemetry.clusters_pruned += local_pruned;

    if (state->trace)
    {
        TraceEvent *ev = trace_emit(state->trace, TRACE_PRUNE_3P);
        if (ev)
        {
            ev->pruned_count = local_pruned;
            ev->cluster_id = cj;
            int active_cnt = 0;
            for (int i = 0; i < state->num_clusters; i++)
            {
                if (state->scratch.clmembflag[i]) active_cnt++;
            }
            ev->active_remaining = active_cnt;
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
                    state->scratch.dcc_min[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                    state->scratch.dcc_min[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
                    state->scratch.dcc_max[cj * config->algo.maxnbclust + cprev] = d_ci_cprev;
                    state->scratch.dcc_max[cprev * config->algo.maxnbclust + cj] = d_ci_cprev;
                    state->scratch.dcc_measured[cj * config->algo.maxnbclust + cprev] = 1;
                    state->scratch.dcc_measured[cprev * config->algo.maxnbclust + cj] = 1;
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

#if defined(__AVX2__)
            if (!config->optim.sparse_dcc_mode)
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
                                state->scratch.dcc_min[cj * config->algo.maxnbclust + kk] = d_ci_ck;
                                state->scratch.dcc_min[kk * config->algo.maxnbclust + cj] = d_ci_ck;
                                state->scratch.dcc_max[cj * config->algo.maxnbclust + kk] = d_ci_ck;
                                state->scratch.dcc_max[kk * config->algo.maxnbclust + cj] = d_ci_ck;
                                state->scratch.dcc_measured[cj * config->algo.maxnbclust + kk] = 1;
                                state->scratch.dcc_measured[kk * config->algo.maxnbclust + cj] = 1;
                            }
                            double d_cprev_ck = row_dcc_cprev[kk];
                            if (d_cprev_ck < 0.0)
                            {
                                d_cprev_ck = get_dist(&state->clusters[cprev].anchor,
                                                      &state->clusters[kk].anchor,
                                                      -1, -1.0, -1.0, config, state);
                                state->scratch.dcc_min[cprev * config->algo.maxnbclust + kk] =
                                    d_cprev_ck;
                                state->scratch.dcc_min[kk * config->algo.maxnbclust + cprev] =
                                    d_cprev_ck;
                                state->scratch.dcc_max[cprev * config->algo.maxnbclust + kk] =
                                    d_cprev_ck;
                                state->scratch.dcc_max[kk * config->algo.maxnbclust + cprev] =
                                    d_cprev_ck;
                                 state->scratch.dcc_measured[
                                     cprev * config->algo.maxnbclust + kk] = 1;
                                 state->scratch.dcc_measured[
                                     kk * config->algo.maxnbclust + cprev] = 1;
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
        double inv_sum_p = 1.0 / sum_p;
        int i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        __m256d v_inv = _mm256_set1_pd(inv_sum_p);
        for (; i <= state->num_clusters - 4; i += 4)
        {
            __m256d vp = _mm256_loadu_pd(&state->scratch.entropy_p_current[i]);
            _mm256_storeu_pd(&state->scratch.entropy_p_current[i], _mm256_mul_pd(vp, v_inv));
        }
#endif

        for (; i < state->num_clusters; i++)
        {
            state->scratch.entropy_p_current[i] *= inv_sum_p;
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

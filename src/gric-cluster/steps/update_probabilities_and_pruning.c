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
 * select_te4_historical_anchors() - Select top informative and orthogonal anchors.
 * @cj:           Current cluster index.
 * @dfc:          Distance from current frame to cj.
 * @config:       Clustering config.
 * @state:        Clustering state.
 * @temp_indices: Array of cluster indices measured so far in this frame.
 * @temp_dists:   Array of frame-to-cluster distances.
 * @num_hist:     Number of historical anchors (temp_count - 1).
 * @max_anchors:  Maximum number of anchors to select.
 * @selected:     Output array to store selected historical indices.
 *
 * Return: Number of selected anchors written to @selected.
 */
static int select_te4_historical_anchors(
    int           cj,
    double        dfc,
    ClusterConfig *config,
    ClusterState  *state,
    const int     *temp_indices,
    const double  *temp_dists,
    int           num_hist,
    int           max_anchors,
    int          *selected)
{
    if (num_hist <= 0 || max_anchors <= 0)
    {
        return 0;
    }

    if (num_hist <= max_anchors)
    {
        int count = 0;
        for (int p = 0; p < num_hist; p++)
        {
            int cprev = temp_indices[p];
            if (config->optim.sparse_dcc_mode &&
                !state->scratch.dcc_measured[cj * config->algo.maxnbclust + cprev])
            {
                continue;
            }
            selected[count++] = p;
        }
        return count;
    }

    /* Candidate pool size: check closest anchors */
    int max_pool = max_anchors * 2;
    if (max_pool > 8)
    {
        max_pool = 8;
    }
    if (max_pool > num_hist)
    {
        max_pool = num_hist;
    }

    int pool[8];
    int pool_sz = 0;
    int maxnbc = config->algo.maxnbclust;

    for (int p = 0; p < num_hist; p++)
    {
        int cprev = temp_indices[p];
        if (config->optim.sparse_dcc_mode &&
            !state->scratch.dcc_measured[cj * maxnbc + cprev])
        {
            continue;
        }

        double d = temp_dists[p];
        if (pool_sz < max_pool)
        {
            int ins = pool_sz;
            while (ins > 0 && temp_dists[pool[ins - 1]] > d)
            {
                pool[ins] = pool[ins - 1];
                ins--;
            }
            pool[ins] = p;
            pool_sz++;
        }
        else if (d < temp_dists[pool[pool_sz - 1]])
        {
            int ins = pool_sz - 1;
            while (ins > 0 && temp_dists[pool[ins - 1]] > d)
            {
                pool[ins] = pool[ins - 1];
                ins--;
            }
            pool[ins] = p;
        }
    }

    if (pool_sz <= max_anchors)
    {
        for (int i = 0; i < pool_sz; i++)
        {
            selected[i] = pool[i];
        }
        return pool_sz;
    }

    /* 1. Pick nearest anchor (pool[0]) for maximum spatial informativeness */
    selected[0] = pool[0];
    int num_sel = 1;

    /* 2. From remaining candidates, pick the one most orthogonal to cj */
    int best_ortho_idx = -1;
    double min_cos_theta = 2.0;

    for (int i = 1; i < pool_sz; i++)
    {
        int p = pool[i];
        int cprev = temp_indices[p];
        double d_ci_cprev = state->scratch.dcc_min[cj * maxnbc + cprev];

        if (d_ci_cprev < 0.0 && !config->optim.sparse_dcc_mode)
        {
            d_ci_cprev = get_dist(
                &state->clusters[cj].anchor,
                &state->clusters[cprev].anchor,
                -1, -1.0, -1.0, config, state);
            set_dcc_pair(state, maxnbc, cj, cprev, d_ci_cprev);
        }

        if (d_ci_cprev <= 1e-9)
        {
            continue;
        }

        double cos_theta = 1.0;
        if (dfc > 1e-9 && temp_dists[p] > 1e-9)
        {
            double num = dfc * dfc + temp_dists[p] * temp_dists[p] - d_ci_cprev * d_ci_cprev;
            double denom = 2.0 * dfc * temp_dists[p];
            cos_theta = fabs(num / denom);
            if (cos_theta > 1.0)
            {
                cos_theta = 1.0;
            }
        }

        if (cos_theta < min_cos_theta)
        {
            min_cos_theta = cos_theta;
            best_ortho_idx = i;
        }
    }

    if (best_ortho_idx >= 0)
    {
        selected[num_sel++] = pool[best_ortho_idx];
    }

    /* 3. Fill remaining slots up to max_anchors with other candidates from pool */
    for (int i = 1; i < pool_sz && num_sel < max_anchors; i++)
    {
        if (i == best_ortho_idx)
        {
            continue;
        }
        selected[num_sel++] = pool[i];
    }

    return num_sel;
}

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
        int active_cnt = state->scratch.num_active_clusters;
        int *act = state->scratch.active_clusters;
        long total_pruned_te4 = 0;
        int num_hist = temp_count - 1;
        int max_te4_anchors = config->optim.te4_max_anchors;
        if (max_te4_anchors <= 0)
        {
            max_te4_anchors = num_hist;
        }

        int selected_anchors[16];
        int num_selected = select_te4_historical_anchors(
            cj, dfc, config, state, temp_indices, temp_dists,
            num_hist, max_te4_anchors, selected_anchors);

        for (int s = 0; s < num_selected && active_cnt > 0; s++)
        {
            int    p = selected_anchors[s];
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
                    d_ci_cprev = get_dist(
                        &state->clusters[cj].anchor,
                        &state->clusters[cprev].anchor,
                        -1, -1.0, -1.0, config, state);
                    set_dcc_pair(state, config->algo.maxnbclust, cj, cprev, d_ci_cprev);
                }
            }

            TE4Ref te4_ref;
            calc_te4_ref_init(&te4_ref, dfc, d_m_cprev, d_ci_cprev);

            const double *row_dcc_cj = &state->scratch.dcc_min[cj * config->algo.maxnbclust];
            const double *row_dcc_cprev =
                &state->scratch.dcc_min[cprev * config->algo.maxnbclust];
            const char *row_meas_cj =
                &state->scratch.dcc_measured[cj * config->algo.maxnbclust];
            const char *row_meas_cprev =
                &state->scratch.dcc_measured[cprev * config->algo.maxnbclust];

            int idx = 0;
            long local_pruned_te4 = 0;

            while (idx < active_cnt)
            {
                int kk = act[idx];
                if (kk == cj || kk == cprev)
                {
                    idx++;
                    continue;
                }

                double d_ci_ck = 0.0;
                double d_cprev_ck = 0.0;

                if (config->optim.sparse_dcc_mode)
                {
                    if (!row_meas_cj[kk] || !row_meas_cprev[kk])
                    {
                        idx++;
                        continue;
                    }
                    d_ci_ck = row_dcc_cj[kk];
                    d_cprev_ck = row_dcc_cprev[kk];
                }
                else
                {
                    d_ci_ck = row_dcc_cj[kk];
                    if (d_ci_ck < 0.0)
                    {
                        d_ci_ck = get_dist(
                            &state->clusters[cj].anchor,
                            &state->clusters[kk].anchor,
                            -1, -1.0, -1.0, config, state);
                        set_dcc_pair(state, config->algo.maxnbclust, cj, kk, d_ci_ck);
                    }

                    d_cprev_ck = row_dcc_cprev[kk];
                    if (d_cprev_ck < 0.0)
                    {
                        d_cprev_ck = get_dist(
                            &state->clusters[cprev].anchor,
                            &state->clusters[kk].anchor,
                            -1, -1.0, -1.0, config, state);
                        set_dcc_pair(state, config->algo.maxnbclust, cprev, kk, d_cprev_ck);
                    }
                }

                double min_d = calc_min_dist_4pt_ref(&te4_ref, d_ci_ck, d_cprev_ck);
                if (min_d > config->algo.rlim)
                {
                    state->scratch.clmembflag[kk] = 0;
                    state->scratch.entropy_p_current[kk] = 0.0;
                    local_pruned_te4++;
                    active_cnt--;
                    act[idx] = act[active_cnt];
                }
                else
                {
                    idx++;
                }
            } // while (idx < active_cnt)

            total_pruned_te4 += local_pruned_te4;
            if (state->trace)
            {
                TraceEvent *ev = trace_emit(state->trace, TRACE_PRUNE_4P);
                if (ev)
                {
                    ev->pruned_count = local_pruned_te4;
                    ev->cluster_id = cj;
                    ev->active_remaining = active_cnt;
                }
            }
        } // for (int s = 0; ...)

        state->scratch.num_active_clusters = active_cnt;
        state->telemetry.clusters_pruned += total_pruned_te4;
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

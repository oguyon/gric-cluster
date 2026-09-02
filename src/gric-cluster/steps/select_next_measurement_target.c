/**
 * @file select_next_measurement_target.c
 * @brief Entropy-based and greedy target selection
 *        for cluster measurement scheduling.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../trace/cluster_trace.h"

/*
 * Max hypotheses sampled per candidate during
 * popcount scoring (limits inner loop cost).
 */
#define ENTROPY_PRUNE_SAMPLE_LIMIT 32

/**
 * fast_log2 - Fast piecewise linear approximation of base-2 logarithm.
 * @val: Input double value.
 *
 * Employs IEEE 754 float bit extraction to compute exponent and linear mantissa fraction.
 *
 * Return: Base-2 logarithm approximation of the value.
 */
static inline __attribute__((always_inline)) double fast_log2(double val)
{
    union { double d; uint64_t i; } vx = { val };
    double exp = (double)((vx.i >> 52) & 0x7FF) - 1023.0;
    vx.i = (vx.i & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    return exp + (vx.d - 1.0);
}
/*
 * Comparators for qsort of TargetScore arrays.
 *
 * Note: inline shellsort was benchmarked as a
 * replacement to eliminate function-pointer overhead,
 * but different sort stability altered the candidate
 * set, causing chaotic degradation of the OMP parallel
 * early-exit in the eval loop (2× regression on
 * high-K patterns).  qsort's particular tie-breaking
 * order produces favorable early-exit convergence.
 */

static int compare_prob_scores(
    const void *a,
    const void *b)
{
    double sa = ((const TargetScore *)a)->score;
    double sb = ((const TargetScore *)b)->score;
    return (sa < sb) - (sa > sb);
}

static int compare_prune_scores(
    const void *a,
    const void *b)
{
    double sa = ((const TargetScore *)a)->score;
    double sb = ((const TargetScore *)b)->score;
    return (sa > sb) - (sa < sb);
}


/**
 * select_next_measurement_target_entropy - Select target based on expected Shannon entropy.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 *
 * Implements the entropy-reduction based optimization. Test hypotheses of true cluster
 * membership for each candidate target, computes hypothetical distributions after triangle
 * inequality pruning, calculates expected Shannon entropy, and returns the target minimizing it.
 *
 * Return: Selected cluster index, or -1 if no active candidates exist.
 */
/**
 * entropy_compute_initial_h() - Calculate current Shannon entropy of active cluster distribution.
 * @state:       Active ClusterState pointer.
 * @p_current:   Array of current posterior probabilities.
 * @meas_idx:    Measurement attempt depth (0 for first attempt).
 *
 * Return: Current Shannon entropy H in bits.
 */
static double entropy_compute_initial_h(
    ClusterState *state,
    const double *p_current,
    int           meas_idx)
{
    double H_current = 0.0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (p_current[i] > 1e-15)
        {
            H_current -= p_current[i] * fast_log2(p_current[i]);
        }
    }

    if (meas_idx == 0)
    {
        state->telemetry.entropy_sum_initial += H_current;
        state->telemetry.entropy_last_initial = H_current;
        if (H_current > state->telemetry.entropy_max_initial)
        {
            state->telemetry.entropy_max_initial = H_current;
        }
    }

    return H_current;
}

/**
 * entropy_check_early_gates() - Test adaptive entropy gate and leader shortcut thresholds.
 * @config:      Active ClusterConfig.
 * @state:       Active ClusterState.
 * @meas_idx:    Measurement index.
 * @H_current:   Calculated current Shannon entropy.
 * @max_p:       Maximum posterior probability among active clusters.
 * @argmax_p:    Cluster index achieving max_p.
 *
 * Return: Target cluster index if gated/short-circuited, or -1 to continue evaluation.
 */
static int entropy_check_early_gates(
    const ClusterConfig *config,
    ClusterState        *state,
    int                  meas_idx,
    double               H_current,
    double               max_p,
    int                  argmax_p)
{
    double gate_bits = (meas_idx == 0)
        ? config->optim.entropy_first_gate_bits
        : config->optim.entropy_gate_bits;

    if (H_current < gate_bits)
    {
        state->telemetry.entropy_frames_gated++;
        if (state->trace)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_ENTROPY_GATE);
            if (ev)
            {
                ev->entropy_h = H_current;
                ev->lower_bound = gate_bits;
                ev->reason = REASON_ENTROPY_GATED;
            }
        }
        return argmax_p;
    }

    if (config->optim.entropy_leader_shortcut &&
        max_p >= config->optim.entropy_leader_cutoff)
    {
        if (meas_idx == 0)
        {
            state->telemetry.entropy_frames_gated++;
        }
        if (state->trace)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
            if (ev)
            {
                ev->reason = REASON_LEADER_SHORTCUT;
                ev->cluster_id = argmax_p;
                ev->entropy_h = H_current;
            }
        }
        return argmax_p;
    }

    return -1;
}

/**
 * entropy_rank_popcount_scores() - Rank candidate targets via fast consistency mask popcount.
 * @config:           Active ClusterConfig.
 * @state:            Active ClusterState.
 * @words:            Bitmask word count.
 * @active_mask:      Bitmask of active clusters.
 * @active_indices:   Active cluster indices.
 * @active_idx_count: Number of active clusters.
 * @dynamic_min_prob: Dynamic hypothesis probability threshold.
 * @prob_scores:      Array of sorted probability scores.
 * @prob_count:       Number of probability scores.
 * @limit:            Candidate limit.
 * @prune_scores:     Output array of ranked popcount pruning scores.
 * @p_current:        Current posterior probability array.
 * @out_M:            Output number of top candidates evaluated.
 */
static void entropy_rank_popcount_scores(
    const ClusterConfig *config,
    ClusterState        *state,
    int                  words,
    const uint64_t      *active_mask,
    const int           *active_indices,
    int                  active_idx_count,
    double               dynamic_min_prob,
    const TargetScore   *prob_scores,
    int                  prob_count,
    int                  limit,
    TargetScore         *prune_scores,
    const double        *p_current,
    int                 *out_M)
{
    int N = config->algo.maxnbclust;
    int nc = state->num_clusters;

    int sampled_indices[ENTROPY_PRUNE_SAMPLE_LIMIT];
    int sampled_count = 0;
    if (active_idx_count <= ENTROPY_PRUNE_SAMPLE_LIMIT)
    {
        sampled_count = active_idx_count;
        for (int idx = 0; idx < active_idx_count; idx++)
        {
            sampled_indices[idx] = active_indices[idx];
        }
    }
    else
    {
        sampled_count = ENTROPY_PRUNE_SAMPLE_LIMIT;
        double step = (double)active_idx_count / (double)ENTROPY_PRUNE_SAMPLE_LIMIT;
        for (int idx = 0; idx < ENTROPY_PRUNE_SAMPLE_LIMIT; idx++)
        {
            sampled_indices[idx] = active_indices[(int)(idx * step)];
        }
    }

    int M = limit * 2;
    if (M > prob_count)
    {
        M = prob_count;
    }
    *out_M = M;

    uint8_t *visited = state->scratch.entropy_visited;
    memset(visited, 0, nc * sizeof(uint8_t));

    #pragma omp parallel for if(M >= 16)
    for (int idx_p = 0; idx_p < M; idx_p++)
    {
        int i = prob_scores[idx_p].id;
        prune_scores[idx_p].id = i;
        prune_scores[idx_p].score = 1e30;
        visited[i] = 1;

        if (p_current[i] >= dynamic_min_prob)
        {
            uint64_t total_pop = 0;
            uint64_t *base_mask_i = &state->scratch.consistency_mask[i * N * words];
            for (int idx = 0; idx < sampled_count; idx++)
            {
                int cj = sampled_indices[idx];
                uint64_t *mask = base_mask_i + cj * words;
                for (int w = 0; w < words; w++)
                {
                    total_pop += __builtin_popcountll(mask[w] & active_mask[w]);
                }
            }
            prune_scores[idx_p].score = (double)total_pop;
        }
    }

    qsort(prune_scores, M, sizeof(TargetScore), compare_prune_scores);

    int prune_count = M;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i] && !visited[i])
        {
            prune_scores[prune_count].id = i;
            prune_scores[prune_count].score = 1e30;
            prune_count++;
        }
    }
}

/**
 * entropy_evaluate_hypotheses() - Simulate hypothesis updates and select target minimizing entropy.
 * @config:               Active ClusterConfig.
 * @state:                Active ClusterState.
 * @words:                Bitmask word count.
 * @active_mask:          Active cluster bitmask.
 * @active_indices:       Active cluster indices.
 * @active_idx_count:     Active cluster count.
 * @candidates:           Target candidates array.
 * @num_targets:          Target count.
 * @p_current:            Current probability array.
 * @plog2p:               Precomputed p*log2(p) array.
 * @H_current:            Current entropy.
 * @expected_h_arr:       Output array for trace logging (or NULL).
 *
 * Return: Best target cluster index.
 */
static int entropy_evaluate_hypotheses(
    const ClusterConfig *config,
    ClusterState        *state,
    int                  words,
    const uint64_t      *active_mask,
    const int           *active_indices,
    int                  active_idx_count,
    const Candidate     *candidates,
    int                  num_targets,
    const double        *p_current,
    const double        *plog2p,
    double               H_current,
    double              *expected_h_arr)
{
    (void)H_current;
    int N = config->algo.maxnbclust;
    int best_target_ci = -1;
    double min_expected_entropy = 1e30;

    #pragma omp parallel for
    for (int tc_idx = 0; tc_idx < num_targets; tc_idx++)
    {
        int target_ci = candidates[tc_idx].id;
        double expected_entropy_for_ci = 0.0;
        uint64_t *base_mask_tc = &state->scratch.consistency_mask[target_ci * N * words];

        double cur_min = 1e30;
        int early_exit = 0;

        for (int h_idx = 0; h_idx < active_idx_count; h_idx++)
        {
            if ((h_idx & 15) == 0)
            {
                #pragma omp atomic read
                cur_min = min_expected_entropy;
            }
            if (expected_entropy_for_ci >= cur_min)
            {
                early_exit = 1;
                break;
            }

            int hypothesis_cj = active_indices[h_idx];
            double hypo_sum = 0.0;
            double plogp_sum = 0.0;
            uint64_t *mask = base_mask_tc + hypothesis_cj * words;

            for (int w = 0; w < words; w++)
            {
                uint64_t mask_val = mask[w] & active_mask[w];
                while (mask_val > 0)
                {
                    int bit = __builtin_ctzll(mask_val);
                    int k = w * 64 + bit;
                    hypo_sum += p_current[k];
                    plogp_sum += plog2p[k];
                    mask_val &= (mask_val - 1);
                }
            }

            double entropy = 0.0;
            if (hypo_sum > 0.0)
            {
                entropy = fast_log2(hypo_sum) - plogp_sum / hypo_sum;
            }
            expected_entropy_for_ci += p_current[hypothesis_cj] * entropy;
        } // for h_idx

        if (!early_exit)
        {
            if (expected_h_arr)
            {
                expected_h_arr[tc_idx] = expected_entropy_for_ci;
            }
            #pragma omp critical
            {
                if (expected_entropy_for_ci < min_expected_entropy)
                {
                    min_expected_entropy = expected_entropy_for_ci;
                    best_target_ci = target_ci;
                }
            }
        }
    } // for tc_idx

    return best_target_ci;
}

/**
 * select_next_measurement_target_entropy - Select target based on expected Shannon entropy.
 * @config:   Config parameters of the clustering execution.
 * @state:    Running state of the clustering execution.
 * @meas_idx: Measurement depth within the current frame (0 = first attempt).
 *
 * Implements the entropy-reduction based optimization. Test hypotheses of true cluster
 * membership for each candidate target, computes hypothetical distributions after triangle
 * inequality pruning, calculates expected Shannon entropy, and returns the target minimizing it.
 *
 * Return: Selected cluster index, or -1 if no active candidates exist.
 */
static int select_next_measurement_target_entropy(
    ClusterConfig *config,
    ClusterState  *state,
    int            meas_idx)
{
    struct timespec start_score;
    clock_gettime(CLOCK_MONOTONIC, &start_score);

    int active_count = 0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i])
        {
            active_count++;
        }
    }

    if (active_count == 0)
    {
        return -1;
    }

    if (active_count == 1)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i])
            {
                return i;
            }
        }
    }

    double *p_current = state->scratch.entropy_p_current;
    double max_p = -1.0;
    int argmax_p = -1;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (p_current[i] > max_p)
        {
            max_p = p_current[i];
            argmax_p = i;
        }
    }

    double H_current = entropy_compute_initial_h(state, p_current, meas_idx);

    int early_target = entropy_check_early_gates(
        config, state, meas_idx, H_current, max_p, argmax_p
    );
    if (early_target >= 0)
    {
        return early_target;
    }

    state->telemetry.entropy_frames_evaluated++;

    double dynamic_min_prob = max_p * 0.01;
    if (dynamic_min_prob < config->optim.entropy_min_prob)
    {
        dynamic_min_prob = config->optim.entropy_min_prob;
    }

    int N = config->algo.maxnbclust;
    int words = (N + 63) / 64;
    uint64_t active_mask[words];
    memset(active_mask, 0, words * sizeof(uint64_t));
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i])
        {
            active_mask[i / 64] |= (1ULL << (i % 64));
        }
    }

    int limit = config->optim.entropy_max_targets;
    if (limit <= 0 || limit > state->num_clusters)
    {
        limit = state->num_clusters;
    }

    int dynamic_limit = (int)(H_current * 2.0 + 1.0);
    if (dynamic_limit < 2)
    {
        dynamic_limit = 2;
    }
    if (dynamic_limit < limit)
    {
        limit = dynamic_limit;
    }

    TargetScore *prob_scores = state->scratch.entropy_prob_scores;
    TargetScore *prune_scores = state->scratch.entropy_prune_scores;
    int *active_indices = state->scratch.entropy_active_indices;

    int active_idx_count = 0;
    for (int j = 0; j < state->num_clusters; j++)
    {
        if (state->scratch.clmembflag[j])
        {
            active_indices[active_idx_count++] = j;
        }
    }

    double *plog2p = state->scratch.entropy_plog2p;
    for (int idx = 0; idx < active_idx_count; idx++)
    {
        int k = active_indices[idx];
        plog2p[k] = (p_current[k] > 1e-15) ? (p_current[k] * fast_log2(p_current[k])) : 0.0;
    }

    int prob_count = 0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i])
        {
            prob_scores[prob_count].id = i;
            prob_scores[prob_count].score = p_current[i];
            prob_count++;
        }
    }

    qsort(prob_scores, prob_count, sizeof(TargetScore), compare_prob_scores);

    int M = 0;
    entropy_rank_popcount_scores(
        config, state, words, active_mask, active_indices, active_idx_count,
        dynamic_min_prob, prob_scores, prob_count, limit, prune_scores,
        p_current, &M
    );

    struct timespec end_score, start_filter;
    clock_gettime(CLOCK_MONOTONIC, &end_score);
    state->telemetry.time_step_3b_score +=
        (end_score.tv_sec - start_score.tv_sec) * 1000.0 +
        (end_score.tv_nsec - start_score.tv_nsec) / 1000000.0;

    if (config->optim.entropy_fast_mode)
    {
        int ret = argmax_p;
        for (int idx = 0; idx < M; idx++)
        {
            if (prune_scores[idx].score < 1e30)
            {
                ret = prune_scores[idx].id;
                break;
            }
        }
        if (state->trace)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
            if (ev)
            {
                ev->reason = REASON_ENTROPY_FAST;
                ev->cluster_id = ret;
                ev->entropy_h = H_current;
            }
        }
        return ret;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_filter);

    int nc = state->num_clusters;
    uint8_t *visited = state->scratch.entropy_visited;
    memset(visited, 0, nc * sizeof(uint8_t));

    Candidate *candidates = state->scratch.entropy_candidates;
    int num_targets = 0;
    int prob_idx = 0;
    int prune_idx = 0;
    int p_limit = (limit / 2 < 1) ? 1 : (limit / 2);

    while (num_targets < p_limit && prob_idx < prob_count)
    {
        int id = prob_scores[prob_idx].id;
        if (prob_scores[prob_idx].score > 0.0 && !visited[id])
        {
            candidates[num_targets].id = id;
            candidates[num_targets].p = prob_scores[prob_idx].score;
            visited[id] = 1;
            num_targets++;
        }
        prob_idx++;
    }

    int prune_count = prob_count;
    while (num_targets < limit && prune_idx < prune_count)
    {
        int id = prune_scores[prune_idx].id;
        if (!visited[id])
        {
            candidates[num_targets].id = id;
            candidates[num_targets].p = p_current[id];
            visited[id] = 1;
            num_targets++;
        }
        prune_idx++;
    }

    struct timespec end_filter, start_eval;
    clock_gettime(CLOCK_MONOTONIC, &end_filter);
    state->telemetry.time_step_3b_filter +=
        (end_filter.tv_sec - start_filter.tv_sec) * 1000.0 +
        (end_filter.tv_nsec - start_filter.tv_nsec) / 1000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start_eval);

    int eval_hypo_count = 0;
    for (int idx = 0; idx < prob_count; idx++)
    {
        int j = prob_scores[idx].id;
        if (p_current[j] >= dynamic_min_prob)
        {
            active_indices[eval_hypo_count++] = j;
        }
    }
    active_idx_count = eval_hypo_count;

    if (num_targets > active_idx_count && active_idx_count > 0)
    {
        num_targets = active_idx_count;
    }

    double *expected_h_arr = NULL;
    if (state->trace)
    {
        expected_h_arr = (double *)malloc(num_targets * sizeof(double));
        if (expected_h_arr)
        {
            for (int i = 0; i < num_targets; i++)
            {
                expected_h_arr[i] = 1e30;
            }
        }
    }

    int best_target_ci = entropy_evaluate_hypotheses(
        config, state, words, active_mask, active_indices, active_idx_count,
        candidates, num_targets, p_current, plog2p, H_current, expected_h_arr
    );

    struct timespec end_eval;
    clock_gettime(CLOCK_MONOTONIC, &end_eval);
    state->telemetry.time_step_3b_eval +=
        (end_eval.tv_sec - start_eval.tv_sec) * 1000.0 +
        (end_eval.tv_nsec - start_eval.tv_nsec) / 1000000.0;

    if (state->trace)
    {
        TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
        if (ev)
        {
            ev->reason = REASON_ENTROPY_FULL;
            ev->cluster_id = best_target_ci;
            ev->entropy_h = H_current;

            if (expected_h_arr)
            {
                typedef struct { int id; double prob; double exp_h; } TCand;
                TCand *tc_arr = malloc(num_targets * sizeof(TCand));
                if (tc_arr)
                {
                    for (int i = 0; i < num_targets; i++)
                    {
                        tc_arr[i].id = candidates[i].id;
                        tc_arr[i].prob = candidates[i].p;
                        tc_arr[i].exp_h = expected_h_arr[i];
                    }

                    for (int i = 0; i < num_targets - 1; i++)
                    {
                        for (int j = i + 1; j < num_targets; j++)
                        {
                            if (tc_arr[i].exp_h > tc_arr[j].exp_h)
                            {
                                TCand tmp = tc_arr[i];
                                tc_arr[i] = tc_arr[j];
                                tc_arr[j] = tmp;
                            }
                        }
                    }

                    ev->num_candidates = num_targets < TRACE_MAX_CANDIDATES
                                         ? num_targets
                                         : TRACE_MAX_CANDIDATES;
                    for (int i = 0; i < ev->num_candidates; i++)
                    {
                        ev->candidates[i].id = tc_arr[i].id;
                        ev->candidates[i].prob = tc_arr[i].prob;
                        ev->candidates[i].expected_h = tc_arr[i].exp_h;
                        ev->candidates[i].info_gain = H_current - tc_arr[i].exp_h;
                    }
                    free(tc_arr);
                }
            }
        }
    }

    if (expected_h_arr)
    {
        free(expected_h_arr);
    }

    return best_target_ci;
}

/**
 * select_next_measurement_target - Select the next cluster candidate to target.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @k_search: Index tracking progression through sorted standard candidates.
 * @pred_candidates: Int array of predicted shortcut candidate indices.
 * @num_preds: Total number of prediction candidates retrieved.
 * @current_pred_idx: Progression index inside the prediction candidates array.
 * @param meas_idx: Measurement depth within the current frame (0 = first attempt).
 *
 * Chooses the next target candidate. Evaluates active trajectory prediction
 * candidates first, and standard candidates next ordered by probability.
 *
 * Return: Cluster index to measure, or -1 if all candidates are exhausted or pruned.
 */
int select_next_measurement_target(
    ClusterConfig *config,
    ClusterState  *state,
    int           *k_search,
    const int     *pred_candidates,
    int            num_preds,
    int           *current_pred_idx,
    int            meas_idx)
{
    /*
     * 1. Prioritize trajectory prediction shortcut candidates.
     * Motivation: Exploits temporal pattern correlation. If the current trajectory
     * matches historical paths, the next frame is highly likely to belong to the
     * predicted sequence, avoiding full database searches.
     */
    if (pred_candidates && *current_pred_idx < num_preds)
    {
        while (*current_pred_idx < num_preds)
        {
            int cj = pred_candidates[*current_pred_idx];
            (*current_pred_idx)++;
            if (cj >= 0 && cj < state->num_clusters && state->scratch.clmembflag[cj])
            {
                if (state->trace)
                {
                    TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
                    if (ev)
                    {
                        ev->reason = REASON_PREDICTION;
                        ev->cluster_id = cj;
                    }
                }
                return cj;
            }
        }
    }

    /*
     * If entropy-based target selection mode is enabled, evaluate expected Shannon
     * entropy for each active cluster and select the one maximizing information gain.
     */
    if (config->optim.entropy_mode)
    {
        return select_next_measurement_target_entropy(
            config, state, meas_idx);
    }

    /*
     * 2. Choose the next candidate when geometric probability mode is disabled.
     * Motivation: Standard linear search ordered statically by prior probability.
     * Since probabilities are not dynamically updated during the search process,
     * we can sequentially return candidates from the pre-sorted list (probsortedclindex)
     * and skip any that have been pruned.
     */
    if (!config->optim.gprob_mode && state->cross_tile_hook == NULL)
    {
        while (*k_search < state->num_clusters &&
               state->scratch.clmembflag[state->scratch.probsortedclindex[*k_search]] == 0)
        {
            (*k_search)++;
        }
        if (*k_search >= state->num_clusters)
        {
            return -1;
        }
        int cj = state->scratch.probsortedclindex[*k_search];
        (*k_search)++;
        
        if (state->trace)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
            if (ev)
            {
                ev->reason = REASON_GREEDY_STATIC;
                ev->cluster_id = cj;
            }
        }
        return cj;
    }
    /*
     * 3. Choose the next candidate when geometric probability mode is enabled.
     * Motivation: Dynamic search driven by ongoing co-measurement evidence.
     * As distance measurements fail, we update and refine geometric probabilities
     * (current_gprobs) of all active candidates. Thus, we must dynamically scan
     * active clusters to select the one with the highest current combined
     * probability (mixed_probs * current_gprobs).
     */
    else
    {
        double max_p = -1.0;
        int cj = -1;
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (state->scratch.clmembflag[i] &&
                state->scratch.entropy_p_current[i] > max_p)
            {
                max_p = state->scratch.entropy_p_current[i];
                cj = i;
            }
        }
        
        if (state->trace && cj != -1)
        {
            TraceEvent *ev = trace_emit(state->trace, TRACE_TARGET_SELECTED);
            if (ev)
            {
                ev->reason = REASON_GREEDY_DYNAMIC;
                ev->cluster_id = cj;
            }
        }
        return cj;
    }
}

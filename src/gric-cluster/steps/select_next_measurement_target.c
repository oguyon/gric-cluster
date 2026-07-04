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

    /*
     * Find the cluster with the highest posterior
     * probability.  This argmax is also the greedy
     * fallback target if the entropy gate decides the
     * distribution is sharp enough to skip evaluation.
     */
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

    /*
     * Proposal 1 — Adaptive entropy gating.
     *
     * Proposal 3 — Two-phase greedy→entropy per frame.
     *
     * The gate threshold is depth-dependent:
     *
     * depth 0 (first measurement in frame): use
     *   entropy_first_gate_bits (default 6.0 bits ≈
     *   64 effective candidates).  At depth 0, gprob
     *   has not yet been updated by any failed
     *   measurement, so the distribution is still
     *   dominated by the static prior.  The greedy
     *   argmax is near-optimal in this phase, and
     *   entropy evaluation is expensive waste.
     *
     * depth >= 1 (after at least one failed measure):
     *   use entropy_gate_bits (default 2.0 bits ≈ 4
     *   effective candidates).  gprob has begun
     *   narrowing the distribution; entropy is now
     *   genuinely useful when the space remains
     *   ambiguous.
     */
    double gate_bits = (meas_idx == 0)
        ? config->optim.entropy_first_gate_bits
        : config->optim.entropy_gate_bits;
    double H_current = 0.0;
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            if (p_current[i] > 1e-15)
            {
                H_current -=
                    p_current[i]
                    * fast_log2(p_current[i]);
            }
        }

        /* Feature 3: accumulate entropy telemetry */
        if (meas_idx == 0)
        {
            state->telemetry.entropy_sum_initial +=
                H_current;
            state->telemetry.entropy_last_initial =
                H_current;
            if (H_current >
                state->telemetry.entropy_max_initial)
            {
                state->telemetry.entropy_max_initial =
                    H_current;
            }
        }

        if (H_current < gate_bits)
        {
            state->telemetry.entropy_frames_gated++;
            return argmax_p;
        }
    }

    state->telemetry.entropy_frames_evaluated++;

    /*
     * Proposal 2 — Dynamic entropy_min_prob threshold.
     *
     * Raise the hypothesis skip threshold to 1% of the
     * leader's probability, floored by the static value.
     */
    double dynamic_min_prob = max_p * 0.01;
    if (dynamic_min_prob < config->optim.entropy_min_prob)
    {
        dynamic_min_prob = config->optim.entropy_min_prob;
    }

    /*
     * Optimization D — All setup below is deferred to
     * after the entropy gate check (above).  Gated frames
     * return immediately from argmax_p and never execute
     * the O(K) mask construction, scoring, or evaluation.
     */

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

    /* Scale limit dynamically based on Shannon entropy */
    int dynamic_limit = (int)(H_current * 2.0 + 1.0);
    if (dynamic_limit < 2)
    {
        dynamic_limit = 2;
    }
    if (dynamic_limit < limit)
    {
        limit = dynamic_limit;
    }

    int nc = state->num_clusters;
    TargetScore *prob_scores =
        state->scratch.entropy_prob_scores;
    TargetScore *prune_scores =
        state->scratch.entropy_prune_scores;
    int *active_indices =
        state->scratch.entropy_active_indices;

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
        if (p_current[k] > 1e-15)
        {
            plog2p[k] =
                p_current[k] * fast_log2(p_current[k]);
        }
        else
        {
            plog2p[k] = 0.0;
        }
    }

    /* Initialize prob_scores for active clusters only */
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

    /* Sort prob_scores descending */
    qsort(prob_scores, prob_count,
          sizeof(TargetScore), compare_prob_scores);

    /* Sampled list for heuristic pruning scores */
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
        double step = (double)active_idx_count
                    / (double)ENTROPY_PRUNE_SAMPLE_LIMIT;
        for (int idx = 0;
             idx < ENTROPY_PRUNE_SAMPLE_LIMIT; idx++)
        {
            sampled_indices[idx] =
                active_indices[(int)(idx * step)];
        }
    }

    /* Compute pruning scores for top M candidates */
    int M = limit * 2;
    if (M > prob_count)
    {
        M = prob_count;
    }

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
            uint64_t *base_mask_i =
                &state->scratch.consistency_mask[
                    i * N * words];
            for (int idx = 0;
                 idx < sampled_count; idx++)
            {
                int cj = sampled_indices[idx];
                uint64_t *mask =
                    base_mask_i + cj * words;
                for (int w = 0; w < words; w++)
                {
                    total_pop +=
                        __builtin_popcountll(
                            mask[w]
                            & active_mask[w]);
                }
            }
            prune_scores[idx_p].score =
                (double)total_pop;
        }
    }

    /* Sort only the M evaluated elements ascending */
    qsort(prune_scores, M,
          sizeof(TargetScore), compare_prune_scores);

    /* Append the remaining non-evaluated active clusters in index order */
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

    struct timespec end_score, start_filter;
    clock_gettime(CLOCK_MONOTONIC, &end_score);
    state->telemetry.time_step_3b_score +=
        (end_score.tv_sec - start_score.tv_sec) * 1000.0
        + (end_score.tv_nsec - start_score.tv_nsec)
            / 1000000.0;

    /*
     * Feature 2: Popcount-only surrogate mode.
     *
     * When entropy_fast_mode is enabled, skip the
     * expensive Shannon entropy evaluation and
     * return the target with the lowest popcount
     * score (most discriminative by support size
     * reduction).  The popcount score is a
     * first-order approximation of entropy.
     */
    if (config->optim.entropy_fast_mode)
    {
        for (int idx = 0; idx < M; idx++)
        {
            if (prune_scores[idx].score < 1e30)
            {
                return prune_scores[idx].id;
            }
        }
        return argmax_p;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_filter);

    Candidate *candidates =
        state->scratch.entropy_candidates;
    memset(visited, 0, nc * sizeof(uint8_t));

    int num_targets = 0;
    int prob_idx = 0;
    int prune_idx = 0;

    int p_limit = limit / 2;
    if (p_limit < 1)
    {
        p_limit = 1;
    }

    while (num_targets < p_limit &&
           prob_idx < prob_count)
    {
        int id = prob_scores[prob_idx].id;
        if (prob_scores[prob_idx].score > 0.0)
        {
            if (!visited[id])
            {
                candidates[num_targets].id = id;
                candidates[num_targets].p =
                    prob_scores[prob_idx].score;
                visited[id] = 1;
                num_targets++;
            }
        }
        prob_idx++;
    }

    while (num_targets < limit &&
           prune_idx < prune_count)
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
        (end_filter.tv_sec - start_filter.tv_sec)
            * 1000.0
        + (end_filter.tv_nsec - start_filter.tv_nsec)
            / 1000000.0;
    clock_gettime(CLOCK_MONOTONIC, &start_eval);

    /*
     * Feature 1: Rebuild active_indices in
     * descending probability order from the
     * pre-sorted prob_scores array.  This makes
     * the early-exit accumulator in the Shannon
     * eval loop grow faster, triggering the bound
     * check sooner.
     */
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

    /*
     * Option 1 — Dynamic Target Capping.
     *
     * Cap num_targets at the number of hypotheses above
     * the dynamic threshold.  If only a few clusters are
     * viable, evaluating 15 targets is wasted work.
     */
    if (num_targets > active_idx_count && active_idx_count > 0)
    {
        num_targets = active_idx_count;
    }

    int best_target_ci = -1;
    double min_expected_entropy = 1e30;

    #pragma omp parallel for
    for (int tc_idx = 0; tc_idx < num_targets; tc_idx++)
    {
        int target_ci = candidates[tc_idx].id;
        double expected_entropy_for_ci = 0.0;

        uint64_t *base_mask_tc =
            &state->scratch.consistency_mask[
                target_ci * N * words];

        double cur_min = 1e30;
        int early_exit = 0;
        for (int h_idx = 0;
             h_idx < active_idx_count; h_idx++)
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
            uint64_t *mask =
                base_mask_tc + hypothesis_cj * words;

            for (int w = 0; w < words; w++)
            {
                uint64_t mask_val =
                    mask[w] & active_mask[w];
                while (mask_val > 0)
                {
                    int bit =
                        __builtin_ctzll(mask_val);
                    int k = w * 64 + bit;
                    hypo_sum += p_current[k];
                    plogp_sum += plog2p[k];
                    mask_val &= (mask_val - 1);
                }
            }

            double entropy = 0.0;
            if (hypo_sum > 0.0)
            {
                entropy = fast_log2(hypo_sum)
                        - plogp_sum / hypo_sum;
            }
            expected_entropy_for_ci +=
                p_current[hypothesis_cj] * entropy;
        } // for h_idx

        if (!early_exit)
        {
            #pragma omp critical
            {
                if (expected_entropy_for_ci <
                    min_expected_entropy)
                {
                    min_expected_entropy =
                        expected_entropy_for_ci;
                    best_target_ci = target_ci;
                }
            }
        }
    } // for tc_idx

    struct timespec end_eval;
    clock_gettime(CLOCK_MONOTONIC, &end_eval);
    state->telemetry.time_step_3b_eval +=
        (end_eval.tv_sec - start_eval.tv_sec) * 1000.0 +
        (end_eval.tv_nsec - start_eval.tv_nsec)
            / 1000000.0;

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
    if (!config->optim.gprob_mode)
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
        return cj;
    }
}

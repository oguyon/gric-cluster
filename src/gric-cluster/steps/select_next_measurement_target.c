#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

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

typedef struct
{
    int    id;
    double score;
} TargetScore;

static int compare_prob_scores(const void *a, const void *b)
{
    double sa = ((const TargetScore *)a)->score;
    double sb = ((const TargetScore *)b)->score;
    return (sa < sb) - (sa > sb);
}

static int compare_prune_scores(const void *a, const void *b)
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
    ClusterState  *state)
{
    struct timespec start_score;
    clock_gettime(CLOCK_MONOTONIC, &start_score);

    int active_count = 0;
    double sum_probs = 0.0;

    for (int i = 0; i < state->num_clusters; i++)
    {
        if (state->scratch.clmembflag[i])
        {
            active_count++;
            sum_probs += state->scratch.mixed_probs[i] * state->scratch.current_gprobs[i];
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

    if (sum_probs > 0.0)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            p_current[i] = state->scratch.clmembflag[i] ?
                ((state->scratch.mixed_probs[i] * state->scratch.current_gprobs[i]) / sum_probs) : 0.0;
        }
    }
    else
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            p_current[i] = state->scratch.clmembflag[i] ? (1.0 / active_count) : 0.0;
        }
    }

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

    if (max_p > 0.5 && argmax_p != -1)
    {
        return argmax_p;
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

    TargetScore prob_scores[state->num_clusters];
    TargetScore prune_scores[state->num_clusters];

    int active_indices[state->num_clusters];
    int active_idx_count = 0;
    for (int j = 0; j < state->num_clusters; j++)
    {
        if (state->scratch.clmembflag[j])
        {
            active_indices[active_idx_count++] = j;
        }
    }

    double plog2p[state->num_clusters];
    for (int idx = 0; idx < active_idx_count; idx++)
    {
        int k = active_indices[idx];
        if (p_current[k] > 1e-15)
        {
            plog2p[k] = p_current[k] * fast_log2(p_current[k]);
        }
        else
        {
            plog2p[k] = 0.0;
        }
    }

    /* Initialize all scores */
    for (int i = 0; i < state->num_clusters; i++)
    {
        prob_scores[i].id = i;
        prune_scores[i].id = i;
        prune_scores[i].score = 1e30; // Default large score

        if (state->scratch.clmembflag[i])
        {
            prob_scores[i].score = state->scratch.mixed_probs[i] * state->scratch.current_gprobs[i];
        }
        else
        {
            prob_scores[i].score = 0.0;
        }
    }

    /* Sort prob_scores descending to find highest probability candidates */
    qsort(prob_scores, state->num_clusters, sizeof(TargetScore), compare_prob_scores);

    /* Pre-compile a sampled list of active clusters for fast heuristic estimation */
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

    /* Compute pruning scores only for top M candidates */
    int M = limit * 2;
    if (M > state->num_clusters)
    {
        M = state->num_clusters;
    }

    #pragma omp parallel for if(M >= OMP_MIN_CLUSTERS)
    for (int idx_p = 0; idx_p < M; idx_p++)
    {
        int i = prob_scores[idx_p].id;
        if (state->scratch.clmembflag[i])
        {
            if (p_current[i] >= config->optim.entropy_min_prob)
            {
                double total_pop = 0.0;
                uint64_t *base_mask_i = &state->scratch.consistency_mask[i * N * words];
                for (int idx = 0; idx < sampled_count; idx++)
                {
                    int cj = sampled_indices[idx];
                    uint64_t *mask = base_mask_i + cj * words;
                    for (int w = 0; w < words; w++)
                    {
                        if (active_mask[w] > 0)
                        {
                            total_pop += (double)__builtin_popcountll(mask[w] & active_mask[w]);
                        }
                    }
                }
                prune_scores[i].score = total_pop;
            }
        }
    }

    /* Sort prune_scores ascending based on pruning scores */
    qsort(prune_scores, state->num_clusters, sizeof(TargetScore), compare_prune_scores);

    struct timespec end_score, start_filter;
    clock_gettime(CLOCK_MONOTONIC, &end_score);
    state->telemetry.time_step_3b_score += (end_score.tv_sec - start_score.tv_sec) * 1000.0 +
                                           (end_score.tv_nsec - start_score.tv_nsec) / 1000000.0;
    clock_gettime(CLOCK_MONOTONIC, &start_filter);

    Candidate *candidates = state->scratch.entropy_candidates;
    uint8_t visited[state->num_clusters];
    memset(visited, 0, state->num_clusters * sizeof(uint8_t));

    int num_targets = 0;
    int prob_idx = 0;
    int prune_idx = 0;

    int p_limit = limit / 2;
    if (p_limit < 1) p_limit = 1;

    while (num_targets < p_limit && prob_idx < state->num_clusters)
    {
        int id = prob_scores[prob_idx].id;
        if (state->scratch.clmembflag[id] && prob_scores[prob_idx].score > 0.0)
        {
            if (!visited[id])
            {
                candidates[num_targets].id = id;
                candidates[num_targets].p = prob_scores[prob_idx].score;
                visited[id] = 1;
                num_targets++;
            }
        }
        prob_idx++;
    }

    while (num_targets < limit && prune_idx < state->num_clusters)
    {
        int id = prune_scores[prune_idx].id;
        if (state->scratch.clmembflag[id] && !visited[id])
        {
            double p = state->scratch.mixed_probs[id] *
                       state->scratch.current_gprobs[id];
            candidates[num_targets].id = id;
            candidates[num_targets].p = p;
            visited[id] = 1;
            num_targets++;
        }
        prune_idx++;
    }

    struct timespec end_filter, start_eval;
    clock_gettime(CLOCK_MONOTONIC, &end_filter);
    state->telemetry.time_step_3b_filter += (end_filter.tv_sec - start_filter.tv_sec) * 1000.0 +
                                            (end_filter.tv_nsec - start_filter.tv_nsec) / 1000000.0;
    clock_gettime(CLOCK_MONOTONIC, &start_eval);

    int best_target_ci = -1;
    double min_expected_entropy = 1e30;

    #pragma omp parallel for
    for (int tc_idx = 0; tc_idx < num_targets; tc_idx++)
    {
        int target_ci = candidates[tc_idx].id;
        double expected_entropy_for_ci = 0.0;

        int matched_indices[state->num_clusters];
        uint64_t *base_mask_tc = &state->scratch.consistency_mask[target_ci * N * words];

        int early_exit = 0;
        for (int h_idx = 0; h_idx < active_idx_count; h_idx++)
        {
            if (expected_entropy_for_ci >= min_expected_entropy)
            {
                early_exit = 1;
                break;
            }

            int hypothesis_cj = active_indices[h_idx];
            if (p_current[hypothesis_cj] < config->optim.entropy_min_prob)
            {
                continue;
            }

            double hypo_sum = 0.0;
            uint64_t *mask = base_mask_tc + hypothesis_cj * words;

            int word_limit = (state->num_clusters + 63) / 64;
            int matched_count = 0;
            for (int w = 0; w < word_limit; w++)
            {
                if (active_mask[w] == 0)
                {
                    continue;
                }
                uint64_t mask_val = mask[w] & active_mask[w];
                while (mask_val > 0)
                {
                    int bit = __builtin_ctzll(mask_val);
                    int k = w * 64 + bit;
                    matched_indices[matched_count++] = k;
                    hypo_sum += p_current[k];
                    mask_val &= (mask_val - 1);
                }
            }

            double entropy = 0.0;
            if (hypo_sum > 0.0)
            {
                double plogp_sum = 0.0;
                for (int m = 0; m < matched_count; m++)
                {
                    plogp_sum += plog2p[matched_indices[m]];
                }
                entropy = fast_log2(hypo_sum) - plogp_sum / hypo_sum;
            }
            expected_entropy_for_ci += p_current[hypothesis_cj] * entropy;
        }

        if (!early_exit)
        {
            #pragma omp critical
            {
                if (expected_entropy_for_ci < min_expected_entropy)
                {
                    min_expected_entropy = expected_entropy_for_ci;
                    best_target_ci = target_ci;
                }
            }
        }
    }

    struct timespec end_eval;
    clock_gettime(CLOCK_MONOTONIC, &end_eval);
    state->telemetry.time_step_3b_eval += (end_eval.tv_sec - start_eval.tv_sec) * 1000.0 +
                                          (end_eval.tv_nsec - start_eval.tv_nsec) / 1000000.0;

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
    int           *current_pred_idx)
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
        return select_next_measurement_target_entropy(config, state);
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
            if (state->scratch.clmembflag[i])
            {
                double p = state->scratch.mixed_probs[i] *
                           state->scratch.current_gprobs[i];
                if (p > max_p)
                {
                    max_p = p;
                    cj = i;
                }
            }
        }
        return cj;
    }
}

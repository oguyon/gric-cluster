#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

    for (int i = 0; i < state->num_clusters; i++)
    {
        prob_scores[i].id = i;
        if (state->scratch.clmembflag[i])
        {
            prob_scores[i].score = state->scratch.mixed_probs[i] * state->scratch.current_gprobs[i];
        }
        else
        {
            prob_scores[i].score = 0.0;
        }

        prune_scores[i].id = i;
        double total_pop = 0.0;
        for (int cj = 0; cj < state->num_clusters; cj++)
        {
            if (state->scratch.clmembflag[cj])
            {
                uint64_t *mask = &state->scratch.consistency_mask[(i * N + cj) * words];
                for (int w = 0; w < words; w++)
                {
                    total_pop += (double)__builtin_popcountll(mask[w] & active_mask[w]);
                }
            }
        }
        prune_scores[i].score = total_pop;
    }

    qsort(prob_scores, state->num_clusters, sizeof(TargetScore), compare_prob_scores);
    qsort(prune_scores, state->num_clusters, sizeof(TargetScore), compare_prune_scores);

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

    int best_target_ci = -1;
    double min_expected_entropy = 1e30;

    for (int tc_idx = 0; tc_idx < num_targets; tc_idx++)
    {
        int target_ci = candidates[tc_idx].id;
        double expected_entropy_for_ci = 0.0;

        for (int hypothesis_cj = 0; hypothesis_cj < state->num_clusters; hypothesis_cj++)
        {
            if (p_current[hypothesis_cj] < config->optim.entropy_min_prob)
            {
                continue;
            }

            double hypo_sum = 0.0;
            uint64_t *mask = &state->scratch.consistency_mask[(target_ci * N + hypothesis_cj) * words];

            int word_limit = (state->num_clusters + 63) / 64;
            for (int w = 0; w < word_limit; w++)
            {
                uint64_t mask_val = mask[w] & active_mask[w];
                while (mask_val > 0)
                {
                    int bit = __builtin_ctzll(mask_val);
                    int k = w * 64 + bit;
                    hypo_sum += p_current[k];
                    mask_val &= (mask_val - 1);
                }
            }

            double entropy = 0.0;
            if (hypo_sum > 0.0)
            {
                for (int w = 0; w < word_limit; w++)
                {
                    uint64_t mask_val = mask[w] & active_mask[w];
                    while (mask_val > 0)
                    {
                        int bit = __builtin_ctzll(mask_val);
                        int k = w * 64 + bit;
                        double p_hypo = p_current[k] / hypo_sum;
                        entropy -= p_hypo * fast_log2(p_hypo);
                        mask_val &= (mask_val - 1);
                    }
                }
            }
            expected_entropy_for_ci += p_current[hypothesis_cj] * entropy;
        }

        if (expected_entropy_for_ci < min_expected_entropy)
        {
            min_expected_entropy = expected_entropy_for_ci;
            best_target_ci = target_ci;
        }
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

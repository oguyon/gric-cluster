/**
 * @file compute_priors_and_mixing.c
 * @brief Prior probability computation with geometric,
 *        transition-matrix, and prediction mixing.
 */
#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_math.h"
#include "framedistance.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../trace/cluster_trace.h"

#include "gric_simd.h"

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#include <immintrin.h>

GRIC_TARGET_AVX512
/**
 * cluster_normalize_probs_avx512() - AVX-512 SIMD vector normalization of cluster probabilities.
 * @probs: Array of unnormalized probabilities [num_clusters].
 * @n:     Number of cluster entries.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked by cluster_normalize_probs() on AVX-512 hardware to reduce probability sums and
 * multiply by the reciprocal sum using 512-bit vector registers, ensuring sum(probs) == 1.0.
 */
static void cluster_normalize_probs_avx512(
    double *restrict probs,
    int              num_clusters)
{
    __m512d acc = _mm512_setzero_pd();
    int i = 0;
    for (; i <= num_clusters - 8; i += 8)
    {
        acc = _mm512_add_pd(acc, _mm512_loadu_pd(probs + i));
    }
    double sum = _mm512_reduce_add_pd(acc);
    for (; i < num_clusters; i++)
    {
        sum += probs[i];
    }

    if (sum <= 0.0)
    {
        double flat_p = 1.0 / (double)num_clusters;
        for (int j = 0; j < num_clusters; j++)
        {
            probs[j] = flat_p;
        }
        return;
    }

    double inv = 1.0 / sum;
    __m512d vinv = _mm512_set1_pd(inv);
    i = 0;
    for (; i <= num_clusters - 8; i += 8)
    {
        __m512d p = _mm512_loadu_pd(probs + i);
        _mm512_storeu_pd(probs + i, _mm512_mul_pd(p, vinv));
    }
    for (; i < num_clusters; i++)
    {
        probs[i] *= inv;
    }
}

GRIC_TARGET_AVX2
/**
 * cluster_normalize_probs_avx2() - AVX2 256-bit SIMD vector normalization of cluster probabilities.
 * @probs: Array of unnormalized probabilities [num_clusters].
 * @n:     Number of cluster entries.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked by cluster_normalize_probs() on AVX2 hardware to normalize candidate cluster
 * prior probabilities to sum to 1.0 using 256-bit FMA vector registers.
 */
static void cluster_normalize_probs_avx2(
    double *restrict probs,
    int              num_clusters)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= num_clusters - 8; i += 8)
    {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(probs + i));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(probs + i + 4));
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    double sum = _mm_cvtsd_f64(s);
    for (; i < num_clusters; i++)
    {
        sum += probs[i];
    }

    if (sum <= 0.0)
    {
        double flat_p = 1.0 / (double)num_clusters;
        for (int j = 0; j < num_clusters; j++)
        {
            probs[j] = flat_p;
        }
        return;
    }

    double inv = 1.0 / sum;
    __m256d vinv = _mm256_set1_pd(inv);
    i = 0;
    for (; i <= num_clusters - 8; i += 8)
    {
        __m256d p0 = _mm256_loadu_pd(probs + i);
        __m256d p1 = _mm256_loadu_pd(probs + i + 4);
        _mm256_storeu_pd(probs + i, _mm256_mul_pd(p0, vinv));
        _mm256_storeu_pd(probs + i + 4, _mm256_mul_pd(p1, vinv));
    }
    for (; i < num_clusters; i++)
    {
        probs[i] *= inv;
    }
}

GRIC_TARGET_AVX2
/**
 * reset_search_scratch_avx2() - Reset frame candidate scratch buffers using AVX2.
 * @state:  Active clustering state.
 * @config: Active clustering configuration.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked at the beginning of each frame step to re-initialize candidate evaluation bitmasks,
 * distance cutoff arrays, and priority scratch arrays before candidate filtering.
 */
static void reset_search_scratch_avx2(
    ClusterState *state,
    int           num_cl)
{
    __m256d ones_d = _mm256_set1_pd(1.0);
    int i = 0;
    for (; i <= num_cl - 4; i += 4)
    {
        _mm256_storeu_pd(state->scratch.current_gprobs + i, ones_d);
    }
    for (; i < num_cl; i++)
    {
        state->scratch.current_gprobs[i] = 1.0;
    }

    __m256i ones_i = _mm256_set1_epi32(1);
    __m256i ramp = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i step8 = _mm256_set1_epi32(8);
    i = 0;
    for (; i <= num_cl - 8; i += 8)
    {
        _mm256_storeu_si256((__m256i *)(state->scratch.clmembflag + i), ones_i);
        _mm256_storeu_si256((__m256i *)(state->scratch.active_clusters + i), ramp);
        ramp = _mm256_add_epi32(ramp, step8);
    }
    for (; i < num_cl; i++)
    {
        state->scratch.clmembflag[i] = 1;
        state->scratch.active_clusters[i] = i;
    }
}
#endif

/**
 * reset_search_scratch_scalar() - Scalar fallback resetting frame candidate scratch buffers.
 * @state:  Active clustering state.
 * @config: Active clustering configuration.
 *
 * Purpose & Context ("What is this used for?"):
 * Portable fallback for reset_search_scratch() when SIMD is unavailable.
 */
static void reset_search_scratch_scalar(
    ClusterState *state,
    int           num_cl)
{
    for (int i = 0; i < num_cl; i++)
    {
        state->scratch.current_gprobs[i] = 1.0;
        state->scratch.clmembflag[i] = 1;
        state->scratch.active_clusters[i] = i;
    }
}

/**
 * reset_search_scratch() - Hardware-dispatched reset of frame candidate scratch buffers.
 * @state:  Active clustering state holding scratch arrays.
 * @num_cl: Number of active clusters to initialize.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked at the start of cluster_frame() to clear and prepare per-frame candidate scratch
 * memory before evaluating incoming vectors. Dispatches to AVX2 or scalar reset.
 */
static inline void reset_search_scratch(
    ClusterState *state,
    int           num_cl)
{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        reset_search_scratch_avx2(state, num_cl);
        return;
    }
#endif
    reset_search_scratch_scalar(state, num_cl);
}

/**
 * cluster_normalize_probs_scalar() - Portable scalar normalization of cluster probabilities.
 * @probs: Array of unnormalized probabilities [num_clusters].
 * @n:     Number of cluster entries.
 *
 * Purpose & Context ("What is this used for?"):
 * Portable fallback for cluster_normalize_probs() when vector hardware is unavailable.
 */
static void cluster_normalize_probs_scalar(
    double *restrict probs,
    int              num_clusters)
{
    double sum = 0.0;
    for (int i = 0; i < num_clusters; i++)
    {
        sum += probs[i];
    }
    if (sum <= 0.0)
    {
        double flat_p = 1.0 / (double)num_clusters;
        for (int i = 0; i < num_clusters; i++)
        {
            probs[i] = flat_p;
        }
        return;
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < num_clusters; i++)
    {
        probs[i] *= inv;
    }
}

/**
 * cluster_normalize_probs() - Fast SIMD normalization of cluster prior probabilities.
 * @probs:        Contiguous array of cluster prior probabilities.
 * @num_clusters: Number of active clusters.
 */
void cluster_normalize_probs(
    double *restrict probs,
    int              num_clusters)
{
    if (probs == NULL || num_clusters <= 0)
    {
        return;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
    GricSimdLevel simd = gric_get_simd_level();
    if (simd >= GRIC_SIMD_AVX512)
    {
        cluster_normalize_probs_avx512(probs, num_clusters);
        return;
    }
    if (simd >= GRIC_SIMD_AVX2)
    {
        cluster_normalize_probs_avx2(probs, num_clusters);
        return;
    }
#endif

    cluster_normalize_probs_scalar(probs, num_clusters);
}

/**
 * calculate_sequence_match_metric - Compute match metric mAB between two cluster histories.
 * @seq_A_cl: Array of cluster assignments for sequence A.
 * @seq_A_d: Array of anchor distances for sequence A.
 * @seq_B_cl: Array of cluster assignments for sequence B.
 * @seq_B_d: Array of anchor distances for sequence B.
 * @n_p: Sequence length (pattern length).
 * @r_c: Cluster radius threshold.
 * @state: Running state of the clustering execution.
 * @config: Config parameters of the clustering execution.
 *
 * Implements the sequence matching formula comparing historical transitions to predict priors.
 *
 * Return: Double value representing the match probability metric mAB.
 */
static double calculate_sequence_match_metric(
    const int     *seq_A_cl,
    const double  *seq_A_d,
    const int     *seq_B_cl,
    const double  *seq_B_d,
    int            n_p,
    double         r_c,
    ClusterState  *state,
    ClusterConfig *config)
{
    double sum_dmax = 0.0;
    double a_param = 1.0 - 1.0 / n_p;
    double a_weight = 1.0;
    double inv_rc = 1.0 / r_c;

    for (int j = 0; j < n_p; j++)
    {
        int clA = seq_A_cl[j];
        int clB = seq_B_cl[j];
        double dA = seq_A_d[j];
        double dB = seq_B_d[j];

        if (clA < 0 || clA >= state->num_clusters ||
            clB < 0 || clB >= state->num_clusters)
        {
            a_weight *= a_param;
            continue;
        }

        int idx = clA * config->algo.maxnbclust + clB;
        double dcc = 0.0;
        if (config->optim.sparse_dcc_mode)
        {
            if (state->scratch.dcc_measured[idx])
            {
                dcc = state->scratch.dcc_min[idx];
            }
            else
            {
                dcc = state->scratch.dcc_max[idx];
            }
        }
        else
        {
            dcc = state->scratch.dcc_min[idx];
            if (dcc < 0.0)
            {
                dcc = framedist(&state->clusters[clA].anchor, &state->clusters[clB].anchor);
                state->scratch.dcc_min[idx] = dcc;
                state->scratch.dcc_min[clB * config->algo.maxnbclust + clA] = dcc;
                state->scratch.dcc_max[idx] = dcc;
                state->scratch.dcc_max[clB * config->algo.maxnbclust + clA] = dcc;
                state->scratch.dcc_measured[idx] = 1;
                state->scratch.dcc_measured[clB * config->algo.maxnbclust + clA] = 1;
            }
        }

        double dmax = dA + dB + dcc;
        if (dmax > r_c)
        {
            return 0.0;
        }
        sum_dmax += a_weight * (dmax * inv_rc);
        a_weight *= a_param;
    }

    /* 1 - 1/e : exponential CDF complement at the mean */
    return exp(-sum_dmax / ((double)n_p * 0.63212055882855767));
}


/**
 * candidate_sort_descending() - Sort candidate cluster indices by probability descending.
 * @indices: In-out array of candidate cluster indices [n].
 * @probs:   Array of matching probabilities for each cluster.
 * @left:    Left recursion index.
 * @right:   Right recursion index.
 *
 * Purpose & Context ("What is this used for?"):
 * Invoked in compute_priors_and_mixing() to sort candidate clusters so that highest-probability
 * clusters are measured first in greedy metric search, maximizing early-cutoff pruning efficiency.
 */
static void candidate_sort_descending(
    Candidate    *cands,
    const double *mixed_probs,
    int          *sorted_indices,
    int           num_cl)
{
    if (num_cl <= 1)
    {
        if (num_cl == 1)
        {
            sorted_indices[0] = 0;
        }
        return;
    }

    for (int i = 0; i < num_cl; i++)
    {
        cands[i].id = i;
        cands[i].p = mixed_probs[i];
    }

    qsort(cands, (size_t)num_cl, sizeof(Candidate), compare_candidates);

    for (int i = 0; i < num_cl; i++)
    {
        sorted_indices[i] = cands[i].id;
    }
}

/**
 * compute_priors_and_mixing - Compute mixed priors using frequency and sequence transitions.
 * @config: Config parameters of the clustering execution.
 * @state: Running state of the clustering execution.
 * @prev_assigned_cluster: The cluster index assigned to the previous frame (-1 if none).
 * @sorting_candidates: Scratch memory array used to sort candidates.
 *
 * Normalizes prior probabilities so they sum to 1. If sequence prediction or transition matrix
 * mixing is enabled, blends normalized priors with temporal transition statistics.
 */
void compute_priors_and_mixing(
    ClusterConfig *config,
    ClusterState  *state,
    int            prev_assigned_cluster,
    Candidate     *sorting_candidates)
{
    int num_cl = state->num_clusters;
    state->scratch.num_active_clusters = num_cl;

    int tm_active = (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1);
    if (config->optim.pred_mode != 2 && !tm_active)
    {
        if (state->scratch.cluster_probs != NULL)
        {
            cluster_normalize_probs(state->scratch.cluster_probs, num_cl);
            memcpy(state->scratch.mixed_probs, state->scratch.cluster_probs,
                   (size_t)num_cl * sizeof(double));
            if (config->optim.entropy_mode || config->optim.gprob_mode ||
                state->cross_tile_hook != NULL)
            {
                memcpy(state->scratch.entropy_p_current, state->scratch.cluster_probs,
                       (size_t)num_cl * sizeof(double));
            }
        }
        else
        {
            double sum_prob = 0.0;
            for (int i = 0; i < num_cl; i++)
            {
                sum_prob += state->clusters[i].prob;
            }
            double inv_sum = (sum_prob > 0.0) ? (1.0 / sum_prob) : 1.0;
            for (int i = 0; i < num_cl; i++)
            {
                double prior = (sum_prob > 0.0) ? (state->clusters[i].prob * inv_sum)
                                                : (1.0 / (double)num_cl);
                state->clusters[i].prob = prior;
                state->scratch.mixed_probs[i] = prior;
                state->scratch.entropy_p_current[i] = prior;
            }
        }
        reset_search_scratch(state, num_cl);
    }
    else
    {
        if (state->scratch.cluster_probs != NULL)
        {
            cluster_normalize_probs(state->scratch.cluster_probs, num_cl);
        }
        else
        {
            double sum_prob = 0.0;
            for (int i = 0; i < num_cl; i++)
            {
                sum_prob += state->clusters[i].prob;
            }
            if (sum_prob > 0.0)
            {
                double inv_sum = 1.0 / sum_prob;
                for (int i = 0; i < num_cl; i++)
                {
                    state->clusters[i].prob *= inv_sum;
                }
            }
        }
        reset_search_scratch(state, num_cl);
    }

    if (config->optim.pred_mode == 2)
    {
        int np = config->optim.pred_len;
        int nl = config->optim.pred_h;
        long t = state->telemetry.total_frames_processed;
        int K = state->num_clusters;
        double rc = config->algo.rlim;

#define MAX_STACK_K 1024
#define MAX_STACK_NP 64

        double p_seq_stack[MAX_STACK_K];
        double *p_seq = (K <= MAX_STACK_K)
            ? p_seq_stack
            : (double *)malloc(K * sizeof(double));
        if (!p_seq)
        {
            for (int i = 0; i < K; i++)
            {
                state->scratch.mixed_probs[i] = 1.0 / K;
            }
        }
        else
        {
            for (int i = 0; i < K; i++)
            {
                p_seq[i] = 1.0 / K;
            }
            if (t >= np)
            {
                double match_scores_stack[MAX_STACK_K];
                double *match_scores = (K <= MAX_STACK_K)
                    ? match_scores_stack
                    : (double *)calloc(K, sizeof(double));
                if (match_scores)
                {
                    if (K <= MAX_STACK_K)
                    {
                        memset(match_scores, 0, K * sizeof(double));
                    }
                    int seq_A_cl_stack[MAX_STACK_NP];
                    double seq_A_d_stack[MAX_STACK_NP];
                    int *seq_A_cl = (np <= MAX_STACK_NP)
                        ? seq_A_cl_stack
                        : (int *)malloc(np * sizeof(int));
                    double *seq_A_d = (np <= MAX_STACK_NP)
                        ? seq_A_d_stack
                        : (double *)malloc(np * sizeof(double));
                    if (seq_A_cl && seq_A_d)
                    {
                        for (int j = 0; j < np; j++)
                        {
                            long idxA = t - 1 - j;
                            seq_A_cl[j] = state->assignments[idxA];
                            seq_A_d[j] = (state->assignment_dists != NULL)
                                ? state->assignment_dists[idxA]
                                : state->frame_infos[idxA].assigned_dist;
                        }

                        long start_s = t - nl;
                        if (start_s < np)
                        {
                            start_s = np;
                        }

                        int seq_B_cl_stack[MAX_STACK_NP];
                        double seq_B_d_stack[MAX_STACK_NP];
                        int *seq_B_cl = (np <= MAX_STACK_NP)
                            ? seq_B_cl_stack
                            : (int *)malloc(np * sizeof(int));
                        double *seq_B_d = (np <= MAX_STACK_NP)
                            ? seq_B_d_stack
                            : (double *)malloc(np * sizeof(double));
                        if (seq_B_cl && seq_B_d)
                        {
                            for (long s = start_s; s < t; s++)
                            {
                                int target_cl = state->assignments[s];
                                if (target_cl < 0 || target_cl >= K)
                                {
                                    continue;
                                }

                                for (int j = 0; j < np; j++)
                                {
                                    long idxB = s - 1 - j;
                                    seq_B_cl[j] = state->assignments[idxB];
                                    seq_B_d[j] = (state->assignment_dists != NULL)
                                        ? state->assignment_dists[idxB]
                                        : state->frame_infos[idxB].assigned_dist;
                                }

                                double mAB = calculate_sequence_match_metric(
                                    seq_A_cl, seq_A_d, seq_B_cl, seq_B_d,
                                    np, rc, state, config);
                                match_scores[target_cl] += mAB;
                            }
                            if (seq_B_cl != seq_B_cl_stack)
                            {
                                free(seq_B_cl);
                            }
                            if (seq_B_d != seq_B_d_stack)
                            {
                                free(seq_B_d);
                            }
                        }
                        if (seq_A_cl != seq_A_cl_stack)
                        {
                            free(seq_A_cl);
                        }
                        if (seq_A_d != seq_A_d_stack)
                        {
                            free(seq_A_d);
                        }
                    }

                    double total_score = 0.0;
                    for (int i = 0; i < K; i++)
                    {
                        total_score += match_scores[i];
                    }
                    if (total_score > 0.0)
                    {
                        for (int i = 0; i < K; i++)
                        {
                            p_seq[i] = match_scores[i] / total_score;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < K; i++)
                        {
                            p_seq[i] = 1.0 / K;
                        }
                    }
                    if (match_scores != match_scores_stack)
                    {
                        free(match_scores);
                    }
                }
            }

            double sum_freq = 0.0;
            for (int i = 0; i < K; i++)
            {
                sum_freq += state->clusters[i].prob;
            }
            if (sum_freq <= 0.0)
            {
                sum_freq = 1.0;
            }

            double sum_final = 0.0;
            for (int i = 0; i < K; i++)
            {
                double p_freq = state->clusters[i].prob / sum_freq;
                state->scratch.mixed_probs[i] = p_freq * p_seq[i];
                sum_final += state->scratch.mixed_probs[i];
            }

            if (sum_final > 0.0)
            {
                for (int i = 0; i < K; i++)
                {
                    state->scratch.mixed_probs[i] /= sum_final;
                }
            }
            else
            {
                for (int i = 0; i < K; i++)
                {
                    state->scratch.mixed_probs[i] = 1.0 / K;
                }
            }
            if (p_seq != p_seq_stack)
            {
                free(p_seq);
            }
        }
    }
    else if (tm_active)
    {
        double trans_prob_sum = 0.0;
        for (int i = 0; i < state->num_clusters; i++)
        {
            trans_prob_sum += (double)state->transition_matrix[
                prev_assigned_cluster * config->algo.maxnbclust + i];
        }

        for (int i = 0; i < state->num_clusters; i++)
        {
            double prior = state->clusters[i].prob;
            double tp = 0.0;
            if (trans_prob_sum > 0.0)
            {
                tp = (double)state->transition_matrix[
                    prev_assigned_cluster * config->algo.maxnbclust + i] / trans_prob_sum;
                state->scratch.mixed_probs[i] =
                    (1.0 - config->algo.tm_mixing_coeff) * prior +
                    config->algo.tm_mixing_coeff * tp;
            }
            else
            {
                state->scratch.mixed_probs[i] = prior;
            }
            state->scratch.entropy_p_current[i] = state->scratch.mixed_probs[i];
        }
    }

    if (!config->optim.gprob_mode)
    {
        candidate_sort_descending(
            sorting_candidates,
            state->scratch.mixed_probs,
            state->scratch.probsortedclindex,
            state->num_clusters);
    }

    /*
     * Initialize the posterior distribution for
     * entropy-based target selection.  This becomes
     * the starting point that the entropy search
     * progressively refines as measurements are taken.
     */
    if (config->optim.pred_mode == 2)
    {
        memcpy(state->scratch.entropy_p_current, state->scratch.mixed_probs,
               (size_t)state->num_clusters * sizeof(double));
    }

    if (state->trace)
    {
        TraceEvent *ev = trace_emit(state->trace, TRACE_PRIOR_MIXING);
        if (ev)
        {
            Candidate *trace_cands = (Candidate *)malloc(state->num_clusters * sizeof(Candidate));
            if (trace_cands)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    trace_cands[i].id = i;
                    trace_cands[i].p = state->scratch.mixed_probs[i];
                }
                qsort(trace_cands, state->num_clusters, sizeof(Candidate), compare_candidates);
                
                ev->num_candidates = state->num_clusters < TRACE_MAX_CANDIDATES
                                     ? state->num_clusters
                                     : TRACE_MAX_CANDIDATES;
                for (int i = 0; i < ev->num_candidates; i++)
                {
                    ev->candidates[i].id = trace_cands[i].id;
                    ev->candidates[i].prob = trace_cands[i].p;
                    ev->candidates[i].expected_h = 0.0;
                    ev->candidates[i].info_gain = 0.0;
                }
                free(trace_cands);
            }
        }
    }
}

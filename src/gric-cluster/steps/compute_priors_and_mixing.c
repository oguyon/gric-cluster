#define _POSIX_C_SOURCE 200809L
#include "cluster_steps.h"
#include "cluster_math.h"
#include "framedistance.h"
#include <math.h>
#include <stdlib.h>

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

    for (int j = 0; j < n_p; j++)
    {
        int clA = seq_A_cl[j];
        int clB = seq_B_cl[j];
        double dA = seq_A_d[j];
        double dB = seq_B_d[j];

        if (clA < 0 || clA >= state->num_clusters ||
            clB < 0 || clB >= state->num_clusters)
        {
            continue;
        }

        double dcc = 0.0;
        if (config->optim.sparse_dcc_mode)
        {
            dcc = state->scratch.dcc_min[clA * config->algo.maxnbclust + clB];
        }
        else
        {
            dcc = state->scratch.dcc_min[clA * config->algo.maxnbclust + clB];
            if (dcc < 0.0)
            {
                dcc = framedist(&state->clusters[clA].anchor, &state->clusters[clB].anchor);
                state->scratch.dcc_min[clA * config->algo.maxnbclust + clB] = dcc;
                state->scratch.dcc_min[clB * config->algo.maxnbclust + clA] = dcc;
                state->scratch.dcc_max[clA * config->algo.maxnbclust + clB] = dcc;
                state->scratch.dcc_max[clB * config->algo.maxnbclust + clA] = dcc;
                state->scratch.dcc_measured[clA * config->algo.maxnbclust + clB] = 1;
                state->scratch.dcc_measured[clB * config->algo.maxnbclust + clA] = 1;
            }
        }

        double dmax = dA + dB + dcc;
        sum_dmax += pow(a_param, j) * (dmax / r_c);
    }

    return exp(-sum_dmax / ((double)n_p * 0.63212055882855767));
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
    double sum_prob = 0.0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        sum_prob += state->clusters[i].prob;
    }
    if (sum_prob > 0)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].prob /= sum_prob;
        }
    }

    for (int i = 0; i < state->num_clusters; i++)
    {
        state->scratch.current_gprobs[i] = 1.0;
    }
    for (int i = 0; i < state->num_clusters; i++)
    {
        state->scratch.clmembflag[i] = 1;
    }

    if (config->optim.pred_mode)
    {
        int np = config->optim.pred_len;
        int nl = config->optim.pred_h;
        long t = state->telemetry.total_frames_processed;
        int K = state->num_clusters;
        double rc = config->algo.rlim;

        double *p_seq = (double *)malloc(K * sizeof(double));
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
                double *match_scores = (double *)calloc(K, sizeof(double));
                if (match_scores)
                {
                    int *seq_A_cl = (int *)malloc(np * sizeof(int));
                    double *seq_A_d = (double *)malloc(np * sizeof(double));
                    if (seq_A_cl && seq_A_d)
                    {
                        for (int j = 0; j < np; j++)
                        {
                            long idxA = t - 1 - j;
                            seq_A_cl[j] = state->assignments[idxA];
                            double d = -1.0;
                            for (int d_idx = 0;
                                 d_idx < state->frame_infos[idxA].num_dists;
                                 d_idx++)
                            {
                                if (state->frame_infos[idxA].cluster_indices[d_idx] ==
                                    seq_A_cl[j])
                                {
                                    d = state->frame_infos[idxA].distances[d_idx];
                                    break;
                                }
                            }
                            seq_A_d[j] = (d >= 0.0) ? d : 0.0;
                        }

                        long start_s = t - nl;
                        if (start_s < np)
                        {
                            start_s = np;
                        }

                        int *seq_B_cl = (int *)malloc(np * sizeof(int));
                        double *seq_B_d = (double *)malloc(np * sizeof(double));
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
                                    double d = -1.0;
                                    for (int d_idx = 0;
                                         d_idx < state->frame_infos[idxB].num_dists;
                                         d_idx++)
                                    {
                                        if (state->frame_infos[idxB].cluster_indices[d_idx] ==
                                            seq_B_cl[j])
                                        {
                                            d = state->frame_infos[idxB].distances[d_idx];
                                            break;
                                        }
                                    }
                                    seq_B_d[j] = (d >= 0.0) ? d : 0.0;
                                }

                                double mAB = calculate_sequence_match_metric(
                                    seq_A_cl, seq_A_d, seq_B_cl, seq_B_d,
                                    np, rc, state, config);
                                match_scores[target_cl] += mAB;
                            }
                            free(seq_B_cl);
                            free(seq_B_d);
                        }
                        free(seq_A_cl);
                        free(seq_A_d);
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
                    free(match_scores);
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
            free(p_seq);
        }
    }
    else
    {
        double trans_prob_sum = 0.0;
        if (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                trans_prob_sum += (double)state->transition_matrix[
                    prev_assigned_cluster * config->algo.maxnbclust + i];
            }
        }

        for (int i = 0; i < state->num_clusters; i++)
        {
            double prior = state->clusters[i].prob;
            double tp = 0.0;
            if (config->algo.tm_mixing_coeff > 0.0 && prev_assigned_cluster != -1 &&
                trans_prob_sum > 0.0)
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
        }
    }

    if (!config->optim.gprob_mode)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            sorting_candidates[i].id = i;
            sorting_candidates[i].p = state->scratch.mixed_probs[i];
        }
        qsort(sorting_candidates, state->num_clusters, sizeof(Candidate),
              compare_candidates);
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->scratch.probsortedclindex[i] = sorting_candidates[i].id;
        }
    }
}

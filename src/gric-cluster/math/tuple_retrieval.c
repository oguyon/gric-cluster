/**
 * @file tuple_retrieval.c
 * @brief Unified tuple-list retrieval for cross-tile
 *        information sharing in multi-tile GRIC.
 *
 * Replaces dense co-occurrence and transition matrices
 * with a scan over observed assignment tuples, using
 * an exponential match kernel.
 */

#include "tuple_retrieval.h"

#include "cluster_math.h"
#include "framedistance.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * soft_match_weight() - Compute soft match weight
 *     between two cluster indices within one tile.
 * @cluster_a: First cluster index.
 * @cluster_b: Second cluster index.
 *
 * Returns 1.0 for exact match, 0.0 otherwise.
 * A future enhancement will use inter-cluster DCC
 * distances for soft matching; Phase 6 uses exact
 * matching for simplicity.
 *
 * Return: Match weight in [0.0, 1.0].
 */
static double soft_match_weight(
    int cluster_a,
    int cluster_b)
{
    return (cluster_a == cluster_b) ? 1.0 : 0.0;
}

/**
 * tuple_retrieve() - Scan tuple history for patterns
 *     matching a spatial+temporal key and accumulate
 *     per-cluster match scores for a target tile.
 * @mts:           Multi-tile state with tuple history.
 * @target_tile:   Index of the tile to retrieve for.
 * @spatial_key:   Array[M] of current cluster IDs for
 *                 spatial context tiles.
 * @spatial_mask:  Array[M]; 1 if tile participates in
 *                 spatial matching, 0 otherwise.
 * @temporal_key:  Array[M] of previous-frame cluster
 *                 IDs for temporal context, or NULL
 *                 to disable temporal matching.
 *                 Entries < 0 are ignored.
 * @match_scores:  Output array[max_clusters] receiving
 *                 normalised match scores.
 * @max_clusters:  Length of match_scores array.
 *
 * The function zero-initialises match_scores, scans
 * the most recent retrieval_window tuples, and
 * normalises the accumulated weights to sum to 1.
 */
void tuple_retrieve(
    const MultiTileState *mts,
    int                   target_tile,
    const int            *spatial_key,
    const int            *spatial_mask,
    const int            *temporal_key,
    double               *match_scores,
    int                   max_clusters)
{
    int  M = mts->num_tiles;
    int  W = mts->retrieval_window;
    long T = mts->tuple_count;

    long start = (T - W > 0) ? (T - W) : 0;

    memset(match_scores, 0,
        (size_t) max_clusters * sizeof(double));

    if (T == 0)
    {
        return;
    }

    /* Scan historical tuples in [start, T) */
    for (long s = start; s < T; s++)
    {
        const int *h_curr =
            &mts->tuple_history[s * M];

        double w = 1.0;

        /* Spatial match across context tiles */
        for (int m = 0; m < M; m++)
        {
            if (spatial_mask[m] == 0)
            {
                continue;
            }
            if (m == target_tile)
            {
                continue;
            }
            w *= soft_match_weight(
                spatial_key[m], h_curr[m]);
            if (w == 0.0)
            {
                break;
            }
        } // for m (spatial)

        if (w == 0.0)
        {
            continue;
        }

        /* Temporal match against previous tuple */
        if (s > 0 && temporal_key != NULL)
        {
            const int *h_prev =
                &mts->tuple_history[(s - 1) * M];

            for (int m = 0; m < M; m++)
            {
                if (temporal_key[m] < 0)
                {
                    continue;
                }
                w *= soft_match_weight(
                    temporal_key[m], h_prev[m]);
                if (w == 0.0)
                {
                    break;
                }
            } // for m (temporal)
        } // if temporal

        if (w > 0.0)
        {
            int cl = h_curr[target_tile];
            if (cl >= 0 && cl < max_clusters)
            {
                match_scores[cl] += w;
            }
        }
    } // for s (tuple scan)

    /* Normalise match_scores to sum to 1 with Laplace smoothing */
    {
        double sum = 0.0;
        double alpha = 0.01; /* Small pseudocount to prevent zero probability */
        for (int k = 0; k < max_clusters; k++)
        {
            sum += match_scores[k];
        }
        if (sum > 0.0)
        {
            double total = sum + (double)max_clusters * alpha;
            for (int k = 0; k < max_clusters; k++)
            {
                match_scores[k] = (match_scores[k] + alpha) / total;
            }
        }
        else
        {
            for (int k = 0; k < max_clusters; k++)
            {
                match_scores[k] = 1.0; /* Flat scores if no match in history */
            }
        }
    } // normalise block
}

/**
 * pass2_fuse() - Run Pass 2 Bayesian fusion for a
 *     single tile.
 * @mts:       Multi-tile state.
 * @tile_idx:  Index of the tile to fuse.
 *
 * Builds spatial and temporal keys from the current
 * pass1 assignments and the most recent tuple, calls
 * tuple_retrieve() to get per-cluster match scores,
 * then multiplies each pass1_posterior element by the
 * corresponding match score.  The assignment is updated
 * to the argmax of the fused posterior.
 */
void pass2_fuse(
    MultiTileState *mts,
    int             tile_idx)
{
    int M = mts->num_tiles;
    TileState *ts = &mts->tile_states[tile_idx];
    int maxcl = ts->config.algo.maxnbclust;

    if (ts->pass1_assignment >= ts->pass1_old_ncl)
    {
        /* New cluster created in Pass 1 - do not override! */
        return;
    }

    /* Allocate scratch arrays on the stack or heap */
    int *spatial_key  = calloc(
        (size_t) M, sizeof(int));
    int *spatial_mask = calloc(
        (size_t) M, sizeof(int));
    int *temporal_key = calloc(
        (size_t) M, sizeof(int));
    double *scores    = calloc(
        (size_t) maxcl, sizeof(double));

    if (spatial_key == NULL || spatial_mask == NULL
        || temporal_key == NULL || scores == NULL)
    {
        goto cleanup;
    }

    /* Build spatial key: all tiles' pass1 assignment */
    for (int m = 0; m < M; m++)
    {
        spatial_key[m] =
            mts->tile_states[m].pass1_assignment;
        spatial_mask[m] = 1;
    }

    /* Build temporal key from previous tuple */
    if (mts->tuple_count > 0)
    {
        long prev_idx =
            (mts->tuple_count - 1) * M;
        for (int m = 0; m < M; m++)
        {
            temporal_key[m] =
                mts->tuple_history[prev_idx + m];
        }
    }
    else
    {
        for (int m = 0; m < M; m++)
        {
            temporal_key[m] = -1;
        }
    }

    /* Retrieve matching tuple scores */
    tuple_retrieve(
        mts, tile_idx,
        spatial_key, spatial_mask,
        temporal_key, scores, maxcl);

    /* Fuse: multiply pass1_posterior by match_scores */
    {
        double best_val = 0.0;
        int    best_k   = ts->pass1_assignment;

        for (int k = 0; k < maxcl; k++)
        {
            ts->pass1_posterior[k] *= scores[k];
            if (ts->pass1_posterior[k] > best_val)
            {
                best_val = ts->pass1_posterior[k];
                best_k   = k;
            }
        }

        ts->pass1_assignment = best_k;
    } // fuse block

cleanup:
    free(spatial_key);
    free(spatial_mask);
    free(temporal_key);
    free(scores);
}

void predict_joint_tuples(
    MultiTileState *mts,
    int             pred_len,
    int             pred_h,
    int             pred_n)
{
    int M = mts->num_tiles;
    long T = mts->tuple_count;
    int L = pred_len;
    int H = pred_h;

    if (T < L)
    {
        /* Not enough history, initialize to uniform priors */
        for (int m = 0; m < M; m++)
        {
            TileState *ts = &mts->tile_states[m];
            int K = ts->state.num_clusters;
            for (int k = 0; k < K; k++)
            {
                ts->state.scratch.mixed_probs[k] = 1.0 / (double)K;
            }
            ts->state.scratch.tuple_pred_count = 0;
        }
        return;
    }

    int *recent_seq = malloc((size_t)(L * M) * sizeof(int));
    if (recent_seq == NULL)
    {
        return;
    }

    for (int j = 0; j < L; j++)
    {
        long frame_idx = T - L + j;
        memcpy(&recent_seq[j * M],
               &mts->tuple_history[frame_idx * M],
               (size_t)M * sizeof(int));
    }

    /* Identify the most selective tile based on cluster prob to dynamically limit H */
    double min_prob = 1.0;
    int m_best = 0;
    for (int m = 0; m < M; m++)
    {
        TileState *ts = &mts->tile_states[m];
        int q = recent_seq[(L - 1) * M + m];
        if (q >= 0 && q < ts->state.num_clusters)
        {
            double p = ts->state.clusters[q].prob;
            if (p < min_prob)
            {
                min_prob = p;
                m_best = m;
            }
        }
        else
        {
            min_prob = 0.0;
        }
    }

    if (min_prob > 0.08)
    {
        H = 100; /* Bypasses deep lookback for background states */
    }

    long search_limit = T - L;
    long search_start = (T > H) ? T - H : 0;
    if (search_start > search_limit)
    {
        search_start = search_limit;
    }

    /* We need maxnbclust for strides */
    int max_clusters = mts->tile_states[0].config.algo.maxnbclust;
    double *accum_scores = calloc((size_t)(M * max_clusters), sizeof(double));
    if (accum_scores == NULL)
    {
        free(recent_seq);
        return;
    }

    double decay = 1.0 - 1.0 / (double)L;

    long n_frames_in_horizon = search_limit - search_start;
    int index_search_ok = 0;
    if (mts->occurrence_head != NULL && mts->occurrence_prev != NULL &&
        n_frames_in_horizon > 0)
    {

        TileState *ts_best = &mts->tile_states[m_best];
        int q_best = recent_seq[(L - 1) * M + m_best];
        if (q_best >= 0 && q_best < ts_best->state.num_clusters)
        {
            index_search_ok = 1;
            double rlim_best = ts_best->config.algo.rlim;
            for (int C = 0; C < ts_best->state.num_clusters; C++)
            {
                double dcc_best = 0.0;
                if (q_best != C)
                {
                    int idx = q_best * max_clusters + C;
                    if (ts_best->state.scratch.dcc_measured[idx])
                    {
                        dcc_best = ts_best->state.scratch.dcc_min[idx];
                    }
                    else
                    {
                        dcc_best = ts_best->state.scratch.dcc_max[idx];
                    }
                    if (dcc_best < 0.0)
                    {
                        dcc_best = 2.0 * rlim_best;
                    }
                }

                if (dcc_best <= 2.0 * rlim_best)
                {
                    /* Traverse LIFO occurrence chain of C on m_best */
                    int t_occ = mts->occurrence_head[m_best * max_clusters + C];
                    while (t_occ != -1)
                    {
                        long s = (long)t_occ + 1;
                        if (s < search_start)
                        {
                            break; /* Link chain is in reverse chronological order */
                        }

                        if (s < search_limit)
                        {
                            /* Verify remaining M-1 tiles directly */
                            int match = 1;
                            for (int m = 0; m < M; m++)
                            {
                                if (m == m_best)
                                {
                                    continue;
                                }

                                int cB = mts->tuple_history[(s - 1) * M + m];
                                int q_m = recent_seq[(L - 1) * M + m];
                                TileState *ts = &mts->tile_states[m];

                                if (cB < 0 || cB >= ts->state.num_clusters ||
                                    q_m < 0 || q_m >= ts->state.num_clusters)
                                {
                                    match = 0;
                                    break;
                                }

                                double dcc = 0.0;
                                if (q_m != cB)
                                {
                                    int idx = q_m * max_clusters + cB;
                                    if (ts->state.scratch.dcc_measured[idx])
                                    {
                                        dcc = ts->state.scratch.dcc_min[idx];
                                    }
                                    else
                                    {
                                        dcc = ts->state.scratch.dcc_max[idx];
                                    }
                                    if (dcc < 0.0)
                                    {
                                        dcc = 2.0 * ts->config.algo.rlim;
                                    }
                                }

                                double rlim = ts->config.algo.rlim;
                                if (dcc > 2.0 * rlim)
                                {
                                    match = 0;
                                    break;
                                }
                            } // for m

                            if (match)
                            {
                                double sum_exponent = 0.0;
                                int sum_exceeded = 0;
                                for (int j = 0; j < L; j++)
                                {
                                    long hist_frame_idx = s - L + j;
                                    double frame_exponent = 0.0;
                                    for (int m = 0; m < M; m++)
                                    {
                                        int cA = recent_seq[j * M + m];
                                        int cB = mts->tuple_history[hist_frame_idx * M + m];
                                        TileState *ts = &mts->tile_states[m];
                                        if (cA < 0 || cA >= ts->state.num_clusters ||
                                            cB < 0 || cB >= ts->state.num_clusters)
                                        {
                                            continue;
                                        }

                                        double dcc = 0.0;
                                        if (cA != cB)
                                        {
                                            int maxcl = ts->config.algo.maxnbclust;
                                            int idx = cA * maxcl + cB;
                                            if (ts->state.scratch.dcc_measured[idx])
                                            {
                                                dcc = ts->state.scratch.dcc_min[idx];
                                            }
                                            else
                                            {
                                                dcc = ts->state.scratch.dcc_max[idx];
                                            }
                                            if (dcc < 0.0)
                                            {
                                                dcc = 2.0 * ts->config.algo.rlim;
                                            }
                                        }
                                        double rlim = ts->config.algo.rlim;
                                        frame_exponent += dcc / rlim;
                                    } // for m
                                    sum_exponent += pow(decay, (double)(L - 1 - j)) *
                                                    frame_exponent;
                                    if (sum_exponent > 10.0)
                                    {
                                        sum_exceeded = 1;
                                        break;
                                    }
                                } // for j

                                if (!sum_exceeded)
                                {
                                    double score = exp(-sum_exponent /
                                                       ((double)L * 0.63212055882855767));
                                    for (int m = 0; m < M; m++)
                                    {
                                        int c_next = mts->tuple_history[s * M + m];
                                        if (c_next >= 0 &&
                                            c_next < mts->tile_states[m].state.num_clusters)
                                        {
                                            accum_scores[m * max_clusters + c_next] += score;
                                        }
                                    }
                                }
                            } // if match
                        } // if s < search_limit
                        t_occ = mts->occurrence_prev[t_occ * M + m_best];
                    } // while t_occ
                } // if dcc_best <= 2.0 * rlim_best
            } // for C
        } // if q_best valid
    }

    if (!index_search_ok)
    {
        /* Fallback: brute force linear scan */
        for (long s = search_start; s < search_limit; s++)
        {
            double sum_exponent = 0.0;
            for (int j = 0; j < L; j++)
            {
                long hist_frame_idx = s - L + j;
                double frame_exponent = 0.0;
                for (int m = 0; m < M; m++)
                {
                    int cA = recent_seq[j * M + m];
                    int cB = mts->tuple_history[hist_frame_idx * M + m];
                    TileState *ts = &mts->tile_states[m];
                    if (cA < 0 || cA >= ts->state.num_clusters ||
                        cB < 0 || cB >= ts->state.num_clusters)
                    {
                        continue;
                    }
                    double dcc = 0.0;
                    if (cA != cB)
                    {
                        int maxcl = ts->config.algo.maxnbclust;
                        int idx = cA * maxcl + cB;
                        if (ts->state.scratch.dcc_measured[idx])
                        {
                            dcc = ts->state.scratch.dcc_min[idx];
                        }
                        else
                        {
                            dcc = ts->state.scratch.dcc_max[idx];
                        }
                        if (dcc < 0.0)
                        {
                            dcc = framedist(&ts->state.clusters[cA].anchor,
                                            &ts->state.clusters[cB].anchor);
                        }
                    }
                    double rlim = mts->tile_states[m].config.algo.rlim;
                    frame_exponent += dcc / rlim;
                } // for m
                sum_exponent += pow(decay, (double)(L - 1 - j)) * frame_exponent;
            } // for j

            double score = exp(-sum_exponent / ((double)L * 0.63212055882855767));
            for (int m = 0; m < M; m++)
            {
                int c_next = mts->tuple_history[s * M + m];
                if (c_next >= 0 && c_next < mts->tile_states[m].state.num_clusters)
                {
                    accum_scores[m * max_clusters + c_next] += score;
                }
            }
        } // for s
    }

    /* Populate mixed_probs and tuple_pred_candidates for each tile */
    for (int m = 0; m < M; m++)
    {
        TileState *ts = &mts->tile_states[m];
        int K = ts->state.num_clusters;
        double sum = 0.0;
        double alpha = 0.01; /* Laplace smoothing pseudocount */

        for (int k = 0; k < K; k++)
        {
            sum += accum_scores[m * max_clusters + k] + alpha;
        }

        for (int k = 0; k < K; k++)
        {
            ts->state.scratch.mixed_probs[k] =
                (accum_scores[m * max_clusters + k] + alpha) / sum;
        }

        Candidate *cand_list = malloc((size_t)K * sizeof(Candidate));
        if (cand_list != NULL)
        {
            for (int k = 0; k < K; k++)
            {
                cand_list[k].id = k;
                cand_list[k].p = accum_scores[m * max_clusters + k];
            }

            qsort(cand_list, (size_t)K, sizeof(Candidate), compare_candidates);

            int n_out = (K < pred_n) ? K : pred_n;
            for (int i = 0; i < n_out; i++)
            {
                ts->state.scratch.tuple_pred_candidates[i] = cand_list[i].id;
            }
            ts->state.scratch.tuple_pred_count = n_out;
            free(cand_list);
        }
    } // for m

    free(accum_scores);
    free(recent_seq);
}

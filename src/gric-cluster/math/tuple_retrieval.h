#ifndef TUPLE_RETRIEVAL_H
#define TUPLE_RETRIEVAL_H

/**
 * @file tuple_retrieval.h
 * @brief Unified tuple-list retrieval and Pass 2
 *        Bayesian fusion for multi-tile GRIC.
 */

#include "tile_state.h"

/**
 * @brief Scan tuple history for matching patterns.
 *
 * Produces per-cluster match scores for a target
 * tile by scanning historical tuples with a combined
 * spatial + temporal exponential match kernel.
 */
void tuple_retrieve(
    const MultiTileState *mts,
    int                   target_tile,
    const int            *spatial_key,
    const int            *spatial_mask,
    const int            *temporal_key,
    double               *match_scores,
    int                   max_clusters);

/**
 * @brief Run Pass 2 Bayesian fusion for one tile.
 *
 * Builds spatial and temporal keys from the current
 * pass1 assignments, retrieves matching tuples, and
 * fuses the match scores with the pass1 posterior.
 * Updates pass1_assignment in place if the fused
 * argmax differs.
 */
void pass2_fuse(
    MultiTileState *mts,
    int             tile_idx);

/**
 * @brief Predict joint transitions for all tiles before Pass 1.
 *
 * Scans tuple history using a fuzzy joint pattern similarity
 * metric, then populates scratch mixed_probs and tuple_pred_candidates
 * for all tiles.
 */
void predict_joint_tuples(
    MultiTileState *mts,
    int             pred_len,
    int             pred_h,
    int             pred_n);

#endif // TUPLE_RETRIEVAL_H

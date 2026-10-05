#ifndef TUPLE_RETRIEVAL_H
#define TUPLE_RETRIEVAL_H

/**
 * @file tuple_retrieval.h
 * @brief Unified tuple-list retrieval and Joint Trajectory Fusion (Pass 2)
 *        for multi-tile GRIC.
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
 * @brief Run Joint Trajectory Fusion (Pass 2) for one tile.
 *
 * Builds spatial and temporal keys from the current
 * Independent Spatial Clustering (Pass 1) assignments, retrieves matching tuples, and
 * fuses the match scores with the Pass 1 posterior.
 * Updates the assignment in place if the fused
 * argmax differs.
 */
void pass2_fuse(
    MultiTileState *mts,
    int             tile_idx,
    Frame          *tile_frame);

/**
 * @brief Predict joint transitions for all tiles before Independent Spatial Clustering (Pass 1).
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

/**
 * @brief Update the Conditional Probability Table incrementally with decay.
 */
void cpt_update_incremental(
    double           *cpt,
    double           *cpt_scale,
    const int        *tuple_history,
    long              tuple_count,
    int               num_tiles,
    int               max_clusters,
    double            decay);

/**
 * @brief Hook callback to inject cross-tile prior probabilities.
 */
void inject_cross_tile_priors(
    void *state,
    void *ctx);

/**
 * @brief Hook callback to inject spatial-temporal cross-tile prior probabilities (Strategy C).
 */
void inject_cross_tile_priors_st(
    void *state,
    void *ctx);

#endif // TUPLE_RETRIEVAL_H

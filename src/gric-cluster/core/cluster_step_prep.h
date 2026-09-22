/**
 * @file cluster_step_prep.h
 * @brief Declarations for frame quantization buffer preparation and prediction retrieval.
 */

#ifndef CLUSTER_STEP_PREP_H
#define CLUSTER_STEP_PREP_H

#include "cluster_defs.h"

/**
 * prepare_frame_quantization() - Allocates buffers and quantizes input frame.
 * @config:        Pointer to clustering configuration.
 * @state:         Pointer to clustering state.
 * @current_frame: Pointer to input frame to quantize.
 */
void prepare_frame_quantization(
    ClusterConfig *config,
    ClusterState  *state,
    const Frame   *current_frame);

/**
 * retrieve_prediction_candidates() - Retrieves prediction candidates into scratch buffer.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state with pre-allocated scratch buffers.
 * @pred_candidates: Pointer to pre-allocated output array of candidate cluster IDs.
 *
 * Return: Number of prediction candidates populated in @pred_candidates.
 */
int retrieve_prediction_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    int           *pred_candidates);

#endif // CLUSTER_STEP_PREP_H

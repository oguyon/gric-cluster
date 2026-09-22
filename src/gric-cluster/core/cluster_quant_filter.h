/**
 * @file cluster_quant_filter.h
 * @brief Multi-tier quantization lower-bound candidate screening and pruning.
 */

#ifndef CLUSTER_QUANT_FILTER_H
#define CLUSTER_QUANT_FILTER_H

#include "cluster_defs.h"
#include <stdint.h>
#include <time.h>

/**
 * cluster_compute_quant_thresholds() - Computes effective cutoff thresholds for quantization.
 * @config:              Pointer to clustering configuration.
 * @out_eq16_adc_cutoff: Output pointer for EQ16 Asymmetric Distance (ADC) cutoff.
 * @out_eq16_ssd_thresh: Output pointer for EQ16 Symmetric SSD cutoff.
 * @out_sq16_ssd_thresh: Output pointer for SQ16 Symmetric SSD cutoff.
 */
void cluster_compute_quant_thresholds(
    const ClusterConfig *config,
    float               *out_eq16_adc_cutoff,
    uint64_t            *out_eq16_ssd_thresh,
    uint64_t            *out_sq16_ssd_thresh);

/**
 * cluster_candidate_is_pruned_by_quant() - Fast metric lower-bound test for a candidate.
 * @cj:              Cluster index to test.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state.
 * @eq16_adc_cutoff: Precomputed EQ16 ADC cutoff threshold.
 * @eq16_ssd_thresh: Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh: Precomputed SQ16 SSD threshold.
 *
 * Return: 1 if candidate is pruned by metric lower bound, 0 if it survives.
 */
int cluster_candidate_is_pruned_by_quant(
    int            cj,
    ClusterConfig *config,
    ClusterState  *state,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh);

/**
 * cluster_quant_filter_initial() - Initial matrix-level screening across all clusters.
 * @config:                Pointer to clustering configuration.
 * @state:                 Pointer to clustering state.
 * @sorting_candidates:    Scratch candidate array.
 * @prev_assigned_cluster: Index of previous cluster assignment (-1 if none).
 * @fast_eq16:             Flag indicating fast EQ16 path is enabled.
 * @fast_sq16:             Flag indicating fast SQ16 path is enabled.
 * @eq16_adc_cutoff:       Precomputed EQ16 ADC cutoff.
 * @eq16_ssd_thresh:       Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh:       Precomputed SQ16 SSD threshold.
 * @step_start:            Timestamp when Step 3a began.
 */
void cluster_quant_filter_initial(
    ClusterConfig  *config,
    ClusterState   *state,
    Candidate      *sorting_candidates,
    int             prev_assigned_cluster,
    int             fast_eq16,
    int             fast_sq16,
    float           eq16_adc_cutoff,
    uint64_t        eq16_ssd_thresh,
    uint64_t        sq16_ssd_thresh,
    struct timespec step_start);

/**
 * cluster_quant_filter_subsequent() - Filters surviving candidate clusters.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state.
 * @eq16_adc_cutoff: Precomputed EQ16 ADC cutoff.
 * @eq16_ssd_thresh: Precomputed EQ16 SSD threshold.
 * @sq16_ssd_thresh: Precomputed SQ16 SSD threshold.
 */
void cluster_quant_filter_subsequent(
    ClusterConfig *config,
    ClusterState  *state,
    float          eq16_adc_cutoff,
    uint64_t       eq16_ssd_thresh,
    uint64_t       sq16_ssd_thresh);

#endif // CLUSTER_QUANT_FILTER_H

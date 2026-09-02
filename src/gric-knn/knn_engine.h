#ifndef KNN_ENGINE_H
#define KNN_ENGINE_H

/**
 * @file knn_engine.h
 * @brief High-performance metric-pruned k-NN solver engine.
 */

#include "knn_defs.h"
#include "knn_reader.h"

/**
 * @brief Execute metric-pruned k-nearest neighbor search across all queries.
 * @param config    Pointer to active KnnConfig configuration.
 * @param model     Pointer to active KnnModel.
 * @param results   Pointer to KnnResults structure to receive sorted indices and distances.
 * @param telemetry Pointer to KnnTelemetry structure to receive performance counters.
 * @return 0 on success, -1 on failure.
 */
int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry);

/**
 * @brief Free heap and buffer allocations inside a KnnResults structure.
 * @param results Pointer to KnnResults structure to free.
 */
void knn_results_free(
    KnnResults *results);

#endif // KNN_ENGINE_H

#ifndef KNN_ENGINE_H
#define KNN_ENGINE_H

/**
 * @file knn_engine.h
 * @brief High-performance metric-pruned k-NN solver engine.
 *
 * Declares the primary search engine driver knn_run_search() and results cleanup
 * function knn_results_free(). Coordinates hardware dispatch, multi-threaded CPU
 * batch execution, and global telemetry reduction for all query frames.
 */

#include "knn_defs.h"
#include "knn_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_run_search() - Execute metric-pruned k-nearest neighbor search across queries
 * @config:    Pointer to active KnnConfig configuration.
 * @model:     Pointer to active KnnModel.
 * @results:   Pointer to KnnResults structure to receive sorted indices and distances.
 * @telemetry: Pointer to KnnTelemetry structure to receive performance counters.
 *
 * Return: 0 on success, or -1 on failure.
 */
int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry);

/**
 * knn_results_free() - Free heap and buffer allocations inside a KnnResults structure
 * @results: Pointer to KnnResults structure to free.
 */
void knn_results_free(
    KnnResults *results);

#ifdef __cplusplus
}
#endif

#endif // KNN_ENGINE_H

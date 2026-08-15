#ifndef KNN_ENGINE_H
#define KNN_ENGINE_H

/**
 * @file knn_engine.h
 * @brief High-performance metric-pruned k-NN solver engine.
 */

#include "knn_defs.h"
#include "knn_reader.h"

int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry);

void knn_results_free(
    KnnResults *results);

#endif // KNN_ENGINE_H

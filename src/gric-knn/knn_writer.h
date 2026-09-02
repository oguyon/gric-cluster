#ifndef KNN_WRITER_H
#define KNN_WRITER_H

/**
 * @file knn_writer.h
 * @brief Output serialization for gric-knn results into FITS or ASCII formats.
 */

#include "knn_defs.h"

/**
 * @brief Write KNN search results to output file (FITS or ASCII).
 * @param config  Pointer to active KnnConfig.
 * @param model   Pointer to active KnnModel.
 * @param results Pointer to completed KnnResults.
 * @return 0 on success, -1 on write error.
 */
int knn_write_results(
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results);

#endif // KNN_WRITER_H

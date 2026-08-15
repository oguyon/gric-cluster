#ifndef KNN_WRITER_H
#define KNN_WRITER_H

/**
 * @file knn_writer.h
 * @brief Output serialization for gric-knn results into FITS or ASCII formats.
 */

#include "knn_defs.h"

int knn_write_results(
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results);

#endif // KNN_WRITER_H

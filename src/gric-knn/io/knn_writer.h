#ifndef KNN_WRITER_H
#define KNN_WRITER_H

/**
 * @file knn_writer.h
 * @brief Output serialization for gric-knn results into FITS or ASCII formats.
 *
 * Declares serialization routines for writing computed k-NN indices, distances, and
 * pairwise mutual distance matrices to disk in binary (GRIC .bin), multi-extension FITS,
 * or ASCII formats.
 */

#include "knn_defs.h"

/**
 * knn_write_results() - Write KNN search results to output file (FITS, binary, or ASCII)
 * @config:  Pointer to active KnnConfig.
 * @model:   Pointer to active KnnModel.
 * @results: Pointer to completed KnnResults containing neighbor indices and distances.
 *
 * Serializes nearest neighbor indices, distances, and optional mutual distances.
 *
 * Return: 0 on success, or -1 on write error.
 */
int knn_write_results(
    const KnnConfig  *config,
    const KnnModel   *model,
    const KnnResults *results);

#endif // KNN_WRITER_H

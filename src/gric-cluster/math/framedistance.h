#ifndef FRAMEDISTANCE_H
#define FRAMEDISTANCE_H

#include "common.h"

#if defined(__AVX512F__)
#define FRAMEDIST_BATCH_SIZE 8
#else
#define FRAMEDIST_BATCH_SIZE 4
#endif

/**
 * framedist() - Computes the Euclidean distance between two frames.
 * @a: Pointer to the first Frame.
 * @b: Pointer to the second Frame.
 *
 * Return: The Euclidean distance, or -1.0 if the frame dimensions mismatch.
 */
double framedist(
    const Frame *a,
    const Frame *b);

/**
 * framedist_float() - Compute Euclidean distance between two single-precision float arrays.
 * @da:   First array pointer.
 * @db:   Second array pointer.
 * @size: Number of elements.
 *
 * Return: Euclidean distance.
 */
double framedist_float(
    const float *restrict da,
    const float *restrict db,
    long                  size);

/**
 * framedist_double() - Compute Euclidean distance between two double-precision arrays.
 * @da:   First array pointer.
 * @db:   Second array pointer.
 * @size: Number of elements.
 *
 * Return: Euclidean distance.
 */
double framedist_double(
    const double *restrict da,
    const double *restrict db,
    long                   size);

/**
 * framedist_squared_cutoff_float() - Squared Euclidean distance with early termination threshold.
 * @da:        First float array pointer.
 * @db:        Second float array pointer.
 * @size:      Number of elements.
 * @cutoff_sq: Squared distance threshold for early exit.
 *
 * Return: Squared distance, or value >= cutoff_sq if early exit triggered.
 */
double framedist_squared_cutoff_float(
    const float *restrict da,
    const float *restrict db,
    long                  size,
    double                cutoff_sq);

/**
 * framedist_squared_cutoff_double() - Squared Euclidean distance with early termination for double.
 * @da:        First double array pointer.
 * @db:        Second double array pointer.
 * @size:      Number of elements.
 * @cutoff_sq: Squared distance threshold for early exit.
 *
 * Return: Squared distance, or value >= cutoff_sq if early exit triggered.
 */
double framedist_squared_cutoff_double(
    const double *restrict da,
    const double *restrict db,
    long                   size,
    double                 cutoff_sq);

/**
 * framedist_batch_1x4_float() - Vectorized batch distance from 1 query to 4 anchor float frames.
 * @q:         Query vector pointer.
 * @anchors:   Array of 4 anchor vector pointers.
 * @out_dists: Output array for 4 computed distances.
 * @size:      Vector length.
 */
void framedist_batch_1x4_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size);

/**
 * framedist_batch_1x4_double() - Vectorized batch distance from 1 query to 4 anchor double frames.
 * @q:         Query vector pointer.
 * @anchors:   Array of 4 anchor vector pointers.
 * @out_dists: Output array for 4 computed distances.
 * @size:      Vector length.
 */
void framedist_batch_1x4_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size);

/**
 * framedist_batch_1x8_float() - Vectorized batch distance from 1 query to 8 anchor float frames.
 * @q:         Query vector pointer.
 * @anchors:   Array of 8 anchor vector pointers.
 * @out_dists: Output array for 8 computed distances.
 * @size:      Vector length.
 */
void framedist_batch_1x8_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size);

/**
 * framedist_batch_1x8_double() - Vectorized batch distance from 1 query to 8 anchor double frames.
 * @q:         Query vector pointer.
 * @anchors:   Array of 8 anchor vector pointers.
 * @out_dists: Output array for 8 computed distances.
 * @size:      Vector length.
 */
void framedist_batch_1x8_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size);

/**
 * framedist_batch_float() - Evaluate batch of distances against multiple float anchors.
 * @q:         Query vector pointer.
 * @anchors:   Array of anchor pointers.
 * @n_anchors: Number of anchors in batch.
 * @out_dists: Output array for computed distances.
 * @size:      Vector length.
 */
void framedist_batch_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size);

/**
 * framedist_batch_double() - Evaluate batch of distances against multiple double anchors.
 * @q:         Query vector pointer.
 * @anchors:   Array of anchor pointers.
 * @n_anchors: Number of anchors in batch.
 * @out_dists: Output array for computed distances.
 * @size:      Vector length.
 */
void framedist_batch_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size);

/**
 * framedist_batch() - High-level batch evaluation across Frame structures.
 * @q:         Query Frame pointer.
 * @anchors:   Array of candidate anchor Frame pointers.
 * @n_anchors: Number of candidate anchors.
 * @out_dists: Output array for computed distances.
 */
void framedist_batch(
    const Frame  *q,
    const Frame **anchors,
    int           n_anchors,
    double       *out_dists);

#endif // FRAMEDISTANCE_H

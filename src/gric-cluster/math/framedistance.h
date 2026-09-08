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

double framedist_float(
    const float *restrict da,
    const float *restrict db,
    long                  size);

double framedist_double(
    const double *restrict da,
    const double *restrict db,
    long                   size);

void framedist_batch_1x4_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size);

void framedist_batch_1x4_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size);

void framedist_batch_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size);

void framedist_batch_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size);

void framedist_batch(
    const Frame  *q,
    const Frame **anchors,
    int           n_anchors,
    double       *out_dists);

#endif // FRAMEDISTANCE_H

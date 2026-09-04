#ifndef FRAMEDISTANCE_H
#define FRAMEDISTANCE_H

#include "common.h"

/**
 * @brief Computes the Euclidean distance between two frames.
 *
 * Checks that the frames have matching dimensions (width and height),
 * and then computes the L2 Euclidean distance between their pixel data.
 * Utilizes SIMD/AVX2 vectorization when compiled on supporting x86 architectures.
 *
 * @param a Pointer to the first Frame.
 * @param b Pointer to the second Frame.
 * @return The Euclidean distance, or -1.0 if the frame dimensions mismatch.
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

#endif // FRAMEDISTANCE_H

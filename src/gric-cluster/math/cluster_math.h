#ifndef CLUSTER_MATH_H
#define CLUSTER_MATH_H

#include "cluster_defs.h"
#include "cluster_locator.h"

int compare_candidates(
    const void *a,
    const void *b);

/**
 * @brief Evaluates match probability weight based on normalized distance.
 *
 * Implements a linear decay matching reward: returns a value interpolating
 * from 'a' (at dr=0) to 'b' (at dr=2). Beyond dr=2, returns 0.
 *
 * @param dr Normalized distance ratio.
 * @param a Start threshold weight.
 * @param b End threshold weight.
 * @return Probability factor in range [0.0, a].
 */
double fmatch(
    double dr,
    double a,
    double b);

#endif // CLUSTER_MATH_H

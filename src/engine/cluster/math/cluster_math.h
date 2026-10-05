#ifndef CLUSTER_MATH_H
#define CLUSTER_MATH_H

#include "cluster_defs.h"
#include "cluster_locator.h"

/**
 * compare_candidates() - Comparator for Candidate structures by distance (ascending).
 * @a: Pointer to first Candidate.
 * @b: Pointer to second Candidate.
 *
 * Return: Negative if a < b, positive if a > b, 0 if equal.
 */
int compare_candidates(
    const void *a,
    const void *b);

/**
 * fmatch() - Evaluates match probability weight based on normalized distance.
 * @dr: Normalized distance ratio.
 * @a:  Start threshold weight (at dr=0).
 * @b:  End threshold weight (at dr=2).
 *
 * Return: Probability factor in range [0.0, a].
 */
double fmatch(
    double dr,
    double a,
    double b);

#endif // CLUSTER_MATH_H

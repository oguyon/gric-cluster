/**
 * @file cluster_math.c
 * @brief High-performance math routines for distance calculation.
 *
 * Implements float matching logic and optimized multi-point triangle-inequality distance
 * calculation heuristics.
 *
 * Main Functions:
 * - fmatch: Checks if a value is close to a target within a small threshold.
 * - calc_min_dist_4pt: Computes the lower distance bound using a 4-point inequality.
 * - calc_min_dist_5pt: Computes the lower distance bound using a 5-point inequality.
 */
#include "cluster_math.h"
#include <math.h>

/**
 * fmatch() - Evaluates match probability weight based on normalized distance.
 * @dr: Normalized distance ratio.
 * @a:  Start threshold weight.
 * @b:  End threshold weight.
 *
 * Return: Probability factor in range [0.0, a].
 */
double fmatch(
    double dr,
    double a,
    double b)
{
    if (dr > 2.0)
        return 0.0;
    return a - (a - b) * dr / 2.0;
}

/**
 * @brief Comparison helper function for candidate sorting.
 *
 * Compares two Candidates by their probability values in descending order.
 *
 * @param a Pointer to the first Candidate.
 * @param b Pointer to the second Candidate.
 * @return 1 if first has lower probability, -1 if first has higher, 0 if equal.
 */
int compare_candidates(
    const void *a,
    const void *b)
{
    const Candidate *ca = (const Candidate *)a;
    const Candidate *cb = (const Candidate *)b;

    if (ca->p < cb->p)
    {
        return 1;
    }
    if (ca->p > cb->p)
    {
        return -1;
    }
    return 0;
}

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

double fmatch(double dr, double a, double b)
{
    if (dr > 2.0)
        return 0.0;
    return a - (a - b) * dr / 2.0;
}

double calc_min_dist_4pt(double d14, double d24, double d12, double d13, double d23)
{
    if (d12 < 1e-9)
        return fabs(d14 - d13);

    double x3 = (d13 * d13 + d12 * d12 - d23 * d23) / (2.0 * d12);
    double y3_sq = d13 * d13 - x3 * x3;
    double y3 = (y3_sq > 0.0) ? sqrt(y3_sq) : 0.0;

    double x4 = (d14 * d14 + d12 * d12 - d24 * d24) / (2.0 * d12);
    double y4_sq = d14 * d14 - x4 * x4;
    double y4 = (y4_sq > 0.0) ? sqrt(y4_sq) : 0.0;

    return sqrt((x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4));
}

double calc_min_dist_5pt(double d_f_c1, double d_f_c2, double d_f_c3, double d_t_c1, double d_t_c2,
                         double d_t_c3, double d_c1_c2, double d_c1_c3, double d_c2_c3)
{
    if (d_c1_c2 < 1e-9)
        return 0.0;

    double x3 = (d_c1_c3 * d_c1_c3 + d_c1_c2 * d_c1_c2 - d_c2_c3 * d_c2_c3) / (2.0 * d_c1_c2);
    double y3_sq = d_c1_c3 * d_c1_c3 - x3 * x3;
    if (y3_sq < 1e-9)
        return 0.0;
    double y3 = sqrt(y3_sq);

    double xF = (d_f_c1 * d_f_c1 + d_c1_c2 * d_c1_c2 - d_f_c2 * d_f_c2) / (2.0 * d_c1_c2);
    double yF =
        (d_f_c1 * d_f_c1 + d_c1_c3 * d_c1_c3 - d_f_c3 * d_f_c3 - 2.0 * xF * x3) / (2.0 * y3);
    double zF_sq = d_f_c1 * d_f_c1 - xF * xF - yF * yF;
    double zF = (zF_sq > 0.0) ? sqrt(zF_sq) : 0.0;

    double xT = (d_t_c1 * d_t_c1 + d_c1_c2 * d_c1_c2 - d_t_c2 * d_t_c2) / (2.0 * d_c1_c2);
    double yT =
        (d_t_c1 * d_t_c1 + d_c1_c3 * d_c1_c3 - d_t_c3 * d_t_c3 - 2.0 * xT * x3) / (2.0 * y3);
    double zT_sq = d_t_c1 * d_t_c1 - xT * xT - yT * yT;
    double zT = (zT_sq > 0.0) ? sqrt(zT_sq) : 0.0;

    return sqrt((xF - xT) * (xF - xT) + (yF - yT) * (yF - yT) + (zF - zT) * (zF - zT));
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

/**
 * @file gric_omp.h
 * @brief OpenMP helpers shared by all GRIC modules.
 *
 * Header-only, no dependency on gric-cluster or gric-knn headers. Builds without OpenMP get
 * no-op versions of the helpers.
 */

#ifndef GRIC_OMP_H
#define GRIC_OMP_H

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * GRIC_OMP_MIN_WORK - Minimum work (element operations) for a parallel region to pay off.
 *
 * Fork/join of an OpenMP team costs a few microseconds, and much more when threads are
 * oversubscribed or asleep. Loops whose total work is below this threshold run serially.
 * Use it as `#pragma omp parallel for if((long)nitems * per_item_ops >= GRIC_OMP_MIN_WORK)`.
 * Can be overridden at compile time for tuning.
 */
#ifndef GRIC_OMP_MIN_WORK
#define GRIC_OMP_MIN_WORK 200000L
#endif

/**
 * gric_omp_set_threads() - Set the thread count for subsequent parallel regions.
 * @nthreads: Requested thread count; values below 1 are treated as 1.
 *
 * Return: The previous thread count (omp_get_max_threads()), so callers can restore it.
 */
static inline int gric_omp_set_threads(
    int nthreads)
{
#ifdef _OPENMP
    int prev = omp_get_max_threads();
    omp_set_num_threads(nthreads > 0 ? nthreads : 1);
    return prev;
#else
    (void)nthreads;
    return 1;
#endif
}

#endif // GRIC_OMP_H

/**
 * @file framedistance_cutoff.c
 * @brief Euclidean distance calculation with early-exit cutoff thresholds.
 *
 * Implements single- and double-precision Euclidean L2 distance calculations
 * that check partial sums periodically and abort early if the squared distance
 * exceeds a specified cutoff threshold.
 */

#include "framedistance.h"
#include "common.h"
#include <math.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * framedist_squared_cutoff_float() - Vectorized single-precision L2 distance with early cutoff.
 * @da:        Pointer to first pixel array.
 * @db:        Pointer to second pixel array.
 * @size:      Number of elements in frame.
 * @cutoff_sq: Squared distance threshold for early termination.
 *
 * Checks partial squared distance periodically (every 64 elements). If the partial sum
 * exceeds @cutoff_sq, returns @cutoff_sq + 1.0 immediately, avoiding remaining computation
 * and the square root.
 *
 * Return: Euclidean L2 distance if <= cutoff, or (cutoff_sq + 1.0) on early exit.
 */
double framedist_squared_cutoff_float(
    const float *restrict da,
    const float *restrict db,
    long                  size,
    double                cutoff_sq)
{
    if (size == 2)
    {
        float d0 = da[0] - db[0];
        float d1 = da[1] - db[1];
        float s = d0 * d0 + d1 * d1;
        if (cutoff_sq > 0.0 && (double)s > cutoff_sq)
        {
            return cutoff_sq + 1.0;
        }
        return (double)sqrtf(s);
    }

    if (size == 3)
    {
        float d0 = da[0] - db[0];
        float d1 = da[1] - db[1];
        float d2 = da[2] - db[2];
        float s = d0 * d0 + d1 * d1 + d2 * d2;
        if (cutoff_sq > 0.0 && (double)s > cutoff_sq)
        {
            return cutoff_sq + 1.0;
        }
        return (double)sqrtf(s);
    }

    float sum = 0.0f;
    long i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 64)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (; i <= size - 32; i += 32)
        {
            __m256 va0 = _mm256_loadu_ps(&da[i]);
            __m256 vb0 = _mm256_loadu_ps(&db[i]);
            __m256 d0 = _mm256_sub_ps(va0, vb0);

            __m256 va1 = _mm256_loadu_ps(&da[i + 8]);
            __m256 vb1 = _mm256_loadu_ps(&db[i + 8]);
            __m256 d1 = _mm256_sub_ps(va1, vb1);

            __m256 va2 = _mm256_loadu_ps(&da[i + 16]);
            __m256 vb2 = _mm256_loadu_ps(&db[i + 16]);
            __m256 d2 = _mm256_sub_ps(va2, vb2);

            __m256 va3 = _mm256_loadu_ps(&da[i + 24]);
            __m256 vb3 = _mm256_loadu_ps(&db[i + 24]);
            __m256 d3 = _mm256_sub_ps(va3, vb3);

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
#endif

            if (cutoff_sq > 0.0)
            {
                __m256 vcut = _mm256_set1_ps((float)cutoff_sq);
                __m256 max01 = _mm256_max_ps(acc0, acc1);
                __m256 max23 = _mm256_max_ps(acc2, acc3);
                __m256 max_lane = _mm256_max_ps(max01, max23);
                if (_mm256_movemask_ps(_mm256_cmp_ps(max_lane, vcut, _CMP_GT_OQ)) != 0)
                {
                    return cutoff_sq + 1.0;
                }

                if ((i + 32) % 128 == 0 || i + 32 >= size)
                {
                    __m256 s01 = _mm256_add_ps(acc0, acc1);
                    __m256 s23 = _mm256_add_ps(acc2, acc3);
                    __m256 s256 = _mm256_add_ps(s01, s23);
                    __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(s256),
                                             _mm256_extractf128_ps(s256, 1));
                    __m128 shuf = _mm_movehl_ps(s128, s128);
                    __m128 squad = _mm_add_ps(s128, shuf);
                    shuf = _mm_shuffle_ps(squad, squad, 1);
                    __m128 sscalar = _mm_add_ss(squad, shuf);
                    float partial = _mm_cvtss_f32(sscalar);
                    if ((double)partial > cutoff_sq)
                    {
                        return cutoff_sq + 1.0;
                    }
                }
            }
        } // for (; i <= size - 32; i += 32)

        for (; i <= size - 8; i += 8)
        {
            __m256 va = _mm256_loadu_ps(&da[i]);
            __m256 vb = _mm256_loadu_ps(&db[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(diff, diff, acc0);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff, diff));
#endif
        }

        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum_vec = _mm256_add_ps(sum01, sum23);

        __m128 vlow = _mm256_castps256_ps128(sum_vec);
        __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehl_ps(vsum, vsum);
        __m128 sum_quad = _mm_add_ps(vsum, shuf);
        shuf = _mm_shuffle_ps(sum_quad, sum_quad, 1);
        __m128 sum_scalar = _mm_add_ss(sum_quad, shuf);
        sum += _mm_cvtss_f32(sum_scalar);
    }
#endif

    for (; i < size; i++)
    {
        float diff = da[i] - db[i];
        sum += diff * diff;
    }

    if (cutoff_sq > 0.0 && (double)sum > cutoff_sq)
    {
        return cutoff_sq + 1.0;
    }

    return sqrt((double)sum);
}

/**
 * framedist_squared_cutoff_double() - Vectorized double-precision L2 distance with early cutoff.
 * @da:        Pointer to first pixel array.
 * @db:        Pointer to second pixel array.
 * @size:      Number of elements in frame.
 * @cutoff_sq: Squared distance threshold for early termination.
 *
 * Return: Euclidean L2 distance if <= cutoff, or (cutoff_sq + 1.0) on early exit.
 */
double framedist_squared_cutoff_double(
    const double *restrict da,
    const double *restrict db,
    long                   size,
    double                 cutoff_sq)
{
    if (size == 2)
    {
        double d0 = da[0] - db[0];
        double d1 = da[1] - db[1];
        double s = d0 * d0 + d1 * d1;
        if (cutoff_sq > 0.0 && s > cutoff_sq)
        {
            return cutoff_sq + 1.0;
        }
        return sqrt(s);
    }

    if (size == 3)
    {
        double d0 = da[0] - db[0];
        double d1 = da[1] - db[1];
        double d2 = da[2] - db[2];
        double s = d0 * d0 + d1 * d1 + d2 * d2;
        if (cutoff_sq > 0.0 && s > cutoff_sq)
        {
            return cutoff_sq + 1.0;
        }
        return sqrt(s);
    }

    double sum = 0.0;
    long i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 32)
    {
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();

        for (; i <= size - 16; i += 16)
        {
            __m256d va0 = _mm256_loadu_pd(&da[i]);
            __m256d vb0 = _mm256_loadu_pd(&db[i]);
            __m256d d0 = _mm256_sub_pd(va0, vb0);

            __m256d va1 = _mm256_loadu_pd(&da[i + 4]);
            __m256d vb1 = _mm256_loadu_pd(&db[i + 4]);
            __m256d d1 = _mm256_sub_pd(va1, vb1);

            __m256d va2 = _mm256_loadu_pd(&da[i + 8]);
            __m256d vb2 = _mm256_loadu_pd(&db[i + 8]);
            __m256d d2 = _mm256_sub_pd(va2, vb2);

            __m256d va3 = _mm256_loadu_pd(&da[i + 12]);
            __m256d vb3 = _mm256_loadu_pd(&db[i + 12]);
            __m256d d3 = _mm256_sub_pd(va3, vb3);

#ifdef __FMA__
            acc0 = _mm256_fmadd_pd(d0, d0, acc0);
            acc1 = _mm256_fmadd_pd(d1, d1, acc1);
            acc2 = _mm256_fmadd_pd(d2, d2, acc2);
            acc3 = _mm256_fmadd_pd(d3, d3, acc3);
#else
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(d2, d2));
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(d3, d3));
#endif

            if (cutoff_sq > 0.0)
            {
                __m256d vcut = _mm256_set1_pd(cutoff_sq);
                __m256d max01 = _mm256_max_pd(acc0, acc1);
                __m256d max23 = _mm256_max_pd(acc2, acc3);
                __m256d max_lane = _mm256_max_pd(max01, max23);
                if (_mm256_movemask_pd(_mm256_cmp_pd(max_lane, vcut, _CMP_GT_OQ)) != 0)
                {
                    return cutoff_sq + 1.0;
                }

                if ((i + 16) % 64 == 0 || i + 16 >= size)
                {
                    __m256d s01 = _mm256_add_pd(acc0, acc1);
                    __m256d s23 = _mm256_add_pd(acc2, acc3);
                    __m256d s256 = _mm256_add_pd(s01, s23);
                    __m128d vlow = _mm256_castpd256_pd128(s256);
                    __m128d vhigh = _mm256_extractf128_pd(s256, 1);
                    __m128d vsum = _mm_add_pd(vlow, vhigh);
                    double partial = _mm_cvtsd_f64(_mm_add_sd(vsum, _mm_unpackhi_pd(vsum, vsum)));
                    if (partial > cutoff_sq)
                    {
                        return cutoff_sq + 1.0;
                    }
                }
            }
        } // for (; i <= size - 16; i += 16)

        for (; i <= size - 4; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&da[i]);
            __m256d vb = _mm256_loadu_pd(&db[i]);
            __m256d diff = _mm256_sub_pd(va, vb);
#ifdef __FMA__
            acc0 = _mm256_fmadd_pd(diff, diff, acc0);
#else
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(diff, diff));
#endif
        }

        __m256d sum01 = _mm256_add_pd(acc0, acc1);
        __m256d sum23 = _mm256_add_pd(acc2, acc3);
        __m256d sum_vec = _mm256_add_pd(sum01, sum23);

        __m128d vlow = _mm256_castpd256_pd128(sum_vec);
        __m128d vhigh = _mm256_extractf128_pd(sum_vec, 1);
        __m128d vsum = _mm_add_pd(vlow, vhigh);
        sum += _mm_cvtsd_f64(_mm_add_sd(vsum, _mm_unpackhi_pd(vsum, vsum)));
    }
#endif

    for (; i < size; i++)
    {
        double diff = da[i] - db[i];
        sum += diff * diff;
    }

    if (cutoff_sq > 0.0 && sum > cutoff_sq)
    {
        return cutoff_sq + 1.0;
    }

    return sqrt(sum);
}

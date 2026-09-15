/**
 * @file framedistance.c
 * @brief Euclidean distance calculation between frames.
 *
 * Implements the standard Euclidean distance metric between two multi-dimensional
 * frame coordinate vectors.
 *
 * Main Functions:
 * - framedist: Computes the Euclidean distance between two frames.
 */
#include "framedistance.h"
#include "common.h"
#include <math.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * framedist() - Computes the Euclidean distance between two frames.
 * @a: Pointer to the first Frame.
 * @b: Pointer to the second Frame.
 *
 * Checks that the frames have matching dimensions (width and height),
 * and then computes the L2 Euclidean distance between their pixel data.
 * Utilizes SIMD/AVX2 vectorization when compiled on supporting x86 architectures.
 *
 * Return: The Euclidean distance, or -1.0 if the frame dimensions mismatch.
 */
/**
 * framedist_float() - Vectorized single-precision Euclidean distance between pixel arrays.
 * @da:   Pointer to first pixel array.
 * @db:   Pointer to second pixel array.
 * @size: Number of elements in frame.
 *
 * Return: Euclidean L2 distance as double.
 */
double framedist_float(
    const float *restrict da,
    const float *restrict db,
    long                  size)
{
    float sum = 0.0f;
    long i = 0;

#if defined(__AVX512F__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 64)
    {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (; i <= size - 64; i += 64)
        {
            __m512 va0 = _mm512_loadu_ps(&da[i]);
            __m512 vb0 = _mm512_loadu_ps(&db[i]);
            __m512 d0 = _mm512_sub_ps(va0, vb0);

            __m512 va1 = _mm512_loadu_ps(&da[i + 16]);
            __m512 vb1 = _mm512_loadu_ps(&db[i + 16]);
            __m512 d1 = _mm512_sub_ps(va1, vb1);

            __m512 va2 = _mm512_loadu_ps(&da[i + 32]);
            __m512 vb2 = _mm512_loadu_ps(&db[i + 32]);
            __m512 d2 = _mm512_sub_ps(va2, vb2);

            __m512 va3 = _mm512_loadu_ps(&da[i + 48]);
            __m512 vb3 = _mm512_loadu_ps(&db[i + 48]);
            __m512 d3 = _mm512_sub_ps(va3, vb3);

            acc0 = _mm512_fmadd_ps(d0, d0, acc0);
            acc1 = _mm512_fmadd_ps(d1, d1, acc1);
            acc2 = _mm512_fmadd_ps(d2, d2, acc2);
            acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        }

        for (; i <= size - 16; i += 16)
        {
            __m512 va = _mm512_loadu_ps(&da[i]);
            __m512 vb = _mm512_loadu_ps(&db[i]);
            __m512 diff = _mm512_sub_ps(va, vb);
            acc0 = _mm512_fmadd_ps(diff, diff, acc0);
        }

        __m512 sum01 = _mm512_add_ps(acc0, acc1);
        __m512 sum23 = _mm512_add_ps(acc2, acc3);
        sum += _mm512_reduce_add_ps(_mm512_add_ps(sum01, sum23));
    }
#elif defined(__AVX__) && \
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
        }

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
        __m128 squad = _mm_add_ps(vsum, shuf);
        shuf = _mm_shuffle_ps(squad, squad, 1);
        sum += _mm_cvtss_f32(_mm_add_ss(squad, shuf));
    }
    else if (size >= 8)
    {
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i <= size - 8; i += 8)
        {
            __m256 va = _mm256_loadu_ps(&da[i]);
            __m256 vb = _mm256_loadu_ps(&db[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
#else
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
#endif
        }
        __m128 vlow = _mm256_castps256_ps128(sum_vec);
        __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehl_ps(vsum, vsum);
        __m128 squad = _mm_add_ps(vsum, shuf);
        shuf = _mm_shuffle_ps(squad, squad, 1);
        sum += _mm_cvtss_f32(_mm_add_ss(squad, shuf));
    }
#endif

    for (; i < size; i++)
    {
        float diff = da[i] - db[i];
        sum += diff * diff;
    }

    return (double)sqrtf(sum);
}

/**
 * framedist_double() - Vectorized double-precision Euclidean distance between pixel arrays.
 * @da:   Pointer to first pixel array.
 * @db:   Pointer to second pixel array.
 * @size: Number of elements in frame.
 *
 * Return: Euclidean L2 distance.
 */
double framedist_double(
    const double *restrict da,
    const double *restrict db,
    long                   size)
{
    double sum = 0.0;
    long i = 0;

#if defined(__AVX512F__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 32)
    {
        __m512d acc0 = _mm512_setzero_pd();
        __m512d acc1 = _mm512_setzero_pd();
        __m512d acc2 = _mm512_setzero_pd();
        __m512d acc3 = _mm512_setzero_pd();

        for (; i <= size - 32; i += 32)
        {
            __m512d va0 = _mm512_loadu_pd(&da[i]);
            __m512d vb0 = _mm512_loadu_pd(&db[i]);
            __m512d d0 = _mm512_sub_pd(va0, vb0);

            __m512d va1 = _mm512_loadu_pd(&da[i + 8]);
            __m512d vb1 = _mm512_loadu_pd(&db[i + 8]);
            __m512d d1 = _mm512_sub_pd(va1, vb1);

            __m512d va2 = _mm512_loadu_pd(&da[i + 16]);
            __m512d vb2 = _mm512_loadu_pd(&db[i + 16]);
            __m512d d2 = _mm512_sub_pd(va2, vb2);

            __m512d va3 = _mm512_loadu_pd(&da[i + 24]);
            __m512d vb3 = _mm512_loadu_pd(&db[i + 24]);
            __m512d d3 = _mm512_sub_pd(va3, vb3);

            acc0 = _mm512_fmadd_pd(d0, d0, acc0);
            acc1 = _mm512_fmadd_pd(d1, d1, acc1);
            acc2 = _mm512_fmadd_pd(d2, d2, acc2);
            acc3 = _mm512_fmadd_pd(d3, d3, acc3);
        }

        for (; i <= size - 8; i += 8)
        {
            __m512d va = _mm512_loadu_pd(&da[i]);
            __m512d vb = _mm512_loadu_pd(&db[i]);
            __m512d diff = _mm512_sub_pd(va, vb);
            acc0 = _mm512_fmadd_pd(diff, diff, acc0);
        }

        __m512d sum01 = _mm512_add_pd(acc0, acc1);
        __m512d sum23 = _mm512_add_pd(acc2, acc3);
        sum += _mm512_reduce_add_pd(_mm512_add_pd(sum01, sum23));
    }
#elif defined(__AVX__) && \
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
        }

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
    else if (size >= 4)
    {
        __m256d sum_vec = _mm256_setzero_pd();
        for (; i <= size - 4; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&da[i]);
            __m256d vb = _mm256_loadu_pd(&db[i]);
            __m256d diff = _mm256_sub_pd(va, vb);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);
#else
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
#endif
        }
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

    return sqrt(sum);
}

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

            if (cutoff_sq > 0.0 && ((i + 32) % 64 == 0 || i + 32 >= size))
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

            if (cutoff_sq > 0.0 && ((i + 16) % 32 == 0 || i + 16 >= size))
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

/**
 * framedist() - Computes the Euclidean distance between two frames.
 * @a: Pointer to the first Frame.
 * @b: Pointer to the second Frame.
 *
 * Checks that the frames have matching dimensions (width and height),
 * and then computes the L2 Euclidean distance between their pixel data,
 * dispatching to float or double precision according to frame precision.
 *
 * Return: The Euclidean distance, or -1.0 if the frame dimensions mismatch.
 */
double framedist(
    const Frame *a,
    const Frame *b)
{
    if (a->width != b->width || a->height != b->height)
    {
        return -1.0;
    }

    long size = a->width * a->height;

    if (a->is_double)
    {
        return framedist_double((const double *)a->data, (const double *)b->data, size);
    }

    return framedist_float((const float *)a->data, (const float *)b->data, size);
}

/**
 * framedist_batch_1x4_float() - Vectorized 1-query vs 4-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @out_dists: Array of 4 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x4_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    long i = 0;

    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];

#if defined(__SSE__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size == 3)
    {
        __m128 vqx = _mm_set1_ps(q[0]);
        __m128 vqy = _mm_set1_ps(q[1]);
        __m128 vqz = _mm_set1_ps(q[2]);
        __m128 va0 = _mm_set_ps(a3[0], a2[0], a1[0], a0[0]);
        __m128 va1 = _mm_set_ps(a3[1], a2[1], a1[1], a0[1]);
        __m128 va2 = _mm_set_ps(a3[2], a2[2], a1[2], a0[2]);
        __m128 dx = _mm_sub_ps(vqx, va0);
        __m128 dy = _mm_sub_ps(vqy, va1);
        __m128 dz = _mm_sub_ps(vqz, va2);
#ifdef __FMA__
        __m128 dsq = _mm_fmadd_ps(dz, dz, _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx)));
#else
        __m128 dsq = _mm_add_ps(_mm_mul_ps(dx, dx),
                                _mm_add_ps(_mm_mul_ps(dy, dy), _mm_mul_ps(dz, dz)));
#endif
        __m128 vd = _mm_sqrt_ps(dsq);
        out_dists[0] = (double)_mm_cvtss_f32(vd);
        out_dists[1] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(1, 1, 1, 1)));
        out_dists[2] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(2, 2, 2, 2)));
        out_dists[3] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(3, 3, 3, 3)));
        return;
    }
    if (size == 2)
    {
        __m128 vqx = _mm_set1_ps(q[0]);
        __m128 vqy = _mm_set1_ps(q[1]);
        __m128 va0 = _mm_set_ps(a3[0], a2[0], a1[0], a0[0]);
        __m128 va1 = _mm_set_ps(a3[1], a2[1], a1[1], a0[1]);
        __m128 dx = _mm_sub_ps(vqx, va0);
        __m128 dy = _mm_sub_ps(vqy, va1);
#ifdef __FMA__
        __m128 dsq = _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx));
#else
        __m128 dsq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
#endif
        __m128 vd = _mm_sqrt_ps(dsq);
        out_dists[0] = (double)_mm_cvtss_f32(vd);
        out_dists[1] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(1, 1, 1, 1)));
        out_dists[2] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(2, 2, 2, 2)));
        out_dists[3] = (double)_mm_cvtss_f32(_mm_shuffle_ps(vd, vd, _MM_SHUFFLE(3, 3, 3, 3)));
        return;
    }
#endif

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 16)
    {
        __m256 a0_0 = _mm256_setzero_ps();
        __m256 a0_1 = _mm256_setzero_ps();
        __m256 a1_0 = _mm256_setzero_ps();
        __m256 a1_1 = _mm256_setzero_ps();
        __m256 a2_0 = _mm256_setzero_ps();
        __m256 a2_1 = _mm256_setzero_ps();
        __m256 a3_0 = _mm256_setzero_ps();
        __m256 a3_1 = _mm256_setzero_ps();

        for (; i <= size - 16; i += 16)
        {
            /* Chunk 0 (8 floats) -> accumulates into *_0 */
            __m256 vq0 = _mm256_loadu_ps(&q[i]);
            __m256 d0_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a0[i]));
            __m256 d1_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a1[i]));
            __m256 d2_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a2[i]));
            __m256 d3_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_ps(d0_0, d0_0, a0_0);
            a1_0 = _mm256_fmadd_ps(d1_0, d1_0, a1_0);
            a2_0 = _mm256_fmadd_ps(d2_0, d2_0, a2_0);
            a3_0 = _mm256_fmadd_ps(d3_0, d3_0, a3_0);
#else
            a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(d0_0, d0_0));
            a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(d1_0, d1_0));
            a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(d2_0, d2_0));
            a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(d3_0, d3_0));
#endif

            /* Chunk 1 (8 floats) -> accumulates into *_1 */
            __m256 vq1 = _mm256_loadu_ps(&q[i + 8]);
            __m256 d0_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a0[i + 8]));
            __m256 d1_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a1[i + 8]));
            __m256 d2_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a2[i + 8]));
            __m256 d3_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a3[i + 8]));

#ifdef __FMA__
            a0_1 = _mm256_fmadd_ps(d0_1, d0_1, a0_1);
            a1_1 = _mm256_fmadd_ps(d1_1, d1_1, a1_1);
            a2_1 = _mm256_fmadd_ps(d2_1, d2_1, a2_1);
            a3_1 = _mm256_fmadd_ps(d3_1, d3_1, a3_1);
#else
            a0_1 = _mm256_add_ps(a0_1, _mm256_mul_ps(d0_1, d0_1));
            a1_1 = _mm256_add_ps(a1_1, _mm256_mul_ps(d1_1, d1_1));
            a2_1 = _mm256_add_ps(a2_1, _mm256_mul_ps(d2_1, d2_1));
            a3_1 = _mm256_add_ps(a3_1, _mm256_mul_ps(d3_1, d3_1));
#endif
        }

        for (; i <= size - 8; i += 8)
        {
            __m256 vq = _mm256_loadu_ps(&q[i]);
            __m256 d0 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a0[i]));
            __m256 d1 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a1[i]));
            __m256 d2 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a2[i]));
            __m256 d3 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_ps(d0, d0, a0_0);
            a1_0 = _mm256_fmadd_ps(d1, d1, a1_0);
            a2_0 = _mm256_fmadd_ps(d2, d2, a2_0);
            a3_0 = _mm256_fmadd_ps(d3, d3, a3_0);
#else
            a0_0 = _mm256_add_ps(a0_0, _mm256_mul_ps(d0, d0));
            a1_0 = _mm256_add_ps(a1_0, _mm256_mul_ps(d1, d1));
            a2_0 = _mm256_add_ps(a2_0, _mm256_mul_ps(d2, d2));
            a3_0 = _mm256_add_ps(a3_0, _mm256_mul_ps(d3, d3));
#endif
        }

        __m256 acc0 = _mm256_add_ps(a0_0, a0_1);
        __m256 acc1 = _mm256_add_ps(a1_0, a1_1);
        __m256 acc2 = _mm256_add_ps(a2_0, a2_1);
        __m256 acc3 = _mm256_add_ps(a3_0, a3_1);

        __m128 lo0 = _mm256_castps256_ps128(acc0);
        __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
        __m128 s0 = _mm_add_ps(lo0, hi0);

        __m128 lo1 = _mm256_castps256_ps128(acc1);
        __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1);

        __m128 lo2 = _mm256_castps256_ps128(acc2);
        __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 s2 = _mm_add_ps(lo2, hi2);

        __m128 lo3 = _mm256_castps256_ps128(acc3);
        __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
        __m128 s3 = _mm_add_ps(lo3, hi3);

        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        __m128 sum01 = _mm_add_ps(s0, s1);
        __m128 sum23 = _mm_add_ps(s2, s3);
        __m128 final_sums = _mm_add_ps(sum01, sum23);

        float fsums[4];
        _mm_storeu_ps(fsums, final_sums);
        sum0 += fsums[0];
        sum1 += fsums[1];
        sum2 += fsums[2];
        sum3 += fsums[3];
    }
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float diff0 = q_val - a0[i];
        float diff1 = q_val - a1[i];
        float diff2 = q_val - a2[i];
        float diff3 = q_val - a3[i];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    out_dists[0] = (double)sqrtf(sum0);
    out_dists[1] = (double)sqrtf(sum1);
    out_dists[2] = (double)sqrtf(sum2);
    out_dists[3] = (double)sqrtf(sum3);
}

/**
 * framedist_batch_1x8_float() - Vectorized 1-query vs 8-anchor Euclidean distance (single).
 * @q:         Pointer to query array.
 * @anchors:   Array of 8 pointers to candidate anchor arrays.
 * @out_dists: Array of 8 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x8_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    double *restrict             out_dists,
    long                         size)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    long i = 0;

    const float *restrict a0 = anchors[0];
    const float *restrict a1 = anchors[1];
    const float *restrict a2 = anchors[2];
    const float *restrict a3 = anchors[3];
    const float *restrict a4 = anchors[4];
    const float *restrict a5 = anchors[5];
    const float *restrict a6 = anchors[6];
    const float *restrict a7 = anchors[7];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 8)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        for (; i <= size - 8; i += 8)
        {
            __m256 vq = _mm256_loadu_ps(&q[i]);
            __m256 d0 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a0[i]));
            __m256 d1 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a1[i]));
            __m256 d2 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a2[i]));
            __m256 d3 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a3[i]));
            __m256 d4 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a4[i]));
            __m256 d5 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a5[i]));
            __m256 d6 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a6[i]));
            __m256 d7 = _mm256_sub_ps(vq, _mm256_loadu_ps(&a7[i]));

#ifdef __FMA__
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            acc4 = _mm256_fmadd_ps(d4, d4, acc4);
            acc5 = _mm256_fmadd_ps(d5, d5, acc5);
            acc6 = _mm256_fmadd_ps(d6, d6, acc6);
            acc7 = _mm256_fmadd_ps(d7, d7, acc7);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(d4, d4));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(d5, d5));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(d6, d6));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(d7, d7));
#endif
        }

        __m128 s0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
        __m128 s1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
        __m128 s2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
        __m128 s3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
        __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc4), _mm256_extractf128_ps(acc4, 1));
        __m128 s5 = _mm_add_ps(_mm256_castps256_ps128(acc5), _mm256_extractf128_ps(acc5, 1));
        __m128 s6 = _mm_add_ps(_mm256_castps256_ps128(acc6), _mm256_extractf128_ps(acc6, 1));
        __m128 s7 = _mm_add_ps(_mm256_castps256_ps128(acc7), _mm256_extractf128_ps(acc7, 1));

        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        _MM_TRANSPOSE4_PS(s4, s5, s6, s7);

        __m128 f0123 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
        __m128 f4567 = _mm_add_ps(_mm_add_ps(s4, s5), _mm_add_ps(s6, s7));

        float f03[4];
        float f47[4];
        _mm_storeu_ps(f03, f0123);
        _mm_storeu_ps(f47, f4567);

        sum0 += f03[0];
        sum1 += f03[1];
        sum2 += f03[2];
        sum3 += f03[3];
        sum4 += f47[0];
        sum5 += f47[1];
        sum6 += f47[2];
        sum7 += f47[3];
    }
#endif

    for (; i < size; i++)
    {
        float q_val = q[i];
        float d0 = q_val - a0[i];
        float d1 = q_val - a1[i];
        float d2 = q_val - a2[i];
        float d3 = q_val - a3[i];
        float d4 = q_val - a4[i];
        float d5 = q_val - a5[i];
        float d6 = q_val - a6[i];
        float d7 = q_val - a7[i];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
    }

    out_dists[0] = (double)sqrtf(sum0);
    out_dists[1] = (double)sqrtf(sum1);
    out_dists[2] = (double)sqrtf(sum2);
    out_dists[3] = (double)sqrtf(sum3);
    out_dists[4] = (double)sqrtf(sum4);
    out_dists[5] = (double)sqrtf(sum5);
    out_dists[6] = (double)sqrtf(sum6);
    out_dists[7] = (double)sqrtf(sum7);
}

/**
 * framedist_batch_1x4_double() - Vectorized 1-query vs 4-anchor Euclidean distance (double).
 * @q:         Pointer to query array.
 * @anchors:   Array of 4 pointers to candidate anchor arrays.
 * @out_dists: Array of 4 doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_1x4_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    double *restrict              out_dists,
    long                          size)
{
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    long i = 0;

    const double *restrict a0 = anchors[0];
    const double *restrict a1 = anchors[1];
    const double *restrict a2 = anchors[2];
    const double *restrict a3 = anchors[3];

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 8)
    {
        __m256d a0_0 = _mm256_setzero_pd();
        __m256d a0_1 = _mm256_setzero_pd();
        __m256d a1_0 = _mm256_setzero_pd();
        __m256d a1_1 = _mm256_setzero_pd();
        __m256d a2_0 = _mm256_setzero_pd();
        __m256d a2_1 = _mm256_setzero_pd();
        __m256d a3_0 = _mm256_setzero_pd();
        __m256d a3_1 = _mm256_setzero_pd();

        for (; i <= size - 8; i += 8)
        {
            /* Chunk 0 (4 doubles) */
            __m256d vq0 = _mm256_loadu_pd(&q[i]);
            __m256d d0_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a0[i]));
            __m256d d1_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a1[i]));
            __m256d d2_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a2[i]));
            __m256d d3_0 = _mm256_sub_pd(vq0, _mm256_loadu_pd(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_pd(d0_0, d0_0, a0_0);
            a1_0 = _mm256_fmadd_pd(d1_0, d1_0, a1_0);
            a2_0 = _mm256_fmadd_pd(d2_0, d2_0, a2_0);
            a3_0 = _mm256_fmadd_pd(d3_0, d3_0, a3_0);
#else
            a0_0 = _mm256_add_pd(a0_0, _mm256_mul_pd(d0_0, d0_0));
            a1_0 = _mm256_add_pd(a1_0, _mm256_mul_pd(d1_0, d1_0));
            a2_0 = _mm256_add_pd(a2_0, _mm256_mul_pd(d2_0, d2_0));
            a3_0 = _mm256_add_pd(a3_0, _mm256_mul_pd(d3_0, d3_0));
#endif

            /* Chunk 1 (4 doubles) */
            __m256d vq1 = _mm256_loadu_pd(&q[i + 4]);
            __m256d d0_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a0[i + 4]));
            __m256d d1_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a1[i + 4]));
            __m256d d2_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a2[i + 4]));
            __m256d d3_1 = _mm256_sub_pd(vq1, _mm256_loadu_pd(&a3[i + 4]));

#ifdef __FMA__
            a0_1 = _mm256_fmadd_pd(d0_1, d0_1, a0_1);
            a1_1 = _mm256_fmadd_pd(d1_1, d1_1, a1_1);
            a2_1 = _mm256_fmadd_pd(d2_1, d2_1, a2_1);
            a3_1 = _mm256_fmadd_pd(d3_1, d3_1, a3_1);
#else
            a0_1 = _mm256_add_pd(a0_1, _mm256_mul_pd(d0_1, d0_1));
            a1_1 = _mm256_add_pd(a1_1, _mm256_mul_pd(d1_1, d1_1));
            a2_1 = _mm256_add_pd(a2_1, _mm256_mul_pd(d2_1, d2_1));
            a3_1 = _mm256_add_pd(a3_1, _mm256_mul_pd(d3_1, d3_1));
#endif
        }

        for (; i <= size - 4; i += 4)
        {
            __m256d vq = _mm256_loadu_pd(&q[i]);
            __m256d d0 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a0[i]));
            __m256d d1 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a1[i]));
            __m256d d2 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a2[i]));
            __m256d d3 = _mm256_sub_pd(vq, _mm256_loadu_pd(&a3[i]));

#ifdef __FMA__
            a0_0 = _mm256_fmadd_pd(d0, d0, a0_0);
            a1_0 = _mm256_fmadd_pd(d1, d1, a1_0);
            a2_0 = _mm256_fmadd_pd(d2, d2, a2_0);
            a3_0 = _mm256_fmadd_pd(d3, d3, a3_0);
#else
            a0_0 = _mm256_add_pd(a0_0, _mm256_mul_pd(d0, d0));
            a1_0 = _mm256_add_pd(a1_0, _mm256_mul_pd(d1, d1));
            a2_0 = _mm256_add_pd(a2_0, _mm256_mul_pd(d2, d2));
            a3_0 = _mm256_add_pd(a3_0, _mm256_mul_pd(d3, d3));
#endif
        }

        __m256d acc0 = _mm256_add_pd(a0_0, a0_1);
        __m256d acc1 = _mm256_add_pd(a1_0, a1_1);
        __m256d acc2 = _mm256_add_pd(a2_0, a2_1);
        __m256d acc3 = _mm256_add_pd(a3_0, a3_1);

        __m128d lo0 = _mm256_castpd256_pd128(acc0);
        __m128d hi0 = _mm256_extractf128_pd(acc0, 1);
        __m128d s0 = _mm_add_pd(lo0, hi0);

        __m128d lo1 = _mm256_castpd256_pd128(acc1);
        __m128d hi1 = _mm256_extractf128_pd(acc1, 1);
        __m128d s1 = _mm_add_pd(lo1, hi1);

        __m128d lo2 = _mm256_castpd256_pd128(acc2);
        __m128d hi2 = _mm256_extractf128_pd(acc2, 1);
        __m128d s2 = _mm_add_pd(lo2, hi2);

        __m128d lo3 = _mm256_castpd256_pd128(acc3);
        __m128d hi3 = _mm256_extractf128_pd(acc3, 1);
        __m128d s3 = _mm_add_pd(lo3, hi3);

        sum0 += _mm_cvtsd_f64(_mm_add_sd(s0, _mm_unpackhi_pd(s0, s0)));
        sum1 += _mm_cvtsd_f64(_mm_add_sd(s1, _mm_unpackhi_pd(s1, s1)));
        sum2 += _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
        sum3 += _mm_cvtsd_f64(_mm_add_sd(s3, _mm_unpackhi_pd(s3, s3)));
    }
#endif

    for (; i < size; i++)
    {
        double q_val = q[i];
        double diff0 = q_val - a0[i];
        double diff1 = q_val - a1[i];
        double diff2 = q_val - a2[i];
        double diff3 = q_val - a3[i];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    out_dists[0] = sqrt(sum0);
    out_dists[1] = sqrt(sum1);
    out_dists[2] = sqrt(sum2);
    out_dists[3] = sqrt(sum3);
}

/**
 * framedist_batch_float() - Batch Euclidean distance from query to N candidate anchors.
 * @q:         Pointer to query array.
 * @anchors:   Array of N pointers to candidate anchor arrays.
 * @n_anchors: Number of candidates in this batch.
 * @out_dists: Array of N doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_float(
    const float *restrict        q,
    const float *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size)
{
    int k = 0;
    while (k + 8 <= n_anchors)
    {
        framedist_batch_1x8_float(q, &anchors[k], &out_dists[k], size);
        k += 8;
    }

    while (k + 4 <= n_anchors)
    {
        framedist_batch_1x4_float(q, &anchors[k], &out_dists[k], size);
        k += 4;
    }

    for (; k < n_anchors; k++)
    {
        out_dists[k] = framedist_float(q, anchors[k], size);
    }
}

/**
 * framedist_batch_double() - Batch Euclidean distance from query to N candidate anchors.
 * @q:         Pointer to query array.
 * @anchors:   Array of N pointers to candidate anchor arrays.
 * @n_anchors: Number of candidates in this batch.
 * @out_dists: Array of N doubles to receive computed distances.
 * @size:      Number of elements in each array.
 */
void framedist_batch_double(
    const double *restrict        q,
    const double *const *restrict anchors,
    int                          n_anchors,
    double *restrict             out_dists,
    long                         size)
{
    int k = 0;
    while (k + 4 <= n_anchors)
    {
        framedist_batch_1x4_double(q, &anchors[k], &out_dists[k], size);
        k += 4;
    }

    for (; k < n_anchors; k++)
    {
        out_dists[k] = framedist_double(q, anchors[k], size);
    }
}

/**
 * framedist_batch() - Computes Euclidean distances from query Frame to N candidate Frames.
 * @q:         Pointer to query Frame.
 * @anchors:   Array of pointers to candidate Frames.
 * @n_anchors: Number of candidate Frames.
 * @out_dists: Array of N doubles to receive computed distances.
 */
void framedist_batch(
    const Frame  *q,
    const Frame **anchors,
    int           n_anchors,
    double       *out_dists)
{
    if (n_anchors <= 0)
    {
        return;
    }

    long size = q->width * q->height;
    const void *ptrs_stack[64];
    const void **ptrs = ptrs_stack;

    if (n_anchors > 64)
    {
        ptrs = (const void **)malloc((size_t)n_anchors * sizeof(const void *));
        if (!ptrs)
        {
            return;
        }
    }

    for (int i = 0; i < n_anchors; i++)
    {
        ptrs[i] = anchors[i]->data;
    }

    if (q->is_double)
    {
        framedist_batch_double(
            (const double *)q->data,
            (const double *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }
    else
    {
        framedist_batch_float(
            (const float *)q->data,
            (const float *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }

    if (ptrs != ptrs_stack)
    {
        free(ptrs);
    }
}


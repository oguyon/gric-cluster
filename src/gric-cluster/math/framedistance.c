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
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        sum += _mm_cvtss_f32(vsum);
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
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        sum += _mm_cvtss_f32(vsum);
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
        vsum = _mm_hadd_pd(vsum, vsum);
        sum += _mm_cvtsd_f64(vsum);
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
        vsum = _mm_hadd_pd(vsum, vsum);
        sum += _mm_cvtsd_f64(vsum);
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

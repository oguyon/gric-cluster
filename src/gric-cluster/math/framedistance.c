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
    if (size >= 8)
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
    if (size >= 4)
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
        __m256d hsum = _mm256_hadd_pd(sum_vec, sum_vec);
        sum += ((double *)&hsum)[0] + ((double *)&hsum)[2];
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

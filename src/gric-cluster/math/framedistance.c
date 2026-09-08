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
                vsum = _mm_hadd_pd(vsum, vsum);
                double partial = _mm_cvtsd_f64(vsum);
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
        vsum = _mm_hadd_pd(vsum, vsum);
        sum += _mm_cvtsd_f64(vsum);
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

        __m128 h01 = _mm_hadd_ps(s0, s1);
        __m128 h23 = _mm_hadd_ps(s2, s3);
        __m128 final_sums = _mm_hadd_ps(h01, h23);

        sum0 += _mm_cvtss_f32(final_sums);
        sum1 += _mm_cvtss_f32(_mm_shuffle_ps(final_sums, final_sums, _MM_SHUFFLE(1, 1, 1, 1)));
        sum2 += _mm_cvtss_f32(_mm_shuffle_ps(final_sums, final_sums, _MM_SHUFFLE(2, 2, 2, 2)));
        sum3 += _mm_cvtss_f32(_mm_shuffle_ps(final_sums, final_sums, _MM_SHUFFLE(3, 3, 3, 3)));
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

        __m128d h0 = _mm_hadd_pd(s0, s0);
        __m128d h1 = _mm_hadd_pd(s1, s1);
        __m128d h2 = _mm_hadd_pd(s2, s2);
        __m128d h3 = _mm_hadd_pd(s3, s3);

        sum0 += _mm_cvtsd_f64(h0);
        sum1 += _mm_cvtsd_f64(h1);
        sum2 += _mm_cvtsd_f64(h2);
        sum3 += _mm_cvtsd_f64(h3);
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
    const void *ptrs_stack[16];
    const void **ptrs = ptrs_stack;

    if (n_anchors > 16)
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


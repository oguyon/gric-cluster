/**
 * @file bench_simd_scaling.c
 * @brief Measures performance scaling from Scalar to AVX2 and AVX-512 across core kernels.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "gric_simd.h"

static inline double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------- */
/* Kernel 1: Single-Vector L2 Distance (Float32)                             */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static float l2_dist_scalar(
    const float *restrict a,
    const float *restrict b,
    int                   dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static float l2_dist_avx2(
    const float *restrict a,
    const float *restrict b,
    int                   dim)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i <= dim - 8; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    __m256 sum256 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    float sum = _mm_cvtss_f32(sum128);
    for (; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

GRIC_TARGET_AVX512
static float l2_dist_avx512(
    const float *restrict a,
    const float *restrict b,
    int                   dim)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i <= dim - 32; i += 32)
    {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 d0  = _mm512_sub_ps(va0, vb0);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);

        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        __m512 d1  = _mm512_sub_ps(va1, vb1);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    }
    for (; i <= dim - 16; i += 16)
    {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d  = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 2: 8-Anchor Batch Distance (Float32)                               */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static void dist8_scalar(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    for (int a = 0; a < 8; a++)
    {
        float sum = 0.0f;
        const float *anc = anchors[a];
        for (int i = 0; i < dim; i++)
        {
            float d = q[i] - anc[i];
            sum += d * d;
        }
        out[a] = sum;
    }
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static void dist8_avx2(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    __m256 acc[8];
    for (int a = 0; a < 8; a++)
    {
        acc[a] = _mm256_setzero_ps();
    }
    int i = 0;
    for (; i <= dim - 8; i += 8)
    {
        __m256 vq = _mm256_loadu_ps(q + i);
        for (int a = 0; a < 8; a++)
        {
            __m256 va = _mm256_loadu_ps(anchors[a] + i);
            __m256 d  = _mm256_sub_ps(vq, va);
            acc[a] = _mm256_fmadd_ps(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 8; a++)
    {
        __m128 lo = _mm256_castps256_ps128(acc[a]);
        __m128 hi = _mm256_extractf128_ps(acc[a], 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
        float sum = _mm_cvtss_f32(sum128);
        for (int rem = i; rem < dim; rem++)
        {
            float d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}

GRIC_TARGET_AVX512
static void dist8_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    __m512 acc[8];
    for (int a = 0; a < 8; a++)
    {
        acc[a] = _mm512_setzero_ps();
    }
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m512 vq = _mm512_loadu_ps(q + i);
        for (int a = 0; a < 8; a++)
        {
            __m512 va = _mm512_loadu_ps(anchors[a] + i);
            __m512 d  = _mm512_sub_ps(vq, va);
            acc[a] = _mm512_fmadd_ps(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 8; a++)
    {
        float sum = _mm512_reduce_add_ps(acc[a]);
        for (int rem = i; rem < dim; rem++)
        {
            float d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 3: SQ8 Quantized L2 Distance (uint8)                               */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static uint32_t sq8_dist_scalar(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    int                     dim)
{
    uint32_t sum = 0;
    for (int i = 0; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint32_t)(diff * diff);
    }
    return sum;
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static uint32_t sq8_dist_avx2(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    int                     dim)
{
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i <= dim - 32; i += 32)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i max_v = _mm256_max_epu8(va, vb);
        __m256i min_v = _mm256_min_epu8(va, vb);
        __m256i diff = _mm256_sub_epi8(max_v, min_v);
        __m256i diff_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diff));
        __m256i diff_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(diff, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_lo, diff_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_hi, diff_hi));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
    uint32_t sum = (uint32_t)_mm_cvtsi128_si32(sum128);
    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint32_t)(diff * diff);
    }
    return sum;
}

GRIC_TARGET_AVX512
static uint32_t sq8_dist_avx512(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    int                     dim)
{
    __m512i acc = _mm512_setzero_si512();
    int i = 0;
    for (; i <= dim - 64; i += 64)
    {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        __m512i max_v = _mm512_max_epu8(va, vb);
        __m512i min_v = _mm512_min_epu8(va, vb);
        __m512i diff = _mm512_sub_epi8(max_v, min_v);
        __m512i diff_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(diff));
        __m512i diff_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(diff, 1));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff_lo, diff_lo));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff_hi, diff_hi));
    }
    for (; i <= dim - 32; i += 32)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i max_v = _mm256_max_epu8(va, vb);
        __m256i min_v = _mm256_min_epu8(va, vb);
        __m256i diff = _mm256_sub_epi8(max_v, min_v);
        __m512i diff16 = _mm512_cvtepu8_epi16(diff);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff16, diff16));
    }
    uint32_t sum = (uint32_t)_mm512_reduce_add_epi32(acc);
    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint32_t)(diff * diff);
    }
    return sum;
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 4: Single-Vector L2 Distance (Float64)                             */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static double l2_d64_scalar(
    const double *restrict a,
    const double *restrict b,
    int                    dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static double l2_d64_avx2(
    const double *restrict a,
    const double *restrict b,
    int                    dim)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= dim - 8; i += 8)
    {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d d0  = _mm256_sub_pd(va0, vb0);
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);

        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d d1  = _mm256_sub_pd(va1, vb1);
        acc1 = _mm256_fmadd_pd(d1, d1, acc1);
    }
    for (; i <= dim - 4; i += 4)
    {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d d  = _mm256_sub_pd(va, vb);
        acc0 = _mm256_fmadd_pd(d, d, acc0);
    }
    __m256d sum256 = _mm256_add_pd(acc0, acc1);
    __m128d lo = _mm256_castpd256_pd128(sum256);
    __m128d hi = _mm256_extractf128_pd(sum256, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    double sum = _mm_cvtsd_f64(_mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128)));
    for (; i < dim; i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

GRIC_TARGET_AVX512
static double l2_d64_avx512(
    const double *restrict a,
    const double *restrict b,
    int                    dim)
{
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m512d va0 = _mm512_loadu_pd(a + i);
        __m512d vb0 = _mm512_loadu_pd(b + i);
        __m512d d0  = _mm512_sub_pd(va0, vb0);
        acc0 = _mm512_fmadd_pd(d0, d0, acc0);

        __m512d va1 = _mm512_loadu_pd(a + i + 8);
        __m512d vb1 = _mm512_loadu_pd(b + i + 8);
        __m512d d1  = _mm512_sub_pd(va1, vb1);
        acc1 = _mm512_fmadd_pd(d1, d1, acc1);
    }
    for (; i <= dim - 8; i += 8)
    {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d d  = _mm512_sub_pd(va, vb);
        acc0 = _mm512_fmadd_pd(d, d, acc0);
    }
    double sum = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
    for (; i < dim; i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 5: 16-Anchor Batch Distance (Float32)                              */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static void dist16_f32_scalar(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    for (int a = 0; a < 16; a++)
    {
        float sum = 0.0f;
        const float *anc = anchors[a];
        for (int i = 0; i < dim; i++)
        {
            float d = q[i] - anc[i];
            sum += d * d;
        }
        out[a] = sum;
    }
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static void dist16_f32_avx2(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    __m256 acc[16];
    for (int a = 0; a < 16; a++)
    {
        acc[a] = _mm256_setzero_ps();
    }
    int i = 0;
    for (; i <= dim - 8; i += 8)
    {
        __m256 vq = _mm256_loadu_ps(q + i);
        for (int a = 0; a < 16; a++)
        {
            __m256 va = _mm256_loadu_ps(anchors[a] + i);
            __m256 d  = _mm256_sub_ps(vq, va);
            acc[a] = _mm256_fmadd_ps(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 16; a++)
    {
        __m128 lo = _mm256_castps256_ps128(acc[a]);
        __m128 hi = _mm256_extractf128_ps(acc[a], 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
        float sum = _mm_cvtss_f32(sum128);
        for (int rem = i; rem < dim; rem++)
        {
            float d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}

GRIC_TARGET_AVX512
static void dist16_f32_avx512(
    const float *restrict        q,
    const float *const *restrict anchors,
    float       *restrict        out,
    int                          dim)
{
    __m512 acc[16];
    for (int a = 0; a < 16; a++)
    {
        acc[a] = _mm512_setzero_ps();
    }
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m512 vq = _mm512_loadu_ps(q + i);
        for (int a = 0; a < 16; a++)
        {
            __m512 va = _mm512_loadu_ps(anchors[a] + i);
            __m512 d  = _mm512_sub_ps(vq, va);
            acc[a] = _mm512_fmadd_ps(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 16; a++)
    {
        float sum = _mm512_reduce_add_ps(acc[a]);
        for (int rem = i; rem < dim; rem++)
        {
            float d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 6: 8-Anchor Batch Distance (Float64)                               */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static void dist8_d64_scalar(
    const double *restrict        q,
    const double *const *restrict anchors,
    double       *restrict        out,
    int                           dim)
{
    for (int a = 0; a < 8; a++)
    {
        double sum = 0.0;
        const double *anc = anchors[a];
        for (int i = 0; i < dim; i++)
        {
            double d = q[i] - anc[i];
            sum += d * d;
        }
        out[a] = sum;
    }
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static void dist8_d64_avx2(
    const double *restrict        q,
    const double *const *restrict anchors,
    double       *restrict        out,
    int                           dim)
{
    __m256d acc[8];
    for (int a = 0; a < 8; a++)
    {
        acc[a] = _mm256_setzero_pd();
    }
    int i = 0;
    for (; i <= dim - 4; i += 4)
    {
        __m256d vq = _mm256_loadu_pd(q + i);
        for (int a = 0; a < 8; a++)
        {
            __m256d va = _mm256_loadu_pd(anchors[a] + i);
            __m256d d  = _mm256_sub_pd(vq, va);
            acc[a] = _mm256_fmadd_pd(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 8; a++)
    {
        __m128d lo = _mm256_castpd256_pd128(acc[a]);
        __m128d hi = _mm256_extractf128_pd(acc[a], 1);
        __m128d sum128 = _mm_add_pd(lo, hi);
        double sum = _mm_cvtsd_f64(_mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128)));
        for (int rem = i; rem < dim; rem++)
        {
            double d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}

GRIC_TARGET_AVX512
static void dist8_d64_avx512(
    const double *restrict        q,
    const double *const *restrict anchors,
    double       *restrict        out,
    int                           dim)
{
    __m512d acc[8];
    for (int a = 0; a < 8; a++)
    {
        acc[a] = _mm512_setzero_pd();
    }
    int i = 0;
    for (; i <= dim - 8; i += 8)
    {
        __m512d vq = _mm512_loadu_pd(q + i);
        for (int a = 0; a < 8; a++)
        {
            __m512d va = _mm512_loadu_pd(anchors[a] + i);
            __m512d d  = _mm512_sub_pd(vq, va);
            acc[a] = _mm512_fmadd_pd(d, d, acc[a]);
        }
    }
    for (int a = 0; a < 8; a++)
    {
        double sum = _mm512_reduce_add_pd(acc[a]);
        for (int rem = i; rem < dim; rem++)
        {
            double d = q[rem] - anchors[a][rem];
            sum += d * d;
        }
        out[a] = sum;
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 7: SQ16 Quantized L2 Distance (int16)                              */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static uint64_t sq16_dist_scalar(
    const int16_t *restrict a,
    const int16_t *restrict b,
    int                     dim)
{
    uint64_t sum = 0;
    for (int i = 0; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint64_t)((int64_t)diff * diff);
    }
    return sum;
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static uint64_t sq16_dist_avx2(
    const int16_t *restrict a,
    const int16_t *restrict b,
    int                     dim)
{
    __m256i acc64_0 = _mm256_setzero_si256();
    __m256i acc64_1 = _mm256_setzero_si256();
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i diff = _mm256_sub_epi16(va, vb);
        __m256i sq32 = _mm256_madd_epi16(diff, diff);
        __m128i sq_lo = _mm256_castsi256_si128(sq32);
        __m128i sq_hi = _mm256_extracti128_si256(sq32, 1);
        acc64_0 = _mm256_add_epi64(acc64_0, _mm256_cvtepu32_epi64(sq_lo));
        acc64_1 = _mm256_add_epi64(acc64_1, _mm256_cvtepu32_epi64(sq_hi));
    }
    __m256i acc_tot = _mm256_add_epi64(acc64_0, acc64_1);
    __m128i lo = _mm256_castsi256_si128(acc_tot);
    __m128i hi = _mm256_extracti128_si256(acc_tot, 1);
    __m128i sum128 = _mm_add_epi64(lo, hi);
    uint64_t sum = (uint64_t)_mm_cvtsi128_si64(sum128) +
                   (uint64_t)_mm_extract_epi64(sum128, 1);
    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint64_t)((int64_t)diff * diff);
    }
    return sum;
}

GRIC_TARGET_AVX512
static uint64_t sq16_dist_avx512(
    const int16_t *restrict a,
    const int16_t *restrict b,
    int                     dim)
{
    __m512i acc64 = _mm512_setzero_si512();
    int i = 0;
    for (; i <= dim - 32; i += 32)
    {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        __m512i diff = _mm512_sub_epi16(va, vb);
        __m512i sq32 = _mm512_madd_epi16(diff, diff);
        __m256i sq_lo = _mm512_castsi512_si256(sq32);
        __m256i sq_hi = _mm512_extracti64x4_epi64(sq32, 1);
        acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(sq_lo));
        acc64 = _mm512_add_epi64(acc64, _mm512_cvtepu32_epi64(sq_hi));
    }
    uint64_t sum = (uint64_t)_mm512_reduce_add_epi64(acc64);
    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint64_t)((int64_t)diff * diff);
    }
    return sum;
}
#endif

/* ------------------------------------------------------------------------- */
/* Kernel 8: PQ FastScan (32-way vs 64-way)                                  */
/* ------------------------------------------------------------------------- */

__attribute__((optimize("no-tree-vectorize")))
static void fastscan_scalar(
    const uint8_t *restrict codes,
    const uint8_t *restrict lut,
    uint16_t      *restrict out_dists,
    int                     num_codes)
{
    for (int i = 0; i < num_codes; i++)
    {
        out_dists[i] += (uint16_t)lut[codes[i] & 0x0F];
    }
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX2
static void fastscan_avx2(
    const uint8_t *restrict codes,
    const uint8_t *restrict lut,
    uint16_t      *restrict out_dists,
    int                     num_codes)
{
    __m128i lut128 = _mm_loadu_si128((const __m128i *)lut);
    __m256i vlut = _mm256_broadcastsi128_si256(lut128);
    __m256i vmask = _mm256_set1_epi8(0x0F);
    int i = 0;
    for (; i <= num_codes - 32; i += 32)
    {
        __m256i raw = _mm256_loadu_si256((const __m256i *)(codes + i));
        __m256i idx = _mm256_and_si256(raw, vmask);
        __m256i looked_up = _mm256_shuffle_epi8(vlut, idx);
        __m128i lo8 = _mm256_castsi256_si128(looked_up);
        __m128i hi8 = _mm256_extracti128_si256(looked_up, 1);
        __m256i val16_lo = _mm256_cvtepu8_epi16(lo8);
        __m256i val16_hi = _mm256_cvtepu8_epi16(hi8);
        __m256i curr_lo = _mm256_loadu_si256((__m256i *)(out_dists + i));
        __m256i curr_hi = _mm256_loadu_si256((__m256i *)(out_dists + i + 16));
        _mm256_storeu_si256((__m256i *)(out_dists + i), _mm256_add_epi16(curr_lo, val16_lo));
        _mm256_storeu_si256((__m256i *)(out_dists + i + 16),
                            _mm256_add_epi16(curr_hi, val16_hi));
    }
    for (; i < num_codes; i++)
    {
        out_dists[i] += (uint16_t)lut[codes[i] & 0x0F];
    }
}

GRIC_TARGET_AVX512
static void fastscan_avx512(
    const uint8_t *restrict codes,
    const uint8_t *restrict lut,
    uint16_t      *restrict out_dists,
    int                     num_codes)
{
    __m128i lut128 = _mm_loadu_si128((const __m128i *)lut);
    __m512i vlut = _mm512_broadcast_i32x4(lut128);
    __m512i vmask = _mm512_set1_epi8(0x0F);
    int i = 0;
    for (; i <= num_codes - 64; i += 64)
    {
        __m512i raw = _mm512_loadu_si512((const void *)(codes + i));
        __m512i idx = _mm512_and_si512(raw, vmask);
        __m512i looked_up = _mm512_shuffle_epi8(vlut, idx);
        __m256i lo8 = _mm512_castsi512_si256(looked_up);
        __m256i hi8 = _mm512_extracti64x4_epi64(looked_up, 1);
        __m512i val16_lo = _mm512_cvtepu8_epi16(lo8);
        __m512i val16_hi = _mm512_cvtepu8_epi16(hi8);
        __m512i curr_lo = _mm512_loadu_si512((void *)(out_dists + i));
        __m512i curr_hi = _mm512_loadu_si512((void *)(out_dists + i + 32));
        _mm512_storeu_si512((void *)(out_dists + i), _mm512_add_epi16(curr_lo, val16_lo));
        _mm512_storeu_si512((void *)(out_dists + i + 32),
                            _mm512_add_epi16(curr_hi, val16_hi));
    }
    for (; i < num_codes; i++)
    {
        out_dists[i] += (uint16_t)lut[codes[i] & 0x0F];
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Benchmark Runner & Pretty Printer                                         */
/* ------------------------------------------------------------------------- */

typedef struct
{
    const char *kernel_name;
    int         dim;
    double      time_scalar;
    double      time_avx2;
    double      time_avx512;
    double      gain_avx2;
    double      gain_avx512_vs_avx2;
    double      gain_total;
    bool        has_avx512;
} SimdBenchResult;

static void run_l2_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    float *a = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
    float *b = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++)
    {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    res->kernel_name = "Float32 L2 Distance";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    /* Scalar */
    volatile float sink = 0.0f;
    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += l2_dist_scalar(a, b, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    /* AVX2 */
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += l2_dist_avx2(a, b, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    /* AVX-512 */
    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            sink += l2_dist_avx512(a, b, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(a);
    free(b);
}

static void run_dist8_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    float *q = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
    float *anchors[8];
    for (int a = 0; a < 8; a++)
    {
        anchors[a] = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
        for (int d = 0; d < dim; d++)
        {
            anchors[a][d] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int d = 0; d < dim; d++)
    {
        q[d] = (float)rand() / (float)RAND_MAX;
    }

    float out[8];
    res->kernel_name = "8-Anchor Batch Dist";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    /* Scalar */
    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist8_scalar(q, (const float *const *)anchors, out, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    /* AVX2 */
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist8_avx2(q, (const float *const *)anchors, out, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    /* AVX-512 */
    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            dist8_avx512(q, (const float *const *)anchors, out, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(q);
    for (int a = 0; a < 8; a++)
    {
        free(anchors[a]);
    }
}

static void run_sq8_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    uint8_t *a = (uint8_t *)aligned_alloc(64, (size_t)dim);
    uint8_t *b = (uint8_t *)aligned_alloc(64, (size_t)dim);
    for (int i = 0; i < dim; i++)
    {
        a[i] = (uint8_t)(rand() % 256);
        b[i] = (uint8_t)(rand() % 256);
    }

    res->kernel_name = "SQ8 Quantized L2";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    volatile uint32_t sink = 0;
    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += sq8_dist_scalar(a, b, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += sq8_dist_avx2(a, b, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            sink += sq8_dist_avx512(a, b, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(a);
    free(b);
}

static void run_l2_d64_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    double *a = (double *)aligned_alloc(64, (size_t)dim * sizeof(double));
    double *b = (double *)aligned_alloc(64, (size_t)dim * sizeof(double));
    for (int i = 0; i < dim; i++)
    {
        a[i] = (double)rand() / (double)RAND_MAX;
        b[i] = (double)rand() / (double)RAND_MAX;
    }

    res->kernel_name = "Float64 L2 Distance";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    volatile double sink = 0.0;
    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += l2_d64_scalar(a, b, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += l2_d64_avx2(a, b, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            sink += l2_d64_avx512(a, b, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(a);
    free(b);
}

static void run_dist16_f32_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    float *q = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
    float *anchors[16];
    for (int a = 0; a < 16; a++)
    {
        anchors[a] = (float *)aligned_alloc(64, (size_t)dim * sizeof(float));
        for (int d = 0; d < dim; d++)
        {
            anchors[a][d] = (float)rand() / (float)RAND_MAX;
        }
    }
    for (int d = 0; d < dim; d++)
    {
        q[d] = (float)rand() / (float)RAND_MAX;
    }

    float out[16];
    res->kernel_name = "16-Anchor Float Batch";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist16_f32_scalar(q, (const float *const *)anchors, out, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist16_f32_avx2(q, (const float *const *)anchors, out, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            dist16_f32_avx512(q, (const float *const *)anchors, out, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(q);
    for (int a = 0; a < 16; a++)
    {
        free(anchors[a]);
    }
}

static void run_dist8_d64_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    double *q = (double *)aligned_alloc(64, (size_t)dim * sizeof(double));
    double *anchors[8];
    for (int a = 0; a < 8; a++)
    {
        anchors[a] = (double *)aligned_alloc(64, (size_t)dim * sizeof(double));
        for (int d = 0; d < dim; d++)
        {
            anchors[a][d] = (double)rand() / (double)RAND_MAX;
        }
    }
    for (int d = 0; d < dim; d++)
    {
        q[d] = (double)rand() / (double)RAND_MAX;
    }

    double out[8];
    res->kernel_name = "8-Anchor Double Batch";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist8_d64_scalar(q, (const double *const *)anchors, out, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        dist8_d64_avx2(q, (const double *const *)anchors, out, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            dist8_d64_avx512(q, (const double *const *)anchors, out, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(q);
    for (int a = 0; a < 8; a++)
    {
        free(anchors[a]);
    }
}

static void run_sq16_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    int16_t *a = (int16_t *)aligned_alloc(64, (size_t)dim * sizeof(int16_t));
    int16_t *b = (int16_t *)aligned_alloc(64, (size_t)dim * sizeof(int16_t));
    for (int i = 0; i < dim; i++)
    {
        a[i] = (int16_t)((rand() % 65536) - 32768);
        b[i] = (int16_t)((rand() % 65536) - 32768);
    }

    res->kernel_name = "SQ16 Quantized L2";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    volatile uint64_t sink = 0;
    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += sq16_dist_scalar(a, b, dim);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        sink += sq16_dist_avx2(a, b, dim);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            sink += sq16_dist_avx512(a, b, dim);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(a);
    free(b);
}

static void run_fastscan_bench(
    int              dim,
    int              reps,
    bool             can_avx512,
    SimdBenchResult *res)
{
    int num_codes = dim * 16;
    uint8_t *codes = (uint8_t *)aligned_alloc(64, (size_t)num_codes);
    uint8_t *lut   = (uint8_t *)aligned_alloc(64, 16);
    uint16_t *dists = (uint16_t *)aligned_alloc(64, (size_t)num_codes * sizeof(uint16_t));
    for (int i = 0; i < num_codes; i++)
    {
        codes[i] = (uint8_t)(rand() % 16);
        dists[i] = 0;
    }
    for (int i = 0; i < 16; i++)
    {
        lut[i] = (uint8_t)(rand() % 256);
    }

    res->kernel_name = "PQ FastScan (32x/64x)";
    res->dim = dim;
    res->has_avx512 = can_avx512;

    double t0 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        fastscan_scalar(codes, lut, dists, num_codes);
    }
    double t1 = get_time_sec();
    res->time_scalar = t1 - t0;

#if GRIC_HAVE_AVX512_TARGET
    double t2 = get_time_sec();
    for (int r = 0; r < reps; r++)
    {
        fastscan_avx2(codes, lut, dists, num_codes);
    }
    double t3 = get_time_sec();
    res->time_avx2 = t3 - t2;

    if (can_avx512)
    {
        double t4 = get_time_sec();
        for (int r = 0; r < reps; r++)
        {
            fastscan_avx512(codes, lut, dists, num_codes);
        }
        double t5 = get_time_sec();
        res->time_avx512 = t5 - t4;
    }
    else
    {
        res->time_avx512 = 0.0;
    }
#else
    res->time_avx2 = res->time_scalar;
    res->time_avx512 = 0.0;
#endif

    res->gain_avx2 = (res->time_avx2 > 0.0) ? (res->time_scalar / res->time_avx2) : 1.0;
    res->gain_avx512_vs_avx2 = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_avx2 / res->time_avx512) : 0.0;
    res->gain_total = (can_avx512 && res->time_avx512 > 0.0) ?
        (res->time_scalar / res->time_avx512) : 0.0;

    free(codes);
    free(lut);
    free(dists);
}

int main(
    int    argc,
    char **argv)
{
    bool markdown = false;
    for (int a = 1; a < argc; a++)
    {
        if (strcmp(argv[a], "--markdown") == 0 || strcmp(argv[a], "-md") == 0)
        {
            markdown = true;
        }
    }

    srand(42);
    GricSimdLevel active_level = gric_get_simd_level();
    bool can_avx512 = (active_level >= GRIC_SIMD_AVX512);

    int dims[] = {64, 128, 256, 512, 1024};
    int num_dims = (int)(sizeof(dims) / sizeof(dims[0]));
    int num_kernels = 8;
    int total_runs = num_dims * num_kernels;
    SimdBenchResult *results = (SimdBenchResult *)calloc(
        (size_t)total_runs, sizeof(SimdBenchResult));

    int idx = 0;
    for (int d = 0; d < num_dims; d++)
    {
        int dim = dims[d];
        int reps = 150000;
        if (dim >= 512) reps = 50000;
        if (dim >= 1024) reps = 25000;

        run_l2_bench(dim, reps, can_avx512, &results[idx++]);
        run_l2_d64_bench(dim, reps, can_avx512, &results[idx++]);
        run_dist8_bench(dim, reps / 4, can_avx512, &results[idx++]);
        run_dist16_f32_bench(dim, reps / 8, can_avx512, &results[idx++]);
        run_dist8_d64_bench(dim, reps / 4, can_avx512, &results[idx++]);
        run_sq8_bench(dim, reps, can_avx512, &results[idx++]);
        run_sq16_bench(dim, reps, can_avx512, &results[idx++]);
        run_fastscan_bench(dim, reps / 4, can_avx512, &results[idx++]);
    }

    if (markdown)
    {
        printf("### SIMD Scaling Benchmark Results (Scalar vs. AVX2 vs. AVX-512)\n\n");
        printf("**Detected SIMD Mode**: `%s`  \n", gric_simd_level_to_string(active_level));
        if (!can_avx512)
        {
            printf("> [!NOTE]\n");
            printf("> Host CPU lacks native AVX-512 instructions. ");
            printf("AVX-512 column will populate on AVX-512 hardware or under QEMU.\n\n");
        }
        printf("| Kernel | Dim | Scalar | AVX2 | AVX-512 | Gain (AVX2/Base) | "
               "Gain (512/AVX2) | Total Gain |\n");
        printf("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n");
        for (int i = 0; i < total_runs; i++)
        {
            SimdBenchResult *r = &results[i];
            if (r->has_avx512)
            {
                printf("| **%s** | %dD | %7.4f s | %7.4f s | %7.4f s | "
                       "**%5.2fx** | **%5.2fx** | **%5.2fx** |\n",
                       r->kernel_name, r->dim, r->time_scalar, r->time_avx2, r->time_avx512,
                       r->gain_avx2, r->gain_avx512_vs_avx2, r->gain_total);
            }
            else
            {
                printf("| **%s** | %dD | %7.4f s | %7.4f s | *N/A (no AVX-512)* | "
                       "**%5.2fx** | *N/A* | *N/A* |\n",
                       r->kernel_name, r->dim, r->time_scalar, r->time_avx2, r->gain_avx2);
            }
        }
    }
    else
    {
        printf("=========================================================================\n");
        printf("        GRIC SIMD SCALING BENCHMARK: SCALAR vs. AVX2 vs. AVX-512         \n");
        printf("=========================================================================\n");
        printf("Active Hardware SIMD: %s\n", gric_simd_level_to_string(active_level));
        printf("-------------------------------------------------------------------------\n");
        printf("%-20s | %-5s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
               "Kernel", "Dim", "Scalar", "AVX2", "AVX-512", "AVX2/Base", "512/AVX2");
        printf("-------------------------------------------------------------------------\n");
        for (int i = 0; i < total_runs; i++)
        {
            SimdBenchResult *r = &results[i];
            if (r->has_avx512)
            {
                printf("%-20s | %-5d | %7.4fs | %7.4fs | %7.4fs | %7.2fx | %7.2fx\n",
                       r->kernel_name, r->dim, r->time_scalar, r->time_avx2, r->time_avx512,
                       r->gain_avx2, r->gain_avx512_vs_avx2);
            }
            else
            {
                printf("%-20s | %-5d | %7.4fs | %7.4fs | %-7s | %7.2fx | %-7s\n",
                       r->kernel_name, r->dim, r->time_scalar, r->time_avx2, "N/A",
                       r->gain_avx2, "N/A");
            }
        }
        printf("=========================================================================\n");
    }

    free(results);
    return 0;
}

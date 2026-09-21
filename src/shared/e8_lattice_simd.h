/**
 * @file e8_lattice_simd.h
 * @brief Vectorized AVX2 and AVX-512 kernels for Conway-Sloane E8 lattice quantization.
 */

#ifndef E8_LATTICE_SIMD_H
#define E8_LATTICE_SIMD_H

#include "gric_simd.h"
#include <stdint.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Align-32 lookup table of single-lane 32-bit selection masks for 8 lanes.
 */
static const uint32_t s_e8_lane_masks[8][8] __attribute__((aligned(32))) = {
    {~0u, 0, 0, 0, 0, 0, 0, 0},
    {0, ~0u, 0, 0, 0, 0, 0, 0},
    {0, 0, ~0u, 0, 0, 0, 0, 0},
    {0, 0, 0, ~0u, 0, 0, 0, 0},
    {0, 0, 0, 0, ~0u, 0, 0, 0},
    {0, 0, 0, 0, 0, ~0u, 0, 0},
    {0, 0, 0, 0, 0, 0, ~0u, 0},
    {0, 0, 0, 0, 0, 0, 0, ~0u}
};

/**
 * hsum256_e8_ps() - Horizontal sum of 8 single-precision floats in __m256.
 * @v: Input 256-bit vector of 8 floats.
 *
 * Return: Sum of all 8 elements.
 */
static inline float hsum256_e8_ps(
    __m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

/**
 * d8_quantize_avx2() - Find closest point in D8 lattice using AVX2.
 * @vx: Input 8D coordinate vector in __m256.
 *
 * Return: Closest D8 lattice vector (all integer coordinates with even sum).
 */
GRIC_TARGET_AVX2
static inline __m256 d8_quantize_avx2(
    __m256 vx)
{
    __m256 vr = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i vi = _mm256_cvtps_epi32(vr);
    __m128i vi_lo = _mm256_castsi256_si128(vi);
    __m128i vi_hi = _mm256_extracti128_si256(vi, 1);
    __m128i xor4  = _mm_xor_si128(vi_lo, vi_hi);
    __m128i xor2  = _mm_xor_si128(xor4, _mm_srli_si128(xor4, 8));
    __m128i xor1  = _mm_xor_si128(xor2, _mm_srli_si128(xor2, 4));
    int parity = _mm_cvtsi128_si32(xor1) & 1;

    if (parity != 0)
    {
        __m256 verr = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_sub_ps(vx, vr));
        __m128 err_lo = _mm256_castps256_ps128(verr);
        __m128 err_hi = _mm256_extractf128_ps(verr, 1);
        __m128 max4   = _mm_max_ps(err_lo, err_hi);
        __m128 max2   = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128 max1   = _mm_max_ps(max2, _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(2, 3, 0, 1)));
        __m256 max8   = _mm256_broadcastss_ps(max1);
        __m256 is_max = _mm256_cmp_ps(verr, max8, _CMP_EQ_OQ);
        int mask = _mm256_movemask_ps(is_max);
        int max_k = __builtin_ctz(mask);

        __m256 ge = _mm256_cmp_ps(vx, vr, _CMP_GE_OQ);
        __m256 delta_all = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_set1_ps(1.0f), ge);
        __m256 lane_mask = _mm256_load_ps((const float *)s_e8_lane_masks[max_k]);
        __m256 delta = _mm256_and_ps(lane_mask, delta_all);
        vr = _mm256_add_ps(vr, delta);
    } // if (parity != 0)

    return vr;
}

/**
 * e8_quantize_point_avx2_vec() - Find closest point in E8 lattice for an 8D vector.
 * @vx: Input 8D coordinate vector in __m256.
 *
 * Return: Closest E8 vector (evaluating D8 and D8 + 1/2 cosets).
 */
GRIC_TARGET_AVX2
static inline __m256 e8_quantize_point_avx2_vec(
    __m256 vx)
{
    __m256 vp0 = d8_quantize_avx2(vx);
    __m256 vy = _mm256_sub_ps(vx, _mm256_set1_ps(0.5f));
    __m256 vq = d8_quantize_avx2(vy);
    __m256 vp1 = _mm256_add_ps(vq, _mm256_set1_ps(0.5f));

    __m256 diff0 = _mm256_sub_ps(vx, vp0);
    __m256 diff1 = _mm256_sub_ps(vx, vp1);
    float dist0 = hsum256_e8_ps(_mm256_mul_ps(diff0, diff0));
    float dist1 = hsum256_e8_ps(_mm256_mul_ps(diff1, diff1));

    return (dist0 <= dist1) ? vp0 : vp1;
}

#ifdef __cplusplus
}
#endif

#endif // x86 SIMD
#endif // E8_LATTICE_SIMD_H

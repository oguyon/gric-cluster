/**
 * @file residual_quant.c
 * @brief 8-bit Residual Vector Quantization (RQ8) implementation.
 */

#include "residual_quant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * rq8_init_params() - Initialize RQ8 parameters from cluster radius and dimension.
 * @params: Pointer to RQ8Params structure to initialize.
 * @rlim:   Maximum cluster covering radius.
 * @dim:    Vector dimension.
 */
void rq8_init_params(
    RQ8Params *params,
    float      rlim,
    long       dim)
{
    if (params == NULL)
    {
        return;
    }

    if (rlim <= 0.0f)
    {
        rlim = 1.0f;
    }

    params->rlim = rlim;
    params->scale = rlim / 127.0f;
    params->inv_scale = 127.0f / rlim;
    params->dim = dim;
    params->err_radius = sqrtf((float)dim) * params->scale * 0.5f;
}

/**
 * rq8_quantize_residual_float() - Quantize a single-precision float residual to int8_t.
 * @src:    Pointer to source float vector [dim].
 * @anchor: Pointer to cluster anchor float vector [dim].
 * @dst:    Pointer to destination int8_t vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
void rq8_quantize_residual_float(
    const float     *restrict src,
    const float     *restrict anchor,
    int8_t          *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    float inv_scale = params->inv_scale;
    long i = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256 v_inv = _mm256_set1_ps(inv_scale);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_nhalf = _mm256_set1_ps(-0.5f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_min = _mm256_set1_ps(-127.0f);
    __m256 v_max = _mm256_set1_ps(127.0f);

    for (; i <= dim - 16; i += 16)
    {
        // Vector 0 (first 8 floats)
        __m256 s0 = _mm256_loadu_ps(&src[i]);
        __m256 a0 = _mm256_loadu_ps(&anchor[i]);
        __m256 res0 = _mm256_mul_ps(_mm256_sub_ps(s0, a0), v_inv);
        __m256 round0 = _mm256_blendv_ps(v_nhalf, v_half, _mm256_cmp_ps(res0, v_zero, _CMP_GE_OQ));
        res0 = _mm256_max_ps(v_min, _mm256_min_ps(_mm256_add_ps(res0, round0), v_max));
        __m256i i32_0 = _mm256_cvttps_epi32(res0);

        // Vector 1 (next 8 floats)
        __m256 s1 = _mm256_loadu_ps(&src[i + 8]);
        __m256 a1 = _mm256_loadu_ps(&anchor[i + 8]);
        __m256 res1 = _mm256_mul_ps(_mm256_sub_ps(s1, a1), v_inv);
        __m256 round1 = _mm256_blendv_ps(v_nhalf, v_half, _mm256_cmp_ps(res1, v_zero, _CMP_GE_OQ));
        res1 = _mm256_max_ps(v_min, _mm256_min_ps(_mm256_add_ps(res1, round1), v_max));
        __m256i i32_1 = _mm256_cvttps_epi32(res1);

        // Pack 16x 32-bit into 16x 8-bit signed ints
        __m128i lo0 = _mm256_castsi256_si128(i32_0);
        __m128i hi0 = _mm256_extracti128_si256(i32_0, 1);
        __m128i lo1 = _mm256_castsi256_si128(i32_1);
        __m128i hi1 = _mm256_extracti128_si256(i32_1, 1);

        __m128i p16_0 = _mm_packs_epi32(lo0, hi0);
        __m128i p16_1 = _mm_packs_epi32(lo1, hi1);
        __m128i p8 = _mm_packs_epi16(p16_0, p16_1);

        _mm_storeu_si128((__m128i *)(dst + i), p8);
    } // for (; i <= dim - 16; i += 16)
#endif

    for (; i < dim; i++)
    {
        float delta = src[i] - anchor[i];
        float val = delta * inv_scale;
        float r_val = (val >= 0.0f) ? (val + 0.5f) : (val - 0.5f);
        if (r_val < -127.0f)
        {
            r_val = -127.0f;
        }
        else if (r_val > 127.0f)
        {
            r_val = 127.0f;
        }
        dst[i] = (int8_t)r_val;
    } // for (; i < dim; i++)
}

/**
 * rq8_quantize_residual_double() - Quantize a double-precision residual to int8_t.
 * @src:    Pointer to source double vector [dim].
 * @anchor: Pointer to cluster anchor double vector [dim].
 * @dst:    Pointer to destination int8_t vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
void rq8_quantize_residual_double(
    const double    *restrict src,
    const double    *restrict anchor,
    int8_t          *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    double inv_scale = (double)params->inv_scale;
    long i = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256d v_inv = _mm256_set1_pd(inv_scale);
    __m256d v_half = _mm256_set1_pd(0.5);
    __m256d v_nhalf = _mm256_set1_pd(-0.5);
    __m256d v_zero = _mm256_setzero_pd();
    __m256d v_min = _mm256_set1_pd(-127.0);
    __m256d v_max = _mm256_set1_pd(127.0);

    for (; i <= dim - 16; i += 16)
    {
        __m256d s0 = _mm256_loadu_pd(&src[i]);
        __m256d a0 = _mm256_loadu_pd(&anchor[i]);
        __m256d res0 = _mm256_mul_pd(_mm256_sub_pd(s0, a0), v_inv);
        __m256d round0 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res0, v_zero, _CMP_GE_OQ));
        res0 = _mm256_max_pd(v_min, _mm256_min_pd(_mm256_add_pd(res0, round0), v_max));
        __m128i i32_0 = _mm256_cvttpd_epi32(res0);

        __m256d s1 = _mm256_loadu_pd(&src[i + 4]);
        __m256d a1 = _mm256_loadu_pd(&anchor[i + 4]);
        __m256d res1 = _mm256_mul_pd(_mm256_sub_pd(s1, a1), v_inv);
        __m256d round1 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res1, v_zero, _CMP_GE_OQ));
        res1 = _mm256_max_pd(v_min, _mm256_min_pd(_mm256_add_pd(res1, round1), v_max));
        __m128i i32_1 = _mm256_cvttpd_epi32(res1);

        __m256d s2 = _mm256_loadu_pd(&src[i + 8]);
        __m256d a2 = _mm256_loadu_pd(&anchor[i + 8]);
        __m256d res2 = _mm256_mul_pd(_mm256_sub_pd(s2, a2), v_inv);
        __m256d round2 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res2, v_zero, _CMP_GE_OQ));
        res2 = _mm256_max_pd(v_min, _mm256_min_pd(_mm256_add_pd(res2, round2), v_max));
        __m128i i32_2 = _mm256_cvttpd_epi32(res2);

        __m256d s3 = _mm256_loadu_pd(&src[i + 12]);
        __m256d a3 = _mm256_loadu_pd(&anchor[i + 12]);
        __m256d res3 = _mm256_mul_pd(_mm256_sub_pd(s3, a3), v_inv);
        __m256d round3 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res3, v_zero, _CMP_GE_OQ));
        res3 = _mm256_max_pd(v_min, _mm256_min_pd(_mm256_add_pd(res3, round3), v_max));
        __m128i i32_3 = _mm256_cvttpd_epi32(res3);

        __m128i p16_0 = _mm_packs_epi32(i32_0, i32_1);
        __m128i p16_1 = _mm_packs_epi32(i32_2, i32_3);
        __m128i p8 = _mm_packs_epi16(p16_0, p16_1);

        _mm_storeu_si128((__m128i *)(dst + i), p8);
    } // for (; i <= dim - 16; i += 16)
#endif

    for (; i < dim; i++)
    {
        double delta = src[i] - anchor[i];
        double val = delta * inv_scale;
        double r_val = (val >= 0.0) ? (val + 0.5) : (val - 0.5);
        if (r_val < -127.0)
        {
            r_val = -127.0;
        }
        else if (r_val > 127.0)
        {
            r_val = 127.0;
        }
        dst[i] = (int8_t)r_val;
    } // for (long i = 0; i < dim; i++)
}

/**
 * rq8_quantize_query_residual_float() - Quantize single-precision query residual to int16_t.
 * @query:  Pointer to query float vector [dim].
 * @anchor: Pointer to cluster anchor float vector [dim].
 * @dst:    Pointer to destination int16_t vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
int rq8_quantize_query_residual_float(
    const float     *restrict query,
    const float     *restrict anchor,
    int16_t         *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    float inv_scale = params->inv_scale;
    long i = 0;
    int clipped = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256 v_inv = _mm256_set1_ps(inv_scale);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_nhalf = _mm256_set1_ps(-0.5f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_min = _mm256_set1_ps(-32767.0f);
    __m256 v_max = _mm256_set1_ps(32767.0f);

    for (; i <= dim - 16; i += 16)
    {
        __m256 q0 = _mm256_loadu_ps(&query[i]);
        __m256 a0 = _mm256_loadu_ps(&anchor[i]);
        __m256 res0 = _mm256_mul_ps(_mm256_sub_ps(q0, a0), v_inv);
        __m256 round0 = _mm256_blendv_ps(
            v_nhalf, v_half, _mm256_cmp_ps(res0, v_zero, _CMP_GE_OQ));
        res0 = _mm256_add_ps(res0, round0);
        if (_mm256_movemask_ps(_mm256_or_ps(
                _mm256_cmp_ps(res0, v_min, _CMP_LT_OQ),
                _mm256_cmp_ps(res0, v_max, _CMP_GT_OQ))) != 0)
        {
            clipped = 1;
        }
        res0 = _mm256_max_ps(v_min, _mm256_min_ps(res0, v_max));
        __m256i i32_0 = _mm256_cvttps_epi32(res0);

        __m256 q1 = _mm256_loadu_ps(&query[i + 8]);
        __m256 a1 = _mm256_loadu_ps(&anchor[i + 8]);
        __m256 res1 = _mm256_mul_ps(_mm256_sub_ps(q1, a1), v_inv);
        __m256 round1 = _mm256_blendv_ps(
            v_nhalf, v_half, _mm256_cmp_ps(res1, v_zero, _CMP_GE_OQ));
        res1 = _mm256_add_ps(res1, round1);
        if (_mm256_movemask_ps(_mm256_or_ps(
                _mm256_cmp_ps(res1, v_min, _CMP_LT_OQ),
                _mm256_cmp_ps(res1, v_max, _CMP_GT_OQ))) != 0)
        {
            clipped = 1;
        }
        res1 = _mm256_max_ps(v_min, _mm256_min_ps(res1, v_max));
        __m256i i32_1 = _mm256_cvttps_epi32(res1);

        __m128i lo0 = _mm256_castsi256_si128(i32_0);
        __m128i hi0 = _mm256_extracti128_si256(i32_0, 1);
        __m128i lo1 = _mm256_castsi256_si128(i32_1);
        __m128i hi1 = _mm256_extracti128_si256(i32_1, 1);

        __m128i p16_0 = _mm_packs_epi32(lo0, hi0);
        __m128i p16_1 = _mm_packs_epi32(lo1, hi1);

        _mm_storeu_si128((__m128i *)(dst + i), p16_0);
        _mm_storeu_si128((__m128i *)(dst + i + 8), p16_1);
    } // for (; i <= dim - 16; i += 16)
#endif

    for (; i < dim; i++)
    {
        float delta = query[i] - anchor[i];
        float val = delta * inv_scale;
        float r_val = (val >= 0.0f) ? (val + 0.5f) : (val - 0.5f);
        if (r_val < -32767.0f)
        {
            r_val = -32767.0f;
            clipped = 1;
        }
        else if (r_val > 32767.0f)
        {
            r_val = 32767.0f;
            clipped = 1;
        }
        dst[i] = (int16_t)r_val;
    } // for (; i < dim; i++)
    return clipped;
}

/**
 * rq8_quantize_query_residual_double() - Quantize double query residual to int16_t.
 * @query:  Pointer to query double vector [dim].
 * @anchor: Pointer to cluster anchor double vector [dim].
 * @dst:    Pointer to destination int16_t vector [dim].
 * @params: Pointer to initialized RQ8Params.
 */
int rq8_quantize_query_residual_double(
    const double    *restrict query,
    const double    *restrict anchor,
    int16_t         *restrict dst,
    const RQ8Params *restrict params)
{
    long dim = params->dim;
    double inv_scale = (double)params->inv_scale;
    long i = 0;
    int clipped = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256d v_inv = _mm256_set1_pd(inv_scale);
    __m256d v_half = _mm256_set1_pd(0.5);
    __m256d v_nhalf = _mm256_set1_pd(-0.5);
    __m256d v_zero = _mm256_setzero_pd();
    __m256d v_min = _mm256_set1_pd(-32767.0);
    __m256d v_max = _mm256_set1_pd(32767.0);

    for (; i <= dim - 8; i += 8)
    {
        __m256d q0 = _mm256_loadu_pd(&query[i]);
        __m256d a0 = _mm256_loadu_pd(&anchor[i]);
        __m256d res0 = _mm256_mul_pd(_mm256_sub_pd(q0, a0), v_inv);
        __m256d round0 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res0, v_zero, _CMP_GE_OQ));
        res0 = _mm256_add_pd(res0, round0);
        if (_mm256_movemask_pd(_mm256_or_pd(
                _mm256_cmp_pd(res0, v_min, _CMP_LT_OQ),
                _mm256_cmp_pd(res0, v_max, _CMP_GT_OQ))) != 0)
        {
            clipped = 1;
        }
        res0 = _mm256_max_pd(v_min, _mm256_min_pd(res0, v_max));
        __m128i i32_0 = _mm256_cvttpd_epi32(res0);

        __m256d q1 = _mm256_loadu_pd(&query[i + 4]);
        __m256d a1 = _mm256_loadu_pd(&anchor[i + 4]);
        __m256d res1 = _mm256_mul_pd(_mm256_sub_pd(q1, a1), v_inv);
        __m256d round1 = _mm256_blendv_pd(
            v_nhalf, v_half, _mm256_cmp_pd(res1, v_zero, _CMP_GE_OQ));
        res1 = _mm256_add_pd(res1, round1);
        if (_mm256_movemask_pd(_mm256_or_pd(
                _mm256_cmp_pd(res1, v_min, _CMP_LT_OQ),
                _mm256_cmp_pd(res1, v_max, _CMP_GT_OQ))) != 0)
        {
            clipped = 1;
        }
        res1 = _mm256_max_pd(v_min, _mm256_min_pd(res1, v_max));
        __m128i i32_1 = _mm256_cvttpd_epi32(res1);

        __m128i p16 = _mm_packs_epi32(i32_0, i32_1);
        _mm_storeu_si128((__m128i *)(dst + i), p16);
    } // for (; i <= dim - 8; i += 8)
#endif

    for (; i < dim; i++)
    {
        double delta = query[i] - anchor[i];
        double val = delta * inv_scale;
        double r_val = (val >= 0.0) ? (val + 0.5) : (val - 0.5);
        if (r_val < -32767.0)
        {
            r_val = -32767.0;
            clipped = 1;
        }
        else if (r_val > 32767.0)
        {
            r_val = 32767.0;
            clipped = 1;
        }
        dst[i] = (int16_t)r_val;
    } // for (long i = 0; i < dim; i++)
    return clipped;
}

/**
 * rq8_save_sidecar() - Save quantized residual dataset buffer and parameters to a binary .rq8 file.
 * @filepath:   Output sidecar file path.
 * @params:     Pointer to initialized RQ8Params.
 * @data:       Quantized dataset buffer [num_frames * dim] of int8_t.
 * @num_frames: Total number of frames in dataset.
 *
 * Return: 0 on success, non-zero on error.
 */
int rq8_save_sidecar(
    const char      *filepath,
    const RQ8Params *params,
    const int8_t    *data,
    long             num_frames,
    uint64_t         fingerprint)
{
    if (filepath == NULL || params == NULL || data == NULL || num_frames <= 0)
    {
        return -1;
    }

    FILE *f = fopen(filepath, "wb");
    if (f == NULL)
    {
        return -1;
    }

    if (fwrite(RQ8_FILE_MAGIC, 1, 8, f) != 8)
    {
        fclose(f);
        return -2;
    }

    float meta_f[4] = {
        params->rlim,
        params->scale,
        params->inv_scale,
        params->err_radius
    };
    if (fwrite(meta_f, sizeof(float), 4, f) != 4)
    {
        fclose(f);
        return -3;
    }

    int64_t meta_i[2] = {(int64_t)params->dim, (int64_t)num_frames};
    if (fwrite(meta_i, sizeof(int64_t), 2, f) != 2)
    {
        fclose(f);
        return -4;
    }

    if (fwrite(&fingerprint, sizeof(uint64_t), 1, f) != 1)
    {
        fclose(f);
        return -5;
    }

    size_t total_elements = (size_t)num_frames * (size_t)params->dim;
    if (fwrite(data, sizeof(int8_t), total_elements, f) != total_elements)
    {
        fclose(f);
        return -6;
    }

    fclose(f);
    return 0;
}

/**
 * rq8_load_sidecar() - Load quantized residual dataset buffer and parameters from a .rq8 file.
 * @filepath:   Input sidecar file path.
 * @params:     Pointer to RQ8Params to populate.
 * @data:       Output pointer to newly allocated int8_t dataset buffer.
 * @num_frames: Output pointer to loaded frame count.
 *
 * Return: 0 on success, non-zero on error.
 */
int rq8_load_sidecar(
    const char *filepath,
    RQ8Params  *params,
    int8_t    **data,
    long       *num_frames,
    uint64_t   *fingerprint)
{
    if (filepath == NULL || params == NULL || data == NULL || num_frames == NULL ||
        fingerprint == NULL)
    {
        return -1;
    }

    FILE *f = fopen(filepath, "rb");
    if (f == NULL)
    {
        return -1;
    }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, RQ8_FILE_MAGIC, 8) != 0)
    {
        fclose(f);
        return -2;
    }

    float meta_f[4];
    if (fread(meta_f, sizeof(float), 4, f) != 4)
    {
        fclose(f);
        return -3;
    }

    int64_t meta_i[2];
    if (fread(meta_i, sizeof(int64_t), 2, f) != 2)
    {
        fclose(f);
        return -4;
    }

    if (fread(fingerprint, sizeof(uint64_t), 1, f) != 1)
    {
        fclose(f);
        return -5;
    }

    params->rlim = meta_f[0];
    params->scale = meta_f[1];
    params->inv_scale = meta_f[2];
    params->err_radius = meta_f[3];
    params->dim = (long)meta_i[0];
    *num_frames = (long)meta_i[1];
    if (*num_frames <= 0)
    {
        fclose(f);
        return -6;
    }

    size_t total_elements = (size_t)(*num_frames) * (size_t)params->dim;
    int8_t *buf = (int8_t *)malloc(total_elements * sizeof(int8_t));
    if (buf == NULL)
    {
        fclose(f);
        return -7;
    }

    if (fread(buf, sizeof(int8_t), total_elements, f) != total_elements)
    {
        free(buf);
        fclose(f);
        return -8;
    }

    *data = buf;
    fclose(f);
    return 0;
}

/**
 * rq8_get_simd_mode() - Query active SIMD mode for FastScan on current CPU.
 *
 * Return: RQ8SimdMode enum.
 */
RQ8SimdMode rq8_get_simd_mode(void)
{
    GricSimdLevel lvl = gric_get_simd_level();
    if (lvl >= GRIC_SIMD_AVX512)
    {
        return RQ8_SIMD_AVX512;
    }
    if (lvl >= GRIC_SIMD_AVX2)
    {
        return RQ8_SIMD_AVX2;
    }
    return RQ8_SIMD_SCALAR;
}

/**
 * rq8_get_simd_mode_str() - Human-readable string for active SIMD register mode.
 *
 * Return: Pointer to constant string description.
 */
const char *rq8_get_simd_mode_str(void)
{
    switch (rq8_get_simd_mode())
    {
        case RQ8_SIMD_AVX512:
            return "AVX-512 (64-byte / 32-lane)";
        case RQ8_SIMD_AVX2:
            return "AVX2 (32-byte / 32-candidate)";
        default:
            return "Scalar (no SIMD)";
    }
}

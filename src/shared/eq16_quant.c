/**
 * @file eq16_quant.c
 * @brief 16-bit E8 Block Lattice Quantization (EQ16) core calibration & quantization.
 */

#include "eq16_quant.h"
#include "e8_lattice.h"
#include "e8_lattice_simd.h"
#include "gric_simd.h"
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * eq16_init_params() - Initialize EQ16 parameters from known min and max values.
 * @params:  Pointer to EQ16Params to initialize.
 * @min_val: Minimum dynamic range value.
 * @max_val: Maximum dynamic range value.
 * @dim:     Vector dimension.
 */
void eq16_init_params(
    EQ16Params *params,
    float       min_val,
    float       max_val,
    long        dim)
{
    if (params == NULL)
    {
        return;
    }

    params->min_val = min_val;
    params->max_val = max_val;
    params->center = 0.5f * (min_val + max_val);
    params->dim = dim;

    float range = max_val - min_val;
    if (range < 1e-12f)
    {
        range = 1.0f;
    }

    params->scale = range / 32767.0f;
    params->inv_scale = 1.0f / params->scale;

    long k = dim / 8;
    long rem = dim % 8;
    float covering_sq = (float)k * (1.0f * 1.0f) + (float)rem * (0.5f * 0.5f);
    params->err_radius = sqrtf(covering_sq) * params->scale;
}

/**
 * eq16_calibrate_float() - Scan a float dataset to find dynamic range and init params.
 * @params:       Pointer to EQ16Params to populate.
 * @data:         Data array to scan.
 * @num_elements: Total number of floats in data.
 * @dim:          Vector dimension.
 */
void eq16_calibrate_float(
    EQ16Params  *params,
    const float *data,
    long         num_elements,
    long         dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    float min_val = data[0];
    float max_val = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        float val = data[i];
        if (val < min_val)
        {
            min_val = val;
        }
        if (val > max_val)
        {
            max_val = val;
        }
    }

    eq16_init_params(params, min_val, max_val, dim);
}

/**
 * eq16_calibrate_double() - Scan a double dataset to find dynamic range and init params.
 * @params:       Pointer to EQ16Params to populate.
 * @data:         Double data array to scan.
 * @num_elements: Total number of doubles in data.
 * @dim:          Vector dimension.
 */
void eq16_calibrate_double(
    EQ16Params   *params,
    const double *data,
    long          num_elements,
    long          dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    double min_val = data[0];
    double max_val = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        double val = data[i];
        if (val < min_val)
        {
            min_val = val;
        }
        if (val > max_val)
        {
            max_val = val;
        }
    }

    eq16_init_params(params, (float)min_val, (float)max_val, dim);
}

/**
 * eq16_quantize_float() - Quantize a float frame onto E8 lattice into doubled int16.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
/**
 * eq16_quantize_float_scalar() - Scalar quantize float frame onto E8 lattice.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
static void eq16_quantize_float_scalar(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params)
{
    long dim = params->dim;
    float center = params->center;
    float inv_scale = params->inv_scale;
    long full_blocks = dim / 8;

    for (long b = 0; b < full_blocks; b++)
    {
        long offset = b * 8;
        float scaled[8];
        float q[8];

        for (int d = 0; d < 8; d++)
        {
            scaled[d] = (src[offset + d] - center) * inv_scale;
        }

        e8_quantize_point_float(scaled, q);

        for (int d = 0; d < 8; d++)
        {
            float doubled = roundf(2.0f * q[d]);
            if (doubled > 32767.0f)
            {
                doubled = 32767.0f;
            }
            else if (doubled < -32768.0f)
            {
                doubled = -32768.0f;
            }
            dst[offset + d] = (int16_t)doubled;
        }
    } // for (long b = 0; b < full_blocks; b++)

    /* Handle trailing dimensions with doubled scalar rounding */
    for (long i = full_blocks * 8; i < dim; i++)
    {
        float s = roundf((src[i] - center) * inv_scale);
        float doubled = roundf(2.0f * s);
        if (doubled > 32767.0f)
        {
            doubled = 32767.0f;
        }
        else if (doubled < -32768.0f)
        {
            doubled = -32768.0f;
        }
        dst[i] = (int16_t)doubled;
    }
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * eq16_quantize_float_avx2() - AVX2 quantize float frame onto E8 lattice.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
GRIC_TARGET_AVX2
static void eq16_quantize_float_avx2(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params)
{
    long dim = params->dim;
    float center = params->center;
    float inv_scale = params->inv_scale;
    long full_blocks = dim / 8;

    __m256 vcenter = _mm256_set1_ps(center);
    __m256 vinv = _mm256_set1_ps(inv_scale);

    for (long b = 0; b < full_blocks; b++)
    {
        long offset = b * 8;
        __m256 vin = _mm256_loadu_ps(src + offset);
        __m256 vscaled = _mm256_mul_ps(_mm256_sub_ps(vin, vcenter), vinv);
        __m256 vq = e8_quantize_point_avx2_vec(vscaled);

        /* 2.0 * vq is an exact integer for all points in E8 */
        __m256 vdoubled = _mm256_add_ps(vq, vq);
        __m256i vi32 = _mm256_cvtps_epi32(vdoubled);
        __m128i lo = _mm256_castsi256_si128(vi32);
        __m128i hi = _mm256_extracti128_si256(vi32, 1);
        __m128i vi16 = _mm_packs_epi32(lo, hi);
        _mm_storeu_si128((__m128i *)(dst + offset), vi16);
    } // for (long b = 0; b < full_blocks; b++)

    for (long i = full_blocks * 8; i < dim; i++)
    {
        float s = roundf((src[i] - center) * inv_scale);
        float doubled = roundf(2.0f * s);
        if (doubled > 32767.0f)
        {
            doubled = 32767.0f;
        }
        else if (doubled < -32768.0f)
        {
            doubled = -32768.0f;
        }
        dst[i] = (int16_t)doubled;
    }
}
#endif // x86 SIMD

#if GRIC_HAVE_AVX512_TARGET
/**
 * eq16_quantize_float_avx512() - AVX-512 quantize float frame onto E8 lattice.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
GRIC_TARGET_AVX512
static void eq16_quantize_float_avx512(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params)
{
    long dim = params->dim;
    float center = params->center;
    float inv_scale = params->inv_scale;
    long full_pairs = dim / 16;

    __m512 vcenter512 = _mm512_set1_ps(center);
    __m512 vinv512 = _mm512_set1_ps(inv_scale);

    for (long p = 0; p < full_pairs; p++)
    {
        long offset = p * 16;
        __m512 vin = _mm512_loadu_ps(src + offset);
        __m512 vscaled = _mm512_mul_ps(_mm512_sub_ps(vin, vcenter512), vinv512);

        __m256 vx0 = _mm512_castps512_ps256(vscaled);
        __m256 vx1 = _mm512_extractf32x8_ps(vscaled, 1);

        __m256 vq0 = e8_quantize_point_avx2_vec(vx0);
        __m256 vq1 = e8_quantize_point_avx2_vec(vx1);

        __m256 v2_0 = _mm256_add_ps(vq0, vq0);
        __m256 v2_1 = _mm256_add_ps(vq1, vq1);

        __m512 vdoubled512 = _mm512_insertf32x8(_mm512_castps256_ps512(v2_0), v2_1, 1);
        __m512i vi32 = _mm512_cvtps_epi32(vdoubled512);
        __m256i vi16 = _mm512_cvtepi32_epi16(vi32);

        _mm256_storeu_si256((__m256i *)(dst + offset), vi16);
    } // for (long p = 0; p < full_pairs; p++)

    long processed = full_pairs * 16;
    if (processed + 8 <= dim)
    {
        long offset = processed;
        __m256 vin = _mm256_loadu_ps(src + offset);
        __m256 vscaled = _mm256_mul_ps(_mm256_sub_ps(vin, _mm256_set1_ps(center)),
                                       _mm256_set1_ps(inv_scale));
        __m256 vq = e8_quantize_point_avx2_vec(vscaled);
        __m256 vdoubled = _mm256_add_ps(vq, vq);
        __m256i vi32 = _mm256_cvtps_epi32(vdoubled);
        __m128i lo = _mm256_castsi256_si128(vi32);
        __m128i hi = _mm256_extracti128_si256(vi32, 1);
        __m128i vi16 = _mm_packs_epi32(lo, hi);
        _mm_storeu_si128((__m128i *)(dst + offset), vi16);
        processed += 8;
    }

    for (long i = processed; i < dim; i++)
    {
        float s = roundf((src[i] - center) * inv_scale);
        float doubled = roundf(2.0f * s);
        if (doubled > 32767.0f)
        {
            doubled = 32767.0f;
        }
        else if (doubled < -32768.0f)
        {
            doubled = -32768.0f;
        }
        dst[i] = (int16_t)doubled;
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_quantize_float() - Quantize a float frame onto E8 lattice into doubled int16.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
void eq16_quantize_float(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params)
{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        eq16_quantize_float_avx512(src, dst, params);
        return;
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        eq16_quantize_float_avx2(src, dst, params);
        return;
    }
#endif
    eq16_quantize_float_scalar(src, dst, params);
}

/**
 * eq16_quantize_double() - Quantize a double frame onto E8 lattice into doubled int16.
 * @src:    Pointer to source double vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
void eq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params)
{
    long dim = params->dim;
    double center = (double)params->center;
    double inv_scale = (double)params->inv_scale;
    long full_blocks = dim / 8;

    for (long b = 0; b < full_blocks; b++)
    {
        long offset = b * 8;
        double scaled[8];
        double q[8];

        for (int d = 0; d < 8; d++)
        {
            scaled[d] = (src[offset + d] - center) * inv_scale;
        }

        e8_quantize_point_double(scaled, q);

        for (int d = 0; d < 8; d++)
        {
            double doubled = round(2.0 * q[d]);
            if (doubled > 32767.0)
            {
                doubled = 32767.0;
            }
            else if (doubled < -32768.0)
            {
                doubled = -32768.0;
            }
            dst[offset + d] = (int16_t)doubled;
        }
    }

    for (long i = full_blocks * 8; i < dim; i++)
    {
        double s = round((src[i] - center) * inv_scale);
        double doubled = round(2.0 * s);
        if (doubled > 32767.0)
        {
            doubled = 32767.0;
        }
        else if (doubled < -32768.0)
        {
            doubled = -32768.0;
        }
        dst[i] = (int16_t)doubled;
    }
}

/**
 * eq16_prepare_query_adc_float() - Normalize float query for Asymmetric Distance Computation.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination float vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
void eq16_prepare_query_adc_float(
    const float      *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params)
{
    float center = params->center;
    float scale_factor = 2.0f * params->inv_scale;
    long dim = params->dim;
    long i = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        __m256 vcenter = _mm256_set1_ps(center);
        __m256 vscale = _mm256_set1_ps(scale_factor);
        for (; i <= dim - 8; i += 8)
        {
            __m256 vsrc = _mm256_loadu_ps(src + i);
            __m256 vdiff = _mm256_sub_ps(vsrc, vcenter);
            __m256 vres = _mm256_mul_ps(vdiff, vscale);
            _mm256_storeu_ps(dst + i, vres);
        }
    }
#endif

    for (; i < dim; i++)
    {
        dst[i] = (src[i] - center) * scale_factor;
    }
}

/**
 * eq16_prepare_query_adc_double() - Normalize double query for Asymmetric Distance Computation.
 * @src:    Pointer to source double vector [dim].
 * @dst:    Pointer to destination float vector [dim].
 * @params: Pointer to initialized EQ16Params.
 */
void eq16_prepare_query_adc_double(
    const double     *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params)
{
    double center = (double)params->center;
    double scale_factor = 2.0 * (double)params->inv_scale;
    long dim = params->dim;

    for (long i = 0; i < dim; i++)
    {
        dst[i] = (float)((src[i] - center) * scale_factor);
    }
}

/**
 * eq16_quantize_float_perm() - Quantize a float frame onto E8 lattice with dimension permutation.
 * @src:      Pointer to source float vector [dim].
 * @dst:      Pointer to destination int16 vector [dim].
 * @params:   Pointer to initialized EQ16Params.
 * @perm_dim: Pointer to dimension permutation array [dim].
 */
void eq16_quantize_float_perm(
    const float      *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim)
{
    if (perm_dim == NULL)
    {
        eq16_quantize_float(src, dst, params);
        return;
    }

    long dim = params->dim;
    float center = params->center;
    float inv_scale = params->inv_scale;
    long full_blocks = dim / 8;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        __m256 vcenter = _mm256_set1_ps(center);
        __m256 vinv = _mm256_set1_ps(inv_scale);
        __m256 vtwo = _mm256_set1_ps(2.0f);

        for (long b = 0; b < full_blocks; b++)
        {
            long offset = b * 8;
            __m256 vin = _mm256_set_ps(
                src[perm_dim[offset + 7]],
                src[perm_dim[offset + 6]],
                src[perm_dim[offset + 5]],
                src[perm_dim[offset + 4]],
                src[perm_dim[offset + 3]],
                src[perm_dim[offset + 2]],
                src[perm_dim[offset + 1]],
                src[perm_dim[offset + 0]]
            );
            __m256 vscaled = _mm256_mul_ps(_mm256_sub_ps(vin, vcenter), vinv);
            __m256 vq = e8_quantize_point_avx2_vec(vscaled);
            __m256 vdoubled = _mm256_round_ps(_mm256_mul_ps(vq, vtwo),
                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i vi32 = _mm256_cvtps_epi32(vdoubled);
            __m128i vlo = _mm256_castsi256_si128(vi32);
            __m128i vhi = _mm256_extracti128_si256(vi32, 1);
            __m128i v16 = _mm_packs_epi32(vlo, vhi);
            _mm_storeu_si128((__m128i *)(dst + offset), v16);
        }

        for (long i = full_blocks * 8; i < dim; i++)
        {
            long src_idx = perm_dim[i];
            float s = roundf((src[src_idx] - center) * inv_scale);
            float doubled = roundf(2.0f * s);
            if (doubled > 32767.0f)
            {
                doubled = 32767.0f;
            }
            else if (doubled < -32768.0f)
            {
                doubled = -32768.0f;
            }
            dst[i] = (int16_t)doubled;
        }
        return;
    }
#endif

    for (long b = 0; b < full_blocks; b++)
    {
        long offset = b * 8;
        float scaled[8];
        float q[8];

        for (int d = 0; d < 8; d++)
        {
            long src_idx = perm_dim[offset + d];
            scaled[d] = (src[src_idx] - center) * inv_scale;
        }

        e8_quantize_point_float(scaled, q);

        for (int d = 0; d < 8; d++)
        {
            float doubled = roundf(2.0f * q[d]);
            if (doubled > 32767.0f)
            {
                doubled = 32767.0f;
            }
            else if (doubled < -32768.0f)
            {
                doubled = -32768.0f;
            }
            dst[offset + d] = (int16_t)doubled;
        }
    }

    for (long i = full_blocks * 8; i < dim; i++)
    {
        long src_idx = perm_dim[i];
        float s = roundf((src[src_idx] - center) * inv_scale);
        float doubled = roundf(2.0f * s);
        if (doubled > 32767.0f)
        {
            doubled = 32767.0f;
        }
        else if (doubled < -32768.0f)
        {
            doubled = -32768.0f;
        }
        dst[i] = (int16_t)doubled;
    }
}

/**
 * eq16_quantize_double_perm() - Quantize a double frame onto E8 lattice with dimension permutation.
 * @src:      Pointer to source double vector [dim].
 * @dst:      Pointer to destination int16 vector [dim].
 * @params:   Pointer to initialized EQ16Params.
 * @perm_dim: Pointer to dimension permutation array [dim].
 */
void eq16_quantize_double_perm(
    const double     *restrict src,
    int16_t          *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim)
{
    if (perm_dim == NULL)
    {
        eq16_quantize_double(src, dst, params);
        return;
    }

    long dim = params->dim;
    double center = (double)params->center;
    double inv_scale = (double)params->inv_scale;
    long full_blocks = dim / 8;

    for (long b = 0; b < full_blocks; b++)
    {
        long offset = b * 8;
        double scaled[8];
        double q[8];

        for (int d = 0; d < 8; d++)
        {
            long src_idx = perm_dim[offset + d];
            scaled[d] = (src[src_idx] - center) * inv_scale;
        }

        e8_quantize_point_double(scaled, q);

        for (int d = 0; d < 8; d++)
        {
            double doubled = round(2.0 * q[d]);
            if (doubled > 32767.0)
            {
                doubled = 32767.0;
            }
            else if (doubled < -32768.0)
            {
                doubled = -32768.0;
            }
            dst[offset + d] = (int16_t)doubled;
        }
    }

    for (long i = full_blocks * 8; i < dim; i++)
    {
        long src_idx = perm_dim[i];
        double s = round((src[src_idx] - center) * inv_scale);
        double doubled = round(2.0 * s);
        if (doubled > 32767.0)
        {
            doubled = 32767.0;
        }
        else if (doubled < -32768.0)
        {
            doubled = -32768.0;
        }
        dst[i] = (int16_t)doubled;
    }
}

/**
 * eq16_prepare_query_adc_float_perm() - Normalize float query for ADC with dimension permutation.
 * @src:      Pointer to source float vector [dim].
 * @dst:      Pointer to destination float vector [dim].
 * @params:   Pointer to initialized EQ16Params.
 * @perm_dim: Pointer to dimension permutation array [dim].
 */
void eq16_prepare_query_adc_float_perm(
    const float      *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim)
{
    if (perm_dim == NULL)
    {
        eq16_prepare_query_adc_float(src, dst, params);
        return;
    }

    float center = params->center;
    float scale_factor = 2.0f * params->inv_scale;
    long dim = params->dim;
    long i = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        __m256 vcenter = _mm256_set1_ps(center);
        __m256 vscale = _mm256_set1_ps(scale_factor);

        for (; i <= dim - 8; i += 8)
        {
            __m256 vsrc = _mm256_set_ps(
                src[perm_dim[i + 7]],
                src[perm_dim[i + 6]],
                src[perm_dim[i + 5]],
                src[perm_dim[i + 4]],
                src[perm_dim[i + 3]],
                src[perm_dim[i + 2]],
                src[perm_dim[i + 1]],
                src[perm_dim[i + 0]]
            );
            __m256 vdiff = _mm256_sub_ps(vsrc, vcenter);
            __m256 vres = _mm256_mul_ps(vdiff, vscale);
            _mm256_storeu_ps(dst + i, vres);
        }
    }
#endif

    for (; i < dim; i++)
    {
        dst[i] = (src[perm_dim[i]] - center) * scale_factor;
    }
}

/**
 * eq16_prepare_query_adc_double_perm() - Normalize double query for ADC with dimension permutation.
 * @src:      Pointer to source double vector [dim].
 * @dst:      Pointer to destination float vector [dim].
 * @params:   Pointer to initialized EQ16Params.
 * @perm_dim: Pointer to dimension permutation array [dim].
 */
void eq16_prepare_query_adc_double_perm(
    const double     *restrict src,
    float            *restrict dst,
    const EQ16Params *restrict params,
    const long       *restrict perm_dim)
{
    if (perm_dim == NULL)
    {
        eq16_prepare_query_adc_double(src, dst, params);
        return;
    }

    double center = (double)params->center;
    double scale_factor = 2.0 * (double)params->inv_scale;
    long dim = params->dim;

    for (long i = 0; i < dim; i++)
    {
        dst[i] = (float)((src[perm_dim[i]] - center) * scale_factor);
    }
}

/**
 * eq16_dist_squared_i16() - Compute sum of squared differences between two int16 vectors.
 * @a:   First vector [dim].
 * @b:   Second vector [dim].
 * @dim: Vector dimension.
 *
 * Return: Sum of squared coordinate differences as uint64_t.
 */

int eq16_save_sidecar(
    const char       *filepath,
    const EQ16Params *params,
    const int16_t    *data,
    long              num_frames)
{
    if (filepath == NULL || params == NULL || data == NULL || num_frames <= 0)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "wb");
    if (fp == NULL)
    {
        return -1;
    }

    if (fwrite(EQ16_FILE_MAGIC, 1, 8, fp) != 8)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[6] = {
        params->min_val,
        params->max_val,
        params->center,
        params->scale,
        params->inv_scale,
        params->err_radius
    };
    if (fwrite(meta_f, sizeof(float), 6, fp) != 6)
    {
        fclose(fp);
        return -1;
    }

    int64_t meta_i[2] = {(int64_t)params->dim, (int64_t)num_frames};
    if (fwrite(meta_i, sizeof(int64_t), 2, fp) != 2)
    {
        fclose(fp);
        return -1;
    }

    size_t total_elements = (size_t)params->dim * (size_t)num_frames;
    if (fwrite(data, sizeof(int16_t), total_elements, fp) != total_elements)
    {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/**
 * eq16_load_sidecar() - Load quantized dataset buffer and parameters from a binary .eq16 file.
 */
int eq16_load_sidecar(
    const char  *filepath,
    EQ16Params  *params,
    int16_t    **data,
    long        *num_frames)
{
    if (filepath == NULL || params == NULL || data == NULL || num_frames == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        return -1;
    }

    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, EQ16_FILE_MAGIC, 8) != 0)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[6];
    if (fread(meta_f, sizeof(float), 6, fp) != 6)
    {
        fclose(fp);
        return -1;
    }

    params->min_val = meta_f[0];
    params->max_val = meta_f[1];
    params->center = meta_f[2];
    params->scale = meta_f[3];
    params->inv_scale = meta_f[4];
    params->err_radius = meta_f[5];

    int64_t meta_i[2];
    if (fread(meta_i, sizeof(int64_t), 2, fp) != 2)
    {
        fclose(fp);
        return -1;
    }

    params->dim = (long)meta_i[0];
    *num_frames = (long)meta_i[1];

    size_t total_elements = (size_t)params->dim * (size_t)(*num_frames);
    *data = (int16_t *)malloc(total_elements * sizeof(int16_t));
    if (*data == NULL)
    {
        fclose(fp);
        return -1;
    }

    if (fread(*data, sizeof(int16_t), total_elements, fp) != total_elements)
    {
        free(*data);
        *data = NULL;
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/**
 * eq16_print_checkpoint_stats() - Print SIMD early exit statistics.
 */

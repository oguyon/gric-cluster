/**
 * @file eq16_quant.c
 * @brief Implementation of 16-bit E8 Block Lattice Quantization (EQ16).
 */

#include "eq16_quant.h"

#include "e8_lattice.h"
#include "e8_lattice_simd.h"
#include "gric_simd.h"
#include "scalar_quant.h"
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif
#include <math.h>
#include <string.h>

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
uint64_t eq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim)
{
    long i = 0;
    uint64_t total = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2 && dim >= 16)
    {
        __m256i sum_lo = _mm256_setzero_si256();
        __m256i sum_hi = _mm256_setzero_si256();

        for (; i <= dim - 16; i += 16)
        {
            __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));

            __m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
            __m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
            __m256i d_lo = _mm256_sub_epi32(a_lo, b_lo);

            __m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
            __m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
            __m256i d_hi = _mm256_sub_epi32(a_hi, b_hi);

            __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
            __m256i d_lo_o = _mm256_srli_si256(d_lo, 4);
            __m256i p_lo_o = _mm256_mul_epi32(d_lo_o, d_lo_o);

            __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
            __m256i d_hi_o = _mm256_srli_si256(d_hi, 4);
            __m256i p_hi_o = _mm256_mul_epi32(d_hi_o, d_hi_o);

            sum_lo = _mm256_add_epi64(sum_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
            sum_hi = _mm256_add_epi64(sum_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
        } // for (; i <= dim - 16; i += 16)

        __m256i s = _mm256_add_epi64(sum_lo, sum_hi);
        __m128i r = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        total = (uint64_t)_mm_cvtsi128_si64(r) + (uint64_t)_mm_extract_epi64(r, 1);
    }
#endif

    for (; i < dim; i++)
    {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i];
        total += (uint64_t)(diff * diff);
    } // for (; i < dim; i++)

    return total;
}

/**
 * eq16_dist_squared_cutoff_i16() - Compute squared difference with early cutoff.
 * @a:          First vector [dim].
 * @b:          Second vector [dim].
 * @dim:        Vector dimension.
 * @ssd_cutoff: Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences, or ssd_cutoff + 1 if threshold exceeded.
 */
uint64_t eq16_dist_squared_cutoff_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    long i = 0;
    uint64_t total = 0;

    if (dim < 16)
    {
        for (long k = 0; k < dim; k++)
        {
            int64_t diff = (int64_t)a[k] - (int64_t)b[k];
            total += (uint64_t)(diff * diff);
            if (total > ssd_cutoff)
            {
                return ssd_cutoff + 1;
            }
        } // for (long k = 0; k < dim; k++)
        return total;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        __m256i sum_lo = _mm256_setzero_si256();
        __m256i sum_hi = _mm256_setzero_si256();

        for (; i <= dim - 16; i += 16)
        {
            __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));

            __m256i a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
            __m256i b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
            __m256i d_lo = _mm256_sub_epi32(a_lo, b_lo);

            __m256i a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
            __m256i b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
            __m256i d_hi = _mm256_sub_epi32(a_hi, b_hi);

            __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
            __m256i d_lo_o = _mm256_srli_si256(d_lo, 4);
            __m256i p_lo_o = _mm256_mul_epi32(d_lo_o, d_lo_o);

            __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
            __m256i d_hi_o = _mm256_srli_si256(d_hi, 4);
            __m256i p_hi_o = _mm256_mul_epi32(d_hi_o, d_hi_o);

            sum_lo = _mm256_add_epi64(sum_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
            sum_hi = _mm256_add_epi64(sum_hi, _mm256_add_epi64(p_hi_e, p_hi_o));

            if ((i & 31) == 16)
            {
                __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
                __m128i slo = _mm256_castsi256_si128(sum);
                __m128i shi = _mm256_extracti128_si256(sum, 1);
                __m128i s128 = _mm_add_epi64(slo, shi);
                uint64_t partial = (uint64_t)_mm_cvtsi128_si64(s128) +
                                   (uint64_t)_mm_extract_epi64(s128, 1);
                if (partial > ssd_cutoff)
                {
                    return ssd_cutoff + 1;
                }
            }
        } // for (; i <= dim - 16; i += 16)

        __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
        __m128i slo = _mm256_castsi256_si128(sum);
        __m128i shi = _mm256_extracti128_si256(sum, 1);
        __m128i s128 = _mm_add_epi64(slo, shi);
        total = (uint64_t)_mm_cvtsi128_si64(s128) +
                (uint64_t)_mm_extract_epi64(s128, 1);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    }
#endif

    for (; i < dim; i++)
    {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i];
        total += (uint64_t)(diff * diff);
        if (total > ssd_cutoff)
        {
            return ssd_cutoff + 1;
        }
    } // for (; i < dim; i++)

    return total;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline float hsum256_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(vlow, vhigh);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    return _mm_cvtss_f32(sum1);
}
#endif

static float eq16_dist_asym_cutoff_scalar(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    float total = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if ((i & 63) == 63 && total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (long i = 0; i < dim; i++)

    return total;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
GRIC_TARGET_AVX2
static float eq16_dist_asym_cutoff_avx2(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    long i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (; i <= dim - 16; i += 16)
    {
        __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i));
        __m256i c32_0 = _mm256_cvtepi16_epi32(c16_0);
        __m256  cf_0  = _mm256_cvtepi32_ps(c32_0);
        __m256  q_0   = _mm256_loadu_ps(q_scaled + i);
        __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i + 8));
        __m256i c32_1 = _mm256_cvtepi16_epi32(c16_1);
        __m256  cf_1  = _mm256_cvtepi32_ps(c32_1);
        __m256  q_1   = _mm256_loadu_ps(q_scaled + i + 8);
        __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        if ((i & 63) == 48)
        {
            float partial = hsum256_ps(_mm256_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    float total = hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}
#endif // x86 / AVX2

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static float eq16_dist_asym_cutoff_avx512(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    long i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    for (; i <= dim - 32; i += 32)
    {
        __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i));
        __m512i c32_0 = _mm512_cvtepi16_epi32(c16_0);
        __m512  cf_0  = _mm512_cvtepi32_ps(c32_0);
        __m512  q_0   = _mm512_loadu_ps(q_scaled + i);
        __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i + 16));
        __m512i c32_1 = _mm512_cvtepi16_epi32(c16_1);
        __m512  cf_1  = _mm512_cvtepi32_ps(c32_1);
        __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);
        __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        if ((i & 63) == 32)
        {
            float partial = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    float total = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_dist_asym_cutoff_f32() - Asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cand_eq16: Quantized candidate int16 vector [dim].
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
float eq16_dist_asym_cutoff_f32(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    dim,
    float                   cutoff)
{
    if (dim < 16)
    {
        return eq16_dist_asym_cutoff_scalar(q_scaled, cand_eq16, dim, cutoff);
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return eq16_dist_asym_cutoff_avx512(q_scaled, cand_eq16, dim, cutoff);
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return eq16_dist_asym_cutoff_avx2(q_scaled, cand_eq16, dim, cutoff);
    }
#endif

    return eq16_dist_asym_cutoff_scalar(q_scaled, cand_eq16, dim, cutoff);
}

/**
 * eq16_dist_asym_cutoff_batch_1x4_scalar() - Scalar 1x4 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 4 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 4 float squared distances.
 */
static void eq16_dist_asym_cutoff_batch_1x4_scalar(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        d0 += diff0 * diff0;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
        d3 += diff3 * diff3;

        if ((i & 63) == 63)
        {
            if (d0 > cutoff && d1 > cutoff && d2 > cutoff && d3 > cutoff)
            {
                out_dists[0] = d0;
                out_dists[1] = d1;
                out_dists[2] = d2;
                out_dists[3] = d3;
                return;
            }
        }
    } // for (long i = 0; i < dim; i++)

    out_dists[0] = d0;
    out_dists[1] = d1;
    out_dists[2] = d2;
    out_dists[3] = d3;
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_scalar() - Scalar 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
static void eq16_dist_asym_cutoff_batch_1x8_scalar(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float d4 = 0.0f;
    float d5 = 0.0f;
    float d6 = 0.0f;
    float d7 = 0.0f;

    for (long i = 0; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        d0 += diff0 * diff0;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
        d3 += diff3 * diff3;
        d4 += diff4 * diff4;
        d5 += diff5 * diff5;
        d6 += diff6 * diff6;
        d7 += diff7 * diff7;

        if ((i & 63) == 63)
        {
            if (d0 > cutoff && d1 > cutoff && d2 > cutoff && d3 > cutoff &&
                d4 > cutoff && d5 > cutoff && d6 > cutoff && d7 > cutoff)
            {
                out_dists[0] = d0;
                out_dists[1] = d1;
                out_dists[2] = d2;
                out_dists[3] = d3;
                out_dists[4] = d4;
                out_dists[5] = d5;
                out_dists[6] = d6;
                out_dists[7] = d7;
                return;
            }
        }
    } // for (long i = 0; i < dim; i++)

    out_dists[0] = d0;
    out_dists[1] = d1;
    out_dists[2] = d2;
    out_dists[3] = d3;
    out_dists[4] = d4;
    out_dists[5] = d5;
    out_dists[6] = d6;
    out_dists[7] = d7;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline __m128 eq16_reduce_4x256_ps(
    __m256 acc0,
    __m256 acc1,
    __m256 acc2,
    __m256 acc3)
{
    __m128 lo0 = _mm256_castps256_ps128(acc0);
    __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
    __m128 s0  = _mm_add_ps(lo0, hi0);

    __m128 lo1 = _mm256_castps256_ps128(acc1);
    __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
    __m128 s1  = _mm_add_ps(lo1, hi1);

    __m128 lo2 = _mm256_castps256_ps128(acc2);
    __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
    __m128 s2  = _mm_add_ps(lo2, hi2);

    __m128 lo3 = _mm256_castps256_ps128(acc3);
    __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
    __m128 s3  = _mm_add_ps(lo3, hi3);

    __m128 h01 = _mm_hadd_ps(s0, s1);
    __m128 h23 = _mm_hadd_ps(s2, s3);
    return _mm_hadd_ps(h01, h23);
}

/**
 * eq16_dist_asym_resume_cutoff_avx2() - Resume single candidate asymmetric distance.
 * @q_scaled:  Normalized query float vector.
 * @cand_eq16: Quantized candidate int16 vector.
 * @start_dim: Starting dimension index (multiple of 16).
 * @dim:       Vector dimension.
 * @init_sum:  Accumulated distance from earlier dimensions.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
GRIC_TARGET_AVX2
static float eq16_dist_asym_resume_cutoff_avx2(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    start_dim,
    long                    dim,
    float                   init_sum,
    float                   cutoff)
{
    long i = start_dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float total = init_sum;

    for (; i <= dim - 16; i += 16)
    {
        __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i));
        __m256i c32_0 = _mm256_cvtepi16_epi32(c16_0);
        __m256  cf_0  = _mm256_cvtepi32_ps(c32_0);
        __m256  q_0   = _mm256_loadu_ps(q_scaled + i);
        __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(cand_eq16 + i + 8));
        __m256i c32_1 = _mm256_cvtepi16_epi32(c16_1);
        __m256  cf_1  = _mm256_cvtepi32_ps(c32_1);
        __m256  q_1   = _mm256_loadu_ps(q_scaled + i + 8);
        __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        if ((i & 63) == 48)
        {
            float partial = total + hsum256_ps(_mm256_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    total += hsum256_ps(_mm256_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}

GRIC_TARGET_AVX2
static void eq16_dist_asym_cutoff_batch_1x4_avx2(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 16; i += 16)
    {
        /* First 8 dimensions */
        {
            __m256 q_0 = _mm256_loadu_ps(q_scaled + i);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_0, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_0, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_0, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        /* Second 8 dimensions */
        {
            __m256 q_1 = _mm256_loadu_ps(q_scaled + i + 8);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i + 8));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_1, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i + 8));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i + 8));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_1, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i + 8));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_1, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        /* Periodic cutoff checkpoint (every 64 dimensions) */
        if ((i & 63) == 48)
        {
            __m128 sums = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
            __m128 cmp = _mm_cmpgt_ps(sums, vcutoff);
            int dead_mask = _mm_movemask_ps(cmp);
            if (dead_mask == 0xF)
            {
                _mm_storeu_ps(out_dists, sums);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 3 && i + 16 < dim)
            {
                _mm_storeu_ps(out_dists, sums);
                for (int k = 0; k < 4; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx2(
                            q_scaled, cands[k], i + 16, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    __m128 sums = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
    _mm_storeu_ps(out_dists, sums);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
    } // for (; i < dim; i++)
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_avx2() - AVX2 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
GRIC_TARGET_AVX2
static void eq16_dist_asym_cutoff_batch_1x8_avx2(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 16; i += 16)
    {
        /* First 8 dimensions */
        {
            __m256 q_0 = _mm256_loadu_ps(q_scaled + i);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_0, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_0, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_0, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_0, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

            __m128i c16_4 = _mm_loadu_si128((const __m128i *)(const void *)(c4 + i));
            __m256  cf_4  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_4));
            __m256  diff4 = _mm256_sub_ps(q_0, cf_4);
            acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);

            __m128i c16_5 = _mm_loadu_si128((const __m128i *)(const void *)(c5 + i));
            __m256  cf_5  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_5));
            __m256  diff5 = _mm256_sub_ps(q_0, cf_5);
            acc5 = _mm256_fmadd_ps(diff5, diff5, acc5);

            __m128i c16_6 = _mm_loadu_si128((const __m128i *)(const void *)(c6 + i));
            __m256  cf_6  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_6));
            __m256  diff6 = _mm256_sub_ps(q_0, cf_6);
            acc6 = _mm256_fmadd_ps(diff6, diff6, acc6);

            __m128i c16_7 = _mm_loadu_si128((const __m128i *)(const void *)(c7 + i));
            __m256  cf_7  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_7));
            __m256  diff7 = _mm256_sub_ps(q_0, cf_7);
            acc7 = _mm256_fmadd_ps(diff7, diff7, acc7);
        }

        /* Second 8 dimensions */
        {
            __m256 q_1 = _mm256_loadu_ps(q_scaled + i + 8);

            __m128i c16_0 = _mm_loadu_si128((const __m128i *)(const void *)(c0 + i + 8));
            __m256  cf_0  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_0));
            __m256  diff0 = _mm256_sub_ps(q_1, cf_0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m128i c16_1 = _mm_loadu_si128((const __m128i *)(const void *)(c1 + i + 8));
            __m256  cf_1  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_1));
            __m256  diff1 = _mm256_sub_ps(q_1, cf_1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            __m128i c16_2 = _mm_loadu_si128((const __m128i *)(const void *)(c2 + i + 8));
            __m256  cf_2  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_2));
            __m256  diff2 = _mm256_sub_ps(q_1, cf_2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            __m128i c16_3 = _mm_loadu_si128((const __m128i *)(const void *)(c3 + i + 8));
            __m256  cf_3  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_3));
            __m256  diff3 = _mm256_sub_ps(q_1, cf_3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

            __m128i c16_4 = _mm_loadu_si128((const __m128i *)(const void *)(c4 + i + 8));
            __m256  cf_4  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_4));
            __m256  diff4 = _mm256_sub_ps(q_1, cf_4);
            acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);

            __m128i c16_5 = _mm_loadu_si128((const __m128i *)(const void *)(c5 + i + 8));
            __m256  cf_5  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_5));
            __m256  diff5 = _mm256_sub_ps(q_1, cf_5);
            acc5 = _mm256_fmadd_ps(diff5, diff5, acc5);

            __m128i c16_6 = _mm_loadu_si128((const __m128i *)(const void *)(c6 + i + 8));
            __m256  cf_6  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_6));
            __m256  diff6 = _mm256_sub_ps(q_1, cf_6);
            acc6 = _mm256_fmadd_ps(diff6, diff6, acc6);

            __m128i c16_7 = _mm_loadu_si128((const __m128i *)(const void *)(c7 + i + 8));
            __m256  cf_7  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(c16_7));
            __m256  diff7 = _mm256_sub_ps(q_1, cf_7);
            acc7 = _mm256_fmadd_ps(diff7, diff7, acc7);
        }

        /* Periodic cutoff checkpoint (every 64 dimensions) */
        if ((i & 63) == 48)
        {
            __m128 sums_lo = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
            __m128 sums_hi = eq16_reduce_4x256_ps(acc4, acc5, acc6, acc7);
            __m128 cmp_lo = _mm_cmpgt_ps(sums_lo, vcutoff);
            __m128 cmp_hi = _mm_cmpgt_ps(sums_hi, vcutoff);
            int m_lo = _mm_movemask_ps(cmp_lo);
            int m_hi = _mm_movemask_ps(cmp_hi);
            int dead_mask = m_lo | (m_hi << 4);

            if (dead_mask == 0xFF)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 6 && i + 16 < dim)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);

                for (int k = 0; k < 8; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx2(
                            q_scaled, cands[k], i + 16, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 16; i += 16)

    __m128 sums_lo = eq16_reduce_4x256_ps(acc0, acc1, acc2, acc3);
    __m128 sums_hi = eq16_reduce_4x256_ps(acc4, acc5, acc6, acc7);
    _mm_storeu_ps(out_dists, sums_lo);
    _mm_storeu_ps(out_dists + 4, sums_hi);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
        out_dists[4] += diff4 * diff4;
        out_dists[5] += diff5 * diff5;
        out_dists[6] += diff6 * diff6;
        out_dists[7] += diff7 * diff7;
    } // for (; i < dim; i++)
}
#endif // x86 AVX2

#if GRIC_HAVE_AVX512_TARGET
/**
 * eq16_dist_asym_resume_cutoff_avx512() - Resume single candidate distance.
 * @q_scaled:  Normalized query float vector.
 * @cand_eq16: Quantized candidate int16 vector.
 * @start_dim: Starting dimension index (multiple of 32).
 * @dim:       Vector dimension.
 * @init_sum:  Accumulated distance from earlier dimensions.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 *
 * Return: Sum of squared differences as float, or cutoff + 1.0f if exceeded.
 */
GRIC_TARGET_AVX512
static float eq16_dist_asym_resume_cutoff_avx512(
    const float   *restrict q_scaled,
    const int16_t *restrict cand_eq16,
    long                    start_dim,
    long                    dim,
    float                   init_sum,
    float                   cutoff)
{
    long i = start_dim;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    float total = init_sum;

    for (; i <= dim - 32; i += 32)
    {
        __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i));
        __m512i c32_0 = _mm512_cvtepi16_epi32(c16_0);
        __m512  cf_0  = _mm512_cvtepi32_ps(c32_0);
        __m512  q_0   = _mm512_loadu_ps(q_scaled + i);
        __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(cand_eq16 + i + 16));
        __m512i c32_1 = _mm512_cvtepi16_epi32(c16_1);
        __m512  cf_1  = _mm512_cvtepi32_ps(c32_1);
        __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);
        __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        if ((i & 63) == 32)
        {
            float partial = total + _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
            if (partial > cutoff)
            {
                return cutoff * 1.01f + 1.0f;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    total += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    if (total > cutoff)
    {
        return cutoff * 1.01f + 1.0f;
    }

    for (; i < dim; i++)
    {
        float diff = q_scaled[i] - (float)cand_eq16[i];
        total += diff * diff;
        if (total > cutoff)
        {
            return cutoff * 1.01f + 1.0f;
        }
    } // for (; i < dim; i++)

    return total;
}

GRIC_TARGET_AVX512
static void eq16_dist_asym_cutoff_batch_1x4_avx512(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 32; i += 32)
    {
        /* First 16 dimensions */
        {
            __m512  q_0   = _mm512_loadu_ps(q_scaled + i);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_0, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_0, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_0, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        /* Second 16 dimensions */
        {
            __m512  q_1   = _mm512_loadu_ps(q_scaled + i + 16);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i + 16));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_1, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i + 16));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i + 16));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_1, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i + 16));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_1, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        /* Periodic cutoff checkpoint (every 64 dimensions) */
        if ((i & 63) == 32)
        {
            __m128 sums = _mm_set_ps(
                _mm512_reduce_add_ps(acc3),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc0));
            __m128 cmp = _mm_cmpgt_ps(sums, vcutoff);
            int dead_mask = _mm_movemask_ps(cmp);
            if (dead_mask == 0xF)
            {
                _mm_storeu_ps(out_dists, sums);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 3 && i + 32 < dim)
            {
                _mm_storeu_ps(out_dists, sums);
                for (int k = 0; k < 4; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx512(
                            q_scaled, cands[k], i + 32, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    __m128 sums = _mm_set_ps(
        _mm512_reduce_add_ps(acc3),
        _mm512_reduce_add_ps(acc2),
        _mm512_reduce_add_ps(acc1),
        _mm512_reduce_add_ps(acc0));
    _mm_storeu_ps(out_dists, sums);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
    } // for (; i < dim; i++)
}

/**
 * eq16_dist_asym_cutoff_batch_1x8_avx512() - AVX512 1x8 asymmetric distance with cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
GRIC_TARGET_AVX512
static void eq16_dist_asym_cutoff_batch_1x8_avx512(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    const int16_t *c0 = cands[0];
    const int16_t *c1 = cands[1];
    const int16_t *c2 = cands[2];
    const int16_t *c3 = cands[3];
    const int16_t *c4 = cands[4];
    const int16_t *c5 = cands[5];
    const int16_t *c6 = cands[6];
    const int16_t *c7 = cands[7];

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m128 vcutoff = _mm_set1_ps(cutoff);

    long i = 0;
    for (; i <= dim - 32; i += 32)
    {
        /* First 16 dimensions */
        {
            __m512 q_0 = _mm512_loadu_ps(q_scaled + i);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_0, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_0, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_0, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_0, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

            __m256i c16_4 = _mm256_loadu_si256((const __m256i *)(const void *)(c4 + i));
            __m512  cf_4  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_4));
            __m512  diff4 = _mm512_sub_ps(q_0, cf_4);
            acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);

            __m256i c16_5 = _mm256_loadu_si256((const __m256i *)(const void *)(c5 + i));
            __m512  cf_5  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_5));
            __m512  diff5 = _mm512_sub_ps(q_0, cf_5);
            acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);

            __m256i c16_6 = _mm256_loadu_si256((const __m256i *)(const void *)(c6 + i));
            __m512  cf_6  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_6));
            __m512  diff6 = _mm512_sub_ps(q_0, cf_6);
            acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);

            __m256i c16_7 = _mm256_loadu_si256((const __m256i *)(const void *)(c7 + i));
            __m512  cf_7  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_7));
            __m512  diff7 = _mm512_sub_ps(q_0, cf_7);
            acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);
        }

        /* Second 16 dimensions */
        {
            __m512 q_1 = _mm512_loadu_ps(q_scaled + i + 16);

            __m256i c16_0 = _mm256_loadu_si256((const __m256i *)(const void *)(c0 + i + 16));
            __m512  cf_0  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_0));
            __m512  diff0 = _mm512_sub_ps(q_1, cf_0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m256i c16_1 = _mm256_loadu_si256((const __m256i *)(const void *)(c1 + i + 16));
            __m512  cf_1  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_1));
            __m512  diff1 = _mm512_sub_ps(q_1, cf_1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m256i c16_2 = _mm256_loadu_si256((const __m256i *)(const void *)(c2 + i + 16));
            __m512  cf_2  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_2));
            __m512  diff2 = _mm512_sub_ps(q_1, cf_2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m256i c16_3 = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i + 16));
            __m512  cf_3  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_3));
            __m512  diff3 = _mm512_sub_ps(q_1, cf_3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

            __m256i c16_4 = _mm256_loadu_si256((const __m256i *)(const void *)(c4 + i + 16));
            __m512  cf_4  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_4));
            __m512  diff4 = _mm512_sub_ps(q_1, cf_4);
            acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);

            __m256i c16_5 = _mm256_loadu_si256((const __m256i *)(const void *)(c5 + i + 16));
            __m512  cf_5  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_5));
            __m512  diff5 = _mm512_sub_ps(q_1, cf_5);
            acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);

            __m256i c16_6 = _mm256_loadu_si256((const __m256i *)(const void *)(c6 + i + 16));
            __m512  cf_6  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_6));
            __m512  diff6 = _mm512_sub_ps(q_1, cf_6);
            acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);

            __m256i c16_7 = _mm256_loadu_si256((const __m256i *)(const void *)(c7 + i + 16));
            __m512  cf_7  = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c16_7));
            __m512  diff7 = _mm512_sub_ps(q_1, cf_7);
            acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);
        }

        /* Periodic cutoff checkpoint (every 64 dimensions) */
        if ((i & 63) == 32)
        {
            __m128 sums_lo = _mm_set_ps(
                _mm512_reduce_add_ps(acc3),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc0));
            __m128 sums_hi = _mm_set_ps(
                _mm512_reduce_add_ps(acc7),
                _mm512_reduce_add_ps(acc6),
                _mm512_reduce_add_ps(acc5),
                _mm512_reduce_add_ps(acc4));
            __m128 cmp_lo = _mm_cmpgt_ps(sums_lo, vcutoff);
            __m128 cmp_hi = _mm_cmpgt_ps(sums_hi, vcutoff);
            int m_lo = _mm_movemask_ps(cmp_lo);
            int m_hi = _mm_movemask_ps(cmp_hi);
            int dead_mask = m_lo | (m_hi << 4);

            if (dead_mask == 0xFF)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);
                return;
            }

            int dead_count = __builtin_popcount((unsigned int)dead_mask);
            if (dead_count >= 6 && i + 32 < dim)
            {
                _mm_storeu_ps(out_dists, sums_lo);
                _mm_storeu_ps(out_dists + 4, sums_hi);

                for (int k = 0; k < 8; k++)
                {
                    if (!(dead_mask & (1 << k)))
                    {
                        out_dists[k] = eq16_dist_asym_resume_cutoff_avx512(
                            q_scaled, cands[k], i + 32, dim, out_dists[k], cutoff);
                    }
                }
                return;
            }
        }
    } // for (; i <= dim - 32; i += 32)

    __m128 sums_lo = _mm_set_ps(
        _mm512_reduce_add_ps(acc3),
        _mm512_reduce_add_ps(acc2),
        _mm512_reduce_add_ps(acc1),
        _mm512_reduce_add_ps(acc0));
    __m128 sums_hi = _mm_set_ps(
        _mm512_reduce_add_ps(acc7),
        _mm512_reduce_add_ps(acc6),
        _mm512_reduce_add_ps(acc5),
        _mm512_reduce_add_ps(acc4));
    _mm_storeu_ps(out_dists, sums_lo);
    _mm_storeu_ps(out_dists + 4, sums_hi);

    /* Remainder dimensions */
    for (; i < dim; i++)
    {
        float q = q_scaled[i];
        float diff0 = q - (float)c0[i];
        float diff1 = q - (float)c1[i];
        float diff2 = q - (float)c2[i];
        float diff3 = q - (float)c3[i];
        float diff4 = q - (float)c4[i];
        float diff5 = q - (float)c5[i];
        float diff6 = q - (float)c6[i];
        float diff7 = q - (float)c7[i];

        out_dists[0] += diff0 * diff0;
        out_dists[1] += diff1 * diff1;
        out_dists[2] += diff2 * diff2;
        out_dists[3] += diff3 * diff3;
        out_dists[4] += diff4 * diff4;
        out_dists[5] += diff5 * diff5;
        out_dists[6] += diff6 * diff6;
        out_dists[7] += diff7 * diff7;
    } // for (; i < dim; i++)
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_dist_asym_cutoff_batch_1x4() - 1x4 asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 4 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 4 float squared distances.
 */
void eq16_dist_asym_cutoff_batch_1x4(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    if (dim < 16)
    {
        eq16_dist_asym_cutoff_batch_1x4_scalar(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 32)
    {
        eq16_dist_asym_cutoff_batch_1x4_avx512(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        eq16_dist_asym_cutoff_batch_1x4_avx2(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

    eq16_dist_asym_cutoff_batch_1x4_scalar(q_scaled, cands, dim, cutoff, out_dists);
}

/**
 * eq16_dist_asym_cutoff_batch_1x8() - 1x8 asymmetric squared distance with early cutoff.
 * @q_scaled:  Normalized query float vector [dim].
 * @cands:     Array of 8 pointers to candidate int16 vectors.
 * @dim:       Vector dimension.
 * @cutoff:    Cutoff threshold on sum of squared differences.
 * @out_dists: Output array of 8 float squared distances.
 */
void eq16_dist_asym_cutoff_batch_1x8(
    const float         *restrict  q_scaled,
    const int16_t *const *restrict cands,
    long                           dim,
    float                          cutoff,
    float               *restrict  out_dists)
{
    if (dim < 16)
    {
        eq16_dist_asym_cutoff_batch_1x8_scalar(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 32)
    {
        eq16_dist_asym_cutoff_batch_1x8_avx512(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        eq16_dist_asym_cutoff_batch_1x8_avx2(q_scaled, cands, dim, cutoff, out_dists);
        return;
    }
#endif

    eq16_dist_asym_cutoff_batch_1x8_scalar(q_scaled, cands, dim, cutoff, out_dists);
}

/**
 * eq16_refine_candidates_adc() - Refine candidate clusters using batched 1x4 ADC computation.
 * @cur_adc:          Normalized query float vector [dim].
 * @mat_eq16:         Quantized anchor matrix [num_clusters * dim].
 * @active_clusters:  Array of surviving candidate cluster indices (compacted in-place).
 * @num_active:       Number of input candidate clusters.
 * @dim:              Vector dimension.
 * @cutoff:           ADC cutoff threshold on sum of squared differences.
 * @clmembflag:       Cluster membership flag array (set to 0 for pruned clusters).
 * @out_num_active:   Output pointer to number of surviving clusters.
 * @out_pruned_count: Output pointer to accumulated pruned cluster count (may be NULL).
 */
void eq16_refine_candidates_adc(
    const float   *restrict cur_adc,
    const int16_t *restrict mat_eq16,
    int           *restrict active_clusters,
    int                     num_active,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    int next_num_active = 0;
    long pruned = 0;
    int idx = 0;

    for (; idx <= num_active - 8; idx += 8)
    {
        const int16_t *cands[8];
        for (int k = 0; k < 8; k++)
        {
            int c = active_clusters[idx + k];
            cands[k] = mat_eq16 + (size_t)c * (size_t)dim;
        }

        float dsq[8];
        eq16_dist_asym_cutoff_batch_1x8(cur_adc, cands, dim, cutoff, dsq);

        for (int k = 0; k < 8; k++)
        {
            int c = active_clusters[idx + k];
            if (dsq[k] > cutoff)
            {
                clmembflag[c] = 0;
                pruned++;
            }
            else
            {
                active_clusters[next_num_active++] = c;
            }
        }
    } // for (; idx <= num_active - 8; idx += 8)

    for (; idx <= num_active - 4; idx += 4)
    {
        int c0 = active_clusters[idx];
        int c1 = active_clusters[idx + 1];
        int c2 = active_clusters[idx + 2];
        int c3 = active_clusters[idx + 3];

        const int16_t *cands[4];
        cands[0] = mat_eq16 + (size_t)c0 * (size_t)dim;
        cands[1] = mat_eq16 + (size_t)c1 * (size_t)dim;
        cands[2] = mat_eq16 + (size_t)c2 * (size_t)dim;
        cands[3] = mat_eq16 + (size_t)c3 * (size_t)dim;

        float dsq[4];
        eq16_dist_asym_cutoff_batch_1x4(cur_adc, cands, dim, cutoff, dsq);

        if (dsq[0] > cutoff)
        {
            clmembflag[c0] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c0;
        }

        if (dsq[1] > cutoff)
        {
            clmembflag[c1] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c1;
        }

        if (dsq[2] > cutoff)
        {
            clmembflag[c2] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c2;
        }

        if (dsq[3] > cutoff)
        {
            clmembflag[c3] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c3;
        }
    } // for (; idx <= num_active - 4; idx += 4)

    for (; idx < num_active; idx++)
    {
        int c = active_clusters[idx];
        const int16_t *cand_anchor = mat_eq16 + (size_t)c * (size_t)dim;
        float dsq = eq16_dist_asym_cutoff_f32(cur_adc, cand_anchor, dim, cutoff);

        if (dsq > cutoff)
        {
            clmembflag[c] = 0;
            pruned++;
        }
        else
        {
            active_clusters[next_num_active++] = c;
        }
    } // for (; idx < num_active; idx++)

    *out_num_active = next_num_active;
    if (out_pruned_count != NULL)
    {
        *out_pruned_count += pruned;
    }
}

/**
 * eq16_compute_lower_bound() - Guaranteed metric lower bound between two EQ16 vectors.
 * @a:       First EQ16 vector.
 * @b:       Second EQ16 vector.
 * @params:  EQ16 parameter struct.
 * @epsilon: Relative relaxation factor (0.0 for exact metric bound).
 *
 * Return: Conservative lower bound on true Euclidean distance.
 */
double eq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const EQ16Params *restrict params,
    double                     epsilon)
{
    uint64_t ssd = eq16_dist_squared_i16(a, b, params->dim);
    double d_quant = sqrt((double)ssd) * ((double)params->scale * 0.5);
    double lb = d_quant - 2.0 * (double)params->err_radius;

    if (lb < 0.0)
    {
        lb = 0.0;
    }

    if (epsilon > 0.0)
    {
        lb /= (1.0 + epsilon);
    }

    return lb;
}

/**
 * eq16_dist_squared_batch_1x4_i16() - Compute SSD from 1 query against 4 anchors in SIMD.
 * @q:            Query EQ16 vector.
 * @anchors:      Array of 4 pointers to anchor vectors.
 * @out_sq_dists: Output array of 4 uint64_t SSD values.
 * @dim:          Vector dimension.
 */
void eq16_dist_squared_batch_1x4_i16(
    const int16_t *restrict        q,
    const int16_t *const *restrict anchors,
    uint64_t *restrict             out_sq_dists,
    long                           dim)
{
    long i = 0;

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2 && dim >= 16)
    {
        __m256i sum0_lo = _mm256_setzero_si256(), sum0_hi = _mm256_setzero_si256();
        __m256i sum1_lo = _mm256_setzero_si256(), sum1_hi = _mm256_setzero_si256();
        __m256i sum2_lo = _mm256_setzero_si256(), sum2_hi = _mm256_setzero_si256();
        __m256i sum3_lo = _mm256_setzero_si256(), sum3_hi = _mm256_setzero_si256();

        const int16_t *a0 = anchors[0];
        const int16_t *a1 = anchors[1];
        const int16_t *a2 = anchors[2];
        const int16_t *a3 = anchors[3];

        for (; i <= dim - 16; i += 16)
        {
            __m256i vq = _mm256_loadu_si256((const __m256i *)(const void *)(q + i));
            __m256i q_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vq));
            __m256i q_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vq, 1));

            /* Anchor 0 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a0 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum0_lo = _mm256_add_epi64(sum0_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum0_hi = _mm256_add_epi64(sum0_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 1 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a1 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum1_lo = _mm256_add_epi64(sum1_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum1_hi = _mm256_add_epi64(sum1_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 2 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a2 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum2_lo = _mm256_add_epi64(sum2_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum2_hi = _mm256_add_epi64(sum2_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }

            /* Anchor 3 */
            {
                __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a3 + i));
                __m256i d_lo = _mm256_sub_epi32(
                    q_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va)));
                __m256i d_hi = _mm256_sub_epi32(
                    q_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1)));
                __m256i p_lo_e = _mm256_mul_epi32(d_lo, d_lo);
                __m256i p_lo_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_lo, 4), _mm256_srli_si256(d_lo, 4));
                __m256i p_hi_e = _mm256_mul_epi32(d_hi, d_hi);
                __m256i p_hi_o = _mm256_mul_epi32(
                    _mm256_srli_si256(d_hi, 4), _mm256_srli_si256(d_hi, 4));
                sum3_lo = _mm256_add_epi64(sum3_lo, _mm256_add_epi64(p_lo_e, p_lo_o));
                sum3_hi = _mm256_add_epi64(sum3_hi, _mm256_add_epi64(p_hi_e, p_hi_o));
            }
        } // for (; i <= dim - 16; i += 16)

        __m256i s0 = _mm256_add_epi64(sum0_lo, sum0_hi);
        __m128i r0 = _mm_add_epi64(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1));
        out_sq_dists[0] = (uint64_t)_mm_cvtsi128_si64(r0) + (uint64_t)_mm_extract_epi64(r0, 1);

        __m256i s1 = _mm256_add_epi64(sum1_lo, sum1_hi);
        __m128i r1 = _mm_add_epi64(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1));
        out_sq_dists[1] = (uint64_t)_mm_cvtsi128_si64(r1) + (uint64_t)_mm_extract_epi64(r1, 1);

        __m256i s2 = _mm256_add_epi64(sum2_lo, sum2_hi);
        __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(s2), _mm256_extracti128_si256(s2, 1));
        out_sq_dists[2] = (uint64_t)_mm_cvtsi128_si64(r2) + (uint64_t)_mm_extract_epi64(r2, 1);

        __m256i s3 = _mm256_add_epi64(sum3_lo, sum3_hi);
        __m128i r3 = _mm_add_epi64(_mm256_castsi256_si128(s3), _mm256_extracti128_si256(s3, 1));
        out_sq_dists[3] = (uint64_t)_mm_cvtsi128_si64(r3) + (uint64_t)_mm_extract_epi64(r3, 1);
    }
    else
#endif
    {
        out_sq_dists[0] = 0;
        out_sq_dists[1] = 0;
        out_sq_dists[2] = 0;
        out_sq_dists[3] = 0;
    }

    for (; i < dim; i++)
    {
        int64_t qv = (int64_t)q[i];
        for (int k = 0; k < 4; k++)
        {
            int64_t diff = qv - (int64_t)anchors[k][i];
            out_sq_dists[k] += (uint64_t)(diff * diff);
        }
    } // for (; i < dim; i++)
}

/**
 * eq16_batch_filter_candidates() - Bulk filter candidates using EQ16 lower bounds.
 */
int eq16_batch_filter_candidates(
    const int16_t *restrict        q_eq16,
    const int16_t *const *restrict anchor_ptrs,
    const int                     *candidate_indices,
    int                            num_candidates,
    double                         cutoff_dist,
    const EQ16Params              *params,
    int *restrict                  clmembflag)
{
    if (num_candidates <= 0)
    {
        return 0;
    }

    double raw_thresh = (cutoff_dist + 2.0 * (double)params->err_radius) /
                        ((double)params->scale * 0.5);
    if (raw_thresh <= 0.0)
    {
        return 0;
    }

    uint64_t ssd_thresh = (uint64_t)(raw_thresh * raw_thresh);
    long dim = params->dim;
    int pruned_count = 0;
    int b_count = num_candidates / 4;

    for (int b = 0; b < b_count; b++)
    {
        int idx = b * 4;
        const int16_t *batch_anchors[4] = {
            anchor_ptrs[idx + 0],
            anchor_ptrs[idx + 1],
            anchor_ptrs[idx + 2],
            anchor_ptrs[idx + 3]
        };
        uint64_t sq_dists[4];

        eq16_dist_squared_batch_1x4_i16(q_eq16, batch_anchors, sq_dists, dim);

        for (int k = 0; k < 4; k++)
        {
            if (sq_dists[k] > ssd_thresh)
            {
                int cl_id = candidate_indices[idx + k];
                clmembflag[cl_id] = 0;
                pruned_count++;
            }
        }
    }

    for (int i = b_count * 4; i < num_candidates; i++)
    {
        uint64_t ssd = eq16_dist_squared_cutoff_i16(
            q_eq16, anchor_ptrs[i], dim, ssd_thresh
        );
        if (ssd > ssd_thresh)
        {
            int cl_id = candidate_indices[i];
            clmembflag[cl_id] = 0;
            pruned_count++;
        }
    }

    return pruned_count;
}

/**
 * eq16_set_anchor_interleaved() - Set coordinates into Block-8 interleaved format.
 * @matrix_interleaved: Interleaved buffer [num_blocks x num_pairs x 8].
 * @cl_idx:             Cluster index.
 * @anchor_coords:      Pointer to cluster's int16_t coordinates [dim].
 * @dim:                Dimension count.
 */
void eq16_set_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    int                     cl_idx,
    const int16_t *restrict anchor_coords,
    long                    dim)
{
    if (matrix_interleaved == NULL || anchor_coords == NULL || dim <= 0)
    {
        return;
    }

    int b = cl_idx / 8;
    int k = cl_idx % 8;
    size_t num_pairs = ((size_t)dim + 1) / 2;
    size_t base = (size_t)b * (num_pairs * 8) + (size_t)k;

    for (size_t j = 0; j < num_pairs; j++)
    {
        uint16_t v0 = (uint16_t)anchor_coords[2 * j];
        uint16_t v1 = (2 * j + 1 < (size_t)dim)
                      ? (uint16_t)anchor_coords[2 * j + 1]
                      : 0;
        matrix_interleaved[base + j * 8] =
            (int32_t)(((uint32_t)v1 << 16) | (uint32_t)v0);
    }
}

/**
 * eq16_rebuild_anchor_interleaved() - Rebuild Block-8 interleaved buffer.
 * @matrix_interleaved: Interleaved buffer [num_blocks x num_pairs x 8].
 * @anchor_matrix:      Row-major anchor matrix [num_clusters x dim].
 * @num_clusters:       Number of clusters.
 * @dim:                Dimension of each cluster anchor.
 */
void eq16_rebuild_anchor_interleaved(
    int32_t       *restrict matrix_interleaved,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim)
{
    if (matrix_interleaved == NULL || anchor_matrix == NULL || num_clusters <= 0 || dim <= 0)
    {
        return;
    }

    size_t num_blocks = ((size_t)num_clusters + 7) / 8;
    size_t num_pairs = ((size_t)dim + 1) / 2;
    size_t total_elements = num_blocks * num_pairs * 8;
    memset(matrix_interleaved, 0, total_elements * sizeof(int32_t));

    for (int c = 0; c < num_clusters; c++)
    {
        eq16_set_anchor_interleaved(
            matrix_interleaved,
            c,
            anchor_matrix + (size_t)c * (size_t)dim,
            dim
        );
    }
}

/**
 * eq16_set_anchor_adc_interleaved() - Set coordinates into Block-16 interleaved float format.
 * @matrix_adc_interleaved: Interleaved float buffer [num_blocks x dim x 16].
 * @cl_idx:                 Cluster index.
 * @anchor_coords:          Pointer to cluster's int16_t coordinates [dim].
 * @dim:                    Dimension count.
 */
void eq16_set_anchor_adc_interleaved(
    float         *restrict matrix_adc_interleaved,
    int                     cl_idx,
    const int16_t *restrict anchor_coords,
    long                    dim)
{
    if (matrix_adc_interleaved == NULL || anchor_coords == NULL || dim <= 0)
    {
        return;
    }

    int b = cl_idx / 16;
    int k = cl_idx % 16;
    size_t base = (size_t)b * ((size_t)dim * 16) + (size_t)k;

    for (long d = 0; d < dim; d++)
    {
        matrix_adc_interleaved[base + (size_t)d * 16] = (float)anchor_coords[d];
    }
}

/**
 * eq16_rebuild_anchor_adc_interleaved() - Rebuild Block-16 interleaved float buffer.
 * @matrix_adc_interleaved: Interleaved float buffer [num_blocks x dim x 16].
 * @anchor_matrix:          Row-major anchor matrix [num_clusters x dim].
 * @num_clusters:           Number of clusters.
 * @dim:                    Dimension of each cluster anchor.
 */
void eq16_rebuild_anchor_adc_interleaved(
    float         *restrict matrix_adc_interleaved,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim)
{
    if (matrix_adc_interleaved == NULL || anchor_matrix == NULL ||
        num_clusters <= 0 || dim <= 0)
    {
        return;
    }

    size_t num_blocks = ((size_t)num_clusters + 15) / 16;
    size_t total_elements = num_blocks * (size_t)dim * 16;
    memset(matrix_adc_interleaved, 0, total_elements * sizeof(float));

    for (int c = 0; c < num_clusters; c++)
    {
        eq16_set_anchor_adc_interleaved(
            matrix_adc_interleaved,
            c,
            anchor_matrix + (size_t)c * (size_t)dim,
            dim
        );
    }
}

/**
 * eq16_filter_anchor_matrix_scalar() - Scalar fallback for anchor matrix filtering.
 */
static void eq16_filter_anchor_matrix_scalar(
    const int16_t *restrict cur_eq16,
    const int16_t *restrict anchor_matrix,
    int                     num_clusters,
    long                    dim,
    uint64_t                eq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    int next_active_count = 0;
    long pruned_this_step = 0;

    for (int idx = 0; idx < num_clusters; idx++)
    {
        int c = active_clusters ? active_clusters[idx] : idx;
        if (!clmembflag[c])
        {
            continue;
        }

        const int16_t *anchor_ptr = anchor_matrix + (size_t)c * (size_t)dim;
        uint64_t ssd = eq16_dist_squared_cutoff_i16(
            cur_eq16, anchor_ptr, dim, eq16_ssd_thresh
        );

        if (ssd > eq16_ssd_thresh)
        {
            clmembflag[c] = 0;
            pruned_this_step++;
        }
        else if (active_clusters)
        {
            active_clusters[next_active_count++] = c;
        }
    }

    if (out_num_active && active_clusters)
    {
        *out_num_active = next_active_count;
    }
    if (out_pruned_count)
    {
        *out_pruned_count += pruned_this_step;
    }
}

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * eq16_compact_active_clusters_avx2() - Vectorized gather of surviving cluster indices.
 */
GRIC_TARGET_AVX2
static inline void eq16_compact_active_clusters_avx2(
    int  *restrict clmembflag,
    int  *restrict active_clusters,
    int            num_clusters,
    int  *restrict out_num_active,
    long *restrict out_pruned_count)
{
    int num_active = 0;
    int i = 0;
    for (; i + 8 <= num_clusters; i += 8)
    {
        __m256i vf = _mm256_loadu_si256((const __m256i *)(clmembflag + i));
        if (_mm256_testz_si256(vf, vf))
        {
            continue;
        }
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpgt_epi32(vf, _mm256_setzero_si256())));
        for (int k = 0; k < 8; k++)
        {
            if (mask & (1 << k))
            {
                if (active_clusters)
                {
                    active_clusters[num_active] = i + k;
                }
                num_active++;
            }
        }
    }
    for (; i < num_clusters; i++)
    {
        if (clmembflag[i])
        {
            if (active_clusters)
            {
                active_clusters[num_active] = i;
            }
            num_active++;
        }
    }
    if (out_num_active)
    {
        *out_num_active = num_active;
    }
    if (out_pruned_count)
    {
        *out_pruned_count += (long)(num_clusters - num_active);
    }
}

#ifdef EQ16_PROFILE_CHECKPOINTS
static uint64_t g_eq16_exit_pairs[512] = {0};
static uint64_t g_eq16_exit_none = 0;
#endif

/**
 * eq16_filter_anchor_matrix_avx2() - AVX2 kernel for anchor filtering.
 */
GRIC_TARGET_AVX2
static void eq16_filter_anchor_matrix_avx2(
    const int16_t *restrict cur_eq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                eq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    uint32_t thresh32 = (eq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)eq16_ssd_thresh;

    if (dim < 16 || anchor_interleaved == NULL)
    {
        eq16_filter_anchor_matrix_scalar(cur_eq16,
                                         anchor_matrix,
                                         num_clusters,
                                         dim,
                                         eq16_ssd_thresh,
                                         clmembflag,
                                         active_clusters,
                                         out_num_active,
                                         out_pruned_count);
        return;
    }

    __m256i v_bias256 = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut256  = _mm256_set1_epi32((int32_t)(thresh32 ^ 0x80000000U));
    __m256i v_min_clamp = _mm256_set1_epi16(-32767);

    int num_pairs = (int)((dim + 1) / 2);
    int num_full_pairs = (int)(dim / 2);
    int num_blocks = num_clusters / 8;
    int rem_start = num_blocks * 8;
    size_t blk_stride = (size_t)num_pairs * 8;

    const int32_t *q_pairs = (const int32_t *)cur_eq16;

#if defined(_OPENMP)
#pragma omp parallel for schedule(guided, 8) if(num_blocks >= 16)
#endif
    for (int b = 0; b < num_blocks; b++)
    {
        const int32_t *blk_ptr = anchor_interleaved + (size_t)b * blk_stride;

        if (b + 1 < num_blocks)
        {
            _mm_prefetch((const char *)(blk_ptr + blk_stride), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk_ptr + blk_stride + 64), _MM_HINT_T0);
        }

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        int pruned = 0;

        /* Checkpoint 0: Early Dim 4 check (pairs 0 and 1) */
        if (num_pairs >= 2)
        {
            __m256i qp0 = _mm256_set1_epi32(q_pairs[0]);
            __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + 0 * 8));
            __m256i df0 = _mm256_max_epi16(
                _mm256_subs_epi16(qp0, c0), v_min_clamp
            );
            acc0 = _mm256_madd_epi16(df0, df0);

            __m256i qp1 = _mm256_set1_epi32(q_pairs[1]);
            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + 1 * 8));
            __m256i df1 = _mm256_max_epi16(
                _mm256_subs_epi16(qp1, c1), v_min_clamp
            );
            acc1 = _mm256_madd_epi16(df1, df1);

            __m256i a_sum01 = _mm256_add_epi32(acc0, acc1);
            __m256i v_acc_b01 = _mm256_xor_si256(a_sum01, v_bias256);
            __m256i cmp01 = _mm256_cmpgt_epi32(v_acc_b01, v_cut256);
            if (_mm256_movemask_ps(_mm256_castsi256_ps(cmp01)) == 0xFF)
            {
                int base = b * 8;
                _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                    _mm256_setzero_si256());
#ifdef EQ16_PROFILE_CHECKPOINTS
                g_eq16_exit_pairs[2]++;
#endif
                continue;
            }
        }

        /* Pairs 2 and 3 */
        if (num_pairs >= 4)
        {
            __m256i qp2 = _mm256_set1_epi32(q_pairs[2]);
            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + 2 * 8));
            __m256i df2 = _mm256_max_epi16(
                _mm256_subs_epi16(qp2, c2), v_min_clamp
            );
            acc2 = _mm256_madd_epi16(df2, df2);

            __m256i qp3 = _mm256_set1_epi32(q_pairs[3]);
            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + 3 * 8));
            __m256i df3 = _mm256_max_epi16(
                _mm256_subs_epi16(qp3, c3), v_min_clamp
            );
            acc3 = _mm256_madd_epi16(df3, df3);

            /* Checkpoint 1: Early Dim 8 check (pairs 0..3) */
            __m256i a_sum03 = _mm256_add_epi32(
                _mm256_add_epi32(acc0, acc1),
                _mm256_add_epi32(acc2, acc3)
            );
            __m256i v_acc_b03 = _mm256_xor_si256(a_sum03, v_bias256);
            __m256i cmp03 = _mm256_cmpgt_epi32(v_acc_b03, v_cut256);
            if (_mm256_movemask_ps(_mm256_castsi256_ps(cmp03)) == 0xFF)
            {
                int base = b * 8;
                _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                    _mm256_setzero_si256());
#ifdef EQ16_PROFILE_CHECKPOINTS
                g_eq16_exit_pairs[4]++;
#endif
                continue;
            }
        }

        int j = 4;
        for (; j + 4 <= num_full_pairs; j += 4)
        {
            __m256i qp0 = _mm256_set1_epi32(q_pairs[j + 0]);
            __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 0) * 8));
            __m256i df0 = _mm256_max_epi16(
                _mm256_subs_epi16(qp0, c0), v_min_clamp
            );
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(df0, df0));

            __m256i qp1 = _mm256_set1_epi32(q_pairs[j + 1]);
            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 1) * 8));
            __m256i df1 = _mm256_max_epi16(
                _mm256_subs_epi16(qp1, c1), v_min_clamp
            );
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(df1, df1));

            __m256i qp2 = _mm256_set1_epi32(q_pairs[j + 2]);
            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 2) * 8));
            __m256i df2 = _mm256_max_epi16(
                _mm256_subs_epi16(qp2, c2), v_min_clamp
            );
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(df2, df2));

            __m256i qp3 = _mm256_set1_epi32(q_pairs[j + 3]);
            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 3) * 8));
            __m256i df3 = _mm256_max_epi16(
                _mm256_subs_epi16(qp3, c3), v_min_clamp
            );
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(df3, df3));

            /* Checkpoints at Dim 16 (j=4), Dim 24 (j=8), Dim 48 (j=20),
             * and every 32 dims ((j & 15) == 12) */
            if ((j == 4 || j == 8 || j == 20 || (j & 15) == 12) && j + 4 < num_pairs)
            {
                __m256i a_sum = _mm256_add_epi32(
                    _mm256_add_epi32(acc0, acc1),
                    _mm256_add_epi32(acc2, acc3));
                __m256i v_acc_b = _mm256_xor_si256(a_sum, v_bias256);
                __m256i cmp = _mm256_cmpgt_epi32(v_acc_b, v_cut256);
                if (_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) == 0xFF)
                {
                    int base = b * 8;
                    _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                        _mm256_setzero_si256());
                    pruned = 1;
#ifdef EQ16_PROFILE_CHECKPOINTS
                    if (j + 4 < 512)
                    {
                        g_eq16_exit_pairs[j + 4]++;
                    }
#endif
                    break;
                }
            }
        } // for (; j + 4 <= num_pairs; j += 4)

        if (!pruned)
        {
#ifdef EQ16_PROFILE_CHECKPOINTS
            g_eq16_exit_none++;
#endif
            for (; j < num_pairs; j++)
            {
                int32_t p = (2 * j + 1 < dim)
                            ? q_pairs[j]
                            : (int32_t)(uint16_t)cur_eq16[2 * j];
                __m256i q_p = _mm256_set1_epi32(p);
                __m256i c = _mm256_load_si256((const __m256i *)(blk_ptr + j * 8));
                __m256i df = _mm256_max_epi16(
                    _mm256_subs_epi16(q_p, c), v_min_clamp
                );
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(df, df));
            } // for (; j < num_pairs; j++)

            __m256i a_sum01 = _mm256_add_epi32(acc0, acc1);
            __m256i a_sum23 = _mm256_add_epi32(acc2, acc3);
            __m256i acc = _mm256_add_epi32(a_sum01, a_sum23);
            __m256i v_acc_b = _mm256_xor_si256(acc, v_bias256);
            __m256i cmp = _mm256_cmpgt_epi32(v_acc_b, v_cut256);

            int base = b * 8;
            __m256i flags = _mm256_andnot_si256(cmp, _mm256_set1_epi32(1));
            _mm256_storeu_si256((__m256i *)(clmembflag + base), flags);
        } // if (!pruned)
    } // for (int b = 0; b < num_blocks; b++)

    /* Remainder clusters (< 8) */
    for (int i = rem_start; i < num_clusters; i++)
    {
        const int16_t *ak_ptr = anchor_matrix + (size_t)i * (size_t)dim;
        uint64_t ssd = eq16_dist_squared_cutoff_i16(cur_eq16, ak_ptr, dim, eq16_ssd_thresh);
        clmembflag[i] = (ssd > eq16_ssd_thresh) ? 0 : 1;
    }

    eq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                      out_num_active, out_pruned_count);
}
#endif // x86 / AVX2

#if GRIC_HAVE_AVX512_TARGET
/**
 * eq16_filter_anchor_matrix_avx512() - AVX-512 kernel for anchor filtering.
 */
GRIC_TARGET_AVX512
static void eq16_filter_anchor_matrix_avx512(
    const int16_t *restrict cur_eq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                eq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    if (dim < 16 || anchor_interleaved == NULL)
    {
        eq16_filter_anchor_matrix_scalar(cur_eq16,
                                         anchor_matrix,
                                         num_clusters,
                                         dim,
                                         eq16_ssd_thresh,
                                         clmembflag,
                                         active_clusters,
                                         out_num_active,
                                         out_pruned_count);
        return;
    }

    uint32_t thresh32 = (eq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)eq16_ssd_thresh;

    int num_pairs = (int)((dim + 1) / 2);
    int num_full_pairs = (int)(dim / 2);
    int num_blocks = num_clusters / 8;
    int num_sb = num_blocks / 2;
    size_t blk_stride = (size_t)num_pairs * 8;
    __m512i v_cut512 = _mm512_set1_epi32((int32_t)thresh32);
    __m512i v_min_clamp512 = _mm512_set1_epi16(-32767);
    const int32_t *q_pairs = (const int32_t *)cur_eq16;

#if defined(_OPENMP)
#pragma omp parallel for schedule(guided, 4) if(num_sb >= 8)
#endif
    for (int sb = 0; sb < num_sb; sb++)
    {
        const int32_t *blk0 = anchor_interleaved + (size_t)(2 * sb + 0) * blk_stride;
        const int32_t *blk1 = anchor_interleaved + (size_t)(2 * sb + 1) * blk_stride;

        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        int pruned = 0;

        /* Checkpoint 0: Early Dim 4 check (pairs 0 and 1) */
        if (num_pairs >= 2)
        {
            __m512i qp0 = _mm512_set1_epi32(q_pairs[0]);
            __m256i c0_0 = _mm256_load_si256((const __m256i *)(blk0 + 0 * 8));
            __m256i c1_0 = _mm256_load_si256((const __m256i *)(blk1 + 0 * 8));
            __m512i c0 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_0), c1_0, 1);
            __m512i df0 = _mm512_max_epi16(
                _mm512_subs_epi16(qp0, c0), v_min_clamp512);
            acc0 = _mm512_madd_epi16(df0, df0);

            __m512i qp1 = _mm512_set1_epi32(q_pairs[1]);
            __m256i c0_1 = _mm256_load_si256((const __m256i *)(blk0 + 1 * 8));
            __m256i c1_1 = _mm256_load_si256((const __m256i *)(blk1 + 1 * 8));
            __m512i c1 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_1), c1_1, 1);
            __m512i df1 = _mm512_max_epi16(
                _mm512_subs_epi16(qp1, c1), v_min_clamp512);
            acc1 = _mm512_madd_epi16(df1, df1);

            __m512i a_sum01 = _mm512_add_epi32(acc0, acc1);
            __mmask16 cmp01 = _mm512_cmpgt_epu32_mask(a_sum01, v_cut512);
            if (cmp01 == 0xFFFF)
            {
                int base = sb * 16;
                _mm512_storeu_si512((void *)(clmembflag + base),
                                    _mm512_setzero_si512());
                continue;
            }
        }

        /* Pairs 2 and 3 */
        if (num_pairs >= 4)
        {
            __m512i qp2 = _mm512_set1_epi32(q_pairs[2]);
            __m256i c0_2 = _mm256_load_si256((const __m256i *)(blk0 + 2 * 8));
            __m256i c1_2 = _mm256_load_si256((const __m256i *)(blk1 + 2 * 8));
            __m512i c2 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_2), c1_2, 1);
            __m512i df2 = _mm512_max_epi16(
                _mm512_subs_epi16(qp2, c2), v_min_clamp512);
            acc2 = _mm512_madd_epi16(df2, df2);

            __m512i qp3 = _mm512_set1_epi32(q_pairs[3]);
            __m256i c0_3 = _mm256_load_si256((const __m256i *)(blk0 + 3 * 8));
            __m256i c1_3 = _mm256_load_si256((const __m256i *)(blk1 + 3 * 8));
            __m512i c3 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_3), c1_3, 1);
            __m512i df3 = _mm512_max_epi16(
                _mm512_subs_epi16(qp3, c3), v_min_clamp512);
            acc3 = _mm512_madd_epi16(df3, df3);

            /* Checkpoint 1: Early Dim 8 check */
            __m512i a_sum03 = _mm512_add_epi32(
                _mm512_add_epi32(acc0, acc1),
                _mm512_add_epi32(acc2, acc3));
            __mmask16 cmp03 = _mm512_cmpgt_epu32_mask(a_sum03, v_cut512);
            if (cmp03 == 0xFFFF)
            {
                int base = sb * 16;
                _mm512_storeu_si512((void *)(clmembflag + base),
                                    _mm512_setzero_si512());
                continue;
            }
        }

        int j = 4;
        for (; j + 4 <= num_full_pairs; j += 4)
        {
            __m512i qp0 = _mm512_set1_epi32(q_pairs[j + 0]);
            __m256i c0_0 = _mm256_load_si256((const __m256i *)(blk0 + (j + 0) * 8));
            __m256i c1_0 = _mm256_load_si256((const __m256i *)(blk1 + (j + 0) * 8));
            __m512i c0 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_0), c1_0, 1);
            __m512i df0 = _mm512_max_epi16(
                _mm512_subs_epi16(qp0, c0), v_min_clamp512);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(df0, df0));

            __m512i qp1 = _mm512_set1_epi32(q_pairs[j + 1]);
            __m256i c0_1 = _mm256_load_si256((const __m256i *)(blk0 + (j + 1) * 8));
            __m256i c1_1 = _mm256_load_si256((const __m256i *)(blk1 + (j + 1) * 8));
            __m512i c1 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_1), c1_1, 1);
            __m512i df1 = _mm512_max_epi16(
                _mm512_subs_epi16(qp1, c1), v_min_clamp512);
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(df1, df1));

            __m512i qp2 = _mm512_set1_epi32(q_pairs[j + 2]);
            __m256i c0_2 = _mm256_load_si256((const __m256i *)(blk0 + (j + 2) * 8));
            __m256i c1_2 = _mm256_load_si256((const __m256i *)(blk1 + (j + 2) * 8));
            __m512i c2 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_2), c1_2, 1);
            __m512i df2 = _mm512_max_epi16(
                _mm512_subs_epi16(qp2, c2), v_min_clamp512);
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(df2, df2));

            __m512i qp3 = _mm512_set1_epi32(q_pairs[j + 3]);
            __m256i c0_3 = _mm256_load_si256((const __m256i *)(blk0 + (j + 3) * 8));
            __m256i c1_3 = _mm256_load_si256((const __m256i *)(blk1 + (j + 3) * 8));
            __m512i c3 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_3), c1_3, 1);
            __m512i df3 = _mm512_max_epi16(
                _mm512_subs_epi16(qp3, c3), v_min_clamp512);
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(df3, df3));

            /* Checkpoints at Dim 16 (j=4), Dim 24 (j=8), Dim 48 (j=20),
             * and every 32 dims ((j & 15) == 12) */
            if ((j == 4 || j == 8 || j == 20 || (j & 15) == 12) && j + 4 < num_pairs)
            {
                __m512i a_sum = _mm512_add_epi32(
                    _mm512_add_epi32(acc0, acc1),
                    _mm512_add_epi32(acc2, acc3));
                __mmask16 cmp = _mm512_cmpgt_epu32_mask(a_sum, v_cut512);
                if (cmp == 0xFFFF)
                {
                    int base = sb * 16;
                    _mm512_storeu_si512((void *)(clmembflag + base),
                                        _mm512_setzero_si512());
                    pruned = 1;
                    break;
                }
            }
        } // for (; j + 4 <= num_full_pairs; j += 4)

        if (!pruned)
        {
            for (; j < num_pairs; j++)
            {
                int32_t p = (2 * j + 1 < dim)
                            ? q_pairs[j]
                            : (int32_t)(uint16_t)cur_eq16[2 * j];
                __m512i q_p = _mm512_set1_epi32(p);
                __m256i c0_p = _mm256_load_si256((const __m256i *)(blk0 + j * 8));
                __m256i c1_p = _mm256_load_si256((const __m256i *)(blk1 + j * 8));
                __m512i c_pair = _mm512_inserti64x4(_mm512_castsi256_si512(c0_p), c1_p, 1);
                __m512i df = _mm512_max_epi16(
                    _mm512_subs_epi16(q_p, c_pair), v_min_clamp512);
                acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(df, df));
            } // for (; j < num_pairs; j++)

            __m512i acc = _mm512_add_epi32(
                _mm512_add_epi32(acc0, acc1),
                _mm512_add_epi32(acc2, acc3));
            __mmask16 mask = _mm512_cmpgt_epu32_mask(acc, v_cut512);

            int base = sb * 16;
            if (mask == 0xFFFF)
            {
                _mm512_storeu_si512((void *)(clmembflag + base),
                                    _mm512_setzero_si512());
            }
            else
            {
                for (int k = 0; k < 16; k++)
                {
                    clmembflag[base + k] = ((mask & (1 << k)) != 0) ? 0 : 1;
                }
            }
        } // if (!pruned)
    } // for (int sb = 0; sb < num_sb; sb++)

    /* Remainder clusters (< 16) */
    int evaluated = num_sb * 16;
    for (int i = evaluated; i < num_clusters; i++)
    {
        const int16_t *ak_ptr = anchor_matrix + (size_t)i * (size_t)dim;
        uint64_t ssd = eq16_dist_squared_cutoff_i16(cur_eq16, ak_ptr, dim, eq16_ssd_thresh);
        clmembflag[i] = (ssd > eq16_ssd_thresh) ? 0 : 1;
    }

    eq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                      out_num_active, out_pruned_count);
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * eq16_filter_anchor_matrix() - Filter contiguous cluster anchors using EQ16 lower bounds.
 */
void eq16_filter_anchor_matrix(
    const int16_t *restrict cur_eq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                eq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    if (num_clusters <= 0)
    {
        if (out_num_active)
        {
            *out_num_active = 0;
        }
        return;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (anchor_interleaved != NULL &&
        gric_get_simd_level() >= GRIC_SIMD_AVX512 &&
        dim >= 16)
    {
        eq16_filter_anchor_matrix_avx512(cur_eq16,
                                         anchor_matrix,
                                         anchor_interleaved,
                                         num_clusters,
                                         dim,
                                         eq16_ssd_thresh,
                                         clmembflag,
                                         active_clusters,
                                         out_num_active,
                                         out_pruned_count);
        return;
    }
#endif
    if (anchor_interleaved != NULL &&
        gric_get_simd_level() >= GRIC_SIMD_AVX2 &&
        dim >= 16)
    {
        eq16_filter_anchor_matrix_avx2(cur_eq16,
                                       anchor_matrix,
                                       anchor_interleaved,
                                       num_clusters,
                                       dim,
                                       eq16_ssd_thresh,
                                       clmembflag,
                                       active_clusters,
                                       out_num_active,
                                       out_pruned_count);
        return;
    }
#endif

    eq16_filter_anchor_matrix_scalar(cur_eq16,
                                     anchor_matrix,
                                     num_clusters,
                                     dim,
                                     eq16_ssd_thresh,
                                     clmembflag,
                                     active_clusters,
                                     out_num_active,
                                     out_pruned_count);
}

#if GRIC_HAVE_AVX512_TARGET
GRIC_TARGET_AVX512
static void eq16_filter_anchor_matrix_adc_avx512(
    const float   *restrict cur_eq16_adc,
    const int16_t *restrict anchor_matrix,
    const float   *restrict anchor_adc_interleaved,
    int                     num_clusters,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    int num_blocks = num_clusters / 16;
    size_t blk_stride = (size_t)dim * 16;
    __m512 v_cut512 = _mm512_set1_ps(cutoff);
    __m512i v_ones = _mm512_set1_epi32(1);
    __m512i v_zeros = _mm512_setzero_si512();

    #define EQ16_ADC_PREBROADCAST_512 512
    __m512 q_stack[EQ16_ADC_PREBROADCAST_512];
    int pre_cnt = (dim < EQ16_ADC_PREBROADCAST_512)
                  ? (int)dim
                  : EQ16_ADC_PREBROADCAST_512;

    for (int d = 0; d < pre_cnt; d++)
    {
        q_stack[d] = _mm512_set1_ps(cur_eq16_adc[d]);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_blocks >= 512)
#endif
    for (int b = 0; b < num_blocks; b++)
    {
        const float *blk_ptr = anchor_adc_interleaved + (size_t)b * blk_stride;

        if (b + 1 < num_blocks)
        {
            _mm_prefetch((const char *)(blk_ptr + blk_stride), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk_ptr + blk_stride + 64), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk_ptr + blk_stride + 128), _MM_HINT_T0);
        }

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        int pruned = 0;

        long d = 0;
        for (; d + 4 <= pre_cnt; d += 4)
        {
            __m512 q0 = q_stack[d + 0];
            __m512 c0 = _mm512_loadu_ps(blk_ptr + (d + 0) * 16);
            __m512 diff0 = _mm512_sub_ps(q0, c0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m512 q1 = q_stack[d + 1];
            __m512 c1 = _mm512_loadu_ps(blk_ptr + (d + 1) * 16);
            __m512 diff1 = _mm512_sub_ps(q1, c1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m512 q2 = q_stack[d + 2];
            __m512 c2 = _mm512_loadu_ps(blk_ptr + (d + 2) * 16);
            __m512 diff2 = _mm512_sub_ps(q2, c2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m512 q3 = q_stack[d + 3];
            __m512 c3 = _mm512_loadu_ps(blk_ptr + (d + 3) * 16);
            __m512 diff3 = _mm512_sub_ps(q3, c3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

            /* Checkpoints at dims 32, 64, 128, 192, 256, 384 */
            if ((d == 28 || d == 60 || d == 124 || d == 188 || d == 252 || d == 380) &&
                d + 4 < dim)
            {
                __m512 a_sum = _mm512_add_ps(
                    _mm512_add_ps(acc0, acc1),
                    _mm512_add_ps(acc2, acc3)
                );
                if (_mm512_cmp_ps_mask(a_sum, v_cut512, _CMP_GT_OQ) == 0xFFFF)
                {
                    int base = b * 16;
                    if (clmembflag != NULL)
                    {
                        _mm512_storeu_si512((void *)(clmembflag + base), v_zeros);
                    }
                    pruned = 1;
                    break;
                }
            }
        } // for (; d + 4 <= pre_cnt; d += 4)

        if (!pruned)
        {
            for (; d < dim; d++)
            {
                __m512 qd = (d < pre_cnt)
                            ? q_stack[d]
                            : _mm512_set1_ps(cur_eq16_adc[d]);
                __m512 cd = _mm512_loadu_ps(blk_ptr + d * 16);
                __m512 diff = _mm512_sub_ps(qd, cd);
                acc0 = _mm512_fmadd_ps(diff, diff, acc0);
            }

            __m512 total = _mm512_add_ps(
                _mm512_add_ps(acc0, acc1),
                _mm512_add_ps(acc2, acc3)
            );
            __mmask16 mask_gt = _mm512_cmp_ps_mask(total, v_cut512, _CMP_GT_OQ);
            int base = b * 16;
            if (clmembflag != NULL)
            {
                if (mask_gt == 0xFFFF)
                {
                    _mm512_storeu_si512((void *)(clmembflag + base), v_zeros);
                }
                else
                {
                    __m512i v_flags = _mm512_mask_mov_epi32(v_ones, mask_gt, v_zeros);
                    _mm512_storeu_si512((void *)(clmembflag + base), v_flags);
                }
            }
        } // if (!pruned)
    } // for (int b = 0; b < num_blocks; b++)

    #undef EQ16_ADC_PREBROADCAST_512

    /* Remainder clusters (< 16) */
    int evaluated = num_blocks * 16;
    for (int i = evaluated; i < num_clusters; i++)
    {
        const int16_t *ak_ptr = anchor_matrix + (size_t)i * (size_t)dim;
        float dist_sq = eq16_dist_asym_cutoff_f32(cur_eq16_adc, ak_ptr, dim, cutoff);
        if (clmembflag != NULL)
        {
            clmembflag[i] = (dist_sq > cutoff) ? 0 : 1;
        }
    }

    eq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                      out_num_active, out_pruned_count);
}
#endif // GRIC_HAVE_AVX512_TARGET

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
GRIC_TARGET_AVX2
static void eq16_filter_anchor_matrix_adc_avx2(
    const float   *restrict cur_eq16_adc,
    const int16_t *restrict anchor_matrix,
    const float   *restrict anchor_adc_interleaved,
    int                     num_clusters,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    int num_blocks = num_clusters / 16;
    size_t blk_stride = (size_t)dim * 16;
    __m256 v_cut256 = _mm256_set1_ps(cutoff);
    __m256i v_one = _mm256_set1_epi32(1);

    #define EQ16_ADC_PREBROADCAST_256 512
    __m256 q_stack[EQ16_ADC_PREBROADCAST_256];
    int pre_cnt = (dim < EQ16_ADC_PREBROADCAST_256)
                  ? (int)dim
                  : EQ16_ADC_PREBROADCAST_256;

    for (int d = 0; d < pre_cnt; d++)
    {
        q_stack[d] = _mm256_set1_ps(cur_eq16_adc[d]);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_blocks >= 512)
#endif
    for (int b = 0; b < num_blocks; b++)
    {
        const float *blk_ptr = anchor_adc_interleaved + (size_t)b * blk_stride;

        if (b + 1 < num_blocks)
        {
            _mm_prefetch((const char *)(blk_ptr + blk_stride), _MM_HINT_T0);
            _mm_prefetch((const char *)(blk_ptr + blk_stride + 64), _MM_HINT_T0);
        }

        __m256 acc0_lo = _mm256_setzero_ps();
        __m256 acc0_hi = _mm256_setzero_ps();
        __m256 acc1_lo = _mm256_setzero_ps();
        __m256 acc1_hi = _mm256_setzero_ps();
        __m256 acc2_lo = _mm256_setzero_ps();
        __m256 acc2_hi = _mm256_setzero_ps();
        __m256 acc3_lo = _mm256_setzero_ps();
        __m256 acc3_hi = _mm256_setzero_ps();
        int pruned = 0;

        long d = 0;
        for (; d + 4 <= pre_cnt; d += 4)
        {
            __m256 q0 = q_stack[d + 0];
            __m256 c0_lo = _mm256_loadu_ps(blk_ptr + (d + 0) * 16 + 0);
            __m256 c0_hi = _mm256_loadu_ps(blk_ptr + (d + 0) * 16 + 8);
            __m256 diff0_lo = _mm256_sub_ps(q0, c0_lo);
            __m256 diff0_hi = _mm256_sub_ps(q0, c0_hi);
            acc0_lo = _mm256_fmadd_ps(diff0_lo, diff0_lo, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(diff0_hi, diff0_hi, acc0_hi);

            __m256 q1 = q_stack[d + 1];
            __m256 c1_lo = _mm256_loadu_ps(blk_ptr + (d + 1) * 16 + 0);
            __m256 c1_hi = _mm256_loadu_ps(blk_ptr + (d + 1) * 16 + 8);
            __m256 diff1_lo = _mm256_sub_ps(q1, c1_lo);
            __m256 diff1_hi = _mm256_sub_ps(q1, c1_hi);
            acc1_lo = _mm256_fmadd_ps(diff1_lo, diff1_lo, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(diff1_hi, diff1_hi, acc1_hi);

            __m256 q2 = q_stack[d + 2];
            __m256 c2_lo = _mm256_loadu_ps(blk_ptr + (d + 2) * 16 + 0);
            __m256 c2_hi = _mm256_loadu_ps(blk_ptr + (d + 2) * 16 + 8);
            __m256 diff2_lo = _mm256_sub_ps(q2, c2_lo);
            __m256 diff2_hi = _mm256_sub_ps(q2, c2_hi);
            acc2_lo = _mm256_fmadd_ps(diff2_lo, diff2_lo, acc2_lo);
            acc2_hi = _mm256_fmadd_ps(diff2_hi, diff2_hi, acc2_hi);

            __m256 q3 = q_stack[d + 3];
            __m256 c3_lo = _mm256_loadu_ps(blk_ptr + (d + 3) * 16 + 0);
            __m256 c3_hi = _mm256_loadu_ps(blk_ptr + (d + 3) * 16 + 8);
            __m256 diff3_lo = _mm256_sub_ps(q3, c3_lo);
            __m256 diff3_hi = _mm256_sub_ps(q3, c3_hi);
            acc3_lo = _mm256_fmadd_ps(diff3_lo, diff3_lo, acc3_lo);
            acc3_hi = _mm256_fmadd_ps(diff3_hi, diff3_hi, acc3_hi);

            /* Checkpoints at dims 32, 64, 128, 192, 256, 384 */
            if ((d == 28 || d == 60 || d == 124 || d == 188 || d == 252 || d == 380) &&
                d + 4 < dim)
            {
                __m256 sum_lo = _mm256_add_ps(
                    _mm256_add_ps(acc0_lo, acc1_lo),
                    _mm256_add_ps(acc2_lo, acc3_lo)
                );
                __m256 sum_hi = _mm256_add_ps(
                    _mm256_add_ps(acc0_hi, acc1_hi),
                    _mm256_add_ps(acc2_hi, acc3_hi)
                );
                int m_lo = _mm256_movemask_ps(_mm256_cmp_ps(sum_lo, v_cut256, _CMP_GT_OQ));
                int m_hi = _mm256_movemask_ps(_mm256_cmp_ps(sum_hi, v_cut256, _CMP_GT_OQ));
                if (m_lo == 0xFF && m_hi == 0xFF)
                {
                    int base = b * 16;
                    if (clmembflag != NULL)
                    {
                        _mm256_storeu_si256((__m256i *)(clmembflag + base + 0),
                                            _mm256_setzero_si256());
                        _mm256_storeu_si256((__m256i *)(clmembflag + base + 8),
                                            _mm256_setzero_si256());
                    }
                    pruned = 1;
                    break;
                }
            }
        } // for (; d + 4 <= pre_cnt; d += 4)

        if (!pruned)
        {
            for (; d < dim; d++)
            {
                __m256 qd = (d < pre_cnt)
                            ? q_stack[d]
                            : _mm256_set1_ps(cur_eq16_adc[d]);
                __m256 cd_lo = _mm256_loadu_ps(blk_ptr + d * 16 + 0);
                __m256 cd_hi = _mm256_loadu_ps(blk_ptr + d * 16 + 8);
                __m256 diff_lo = _mm256_sub_ps(qd, cd_lo);
                __m256 diff_hi = _mm256_sub_ps(qd, cd_hi);
                acc0_lo = _mm256_fmadd_ps(diff_lo, diff_lo, acc0_lo);
                acc0_hi = _mm256_fmadd_ps(diff_hi, diff_hi, acc0_hi);
            }

            __m256 sum_lo = _mm256_add_ps(
                _mm256_add_ps(acc0_lo, acc1_lo),
                _mm256_add_ps(acc2_lo, acc3_lo)
            );
            __m256 sum_hi = _mm256_add_ps(
                _mm256_add_ps(acc0_hi, acc1_hi),
                _mm256_add_ps(acc2_hi, acc3_hi)
            );
            __m256 cmp_lo = _mm256_cmp_ps(sum_lo, v_cut256, _CMP_GT_OQ);
            __m256 cmp_hi = _mm256_cmp_ps(sum_hi, v_cut256, _CMP_GT_OQ);
            int m_lo = _mm256_movemask_ps(cmp_lo);
            int m_hi = _mm256_movemask_ps(cmp_hi);
            int base = b * 16;
            if (clmembflag != NULL)
            {
                if (m_lo == 0xFF && m_hi == 0xFF)
                {
                    _mm256_storeu_si256((__m256i *)(clmembflag + base + 0),
                                        _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i *)(clmembflag + base + 8),
                                        _mm256_setzero_si256());
                }
                else
                {
                    __m256i flag_lo = _mm256_andnot_si256(
                        _mm256_castps_si256(cmp_lo), v_one
                    );
                    __m256i flag_hi = _mm256_andnot_si256(
                        _mm256_castps_si256(cmp_hi), v_one
                    );
                    _mm256_storeu_si256((__m256i *)(clmembflag + base + 0), flag_lo);
                    _mm256_storeu_si256((__m256i *)(clmembflag + base + 8), flag_hi);
                }
            }
        } // if (!pruned)
    } // for (int b = 0; b < num_blocks; b++)

    #undef EQ16_ADC_PREBROADCAST_256

    /* Remainder clusters (< 16) */
    int evaluated = num_blocks * 16;
    for (int i = evaluated; i < num_clusters; i++)
    {
        const int16_t *ak_ptr = anchor_matrix + (size_t)i * (size_t)dim;
        float dist_sq = eq16_dist_asym_cutoff_f32(cur_eq16_adc, ak_ptr, dim, cutoff);
        if (clmembflag != NULL)
        {
            clmembflag[i] = (dist_sq > cutoff) ? 0 : 1;
        }
    }

    eq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                      out_num_active, out_pruned_count);
}
#endif // AVX2

/**
 * eq16_filter_anchor_matrix_adc() - Filter cluster anchors using Asymmetric Distance Computation.
 * @cur_eq16_adc:           Normalized float query vector [dim].
 * @anchor_matrix:          Contiguous [num_clusters x dim] array of int16_t anchor coordinates.
 * @anchor_adc_interleaved: Optional Block-16 interleaved float matrix [num_blocks x dim x 16].
 * @num_clusters:           Total number of clusters to filter.
 * @dim:                    Dimension of vectors.
 * @cutoff:                 Normalized ADC distance squared cutoff threshold.
 * @clmembflag:             In/out membership flags (1 = candidate active, 0 = pruned).
 * @active_clusters:        Output array of surviving active cluster indices.
 * @out_num_active:         Output count of surviving active clusters.
 * @out_pruned_count:       Output count of clusters pruned during this filtering call.
 */
void eq16_filter_anchor_matrix_adc(
    const float   *restrict cur_eq16_adc,
    const int16_t *restrict anchor_matrix,
    const float   *restrict anchor_adc_interleaved,
    int                     num_clusters,
    long                    dim,
    float                   cutoff,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    long          *restrict out_pruned_count)
{
    if (num_clusters <= 0)
    {
        if (out_num_active != NULL)
        {
            *out_num_active = 0;
        }
        return;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (anchor_adc_interleaved != NULL &&
        gric_get_simd_level() >= GRIC_SIMD_AVX512 &&
        dim >= 16)
    {
        eq16_filter_anchor_matrix_adc_avx512(cur_eq16_adc,
                                             anchor_matrix,
                                             anchor_adc_interleaved,
                                             num_clusters,
                                             dim,
                                             cutoff,
                                             clmembflag,
                                             active_clusters,
                                             out_num_active,
                                             out_pruned_count);
        return;
    }
#endif
    if (anchor_adc_interleaved != NULL &&
        gric_get_simd_level() >= GRIC_SIMD_AVX2 &&
        dim >= 16)
    {
        eq16_filter_anchor_matrix_adc_avx2(cur_eq16_adc,
                                           anchor_matrix,
                                           anchor_adc_interleaved,
                                           num_clusters,
                                           dim,
                                           cutoff,
                                           clmembflag,
                                           active_clusters,
                                           out_num_active,
                                           out_pruned_count);
        return;
    }
#endif

    int next_active_count = 0;
    long pruned_this_step = 0;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(+:pruned_this_step) if(num_clusters >= 1024)
#endif
    for (int c = 0; c < num_clusters; c++)
    {
        const int16_t *anchor_ptr = anchor_matrix + (size_t)c * (size_t)dim;
        float dist_sq = eq16_dist_asym_cutoff_f32(cur_eq16_adc, anchor_ptr, dim, cutoff);

        if (dist_sq > cutoff)
        {
            if (clmembflag != NULL)
            {
                clmembflag[c] = 0;
            }
            pruned_this_step++;
        }
        else
        {
            if (clmembflag != NULL)
            {
                clmembflag[c] = 1;
            }
        }
    }

    if (active_clusters != NULL && clmembflag != NULL)
    {
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
        {
            int num_active = 0;
            long dummy_pruned = 0;
            eq16_compact_active_clusters_avx2(
                clmembflag, active_clusters, num_clusters, &num_active, &dummy_pruned
            );
            next_active_count = num_active;
        }
        else
#endif
        {
            for (int c = 0; c < num_clusters; c++)
            {
                if (clmembflag[c])
                {
                    active_clusters[next_active_count++] = c;
                }
            }
        }
    }
    else if (active_clusters != NULL)
    {
        next_active_count = num_clusters - (int)pruned_this_step;
    }

    if (out_num_active != NULL)
    {
        *out_num_active = next_active_count;
    }
    if (out_pruned_count != NULL)
    {
        *out_pruned_count += pruned_this_step;
    }
}

/**
 * eq16_save_sidecar() - Save quantized dataset buffer and parameters to a binary .eq16 file.
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
void eq16_print_checkpoint_stats(void)
{
#ifdef EQ16_PROFILE_CHECKPOINTS
    uint64_t total = g_eq16_exit_none;
    for (int p = 0; p < 512; p++)
    {
        total += g_eq16_exit_pairs[p];
    }
    if (total == 0)
    {
        return;
    }

    printf("  EQ16 AVX2 Block Early Exit Checkpoints (Total: %lu blocks):\n", total);
    for (int p = 0; p < 512; p++)
    {
        if (g_eq16_exit_pairs[p] > 0)
        {
            printf("    Exit at Dim %3d: %10lu (%5.1f%%)\n",
                   p * 2,
                   g_eq16_exit_pairs[p],
                   100.0 * (double)g_eq16_exit_pairs[p] / (double)total);
        }
    }
    printf("    Evaluated to end:%10lu (%5.1f%%)\n",
           g_eq16_exit_none, 100.0 * (double)g_eq16_exit_none / (double)total);
#endif
}

/**
 * eq16_fastscan_32x_adc_scalar() - Scalar fallback for 32-candidate EQ16 ADC FastScan.
 * @q_adc:        Normalized float query coordinates.
 * @block_coords: Transposed int16 candidate block coordinates [dim * 32].
 * @dim:          Vector dimension.
 * @cutoff_f:     Cutoff squared distance threshold.
 *
 * Return: 32-bit pass bitmask.
 */
static inline uint32_t eq16_fastscan_32x_adc_scalar(
    const float   *restrict q_adc,
    const int16_t *restrict block_coords,
    long                    dim,
    float                   cutoff_f)
{
    float acc[32];
    for (int i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    for (long d = 0; d < dim; d++)
    {
        float qd = q_adc[d];
        const int16_t *cd_ptr = block_coords + d * 32;
        for (int i = 0; i < 32; i++)
        {
            float diff = qd - (float)cd_ptr[i];
            acc[i] += diff * diff;
        }

        if (d == 31 || d == 63 || d == 127 || d == 255)
        {
            int any_alive = 0;
            for (int i = 0; i < 32; i++)
            {
                if (acc[i] <= cutoff_f)
                {
                    any_alive = 1;
                    break;
                }
            }
            if (!any_alive)
            {
                return 0;
            }
        }
    } // for (long d = 0; d < dim; d++)

    uint32_t pass_mask = 0;
    for (int i = 0; i < 32; i++)
    {
        if (acc[i] <= cutoff_f)
        {
            pass_mask |= (1U << i);
        }
    }

    return pass_mask;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * eq16_fastscan_32x_adc_avx2() - AVX2 256-bit SIMD kernel for 32-candidate EQ16 ADC.
 * @q_adc:        Normalized float query coordinates.
 * @block_coords: Transposed int16 candidate block coordinates [dim * 32].
 * @dim:          Vector dimension.
 * @cutoff_f:     Cutoff squared distance threshold.
 *
 * Return: 32-bit pass bitmask.
 */
static inline uint32_t eq16_fastscan_32x_adc_avx2(
    const float   *restrict q_adc,
    const int16_t *restrict block_coords,
    long                    dim,
    float                   cutoff_f)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (long d = 0; d < dim; d++)
    {
        __m256 v_qd = _mm256_set1_ps(q_adc[d]);
        const int16_t *cd_ptr = block_coords + d * 32;

        __m256i c_lo = _mm256_loadu_si256((const __m256i *)(const void *)cd_ptr);
        __m256i c0_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c_lo));
        __m256i c1_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c_lo, 1));
        __m256 diff0 = _mm256_sub_ps(v_qd, _mm256_cvtepi32_ps(c0_i32));
        __m256 diff1 = _mm256_sub_ps(v_qd, _mm256_cvtepi32_ps(c1_i32));
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        __m256i c_hi = _mm256_loadu_si256((const __m256i *)(const void *)(cd_ptr + 16));
        __m256i c2_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c_hi));
        __m256i c3_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c_hi, 1));
        __m256 diff2 = _mm256_sub_ps(v_qd, _mm256_cvtepi32_ps(c2_i32));
        __m256 diff3 = _mm256_sub_ps(v_qd, _mm256_cvtepi32_ps(c3_i32));
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        if (d == 31 || d == 63 || d == 127 || d == 255)
        {
            __m256 v_cut = _mm256_set1_ps(cutoff_f);
            __m256 cmp0 = _mm256_cmp_ps(acc0, v_cut, _CMP_LE_OQ);
            __m256 cmp1 = _mm256_cmp_ps(acc1, v_cut, _CMP_LE_OQ);
            __m256 cmp2 = _mm256_cmp_ps(acc2, v_cut, _CMP_LE_OQ);
            __m256 cmp3 = _mm256_cmp_ps(acc3, v_cut, _CMP_LE_OQ);
            __m256 any_le = _mm256_or_ps(_mm256_or_ps(cmp0, cmp1),
                                         _mm256_or_ps(cmp2, cmp3));
            if (_mm256_movemask_ps(any_le) == 0)
            {
                return 0;
            }
        }
    } // for (long d = 0; d < dim; d++)

    __m256 v_cut = _mm256_set1_ps(cutoff_f);
    int m0 = _mm256_movemask_ps(_mm256_cmp_ps(acc0, v_cut, _CMP_LE_OQ));
    int m1 = _mm256_movemask_ps(_mm256_cmp_ps(acc1, v_cut, _CMP_LE_OQ));
    int m2 = _mm256_movemask_ps(_mm256_cmp_ps(acc2, v_cut, _CMP_LE_OQ));
    int m3 = _mm256_movemask_ps(_mm256_cmp_ps(acc3, v_cut, _CMP_LE_OQ));

    return (uint32_t)m0 | ((uint32_t)m1 << 8) |
           ((uint32_t)m2 << 16) | ((uint32_t)m3 << 24);
}
#endif // x86/x64 AVX2

#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * eq16_fastscan_32x_adc_avx512() - AVX-512 512-bit SIMD kernel for 32-candidate EQ16 ADC.
 * @q_adc:        Normalized float query coordinates.
 * @block_coords: Transposed int16 candidate block coordinates [dim * 32].
 * @dim:          Vector dimension.
 * @cutoff_f:     Cutoff squared distance threshold.
 *
 * Return: 32-bit pass bitmask.
 */
static inline uint32_t eq16_fastscan_32x_adc_avx512(
    const float   *restrict q_adc,
    const int16_t *restrict block_coords,
    long                    dim,
    float                   cutoff_f)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 v_cut = _mm512_set1_ps(cutoff_f);

    for (long d = 0; d < dim; d++)
    {
        __m512 v_qd = _mm512_set1_ps(q_adc[d]);
        const int16_t *cd_ptr = block_coords + d * 32;

        __m256i c0 = _mm256_loadu_si256((const __m256i *)(const void *)cd_ptr);
        __m512 c0_f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c0));
        __m512 diff0 = _mm512_sub_ps(v_qd, c0_f);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        __m256i c1 = _mm256_loadu_si256((const __m256i *)(const void *)(cd_ptr + 16));
        __m512 c1_f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c1));
        __m512 diff1 = _mm512_sub_ps(v_qd, c1_f);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        if (d == 31 || d == 63 || d == 127 || d == 255)
        {
            __mmask16 k0 = _mm512_cmp_ps_mask(acc0, v_cut, _CMP_LE_OQ);
            __mmask16 k1 = _mm512_cmp_ps_mask(acc1, v_cut, _CMP_LE_OQ);
            if ((k0 | k1) == 0)
            {
                return 0;
            }
        }
    } // for (long d = 0; d < dim; d++)

    __mmask16 k0 = _mm512_cmp_ps_mask(acc0, v_cut, _CMP_LE_OQ);
    __mmask16 k1 = _mm512_cmp_ps_mask(acc1, v_cut, _CMP_LE_OQ);

    return (uint32_t)k0 | ((uint32_t)k1 << 16);
}
#endif // AVX-512

/**
 * eq16_fastscan_32x_adc() - Evaluate EQ16 ADC distance for 32 candidates in SIMD.
 * @q_adc:        Normalized float query coordinates.
 * @block_coords: Transposed int16 candidate block coordinates [dim * 32].
 * @dim:          Vector dimension.
 * @cutoff_f:     Cutoff squared distance threshold.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i distance <= cutoff_f.
 */
uint32_t eq16_fastscan_32x_adc(
    const float   *restrict q_adc,
    const int16_t *restrict block_coords,
    long                    dim,
    float                   cutoff_f)
{
#if !defined(__CUDACC__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return eq16_fastscan_32x_adc_avx512(q_adc, block_coords, dim, cutoff_f);
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return eq16_fastscan_32x_adc_avx2(q_adc, block_coords, dim, cutoff_f);
    }
#endif

    return eq16_fastscan_32x_adc_scalar(q_adc, block_coords, dim, cutoff_f);
}

/**
 * eq16_fastscan_32x_i16() - Evaluate EQ16 SDC squared distance for 32 candidates in SIMD.
 * @query_eq16:   Quantized int16 query coordinates.
 * @block_coords: Transposed int16 candidate block coordinates [dim * 32].
 * @dim:          Vector dimension.
 * @ssd_cutoff:   Cutoff sum-of-squared-differences threshold.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i SSD <= ssd_cutoff.
 */
uint32_t eq16_fastscan_32x_i16(
    const int16_t *restrict query_eq16,
    const int16_t *restrict block_coords,
    long                    dim,
    uint64_t                ssd_cutoff)
{
    return sq16_fastscan_32x(query_eq16, block_coords, dim, ssd_cutoff);
}

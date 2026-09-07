/**
 * @file scalar_quant.c
 * @brief 8-bit Scalar Quantization (SQ8) implementation with SIMD kernels.
 */

#include "scalar_quant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/**
 * sq8_init_params() - Initialize SQ8 parameters from known min and max values.
 * @params:  Pointer to SQ8Params structure to initialize.
 * @min_val: Minimum coordinate value.
 * @max_val: Maximum coordinate value.
 * @dim:     Vector dimension.
 */
void sq8_init_params(
    SQ8Params *params,
    float      min_val,
    float      max_val,
    long       dim)
{
    if (params == NULL)
    {
        return;
    }

    if (max_val <= min_val)
    {
        max_val = min_val + 1.0f;
    }

    params->min_val = min_val;
    params->max_val = max_val;
    params->scale = (max_val - min_val) / 255.0f;
    params->inv_scale = 255.0f / (max_val - min_val);
    params->dim = dim;
    params->err_radius = sqrtf((float)dim) * params->scale * 0.5f;
}

/**
 * sq8_calibrate_float() - Calibrate SQ8 parameters by scanning a float array.
 * @params:       Pointer to SQ8Params structure to populate.
 * @data:         Pointer to input float array.
 * @num_elements: Total number of float values in the array.
 * @dim:          Vector dimension per frame.
 */
void sq8_calibrate_float(
    SQ8Params   *params,
    const float *data,
    long         num_elements,
    long         dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    float min_v = data[0];
    float max_v = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        float val = data[i];
        if (val < min_v)
        {
            min_v = val;
        }
        if (val > max_v)
        {
            max_v = val;
        }
    } // for (long i = 1; i < num_elements; i++)

    sq8_init_params(params, min_v, max_v, dim);
}

/**
 * sq8_calibrate_double() - Calibrate SQ8 parameters by scanning a double array.
 * @params:       Pointer to SQ8Params structure to populate.
 * @data:         Pointer to input double array.
 * @num_elements: Total number of double values in the array.
 * @dim:          Vector dimension per frame.
 */
void sq8_calibrate_double(
    SQ8Params    *params,
    const double *data,
    long          num_elements,
    long          dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    double min_v = data[0];
    double max_v = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        double val = data[i];
        if (val < min_v)
        {
            min_v = val;
        }
        if (val > max_v)
        {
            max_v = val;
        }
    } // for (long i = 1; i < num_elements; i++)

    sq8_init_params(params, (float)min_v, (float)max_v, dim);
}

/**
 * sq8_quantize_float() - Quantize a single-precision float frame to uint8.
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination uint8 vector [dim].
 * @params: Pointer to initialized SQ8Params.
 */
void sq8_quantize_float(
    const float     *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params)
{
    long dim = params->dim;
    float min_val = params->min_val;
    float inv_scale = params->inv_scale;
    long i = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256 v_min = _mm256_set1_ps(min_val);
    __m256 v_inv = _mm256_set1_ps(inv_scale);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_max = _mm256_set1_ps(255.0f);

    for (; i <= dim - 8; i += 8)
    {
        __m256 in_vec = _mm256_loadu_ps(&src[i]);
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(in_vec, v_min), v_inv);
        norm = _mm256_add_ps(norm, v_half);
        norm = _mm256_max_ps(v_zero, _mm256_min_ps(norm, v_max));
        __m256i i32 = _mm256_cvttps_epi32(norm);

        // Pack 8x 32-bit integers into 8x 8-bit unsigned integers
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i p16 = _mm_packs_epi32(lo, hi);
        __m128i p8 = _mm_packus_epi16(p16, p16);

        // Store 8 bytes
        uint64_t bytes8 = (uint64_t)_mm_cvtsi128_si64(p8);
        memcpy(&dst[i], &bytes8, 8);
    } // for (; i <= dim - 8; i += 8)
#endif

    for (; i < dim; i++)
    {
        float val = (src[i] - min_val) * inv_scale + 0.5f;
        if (val < 0.0f)
        {
            val = 0.0f;
        }
        else if (val > 255.0f)
        {
            val = 255.0f;
        }
        dst[i] = (uint8_t)val;
    } // for (; i < dim; i++)
}

/**
 * sq8_quantize_double() - Quantize a double-precision frame to uint8.
 * @src:    Pointer to source double vector [dim].
 * @dst:    Pointer to destination uint8 vector [dim].
 * @params: Pointer to initialized SQ8Params.
 */
void sq8_quantize_double(
    const double    *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params)
{
    long dim = params->dim;
    double min_val = (double)params->min_val;
    double inv_scale = (double)params->inv_scale;

    for (long i = 0; i < dim; i++)
    {
        double val = (src[i] - min_val) * inv_scale + 0.5;
        if (val < 0.0)
        {
            val = 0.0;
        }
        else if (val > 255.0)
        {
            val = 255.0;
        }
        dst[i] = (uint8_t)val;
    } // for (long i = 0; i < dim; i++)
}

/**
 * sq8_dist_squared_u8() - Compute sum of squared differences between two uint8 vectors.
 * @a:   Pointer to first uint8 array [dim].
 * @b:   Pointer to second uint8 array [dim].
 * @dim: Vector dimension.
 *
 * Return: Total sum of squared differences as uint64_t.
 */
uint64_t sq8_dist_squared_u8(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    long                    dim)
{
    uint64_t total = 0;
    long i = 0;

#if defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    const __m512i zero512 = _mm512_setzero_si512();
    __m512i sum_vec512 = _mm512_setzero_si512();
    long chunk_count512 = 0;

    for (; i <= dim - 64; i += 64)
    {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));

        __m512i va_lo = _mm512_unpacklo_epi8(va, zero512);
        __m512i vb_lo = _mm512_unpacklo_epi8(vb, zero512);
        __m512i diff_lo = _mm512_sub_epi16(va_lo, vb_lo);

        __m512i va_hi = _mm512_unpackhi_epi8(va, zero512);
        __m512i vb_hi = _mm512_unpackhi_epi8(vb, zero512);
        __m512i diff_hi = _mm512_sub_epi16(va_hi, vb_hi);

        __m512i prod_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        __m512i prod_hi = _mm512_madd_epi16(diff_hi, diff_hi);

        sum_vec512 = _mm512_add_epi32(sum_vec512, prod_lo);
        sum_vec512 = _mm512_add_epi32(sum_vec512, prod_hi);

        chunk_count512 += 64;
        if (chunk_count512 >= 16384)
        {
            total += (uint64_t)_mm512_reduce_add_epi32(sum_vec512);
            sum_vec512 = _mm512_setzero_si512();
            chunk_count512 = 0;
        }
    } // for (; i <= dim - 64; i += 64)

    total += (uint64_t)_mm512_reduce_add_epi32(sum_vec512);
#elif defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    const __m256i zero256 = _mm256_setzero_si256();
    __m256i sum_vec256 = _mm256_setzero_si256();
    long chunk_count256 = 0;

    for (; i <= dim - 32; i += 32)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));

        __m256i va_lo = _mm256_unpacklo_epi8(va, zero256);
        __m256i vb_lo = _mm256_unpacklo_epi8(vb, zero256);
        __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);

        __m256i va_hi = _mm256_unpackhi_epi8(va, zero256);
        __m256i vb_hi = _mm256_unpackhi_epi8(vb, zero256);
        __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

        __m256i prod_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        __m256i prod_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum_vec256 = _mm256_add_epi32(sum_vec256, prod_lo);
        sum_vec256 = _mm256_add_epi32(sum_vec256, prod_hi);

        chunk_count256 += 32;
        if (chunk_count256 >= 16384)
        {
            __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum_vec256),
                                         _mm256_extracti128_si256(sum_vec256, 1));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
            s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
            total += (uint64_t)(uint32_t)_mm_cvtsi128_si32(s128);
            sum_vec256 = _mm256_setzero_si256();
            chunk_count256 = 0;
        }
    } // for (; i <= dim - 32; i += 32)

    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum_vec256),
                                 _mm256_extracti128_si256(sum_vec256, 1));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
    total += (uint64_t)(uint32_t)_mm_cvtsi128_si32(s128);
#endif

    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        total += (uint64_t)(diff * diff);
    } // for (; i < dim; i++)

    return total;
}

/**
 * sq8_compute_lower_bound() - Guaranteed metric lower bound between two quantized vectors.
 * @a:       Pointer to first uint8 array.
 * @b:       Pointer to second uint8 array.
 * @params:  Pointer to initialized SQ8Params.
 * @epsilon: Relative relaxation factor (0.0 for exact, > 0.0 for (1+eps)-ANN).
 *
 * Return: Lower-bound distance as double.
 */
double sq8_compute_lower_bound(
    const uint8_t   *restrict a,
    const uint8_t   *restrict b,
    const SQ8Params *restrict params,
    double                    epsilon)
{
    uint64_t ssd = sq8_dist_squared_u8(a, b, params->dim);
    double q_dist = (double)params->scale * sqrt((double)ssd);
    double lb = q_dist - (2.0 * (double)params->err_radius);

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
 * sq8_save_sidecar() - Save quantized dataset buffer and parameters to a binary .sq8 file.
 * @filepath:   Path to destination .sq8 file.
 * @params:     Pointer to SQ8Params.
 * @data:       Pointer to quantized byte array [num_frames * dim].
 * @num_frames: Number of frames in the dataset.
 *
 * Return: 0 on success, -1 on I/O error.
 */
int sq8_save_sidecar(
    const char      *filepath,
    const SQ8Params *params,
    const uint8_t   *data,
    long             num_frames)
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

    char magic[8];
    memcpy(magic, SQ8_FILE_MAGIC, 8);
    if (fwrite(magic, 1, 8, fp) != 8)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[5] = {
        params->min_val,
        params->max_val,
        params->scale,
        params->inv_scale,
        params->err_radius
    };
    if (fwrite(meta_f, sizeof(float), 5, fp) != 5)
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

    size_t total_bytes = (size_t)params->dim * (size_t)num_frames;
    if (fwrite(data, 1, total_bytes, fp) != total_bytes)
    {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/**
 * sq8_load_sidecar() - Load quantized dataset buffer and parameters from a binary .sq8 file.
 * @filepath:   Path to source .sq8 file.
 * @params:     Pointer to SQ8Params to receive metadata.
 * @data:       Output pointer to dynamically allocated uint8 array [num_frames * dim].
 * @num_frames: Output pointer to received frame count.
 *
 * Return: 0 on success, -1 on I/O or format error.
 */
int sq8_load_sidecar(
    const char *filepath,
    SQ8Params  *params,
    uint8_t   **data,
    long       *num_frames)
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
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, SQ8_FILE_MAGIC, 8) != 0)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[5];
    if (fread(meta_f, sizeof(float), 5, fp) != 5)
    {
        fclose(fp);
        return -1;
    }

    int64_t meta_i[2];
    if (fread(meta_i, sizeof(int64_t), 2, fp) != 2)
    {
        fclose(fp);
        return -1;
    }

    params->min_val = meta_f[0];
    params->max_val = meta_f[1];
    params->scale = meta_f[2];
    params->inv_scale = meta_f[3];
    params->err_radius = meta_f[4];
    params->dim = (long)meta_i[0];
    *num_frames = (long)meta_i[1];

    size_t total_bytes = (size_t)params->dim * (size_t)(*num_frames);
    uint8_t *buf = (uint8_t *)malloc(total_bytes);
    if (buf == NULL)
    {
        fclose(fp);
        return -1;
    }

    if (fread(buf, 1, total_bytes, fp) != total_bytes)
    {
        free(buf);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *data = buf;
    return 0;
}

/**
 * sq16_init_params() - Initialize SQ16 parameters from known min and max values.
 * @params:  Pointer to SQ16Params structure to initialize.
 * @min_val: Minimum coordinate value.
 * @max_val: Maximum coordinate value.
 * @dim:     Vector dimension.
 */
void sq16_init_params(
    SQ16Params *params,
    float       min_val,
    float       max_val,
    long        dim)
{
    if (params == NULL)
    {
        return;
    }

    if (max_val <= min_val)
    {
        max_val = min_val + 1.0f;
    }

    params->min_val = min_val;
    params->max_val = max_val;
    params->scale = (max_val - min_val) / 32767.0f;
    params->inv_scale = 32767.0f / (max_val - min_val);
    params->dim = dim;
    params->err_radius = sqrtf((float)dim) * params->scale * 0.5f;
}

/**
 * sq16_calibrate_float() - Calibrate SQ16 parameters by scanning a float array.
 * @params:       Pointer to SQ16Params structure to populate.
 * @data:         Pointer to input float array.
 * @num_elements: Total number of float values in the array.
 * @dim:          Vector dimension per frame.
 */
void sq16_calibrate_float(
    SQ16Params  *params,
    const float *data,
    long         num_elements,
    long         dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    float min_v = data[0];
    float max_v = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        float val = data[i];
        if (val < min_v)
        {
            min_v = val;
        }
        if (val > max_v)
        {
            max_v = val;
        }
    } // for (long i = 1; i < num_elements; i++)

    sq16_init_params(params, min_v, max_v, dim);
}

/**
 * sq16_calibrate_double() - Calibrate SQ16 parameters by scanning a double array.
 * @params:       Pointer to SQ16Params structure to populate.
 * @data:         Pointer to input double array.
 * @num_elements: Total number of double values in the array.
 * @dim:          Vector dimension per frame.
 */
void sq16_calibrate_double(
    SQ16Params   *params,
    const double *data,
    long          num_elements,
    long          dim)
{
    if (params == NULL || data == NULL || num_elements <= 0)
    {
        return;
    }

    double min_v = data[0];
    double max_v = data[0];

    for (long i = 1; i < num_elements; i++)
    {
        double val = data[i];
        if (val < min_v)
        {
            min_v = val;
        }
        if (val > max_v)
        {
            max_v = val;
        }
    } // for (long i = 1; i < num_elements; i++)

    sq16_init_params(params, (float)min_v, (float)max_v, dim);
}

/**
 * sq16_quantize_float() - Quantize a single-precision float frame to int16 [0, 32767].
 * @src:    Pointer to source float vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized SQ16Params.
 */
void sq16_quantize_float(
    const float      *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params)
{
    long dim = params->dim;
    float min_val = params->min_val;
    float inv_scale = params->inv_scale;
    long i = 0;

#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256 v_min = _mm256_set1_ps(min_val);
    __m256 v_inv = _mm256_set1_ps(inv_scale);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_max = _mm256_set1_ps(32767.0f);

    for (; i <= dim - 16; i += 16)
    {
        __m256 in0 = _mm256_loadu_ps(&src[i]);
        __m256 norm0 = _mm256_mul_ps(_mm256_sub_ps(in0, v_min), v_inv);
        norm0 = _mm256_add_ps(norm0, v_half);
        norm0 = _mm256_max_ps(v_zero, _mm256_min_ps(norm0, v_max));
        __m256i i32_0 = _mm256_cvttps_epi32(norm0);

        __m256 in1 = _mm256_loadu_ps(&src[i + 8]);
        __m256 norm1 = _mm256_mul_ps(_mm256_sub_ps(in1, v_min), v_inv);
        norm1 = _mm256_add_ps(norm1, v_half);
        norm1 = _mm256_max_ps(v_zero, _mm256_min_ps(norm1, v_max));
        __m256i i32_1 = _mm256_cvttps_epi32(norm1);

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
        float val = (src[i] - min_val) * inv_scale + 0.5f;
        if (val < 0.0f)
        {
            val = 0.0f;
        }
        else if (val > 32767.0f)
        {
            val = 32767.0f;
        }
        dst[i] = (int16_t)val;
    } // for (; i < dim; i++)
}

/**
 * sq16_quantize_double() - Quantize a double-precision frame to int16 [0, 32767].
 * @src:    Pointer to source double vector [dim].
 * @dst:    Pointer to destination int16 vector [dim].
 * @params: Pointer to initialized SQ16Params.
 */
void sq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params)
{
    long dim = params->dim;
    double min_val = (double)params->min_val;
    double inv_scale = (double)params->inv_scale;

    for (long i = 0; i < dim; i++)
    {
        double val = (src[i] - min_val) * inv_scale + 0.5;
        if (val < 0.0)
        {
            val = 0.0;
        }
        else if (val > 32767.0)
        {
            val = 32767.0;
        }
        dst[i] = (int16_t)val;
    } // for (long i = 0; i < dim; i++)
}

/**
 * sq16_dist_squared_i16() - Compute sum of squared differences between two int16 vectors.
 * @a:   Pointer to first int16 array [dim].
 * @b:   Pointer to second int16 array [dim].
 * @dim: Vector dimension.
 *
 * Return: Total sum of squared differences as uint64_t.
 */
uint64_t sq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim)
{
    uint64_t total = 0;
    long i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m512i sum_vec512_0 = _mm512_setzero_si512();
    __m512i sum_vec512_1 = _mm512_setzero_si512();

    for (; i <= dim - 32; i += 32)
    {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        __m512i diff = _mm512_sub_epi16(va, vb);
        __m512i prod = _mm512_madd_epi16(diff, diff);

        __m256i prod_lo = _mm512_castsi512_si256(prod);
        __m256i prod_hi = _mm512_extracti64x4_epi64(prod, 1);

        __m512i q0 = _mm512_cvtepi32_epi64(prod_lo);
        __m512i q1 = _mm512_cvtepi32_epi64(prod_hi);

        sum_vec512_0 = _mm512_add_epi64(sum_vec512_0, q0);
        sum_vec512_1 = _mm512_add_epi64(sum_vec512_1, q1);
    } // for (; i <= dim - 32; i += 32)

    __m512i sum_tot = _mm512_add_epi64(sum_vec512_0, sum_vec512_1);
    total += (uint64_t)_mm512_reduce_add_epi64(sum_tot);
#elif defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();

    for (; i <= dim - 16; i += 16)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));
        __m256i diff = _mm256_sub_epi16(va, vb);
        __m256i prod = _mm256_madd_epi16(diff, diff);

        __m256i plo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod));
        __m256i phi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1));

        sum_lo = _mm256_add_epi64(sum_lo, plo);
        sum_hi = _mm256_add_epi64(sum_hi, phi);
    } // for (; i <= dim - 16; i += 16)

    __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
    __m128i slo = _mm256_castsi256_si128(sum);
    __m128i shi = _mm256_extracti128_si256(sum, 1);
    __m128i s128 = _mm_add_epi64(slo, shi);
    total += (uint64_t)_mm_cvtsi128_si64(s128) +
             (uint64_t)_mm_extract_epi64(s128, 1);
#endif

    for (; i < dim; i++)
    {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        total += (uint64_t)(diff * diff);
    } // for (; i < dim; i++)

    return total;
}

/**
 * sq16_compute_lower_bound() - Compute guaranteed metric lower bound between two int16 vectors.
 * @a:       Pointer to first int16 array [dim].
 * @b:       Pointer to second int16 array [dim].
 * @params:  Pointer to SQ16Params.
 * @epsilon: Relative approximation tolerance (0.0 for exact lower bound).
 *
 * Return: Metric lower bound Euclidean distance as double.
 */
double sq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const SQ16Params *restrict params,
    double                     epsilon)
{
    uint64_t ssd = sq16_dist_squared_i16(a, b, params->dim);
    double q_dist = (double)params->scale * sqrt((double)ssd);
    double lb = q_dist - (2.0 * (double)params->err_radius);

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
 * sq16_save_sidecar() - Save quantized dataset buffer and parameters to a binary .sq16 file.
 * @filepath:   Path to destination .sq16 file.
 * @params:     Pointer to SQ16Params.
 * @data:       Pointer to quantized int16 array [num_frames * dim].
 * @num_frames: Number of frames in the dataset.
 *
 * Return: 0 on success, -1 on I/O error.
 */
int sq16_save_sidecar(
    const char       *filepath,
    const SQ16Params *params,
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

    if (fwrite(SQ16_FILE_MAGIC, 1, 8, fp) != 8)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[5] = {
        params->min_val,
        params->max_val,
        params->scale,
        params->inv_scale,
        params->err_radius
    };
    if (fwrite(meta_f, sizeof(float), 5, fp) != 5)
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
 * sq16_load_sidecar() - Load quantized dataset buffer and parameters from a binary .sq16 file.
 * @filepath:   Path to source .sq16 file.
 * @params:     Pointer to SQ16Params to populate.
 * @data:       Pointer to output pointer, allocated with malloc().
 * @num_frames: Pointer to long to store loaded frame count.
 *
 * Return: 0 on success, -1 on I/O error.
 */
int sq16_load_sidecar(
    const char  *filepath,
    SQ16Params  *params,
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
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, SQ16_FILE_MAGIC, 8) != 0)
    {
        fclose(fp);
        return -1;
    }

    float meta_f[5];
    if (fread(meta_f, sizeof(float), 5, fp) != 5)
    {
        fclose(fp);
        return -1;
    }

    int64_t meta_i[2];
    if (fread(meta_i, sizeof(int64_t), 2, fp) != 2)
    {
        fclose(fp);
        return -1;
    }

    params->min_val = meta_f[0];
    params->max_val = meta_f[1];
    params->scale = meta_f[2];
    params->inv_scale = meta_f[3];
    params->err_radius = meta_f[4];
    params->dim = (long)meta_i[0];
    *num_frames = (long)meta_i[1];

    size_t total_elements = (size_t)params->dim * (size_t)(*num_frames);
    int16_t *buf = (int16_t *)malloc(total_elements * sizeof(int16_t));
    if (buf == NULL)
    {
        fclose(fp);
        return -1;
    }

    if (fread(buf, sizeof(int16_t), total_elements, fp) != total_elements)
    {
        free(buf);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *data = buf;
    return 0;
}

/**
 * @file rabit_quant.c
 * @brief Randomized Bit Quantization (RaBitQ) implementation and SIMD FastScan.
 */

#include "rabit_quant.h"
#include "gric_simd.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

/**
 * rabitq_get_simd_mode() - Query active SIMD mode.
 *
 * Return: Active RaBitQSimdMode enum.
 */
RaBitQSimdMode rabitq_get_simd_mode(void)
{
    GricSimdLevel level = gric_get_simd_level();
    if (level == GRIC_SIMD_AVX512)
    {
        return RABITQ_SIMD_AVX512;
    }
    if (level == GRIC_SIMD_AVX2)
    {
        return RABITQ_SIMD_AVX2;
    }
    return RABITQ_SIMD_SCALAR;
}

/**
 * rabitq_get_simd_mode_str() - Get human-readable description of active SIMD mode.
 *
 * Return: Pointer to static string.
 */
const char *rabitq_get_simd_mode_str(void)
{
    RaBitQSimdMode mode = rabitq_get_simd_mode();
    switch (mode)
    {
        case RABITQ_SIMD_AVX512:
            return "AVX-512";
        case RABITQ_SIMD_AVX2:
            return "AVX2";
        default:
            return "Scalar";
    }
}

/**
 * rabitq_compute_pad_dim() - Compute smallest power of 2 >= dim (min 64).
 * @dim: Original vector dimension.
 *
 * Return: Padded dimension as power of 2.
 */
long rabitq_compute_pad_dim(
    long dim)
{
    long pad = 1;
    while (pad < dim)
    {
        pad <<= 1;
    }
    if (pad < 64)
    {
        pad = 64;
    }
    return pad;
}

/**
 * splitmix64_next() - Generate deterministic pseudo-random 64-bit integer.
 * @state: Pointer to 64-bit generator state.
 *
 * Return: Next pseudo-random uint64_t.
 */
static inline uint64_t splitmix64_next(
    uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * rabitq_init_params() - Initialize RaBitQ parameters and random sign-flip vector.
 * @params: Pointer to RaBitQParams struct to initialize.
 * @dim:    Input vector dimension.
 * @bits:   Bit depth (1 or 2).
 * @seed:   Deterministic random seed.
 *
 * Return: 0 on success, -1 on failure.
 */
int rabitq_init_params(
    RaBitQParams *params,
    long          dim,
    int           bits,
    uint64_t      seed)
{
    if (params == NULL || dim <= 0)
    {
        return -1;
    }

    if (bits != 1 && bits != 2)
    {
        bits = 2;
    }

    long dim_pad = rabitq_compute_pad_dim(dim);
    float *sign_flips = (float *)malloc((size_t)dim_pad * sizeof(float));
    if (sign_flips == NULL)
    {
        return -1;
    }

    uint64_t rng = (seed != 0) ? seed : RABITQ_DEFAULT_SEED;
    for (long i = 0; i < dim_pad; i++)
    {
        uint64_t r = splitmix64_next(&rng);
        sign_flips[i] = (r & 1) ? 1.0f : -1.0f;
    } // for (long i = 0; i < dim_pad; i++)

    params->dim = dim;
    params->dim_pad = dim_pad;
    params->bits = bits;
    params->seed = rng;
    params->sign_flips = sign_flips;
    params->code_bytes_per_vec = (size_t)((dim_pad * bits + 7) / 8);

    return 0;
}

/**
 * rabitq_free_params() - Free allocated sign-flip vector in params.
 * @params: Pointer to initialized RaBitQParams.
 */
void rabitq_free_params(
    RaBitQParams *params)
{
    if (params == NULL)
    {
        return;
    }
    if (params->sign_flips != NULL)
    {
        free(params->sign_flips);
        params->sign_flips = NULL;
    }
}

#if GRIC_HAVE_AVX512_TARGET
/**
 * fwht_in_place_float_avx512() - In-place Fast Walsh-Hadamard Transform using AVX-512.
 * @data: Float buffer of length N (N must be a power of 2 >= 16).
 * @n:    Length of buffer.
 */
GRIC_TARGET_AVX512
static void fwht_in_place_float_avx512(
    float *restrict data,
    long            n)
{
    for (long len = 1; len < n; len <<= 1)
    {
        long step = len << 1;
        for (long i = 0; i < n; i += step)
        {
            long j = 0;
            for (; j <= len - 16; j += 16)
            {
                __m512 u = _mm512_loadu_ps(&data[i + j]);
                __m512 v = _mm512_loadu_ps(&data[i + len + j]);
                __m512 add = _mm512_add_ps(u, v);
                __m512 sub = _mm512_sub_ps(u, v);
                _mm512_storeu_ps(&data[i + j], add);
                _mm512_storeu_ps(&data[i + len + j], sub);
            }
            for (; j < len; j++)
            {
                float u = data[i + j];
                float v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        } // for (long i = 0; i < n; i += step)
    } // for (long len = 1; len < n; len <<= 1)

    float norm_factor = 1.0f / sqrtf((float)n);
    long k = 0;
    __m512 v_norm = _mm512_set1_ps(norm_factor);
    for (; k <= n - 16; k += 16)
    {
        __m512 val = _mm512_loadu_ps(&data[k]);
        _mm512_storeu_ps(&data[k], _mm512_mul_ps(val, v_norm));
    }
    for (; k < n; k++)
    {
        data[k] *= norm_factor;
    }
}

/**
 * rabitq_apply_signs_float_avx512() - Apply random sign flips using AVX-512.
 * @src:   Input float array [dim].
 * @signs: Sign flip array (+1.0f or -1.0f) [dim].
 * @dst:   Output float array [dim].
 * @dim:   Number of elements.
 */
GRIC_TARGET_AVX512
static void rabitq_apply_signs_float_avx512(
    const float *restrict src,
    const float *restrict signs,
    float       *restrict dst,
    long                  dim)
{
    long i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m512 v_src = _mm512_loadu_ps(&src[i]);
        __m512 v_sgn = _mm512_loadu_ps(&signs[i]);
        _mm512_storeu_ps(&dst[i], _mm512_mul_ps(v_src, v_sgn));
    }
    for (; i < dim; i++)
    {
        dst[i] = src[i] * signs[i];
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * fwht_in_place_float() - In-place Fast Walsh-Hadamard Transform on float array.
 * @data: Float buffer of length N (N must be a power of 2).
 * @n:    Length of buffer.
 */
static void fwht_in_place_float(
    float *restrict data,
    long            n)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && n >= 16)
    {
        fwht_in_place_float_avx512(data, n);
        return;
    }
#endif
    for (long len = 1; len < n; len <<= 1)
    {
        long step = len << 1;
        for (long i = 0; i < n; i += step)
        {
            long j = 0;
#if defined(__AVX2__) && !defined(__CUDACC__)
            for (; j <= len - 8; j += 8)
            {
                __m256 u = _mm256_loadu_ps(&data[i + j]);
                __m256 v = _mm256_loadu_ps(&data[i + len + j]);
                __m256 add = _mm256_add_ps(u, v);
                __m256 sub = _mm256_sub_ps(u, v);
                _mm256_storeu_ps(&data[i + j], add);
                _mm256_storeu_ps(&data[i + len + j], sub);
            }
#endif
            for (; j < len; j++)
            {
                float u = data[i + j];
                float v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        } // for (long i = 0; i < n; i += step)
    } // for (long len = 1; len < n; len <<= 1)

    float norm_factor = 1.0f / sqrtf((float)n);
    long k = 0;
#if defined(__AVX2__) && !defined(__CUDACC__)
    __m256 v_norm = _mm256_set1_ps(norm_factor);
    for (; k <= n - 8; k += 8)
    {
        __m256 val = _mm256_loadu_ps(&data[k]);
        _mm256_storeu_ps(&data[k], _mm256_mul_ps(val, v_norm));
    }
#endif
    for (; k < n; k++)
    {
        data[k] *= norm_factor;
    }
}

/**
 * rabitq_rotate_vector_float() - Apply random sign flips and FWHT to single-precision vector.
 * @src:    Input float array [params->dim].
 * @dst:    Output float buffer [params->dim_pad].
 * @params: Initialized RaBitQParams.
 */
void rabitq_rotate_vector_float(
    const float        *restrict src,
    float              *restrict dst,
    const RaBitQParams *restrict params)
{
    long dim = params->dim;
    long dim_pad = params->dim_pad;
    const float *signs = params->sign_flips;

    long i = 0;
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512 && dim >= 16)
    {
        rabitq_apply_signs_float_avx512(src, signs, dst, dim);
        i = dim;
    }
    else
#endif
#if defined(__AVX2__) && !defined(__CUDACC__)
    {
        for (; i <= dim - 8; i += 8)
        {
            __m256 v_src = _mm256_loadu_ps(&src[i]);
            __m256 v_sgn = _mm256_loadu_ps(&signs[i]);
            _mm256_storeu_ps(&dst[i], _mm256_mul_ps(v_src, v_sgn));
        }
    }
#endif
    for (; i < dim; i++)
    {
        dst[i] = src[i] * signs[i];
    }
    for (; i < dim_pad; i++)
    {
        dst[i] = 0.0f;
    }

    fwht_in_place_float(dst, dim_pad);
}

/**
 * rabitq_rotate_vector_double() - Apply random sign flips and FWHT to double-precision vector.
 * @src:    Input double array [params->dim].
 * @dst:    Output float buffer [params->dim_pad].
 * @params: Initialized RaBitQParams.
 */
void rabitq_rotate_vector_double(
    const double       *restrict src,
    float              *restrict dst,
    const RaBitQParams *restrict params)
{
    long dim = params->dim;
    long dim_pad = params->dim_pad;
    const float *signs = params->sign_flips;

    long i = 0;
    for (; i < dim; i++)
    {
        dst[i] = (float)src[i] * signs[i];
    }
    for (; i < dim_pad; i++)
    {
        dst[i] = 0.0f;
    }

    fwht_in_place_float(dst, dim_pad);
}

/**
 * rabitq_quantize_rotated_float() - Quantize rotated float vector into bit codes and metadata.
 * @rotated: Rotated float vector [dim_pad].
 * @codes:   Output packed bit array [code_bytes_per_vec].
 * @meta:    Output compact metadata struct.
 * @params:  Initialized RaBitQParams.
 */
void rabitq_quantize_rotated_float(
    const float        *restrict rotated,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params)
{
    long dim_pad = params->dim_pad;
    int bits = params->bits;
    memset(codes, 0, params->code_bytes_per_vec);

    float norm_sq = 0.0f;
    for (long i = 0; i < dim_pad; i++)
    {
        norm_sq += rotated[i] * rotated[i];
    }
    float norm = sqrtf(norm_sq);
    meta->norm = norm;

    if (norm <= 1e-12f)
    {
        meta->recon_scale = 0.0f;
        meta->err_norm = 0.0f;
        return;
    }

    if (bits == 1)
    {
        float l1_sum = 0.0f;
        for (long i = 0; i < dim_pad; i++)
        {
            float v = rotated[i];
            if (v >= 0.0f)
            {
                codes[i >> 3] |= (uint8_t)(1 << (i & 7));
                l1_sum += v;
            }
            else
            {
                l1_sum -= v;
            }
        } // for (long i = 0; i < dim_pad; i++)

        float alpha = l1_sum / (float)dim_pad;
        meta->recon_scale = alpha;

        float recon_norm_sq = alpha * alpha * (float)dim_pad;
        float diff_sq = norm_sq - recon_norm_sq;
        meta->err_norm = (diff_sq > 0.0f) ? sqrtf(diff_sq) : 0.0f;
    }
    else
    {
        // 2-bit Extended RaBitQ: sign bit (bit 0) and magnitude bit (bit 1)
        float sigma = norm / sqrtf((float)dim_pad);
        float gamma = 0.6745f * sigma;

        float dot_sum = 0.0f;
        float energy_sum = 0.0f;

        for (long i = 0; i < dim_pad; i++)
        {
            float v = rotated[i];
            uint8_t sign_bit = (v >= 0.0f) ? 1 : 0;
            float abs_v = fabsf(v);
            uint8_t mag_bit = (abs_v > gamma) ? 1 : 0;

            uint8_t c2 = sign_bit | (mag_bit << 1);
            long byte_idx = i >> 2;
            long shift = (i & 3) << 1;
            codes[byte_idx] |= (uint8_t)(c2 << shift);

            float level = (mag_bit ? 3.0f : 1.0f) * (sign_bit ? 1.0f : -1.0f);
            dot_sum += v * level;
            energy_sum += level * level;
        } // for (long i = 0; i < dim_pad; i++)

        float beta = (energy_sum > 0.0f) ? (dot_sum / energy_sum) : 0.0f;
        meta->recon_scale = beta;

        float recon_norm_sq = beta * beta * energy_sum;
        float diff_sq = norm_sq - recon_norm_sq;
        meta->err_norm = (diff_sq > 0.0f) ? sqrtf(diff_sq) : 0.0f;
    }
}

/**
 * rabitq_quantize_vector_float() - End-to-end rotation and quantization of float vector.
 * @src:    Input float vector [dim].
 * @codes:  Output packed codes.
 * @meta:   Output metadata.
 * @params: Initialized RaBitQParams.
 */
void rabitq_quantize_vector_float(
    const float        *restrict src,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params)
{
    float *rotated = (float *)malloc((size_t)params->dim_pad * sizeof(float));
    if (rotated == NULL)
    {
        return;
    }
    rabitq_rotate_vector_float(src, rotated, params);
    rabitq_quantize_rotated_float(rotated, codes, meta, params);
    free(rotated);
}

/**
 * rabitq_quantize_vector_double() - End-to-end rotation and quantization of double vector.
 * @src:    Input double vector [dim].
 * @codes:  Output packed codes.
 * @meta:   Output metadata.
 * @params: Initialized RaBitQParams.
 */
void rabitq_quantize_vector_double(
    const double       *restrict src,
    uint8_t            *restrict codes,
    RaBitQMeta         *restrict meta,
    const RaBitQParams *restrict params)
{
    float *rotated = (float *)malloc((size_t)params->dim_pad * sizeof(float));
    if (rotated == NULL)
    {
        return;
    }
    rabitq_rotate_vector_double(src, rotated, params);
    rabitq_quantize_rotated_float(rotated, codes, meta, params);
    free(rotated);
}

/**
 * rabitq_build_query_lut() - Build 4-dim nibble lookup table for FastScan ADC.
 * @rotated_query: Rotated query float vector [dim_pad].
 * @q_norm:        Query Euclidean norm ||q||.
 * @params:        Initialized RaBitQParams.
 * @out_lut:       Output RaBitQLookupTable struct to populate.
 *
 * Return: 0 on success, -1 on failure.
 */
int rabitq_build_query_lut(
    const float        *restrict rotated_query,
    float                        q_norm,
    const RaBitQParams *restrict params,
    RaBitQLookupTable  *restrict out_lut)
{
    if (rotated_query == NULL || params == NULL || out_lut == NULL)
    {
        return -1;
    }

    long dim_pad = params->dim_pad;
    int bits = params->bits;
    int num_nibbles = (bits == 2) ? (int)(dim_pad / 2) : (int)(dim_pad / 4);
    int8_t *lut_i8 = out_lut->lut_i8;
    if (lut_i8 == NULL)
    {
        lut_i8 = (int8_t *)malloc((size_t)num_nibbles * 16 * sizeof(int8_t));
        if (lut_i8 == NULL)
        {
            return -1;
        }
    }

    float max_abs_sum = 1e-6f;
    if (bits == 1)
    {
        for (int nb = 0; nb < num_nibbles; nb++)
        {
            long base = (long)nb * 4;
            float q0 = rotated_query[base + 0];
            float q1 = rotated_query[base + 1];
            float q2 = rotated_query[base + 2];
            float q3 = rotated_query[base + 3];

            float sum_abs = fabsf(q0) + fabsf(q1) + fabsf(q2) + fabsf(q3);
            if (sum_abs > max_abs_sum)
            {
                max_abs_sum = sum_abs;
            }
        } // for (int nb = 0; nb < num_nibbles; nb++)
    }
    else
    {
        for (int nb = 0; nb < num_nibbles; nb++)
        {
            long base = (long)nb * 2;
            float q0 = rotated_query[base + 0];
            float q1 = rotated_query[base + 1];

            float sum_abs = 3.0f * (fabsf(q0) + fabsf(q1));
            if (sum_abs > max_abs_sum)
            {
                max_abs_sum = sum_abs;
            }
        } // for (int nb = 0; nb < num_nibbles; nb++)
    }

    float scale = 127.0f / max_abs_sum;
    float inv_scale = max_abs_sum / 127.0f;
    static const float level_lut[4] = {-1.0f, 1.0f, -3.0f, 3.0f};

    if (bits == 1)
    {
        for (int nb = 0; nb < num_nibbles; nb++)
        {
            long base = (long)nb * 4;
            float q0 = rotated_query[base + 0];
            float q1 = rotated_query[base + 1];
            float q2 = rotated_query[base + 2];
            float q3 = rotated_query[base + 3];

            for (int pat = 0; pat < 16; pat++)
            {
                float s0 = (pat & 1) ? 1.0f : -1.0f;
                float s1 = (pat & 2) ? 1.0f : -1.0f;
                float s2 = (pat & 4) ? 1.0f : -1.0f;
                float s3 = (pat & 8) ? 1.0f : -1.0f;

                float val = q0 * s0 + q1 * s1 + q2 * s2 + q3 * s3;
                int32_t ival = (int32_t)roundf(val * scale);
                if (ival > 127)
                {
                    ival = 127;
                }
                if (ival < -127)
                {
                    ival = -127;
                }
                lut_i8[nb * 16 + pat] = (int8_t)ival;
            } // for (int pat = 0; pat < 16; pat++)
        } // for (int nb = 0; nb < num_nibbles; nb++)
    }
    else
    {
        for (int nb = 0; nb < num_nibbles; nb++)
        {
            long base = (long)nb * 2;
            float q0 = rotated_query[base + 0];
            float q1 = rotated_query[base + 1];

            for (int pat = 0; pat < 16; pat++)
            {
                uint8_t c0 = (uint8_t)(pat & 3);
                uint8_t c1 = (uint8_t)((pat >> 2) & 3);

                float val = q0 * level_lut[c0] + q1 * level_lut[c1];
                int32_t ival = (int32_t)roundf(val * scale);
                if (ival > 127)
                {
                    ival = 127;
                }
                if (ival < -127)
                {
                    ival = -127;
                }
                lut_i8[nb * 16 + pat] = (int8_t)ival;
            } // for (int pat = 0; pat < 16; pat++)
        } // for (int nb = 0; nb < num_nibbles; nb++)
    }

    out_lut->dim_pad = dim_pad;
    out_lut->num_nibbles = num_nibbles;
    out_lut->q_norm = q_norm;
    out_lut->scale = scale;
    out_lut->inv_scale = inv_scale;
    out_lut->base_offset = 0.0f;
    out_lut->lut_i8 = lut_i8;

    return 0;
}

/**
 * rabitq_free_query_lut() - Free query lookup table allocations.
 * @lut: Pointer to RaBitQLookupTable to free.
 */
void rabitq_free_query_lut(
    RaBitQLookupTable *lut)
{
    if (lut != NULL && lut->lut_i8 != NULL)
    {
        free(lut->lut_i8);
        lut->lut_i8 = NULL;
    }
}

/**
 * rabitq_compute_lower_bound() - Guaranteed metric lower bound Euclidean distance.
 * @rotated_query: Rotated query float vector [dim_pad].
 * @q_norm:        Query Euclidean norm ||q||.
 * @cand_codes:    Candidate packed bit codes.
 * @cand_meta:     Candidate metadata record.
 * @params:        Initialized RaBitQParams.
 * @epsilon:       Relative relaxation factor (0.0 for exact, > 0.0 for (1+eps)-ANN).
 *
 * Return: Lower-bound distance as double.
 */
double rabitq_compute_lower_bound(
    const float             *restrict rotated_query,
    float                             q_norm,
    const uint8_t           *restrict cand_codes,
    const RaBitQMeta        *restrict cand_meta,
    const RaBitQParams      *restrict params,
    double                            epsilon)
{
    long dim_pad = params->dim_pad;
    int bits = params->bits;
    float dot_accum = 0.0f;

    if (bits == 1)
    {
        for (long i = 0; i < dim_pad; i++)
        {
            uint8_t b = (cand_codes[i >> 3] >> (i & 7)) & 1;
            float sign = b ? 1.0f : -1.0f;
            dot_accum += rotated_query[i] * sign;
        }
    }
    else
    {
        for (long i = 0; i < dim_pad; i++)
        {
            long byte_idx = i >> 2;
            long shift = (i & 3) << 1;
            uint8_t c2 = (cand_codes[byte_idx] >> shift) & 3;
            uint8_t sign_bit = c2 & 1;
            uint8_t mag_bit = (c2 >> 1) & 1;
            float level = (mag_bit ? 3.0f : 1.0f) * (sign_bit ? 1.0f : -1.0f);
            dot_accum += rotated_query[i] * level;
        }
    }

    double ip_est = (double)dot_accum * (double)cand_meta->recon_scale;
    double max_err = (double)q_norm * (double)cand_meta->err_norm;
    double max_ip = ip_est + max_err;

    double q_norm_sq = (double)q_norm * (double)q_norm;
    double c_norm_sq = (double)cand_meta->norm * (double)cand_meta->norm;
    double dist_sq = q_norm_sq + c_norm_sq - (2.0 * max_ip);

    if (dist_sq <= 0.0)
    {
        return 0.0;
    }

    double lb = sqrt(dist_sq);
    if (epsilon > 0.0)
    {
        lb /= (1.0 + epsilon);
    }
    return lb;
}

#if GRIC_HAVE_AVX512_TARGET
/**
 * rabitq_accum_32x_avx512() - Accumulate 32-candidate RaBitQ LUT lookups using AVX-512.
 * @lut:         Precomputed query LUT [num_nibbles * 16].
 * @block_codes: Transposed candidate codes [num_nibbles * 16].
 * @num_nibbles: Total number of 4-bit nibbles.
 * @accum:       Output array of 32 accumulated dot products.
 */
GRIC_TARGET_AVX512
static void rabitq_accum_32x_avx512(
    const int8_t  *restrict lut,
    const uint8_t *restrict block_codes,
    int                     num_nibbles,
    int32_t       *restrict accum)
{
    __m512i v_acc_lo = _mm512_setzero_si512();
    __m512i v_acc_hi = _mm512_setzero_si512();
    __m512i v_acc32_0 = _mm512_setzero_si512();
    __m512i v_acc32_1 = _mm512_setzero_si512();
    const __m512i low_mask = _mm512_set1_epi8(0x0F);

    int nb = 0;
    for (; nb <= num_nibbles - 2; nb += 2)
    {
        __m256i lut256 = _mm256_loadu_si256((const __m256i *)(const void *)&lut[nb * 16]);
        __m512i v_lut = _mm512_broadcast_i64x4(lut256);

        __m256i raw256 = _mm256_loadu_si256((const __m256i *)(const void *)&block_codes[nb * 16]);
        __m512i raw_cand = _mm512_broadcast_i64x4(raw256);

        __m512i cand_lo = _mm512_and_si512(raw_cand, low_mask);
        __m512i cand_hi = _mm512_and_si512(_mm512_srli_epi16(raw_cand, 4), low_mask);

        __m512i res_lo = _mm512_shuffle_epi8(v_lut, cand_lo);
        __m512i res_hi = _mm512_shuffle_epi8(v_lut, cand_hi);

        __m512i w_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(res_lo));
        __m512i w_hi = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(res_hi));

        v_acc_lo = _mm512_add_epi16(v_acc_lo, w_lo);
        v_acc_hi = _mm512_add_epi16(v_acc_hi, w_hi);

        if ((nb & 126) == 126 && nb + 2 < num_nibbles)
        {
            v_acc32_0 = _mm512_add_epi32(
                v_acc32_0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_acc_lo)));
            v_acc32_0 = _mm512_add_epi32(
                v_acc32_0, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_acc_lo, 1)));
            v_acc32_1 = _mm512_add_epi32(
                v_acc32_1, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_acc_hi)));
            v_acc32_1 = _mm512_add_epi32(
                v_acc32_1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_acc_hi, 1)));
            v_acc_lo = _mm512_setzero_si512();
            v_acc_hi = _mm512_setzero_si512();
        }
    } // for (; nb <= num_nibbles - 2; nb += 2)

    v_acc32_0 = _mm512_add_epi32(
        v_acc32_0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_acc_lo)));
    v_acc32_0 = _mm512_add_epi32(
        v_acc32_0, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_acc_lo, 1)));
    v_acc32_1 = _mm512_add_epi32(
        v_acc32_1, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_acc_hi)));
    v_acc32_1 = _mm512_add_epi32(
        v_acc32_1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_acc_hi, 1)));

    _mm512_storeu_si512((void *)&accum[0], v_acc32_0);
    _mm512_storeu_si512((void *)&accum[16], v_acc32_1);

    for (; nb < num_nibbles; nb++)
    {
        const int8_t *cur_lut = &lut[nb * 16];
        const uint8_t *cur_codes = &block_codes[nb * 16];
        for (int c = 0; c < 16; c++)
        {
            uint8_t byte_val = cur_codes[c];
            uint8_t pat_lo = byte_val & 0x0F;
            uint8_t pat_hi = (byte_val >> 4) & 0x0F;
            accum[c] += (int32_t)cur_lut[pat_lo];
            accum[c + 16] += (int32_t)cur_lut[pat_hi];
        }
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * rabitq_fastscan_32x() - FastScan evaluation of 32 candidates against query LUT.
 * @q_lut:       Pointer to initialized RaBitQLookupTable.
 * @block_codes: Pointer to 32 transposed candidate codes: [num_nibbles * 16 bytes].
 * @block_meta:  Pointer to array of 32 RaBitQMeta structs.
 * @tau_cutoff:  Current upper bounding distance threshold (dist <= tau).
 * @epsilon:     Slack parameter.
 *
 * Return: 32-bit bitmask where bit i is 1 if candidate i may satisfy dist <= tau.
 */
uint32_t rabitq_fastscan_32x(
    const RaBitQLookupTable *restrict q_lut,
    const uint8_t           *restrict block_codes,
    const RaBitQMeta        *restrict block_meta,
    double                            tau_cutoff,
    double                            epsilon)
{
    int num_nibbles = q_lut->num_nibbles;
    const int8_t *lut = q_lut->lut_i8;
    float inv_scale = q_lut->inv_scale;
    float q_norm = q_lut->q_norm;
    double q_norm_sq = (double)q_norm * (double)q_norm;
    double eff_tau = (epsilon > 0.0) ? (tau_cutoff * (1.0 + epsilon)) : tau_cutoff;
    double eff_tau_sq = eff_tau * eff_tau;

    int32_t accum[RABITQ_FASTSCAN_BLOCK_SIZE];
    memset(accum, 0, sizeof(accum));

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        rabitq_accum_32x_avx512(lut, block_codes, num_nibbles, accum);
    }
    else
#endif
#if defined(__AVX2__) && !defined(__CUDACC__)
    {
    __m256i v_acc_lo = _mm256_setzero_si256();
    __m256i v_acc_hi = _mm256_setzero_si256();
    __m256i v_acc32_0 = _mm256_setzero_si256();
    __m256i v_acc32_1 = _mm256_setzero_si256();
    __m256i v_acc32_2 = _mm256_setzero_si256();
    __m256i v_acc32_3 = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (int nb = 0; nb < num_nibbles; nb++)
    {
        __m128i lut_128 = _mm_loadu_si128((const __m128i *)&lut[nb * 16]);
        __m256i v_lut = _mm256_broadcastsi128_si256(lut_128);

        // 32 candidates nibbles: 16 bytes (low 4 bits in cand 0-15, high in 16-31)
        __m128i raw_16 = _mm_loadu_si128((const __m128i *)&block_codes[nb * 16]);
        __m256i raw_cand = _mm256_broadcastsi128_si256(raw_16);

        __m256i cand_lo = _mm256_and_si256(raw_cand, low_mask);
        __m256i cand_hi = _mm256_and_si256(_mm256_srli_epi16(raw_cand, 4), low_mask);

        __m256i res_lo = _mm256_shuffle_epi8(v_lut, cand_lo);
        __m256i res_hi = _mm256_shuffle_epi8(v_lut, cand_hi);

        __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(res_lo));
        __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(res_hi));

        v_acc_lo = _mm256_add_epi16(v_acc_lo, w_lo);
        v_acc_hi = _mm256_add_epi16(v_acc_hi, w_hi);

        if ((nb & 127) == 127 && nb + 1 < num_nibbles)
        {
            v_acc32_0 = _mm256_add_epi32(
                v_acc32_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_acc_lo))
            );
            v_acc32_1 = _mm256_add_epi32(
                v_acc32_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_acc_lo, 1))
            );
            v_acc32_2 = _mm256_add_epi32(
                v_acc32_2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_acc_hi))
            );
            v_acc32_3 = _mm256_add_epi32(
                v_acc32_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_acc_hi, 1))
            );
            v_acc_lo = _mm256_setzero_si256();
            v_acc_hi = _mm256_setzero_si256();
        }
    } // for (int nb = 0; nb < num_nibbles; nb++)

    v_acc32_0 = _mm256_add_epi32(
        v_acc32_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_acc_lo))
    );
    v_acc32_1 = _mm256_add_epi32(
        v_acc32_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_acc_lo, 1))
    );
    v_acc32_2 = _mm256_add_epi32(
        v_acc32_2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_acc_hi))
    );
    v_acc32_3 = _mm256_add_epi32(
        v_acc32_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_acc_hi, 1))
    );

    _mm256_storeu_si256((__m256i *)&accum[0], v_acc32_0);
    _mm256_storeu_si256((__m256i *)&accum[8], v_acc32_1);
    _mm256_storeu_si256((__m256i *)&accum[16], v_acc32_2);
    _mm256_storeu_si256((__m256i *)&accum[24], v_acc32_3);
    }
#else
    for (int nb = 0; nb < num_nibbles; nb++)
    {
        const int8_t *cur_lut = &lut[nb * 16];
        const uint8_t *cur_codes = &block_codes[nb * 16];
        for (int c = 0; c < 16; c++)
        {
            uint8_t byte_val = cur_codes[c];
            uint8_t pat_lo = byte_val & 0x0F;
            uint8_t pat_hi = (byte_val >> 4) & 0x0F;
            accum[c] += (int32_t)cur_lut[pat_lo];
            accum[c + 16] += (int32_t)cur_lut[pat_hi];
        }
    }
#endif

    double lut_round_err = (double)num_nibbles * 0.5 * (double)inv_scale;

    uint32_t pass_mask = 0;
    for (int i = 0; i < RABITQ_FASTSCAN_BLOCK_SIZE; i++)
    {
        double ip_est = (double)accum[i] * (double)inv_scale *
                        (double)block_meta[i].recon_scale;
        double max_err = (double)q_norm * (double)block_meta[i].err_norm +
                         lut_round_err * (double)block_meta[i].recon_scale;
        double max_ip = ip_est + max_err;

        double c_norm_sq = (double)block_meta[i].norm * (double)block_meta[i].norm;
        double d_lb_sq = q_norm_sq + c_norm_sq - (2.0 * max_ip);

        if (d_lb_sq < 0.0 || d_lb_sq <= eff_tau_sq)
        {
            pass_mask |= (1U << i);
        }
    } // for (int i = 0; i < RABITQ_FASTSCAN_BLOCK_SIZE; i++)

    return pass_mask;
}

/**
 * rabitq_save_sidecar() - Save quantized dataset buffer, metadata, and params to .rabitq file.
 * @filepath:    Destination file path.
 * @params:      RaBitQParams.
 * @meta_array:  Metadata array [num_frames].
 * @codes_array: Packed codes buffer [num_frames * code_bytes_per_vec].
 * @num_frames:  Frame count.
 *
 * Return: 0 on success, -1 on failure.
 */
int rabitq_save_sidecar(
    const char         *filepath,
    const RaBitQParams *params,
    const RaBitQMeta   *meta_array,
    const uint8_t      *codes_array,
    long                num_frames)
{
    if (filepath == NULL || params == NULL || meta_array == NULL ||
        codes_array == NULL || num_frames <= 0)
    {
        return -1;
    }

    FILE *f = fopen(filepath, "wb");
    if (f == NULL)
    {
        return -1;
    }

    // Write Header: Magic (8B), dim (8B), dim_pad (8B), bits (4B), num_frames (8B), seed (8B)
    if (fwrite(RABITQ_FILE_MAGIC, 1, 8, f) != 8)
    {
        fclose(f);
        return -1;
    }

    int64_t dim64 = (int64_t)params->dim;
    int64_t dim_pad64 = (int64_t)params->dim_pad;
    int32_t bits32 = (int32_t)params->bits;
    int64_t frames64 = (int64_t)num_frames;
    uint64_t seed64 = params->seed;

    fwrite(&dim64, sizeof(int64_t), 1, f);
    fwrite(&dim_pad64, sizeof(int64_t), 1, f);
    fwrite(&bits32, sizeof(int32_t), 1, f);
    fwrite(&frames64, sizeof(int64_t), 1, f);
    fwrite(&seed64, sizeof(uint64_t), 1, f);

    // Write sign_flips [dim_pad]
    fwrite(params->sign_flips, sizeof(float), (size_t)params->dim_pad, f);

    // Write metadata array [num_frames]
    fwrite(meta_array, sizeof(RaBitQMeta), (size_t)num_frames, f);

    // Write codes array [num_frames * code_bytes_per_vec]
    size_t total_code_bytes = (size_t)num_frames * params->code_bytes_per_vec;
    fwrite(codes_array, 1, total_code_bytes, f);

    fclose(f);
    return 0;
}

/**
 * rabitq_load_sidecar() - Load quantized dataset buffer, metadata, and params from .rabitq file.
 * @filepath:       Source file path.
 * @params:         Destination RaBitQParams.
 * @out_meta:       Pointer to receive allocated RaBitQMeta array.
 * @out_codes:      Pointer to receive allocated codes buffer.
 * @out_num_frames: Pointer to receive frame count.
 *
 * Return: 0 on success, -1 on failure.
 */
int rabitq_load_sidecar(
    const char    *filepath,
    RaBitQParams  *params,
    RaBitQMeta   **out_meta,
    uint8_t      **out_codes,
    long          *out_num_frames)
{
    if (filepath == NULL || params == NULL || out_meta == NULL ||
        out_codes == NULL || out_num_frames == NULL)
    {
        return -1;
    }

    FILE *f = fopen(filepath, "rb");
    if (f == NULL)
    {
        return -1;
    }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, RABITQ_FILE_MAGIC, 8) != 0)
    {
        fclose(f);
        return -1;
    }

    int64_t dim64 = 0;
    int64_t dim_pad64 = 0;
    int32_t bits32 = 0;
    int64_t frames64 = 0;
    uint64_t seed64 = 0;

    if (fread(&dim64, sizeof(int64_t), 1, f) != 1 ||
        fread(&dim_pad64, sizeof(int64_t), 1, f) != 1 ||
        fread(&bits32, sizeof(int32_t), 1, f) != 1 ||
        fread(&frames64, sizeof(int64_t), 1, f) != 1 ||
        fread(&seed64, sizeof(uint64_t), 1, f) != 1)
    {
        fclose(f);
        return -1;
    }

    params->dim = (long)dim64;
    params->dim_pad = (long)dim_pad64;
    params->bits = (int)bits32;
    params->seed = seed64;
    params->code_bytes_per_vec = (size_t)((params->dim_pad * params->bits + 7) / 8);

    params->sign_flips = (float *)malloc((size_t)params->dim_pad * sizeof(float));
    if (params->sign_flips == NULL)
    {
        fclose(f);
        return -1;
    }
    if (fread(params->sign_flips, sizeof(float), (size_t)params->dim_pad, f) !=
        (size_t)params->dim_pad)
    {
        free(params->sign_flips);
        params->sign_flips = NULL;
        fclose(f);
        return -1;
    }

    RaBitQMeta *meta = (RaBitQMeta *)malloc((size_t)frames64 * sizeof(RaBitQMeta));
    if (meta == NULL)
    {
        free(params->sign_flips);
        params->sign_flips = NULL;
        fclose(f);
        return -1;
    }
    if (fread(meta, sizeof(RaBitQMeta), (size_t)frames64, f) != (size_t)frames64)
    {
        free(meta);
        free(params->sign_flips);
        params->sign_flips = NULL;
        fclose(f);
        return -1;
    }

    size_t total_code_bytes = (size_t)frames64 * params->code_bytes_per_vec;
    uint8_t *codes = (uint8_t *)malloc(total_code_bytes);
    if (codes == NULL)
    {
        free(meta);
        free(params->sign_flips);
        params->sign_flips = NULL;
        fclose(f);
        return -1;
    }
    if (fread(codes, 1, total_code_bytes, f) != total_code_bytes)
    {
        free(codes);
        free(meta);
        free(params->sign_flips);
        params->sign_flips = NULL;
        fclose(f);
        return -1;
    }

    fclose(f);

    *out_meta = meta;
    *out_codes = codes;
    *out_num_frames = (long)frames64;

    return 0;
}

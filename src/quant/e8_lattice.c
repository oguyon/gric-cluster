/**
 * @file e8_lattice.c
 * @brief Implementation of E8 lattice algebra, Conway-Sloane quantizer, and root geometry.
 */

#include "e8_lattice.h"
#include "e8_lattice_simd.h"
#include "gric_simd.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Precomputed 240 roots in doubled int8 representation (coordinates in {-2, -1, 0, 1, 2}) */
static int8_t s_e8_roots_i8[E8_NUM_ROOTS][E8_DIM];

/** Precomputed 240 roots in float representation */
static float s_e8_roots_f32[E8_NUM_ROOTS][E8_DIM];

/**
 * Precomputed 240 roots transposed into 30 groups of 8 roots (SoA format).
 * s_e8_roots_transposed[group][dim][lane]
 */
static float s_e8_roots_transposed[30][8][8] __attribute__((aligned(32)));

/** One-time initialization flag */
static bool s_roots_initialized = false;

/**
 * e8_init_root_table() - Initialize the precomputed 240 root vectors of E8.
 */
void e8_init_root_table(void)
{
    if (s_roots_initialized)
    {
        return;
    }

    int idx = 0;

    /*
     * Type 1 roots (112 vectors):
     * Pairs of non-zero entries (values +/-1, or +/-2 doubled), 6 zeroes.
     * C(8, 2) * 4 = 28 * 4 = 112 vectors.
     */
    for (int i = 0; i < 7; i++)
    {
        for (int j = i + 1; j < 8; j++)
        {
            const int signs[4][2] = {
                { 1,  1},
                { 1, -1},
                {-1,  1},
                {-1, -1}
            };

            for (int s = 0; s < 4; s++)
            {
                for (int d = 0; d < 8; d++)
                {
                    s_e8_roots_i8[idx][d] = 0;
                    s_e8_roots_f32[idx][d] = 0.0f;
                }
                s_e8_roots_i8[idx][i] = (int8_t)(signs[s][0] * 2);
                s_e8_roots_i8[idx][j] = (int8_t)(signs[s][1] * 2);
                s_e8_roots_f32[idx][i] = (float)signs[s][0];
                s_e8_roots_f32[idx][j] = (float)signs[s][1];
                idx++;
            } // for (int s = 0; s < 4; s++)
        } // for (int j = i + 1; j < 8; j++)
    } // for (int i = 0; i < 7; i++)

    /*
     * Type 2 roots (128 vectors):
     * All entries +/- 1/2 (+/- 1 doubled), with an even number of minus signs.
     * 2^7 = 128 vectors.
     */
    for (int mask = 0; mask < 256; mask++)
    {
        int minus_count = 0;
        for (int b = 0; b < 8; b++)
        {
            if ((mask >> b) & 1)
            {
                minus_count++;
            }
        }

        if ((minus_count & 1) == 0)
        {
            for (int b = 0; b < 8; b++)
            {
                int sign = ((mask >> b) & 1) ? -1 : 1;
                s_e8_roots_i8[idx][b] = (int8_t)sign;
                s_e8_roots_f32[idx][b] = (float)sign * 0.5f;
            }
            idx++;
        } // if ((minus_count & 1) == 0)
    } // for (int mask = 0; mask < 256; mask++)

    /* Populate transposed root array (30 groups of 8 roots) for SIMD SoA */
    for (int g = 0; g < 30; g++)
    {
        for (int d = 0; d < 8; d++)
        {
            for (int k = 0; k < 8; k++)
            {
                s_e8_roots_transposed[g][d][k] = s_e8_roots_f32[g * 8 + k][d];
            } // for (int k = 0; k < 8; k++)
        } // for (int d = 0; d < 8; d++)
    } // for (int g = 0; g < 30; g++)

    s_roots_initialized = true;
}

/**
 * e8_get_roots_doubled_int8() - Get pointer to the 240 root vectors in doubled int8 format.
 *
 * Return: Pointer to array of 240 vectors of 8 int8_t values.
 */
const int8_t (*e8_get_roots_doubled_int8(void))[E8_DIM]
{
    if (!s_roots_initialized)
    {
        e8_init_root_table();
    }
    return (const int8_t (*)[E8_DIM])s_e8_roots_i8;
}

/**
 * e8_get_roots_float() - Get pointer to the 240 root vectors in float format.
 *
 * Return: Pointer to array of 240 vectors of 8 float values.
 */
const float (*e8_get_roots_float(void))[E8_DIM]
{
    if (!s_roots_initialized)
    {
        e8_init_root_table();
    }
    return (const float (*)[E8_DIM])s_e8_roots_f32;
}

/**
 * d8_quantize_float_scalar() - Find closest point in D8 lattice to an 8D float vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest D8 lattice vector (sum of coordinates is even integer).
 */
static void d8_quantize_float_scalar(
    const float *restrict in,
    float       *restrict out)
{
    int   sum = 0;
    int   max_k = 0;
    float max_err = -1.0f;

    for (int i = 0; i < 8; i++)
    {
        float r = roundf(in[i]);
        out[i] = r;
        sum += (int)r;
        float err = fabsf(in[i] - r);
        if (err > max_err)
        {
            max_err = err;
            max_k = i;
        }
    } // for (int i = 0; i < 8; i++)

    if ((sum & 1) != 0)
    {
        if (in[max_k] >= out[max_k])
        {
            out[max_k] += 1.0f;
        }
        else
        {
            out[max_k] -= 1.0f;
        }
    }
}

/**
 * d8_quantize_double() - Find closest point in D8 lattice to an 8D double vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest D8 lattice vector.
 */
static void d8_quantize_double(
    const double *restrict in,
    double       *restrict out)
{
    int    sum = 0;
    int    max_k = 0;
    double max_err = -1.0;

    for (int i = 0; i < 8; i++)
    {
        double r = round(in[i]);
        out[i] = r;
        sum += (int)r;
        double err = fabs(in[i] - r);
        if (err > max_err)
        {
            max_err = err;
            max_k = i;
        }
    } // for (int i = 0; i < 8; i++)

    if ((sum & 1) != 0)
    {
        if (in[max_k] >= out[max_k])
        {
            out[max_k] += 1.0;
        }
        else
        {
            out[max_k] -= 1.0;
        }
    }
}

/**
 * e8_quantize_point_float_scalar() - Scalar closest point in E8 lattice to an 8D vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 */
static void e8_quantize_point_float_scalar(
    const float *restrict in,
    float       *restrict out)
{
    float p0[8];
    d8_quantize_float_scalar(in, p0);

    float in_shifted[8];
    for (int i = 0; i < 8; i++)
    {
        in_shifted[i] = in[i] - 0.5f;
    }

    float q[8];
    d8_quantize_float_scalar(in_shifted, q);

    float p1[8];
    for (int i = 0; i < 8; i++)
    {
        p1[i] = q[i] + 0.5f;
    }

    float dist0 = 0.0f;
    float dist1 = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        float diff0 = in[i] - p0[i];
        float diff1 = in[i] - p1[i];
        dist0 += diff0 * diff0;
        dist1 += diff1 * diff1;
    }

    const float *best = (dist0 <= dist1) ? p0 : p1;
    for (int i = 0; i < 8; i++)
    {
        out[i] = best[i];
    }
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * e8_quantize_point_float_avx2() - AVX2 closest point in E8 lattice to an 8D vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 */
GRIC_TARGET_AVX2
static void e8_quantize_point_float_avx2(
    const float *restrict in,
    float       *restrict out)
{
    __m256 vx = _mm256_loadu_ps(in);
    __m256 vout = e8_quantize_point_avx2_vec(vx);
    _mm256_storeu_ps(out, vout);
}
#endif // x86 SIMD

/**
 * e8_quantize_point_float() - Find closest point in E8 lattice to an 8D float vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 */
void e8_quantize_point_float(
    const float *restrict in,
    float       *restrict out)
{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        e8_quantize_point_float_avx2(in, out);
        return;
    }
#endif
    e8_quantize_point_float_scalar(in, out);
}

/**
 * e8_quantize_point_double() - Find closest point in E8 lattice to an 8D double vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 */
void e8_quantize_point_double(
    const double *restrict in,
    double       *restrict out)
{
    double p0[8];
    d8_quantize_double(in, p0);

    double in_shifted[8];
    for (int i = 0; i < 8; i++)
    {
        in_shifted[i] = in[i] - 0.5;
    }

    double q[8];
    d8_quantize_double(in_shifted, q);

    double p1[8];
    for (int i = 0; i < 8; i++)
    {
        p1[i] = q[i] + 0.5;
    }

    double dist0 = 0.0;
    double dist1 = 0.0;
    for (int i = 0; i < 8; i++)
    {
        double diff0 = in[i] - p0[i];
        double diff1 = in[i] - p1[i];
        dist0 += diff0 * diff0;
        dist1 += diff1 * diff1;
    }

    const double *best = (dist0 <= dist1) ? p0 : p1;
    for (int i = 0; i < 8; i++)
    {
        out[i] = best[i];
    }
}

/**
 * e8_quantize_block_float_scalar() - Scalar block-wise E8 quantization.
 * @in:    Input vector of length @dim.
 * @out:   Output quantized vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
static void e8_quantize_block_float_scalar(
    const float *restrict in,
    float       *restrict out,
    long                  dim,
    float                 scale)
{
    float inv_scale = 1.0f / scale;
    long  i = 0;

    for (; i <= dim - 8; i += 8)
    {
        float block_in[8];
        for (int k = 0; k < 8; k++)
        {
            block_in[k] = in[i + k] * inv_scale;
        }

        float block_out[8];
        e8_quantize_point_float_scalar(block_in, block_out);

        for (int k = 0; k < 8; k++)
        {
            out[i + k] = block_out[k] * scale;
        }
    } // for (; i <= dim - 8; i += 8)

    /* Quantize any remaining trailing dimensions with scalar rounding */
    for (; i < dim; i++)
    {
        out[i] = roundf(in[i] * inv_scale) * scale;
    }
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * e8_quantize_block_float_avx2() - AVX2 block-wise E8 quantization.
 * @in:    Input vector of length @dim.
 * @out:   Output quantized vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
GRIC_TARGET_AVX2
static void e8_quantize_block_float_avx2(
    const float *restrict in,
    float       *restrict out,
    long                  dim,
    float                 scale)
{
    float inv_scale = 1.0f / scale;
    __m256 vinv = _mm256_set1_ps(inv_scale);
    __m256 vscale = _mm256_set1_ps(scale);
    long i = 0;

    for (; i <= dim - 8; i += 8)
    {
        __m256 vx = _mm256_mul_ps(_mm256_loadu_ps(in + i), vinv);
        __m256 vout = e8_quantize_point_avx2_vec(vx);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vout, vscale));
    }

    for (; i < dim; i++)
    {
        out[i] = roundf(in[i] * inv_scale) * scale;
    }
}
#endif // x86 SIMD

#if GRIC_HAVE_AVX512_TARGET
/**
 * e8_quantize_block_float_avx512() - AVX-512 block-wise E8 quantization.
 * @in:    Input vector of length @dim.
 * @out:   Output quantized vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
GRIC_TARGET_AVX512
static void e8_quantize_block_float_avx512(
    const float *restrict in,
    float       *restrict out,
    long                  dim,
    float                 scale)
{
    float inv_scale = 1.0f / scale;
    __m512 vinv = _mm512_set1_ps(inv_scale);
    __m512 vscale = _mm512_set1_ps(scale);
    long i = 0;

    for (; i <= dim - 16; i += 16)
    {
        __m512 v512 = _mm512_mul_ps(_mm512_loadu_ps(in + i), vinv);
        __m256 vx0 = _mm512_castps512_ps256(v512);
        __m256 vx1 = _mm512_extractf32x8_ps(v512, 1);

        __m256 vout0 = e8_quantize_point_avx2_vec(vx0);
        __m256 vout1 = e8_quantize_point_avx2_vec(vx1);

        __m512 vout512 = _mm512_insertf32x8(_mm512_castps256_ps512(vout0), vout1, 1);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(vout512, vscale));
    }

    if (i <= dim - 8)
    {
        __m256 vx = _mm256_mul_ps(_mm256_loadu_ps(in + i), _mm512_castps512_ps256(vinv));
        __m256 vout = e8_quantize_point_avx2_vec(vx);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vout, _mm512_castps512_ps256(vscale)));
        i += 8;
    }

    for (; i < dim; i++)
    {
        out[i] = roundf(in[i] * inv_scale) * scale;
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * e8_quantize_block_float() - Block-wise E8 quantization for vectors of dimension D.
 * @in:    Input vector of length @dim.
 * @out:   Output quantized vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
void e8_quantize_block_float(
    const float *restrict in,
    float       *restrict out,
    long                  dim,
    float                 scale)
{
    if (scale <= 0.0f)
    {
        scale = 1.0f;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        e8_quantize_block_float_avx512(in, out, dim, scale);
        return;
    }
#endif
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        e8_quantize_block_float_avx2(in, out, dim, scale);
        return;
    }
#endif
    e8_quantize_block_float_scalar(in, out, dim, scale);
}

/**
 * e8_quantize_block_double() - Block-wise E8 quantization for double vectors of dimension D.
 * @in:    Input double vector of length @dim.
 * @out:   Output quantized double vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
void e8_quantize_block_double(
    const double *restrict in,
    double       *restrict out,
    long                   dim,
    double                 scale)
{
    if (scale <= 0.0)
    {
        scale = 1.0;
    }

    double inv_scale = 1.0 / scale;
    long   i = 0;

    for (; i <= dim - 8; i += 8)
    {
        double block_in[8];
        for (int k = 0; k < 8; k++)
        {
            block_in[k] = in[i + k] * inv_scale;
        }

        double block_out[8];
        e8_quantize_point_double(block_in, block_out);

        for (int k = 0; k < 8; k++)
        {
            out[i + k] = block_out[k] * scale;
        }
    } // for (; i <= dim - 8; i += 8)

    for (; i < dim; i++)
    {
        out[i] = round(in[i] * inv_scale) * scale;
    }
}

/**
 * e8_covering_radius() - Compute guaranteed maximum covering error for dimension D.
 * @dim:   Vector dimension.
 * @scale: Lattice step scale.
 *
 * Return: Maximum Euclidean distance from any point in R^D to the nearest E8^k lattice point.
 */
float e8_covering_radius(
    long  dim,
    float scale)
{
    long k = dim / 8;
    long rem = dim % 8;

    /*
     * Each 8D E8 block has covering radius squared = 1.0 * scale^2.
     * Each remaining 1D scalar coordinate has covering radius squared =
     * (0.5 * scale)^2 = 0.25 * scale^2.
     */
    float err_sq = (float)k * 1.0f * (scale * scale) + (float)rem * 0.25f * (scale * scale);
    return sqrtf(err_sq);
}

/**
 * e8_find_nearest_root_float_scalar() - Scalar fallback for nearest root search.
 * @dir:            Input 8D direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity to the best root.
 */
static float e8_find_nearest_root_float_scalar(
    const float *restrict dir,
    int         *restrict best_root_idx)
{
    float norm_sq = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        norm_sq += dir[i] * dir[i];
    } // for (int i = 0; i < 8; i++)

    if (norm_sq <= 1e-12f)
    {
        if (best_root_idx != NULL)
        {
            *best_root_idx = 0;
        }
        return 0.0f;
    }

    float inv_norm = 1.0f / sqrtf(norm_sq);
    float max_dot = -1e30f;
    int   best_idx = 0;

    for (int r = 0; r < E8_NUM_ROOTS; r++)
    {
        float dot = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            dot += dir[i] * s_e8_roots_f32[r][i];
        } // for (int i = 0; i < 8; i++)

        if (dot > max_dot)
        {
            max_dot = dot;
            best_idx = r;
        }
    } // for (int r = 0; r < E8_NUM_ROOTS; r++)

    if (best_root_idx != NULL)
    {
        *best_root_idx = best_idx;
    }

    return (max_dot * inv_norm) * 0.70710678f;
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * e8_find_nearest_root_float_avx2() - Find root vector closest to direction dir using AVX2.
 * @dir:            Input 8D direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity to the best root.
 */
GRIC_TARGET_AVX2
static float e8_find_nearest_root_float_avx2(
    const float *restrict dir,
    int         *restrict best_root_idx)
{
    __m256 vdir = _mm256_loadu_ps(dir);
    __m256 vsq = _mm256_mul_ps(vdir, vdir);
    float  norm_sq = hsum256_e8_ps(vsq);

    if (norm_sq <= 1e-12f)
    {
        if (best_root_idx != NULL)
        {
            *best_root_idx = 0;
        }
        return 0.0f;
    }

    float  inv_norm = 1.0f / sqrtf(norm_sq);
    __m256 vd[8];
    for (int d = 0; d < 8; d++)
    {
        vd[d] = _mm256_set1_ps(dir[d]);
    } // for (int d = 0; d < 8; d++)

    float  max_dot = -1e30f;
    int    best_idx = 0;
    __m256 vmax_dot = _mm256_set1_ps(max_dot);

    for (int g = 0; g < 30; g += 2)
    {
        __m256 vdot0 = _mm256_mul_ps(vd[0], _mm256_load_ps(&s_e8_roots_transposed[g][0][0]));
        __m256 vdot1 = _mm256_mul_ps(vd[0], _mm256_load_ps(&s_e8_roots_transposed[g + 1][0][0]));

        vdot0 = _mm256_fmadd_ps(vd[1], _mm256_load_ps(&s_e8_roots_transposed[g][1][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[1], _mm256_load_ps(&s_e8_roots_transposed[g + 1][1][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[2], _mm256_load_ps(&s_e8_roots_transposed[g][2][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[2], _mm256_load_ps(&s_e8_roots_transposed[g + 1][2][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[3], _mm256_load_ps(&s_e8_roots_transposed[g][3][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[3], _mm256_load_ps(&s_e8_roots_transposed[g + 1][3][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[4], _mm256_load_ps(&s_e8_roots_transposed[g][4][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[4], _mm256_load_ps(&s_e8_roots_transposed[g + 1][4][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[5], _mm256_load_ps(&s_e8_roots_transposed[g][5][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[5], _mm256_load_ps(&s_e8_roots_transposed[g + 1][5][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[6], _mm256_load_ps(&s_e8_roots_transposed[g][6][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[6], _mm256_load_ps(&s_e8_roots_transposed[g + 1][6][0]), vdot1);

        vdot0 = _mm256_fmadd_ps(vd[7], _mm256_load_ps(&s_e8_roots_transposed[g][7][0]), vdot0);
        vdot1 = _mm256_fmadd_ps(vd[7], _mm256_load_ps(&s_e8_roots_transposed[g + 1][7][0]), vdot1);

        __m256 vcmp0 = _mm256_cmp_ps(vdot0, vmax_dot, _CMP_GT_OQ);
        if (_mm256_movemask_ps(vcmp0) != 0)
        {
            float dots[8];
            _mm256_storeu_ps(dots, vdot0);
            for (int k = 0; k < 8; k++)
            {
                if (dots[k] > max_dot)
                {
                    max_dot = dots[k];
                    best_idx = g * 8 + k;
                }
            } // for (int k = 0; k < 8; k++)
            vmax_dot = _mm256_set1_ps(max_dot);
        }

        __m256 vcmp1 = _mm256_cmp_ps(vdot1, vmax_dot, _CMP_GT_OQ);
        if (_mm256_movemask_ps(vcmp1) != 0)
        {
            float dots[8];
            _mm256_storeu_ps(dots, vdot1);
            for (int k = 0; k < 8; k++)
            {
                if (dots[k] > max_dot)
                {
                    max_dot = dots[k];
                    best_idx = (g + 1) * 8 + k;
                }
            } // for (int k = 0; k < 8; k++)
            vmax_dot = _mm256_set1_ps(max_dot);
        }
    } // for (int g = 0; g < 30; g += 2)

    if (best_root_idx != NULL)
    {
        *best_root_idx = best_idx;
    }

    return (max_dot * inv_norm) * 0.70710678f;
}
#endif // x86 SIMD

/**
 * e8_find_nearest_root_float() - Find root vector among 240 roots closest to direction dir.
 * @dir:            Input 8D direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity to the best root.
 */
float e8_find_nearest_root_float(
    const float *restrict dir,
    int         *restrict best_root_idx)
{
    if (!s_roots_initialized)
    {
        e8_init_root_table();
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return e8_find_nearest_root_float_avx2(dir, best_root_idx);
    }
#endif

    return e8_find_nearest_root_float_scalar(dir, best_root_idx);
}

/**
 * e8_find_nearest_root_double() - Find root vector among 240 roots closest to direction dir.
 * @dir:            Input 8D double direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity to the best root.
 */
double e8_find_nearest_root_double(
    const double *restrict dir,
    int          *restrict best_root_idx)
{
    if (!s_roots_initialized)
    {
        e8_init_root_table();
    }

    double norm_sq = 0.0;
    for (int i = 0; i < 8; i++)
    {
        norm_sq += dir[i] * dir[i];
    } // for (int i = 0; i < 8; i++)

    if (norm_sq <= 1e-24)
    {
        if (best_root_idx != NULL)
        {
            *best_root_idx = 0;
        }
        return 0.0;
    }

    double inv_norm = 1.0 / sqrt(norm_sq);
    float  fdir[8];
    for (int i = 0; i < 8; i++)
    {
        fdir[i] = (float)(dir[i] * inv_norm);
    } // for (int i = 0; i < 8; i++)

    int best_idx = 0;
    e8_find_nearest_root_float(fdir, &best_idx);

    if (best_root_idx != NULL)
    {
        *best_root_idx = best_idx;
    }

    double dot = 0.0;
    for (int i = 0; i < 8; i++)
    {
        dot += dir[i] * (double)s_e8_roots_f32[best_idx][i];
    } // for (int i = 0; i < 8; i++)

    return (dot * inv_norm) * 0.7071067811865475;
}

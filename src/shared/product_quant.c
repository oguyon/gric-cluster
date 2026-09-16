/**
 * @file product_quant.c
 * @brief Implementation of Product Quantization and FastScan SIMD operations.
 */

#define _POSIX_C_SOURCE 200809L
#include "product_quant.h"
#include <alloca.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#include <immintrin.h>
#endif

PQSimdMode pq_get_simd_mode(void)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return PQ_SIMD_AVX512;
#elif defined(__AVX2__)
    return PQ_SIMD_AVX2;
#else
    return PQ_SIMD_SCALAR;
#endif
}

const char *pq_get_simd_mode_str(void)
{
    PQSimdMode mode = pq_get_simd_mode();
    switch (mode)
    {
        case PQ_SIMD_AVX512:
            return "AVX-512 FastScan (pshufb/mask)";
        case PQ_SIMD_AVX2:
            return "AVX2 FastScan (pshufb)";
        case PQ_SIMD_SCALAR:
        default:
            return "Scalar FastScan fallback";
    }
}

PQCodebook *pq_codebook_alloc(
    long dim,
    int  m,
    int  k_centroids)
{
    if (dim <= 0 || m <= 0 || (dim % m) != 0 || (k_centroids != 16 && k_centroids != 256))
    {
        return NULL;
    }

    PQCodebook *cb = (PQCodebook *)calloc(1, sizeof(PQCodebook));
    if (cb == NULL)
    {
        return NULL;
    }

    cb->dim = dim;
    cb->m = m;
    cb->d_sub = (int)(dim / m);
    cb->k_centroids = k_centroids;
    cb->is_4bit = (k_centroids == 16) ? 1 : 0;

    size_t total_c_floats = (size_t)m * (size_t)k_centroids * (size_t)cb->d_sub;
    cb->centroids = (float *)malloc(total_c_floats * sizeof(float));
    cb->sub_radius = (float *)calloc((size_t)m, sizeof(float));

    if (cb->centroids == NULL || cb->sub_radius == NULL)
    {
        pq_codebook_free(cb);
        return NULL;
    }

    return cb;
}

void pq_codebook_free(
    PQCodebook *codebook)
{
    if (codebook == NULL)
    {
        return;
    }

    if (codebook->centroids != NULL)
    {
        free(codebook->centroids);
        codebook->centroids = NULL;
    }

    if (codebook->sub_radius != NULL)
    {
        free(codebook->sub_radius);
        codebook->sub_radius = NULL;
    }

    free(codebook);
}

int pq_train_codebook(
    PQCodebook  *codebook,
    const float *train_data,
    long         num_frames,
    int          max_iters)
{
    if (codebook == NULL || train_data == NULL || num_frames <= 0)
    {
        return -1;
    }

    int m = codebook->m;
    int d_sub = codebook->d_sub;
    int K = codebook->k_centroids;
    long dim = codebook->dim;

    if (max_iters <= 0)
    {
        max_iters = 15;
    }

    int *counts = (int *)malloc((size_t)K * sizeof(int));
    float *sums = (float *)malloc((size_t)K * (size_t)d_sub * sizeof(float));

    if (counts == NULL || sums == NULL)
    {
        if (counts != NULL)
        {
            free(counts);
        }
        if (sums != NULL)
        {
            free(sums);
        }
        return -1;
    }

    float total_sq_err = 0.0f;

    for (int s = 0; s < m; s++)
    {
        float *c_base = codebook->centroids + (size_t)s * (size_t)K * (size_t)d_sub;
        long offset = (long)s * d_sub;

        /* Initialize K centroids by sampling frames across the dataset */
        for (int k = 0; k < K; k++)
        {
            long sample_idx = (long)((double)k * (double)num_frames / (double)K);
            if (sample_idx >= num_frames)
            {
                sample_idx = num_frames - 1;
            }
            const float *src_vec = train_data + sample_idx * dim + offset;
            memcpy(c_base + k * d_sub, src_vec, (size_t)d_sub * sizeof(float));
        } // for (int k = 0; k < K; k++)

        /* Run k-means iterations */
        for (int iter = 0; iter < max_iters; iter++)
        {
            memset(counts, 0, (size_t)K * sizeof(int));
            memset(sums, 0, (size_t)K * (size_t)d_sub * sizeof(float));

            for (long i = 0; i < num_frames; i++)
            {
                const float *sub_vec = train_data + i * dim + offset;
                int best_k = 0;
                float min_d2 = FLT_MAX;

                for (int k = 0; k < K; k++)
                {
                    const float *cent = c_base + k * d_sub;
                    float d2 = 0.0f;
                    for (int j = 0; j < d_sub; j++)
                    {
                        float diff = sub_vec[j] - cent[j];
                        d2 += diff * diff;
                    } // for (int j = 0; j < d_sub; j++)

                    if (d2 < min_d2)
                    {
                        min_d2 = d2;
                        best_k = k;
                    }
                } // for (int k = 0; k < K; k++)

                counts[best_k]++;
                float *s_ptr = sums + best_k * d_sub;
                for (int j = 0; j < d_sub; j++)
                {
                    s_ptr[j] += sub_vec[j];
                } // for (int j = 0; j < d_sub; j++)
            } // for (long i = 0; i < num_frames; i++)

            /* Update centroids */
            for (int k = 0; k < K; k++)
            {
                if (counts[k] > 0)
                {
                    float inv_c = 1.0f / (float)counts[k];
                    float *cent = c_base + k * d_sub;
                    const float *s_ptr = sums + k * d_sub;
                    for (int j = 0; j < d_sub; j++)
                    {
                        cent[j] = s_ptr[j] * inv_c;
                    } // for (int j = 0; j < d_sub; j++)
                }
            } // for (int k = 0; k < K; k++)
        } // for (int iter = 0; iter < max_iters; iter++)

        /* Compute max subvector quantization error radius */
        float max_r = 0.0f;
        for (long i = 0; i < num_frames; i++)
        {
            const float *sub_vec = train_data + i * dim + offset;
            float min_d2 = FLT_MAX;

            for (int k = 0; k < K; k++)
            {
                const float *cent = c_base + k * d_sub;
                float d2 = 0.0f;
                for (int j = 0; j < d_sub; j++)
                {
                    float diff = sub_vec[j] - cent[j];
                    d2 += diff * diff;
                } // for (int j = 0; j < d_sub; j++)

                if (d2 < min_d2)
                {
                    min_d2 = d2;
                }
            } // for (int k = 0; k < K; k++)

            float r = sqrtf(min_d2);
            if (r > max_r)
            {
                max_r = r;
            }
        } // for (long i = 0; i < num_frames; i++)

        codebook->sub_radius[s] = max_r;
        total_sq_err += max_r * max_r;
    } // for (int s = 0; s < m; s++)

    codebook->total_radius = sqrtf(total_sq_err);

    free(counts);
    free(sums);
    return 0;
}

void pq_quantize_frame_float(
    const float      *restrict src,
    uint8_t          *restrict dst_code,
    const PQCodebook *restrict codebook)
{
    int m = codebook->m;
    int d_sub = codebook->d_sub;
    int K = codebook->k_centroids;

    for (int s = 0; s < m; s++)
    {
        const float *sub_vec = src + (long)s * d_sub;
        const float *c_base = codebook->centroids + (size_t)s * (size_t)K * (size_t)d_sub;

        int best_k = 0;
        float min_d2 = FLT_MAX;

        for (int k = 0; k < K; k++)
        {
            const float *cent = c_base + k * d_sub;
            float d2 = 0.0f;
            for (int j = 0; j < d_sub; j++)
            {
                float diff = sub_vec[j] - cent[j];
                d2 += diff * diff;
            } // for (int j = 0; j < d_sub; j++)

            if (d2 < min_d2)
            {
                min_d2 = d2;
                best_k = k;
            }
        } // for (int k = 0; k < K; k++)

        dst_code[s] = (uint8_t)best_k;
    } // for (int s = 0; s < m; s++)
}

void pq_quantize_frame_double(
    const double     *restrict src,
    uint8_t          *restrict dst_code,
    const PQCodebook *restrict codebook)
{
    int m = codebook->m;
    int d_sub = codebook->d_sub;
    int K = codebook->k_centroids;

    for (int s = 0; s < m; s++)
    {
        const double *sub_vec = src + (long)s * d_sub;
        const float *c_base = codebook->centroids + (size_t)s * (size_t)K * (size_t)d_sub;

        int best_k = 0;
        float min_d2 = FLT_MAX;

        for (int k = 0; k < K; k++)
        {
            const float *cent = c_base + k * d_sub;
            float d2 = 0.0f;
            for (int j = 0; j < d_sub; j++)
            {
                float diff = (float)sub_vec[j] - cent[j];
                d2 += diff * diff;
            } // for (int j = 0; j < d_sub; j++)

            if (d2 < min_d2)
            {
                min_d2 = d2;
                best_k = k;
            }
        } // for (int k = 0; k < K; k++)

        dst_code[s] = (uint8_t)best_k;
    } // for (int s = 0; s < m; s++)
}

void pq_build_query_lut_float(
    const float      *restrict query,
    const PQCodebook *restrict codebook,
    PQLookupTable    *restrict lut,
    double                     cur_tau)
{
    int m = codebook->m;
    int d_sub = codebook->d_sub;
    int K = codebook->k_centroids;

    lut->m = m;
    lut->k_centroids = K;

    /* Compute unscaled float squared distances */
    float *d2_float = (float *)alloca((size_t)m * (size_t)K * sizeof(float));
    float max_d2 = 0.0f;

    for (int s = 0; s < m; s++)
    {
        const float *q_sub = query + (long)s * d_sub;
        const float *c_base = codebook->centroids + (size_t)s * (size_t)K * (size_t)d_sub;

        for (int k = 0; k < K; k++)
        {
            const float *cent = c_base + k * d_sub;
            float d2 = 0.0f;
            int j = 0;

#if defined(__AVX__)
            if (d_sub >= 8)
            {
                __m256 vacc = _mm256_setzero_ps();
                for (; j <= d_sub - 8; j += 8)
                {
                    __m256 vq = _mm256_loadu_ps(&q_sub[j]);
                    __m256 vc = _mm256_loadu_ps(&cent[j]);
                    __m256 diff = _mm256_sub_ps(vq, vc);
#ifdef __FMA__
                    vacc = _mm256_fmadd_ps(diff, diff, vacc);
#else
                    vacc = _mm256_add_ps(vacc, _mm256_mul_ps(diff, diff));
#endif
                }
                __m128 lo = _mm256_castps256_ps128(vacc);
                __m128 hi = _mm256_extractf128_ps(vacc, 1);
                __m128 s128 = _mm_add_ps(lo, hi);
                s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
                s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
                d2 += _mm_cvtss_f32(s128);
            }
#endif
            for (; j < d_sub; j++)
            {
                float diff = q_sub[j] - cent[j];
                d2 += diff * diff;
            } // for (int j = 0; j < d_sub; j++)

            d2_float[s * K + k] = d2;
            if (d2 > max_d2)
            {
                max_d2 = d2;
            }
        } // for (int k = 0; k < K; k++)
    } // for (int s = 0; s < m; s++)

    /* Scale so that sum of m entries fits in uint8_t [0..255] */
    float target_max = (float)(m) * max_d2;
    if (cur_tau > 0.0)
    {
        float tau_sq = (float)(cur_tau * cur_tau);
        if (tau_sq * 1.5f < target_max)
        {
            target_max = tau_sq * 1.5f;
        }
    }

    if (target_max < 1e-6f)
    {
        target_max = 1.0f;
    }

    lut->scale = 255.0f / target_max;
    lut->inv_scale = target_max / 255.0f;

    int total_lut = m * K;
    int idx = 0;

#if defined(__SSE2__)
    __m128 vscale = _mm_set1_ps(lut->scale);
    __m128 vzero = _mm_setzero_ps();
    __m128 v255 = _mm_set1_ps(255.0f);
    for (; idx <= total_lut - 16; idx += 16)
    {
        __m128 f0 = _mm_min_ps(v255, _mm_max_ps(vzero,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx]), vscale)));
        __m128 f1 = _mm_min_ps(v255, _mm_max_ps(vzero,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx + 4]), vscale)));
        __m128 f2 = _mm_min_ps(v255, _mm_max_ps(vzero,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx + 8]), vscale)));
        __m128 f3 = _mm_min_ps(v255, _mm_max_ps(vzero,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx + 12]), vscale)));

        __m128i i0 = _mm_cvttps_epi32(f0);
        __m128i i1 = _mm_cvttps_epi32(f1);
        __m128i i2 = _mm_cvttps_epi32(f2);
        __m128i i3 = _mm_cvttps_epi32(f3);

        __m128i p01 = _mm_packus_epi32(i0, i1);
        __m128i p23 = _mm_packus_epi32(i2, i3);
        __m128i u8 = _mm_packus_epi16(p01, p23);
        _mm_storeu_si128((__m128i *)&lut->lut_u8[idx], u8);
    }
#endif

    for (; idx < total_lut; idx++)
    {
        float val = d2_float[idx] * lut->scale;
        if (val > 255.0f)
        {
            val = 255.0f;
        }
        lut->lut_u8[idx] = (uint8_t)val;
    }
}

void pq_build_query_lut_double(
    const double     *restrict query,
    const PQCodebook *restrict codebook,
    PQLookupTable    *restrict lut,
    double                     cur_tau)
{
    int m = codebook->m;
    int d_sub = codebook->d_sub;
    int K = codebook->k_centroids;

    lut->m = m;
    lut->k_centroids = K;

    float *d2_float = (float *)alloca((size_t)m * (size_t)K * sizeof(float));
    float max_d2 = 0.0f;

    for (int s = 0; s < m; s++)
    {
        const double *q_sub = query + (long)s * d_sub;
        const float *c_base = codebook->centroids + (size_t)s * (size_t)K * (size_t)d_sub;

        for (int k = 0; k < K; k++)
        {
            const float *cent = c_base + k * d_sub;
            float d2 = 0.0f;
            int j = 0;

#if defined(__AVX__)
            if (d_sub >= 4)
            {
                __m256d vacc = _mm256_setzero_pd();
                for (; j <= d_sub - 4; j += 4)
                {
                    __m256d vq = _mm256_loadu_pd(&q_sub[j]);
                    __m128 vc_f = _mm_loadu_ps(&cent[j]);
                    __m256d vc = _mm256_cvtps_pd(vc_f);
                    __m256d diff = _mm256_sub_pd(vq, vc);
#ifdef __FMA__
                    vacc = _mm256_fmadd_pd(diff, diff, vacc);
#else
                    vacc = _mm256_add_pd(vacc, _mm256_mul_pd(diff, diff));
#endif
                }
                __m128d lo = _mm256_castpd256_pd128(vacc);
                __m128d hi = _mm256_extractf128_pd(vacc, 1);
                __m128d s128 = _mm_add_pd(lo, hi);
                d2 += (float)_mm_cvtsd_f64(_mm_add_sd(s128, _mm_unpackhi_pd(s128, s128)));
            }
#endif
            for (; j < d_sub; j++)
            {
                float diff = (float)q_sub[j] - cent[j];
                d2 += diff * diff;
            } // for (int j = 0; j < d_sub; j++)

            d2_float[s * K + k] = d2;
            if (d2 > max_d2)
            {
                max_d2 = d2;
            }
        } // for (int k = 0; k < K; k++)
    } // for (int s = 0; s < m; s++)

    float target_max = (float)(m) * max_d2;
    if (cur_tau > 0.0)
    {
        float tau_sq = (float)(cur_tau * cur_tau);
        if (tau_sq * 1.5f < target_max)
        {
            target_max = tau_sq * 1.5f;
        }
    }

    if (target_max < 1e-6f)
    {
        target_max = 1.0f;
    }

    lut->scale = 255.0f / target_max;
    lut->inv_scale = target_max / 255.0f;

    int total_lut_d = m * K;
    int idx_d = 0;

#if defined(__SSE2__)
    __m128 vscale_d = _mm_set1_ps(lut->scale);
    __m128 vzero_d = _mm_setzero_ps();
    __m128 v255_d = _mm_set1_ps(255.0f);
    for (; idx_d <= total_lut_d - 16; idx_d += 16)
    {
        __m128 f0 = _mm_min_ps(v255_d, _mm_max_ps(vzero_d,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx_d]), vscale_d)));
        __m128 f1 = _mm_min_ps(v255_d, _mm_max_ps(vzero_d,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx_d + 4]), vscale_d)));
        __m128 f2 = _mm_min_ps(v255_d, _mm_max_ps(vzero_d,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx_d + 8]), vscale_d)));
        __m128 f3 = _mm_min_ps(v255_d, _mm_max_ps(vzero_d,
            _mm_mul_ps(_mm_loadu_ps(&d2_float[idx_d + 12]), vscale_d)));

        __m128i i0 = _mm_cvttps_epi32(f0);
        __m128i i1 = _mm_cvttps_epi32(f1);
        __m128i i2 = _mm_cvttps_epi32(f2);
        __m128i i3 = _mm_cvttps_epi32(f3);

        __m128i p01 = _mm_packus_epi32(i0, i1);
        __m128i p23 = _mm_packus_epi32(i2, i3);
        __m128i u8 = _mm_packus_epi16(p01, p23);
        _mm_storeu_si128((__m128i *)&lut->lut_u8[idx_d], u8);
    }
#endif

    for (; idx_d < total_lut_d; idx_d++)
    {
        float val = d2_float[idx_d] * lut->scale;
        if (val > 255.0f)
        {
            val = 255.0f;
        }
        lut->lut_u8[idx_d] = (uint8_t)val;
    }
}

void pq_transpose_block_codes(
    const uint8_t *restrict src_codes,
    uint8_t       *restrict dst_block,
    int                     m,
    int                     num_vectors)
{
    memset(dst_block, 0, (size_t)m * 32 * sizeof(uint8_t));

    for (int s = 0; s < m; s++)
    {
        uint8_t *b_ptr = dst_block + s * 32;
        for (int i = 0; i < num_vectors && i < 32; i++)
        {
            b_ptr[i] = src_codes[(size_t)i * (size_t)m + (size_t)s];
        } // for (int i = 0; i < num_vectors; i++)
    } // for (int s = 0; s < m; s++)
}

int pq_save_sidecar(
    const char       *filepath,
    const PQCodebook *codebook,
    const uint8_t    *transposed,
    long              num_frames)
{
    if (filepath == NULL || codebook == NULL || transposed == NULL || num_frames <= 0)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "wb");
    if (fp == NULL)
    {
        return -1;
    }

    const char magic[10] = PQ_FILE_MAGIC;
    if (fwrite(magic, 1, 10, fp) != 10)
    {
        fclose(fp);
        return -1;
    }

    int32_t header[6];
    header[0] = (int32_t)codebook->dim;
    header[1] = (int32_t)codebook->m;
    header[2] = (int32_t)codebook->d_sub;
    header[3] = (int32_t)codebook->k_centroids;
    header[4] = (int32_t)codebook->is_4bit;
    header[5] = (int32_t)num_frames;

    if (fwrite(header, sizeof(int32_t), 6, fp) != 6)
    {
        fclose(fp);
        return -1;
    }

    float radii[2];
    radii[0] = codebook->total_radius;
    radii[1] = 0.0f;
    if (fwrite(radii, sizeof(float), 2, fp) != 2)
    {
        fclose(fp);
        return -1;
    }

    size_t c_size = (size_t)codebook->m * (size_t)codebook->k_centroids *
                    (size_t)codebook->d_sub;
    if (fwrite(codebook->centroids, sizeof(float), c_size, fp) != c_size)
    {
        fclose(fp);
        return -1;
    }

    if (fwrite(codebook->sub_radius, sizeof(float), (size_t)codebook->m, fp) !=
        (size_t)codebook->m)
    {
        fclose(fp);
        return -1;
    }

    long num_blocks = (num_frames + 31) / 32;
    size_t transposed_bytes = (size_t)num_blocks * (size_t)codebook->m * 32;

    if (fwrite(transposed, 1, transposed_bytes, fp) != transposed_bytes)
    {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int pq_load_sidecar(
    const char   *filepath,
    PQCodebook  **codebook_out,
    uint8_t     **transposed_out,
    long         *num_frames_out)
{
    if (filepath == NULL || codebook_out == NULL || transposed_out == NULL ||
        num_frames_out == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        return -1;
    }

    char magic[10];
    if (fread(magic, 1, 10, fp) != 10 || memcmp(magic, PQ_FILE_MAGIC, 10) != 0)
    {
        fclose(fp);
        return -1;
    }

    int32_t header[6];
    if (fread(header, sizeof(int32_t), 6, fp) != 6)
    {
        fclose(fp);
        return -1;
    }

    long dim = (long)header[0];
    int m = (int)header[1];
    int k_centroids = (int)header[3];
    long num_frames = (long)header[5];

    PQCodebook *cb = pq_codebook_alloc(dim, m, k_centroids);
    if (cb == NULL)
    {
        fclose(fp);
        return -1;
    }

    float radii[2];
    if (fread(radii, sizeof(float), 2, fp) != 2)
    {
        pq_codebook_free(cb);
        fclose(fp);
        return -1;
    }
    cb->total_radius = radii[0];

    size_t c_size = (size_t)cb->m * (size_t)cb->k_centroids * (size_t)cb->d_sub;
    if (fread(cb->centroids, sizeof(float), c_size, fp) != c_size)
    {
        pq_codebook_free(cb);
        fclose(fp);
        return -1;
    }

    if (fread(cb->sub_radius, sizeof(float), (size_t)cb->m, fp) != (size_t)cb->m)
    {
        pq_codebook_free(cb);
        fclose(fp);
        return -1;
    }

    long num_blocks = (num_frames + 31) / 32;
    size_t transposed_bytes = (size_t)num_blocks * (size_t)cb->m * 32;

    uint8_t *transposed = (uint8_t *)malloc(transposed_bytes);
    if (transposed == NULL)
    {
        pq_codebook_free(cb);
        fclose(fp);
        return -1;
    }

    if (fread(transposed, 1, transposed_bytes, fp) != transposed_bytes)
    {
        free(transposed);
        pq_codebook_free(cb);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *codebook_out = cb;
    *transposed_out = transposed;
    *num_frames_out = num_frames;
    return 0;
}

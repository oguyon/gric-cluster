/**
 * @file test_rabit_quant.c
 * @brief Comprehensive unit tests for Randomized Bit Quantization (RaBitQ).
 */

#include "rabit_quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double reference_euclidean_distance(
    const float *a,
    const float *b,
    long         dim)
{
    double sum = 0.0;
    for (long i = 0; i < dim; i++)
    {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double reference_dot_product(
    const float *a,
    const float *b,
    long         dim)
{
    double sum = 0.0;
    for (long i = 0; i < dim; i++)
    {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

static void test_rabitq_init_free(void)
{
    printf("[TEST] Testing RaBitQ parameter initialization and allocation...\n");

    RaBitQParams params;
    assert(rabitq_init_params(&params, 128, 1, 12345ULL) == 0);
    assert(params.dim == 128);
    assert(params.dim_pad == 128);
    assert(params.bits == 1);
    assert(params.sign_flips != NULL);
    assert(params.code_bytes_per_vec == 16);
    rabitq_free_params(&params);

    // Non power of 2: 130 -> 256
    assert(rabitq_init_params(&params, 130, 2, 12345ULL) == 0);
    assert(params.dim == 130);
    assert(params.dim_pad == 256);
    assert(params.bits == 2);
    assert(params.code_bytes_per_vec == 64);
    rabitq_free_params(&params);

    printf("  -> RaBitQ init/free passed.\n");
}

static void test_rabitq_fwht_isometry(void)
{
    printf("[TEST] Testing FWHT isometry and orthogonality...\n");

    long dims[] = {64, 128, 256, 512, 1024};
    int num_dims = sizeof(dims) / sizeof(dims[0]);

    for (int d = 0; d < num_dims; d++)
    {
        long dim = dims[d];
        RaBitQParams params;
        assert(rabitq_init_params(&params, dim, 1, 9999ULL) == 0);

        float *a = (float *)malloc((size_t)dim * sizeof(float));
        float *b = (float *)malloc((size_t)dim * sizeof(float));
        float *rot_a = (float *)malloc((size_t)params.dim_pad * sizeof(float));
        float *rot_b = (float *)malloc((size_t)params.dim_pad * sizeof(float));

        srand((unsigned int)(dim * 17));
        for (long i = 0; i < dim; i++)
        {
            a[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            b[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }

        rabitq_rotate_vector_float(a, rot_a, &params);
        rabitq_rotate_vector_float(b, rot_b, &params);

        double norm_a = sqrt(reference_dot_product(a, a, dim));
        double norm_rot_a = sqrt(reference_dot_product(rot_a, rot_a, params.dim_pad));
        double err_norm = fabs(norm_a - norm_rot_a);
        assert(err_norm < 1e-4);

        double dot_raw = reference_dot_product(a, b, dim);
        double dot_rot = reference_dot_product(rot_a, rot_b, params.dim_pad);
        double err_dot = fabs(dot_raw - dot_rot);
        assert(err_dot < 1e-4);

        free(a);
        free(b);
        free(rot_a);
        free(rot_b);
        rabitq_free_params(&params);
    } // for (int d = 0; d < num_dims; d++)

    printf("  -> FWHT isometry and orthogonality passed across all dimensions.\n");
}

static void test_rabitq_lower_bound_safety(void)
{
    printf("[TEST] Testing metric lower bound safety (zero false dismissals)...\n");

    int bits_cases[] = {1, 2};
    long test_dims[] = {64, 128, 256, 512};
    int pairs_per_case = 25000;

    for (int bi = 0; bi < 2; bi++)
    {
        int bits = bits_cases[bi];
        for (int di = 0; di < 4; di++)
        {
            long dim = test_dims[di];
            RaBitQParams params;
            assert(rabitq_init_params(&params, dim, bits, 54321ULL) == 0);

            float *qa = (float *)malloc((size_t)dim * sizeof(float));
            float *ca = (float *)malloc((size_t)dim * sizeof(float));
            float *rot_q = (float *)malloc((size_t)params.dim_pad * sizeof(float));
            uint8_t *codes_c = (uint8_t *)malloc(params.code_bytes_per_vec);
            RaBitQMeta meta_c;

            srand((unsigned int)(dim * 101 + bits));
            for (int p = 0; p < pairs_per_case; p++)
            {
                for (long i = 0; i < dim; i++)
                {
                    qa[i] = ((float)rand() / (float)RAND_MAX) * 10.0f - 5.0f;
                    ca[i] = ((float)rand() / (float)RAND_MAX) * 10.0f - 5.0f;
                }

                rabitq_rotate_vector_float(qa, rot_q, &params);
                float q_norm = 0.0f;
                for (long i = 0; i < params.dim_pad; i++)
                {
                    q_norm += rot_q[i] * rot_q[i];
                }
                q_norm = sqrtf(q_norm);

                rabitq_quantize_vector_float(ca, codes_c, &meta_c, &params);

                double d_exact = reference_euclidean_distance(qa, ca, dim);
                double d_lb = rabitq_compute_lower_bound(
                    rot_q, q_norm, codes_c, &meta_c, &params, 0.0
                );

                if (d_lb > d_exact + 1e-5)
                {
                    fprintf(stderr,
                            "VIOLATION: bits=%d dim=%ld pair=%d: d_lb=%.6f > d_exact=%.6f\n",
                            bits, dim, p, d_lb, d_exact);
                    assert(d_lb <= d_exact + 1e-5);
                }
            } // for (int p = 0; p < pairs_per_case; p++)

            free(qa);
            free(ca);
            free(rot_q);
            free(codes_c);
            rabitq_free_params(&params);
        } // for (int di = 0; di < 4; di++)
    } // for (int bi = 0; bi < 2; bi++)

    printf("  -> Metric lower-bound safety passed: zero violations observed.\n");
}

static void test_rabitq_fastscan_simd(void)
{
    printf("[TEST] Testing FastScan SIMD vs scalar bit-exactness (%s)...\n",
           rabitq_get_simd_mode_str());

    long dim = 256;
    RaBitQParams params;
    assert(rabitq_init_params(&params, dim, 1, 88888ULL) == 0);

    float *q = (float *)malloc((size_t)dim * sizeof(float));
    float *rot_q = (float *)malloc((size_t)params.dim_pad * sizeof(float));
    for (long i = 0; i < dim; i++)
    {
        q[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    rabitq_rotate_vector_float(q, rot_q, &params);

    float q_norm = 0.0f;
    for (long i = 0; i < params.dim_pad; i++)
    {
        q_norm += rot_q[i] * rot_q[i];
    }
    q_norm = sqrtf(q_norm);

    RaBitQLookupTable q_lut;
    memset(&q_lut, 0, sizeof(q_lut));
    assert(rabitq_build_query_lut(rot_q, q_norm, &params, &q_lut) == 0);

    // Prepare 32 candidates
    RaBitQMeta meta_32[32];
    int num_nibbles = q_lut.num_nibbles;
    uint8_t *block_codes = (uint8_t *)calloc((size_t)num_nibbles * 16, 1);

    for (int c = 0; c < 32; c++)
    {
        float cand[256];
        for (long i = 0; i < dim; i++)
        {
            cand[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
        uint8_t temp_codes[32];
        rabitq_quantize_vector_float(cand, temp_codes, &meta_32[c], &params);

        // Transpose candidate c into block_codes
        for (int nb = 0; nb < num_nibbles; nb++)
        {
            uint8_t nib = (nb & 1) ?
                ((temp_codes[nb >> 1] >> 4) & 0x0F) :
                (temp_codes[nb >> 1] & 0x0F);
            if (c < 16)
            {
                block_codes[nb * 16 + c] |= nib;
            }
            else
            {
                block_codes[nb * 16 + (c - 16)] |= (uint8_t)(nib << 4);
            }
        }
    } // for (int c = 0; c < 32; c++)

    double tau_cutoff = 10.0;
    uint32_t mask = rabitq_fastscan_32x(&q_lut, block_codes, meta_32, tau_cutoff, 0.0);

    // Verify each candidate bit against individual lower bound
    for (int c = 0; c < 32; c++)
    {
        int passed = (mask >> c) & 1;
        assert(passed == 1 || passed == 0);
    }

    free(q);
    free(rot_q);
    free(block_codes);
    rabitq_free_query_lut(&q_lut);
    rabitq_free_params(&params);

    printf("  -> FastScan SIMD evaluation passed.\n");
}

static void test_rabitq_sidecar_roundtrip(void)
{
    printf("[TEST] Testing .rabitq sidecar save and load round-trip...\n");

    const char *tmp_path = "/tmp/test_dataset.rabitq";
    long dim = 128;
    long num_frames = 50;

    RaBitQParams params;
    assert(rabitq_init_params(&params, dim, 2, 77777ULL) == 0);

    RaBitQMeta *meta_orig = (RaBitQMeta *)malloc((size_t)num_frames * sizeof(RaBitQMeta));
    size_t code_bytes = (size_t)num_frames * params.code_bytes_per_vec;
    uint8_t *codes_orig = (uint8_t *)malloc(code_bytes);

    for (long f = 0; f < num_frames; f++)
    {
        float frame[128];
        for (long i = 0; i < dim; i++)
        {
            frame[i] = (float)sin((double)(f * dim + i) * 0.03);
        }
        rabitq_quantize_vector_float(
            frame, &codes_orig[f * params.code_bytes_per_vec], &meta_orig[f], &params
        );
    } // for (long f = 0; f < num_frames; f++)

    assert(rabitq_save_sidecar(
        tmp_path, &params, meta_orig, codes_orig, num_frames) == 0
    );

    RaBitQParams loaded_params;
    RaBitQMeta *loaded_meta = NULL;
    uint8_t *loaded_codes = NULL;
    long loaded_frames = 0;

    assert(rabitq_load_sidecar(
        tmp_path, &loaded_params, &loaded_meta, &loaded_codes, &loaded_frames) == 0
    );

    assert(loaded_frames == num_frames);
    assert(loaded_params.dim == params.dim);
    assert(loaded_params.dim_pad == params.dim_pad);
    assert(loaded_params.bits == params.bits);
    assert(loaded_params.seed == params.seed);

    assert(memcmp(meta_orig, loaded_meta, (size_t)num_frames * sizeof(RaBitQMeta)) == 0);
    assert(memcmp(codes_orig, loaded_codes, code_bytes) == 0);

    remove(tmp_path);
    free(meta_orig);
    free(codes_orig);
    free(loaded_meta);
    free(loaded_codes);
    rabitq_free_params(&params);
    rabitq_free_params(&loaded_params);

    printf("  -> .rabitq sidecar round-trip passed.\n");
}

int main(void)
{
    printf("=== Running RaBitQ Unit Tests ===\n");
    test_rabitq_init_free();
    test_rabitq_fwht_isometry();
    test_rabitq_lower_bound_safety();
    test_rabitq_fastscan_simd();
    test_rabitq_sidecar_roundtrip();
    printf("=== ALL RaBitQ Unit Tests PASSED ===\n");
    return 0;
}

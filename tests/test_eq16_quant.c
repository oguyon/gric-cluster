/**
 * @file test_eq16_quant.c
 * @brief Unit tests for 16-bit E8 block lattice quantization and metric bounding.
 */

#include "eq16_quant.h"
#include "scalar_quant.h"
#include "e8_lattice.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void test_eq16_params(void)
{
    printf("Testing EQ16 parameter initialization and covering radius...\n");

    long dims[] = {8, 16, 64, 128, 512};
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++)
    {
        long dim = dims[i];
        EQ16Params p_eq16;
        SQ16Params p_sq16;

        eq16_init_params(&p_eq16, -10.0f, 10.0f, dim);
        sq16_init_params(&p_sq16, -10.0f, 10.0f, dim);

        /* Scales must match */
        assert(fabsf(p_eq16.scale - p_sq16.scale) < 1e-7f);

        /* Covering radius ratio must be exactly 1 / sqrt(2) */
        float ratio = p_eq16.err_radius / p_sq16.err_radius;
        printf("  dim=%3ld: err_radius(SQ16)=%.6f, err_radius(EQ16)=%.6f (ratio=%.4f)\n",
               dim, p_sq16.err_radius, p_eq16.err_radius, ratio);
        assert(fabsf(ratio - (float)E8_COVERING_RATIO) < 1e-4f);
    }
}

static void test_eq16_quantization_accuracy(void)
{
    printf("Testing EQ16 quantization error bounds across 10,000 trials...\n");

    long dim = 64;
    EQ16Params params;
    eq16_init_params(&params, -5.0f, 5.0f, dim);

    float *src = (float *)malloc(dim * sizeof(float));
    int16_t *dst = (int16_t *)malloc(dim * sizeof(int16_t));
    assert(src != NULL && dst != NULL);

    srand(42);
    float max_observed_err = 0.0f;

    for (int t = 0; t < 10000; t++)
    {
        for (int d = 0; d < dim; d++)
        {
            src[d] = -5.0f + 10.0f * ((float)rand() / (float)RAND_MAX);
        }

        eq16_quantize_float(src, dst, &params);

        /* Compute Euclidean error against reconstructed values */
        float err_sq = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            float recon = params.center + (float)dst[d] * (params.scale * 0.5f);
            float diff = src[d] - recon;
            err_sq += diff * diff;
        }
        float err = sqrtf(err_sq);
        if (err > max_observed_err)
        {
            max_observed_err = err;
        }

        /* Strict guarantee: observed error must never exceed err_radius */
        assert(err <= params.err_radius + 1e-4f);
    }

    printf("  dim=%ld: err_radius=%.6f, max_observed_err=%.6f (Passed)\n",
           dim, params.err_radius, max_observed_err);

    free(src);
    free(dst);
}

static void test_eq16_metric_lower_bound(void)
{
    printf("Testing EQ16 metric lower bounding guarantees...\n");

    long dim = 64;
    EQ16Params p_eq16;
    SQ16Params p_sq16;
    eq16_init_params(&p_eq16, -5.0f, 5.0f, dim);
    sq16_init_params(&p_sq16, -5.0f, 5.0f, dim);

    float *x = (float *)malloc(dim * sizeof(float));
    float *y = (float *)malloc(dim * sizeof(float));
    int16_t *qx_eq = (int16_t *)malloc(dim * sizeof(int16_t));
    int16_t *qy_eq = (int16_t *)malloc(dim * sizeof(int16_t));
    int16_t *qx_sq = (int16_t *)malloc(dim * sizeof(int16_t));
    int16_t *qy_sq = (int16_t *)malloc(dim * sizeof(int16_t));

    srand(12345);
    int tighter_count = 0;

    for (int t = 0; t < 10000; t++)
    {
        float true_sq = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            x[d] = -5.0f + 10.0f * ((float)rand() / (float)RAND_MAX);
            y[d] = -5.0f + 10.0f * ((float)rand() / (float)RAND_MAX);
            float diff = x[d] - y[d];
            true_sq += diff * diff;
        }
        double d_true = (double)sqrtf(true_sq);

        eq16_quantize_float(x, qx_eq, &p_eq16);
        eq16_quantize_float(y, qy_eq, &p_eq16);

        sq16_quantize_float(x, qx_sq, &p_sq16);
        sq16_quantize_float(y, qy_sq, &p_sq16);

        double lb_eq = eq16_compute_lower_bound(qx_eq, qy_eq, &p_eq16, 0.0);
        double lb_sq = sq16_compute_lower_bound(qx_sq, qy_sq, &p_sq16, 0.0);

        /* Lower bound must be conservative: never exceed true distance */
        assert(lb_eq <= d_true + 1e-4);

        /* Check bound tightness */
        if (lb_eq >= lb_sq - 1e-6)
        {
            tighter_count++;
        }
    }

    printf("  10,000 trials: 0 false dismissals, EQ16 bound tighter/equal in %d / 10000 trials\n",
           tighter_count);
    assert(tighter_count >= 9900);

    free(x);
    free(y);
    free(qx_eq);
    free(qy_eq);
    free(qx_sq);
    free(qy_sq);
}

static void test_eq16_sidecar_roundtrip(void)
{
    printf("Testing EQ16 sidecar save and load...\n");

    long dim = 32;
    long num_frames = 100;
    EQ16Params p_save;
    eq16_init_params(&p_save, -3.0f, 7.0f, dim);

    int16_t *data_save = (int16_t *)malloc(dim * num_frames * sizeof(int16_t));
    for (long i = 0; i < dim * num_frames; i++)
    {
        data_save[i] = (int16_t)(i % 1000 - 500);
    }

    const char *tmp_path = "/tmp/test_eq16_sidecar.eq16";
    int res_save = eq16_save_sidecar(tmp_path, &p_save, data_save, num_frames);
    assert(res_save == 0);

    EQ16Params p_load;
    int16_t *data_load = NULL;
    long loaded_frames = 0;
    int res_load = eq16_load_sidecar(tmp_path, &p_load, &data_load, &loaded_frames);
    assert(res_load == 0);
    assert(loaded_frames == num_frames);
    assert(p_load.dim == dim);
    assert(fabsf(p_load.center - p_save.center) < 1e-7f);
    assert(fabsf(p_load.scale - p_save.scale) < 1e-7f);
    assert(fabsf(p_load.err_radius - p_save.err_radius) < 1e-7f);
    assert(memcmp(data_save, data_load, dim * num_frames * sizeof(int16_t)) == 0);

    free(data_save);
    free(data_load);
    unlink(tmp_path);
    printf("  Sidecar roundtrip verified (Passed)\n");
}

static void test_eq16_adc_bounding_and_cutoff(void)
{
    printf("Testing EQ16 Asymmetric Distance Computation (ADC) guarantees...\n");

    long dim = 64;
    EQ16Params p_eq16;
    eq16_init_params(&p_eq16, -5.0f, 5.0f, dim);

    float *q = (float *)malloc(dim * sizeof(float));
    float *x = (float *)malloc(dim * sizeof(float));
    float *q_adc = (float *)malloc(dim * sizeof(float));
    int16_t *qx_eq = (int16_t *)malloc(dim * sizeof(int16_t));
    int16_t *cand_eq = (int16_t *)malloc(dim * sizeof(int16_t));

    srand(54321);
    int adc_tighter_count = 0;

    for (int t = 0; t < 10000; t++)
    {
        float true_sq = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            q[d] = -5.0f + 10.0f * ((float)rand() / (float)RAND_MAX);
            x[d] = -5.0f + 10.0f * ((float)rand() / (float)RAND_MAX);
            float diff = q[d] - x[d];
            true_sq += diff * diff;
        }
        double d_true = (double)sqrtf(true_sq);

        eq16_prepare_query_adc_float(q, q_adc, &p_eq16);
        eq16_quantize_float(q, qx_eq, &p_eq16);
        eq16_quantize_float(x, cand_eq, &p_eq16);

        /* SDC distance and lower bound: lb = d(hat{q}, hat{x}) - 2*Rc */
        uint64_t sdc_ssd = eq16_dist_squared_i16(qx_eq, cand_eq, dim);
        double d_sdc = sqrt((double)sdc_ssd) * ((double)p_eq16.scale * 0.5);
        double lb_sdc = d_sdc - 2.0 * (double)p_eq16.err_radius;

        /* ADC distance and lower bound: lb = d(q, hat{x}) - 1*Rc */
        float adc_dist_sq = eq16_dist_asym_cutoff_f32(q_adc, cand_eq, dim, 1e30f);
        double d_adc = sqrt((double)adc_dist_sq) * ((double)p_eq16.scale * 0.5);
        double lb_adc = d_adc - 1.0 * (double)p_eq16.err_radius;

        /* Safety check: ADC lower bound must NEVER exceed true Euclidean distance */
        assert(lb_adc <= d_true + 1e-4);

        /* ADC lower bound must be tighter than or equal to SDC in virtually all cases */
        if (lb_adc >= lb_sdc - 1e-6)
        {
            adc_tighter_count++;
        }

        /* Test cutoff threshold early-exit */
        float exact_cutoff = adc_dist_sq * 0.5f;
        float early_val = eq16_dist_asym_cutoff_f32(q_adc, cand_eq, dim, exact_cutoff);
        assert(early_val > exact_cutoff);
    } // for (int t = 0; t < 10000; t++)

    printf("  10,000 trials: 0 false dismissals, ADC bound tighter in %d / 10000 trials (Passed)\n",
           adc_tighter_count);
    assert(adc_tighter_count >= 9950);

    free(q);
    free(x);
    free(q_adc);
    free(qx_eq);
    free(cand_eq);
}

static void test_eq16_filter_anchor_matrix_adc(void)
{
    printf("Testing EQ16 ADC Anchor Matrix Filtering (SIMD & Early Exit)...\n");

    long dim = 128;
    int num_clusters = 512;
    EQ16Params p_eq16;
    eq16_init_params(&p_eq16, -10.0f, 10.0f, dim);

    float *anchors_raw = (float *)malloc(num_clusters * dim * sizeof(float));
    int16_t *anchor_matrix = (int16_t *)malloc(num_clusters * dim * sizeof(int16_t));
    float *query = (float *)malloc(dim * sizeof(float));
    float *query_adc = (float *)malloc(dim * sizeof(float));
    int16_t *query_eq16 = (int16_t *)malloc(dim * sizeof(int16_t));

    int *clmembflag_adc = (int *)malloc(num_clusters * sizeof(int));
    int *clmembflag_sdc = (int *)malloc(num_clusters * sizeof(int));
    int *active_clusters_adc = (int *)malloc(num_clusters * sizeof(int));
    int *active_clusters_sdc = (int *)malloc(num_clusters * sizeof(int));

    srand(98765);
    for (int i = 0; i < num_clusters * dim; i++)
    {
        anchors_raw[i] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
    }
    for (int c = 0; c < num_clusters; c++)
    {
        eq16_quantize_float(&anchors_raw[c * dim], &anchor_matrix[c * dim], &p_eq16);
    }

    size_t num_blocks_adc = ((size_t)num_clusters + 15) / 16;
    size_t adc_elements = num_blocks_adc * (size_t)dim * 16;
    float *anchor_adc_interleaved = NULL;
    int mem_ret = posix_memalign(
        (void **)&anchor_adc_interleaved, 64, adc_elements * sizeof(float)
    );
    assert(mem_ret == 0 && anchor_adc_interleaved != NULL);
    eq16_rebuild_anchor_adc_interleaved(anchor_adc_interleaved, anchor_matrix,
                                        num_clusters, dim);

    int *clmembflag_fallback = (int *)malloc(num_clusters * sizeof(int));
    int *active_clusters_fallback = (int *)malloc(num_clusters * sizeof(int));

    int test_queries = 20;
    double rlim = 3.5;
    double inv_step = 2.0 / (double)p_eq16.scale;
    double cut_adc_unsq = (rlim + (double)p_eq16.err_radius) * inv_step;
    float cutoff_adc = (float)(cut_adc_unsq * cut_adc_unsq);

    double cut_sdc_unsq = (rlim + 2.0 * (double)p_eq16.err_radius) * inv_step;
    uint64_t cutoff_sdc = (uint64_t)ceil(cut_sdc_unsq * cut_sdc_unsq);

    for (int q_idx = 0; q_idx < test_queries; q_idx++)
    {
        for (int d = 0; d < dim; d++)
        {
            query[d] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
        }
        eq16_prepare_query_adc_float(query, query_adc, &p_eq16);
        eq16_quantize_float(query, query_eq16, &p_eq16);

        for (int c = 0; c < num_clusters; c++)
        {
            clmembflag_adc[c] = 1;
            clmembflag_fallback[c] = 1;
            clmembflag_sdc[c] = 1;
            active_clusters_sdc[c] = c;
        }

        int num_active_adc = 0;
        long pruned_adc = 0;
        eq16_filter_anchor_matrix_adc(
            query_adc,
            anchor_matrix,
            anchor_adc_interleaved,
            num_clusters,
            dim,
            cutoff_adc,
            clmembflag_adc,
            active_clusters_adc,
            &num_active_adc,
            &pruned_adc
        );

        int num_active_fb = 0;
        long pruned_fb = 0;
        eq16_filter_anchor_matrix_adc(
            query_adc,
            anchor_matrix,
            NULL,
            num_clusters,
            dim,
            cutoff_adc,
            clmembflag_fallback,
            active_clusters_fallback,
            &num_active_fb,
            &pruned_fb
        );

        /* Verify Block-16 interleaved kernel matches fallback exactly */
        assert(num_active_adc == num_active_fb);
        assert(pruned_adc == pruned_fb);
        for (int c = 0; c < num_clusters; c++)
        {
            assert(clmembflag_adc[c] == clmembflag_fallback[c]);
        }

        int num_active_sdc = 0;
        long pruned_sdc = 0;
        eq16_filter_anchor_matrix(
            query_eq16,
            anchor_matrix,
            NULL,
            num_clusters,
            dim,
            cutoff_sdc,
            clmembflag_sdc,
            active_clusters_sdc,
            &num_active_sdc,
            &pruned_sdc
        );

        /* Verify active list matches flags */
        assert(num_active_adc + pruned_adc == num_clusters);
        for (int a = 0; a < num_active_adc; a++)
        {
            int c = active_clusters_adc[a];
            assert(clmembflag_adc[c] == 1);
        }

        /* Verify zero false dismissals against true Euclidean distance */
        for (int c = 0; c < num_clusters; c++)
        {
            double dist_sq = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double diff = (double)query[d] - (double)anchors_raw[c * dim + d];
                dist_sq += diff * diff;
            }
            double true_d = sqrt(dist_sq);

            if (true_d < rlim)
            {
                /* Ground truth neighbor must NEVER be pruned */
                assert(clmembflag_adc[c] == 1);
            }
        }

        /* ADC slack is half of SDC (1*Rc vs 2*Rc), so ADC active candidates <= SDC */
        assert(num_active_adc <= num_active_sdc);
    } // for (int q_idx = 0; q_idx < test_queries; q_idx++)

    printf("  EQ16 ADC Anchor Matrix Filtering: 0 false dismissals, "
           "strictly tighter pruning than SDC (Passed)\n");

    free(anchors_raw);
    free(anchor_matrix);
    free(anchor_adc_interleaved);
    free(query);
    free(query_adc);
    free(query_eq16);
    free(clmembflag_adc);
    free(clmembflag_fallback);
    free(clmembflag_sdc);
    free(active_clusters_adc);
    free(active_clusters_fallback);
    free(active_clusters_sdc);
}

static void test_eq16_cascaded_screening(void)
{
    printf("Testing EQ16 Cascaded Two-Tier Screening (SDC -> ADC Refinement)...\n");

    long dim = 128;
    int num_clusters = 512;
    EQ16Params p_eq16;
    eq16_init_params(&p_eq16, -10.0f, 10.0f, dim);

    int16_t *anchor_matrix = (int16_t *)malloc(num_clusters * dim * sizeof(int16_t));
    float *query = (float *)malloc(dim * sizeof(float));
    float *query_adc = (float *)malloc(dim * sizeof(float));
    int16_t *query_eq16 = (int16_t *)malloc(dim * sizeof(int16_t));

    int *clmembflag_adc = (int *)malloc(num_clusters * sizeof(int));
    int *active_clusters_adc = (int *)malloc(num_clusters * sizeof(int));
    int *clmembflag_cascaded = (int *)malloc(num_clusters * sizeof(int));
    int *active_clusters_cascaded = (int *)malloc(num_clusters * sizeof(int));

    srand(54321);
    for (int c = 0; c < num_clusters; c++)
    {
        float raw[128];
        for (int d = 0; d < dim; d++)
        {
            raw[d] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
        }
        eq16_quantize_float(raw, &anchor_matrix[c * dim], &p_eq16);
    }

    size_t num_blocks_adc = ((size_t)num_clusters + 15) / 16;
    size_t adc_elements = num_blocks_adc * (size_t)dim * 16;
    float *anchor_adc_interleaved = NULL;
    int mem_ret = posix_memalign(
        (void **)&anchor_adc_interleaved, 64, adc_elements * sizeof(float)
    );
    assert(mem_ret == 0 && anchor_adc_interleaved != NULL);
    eq16_rebuild_anchor_adc_interleaved(
        anchor_adc_interleaved, anchor_matrix, num_clusters, dim
    );

    int num_pairs = (int)((dim + 1) / 2);
    int num_blocks_sdc = (num_clusters + 7) / 8;
    size_t sdc_elements = (size_t)num_blocks_sdc * (size_t)num_pairs * 8;
    int32_t *anchor_sdc_interleaved = NULL;
    mem_ret = posix_memalign(
        (void **)&anchor_sdc_interleaved, 64, sdc_elements * sizeof(int32_t)
    );
    assert(mem_ret == 0 && anchor_sdc_interleaved != NULL);
    eq16_rebuild_anchor_interleaved(
        anchor_sdc_interleaved, anchor_matrix, num_clusters, dim
    );

    int test_queries = 25;
    double rlim = 3.5;
    double inv_step = 2.0 / (double)p_eq16.scale;
    double cut_adc_unsq = (rlim + (double)p_eq16.err_radius) * inv_step;
    float cutoff_adc = (float)(cut_adc_unsq * cut_adc_unsq);

    double cut_sdc_unsq = (rlim + 2.0 * (double)p_eq16.err_radius) * inv_step;
    uint64_t cutoff_sdc = (uint64_t)ceil(cut_sdc_unsq * cut_sdc_unsq);

    for (int q_idx = 0; q_idx < test_queries; q_idx++)
    {
        for (int d = 0; d < dim; d++)
        {
            query[d] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
        }
        eq16_prepare_query_adc_float(query, query_adc, &p_eq16);
        eq16_quantize_float(query, query_eq16, &p_eq16);

        for (int c = 0; c < num_clusters; c++)
        {
            clmembflag_adc[c] = 1;
            clmembflag_cascaded[c] = 1;
        }

        /* 1. Direct ADC screening */
        int num_active_adc = 0;
        long pruned_adc = 0;
        eq16_filter_anchor_matrix_adc(
            query_adc,
            anchor_matrix,
            anchor_adc_interleaved,
            num_clusters,
            dim,
            cutoff_adc,
            clmembflag_adc,
            active_clusters_adc,
            &num_active_adc,
            &pruned_adc
        );

        /* 2. Cascaded screening: Tier 1 SDC */
        int num_active_cascaded = 0;
        long pruned_cascaded = 0;
        eq16_filter_anchor_matrix(
            query_eq16,
            anchor_matrix,
            anchor_sdc_interleaved,
            num_clusters,
            dim,
            cutoff_sdc,
            clmembflag_cascaded,
            active_clusters_cascaded,
            &num_active_cascaded,
            &pruned_cascaded
        );

        /* Tier 2 ADC refinement */
        int next_num_active = 0;
        for (int idx = 0; idx < num_active_cascaded; idx++)
        {
            int c = active_clusters_cascaded[idx];
            const int16_t *cand_anchor = anchor_matrix + (size_t)c * (size_t)dim;
            float dsq = eq16_dist_asym_cutoff_f32(
                query_adc, cand_anchor, dim, cutoff_adc
            );
            if (dsq > cutoff_adc)
            {
                clmembflag_cascaded[c] = 0;
                pruned_cascaded++;
            }
            else
            {
                active_clusters_cascaded[next_num_active++] = c;
            }
        }
        num_active_cascaded = next_num_active;

        /* Verify exact mathematical equivalence */
        assert(num_active_cascaded == num_active_adc);
        assert(pruned_cascaded == pruned_adc);
        for (int c = 0; c < num_clusters; c++)
        {
            assert(clmembflag_cascaded[c] == clmembflag_adc[c]);
        }
        for (int k = 0; k < num_active_adc; k++)
        {
            assert(active_clusters_cascaded[k] == active_clusters_adc[k]);
        }
    } // for (int q_idx = 0; q_idx < test_queries; q_idx++)

    printf("  EQ16 Cascaded Screening: Exact equivalence to direct ADC verified (Passed)\n");

    free(anchor_matrix);
    free(anchor_adc_interleaved);
    free(anchor_sdc_interleaved);
    free(query);
    free(query_adc);
    free(query_eq16);
    free(clmembflag_adc);
    free(active_clusters_adc);
    free(clmembflag_cascaded);
    free(active_clusters_cascaded);
}

static void test_eq16_permutation(void)
{
    printf("Testing EQ16 Dimension Permutation (Float & Double)...\n");

    long dim = 64;
    EQ16Params params;
    eq16_init_params(&params, -5.0f, 5.0f, dim);

    float *src_f = (float *)malloc((size_t)dim * sizeof(float));
    float *src_f_perm = (float *)malloc((size_t)dim * sizeof(float));
    double *src_d = (double *)malloc((size_t)dim * sizeof(double));
    double *src_d_perm = (double *)malloc((size_t)dim * sizeof(double));
    int16_t *dst_f1 = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
    int16_t *dst_f2 = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
    int16_t *dst_d1 = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
    int16_t *dst_d2 = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
    float *adc_f1 = (float *)malloc((size_t)dim * sizeof(float));
    float *adc_f2 = (float *)malloc((size_t)dim * sizeof(float));
    float *adc_d1 = (float *)malloc((size_t)dim * sizeof(float));
    float *adc_d2 = (float *)malloc((size_t)dim * sizeof(float));
    long *perm_dim = (long *)malloc((size_t)dim * sizeof(long));

    assert(src_f && src_f_perm && src_d && src_d_perm &&
           dst_f1 && dst_f2 && dst_d1 && dst_d2 &&
           adc_f1 && adc_f2 && adc_d1 && adc_d2 && perm_dim);

    /* Generate random permutation using Fisher-Yates */
    for (long i = 0; i < dim; i++)
    {
        perm_dim[i] = i;
    }
    for (long i = dim - 1; i > 0; i--)
    {
        long j = (long)(rand() % (i + 1));
        long tmp = perm_dim[i];
        perm_dim[i] = perm_dim[j];
        perm_dim[j] = tmp;
    }

    for (int trial = 0; trial < 100; trial++)
    {
        for (long i = 0; i < dim; i++)
        {
            float val = -4.0f + 8.0f * ((float)rand() / (float)RAND_MAX);
            src_f[i] = val;
            src_d[i] = (double)val;
        }
        for (long i = 0; i < dim; i++)
        {
            src_f_perm[i] = src_f[perm_dim[i]];
            src_d_perm[i] = src_d[perm_dim[i]];
        }

        /* Test float quantization */
        eq16_quantize_float_perm(src_f, dst_f1, &params, perm_dim);
        eq16_quantize_float(src_f_perm, dst_f2, &params);
        for (long i = 0; i < dim; i++)
        {
            assert(dst_f1[i] == dst_f2[i]);
        }

        /* Test float ADC query */
        eq16_prepare_query_adc_float_perm(src_f, adc_f1, &params, perm_dim);
        eq16_prepare_query_adc_float(src_f_perm, adc_f2, &params);
        for (long i = 0; i < dim; i++)
        {
            assert(fabsf(adc_f1[i] - adc_f2[i]) < 1e-5f);
        }

        /* Test double quantization */
        eq16_quantize_double_perm(src_d, dst_d1, &params, perm_dim);
        eq16_quantize_double(src_d_perm, dst_d2, &params);
        for (long i = 0; i < dim; i++)
        {
            assert(dst_d1[i] == dst_d2[i]);
        }

        /* Test double ADC query */
        eq16_prepare_query_adc_double_perm(src_d, adc_d1, &params, perm_dim);
        eq16_prepare_query_adc_double(src_d_perm, adc_d2, &params);
        for (long i = 0; i < dim; i++)
        {
            assert(fabsf(adc_d1[i] - adc_d2[i]) < 1e-5f);
        }
    }

    printf("  EQ16 Dimension Permutation: Float & Double verified across 100 trials (Passed)\n");

    free(src_f);
    free(src_f_perm);
    free(src_d);
    free(src_d_perm);
    free(dst_f1);
    free(dst_f2);
    free(dst_d1);
    free(dst_d2);
    free(adc_f1);
    free(adc_f2);
    free(adc_d1);
    free(adc_d2);
    free(perm_dim);
}

static void test_eq16_adc_batch_1x4(void)
{
    printf("Testing EQ16 ADC Batch 1x4 and Candidate Refinement...\n");

    long test_dims[] = {8, 16, 24, 32, 64, 128, 512};
    size_t num_test_dims = sizeof(test_dims) / sizeof(test_dims[0]);

    srand(12345);

    for (size_t d_idx = 0; d_idx < num_test_dims; d_idx++)
    {
        long dim = test_dims[d_idx];
        EQ16Params params;
        eq16_init_params(&params, -10.0f, 10.0f, dim);

        float *raw_q = (float *)malloc((size_t)dim * sizeof(float));
        float *q_adc = (float *)malloc((size_t)dim * sizeof(float));
        int16_t *cands_buf = (int16_t *)malloc(8 * (size_t)dim * sizeof(int16_t));
        const int16_t *cands[8];
        for (int k = 0; k < 8; k++)
        {
            cands[k] = cands_buf + (size_t)k * (size_t)dim;
        }

        assert(raw_q != NULL && q_adc != NULL && cands_buf != NULL);

        for (int trial = 0; trial < 20; trial++)
        {
            for (long i = 0; i < dim; i++)
            {
                raw_q[i] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
            }
            eq16_prepare_query_adc_float(raw_q, q_adc, &params);

            for (int k = 0; k < 8; k++)
            {
                float tmp[512];
                for (long i = 0; i < dim; i++)
                {
                    tmp[i] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
                }
                eq16_quantize_float(tmp, (int16_t *)cands[k], &params);
            }

            /* 1. Full distance match without early cutoff (1x4 and 1x8) */
            float d_batch4[4];
            eq16_dist_asym_cutoff_batch_1x4(q_adc, cands, dim, 1e15f, d_batch4);

            float d_batch8[8];
            eq16_dist_asym_cutoff_batch_1x8(q_adc, cands, dim, 1e15f, d_batch8);

            float d_seq[8];
            for (int k = 0; k < 8; k++)
            {
                d_seq[k] = eq16_dist_asym_cutoff_f32(q_adc, cands[k], dim, 1e15f);
                float rel_diff = fabsf(d_batch8[k] - d_seq[k]) / (d_seq[k] + 1.0f);
                assert(rel_diff < 1e-4f);
                if (k < 4)
                {
                    float rel_diff4 = fabsf(d_batch4[k] - d_seq[k]) / (d_seq[k] + 1.0f);
                    assert(rel_diff4 < 1e-4f);
                }
            }

            /* 2. Pruning decision consistency with realistic cutoff (1x4 and 1x8) */
            float mid_cutoff = 0.0f;
            for (int k = 0; k < 8; k++)
            {
                mid_cutoff += d_seq[k];
            }
            mid_cutoff *= (1.0f / 8.0f);

            float d_batch8_cut[8];
            eq16_dist_asym_cutoff_batch_1x8(q_adc, cands, dim, mid_cutoff, d_batch8_cut);

            float d_batch4_cut[4];
            eq16_dist_asym_cutoff_batch_1x4(q_adc, cands, dim, mid_cutoff, d_batch4_cut);

            for (int k = 0; k < 8; k++)
            {
                bool pruned_batch8 = (d_batch8_cut[k] > mid_cutoff);
                bool pruned_seq = (d_seq[k] > mid_cutoff);
                assert(pruned_batch8 == pruned_seq);
                if (k < 4)
                {
                    bool pruned_batch4 = (d_batch4_cut[k] > mid_cutoff);
                    assert(pruned_batch4 == pruned_seq);
                }
            }
        } // for (int trial = 0; trial < 20; trial++)

        free(raw_q);
        free(q_adc);
        free(cands_buf);
    } // for (size_t d_idx = 0; d_idx < num_test_dims; d_idx++)

    /* 3. Test eq16_refine_candidates_adc against sequential reference */
    {
        long dim = 512;
        int num_clusters = 50;
        int num_active = 27; /* Not a multiple of 4, tests remainder loop */

        EQ16Params params;
        eq16_init_params(&params, -10.0f, 10.0f, dim);

        float *raw_q = (float *)malloc((size_t)dim * sizeof(float));
        float *q_adc = (float *)malloc((size_t)dim * sizeof(float));
        int16_t *mat_eq16 = (int16_t *)malloc(
            (size_t)num_clusters * (size_t)dim * sizeof(int16_t)
        );

        int *active_clusters_ref = (int *)malloc((size_t)num_active * sizeof(int));
        int *active_clusters_bat = (int *)malloc((size_t)num_active * sizeof(int));
        int *clmembflag_ref = (int *)malloc((size_t)num_clusters * sizeof(int));
        int *clmembflag_bat = (int *)malloc((size_t)num_clusters * sizeof(int));

        assert(raw_q && q_adc && mat_eq16 && active_clusters_ref &&
               active_clusters_bat && clmembflag_ref && clmembflag_bat);

        for (long i = 0; i < dim; i++)
        {
            raw_q[i] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
        }
        eq16_prepare_query_adc_float(raw_q, q_adc, &params);

        for (int c = 0; c < num_clusters; c++)
        {
            float tmp[512];
            for (long i = 0; i < dim; i++)
            {
                tmp[i] = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
            }
            eq16_quantize_float(tmp, mat_eq16 + (size_t)c * (size_t)dim, &params);
        }

        for (int c = 0; c < num_clusters; c++)
        {
            clmembflag_ref[c] = 0;
            clmembflag_bat[c] = 0;
        }

        for (int a = 0; a < num_active; a++)
        {
            int c = a * 1;
            active_clusters_ref[a] = c;
            active_clusters_bat[a] = c;
            clmembflag_ref[c] = 1;
            clmembflag_bat[c] = 1;
        }

        float cutoff = 250.0f;

        /* Sequential reference */
        int next_num_active_ref = 0;
        long pruned_count_ref = 0;
        for (int idx = 0; idx < num_active; idx++)
        {
            int c = active_clusters_ref[idx];
            const int16_t *cand_anchor = mat_eq16 + (size_t)c * (size_t)dim;
            float dsq = eq16_dist_asym_cutoff_f32(q_adc, cand_anchor, dim, cutoff);
            if (dsq > cutoff)
            {
                clmembflag_ref[c] = 0;
                pruned_count_ref++;
            }
            else
            {
                active_clusters_ref[next_num_active_ref++] = c;
            }
        }

        /* Batched refinement */
        int next_num_active_bat = 0;
        long pruned_count_bat = 0;
        eq16_refine_candidates_adc(
            q_adc,
            mat_eq16,
            active_clusters_bat,
            num_active,
            dim,
            cutoff,
            clmembflag_bat,
            &next_num_active_bat,
            &pruned_count_bat
        );

        assert(next_num_active_ref == next_num_active_bat);
        assert(pruned_count_ref == pruned_count_bat);

        for (int i = 0; i < next_num_active_ref; i++)
        {
            assert(active_clusters_ref[i] == active_clusters_bat[i]);
        }

        for (int c = 0; c < num_clusters; c++)
        {
            assert(clmembflag_ref[c] == clmembflag_bat[c]);
        }

        free(raw_q);
        free(q_adc);
        free(mat_eq16);
        free(active_clusters_ref);
        free(active_clusters_bat);
        free(clmembflag_ref);
        free(clmembflag_bat);
    }

    printf("  EQ16 ADC Batch 1x4 and Refinement: verified (Passed)\n");
}

static void test_eq16_fastscan_32x(void)
{
    printf("Testing EQ16 Block-Transposed FastScan (ADC and SDC) across dimensions...\n");

    long test_dims[] = {8, 16, 64, 128, 512};
    srand(12345);

    for (size_t di = 0; di < sizeof(test_dims) / sizeof(test_dims[0]); di++)
    {
        long dim = test_dims[di];

        float *q_adc = (float *)malloc((size_t)dim * sizeof(float));
        int16_t *query_i16 = (int16_t *)malloc((size_t)dim * sizeof(int16_t));
        int16_t *block = (int16_t *)malloc((size_t)dim * 32 * sizeof(int16_t));
        assert(q_adc != NULL && query_i16 != NULL && block != NULL);

        for (long d = 0; d < dim; d++)
        {
            q_adc[d] = (float)(rand() % 2000 - 1000);
            query_i16[d] = (int16_t)(rand() % 2000 - 1000);
            for (int i = 0; i < 32; i++)
            {
                block[d * 32 + i] = (int16_t)(rand() % 2000 - 1000);
            }
        }

        /* Calculate ground truth distances */
        float gt_adc[32];
        uint64_t gt_sdc[32];
        for (int i = 0; i < 32; i++)
        {
            gt_adc[i] = 0.0f;
            gt_sdc[i] = 0;
            for (long d = 0; d < dim; d++)
            {
                float diff_adc = q_adc[d] - (float)block[d * 32 + i];
                gt_adc[i] += diff_adc * diff_adc;

                int64_t diff_sdc = (int64_t)query_i16[d] - (int64_t)block[d * 32 + i];
                gt_sdc[i] += (uint64_t)(diff_sdc * diff_sdc);
            }
        }

        /* Test ADC FastScan: low, median, and high cutoff */
        float median_adc = gt_adc[16];
        float cutoffs_adc[] = {-1.0f, median_adc, 1e20f};
        for (int c = 0; c < 3; c++)
        {
            float cut = cutoffs_adc[c];
            uint32_t expected_mask = 0;
            for (int i = 0; i < 32; i++)
            {
                if (gt_adc[i] <= cut)
                {
                    expected_mask |= (1U << i);
                }
            }
            uint32_t simd_mask = eq16_fastscan_32x_adc(q_adc, block, dim, cut);
            assert(simd_mask == expected_mask);
        }

        /* Test SDC FastScan: low, median, and high cutoff */
        uint64_t median_sdc = gt_sdc[16];
        uint64_t cutoffs_sdc[] = {0, median_sdc, UINT64_MAX / 2};
        for (int c = 0; c < 3; c++)
        {
            uint64_t cut = cutoffs_sdc[c];
            uint32_t expected_mask = 0;
            for (int i = 0; i < 32; i++)
            {
                if (gt_sdc[i] <= cut)
                {
                    expected_mask |= (1U << i);
                }
            }
            uint32_t simd_mask = eq16_fastscan_32x_i16(query_i16, block, dim, cut);
            assert(simd_mask == expected_mask);
        }

        free(q_adc);
        free(query_i16);
        free(block);
    } // for (size_t di = 0; di < sizeof(test_dims) / sizeof(test_dims[0]); di++)

    printf("  EQ16 FastScan (ADC and SDC): verified (Passed)\n");
}

/**
 * test_eq16_dist_squared_i16_kernels() - Verify EQ16 i16 distance kernels against exact math.
 */
static void test_eq16_dist_squared_i16_kernels(void)
{
    printf("Testing EQ16 dist_squared_i16, cutoff, and batch_1x4 kernels...\n");

    long test_dims[] = {8, 16, 24, 64, 128, 512};
    long max_dim = 512;
    int16_t *q = (int16_t *)malloc((size_t)max_dim * sizeof(int16_t));
    int16_t *anchors[4] = {
        (int16_t *)malloc((size_t)max_dim * sizeof(int16_t)),
        (int16_t *)malloc((size_t)max_dim * sizeof(int16_t)),
        (int16_t *)malloc((size_t)max_dim * sizeof(int16_t)),
        (int16_t *)malloc((size_t)max_dim * sizeof(int16_t))
    };

    srand(54321);

    for (size_t di = 0; di < sizeof(test_dims) / sizeof(test_dims[0]); di++)
    {
        long dim = test_dims[di];

        /* 1. Test moderate ranges (typical clustering) */
        for (int trial = 0; trial < 100; trial++)
        {
            for (long d = 0; d < dim; d++)
            {
                q[d] = (int16_t)(rand() % 4000 - 2000);
                for (int k = 0; k < 4; k++)
                {
                    anchors[k][d] = (int16_t)(rand() % 4000 - 2000);
                }
            }

            uint64_t batch_out[4];
            eq16_dist_squared_batch_1x4_i16(
                q, (const int16_t *const *)anchors, batch_out, dim);

            for (int k = 0; k < 4; k++)
            {
                uint64_t exact = 0;
                for (long d = 0; d < dim; d++)
                {
                    int64_t diff = (int64_t)q[d] - (int64_t)anchors[k][d];
                    exact += (uint64_t)(diff * diff);
                }

                uint64_t dist = eq16_dist_squared_i16(q, anchors[k], dim);
                assert(dist == exact);
                assert(batch_out[k] == exact);

                uint64_t cut_pass = eq16_dist_squared_cutoff_i16(
                    q, anchors[k], dim, exact + 10);
                assert(cut_pass == exact);

                if (exact > 10)
                {
                    uint64_t cut_fail = eq16_dist_squared_cutoff_i16(
                        q, anchors[k], dim, exact - 10);
                    assert(cut_fail > exact - 10);
                }
            }
        }

        /* 2. Test full-range extreme differences (-32768 to 32767) */
        for (int trial = 0; trial < 100; trial++)
        {
            for (long d = 0; d < dim; d++)
            {
                q[d] = (int16_t)(rand() % 65536 - 32768);
                for (int k = 0; k < 4; k++)
                {
                    anchors[k][d] = (int16_t)(rand() % 65536 - 32768);
                }
            }

            uint64_t batch_out[4];
            eq16_dist_squared_batch_1x4_i16(
                q, (const int16_t *const *)anchors, batch_out, dim);

            for (int k = 0; k < 4; k++)
            {
                uint64_t exact = 0;
                for (long d = 0; d < dim; d++)
                {
                    int64_t diff = (int64_t)q[d] - (int64_t)anchors[k][d];
                    exact += (uint64_t)(diff * diff);
                }

                uint64_t dist = eq16_dist_squared_i16(q, anchors[k], dim);
                assert(dist == exact);
                assert(batch_out[k] == exact);
            }
        }
    } // for (size_t di = 0; di < sizeof(test_dims) / sizeof(test_dims[0]); di++)

    free(q);
    for (int k = 0; k < 4; k++)
    {
        free(anchors[k]);
    }

    printf("  EQ16 dist_squared_i16, cutoff, and batch_1x4: verified (Passed)\n");
}

int main(void)
{
    printf("=========================================\n");
    printf(" RUNNING EQ16 UNIT TESTS\n");
    printf("=========================================\n");

    e8_init_root_table();

    test_eq16_params();
    test_eq16_quantization_accuracy();
    test_eq16_metric_lower_bound();
    test_eq16_dist_squared_i16_kernels();
    test_eq16_adc_bounding_and_cutoff();
    test_eq16_filter_anchor_matrix_adc();
    test_eq16_cascaded_screening();
    test_eq16_permutation();
    test_eq16_adc_batch_1x4();
    test_eq16_fastscan_32x();
    test_eq16_sidecar_roundtrip();

    printf("\nALL EQ16 TESTS PASSED SUCCESSFULLY.\n");
    return 0;
}

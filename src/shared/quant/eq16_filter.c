/**
 * @file eq16_filter.c
 * @brief High-throughput anchor matrix filtering and interleaving for EQ16.
 */

#include "eq16_quant.h"
#include "scalar_quant.h"
#include "gric_simd.h"
#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif
#include <math.h>
#include <string.h>
#include <stdio.h>

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

/**
 * eq16_filter_anchor_matrix_avx_vnni() - AVX-VNNI kernel for anchor filtering.
 * @cur_eq16:           Quantized query coordinate vector [dim].
 * @anchor_matrix:      Row-major fallback anchor matrix [num_clusters x dim].
 * @anchor_interleaved: Block-8 interleaved anchor matrix.
 * @num_clusters:       Number of clusters.
 * @dim:                Vector dimensionality.
 * @eq16_ssd_thresh:    Squared distance cutoff threshold.
 * @clmembflag:         Candidate flags array (0 = pruned, 1 = active).
 * @active_clusters:    Surviving cluster indices array.
 * @out_num_active:     Count of surviving clusters.
 * @out_pruned_count:   Count of pruned clusters.
 *
 * Accelerates EQ16 anchor matrix screening on CPUs with hardware AVX-VNNI
 * support using _mm256_dpwssd_epi32 and pre-broadcasted query coordinate pairs.
 */
GRIC_TARGET_AVX_VNNI
static void eq16_filter_anchor_matrix_avx_vnni(
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

    #define EQ16_PREBROADCAST_PAIRS_VNNI 512
    __m256i q_pairs_stack[EQ16_PREBROADCAST_PAIRS_VNNI];
    int pre_broadcast_count = (num_pairs < EQ16_PREBROADCAST_PAIRS_VNNI)
                              ? num_pairs
                              : EQ16_PREBROADCAST_PAIRS_VNNI;

    for (int j = 0; j < pre_broadcast_count; j++)
    {
        uint32_t v0 = (uint32_t)(uint16_t)cur_eq16[2 * j];
        uint32_t v1 = (2 * j + 1 < dim)
                      ? (uint32_t)(uint16_t)cur_eq16[2 * j + 1]
                      : 0;
        uint32_t p = (v1 << 16) | v0;
        q_pairs_stack[j] = _mm256_set1_epi32((int32_t)p);
    }

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
            __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + 0 * 8));
            __m256i df0 = _mm256_max_epi16(
                _mm256_subs_epi16(q_pairs_stack[0], c0), v_min_clamp
            );
            acc0 = _mm256_dpwssd_epi32(acc0, df0, df0);

            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + 1 * 8));
            __m256i df1 = _mm256_max_epi16(
                _mm256_subs_epi16(q_pairs_stack[1], c1), v_min_clamp
            );
            acc1 = _mm256_dpwssd_epi32(acc1, df1, df1);

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
        } // if (num_pairs >= 2)

        /* Pairs 2 and 3 */
        if (num_pairs >= 4)
        {
            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + 2 * 8));
            __m256i df2 = _mm256_max_epi16(
                _mm256_subs_epi16(q_pairs_stack[2], c2), v_min_clamp
            );
            acc2 = _mm256_dpwssd_epi32(acc2, df2, df2);

            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + 3 * 8));
            __m256i df3 = _mm256_max_epi16(
                _mm256_subs_epi16(q_pairs_stack[3], c3), v_min_clamp
            );
            acc3 = _mm256_dpwssd_epi32(acc3, df3, df3);

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
        } // if (num_pairs >= 4)

        int j = 4;
        int unroll_limit = (num_full_pairs < pre_broadcast_count)
                           ? num_full_pairs
                           : pre_broadcast_count;
        for (; j + 4 <= unroll_limit; j += 4)
        {
            __m256i qp0 = q_pairs_stack[j + 0];
            __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 0) * 8));
            __m256i df0 = _mm256_max_epi16(
                _mm256_subs_epi16(qp0, c0), v_min_clamp
            );
            acc0 = _mm256_dpwssd_epi32(acc0, df0, df0);

            __m256i qp1 = q_pairs_stack[j + 1];
            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 1) * 8));
            __m256i df1 = _mm256_max_epi16(
                _mm256_subs_epi16(qp1, c1), v_min_clamp
            );
            acc1 = _mm256_dpwssd_epi32(acc1, df1, df1);

            __m256i qp2 = q_pairs_stack[j + 2];
            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 2) * 8));
            __m256i df2 = _mm256_max_epi16(
                _mm256_subs_epi16(qp2, c2), v_min_clamp
            );
            acc2 = _mm256_dpwssd_epi32(acc2, df2, df2);

            __m256i qp3 = q_pairs_stack[j + 3];
            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 3) * 8));
            __m256i df3 = _mm256_max_epi16(
                _mm256_subs_epi16(qp3, c3), v_min_clamp
            );
            acc3 = _mm256_dpwssd_epi32(acc3, df3, df3);

            /* Checkpoints at Dim 16 (j=4), Dim 24 (j=8), Dim 48 (j=20),
             * and every 32 dims ((j & 15) == 12) */
            if ((j == 4 || j == 8 || j == 20 || (j & 15) == 12) && j + 4 < num_pairs)
            {
                __m256i a_sum = _mm256_add_epi32(
                    _mm256_add_epi32(acc0, acc1),
                    _mm256_add_epi32(acc2, acc3)
                );
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
        } // for (; j + 4 <= unroll_limit; j += 4)

        if (!pruned)
        {
#ifdef EQ16_PROFILE_CHECKPOINTS
            g_eq16_exit_none++;
#endif
            for (; j < num_pairs; j++)
            {
                __m256i q_p;
                if (j < pre_broadcast_count)
                {
                    q_p = q_pairs_stack[j];
                }
                else
                {
                    uint32_t v0 = (uint32_t)(uint16_t)cur_eq16[2 * j];
                    uint32_t v1 = (2 * j + 1 < dim)
                                  ? (uint32_t)(uint16_t)cur_eq16[2 * j + 1]
                                  : 0;
                    q_p = _mm256_set1_epi32((int32_t)((v1 << 16) | v0));
                }

                __m256i c = _mm256_load_si256((const __m256i *)(blk_ptr + j * 8));
                __m256i df = _mm256_max_epi16(
                    _mm256_subs_epi16(q_p, c), v_min_clamp
                );
                acc0 = _mm256_dpwssd_epi32(acc0, df, df);
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
    } // for (int i = rem_start; i < num_clusters; i++)

    #undef EQ16_PREBROADCAST_PAIRS_VNNI
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
#if !defined(__CUDACC__)
    if (anchor_interleaved != NULL &&
        gric_get_simd_level() >= GRIC_SIMD_AVX2 &&
        gric_has_avx_vnni() &&
        dim >= 16)
    {
        eq16_filter_anchor_matrix_avx_vnni(cur_eq16,
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

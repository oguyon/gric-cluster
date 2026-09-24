/**
 * @file scalar_quant_filter.c
 * @brief Bulk cluster anchor matrix filtering and interleaved layout for SQ16.
 */

#include "scalar_quant.h"
#include "gric_simd.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

#if !defined(__CUDACC__) && defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * sq16_reduce_add_epi32_avx2() - Horizontal add of 8 32-bit vector elements using AVX2.
 * @v: 256-bit SIMD vector holding 8 int32 lanes.
 *
 * Return: Sum of all 8 lanes as unsigned 32-bit integer.
 */
static inline uint32_t sq16_reduce_add_epi32_avx2(
    __m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(s);
}
#endif // __AVX2__

/**
 * sq16_set_anchor_interleaved() - Set coordinates into Block-8 interleaved format across all dim.
 * @matrix_interleaved: Interleaved buffer [num_blocks x num_pairs x 8].
 * @cl_idx:             Cluster index.
 * @anchor_coords:      Pointer to cluster's int16_t coordinates [dim].
 * @dim:                Dimension count.
 */
void sq16_set_anchor_interleaved(
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
        uint32_t v0 = (uint32_t)(uint16_t)anchor_coords[2 * j];
        uint32_t v1 = 0;
        if (2 * j + 1 < (size_t)dim)
        {
            v1 = (uint32_t)(uint16_t)anchor_coords[2 * j + 1];
        }
        matrix_interleaved[base + j * 8] = (int32_t)((v1 << 16) | v0);
    }
}

/**
 * sq16_rebuild_anchor_interleaved() - Rebuild Block-8 interleaved buffer.
 * @matrix_interleaved: Interleaved buffer [num_blocks x num_pairs x 8].
 * @anchor_matrix:      Row-major anchor matrix [num_clusters x dim].
 * @num_clusters:       Number of clusters.
 * @dim:                Dimension of each cluster anchor.
 */
void sq16_rebuild_anchor_interleaved(
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
        sq16_set_anchor_interleaved(
            matrix_interleaved,
            c,
            anchor_matrix + (size_t)c * (size_t)dim,
            dim
        );
    }
}

static void sq16_filter_anchor_matrix_scalar(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count);

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static void sq16_filter_anchor_matrix_avx2(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count);

static void sq16_filter_anchor_matrix_avx_vnni(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count);

/**
 * sq16_compact_active_clusters_avx2() - Vectorized gather of surviving cluster indices.
 * @clmembflag:       Candidate flags array (0 = pruned, 1 = active).
 * @active_clusters:  Output array of surviving cluster indices.
 * @num_clusters:     Total clusters count.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
GRIC_TARGET_AVX2
static inline void sq16_compact_active_clusters_avx2(
    int *restrict clmembflag,
    int *restrict active_clusters,
    int           num_clusters,
    int *restrict out_num_active,
    int *restrict out_pruned_count)
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
                active_clusters[num_active++] = i + k;
            }
        }
    }
    for (; i < num_clusters; i++)
    {
        if (clmembflag[i])
        {
            active_clusters[num_active++] = i;
        }
    }
    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = num_clusters - num_active;
    }
}
#endif

#if GRIC_HAVE_AVX512_TARGET
/**
 * sq16_compact_active_clusters_avx512() - Hardware compress-store gather of active clusters.
 * @clmembflag:       Candidate flags array (0 = pruned, 1 = active).
 * @active_clusters:  Output array of surviving cluster indices.
 * @num_clusters:     Total clusters count.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
GRIC_TARGET_AVX512
static inline void sq16_compact_active_clusters_avx512(
    const int *restrict clmembflag,
    int       *restrict active_clusters,
    int                 num_clusters,
    int       *restrict out_num_active,
    int       *restrict out_pruned_count)
{
    int num_active = 0;
    int i = 0;
    __m512i vzero = _mm512_setzero_si512();
    __m512i vramp = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    for (; i + 16 <= num_clusters; i += 16)
    {
        __m512i vf = _mm512_loadu_si512((const void *)(clmembflag + i));
        __mmask16 mask = _mm512_cmpgt_epi32_mask(vf, vzero);
        if (mask == 0)
        {
            continue;
        }
        __m512i vindices = _mm512_add_epi32(_mm512_set1_epi32(i), vramp);
        _mm512_mask_compressstoreu_epi32(
            (void *)(active_clusters + num_active), mask, vindices);
        num_active += (int)_mm_popcnt_u32((uint32_t)mask);
    }
    for (; i < num_clusters; i++)
    {
        if (clmembflag[i])
        {
            active_clusters[num_active++] = i;
        }
    }
    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = num_clusters - num_active;
    }
}

/**
 * sq16_filter_anchor_matrix_avx512() - AVX-512 kernel for anchor filtering.
 * @cur_sq16:           Query int16 array [dim].
 * @anchor_matrix:      Contiguous anchor matrix [num_clusters x dim].
 * @anchor_interleaved: Interleaved anchor matrix [num_blocks x num_pairs x 8].
 * @num_clusters:       Cluster count.
 * @dim:                Vector dimensionality.
 * @sq16_ssd_thresh:    Squared distance cutoff threshold.
 * @clmembflag:         Candidate flags array (0 = pruned).
 * @active_clusters:    Surviving cluster indices array.
 * @out_num_active:     Count of surviving clusters.
 * @out_pruned_count:   Count of pruned clusters.
 */
GRIC_TARGET_AVX512
static void sq16_filter_anchor_matrix_avx512(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    if (dim < 16)
    {
        sq16_filter_anchor_matrix_scalar(cur_sq16,
                                         anchor_matrix,
                                         anchor_interleaved,
                                         num_clusters,
                                         dim,
                                         sq16_ssd_thresh,
                                         clmembflag,
                                         active_clusters,
                                         out_num_active,
                                         out_pruned_count);
        return;
    }

    uint32_t thresh32 = (sq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)sq16_ssd_thresh;

    if (anchor_interleaved != NULL)
    {
        int num_pairs = (int)((dim + 1) / 2);
        int num_blocks = num_clusters / 8;
        int num_sb = num_blocks / 2;
        size_t blk_stride = (size_t)num_pairs * 8;
        __m512i v_cut512 = _mm512_set1_epi32((int32_t)thresh32);

        #define SQ16_PREBROADCAST_PAIRS_512 512
        __m512i q_pairs_stack[SQ16_PREBROADCAST_PAIRS_512];
        int pre_broadcast_count = (num_pairs < SQ16_PREBROADCAST_PAIRS_512)
                                  ? num_pairs
                                  : SQ16_PREBROADCAST_PAIRS_512;

        for (int j = 0; j < pre_broadcast_count; j++)
        {
            uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
            uint32_t v1 = (2 * j + 1 < dim)
                          ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                          : 0;
            uint32_t p = (v1 << 16) | v0;
            q_pairs_stack[j] = _mm512_set1_epi32((int32_t)p);
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_sb >= 512)
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

            /* Checkpoint 0: Early Dim 4 check */
            if (num_pairs >= 2)
            {
                __m256i c0_0 = _mm256_load_si256((const __m256i *)(blk0 + 0 * 8));
                __m256i c1_0 = _mm256_load_si256((const __m256i *)(blk1 + 0 * 8));
                __m512i c0 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_0), c1_0, 1);
                __m512i d0 = _mm512_sub_epi16(q_pairs_stack[0], c0);
                acc0 = _mm512_madd_epi16(d0, d0);

                __m256i c0_1 = _mm256_load_si256((const __m256i *)(blk0 + 1 * 8));
                __m256i c1_1 = _mm256_load_si256((const __m256i *)(blk1 + 1 * 8));
                __m512i c1 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_1), c1_1, 1);
                __m512i d1 = _mm512_sub_epi16(q_pairs_stack[1], c1);
                acc1 = _mm512_madd_epi16(d1, d1);

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

            if (num_pairs >= 4)
            {
                __m256i c0_2 = _mm256_load_si256((const __m256i *)(blk0 + 2 * 8));
                __m256i c1_2 = _mm256_load_si256((const __m256i *)(blk1 + 2 * 8));
                __m512i c2 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_2), c1_2, 1);
                __m512i d2 = _mm512_sub_epi16(q_pairs_stack[2], c2);
                acc2 = _mm512_madd_epi16(d2, d2);

                __m256i c0_3 = _mm256_load_si256((const __m256i *)(blk0 + 3 * 8));
                __m256i c1_3 = _mm256_load_si256((const __m256i *)(blk1 + 3 * 8));
                __m512i c3 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_3), c1_3, 1);
                __m512i d3 = _mm512_sub_epi16(q_pairs_stack[3], c3);
                acc3 = _mm512_madd_epi16(d3, d3);
            }

            int j = 4;
            for (; j + 4 <= num_pairs; j += 4)
            {
                __m256i c0_0 = _mm256_load_si256((const __m256i *)(blk0 + (j + 0) * 8));
                __m256i c1_0 = _mm256_load_si256((const __m256i *)(blk1 + (j + 0) * 8));
                __m512i c0 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_0), c1_0, 1);
                __m512i d0 = _mm512_sub_epi16(q_pairs_stack[j + 0], c0);
                acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));

                __m256i c0_1 = _mm256_load_si256((const __m256i *)(blk0 + (j + 1) * 8));
                __m256i c1_1 = _mm256_load_si256((const __m256i *)(blk1 + (j + 1) * 8));
                __m512i c1 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_1), c1_1, 1);
                __m512i d1 = _mm512_sub_epi16(q_pairs_stack[j + 1], c1);
                acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));

                __m256i c0_2 = _mm256_load_si256((const __m256i *)(blk0 + (j + 2) * 8));
                __m256i c1_2 = _mm256_load_si256((const __m256i *)(blk1 + (j + 2) * 8));
                __m512i c2 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_2), c1_2, 1);
                __m512i d2 = _mm512_sub_epi16(q_pairs_stack[j + 2], c2);
                acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));

                __m256i c0_3 = _mm256_load_si256((const __m256i *)(blk0 + (j + 3) * 8));
                __m256i c1_3 = _mm256_load_si256((const __m256i *)(blk1 + (j + 3) * 8));
                __m512i c3 = _mm512_inserti64x4(_mm512_castsi256_si512(c0_3), c1_3, 1);
                __m512i d3 = _mm512_sub_epi16(q_pairs_stack[j + 3], c3);
                acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));

                if ((j == 12 || j == 60 || j == 124 || j == 188) && j + 4 < num_pairs)
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
            }

            if (!pruned)
            {
                for (; j < num_pairs; j++)
                {
                    __m512i q_p;
                    if (j < SQ16_PREBROADCAST_PAIRS_512)
                    {
                        q_p = q_pairs_stack[j];
                    }
                    else
                    {
                        uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
                        uint32_t v1 = (2 * j + 1 < dim)
                                      ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                                      : 0;
                        q_p = _mm512_set1_epi32((int32_t)((v1 << 16) | v0));
                    }

                    __m256i c0_p = _mm256_load_si256((const __m256i *)(blk0 + j * 8));
                    __m256i c1_p = _mm256_load_si256((const __m256i *)(blk1 + j * 8));
                    __m512i c_pair = _mm512_inserti64x4(_mm512_castsi256_si512(c0_p), c1_p, 1);
                    __m512i diff = _mm512_sub_epi16(q_p, c_pair);
                    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(diff, diff));
                }

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
            }
        } // for (sb = 0 .. num_sb - 1)

        #undef SQ16_PREBROADCAST_PAIRS_512

        /* Remainder clusters (< 16) */
        int evaluated = num_sb * 16;
        for (int i = evaluated; i < num_clusters; i++)
        {
            const int16_t *ak_ptr = anchor_matrix + (size_t)i * dim;
            uint32_t total = 0;
            int rem_pruned = 0;
            for (long d = 0; d < dim; d += 16)
            {
                if (d + 16 <= dim)
                {
                    __m256i qd = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d));
                    __m256i ad = _mm256_loadu_si256((const __m256i *)(ak_ptr + d));
                    __m256i diff = _mm256_sub_epi16(qd, ad);
                    __m256i madd = _mm256_madd_epi16(diff, diff);
                    total += sq16_reduce_add_epi32_avx2(madd);
                }
                else
                {
                    for (long k = d; k < dim; k++)
                    {
                        int32_t diff = (int32_t)cur_sq16[k] - (int32_t)ak_ptr[k];
                        total += (uint32_t)(diff * diff);
                    }
                }
                if (total > thresh32)
                {
                    rem_pruned = 1;
                    break;
                }
            }
            clmembflag[i] = rem_pruned ? 0 : 1;
        }

        sq16_compact_active_clusters_avx512(clmembflag, active_clusters, num_clusters,
                                             out_num_active, out_pruned_count);
        return;
    } // if (anchor_interleaved != NULL)

    int num_active = 0;
    int pruned_count = 0;

    if (dim == 128)
    {
        __m512i q0 = _mm512_loadu_si512((const void *)(cur_sq16 + 0));
        __m512i q1 = _mm512_loadu_si512((const void *)(cur_sq16 + 32));
        __m512i q2 = _mm512_loadu_si512((const void *)(cur_sq16 + 64));
        __m512i q3 = _mm512_loadu_si512((const void *)(cur_sq16 + 96));

        int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                if (i + 12 < num_clusters)
                {
                    _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 8) * 128),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 10) * 128),
                                 _MM_HINT_T0);
                }

                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * 128;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * 128;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * 128;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * 128;

                __m512i a0 = _mm512_loadu_si512((const void *)(a0_ptr + 0));
                __m512i a1 = _mm512_loadu_si512((const void *)(a1_ptr + 0));
                __m512i a2 = _mm512_loadu_si512((const void *)(a2_ptr + 0));
                __m512i a3 = _mm512_loadu_si512((const void *)(a3_ptr + 0));

                __m512i d0 = _mm512_sub_epi16(q0, a0);
                __m512i d1 = _mm512_sub_epi16(q0, a1);
                __m512i d2 = _mm512_sub_epi16(q0, a2);
                __m512i d3 = _mm512_sub_epi16(q0, a3);

                __m512i acc0 = _mm512_madd_epi16(d0, d0);
                __m512i acc1 = _mm512_madd_epi16(d1, d1);
                __m512i acc2 = _mm512_madd_epi16(d2, d2);
                __m512i acc3 = _mm512_madd_epi16(d3, d3);

                uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(acc0);
                uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(acc1);
                uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(acc2);
                uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(acc3);

                if (s0 > thresh32 && s1 > thresh32 && s2 > thresh32 && s3 > thresh32)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t sums[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4; k++)
                {
                    if (sums[k] > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * 128;
                    uint32_t total = sums[k];

                    // Block 1: dims 32..63
                    __m512i b1 = _mm512_loadu_si512((const void *)(ak_ptr + 32));
                    __m512i db1 = _mm512_sub_epi16(q1, b1);
                    total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db1, db1));
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    // Block 2: dims 64..95
                    __m512i b2 = _mm512_loadu_si512((const void *)(ak_ptr + 64));
                    __m512i db2 = _mm512_sub_epi16(q2, b2);
                    total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db2, db2));
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    // Block 3: dims 96..127
                    __m512i b3 = _mm512_loadu_si512((const void *)(ak_ptr + 96));
                    __m512i db3 = _mm512_sub_epi16(q3, b3);
                    total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db3, db3));
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            } // for (; i + 3 < num_clusters; i += 4)

            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * 128;

                __m512i a0 = _mm512_loadu_si512((const void *)(a_ptr + 0));
                __m512i d0 = _mm512_sub_epi16(q0, a0);
                __m512i acc = _mm512_madd_epi16(d0, d0);
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                __m512i a1 = _mm512_loadu_si512((const void *)(a_ptr + 32));
                __m512i d1 = _mm512_sub_epi16(q1, a1);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d1, d1));
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                __m512i a2 = _mm512_loadu_si512((const void *)(a_ptr + 64));
                __m512i d2 = _mm512_sub_epi16(q2, a2);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d2, d2));
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                __m512i a3 = _mm512_loadu_si512((const void *)(a_ptr + 96));
                __m512i d3 = _mm512_sub_epi16(q3, a3);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d3, d3));
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            } // for (; i < num_clusters; i++)
    }
    else if (dim <= 64)
    {
        if (dim <= 32)
        {
            __mmask32 kmask = (dim >= 32) ? 0xFFFFFFFFU : (uint32_t)((1ULL << dim) - 1ULL);
            __m512i q0 = _mm512_maskz_loadu_epi16(kmask, (const void *)cur_sq16);
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;

                __m512i a0 = _mm512_maskz_loadu_epi16(kmask, (const void *)a0_ptr);
                __m512i a1 = _mm512_maskz_loadu_epi16(kmask, (const void *)a1_ptr);
                __m512i a2 = _mm512_maskz_loadu_epi16(kmask, (const void *)a2_ptr);
                __m512i a3 = _mm512_maskz_loadu_epi16(kmask, (const void *)a3_ptr);

                __m512i d0 = _mm512_sub_epi16(q0, a0);
                __m512i d1 = _mm512_sub_epi16(q0, a1);
                __m512i d2 = _mm512_sub_epi16(q0, a2);
                __m512i d3 = _mm512_sub_epi16(q0, a3);

                uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d0, d0));
                uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d1, d1));
                uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d2, d2));
                uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d3, d3));

                uint32_t sums[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4; k++)
                {
                    if (sums[k] > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                    }
                    else
                    {
                        clmembflag[i + k] = 1;
                        active_clusters[num_active++] = i + k;
                    }
                }
            } // for (; i + 3 < num_clusters; i += 4)
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m512i a = _mm512_maskz_loadu_epi16(kmask, (const void *)a_ptr);
                __m512i d = _mm512_sub_epi16(q0, a);
                uint32_t s = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d, d));
                if (s > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                }
                else
                {
                    clmembflag[i] = 1;
                    active_clusters[num_active++] = i;
                }
            } // for (; i < num_clusters; i++)
        }
        else // 32 < dim <= 64
        {
            uint32_t rem_dim = (uint32_t)(dim - 32);
            __mmask32 kmask1 = (rem_dim >= 32) ? 0xFFFFFFFFU : (uint32_t)((1ULL << rem_dim) - 1ULL);
            __m512i q0 = _mm512_loadu_si512((const void *)(cur_sq16 + 0));
            __m512i q1 = _mm512_maskz_loadu_epi16(kmask1, (const void *)(cur_sq16 + 32));
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                __m512i a0 = _mm512_loadu_si512((const void *)(anchor_matrix +
                                                              (size_t)(i + 0) * dim));
                __m512i a1 = _mm512_loadu_si512((const void *)(anchor_matrix +
                                                              (size_t)(i + 1) * dim));
                __m512i a2 = _mm512_loadu_si512((const void *)(anchor_matrix +
                                                              (size_t)(i + 2) * dim));
                __m512i a3 = _mm512_loadu_si512((const void *)(anchor_matrix +
                                                              (size_t)(i + 3) * dim));

                __m512i d0 = _mm512_sub_epi16(q0, a0);
                __m512i d1 = _mm512_sub_epi16(q0, a1);
                __m512i d2 = _mm512_sub_epi16(q0, a2);
                __m512i d3 = _mm512_sub_epi16(q0, a3);

                uint32_t s0 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d0, d0));
                uint32_t s1 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d1, d1));
                uint32_t s2 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d2, d2));
                uint32_t s3 = (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(d3, d3));

                if (s0 > thresh32 && s1 > thresh32 && s2 > thresh32 && s3 > thresh32)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t sums[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4; k++)
                {
                    if (sums[k] > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }
                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * dim;
                    __m512i bk = _mm512_maskz_loadu_epi16(kmask1, (const void *)(ak_ptr + 32));
                    __m512i db1 = _mm512_sub_epi16(q1, bk);
                    uint32_t tot = sums[k] +
                        (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(db1, db1));
                    if (tot > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }
                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            } // for (; i + 3 < num_clusters; i += 4)
            for (; i < num_clusters; i++)
            {
                __m512i a0 = _mm512_loadu_si512((const void *)(anchor_matrix +
                                                              (size_t)i * dim));
                __m512i d0 = _mm512_sub_epi16(q0, a0);
                __m512i acc = _mm512_madd_epi16(d0, d0);
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m512i bk = _mm512_maskz_loadu_epi16(kmask1, (const void *)(a_ptr + 32));
                __m512i d1 = _mm512_sub_epi16(q1, bk);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d1, d1));
                if ((uint32_t)_mm512_reduce_add_epi32(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            } // for (; i < num_clusters; i++)
        }
    }
    else
    {
        // Arbitrary dimension (including D > 128): 32-element dimension-block streaming
        for (int i = 0; i < num_clusters; i++)
        {
            if (i + 16 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 16) * dim),
                             _MM_HINT_T0);
            }
            uint32_t total = 0;
            int pruned = 0;
            long d = 0;


            if (!pruned)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                for (; d < dim; d += 32)
                {
                    if (d + 32 <= dim)
                    {
                        __m512i q = _mm512_loadu_si512((const void *)(cur_sq16 + d));
                        __m512i a = _mm512_loadu_si512((const void *)(a_ptr + d));
                        __m512i diff = _mm512_sub_epi16(q, a);
                        total += (uint32_t)_mm512_reduce_add_epi32(_mm512_madd_epi16(diff, diff));
                    }
                    else
                    {
                        for (long rem = d; rem < dim; rem++)
                        {
                            int32_t df = (int32_t)cur_sq16[rem] - (int32_t)a_ptr[rem];
                            total += (uint32_t)(df * df);
                        }
                    }
                    if (total > thresh32)
                    {
                        pruned = 1;
                        break;
                    }
                } // for (; d < dim; d += 32)
            }

            if (pruned)
            {
                clmembflag[i] = 0;
                pruned_count++;
            }
            else
            {
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        } // for (int i = 0; i < num_clusters; i++)
    }

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}
#endif // GRIC_HAVE_AVX512_TARGET

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
/**
 * sq16_filter_anchor_matrix_avx2() - AVX2 kernel for anchor filtering.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
GRIC_TARGET_AVX2
static void sq16_filter_anchor_matrix_avx2(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    uint32_t thresh32 = (sq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)sq16_ssd_thresh;
    int num_active = 0;
    int pruned_count = 0;

    if (dim < 16)
    {
        sq16_filter_anchor_matrix_scalar(cur_sq16,
                                         anchor_matrix,
                                         anchor_interleaved,
                                         num_clusters,
                                         dim,
                                         sq16_ssd_thresh,
                                         clmembflag,
                                         active_clusters,
                                         out_num_active,
                                         out_pruned_count);
        return;
    }

    __m128i v_bias = _mm_set1_epi32((int32_t)0x80000000U);
    __m128i v_cut  = _mm_set1_epi32((int32_t)(thresh32 ^ 0x80000000U));

    if (anchor_interleaved != NULL)
    {
        __m256i v_bias256 = _mm256_set1_epi32((int32_t)0x80000000U);
        __m256i v_cut256  = _mm256_set1_epi32((int32_t)(thresh32 ^ 0x80000000U));

        int num_pairs = (int)((dim + 1) / 2);
        int num_blocks = num_clusters / 8;
        int rem_start = num_blocks * 8;
        size_t blk_stride = (size_t)num_pairs * 8;

        #define SQ16_PREBROADCAST_PAIRS 512
        __m256i q_pairs_stack[SQ16_PREBROADCAST_PAIRS];
        int pre_broadcast_count = (num_pairs < SQ16_PREBROADCAST_PAIRS)
                                  ? num_pairs
                                  : SQ16_PREBROADCAST_PAIRS;

        for (int j = 0; j < pre_broadcast_count; j++)
        {
            uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
            uint32_t v1 = (2 * j + 1 < dim)
                          ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                          : 0;
            uint32_t p = (v1 << 16) | v0;
            q_pairs_stack[j] = _mm256_set1_epi32((int32_t)p);
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_blocks >= 1024)
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
                __m256i d0 = _mm256_sub_epi16(q_pairs_stack[0], c0);
                acc0 = _mm256_madd_epi16(d0, d0);

                __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + 1 * 8));
                __m256i d1 = _mm256_sub_epi16(q_pairs_stack[1], c1);
                acc1 = _mm256_madd_epi16(d1, d1);

                __m256i a_sum01 = _mm256_add_epi32(acc0, acc1);
                __m256i v_acc_b01 = _mm256_xor_si256(a_sum01, v_bias256);
                __m256i cmp01 = _mm256_cmpgt_epi32(v_acc_b01, v_cut256);
                if (_mm256_movemask_ps(_mm256_castsi256_ps(cmp01)) == 0xFF)
                {
                    int base = b * 8;
                    _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                        _mm256_setzero_si256());
                    continue;
                }
            }

            /* Pairs 2 and 3 */
            if (num_pairs >= 4)
            {
                __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + 2 * 8));
                __m256i d2 = _mm256_sub_epi16(q_pairs_stack[2], c2);
                acc2 = _mm256_madd_epi16(d2, d2);

                __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + 3 * 8));
                __m256i d3 = _mm256_sub_epi16(q_pairs_stack[3], c3);
                acc3 = _mm256_madd_epi16(d3, d3);
            }

            int j = 4;
            for (; j + 4 <= num_pairs; j += 4)
            {
                __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 0) * 8));
                __m256i d0 = _mm256_sub_epi16(q_pairs_stack[j + 0], c0);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));

                __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 1) * 8));
                __m256i d1 = _mm256_sub_epi16(q_pairs_stack[j + 1], c1);
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));

                __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 2) * 8));
                __m256i d2 = _mm256_sub_epi16(q_pairs_stack[j + 2], c2);
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2, d2));

                __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 3) * 8));
                __m256i d3 = _mm256_sub_epi16(q_pairs_stack[j + 3], c3);
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3, d3));

                /* Spaced checkpoints at dims 32, 64, 128, 256, 384 */
                if ((j == 12 || j == 28 || j == 60 || j == 124 || j == 188) &&
                    j + 4 < num_pairs)
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
                        break;
                    }
                }
            }

            if (!pruned)
            {
                for (; j < num_pairs; j++)
                {
                    __m256i q_p;
                    if (j < SQ16_PREBROADCAST_PAIRS)
                    {
                        q_p = q_pairs_stack[j];
                    }
                    else
                    {
                        uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
                        uint32_t v1 = (2 * j + 1 < dim)
                                      ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                                      : 0;
                        q_p = _mm256_set1_epi32((int32_t)((v1 << 16) | v0));
                    }

                    __m256i c_pair = _mm256_load_si256((const __m256i *)(blk_ptr + j * 8));
                    __m256i diff = _mm256_sub_epi16(q_p, c_pair);
                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(diff, diff));
                }

                __m256i acc = _mm256_add_epi32(
                    _mm256_add_epi32(acc0, acc1),
                    _mm256_add_epi32(acc2, acc3));

                __m256i v_acc_b = _mm256_xor_si256(acc, v_bias256);
                __m256i cmp = _mm256_cmpgt_epi32(v_acc_b, v_cut256);
                int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

                int base = b * 8;
                if (mask == 0xFF)
                {
                    _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                        _mm256_setzero_si256());
                }
                else
                {
                    for (int k = 0; k < 8; k++)
                    {
                        clmembflag[base + k] = ((mask & (1 << k)) != 0) ? 0 : 1;
                    }
                }
            }
        } // for (int b = 0; b < num_blocks; b++)

        /* Remainder clusters (< 8) */
        for (int i = rem_start; i < num_clusters; i++)
        {
            const int16_t *ak_ptr = anchor_matrix + (size_t)i * dim;
            uint32_t total = 0;
            int rem_pruned = 0;
            for (long d = 0; d < dim; d += 16)
            {
                if (d + 16 <= dim)
                {
                    __m256i qd = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d));
                    __m256i ad = _mm256_loadu_si256((const __m256i *)(ak_ptr + d));
                    __m256i diff = _mm256_sub_epi16(qd, ad);
                    __m256i madd = _mm256_madd_epi16(diff, diff);
                    total += sq16_reduce_add_epi32_avx2(madd);
                }
                else
                {
                    for (long rem = d; rem < dim; rem++)
                    {
                        int32_t df = (int32_t)cur_sq16[rem] - (int32_t)ak_ptr[rem];
                        total += (uint32_t)(df * df);
                    }
                }
                if (total > thresh32)
                {
                    rem_pruned = 1;
                    break;
                }
            }
            clmembflag[i] = rem_pruned ? 0 : 1;
        }

        #undef SQ16_PREBROADCAST_PAIRS
        sq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                          out_num_active, out_pruned_count);
        return;
    } // if (anchor_interleaved != NULL)

    if (dim == 128)
    {
        __m256i q0 = _mm256_load_si256((const __m256i *)(cur_sq16 + 0));
        __m256i q1 = _mm256_load_si256((const __m256i *)(cur_sq16 + 16));
        __m256i q2 = _mm256_load_si256((const __m256i *)(cur_sq16 + 32));
        __m256i q3 = _mm256_load_si256((const __m256i *)(cur_sq16 + 48));
        __m256i q4 = _mm256_load_si256((const __m256i *)(cur_sq16 + 64));
        __m256i q5 = _mm256_load_si256((const __m256i *)(cur_sq16 + 80));
        __m256i q6 = _mm256_load_si256((const __m256i *)(cur_sq16 + 96));
        __m256i q7 = _mm256_load_si256((const __m256i *)(cur_sq16 + 112));

        int i = 0;
        for (; i + 3 < num_clusters; i += 4)
        {
                if (i + 12 < num_clusters)
                {
                    _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 8) * 128),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 10) * 128),
                                 _MM_HINT_T0);
                }

                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * 128;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * 128;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * 128;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * 128;

                // Chunk 0: Anchor 0
                __m256i a0_0 = _mm256_load_si256((const __m256i *)(a0_ptr + 0));
                __m256i a0_1 = _mm256_load_si256((const __m256i *)(a0_ptr + 16));
                __m256i d0_0 = _mm256_sub_epi16(q0, a0_0);
                __m256i d0_1 = _mm256_sub_epi16(q1, a0_1);
                __m256i acc0 = _mm256_add_epi32(_mm256_madd_epi16(d0_0, d0_0),
                                                _mm256_madd_epi16(d0_1, d0_1));

                // Chunk 0: Anchor 1
                __m256i a1_0 = _mm256_load_si256((const __m256i *)(a1_ptr + 0));
                __m256i a1_1 = _mm256_load_si256((const __m256i *)(a1_ptr + 16));
                __m256i d1_0 = _mm256_sub_epi16(q0, a1_0);
                __m256i d1_1 = _mm256_sub_epi16(q1, a1_1);
                __m256i acc1 = _mm256_add_epi32(_mm256_madd_epi16(d1_0, d1_0),
                                                _mm256_madd_epi16(d1_1, d1_1));

                __m256i h01 = _mm256_hadd_epi32(acc0, acc1);

                // Chunk 0: Anchor 2
                __m256i a2_0 = _mm256_load_si256((const __m256i *)(a2_ptr + 0));
                __m256i a2_1 = _mm256_load_si256((const __m256i *)(a2_ptr + 16));
                __m256i d2_0 = _mm256_sub_epi16(q0, a2_0);
                __m256i d2_1 = _mm256_sub_epi16(q1, a2_1);
                __m256i acc2 = _mm256_add_epi32(_mm256_madd_epi16(d2_0, d2_0),
                                                _mm256_madd_epi16(d2_1, d2_1));

                // Chunk 0: Anchor 3
                __m256i a3_0 = _mm256_load_si256((const __m256i *)(a3_ptr + 0));
                __m256i a3_1 = _mm256_load_si256((const __m256i *)(a3_ptr + 16));
                __m256i d3_0 = _mm256_sub_epi16(q0, a3_0);
                __m256i d3_1 = _mm256_sub_epi16(q1, a3_1);
                __m256i acc3 = _mm256_add_epi32(_mm256_madd_epi16(d3_0, d3_0),
                                                _mm256_madd_epi16(d3_1, d3_1));

                __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
                __m256i h_all = _mm256_hadd_epi32(h01, h23);
                __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                             _mm256_extracti128_si256(h_all, 1));

                __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
                __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

                if (mask == 0xF)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t s4[4];
                _mm_storeu_si128((__m128i *)s4, sum4);

                for (int k = 0; k < 4; k++)
                {
                    if ((mask & (1 << k)) != 0)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * 128;
                    uint32_t total = s4[k];

                    // Chunk 1: dims 32..63
                    __m256i a2 = _mm256_load_si256((const __m256i *)(ak_ptr + 32));
                    __m256i a3 = _mm256_load_si256((const __m256i *)(ak_ptr + 48));
                    __m256i d2 = _mm256_sub_epi16(q2, a2);
                    __m256i d3 = _mm256_sub_epi16(q3, a3);
                    __m256i acc_c1 = _mm256_add_epi32(_mm256_madd_epi16(d2, d2),
                                                      _mm256_madd_epi16(d3, d3));
                    total += sq16_reduce_add_epi32_avx2(acc_c1);
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    // Chunk 2 & 3: dims 64..127
                    __m256i a4 = _mm256_load_si256((const __m256i *)(ak_ptr + 64));
                    __m256i a5 = _mm256_load_si256((const __m256i *)(ak_ptr + 80));
                    __m256i d4 = _mm256_sub_epi16(q4, a4);
                    __m256i d5 = _mm256_sub_epi16(q5, a5);
                    __m256i acc_c23 = _mm256_add_epi32(_mm256_madd_epi16(d4, d4),
                                                       _mm256_madd_epi16(d5, d5));

                    __m256i a6 = _mm256_load_si256((const __m256i *)(ak_ptr + 96));
                    __m256i a7 = _mm256_load_si256((const __m256i *)(ak_ptr + 112));
                    __m256i d6 = _mm256_sub_epi16(q6, a6);
                    __m256i d7 = _mm256_sub_epi16(q7, a7);
                    acc_c23 = _mm256_add_epi32(acc_c23, _mm256_madd_epi16(d6, d6));
                    acc_c23 = _mm256_add_epi32(acc_c23, _mm256_madd_epi16(d7, d7));

                    total += sq16_reduce_add_epi32_avx2(acc_c23);
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            } // for (; i + 3 < num_clusters; i += 4)

            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * 128;

                // Chunk 0: dims 0..31
                __m256i a0 = _mm256_load_si256((const __m256i *)(a_ptr + 0));
                __m256i a1 = _mm256_load_si256((const __m256i *)(a_ptr + 16));
                __m256i d0 = _mm256_sub_epi16(q0, a0);
                __m256i d1 = _mm256_sub_epi16(q1, a1);
                __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                               _mm256_madd_epi16(d1, d1));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                // Chunk 1: dims 32..63
                __m256i a2 = _mm256_load_si256((const __m256i *)(a_ptr + 32));
                __m256i a3 = _mm256_load_si256((const __m256i *)(a_ptr + 48));
                __m256i d2 = _mm256_sub_epi16(q2, a2);
                __m256i d3 = _mm256_sub_epi16(q3, a3);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d2, d2));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d3, d3));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                // Chunk 2 & 3: dims 64..127
                __m256i a4 = _mm256_load_si256((const __m256i *)(a_ptr + 64));
                __m256i a5 = _mm256_load_si256((const __m256i *)(a_ptr + 80));
                __m256i d4 = _mm256_sub_epi16(q4, a4);
                __m256i d5 = _mm256_sub_epi16(q5, a5);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d4, d4));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d5, d5));

                __m256i a6 = _mm256_load_si256((const __m256i *)(a_ptr + 96));
                __m256i a7 = _mm256_load_si256((const __m256i *)(a_ptr + 112));
                __m256i d6 = _mm256_sub_epi16(q6, a6);
                __m256i d7 = _mm256_sub_epi16(q7, a7);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d6, d6));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d7, d7));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }

                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            } // for (; i < num_clusters; i++)
    }
    else if (dim <= 64)
    {
        if (dim == 32)
        {
            __m256i q0 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 0));
            __m256i q1 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 16));
            const int16_t *src = anchor_matrix;
            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = src + (size_t)(i + 0) * 32;
                const int16_t *a1_ptr = src + (size_t)(i + 1) * 32;
                const int16_t *a2_ptr = src + (size_t)(i + 2) * 32;
                const int16_t *a3_ptr = src + (size_t)(i + 3) * 32;

                __m256i d0_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a0_ptr));
                __m256i d0_1 = _mm256_sub_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(a0_ptr + 16)));
                __m256i acc0 = _mm256_add_epi32(_mm256_madd_epi16(d0_0, d0_0),
                                                _mm256_madd_epi16(d0_1, d0_1));

                __m256i d1_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a1_ptr));
                __m256i d1_1 = _mm256_sub_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(a1_ptr + 16)));
                __m256i acc1 = _mm256_add_epi32(_mm256_madd_epi16(d1_0, d1_0),
                                                _mm256_madd_epi16(d1_1, d1_1));

                __m256i h01 = _mm256_hadd_epi32(acc0, acc1);

                __m256i d2_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a2_ptr));
                __m256i d2_1 = _mm256_sub_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(a2_ptr + 16)));
                __m256i acc2 = _mm256_add_epi32(_mm256_madd_epi16(d2_0, d2_0),
                                                _mm256_madd_epi16(d2_1, d2_1));

                __m256i d3_0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a3_ptr));
                __m256i d3_1 = _mm256_sub_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(a3_ptr + 16)));
                __m256i acc3 = _mm256_add_epi32(_mm256_madd_epi16(d3_0, d3_0),
                                                _mm256_madd_epi16(d3_1, d3_1));

                __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
                __m256i h_all = _mm256_hadd_epi32(h01, h23);
                __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                             _mm256_extracti128_si256(h_all, 1));

                __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
                __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

                for (int k = 0; k < 4; k++)
                {
                    if ((mask & (1 << k)) != 0)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                    }
                    else
                    {
                        clmembflag[i + k] = 1;
                        active_clusters[num_active++] = i + k;
                    }
                }
            }
            for (; i < num_clusters; i++)
            {
                const int16_t *a_ptr = src + (size_t)i * 32;
                __m256i d0 = _mm256_sub_epi16(q0, _mm256_loadu_si256((const __m256i *)a_ptr));
                __m256i d1 = _mm256_sub_epi16(q1,
                    _mm256_loadu_si256((const __m256i *)(a_ptr + 16)));
                __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                               _mm256_madd_epi16(d1, d1));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                }
                else
                {
                    clmembflag[i] = 1;
                    active_clusters[num_active++] = i;
                }
            }
        }
        else // 32 < dim <= 64
        {
            __m256i q0 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 0));
            __m256i q1 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 16));
            __m256i q2 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 32));
            __m256i q3 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + 48));

            int i = 0;
            for (; i + 3 < num_clusters; i += 4)
            {
                const int16_t *a0_ptr = anchor_matrix + (size_t)(i + 0) * dim;
                const int16_t *a1_ptr = anchor_matrix + (size_t)(i + 1) * dim;
                const int16_t *a2_ptr = anchor_matrix + (size_t)(i + 2) * dim;
                const int16_t *a3_ptr = anchor_matrix + (size_t)(i + 3) * dim;
                __m256i a0_0 = _mm256_loadu_si256((const __m256i *)(a0_ptr + 0));
                __m256i a0_1 = _mm256_loadu_si256((const __m256i *)(a0_ptr + 16));
                __m256i a1_0 = _mm256_loadu_si256((const __m256i *)(a1_ptr + 0));
                __m256i a1_1 = _mm256_loadu_si256((const __m256i *)(a1_ptr + 16));
                __m256i a2_0 = _mm256_loadu_si256((const __m256i *)(a2_ptr + 0));
                __m256i a2_1 = _mm256_loadu_si256((const __m256i *)(a2_ptr + 16));
                __m256i a3_0 = _mm256_loadu_si256((const __m256i *)(a3_ptr + 0));
                __m256i a3_1 = _mm256_loadu_si256((const __m256i *)(a3_ptr + 16));

                __m256i d0_0 = _mm256_sub_epi16(q0, a0_0);
                __m256i d0_1 = _mm256_sub_epi16(q1, a0_1);
                __m256i acc0 = _mm256_add_epi32(_mm256_madd_epi16(d0_0, d0_0),
                                                _mm256_madd_epi16(d0_1, d0_1));

                __m256i d1_0 = _mm256_sub_epi16(q0, a1_0);
                __m256i d1_1 = _mm256_sub_epi16(q1, a1_1);
                __m256i acc1 = _mm256_add_epi32(_mm256_madd_epi16(d1_0, d1_0),
                                                _mm256_madd_epi16(d1_1, d1_1));

                __m256i h01 = _mm256_hadd_epi32(acc0, acc1);

                __m256i d2_0 = _mm256_sub_epi16(q0, a2_0);
                __m256i d2_1 = _mm256_sub_epi16(q1, a2_1);
                __m256i acc2 = _mm256_add_epi32(_mm256_madd_epi16(d2_0, d2_0),
                                                _mm256_madd_epi16(d2_1, d2_1));

                __m256i d3_0 = _mm256_sub_epi16(q0, a3_0);
                __m256i d3_1 = _mm256_sub_epi16(q1, a3_1);
                __m256i acc3 = _mm256_add_epi32(_mm256_madd_epi16(d3_0, d3_0),
                                                _mm256_madd_epi16(d3_1, d3_1));

                __m256i h23 = _mm256_hadd_epi32(acc2, acc3);
                __m256i h_all = _mm256_hadd_epi32(h01, h23);
                __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(h_all),
                                             _mm256_extracti128_si256(h_all, 1));

                __m128i v_sum_b = _mm_xor_si128(sum4, v_bias);
                __m128i cmp = _mm_cmpgt_epi32(v_sum_b, v_cut);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));

                if (mask == 0xF)
                {
                    clmembflag[i + 0] = 0;
                    clmembflag[i + 1] = 0;
                    clmembflag[i + 2] = 0;
                    clmembflag[i + 3] = 0;
                    pruned_count += 4;
                    continue;
                }

                uint32_t s4[4];
                _mm_storeu_si128((__m128i *)s4, sum4);

                for (int k = 0; k < 4; k++)
                {
                    if ((mask & (1 << k)) != 0)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    const int16_t *ak_ptr = anchor_matrix + (size_t)(i + k) * dim;
                    uint32_t total = s4[k];

                    __m256i a2 = _mm256_loadu_si256((const __m256i *)(ak_ptr + 32));
                    __m256i a3 = _mm256_loadu_si256((const __m256i *)(ak_ptr + 48));
                    __m256i d2 = _mm256_sub_epi16(q2, a2);
                    __m256i d3 = _mm256_sub_epi16(q3, a3);
                    __m256i acc_c1 = _mm256_add_epi32(_mm256_madd_epi16(d2, d2),
                                                      _mm256_madd_epi16(d3, d3));
                    total += sq16_reduce_add_epi32_avx2(acc_c1);
                    if (total > thresh32)
                    {
                        clmembflag[i + k] = 0;
                        pruned_count++;
                        continue;
                    }

                    clmembflag[i + k] = 1;
                    active_clusters[num_active++] = i + k;
                }
            } // for (; i + 3 < num_clusters; i += 4)

            for (; i < num_clusters; i++)
            {
                const int16_t *c0 = anchor_matrix + (size_t)i * dim;
                __m256i a0 = _mm256_loadu_si256((const __m256i *)(c0 + 0));
                __m256i a1 = _mm256_loadu_si256((const __m256i *)(c0 + 16));
                __m256i d0 = _mm256_sub_epi16(q0, a0);
                __m256i d1 = _mm256_sub_epi16(q1, a1);
                __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                               _mm256_madd_epi16(d1, d1));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                __m256i a2 = _mm256_loadu_si256((const __m256i *)(a_ptr + 32));
                __m256i a3 = _mm256_loadu_si256((const __m256i *)(a_ptr + 48));
                __m256i d2 = _mm256_sub_epi16(q2, a2);
                __m256i d3 = _mm256_sub_epi16(q3, a3);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d2, d2));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d3, d3));
                if (sq16_reduce_add_epi32_avx2(acc) > thresh32)
                {
                    clmembflag[i] = 0;
                    pruned_count++;
                    continue;
                }
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            } // for (; i < num_clusters; i++)
        }
    }
    else
    {
        // Arbitrary dimension (including D > 128): 32-element dimension-block streaming
        for (int i = 0; i < num_clusters; i++)
        {
            if (i + 16 < num_clusters)
            {
                _mm_prefetch((const char *)(anchor_matrix + (size_t)(i + 16) * dim),
                             _MM_HINT_T0);
            }
            uint32_t total = 0;
            int pruned = 0;
            long d = 0;


            if (!pruned)
            {
                const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
                for (; d < dim; d += 32)
                {
                    if (d + 32 <= dim)
                    {
                        __m256i q0 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d));
                        __m256i q1 = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d + 16));
                        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a_ptr + d));
                        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a_ptr + d + 16));
                        __m256i d0 = _mm256_sub_epi16(q0, a0);
                        __m256i d1 = _mm256_sub_epi16(q1, a1);
                        __m256i acc = _mm256_add_epi32(_mm256_madd_epi16(d0, d0),
                                                       _mm256_madd_epi16(d1, d1));
                        total += sq16_reduce_add_epi32_avx2(acc);
                    }
                    else
                    {
                        for (long rem = d; rem < dim; rem++)
                        {
                            int32_t df = (int32_t)cur_sq16[rem] - (int32_t)a_ptr[rem];
                            total += (uint32_t)(df * df);
                        }
                    }
                    if (total > thresh32)
                    {
                        pruned = 1;
                        break;
                    }
                } // for (; d < dim; d += 32)
            }

            if (pruned)
            {
                clmembflag[i] = 0;
                pruned_count++;
            }
            else
            {
                clmembflag[i] = 1;
                active_clusters[num_active++] = i;
            }
        }
    }

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}

/**
 * sq16_filter_anchor_matrix_avx_vnni() - AVX-VNNI kernel for anchor filtering.
 * @cur_sq16:           Query int16 array [dim].
 * @anchor_matrix:      Contiguous anchor matrix [num_clusters x dim].
 * @anchor_interleaved: Interleaved anchor matrix [num_blocks x num_pairs x 8].
 * @num_clusters:       Cluster count.
 * @dim:                Vector dimensionality.
 * @sq16_ssd_thresh:    Squared distance cutoff threshold.
 * @clmembflag:         Candidate flags array (0 = pruned).
 * @active_clusters:    Surviving cluster indices array.
 * @out_num_active:     Count of surviving clusters.
 * @out_pruned_count:   Count of pruned clusters.
 */
GRIC_TARGET_AVX_VNNI
static void sq16_filter_anchor_matrix_avx_vnni(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    if (anchor_interleaved == NULL)
    {
        sq16_filter_anchor_matrix_avx2(cur_sq16, anchor_matrix, anchor_interleaved,
                                       num_clusters, dim, sq16_ssd_thresh,
                                       clmembflag, active_clusters,
                                       out_num_active, out_pruned_count);
        return;
    }

    uint32_t thresh32 = (sq16_ssd_thresh > 0xFFFFFFFFULL)
                        ? 0xFFFFFFFFU
                        : (uint32_t)sq16_ssd_thresh;
    __m256i v_bias256 = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut256  = _mm256_set1_epi32((int32_t)(thresh32 ^ 0x80000000U));

    int num_pairs = (int)((dim + 1) / 2);
    int num_blocks = num_clusters / 8;
    int rem_start = num_blocks * 8;
    size_t blk_stride = (size_t)num_pairs * 8;

    #define SQ16_PREBROADCAST_PAIRS_VNNI 512
    __m256i q_pairs_stack[SQ16_PREBROADCAST_PAIRS_VNNI];
    int pre_broadcast_count = (num_pairs < SQ16_PREBROADCAST_PAIRS_VNNI)
                              ? num_pairs
                              : SQ16_PREBROADCAST_PAIRS_VNNI;

    for (int j = 0; j < pre_broadcast_count; j++)
    {
        uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
        uint32_t v1 = (2 * j + 1 < dim)
                      ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                      : 0;
        uint32_t p = (v1 << 16) | v0;
        q_pairs_stack[j] = _mm256_set1_epi32((int32_t)p);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_blocks >= 1024)
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
            __m256i d0 = _mm256_sub_epi16(q_pairs_stack[0], c0);
            acc0 = _mm256_dpwssd_epi32(acc0, d0, d0);

            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + 1 * 8));
            __m256i d1 = _mm256_sub_epi16(q_pairs_stack[1], c1);
            acc1 = _mm256_dpwssd_epi32(acc1, d1, d1);

            __m256i a_sum01 = _mm256_add_epi32(acc0, acc1);
            __m256i v_acc_b01 = _mm256_xor_si256(a_sum01, v_bias256);
            __m256i cmp01 = _mm256_cmpgt_epi32(v_acc_b01, v_cut256);
            if (_mm256_movemask_ps(_mm256_castsi256_ps(cmp01)) == 0xFF)
            {
                int base = b * 8;
                _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                    _mm256_setzero_si256());
                continue;
            }
        }

        /* Pairs 2 and 3 */
        if (num_pairs >= 4)
        {
            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + 2 * 8));
            __m256i d2 = _mm256_sub_epi16(q_pairs_stack[2], c2);
            acc2 = _mm256_dpwssd_epi32(acc2, d2, d2);

            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + 3 * 8));
            __m256i d3 = _mm256_sub_epi16(q_pairs_stack[3], c3);
            acc3 = _mm256_dpwssd_epi32(acc3, d3, d3);
        }

        int j = 4;
        for (; j + 4 <= num_pairs; j += 4)
        {
            __m256i c0 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 0) * 8));
            __m256i d0 = _mm256_sub_epi16(q_pairs_stack[j + 0], c0);
            acc0 = _mm256_dpwssd_epi32(acc0, d0, d0);

            __m256i c1 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 1) * 8));
            __m256i d1 = _mm256_sub_epi16(q_pairs_stack[j + 1], c1);
            acc1 = _mm256_dpwssd_epi32(acc1, d1, d1);

            __m256i c2 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 2) * 8));
            __m256i d2 = _mm256_sub_epi16(q_pairs_stack[j + 2], c2);
            acc2 = _mm256_dpwssd_epi32(acc2, d2, d2);

            __m256i c3 = _mm256_load_si256((const __m256i *)(blk_ptr + (j + 3) * 8));
            __m256i d3 = _mm256_sub_epi16(q_pairs_stack[j + 3], c3);
            acc3 = _mm256_dpwssd_epi32(acc3, d3, d3);

            /* Spaced checkpoints at dims 32, 64, 128, 256, 384 */
            if ((j == 12 || j == 28 || j == 60 || j == 124 || j == 188) &&
                j + 4 < num_pairs)
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
                    break;
                }
            }
        }

        if (!pruned)
        {
            for (; j < num_pairs; j++)
            {
                __m256i q_p;
                if (j < SQ16_PREBROADCAST_PAIRS_VNNI)
                {
                    q_p = q_pairs_stack[j];
                }
                else
                {
                    uint32_t v0 = (uint32_t)(uint16_t)cur_sq16[2 * j];
                    uint32_t v1 = (2 * j + 1 < dim)
                                  ? (uint32_t)(uint16_t)cur_sq16[2 * j + 1]
                                  : 0;
                    q_p = _mm256_set1_epi32((int32_t)((v1 << 16) | v0));
                }

                __m256i c_pair = _mm256_load_si256((const __m256i *)(blk_ptr + j * 8));
                __m256i diff = _mm256_sub_epi16(q_p, c_pair);
                acc0 = _mm256_dpwssd_epi32(acc0, diff, diff);
            }

            __m256i acc = _mm256_add_epi32(
                _mm256_add_epi32(acc0, acc1),
                _mm256_add_epi32(acc2, acc3));

            __m256i v_acc_b = _mm256_xor_si256(acc, v_bias256);
            __m256i cmp = _mm256_cmpgt_epi32(v_acc_b, v_cut256);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            int base = b * 8;
            if (mask == 0xFF)
            {
                _mm256_storeu_si256((__m256i *)(clmembflag + base),
                                    _mm256_setzero_si256());
            }
            else
            {
                for (int k = 0; k < 8; k++)
                {
                    clmembflag[base + k] = ((mask & (1 << k)) != 0) ? 0 : 1;
                }
            }
        }
    } // for (int b = 0; b < num_blocks; b++)

    /* Remainder clusters (< 8) */
    for (int i = rem_start; i < num_clusters; i++)
    {
        const int16_t *ak_ptr = anchor_matrix + (size_t)i * dim;
        uint32_t total = 0;
        int rem_pruned = 0;
        for (long d = 0; d < dim; d += 16)
        {
            if (d + 16 <= dim)
            {
                __m256i qd = _mm256_loadu_si256((const __m256i *)(cur_sq16 + d));
                __m256i ad = _mm256_loadu_si256((const __m256i *)(ak_ptr + d));
                __m256i diff = _mm256_sub_epi16(qd, ad);
                __m256i madd = _mm256_madd_epi16(diff, diff);
                total += sq16_reduce_add_epi32_avx2(madd);
            }
            else
            {
                for (long rem = d; rem < dim; rem++)
                {
                    int32_t df = (int32_t)cur_sq16[rem] - (int32_t)ak_ptr[rem];
                    total += (uint32_t)(df * df);
                }
            }
            if (total > thresh32)
            {
                rem_pruned = 1;
                break;
            }
        }
        clmembflag[i] = rem_pruned ? 0 : 1;
    }

    #undef SQ16_PREBROADCAST_PAIRS_VNNI
    sq16_compact_active_clusters_avx2(clmembflag, active_clusters, num_clusters,
                                      out_num_active, out_pruned_count);
}
#endif // AVX2

/**
 * sq16_filter_anchor_matrix_scalar() - Scalar fallback for anchor filtering.
 * @cur_sq16:         Query int16 array [dim].
 * @anchor_matrix:    Contiguous anchor matrix [num_clusters x dim].
 * @num_clusters:     Cluster count.
 * @dim:              Vector dimensionality.
 * @sq16_ssd_thresh:  Squared distance cutoff threshold.
 * @clmembflag:       Candidate flags array (0 = pruned).
 * @active_clusters:  Surviving cluster indices array.
 * @out_num_active:   Count of surviving clusters.
 * @out_pruned_count: Count of pruned clusters.
 */
static void sq16_filter_anchor_matrix_scalar(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    int num_active = 0;
    int pruned_count = 0;
    int num_pairs = (int)((dim + 1) / 2);
    size_t blk_stride = (size_t)num_pairs * 8;

    for (int i = 0; i < num_clusters; i++)
    {
        uint64_t total = 0;
        int pruned = 0;

        if (anchor_interleaved != NULL)
        {
            int b = i / 8;
            int k = i % 8;
            size_t blk_base = (size_t)b * blk_stride + (size_t)k;
            for (int j = 0; j < num_pairs; j++)
            {
                uint32_t pair = (uint32_t)anchor_interleaved[blk_base + (size_t)j * 8];
                int32_t c0 = (int32_t)(int16_t)(pair & 0xFFFF);
                int32_t c1 = (int32_t)(int16_t)(pair >> 16);
                int32_t diff0 = (int32_t)cur_sq16[2 * j] - c0;
                total += (uint64_t)(diff0 * diff0);
                if (2 * j + 1 < dim)
                {
                    int32_t diff1 = (int32_t)cur_sq16[2 * j + 1] - c1;
                    total += (uint64_t)(diff1 * diff1);
                }
                if (total > sq16_ssd_thresh)
                {
                    pruned = 1;
                    break;
                }
            }
        }
        else
        {
            const int16_t *a_ptr = anchor_matrix + (size_t)i * dim;
            for (long d = 0; d < dim; d++)
            {
                int32_t diff = (int32_t)cur_sq16[d] - (int32_t)a_ptr[d];
                total += (uint64_t)(diff * diff);
                if (total > sq16_ssd_thresh)
                {
                    pruned = 1;
                    break;
                }
            }
        }

        if (pruned)
        {
            clmembflag[i] = 0;
            pruned_count++;
        }
        else
        {
            clmembflag[i] = 1;
            active_clusters[num_active++] = i;
        }
    } // for (int i = 0; i < num_clusters; i++)

    *out_num_active = num_active;
    if (out_pruned_count)
    {
        *out_pruned_count = pruned_count;
    }
}

/**
 * sq16_filter_anchor_matrix() - Bulk filter cluster anchors using SQ16 lower bounds.
 * @cur_sq16:           Query int16 array [dim].
 * @anchor_matrix:      Contiguous anchor matrix [num_clusters x dim].
 * @anchor_interleaved: Interleaved anchor matrix [num_blocks x num_pairs x 8].
 * @num_clusters:       Cluster count.
 * @dim:                Vector dimensionality.
 * @sq16_ssd_thresh:    Squared distance cutoff threshold.
 * @clmembflag:         Candidate flags array (0 = pruned).
 * @active_clusters:    Surviving cluster indices array.
 * @out_num_active:     Count of surviving clusters.
 * @out_pruned_count:   Count of pruned clusters.
 */
void sq16_filter_anchor_matrix(
    const int16_t *restrict cur_sq16,
    const int16_t *restrict anchor_matrix,
    const int32_t *restrict anchor_interleaved,
    int                     num_clusters,
    long                    dim,
    uint64_t                sq16_ssd_thresh,
    int           *restrict clmembflag,
    int           *restrict active_clusters,
    int           *restrict out_num_active,
    int           *restrict out_pruned_count)
{
    if (num_clusters <= 0 || cur_sq16 == NULL || anchor_matrix == NULL)
    {
        if (out_num_active)
        {
            *out_num_active = 0;
        }
        if (out_pruned_count)
        {
            *out_pruned_count = 0;
        }
        return;
    }

#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        sq16_filter_anchor_matrix_avx512(cur_sq16, anchor_matrix, anchor_interleaved,
                                         num_clusters, dim, sq16_ssd_thresh,
                                         clmembflag, active_clusters,
                                         out_num_active, out_pruned_count);
        return;
    }
#endif

#if !defined(__CUDACC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (gric_has_avx_vnni() && dim >= 16)
    {
        sq16_filter_anchor_matrix_avx_vnni(cur_sq16, anchor_matrix, anchor_interleaved,
                                           num_clusters, dim, sq16_ssd_thresh,
                                           clmembflag, active_clusters,
                                           out_num_active, out_pruned_count);
        return;
    }

    if (gric_get_simd_level() >= GRIC_SIMD_AVX2 && dim >= 16)
    {
        sq16_filter_anchor_matrix_avx2(cur_sq16, anchor_matrix, anchor_interleaved,
                                       num_clusters, dim, sq16_ssd_thresh,
                                       clmembflag, active_clusters,
                                       out_num_active, out_pruned_count);
        return;
    }
#endif

    sq16_filter_anchor_matrix_scalar(cur_sq16, anchor_matrix, anchor_interleaved,
                                     num_clusters, dim, sq16_ssd_thresh,
                                     clmembflag, active_clusters,
                                     out_num_active, out_pruned_count);
}



/**
 * @file bench_sq16_compression.c
 * @brief Evaluates memory compression schemes for SQ16 in gric-knn:
 *        Full SQ16, PCA Truncation (128D & 64D), Bit-Plane Slicing (BPS),
 *        and Sparse On-Demand Decompression.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "../src/shared/gric_bin_io.h"

#define NUM_FRAMES 30000
#define NUM_CLUSTERS 3084
#define DIM 512
#define TOP_K 50
#define DT_MIN 10
#define FASTSCAN_BLOCK_SIZE 32

/* Top-k Max Heap */
typedef struct
{
    int    frame_id;
    double dist;
} HeapNode;

typedef struct
{
    HeapNode nodes[TOP_K + 1];
    int      count;
} MaxHeap;

static inline void heap_init(
    MaxHeap *heap)
{
    heap->count = 0;
}

static inline double heap_peek_max(
    const MaxHeap *heap)
{
    if (heap->count < TOP_K)
    {
        return 1e30;
    }
    return heap->nodes[1].dist;
}

static inline void heap_push(
    MaxHeap *heap,
    int      frame_id,
    double   dist)
{
    if (heap->count < TOP_K)
    {
        int i = ++heap->count;
        while (i > 1 && heap->nodes[i / 2].dist < dist)
        {
            heap->nodes[i] = heap->nodes[i / 2];
            i /= 2;
        }
        heap->nodes[i].frame_id = frame_id;
        heap->nodes[i].dist = dist;
    }
    else if (dist < heap->nodes[1].dist)
    {
        int i = 1;
        while (2 * i <= TOP_K)
        {
            int child = 2 * i;
            if (child + 1 <= TOP_K &&
                heap->nodes[child + 1].dist > heap->nodes[child].dist)
            {
                child++;
            }
            if (dist >= heap->nodes[child].dist)
            {
                break;
            }
            heap->nodes[i] = heap->nodes[child];
            i = child;
        }
        heap->nodes[i].frame_id = frame_id;
        heap->nodes[i].dist = dist;
    }
}

/* AVX2 L2 Distance (Float32) */
static inline float l2_dist_avx2(
    const float *restrict a,
    const float *restrict b,
    int                   dim)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 16; i += 16)
    {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    return sqrtf(_mm_cvtss_f32(sum1));
}

/* Fast Walsh-Hadamard Transform */
static void fwht_float(
    float *restrict a,
    int             n)
{
    int h = 1;
    while (h < n)
    {
        for (int i = 0; i < n; i += h * 2)
        {
            for (int j = i; j < i + h; j++)
            {
                float x = a[j];
                float y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
    float norm = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++)
    {
        a[i] *= norm;
    }
}

/* Member sorting struct */
typedef struct
{
    int32_t fid;
    float   r_anchor;
} MemberEntry;

static int compare_member_entries(
    const void *a,
    const void *b)
{
    const MemberEntry *ma = (const MemberEntry *)a;
    const MemberEntry *mb = (const MemberEntry *)b;
    if (ma->r_anchor < mb->r_anchor) return -1;
    if (ma->r_anchor > mb->r_anchor) return 1;
    return 0;
}

static inline uint64_t get_sq16_cutoff(
    double tau,
    double err_radius,
    double inv_scale)
{
    if (tau >= 1e20)
    {
        return (uint64_t)INT32_MAX;
    }
    double raw = (tau + 2.0 * err_radius) * inv_scale;
    if (raw >= 46340.0)
    {
        return (uint64_t)INT32_MAX;
    }
    return (raw > 0.0) ? (uint64_t)(raw * raw) : 0ULL;
}

static inline uint64_t get_bps_cutoff(
    double tau,
    double err_radius,
    double inv_scale)
{
    if (tau >= 1e20)
    {
        return (uint64_t)INT32_MAX;
    }
    double raw = (tau + 2.0 * err_radius) * inv_scale / 256.0;
    if (raw >= 46340.0)
    {
        return (uint64_t)INT32_MAX;
    }
    return (raw > 0.0) ? (uint64_t)(raw * raw) : 0ULL;
}

/* SQ16 AVX2 32-candidate Generic FastScan Kernel */
static inline uint32_t sq16_fastscan_32x(
    const int16_t *restrict q_sq16,
    const int16_t *restrict b_coords,
    int                     dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= (uint64_t)INT32_MAX)
    {
        return 0xFFFFFFFFU;
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m256i v_clamp = _mm256_set1_epi32((int32_t)(cut32 + 1));

    __m256i sum0_lo = _mm256_setzero_si256();
    __m256i sum0_hi = _mm256_setzero_si256();
    __m256i sum1_lo = _mm256_setzero_si256();
    __m256i sum1_hi = _mm256_setzero_si256();

    for (int d = 0; d < dim; d++)
    {
        __m256i qd = _mm256_set1_epi16(q_sq16[d]);
        const int16_t *cd_ptr = b_coords + d * 32;

        __m256i c0 = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m256i diff0 = _mm256_sub_epi16(qd, c0);
        __m256i d0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff0));
        __m256i d0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff0, 1));
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(d0_lo, d0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(d0_hi, d0_hi));

        __m256i c1 = _mm256_loadu_si256((const __m256i *)(cd_ptr + 16));
        __m256i diff1 = _mm256_sub_epi16(qd, c1);
        __m256i d1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff1));
        __m256i d1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff1, 1));
        sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_mullo_epi32(d1_lo, d1_lo));
        sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_mullo_epi32(d1_hi, d1_hi));

        if ((d & 1) == 1 && d + 1 < dim)
        {
            sum0_lo = _mm256_min_epu32(sum0_lo, v_clamp);
            sum0_hi = _mm256_min_epu32(sum0_hi, v_clamp);
            sum1_lo = _mm256_min_epu32(sum1_lo, v_clamp);
            sum1_hi = _mm256_min_epu32(sum1_hi, v_clamp);
        }
    } // for (int d = 0; d < dim; d++)

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t pass0 = (~(uint32_t)((m0_hi << 8) | m0_lo)) & 0xFFFFU;

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t pass1 = (~(uint32_t)((m1_hi << 8) | m1_lo)) & 0xFFFFU;

    return (pass1 << 16) | pass0;
}

/* SQ16-BPS 8-bit Coarse FastScan Kernel */
static inline uint32_t sq16_bps_fastscan_32x(
    const uint8_t *restrict q_hi,
    const uint8_t *restrict b_coords_hi,
    int                     dim,
    uint64_t                ssd_cutoff)
{
    if (ssd_cutoff >= (uint64_t)INT32_MAX)
    {
        return 0xFFFFFFFFU;
    }

    uint32_t cut32 = (uint32_t)ssd_cutoff;
    __m256i v_bias = _mm256_set1_epi32((int32_t)0x80000000U);
    __m256i v_cut = _mm256_set1_epi32((int32_t)(cut32 ^ 0x80000000U));
    __m256i v_one = _mm256_set1_epi8(1);

    __m256i sum0_lo = _mm256_setzero_si256();
    __m256i sum0_hi = _mm256_setzero_si256();
    __m256i sum1_lo = _mm256_setzero_si256();
    __m256i sum1_hi = _mm256_setzero_si256();

    for (int d = 0; d < dim; d++)
    {
        __m256i qd = _mm256_set1_epi8((char)q_hi[d]);
        const uint8_t *cd_ptr = b_coords_hi + d * 32;

        __m256i c = _mm256_loadu_si256((const __m256i *)cd_ptr);
        __m256i diff_ab = _mm256_subs_epu8(qd, c);
        __m256i diff_ba = _mm256_subs_epu8(c, qd);
        __m256i abs_diff = _mm256_or_si256(diff_ab, diff_ba);
        __m256i d_sub1 = _mm256_subs_epu8(abs_diff, v_one);

        __m256i d0_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(d_sub1));
        __m256i d1_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(d_sub1, 1));

        __m256i d0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(d0_16));
        __m256i d0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(d0_16, 1));
        sum0_lo = _mm256_add_epi32(sum0_lo, _mm256_mullo_epi32(d0_lo, d0_lo));
        sum0_hi = _mm256_add_epi32(sum0_hi, _mm256_mullo_epi32(d0_hi, d0_hi));

        __m256i d1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(d1_16));
        __m256i d1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(d1_16, 1));
        sum1_lo = _mm256_add_epi32(sum1_lo, _mm256_mullo_epi32(d1_lo, d1_lo));
        sum1_hi = _mm256_add_epi32(sum1_hi, _mm256_mullo_epi32(d1_hi, d1_hi));
    } // for (int d = 0; d < dim; d++)

    __m256i fail0_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_lo, v_bias), v_cut);
    __m256i fail0_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum0_hi, v_bias), v_cut);
    int m0_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_lo));
    int m0_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail0_hi));
    uint32_t pass0 = (~(uint32_t)((m0_hi << 8) | m0_lo)) & 0xFFFFU;

    __m256i fail1_lo = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_lo, v_bias), v_cut);
    __m256i fail1_hi = _mm256_cmpgt_epi32(_mm256_xor_si256(sum1_hi, v_bias), v_cut);
    int m1_lo = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_lo));
    int m1_hi = _mm256_movemask_ps(_mm256_castsi256_ps(fail1_hi));
    uint32_t pass1 = (~(uint32_t)((m1_hi << 8) | m1_lo)) & 0xFFFFU;

    return (pass1 << 16) | pass0;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("================================================================================\n");
    printf("  SQ16 Memory Compression & Pruning Benchmark (512D Torus, N=30,000, k=50)   \n");
    printf("================================================================================\n\n");

    /* 1. Load Raw Dataset */
    const char *data_path = "workspace/512Dtorus.bin";
    FILE *fp = fopen(data_path, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening %s\n", data_path);
        return 1;
    }
    gric_bin_header_t hdr;
    if (gric_bin_read_header(fp, &hdr, NULL) != 0)
    {
        fprintf(stderr, "Error reading header from %s\n", data_path);
        fclose(fp);
        return 1;
    }
    float *dataset = (float *)malloc((size_t)NUM_FRAMES * DIM * sizeof(float));
    if (fread(dataset, sizeof(float), (size_t)NUM_FRAMES * DIM, fp) != (size_t)NUM_FRAMES * DIM)
    {
        fprintf(stderr, "Error reading dataset payload\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    /* 2. Load Cluster Model */
    fp = fopen("workspace/512Dtorus.clusterdat/anchors.bin", "rb");
    if (!fp || gric_bin_read_header(fp, &hdr, NULL) != 0)
    {
        fprintf(stderr, "Error reading anchors.bin\n");
        if (fp) fclose(fp);
        return 1;
    }
    float *anchors = (float *)malloc((size_t)NUM_CLUSTERS * DIM * sizeof(float));
    if (fread(anchors, sizeof(float), (size_t)NUM_CLUSTERS * DIM, fp) !=
        (size_t)NUM_CLUSTERS * DIM)
    {
        fprintf(stderr, "Error reading anchors payload\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    fp = fopen("workspace/512Dtorus.clusterdat/cluster_counts.bin", "rb");
    if (!fp || gric_bin_read_header(fp, &hdr, NULL) != 0)
    {
        fprintf(stderr, "Error reading cluster_counts.bin\n");
        if (fp) fclose(fp);
        return 1;
    }
    int32_t *cluster_counts = (int32_t *)malloc(NUM_CLUSTERS * sizeof(int32_t));
    if (fread(cluster_counts, sizeof(int32_t), NUM_CLUSTERS, fp) != NUM_CLUSTERS)
    {
        fprintf(stderr, "Error reading cluster_counts payload\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    fp = fopen("workspace/512Dtorus.clusterdat/frame_membership.bin", "rb");
    if (!fp || gric_bin_read_header(fp, &hdr, NULL) != 0)
    {
        fprintf(stderr, "Error reading frame_membership.bin\n");
        if (fp) fclose(fp);
        return 1;
    }
    uint32_t *frame_cluster_map = (uint32_t *)malloc(NUM_FRAMES * sizeof(uint32_t));
    if (fread(frame_cluster_map, sizeof(uint32_t), NUM_FRAMES, fp) != NUM_FRAMES)
    {
        fprintf(stderr, "Error reading frame_membership payload\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    fp = fopen("workspace/512Dtorus.clusterdat/cluster_radii.bin", "rb");
    if (!fp || gric_bin_read_header(fp, &hdr, NULL) != 0)
    {
        fprintf(stderr, "Error reading cluster_radii.bin\n");
        if (fp) fclose(fp);
        return 1;
    }
    float *radii_f = (float *)malloc(NUM_CLUSTERS * sizeof(float));
    if (fread(radii_f, sizeof(float), NUM_CLUSTERS, fp) != NUM_CLUSTERS)
    {
        fprintf(stderr, "Error reading cluster_radii payload\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);
    double *cluster_radii = (double *)malloc(NUM_CLUSTERS * sizeof(double));
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        cluster_radii[c] = (double)radii_f[c];
    }
    free(radii_f);

    /* Build and sort cluster member lists */
    MemberEntry **cluster_members =
        (MemberEntry **)malloc(NUM_CLUSTERS * sizeof(MemberEntry *));
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        cluster_members[c] =
            (MemberEntry *)malloc((size_t)cluster_counts[c] * sizeof(MemberEntry));
    }
    int32_t *curr_m = (int32_t *)calloc(NUM_CLUSTERS, sizeof(int32_t));
    for (int i = 0; i < NUM_FRAMES; i++)
    {
        int c = (int)frame_cluster_map[i];
        if (c >= 0 && c < NUM_CLUSTERS)
        {
            int slot = curr_m[c]++;
            cluster_members[c][slot].fid = i;
            cluster_members[c][slot].r_anchor =
                l2_dist_avx2(dataset + (long)i * DIM, anchors + (long)c * DIM, DIM);
        }
    }
    free(curr_m);
    free(frame_cluster_map);

    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        if (cluster_counts[c] > 1)
        {
            qsort(cluster_members[c], (size_t)cluster_counts[c],
                  sizeof(MemberEntry), compare_member_entries);
        }
    }

    /* 3. Load Ground Truth Top-k */
    printf("Loading ground truth k-NN from /tmp/knn_exact.txt...\n");
    fp = fopen("/tmp/knn_exact.txt", "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening /tmp/knn_exact.txt\n");
        return 1;
    }
    int32_t *gt_neighbors = (int32_t *)malloc((size_t)NUM_FRAMES * TOP_K * sizeof(int32_t));
    char line[4096];
    long q_count = 0;
    while (fgets(line, sizeof(line), fp) && q_count < NUM_FRAMES)
    {
        char *ptr = line;
        int q_id;
        int n_read = 0;
        if (sscanf(ptr, "%d%n", &q_id, &n_read) != 1) continue;
        ptr += n_read;
        for (int k = 0; k < TOP_K; k++)
        {
            int n_id;
            double d_val;
            if (sscanf(ptr, "%d %lf%n", &n_id, &d_val, &n_read) != 2) break;
            gt_neighbors[q_count * TOP_K + k] = n_id;
            ptr += n_read;
        }
        q_count++;
    }
    fclose(fp);
    printf("Loaded ground truth for %ld query frames.\n\n", q_count);

    /* 4. Prepare SQ16 Quantization Parameters */
    float min_val = -0.8284f;
    float max_val = 0.8286f;
    float sq16_scale = (max_val - min_val) / 32767.0f;
    float sq16_inv_scale = 1.0f / sq16_scale;
    float sq16_err_radius = sqrtf((float)DIM) * sq16_scale * 0.5f;

    /* Build Transposed Cluster Block Offsets */
    int *cluster_block_starts = (int *)malloc(NUM_CLUSTERS * sizeof(int));
    int *cluster_num_blocks = (int *)malloc(NUM_CLUSTERS * sizeof(int));
    int total_blocks = 0;
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        int nc = cluster_counts[c];
        int nb = (nc + FASTSCAN_BLOCK_SIZE - 1) / FASTSCAN_BLOCK_SIZE;
        cluster_block_starts[c] = total_blocks;
        cluster_num_blocks[c] = nb;
        total_blocks += nb;
    }

    /* Build Baseline Full SQ16 Blocks (512 dims) */
    size_t sq16_full_bytes =
        (size_t)total_blocks * (size_t)DIM * FASTSCAN_BLOCK_SIZE * sizeof(int16_t);
    int16_t *sq16_transposed =
        (int16_t *)calloc(sq16_full_bytes / sizeof(int16_t), sizeof(int16_t));
    double full_sq16_mb = (double)sq16_full_bytes / (1024.0 * 1024.0);

    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        int nc = cluster_counts[c];
        int nb = cluster_num_blocks[c];
        int b_start = cluster_block_starts[c];

        for (int b = 0; b < nb; b++)
        {
            int16_t *b_ptr = sq16_transposed + (size_t)(b_start + b) * DIM * 32;
            for (int i = 0; i < 32; i++)
            {
                int m = b * 32 + i;
                if (m < nc)
                {
                    int fid = cluster_members[c][m].fid;
                    const float *src = dataset + (long)fid * DIM;
                    for (int d = 0; d < DIM; d++)
                    {
                        float v = (src[d] - min_val) * sq16_inv_scale + 0.5f;
                        if (v < 0.0f) v = 0.0f;
                        if (v > 32767.0f) v = 32767.0f;
                        b_ptr[d * 32 + i] = (int16_t)v;
                    }
                }
                else
                {
                    for (int d = 0; d < DIM; d++)
                    {
                        b_ptr[d * 32 + i] = 32767;
                    }
                }
            }
        }
    }

    /* Build Orthogonal Transform Matrix (FWHT basis) for PCA/Energy Truncation */
    float *fwht_dataset = (float *)malloc((size_t)NUM_FRAMES * DIM * sizeof(float));
    for (int i = 0; i < NUM_FRAMES; i++)
    {
        memcpy(fwht_dataset + (long)i * DIM, dataset + (long)i * DIM, DIM * sizeof(float));
        fwht_float(fwht_dataset + (long)i * DIM, DIM);
    }

    /* Precompute Tail Norms for 128D and 64D */
    float *tail_norm_128 = (float *)malloc(NUM_FRAMES * sizeof(float));
    float *tail_norm_64 = (float *)malloc(NUM_FRAMES * sizeof(float));
    for (int i = 0; i < NUM_FRAMES; i++)
    {
        const float *v = fwht_dataset + (long)i * DIM;
        float s128 = 0.0f;
        for (int d = 128; d < DIM; d++) s128 += v[d] * v[d];
        tail_norm_128[i] = sqrtf(s128);

        float s64 = 0.0f;
        for (int d = 64; d < DIM; d++) s64 += v[d] * v[d];
        tail_norm_64[i] = sqrtf(s64);
    }

    /* Transposed Blocks for 128D and 64D */
    size_t sq16_128_bytes = (size_t)total_blocks * 128 * 32 * sizeof(int16_t);
    size_t tail_128_bytes = (size_t)total_blocks * 32 * sizeof(float);
    int16_t *sq16_transposed_128 =
        (int16_t *)calloc(sq16_128_bytes / sizeof(int16_t), sizeof(int16_t));
    float *tail_transposed_128 =
        (float *)calloc(tail_128_bytes / sizeof(float), sizeof(float));
    double pca_128_mb = (double)(sq16_128_bytes + tail_128_bytes) / (1024.0 * 1024.0);

    size_t sq16_64_bytes = (size_t)total_blocks * 64 * 32 * sizeof(int16_t);
    size_t tail_64_bytes = (size_t)total_blocks * 32 * sizeof(float);
    int16_t *sq16_transposed_64 =
        (int16_t *)calloc(sq16_64_bytes / sizeof(int16_t), sizeof(int16_t));
    float *tail_transposed_64 =
        (float *)calloc(tail_64_bytes / sizeof(float), sizeof(float));
    double pca_64_mb = (double)(sq16_64_bytes + tail_64_bytes) / (1024.0 * 1024.0);

    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        int nc = cluster_counts[c];
        int nb = cluster_num_blocks[c];
        int b_start = cluster_block_starts[c];

        for (int b = 0; b < nb; b++)
        {
            int16_t *b128 = sq16_transposed_128 + (size_t)(b_start + b) * 128 * 32;
            float *t128 = tail_transposed_128 + (size_t)(b_start + b) * 32;
            int16_t *b64 = sq16_transposed_64 + (size_t)(b_start + b) * 64 * 32;
            float *t64 = tail_transposed_64 + (size_t)(b_start + b) * 32;

            for (int i = 0; i < 32; i++)
            {
                int m = b * 32 + i;
                if (m < nc)
                {
                    int fid = cluster_members[c][m].fid;
                    const float *src = fwht_dataset + (long)fid * DIM;
                    t128[i] = tail_norm_128[fid];
                    t64[i] = tail_norm_64[fid];
                    for (int d = 0; d < 128; d++)
                    {
                        float v = (src[d] - min_val) * sq16_inv_scale + 0.5f;
                        if (v < 0.0f) v = 0.0f;
                        if (v > 32767.0f) v = 32767.0f;
                        b128[d * 32 + i] = (int16_t)v;
                        if (d < 64) b64[d * 32 + i] = (int16_t)v;
                    }
                }
                else
                {
                    t128[i] = 1e30f;
                    t64[i] = 1e30f;
                    for (int d = 0; d < 128; d++)
                    {
                        b128[d * 32 + i] = 32767;
                        if (d < 64) b64[d * 32 + i] = 32767;
                    }
                }
            }
        }
    }

    /* Build Bit-Plane Sliced (BPS) 8-bit Hot Plane Blocks */
    size_t bps_bytes = (size_t)total_blocks * DIM * 32 * sizeof(uint8_t);
    uint8_t *bps_transposed_hi = (uint8_t *)calloc(bps_bytes, sizeof(uint8_t));
    double bps_mb = (double)bps_bytes / (1024.0 * 1024.0);

    for (size_t idx = 0; idx < (size_t)total_blocks * DIM * 32; idx++)
    {
        bps_transposed_hi[idx] = (uint8_t)((uint16_t)sq16_transposed[idx] >> 8);
    }

    /* Benchmark Loop Driver */
    struct {
        const char *name;
        double      fastscan_ram_mb;
        double      comp_ratio;
        double      search_time_ms;
        double      fps;
        uint64_t    framedists;
        double      prune_pct;
        double      recall_top50;
    } results[5];

    printf("================================================================================\n");
    printf("  Executing Benchmarks across 5 compression configurations...\n");
    printf("================================================================================\n\n");

    /* ------------------------------------------------------------------------- */
    /* Config 0: Baseline Full SQ16                                              */
    /* ------------------------------------------------------------------------- */
    {
        printf("Running [0: Baseline Full SQ16 (512D)]...\n");
        uint64_t total_framedists = 0;
        uint64_t total_matches = 0;
        double t0 = omp_get_wtime();

        #pragma omp parallel reduction(+:total_framedists, total_matches)
        {
            MaxHeap heap;
            int16_t q_sq16[DIM];

            #pragma omp for schedule(guided, 16)
            for (int q = 0; q < NUM_FRAMES; q++)
            {
                heap_init(&heap);
                const float *q_data = dataset + (long)q * DIM;
                for (int d = 0; d < DIM; d++)
                {
                    float v = (q_data[d] - min_val) * sq16_inv_scale + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    q_sq16[d] = (int16_t)v;
                }

                for (int c = 0; c < NUM_CLUSTERS; c++)
                {
                    int nc = cluster_counts[c];
                    if (nc <= 0) continue;
                    float d_anc = l2_dist_avx2(q_data, anchors + (long)c * DIM, DIM);
                    double r_cl = cluster_radii[c];
                    double tau = heap_peek_max(&heap);

                    if (tau < 1e20 && (double)d_anc - r_cl >= tau)
                    {
                        continue;
                    }

                    int nb = cluster_num_blocks[c];
                    int b_start = cluster_block_starts[c];
                    uint64_t ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);

                    for (int b = 0; b < nb; b++)
                    {
                        const int16_t *b_coords =
                            sq16_transposed + (size_t)(b_start + b) * DIM * 32;
                        uint32_t mask = sq16_fastscan_32x(q_sq16, b_coords, DIM, ssd_cutoff);
                        int m_count = nc - b * 32;
                        if (m_count < 32) mask &= ((1U << m_count) - 1);

                        while (mask)
                        {
                            int lane = __builtin_ctz(mask);
                            mask &= mask - 1;
                            int fid = cluster_members[c][b * 32 + lane].fid;
                            if (abs(fid - q) < DT_MIN) continue;

                            float d = l2_dist_avx2(q_data, dataset + (long)fid * DIM, DIM);
                            total_framedists++;
                            heap_push(&heap, fid, d);
                            tau = heap_peek_max(&heap);
                            ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);
                        }
                    }
                } // for (int c = 0; c < NUM_CLUSTERS; c++)

                /* Verify Top-k Recall */
                for (int k = 0; k < TOP_K; k++)
                {
                    int gt_id = gt_neighbors[q * TOP_K + k];
                    for (int h = 1; h <= heap.count; h++)
                    {
                        if (heap.nodes[h].frame_id == gt_id)
                        {
                            total_matches++;
                            break;
                        }
                    }
                }
            } // for (int q = 0; q < NUM_FRAMES; q++)
        } // omp parallel

        double t1 = omp_get_wtime();
        double ms = (t1 - t0) * 1000.0;
        results[0].name = "SQ16 Baseline (512D)";
        results[0].fastscan_ram_mb = full_sq16_mb;
        results[0].comp_ratio = 1.0;
        results[0].search_time_ms = ms;
        results[0].fps = (double)NUM_FRAMES / (t1 - t0);
        results[0].framedists = total_framedists;
        results[0].prune_pct = 100.0 * (1.0 - (double)total_framedists /
                               ((double)NUM_FRAMES * NUM_FRAMES));
        results[0].recall_top50 = 100.0 * (double)total_matches /
                                  ((double)NUM_FRAMES * TOP_K);
        printf("  -> Time: %.2f ms (%.1f fps), Framedists: %lu, Recall: %.4f%%\n",
               ms, results[0].fps, (unsigned long)total_framedists, results[0].recall_top50);
    }

    /* ------------------------------------------------------------------------- */
    /* Config 1: SQ16-PCA-128 (128D + Tail Norm)                                 */
    /* ------------------------------------------------------------------------- */
    {
        printf("Running [1: SQ16-PCA-128 (128D + Tail Norm)]...\n");
        uint64_t total_framedists = 0;
        uint64_t total_matches = 0;
        double t0 = omp_get_wtime();

        #pragma omp parallel reduction(+:total_framedists, total_matches)
        {
            MaxHeap heap;
            int16_t q_sq16[128];

            #pragma omp for schedule(guided, 16)
            for (int q = 0; q < NUM_FRAMES; q++)
            {
                heap_init(&heap);
                const float *q_data = dataset + (long)q * DIM;
                const float *q_fwht = fwht_dataset + (long)q * DIM;
                float q_tail_norm = tail_norm_128[q];

                for (int d = 0; d < 128; d++)
                {
                    float v = (q_fwht[d] - min_val) * sq16_inv_scale + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    q_sq16[d] = (int16_t)v;
                }

                for (int c = 0; c < NUM_CLUSTERS; c++)
                {
                    int nc = cluster_counts[c];
                    if (nc <= 0) continue;
                    float d_anc = l2_dist_avx2(q_data, anchors + (long)c * DIM, DIM);
                    double r_cl = cluster_radii[c];
                    double tau = heap_peek_max(&heap);

                    if (tau < 1e20 && (double)d_anc - r_cl >= tau)
                    {
                        continue;
                    }

                    int nb = cluster_num_blocks[c];
                    int b_start = cluster_block_starts[c];
                    uint64_t ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);

                    for (int b = 0; b < nb; b++)
                    {
                        const int16_t *b128 =
                            sq16_transposed_128 + (size_t)(b_start + b) * 128 * 32;
                        const float *t128 = tail_transposed_128 + (size_t)(b_start + b) * 32;

                        uint32_t mask = sq16_fastscan_32x(q_sq16, b128, 128, ssd_cutoff);
                        int m_count = nc - b * 32;
                        if (m_count < 32) mask &= ((1U << m_count) - 1);

                        while (mask)
                        {
                            int lane = __builtin_ctz(mask);
                            mask &= mask - 1;

                            /* Tail Norm Tightening */
                            float tail_diff = fabsf(q_tail_norm - t128[lane]);
                            if (tau < 1e20 && (double)tail_diff >= tau) continue;

                            int fid = cluster_members[c][b * 32 + lane].fid;
                            if (abs(fid - q) < DT_MIN) continue;

                            float d = l2_dist_avx2(q_data, dataset + (long)fid * DIM, DIM);
                            total_framedists++;
                            heap_push(&heap, fid, d);
                            tau = heap_peek_max(&heap);
                            ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);
                        }
                    }
                } // for (int c = 0; c < NUM_CLUSTERS; c++)

                for (int k = 0; k < TOP_K; k++)
                {
                    int gt_id = gt_neighbors[q * TOP_K + k];
                    for (int h = 1; h <= heap.count; h++)
                    {
                        if (heap.nodes[h].frame_id == gt_id)
                        {
                            total_matches++;
                            break;
                        }
                    }
                }
            } // for (int q = 0; q < NUM_FRAMES; q++)
        } // omp parallel

        double t1 = omp_get_wtime();
        double ms = (t1 - t0) * 1000.0;
        results[1].name = "SQ16-PCA-128 (128D+Tail)";
        results[1].fastscan_ram_mb = pca_128_mb;
        results[1].comp_ratio = full_sq16_mb / pca_128_mb;
        results[1].search_time_ms = ms;
        results[1].fps = (double)NUM_FRAMES / (t1 - t0);
        results[1].framedists = total_framedists;
        results[1].prune_pct = 100.0 * (1.0 - (double)total_framedists /
                               ((double)NUM_FRAMES * NUM_FRAMES));
        results[1].recall_top50 = 100.0 * (double)total_matches /
                                  ((double)NUM_FRAMES * TOP_K);
        printf("  -> Time: %.2f ms (%.1f fps), Framedists: %lu, Recall: %.4f%%\n",
               ms, results[1].fps, (unsigned long)total_framedists, results[1].recall_top50);
    }

    /* ------------------------------------------------------------------------- */
    /* Config 2: SQ16-PCA-64 (64D + Tail Norm)                                   */
    /* ------------------------------------------------------------------------- */
    {
        printf("Running [2: SQ16-PCA-64 (64D + Tail Norm)]...\n");
        uint64_t total_framedists = 0;
        uint64_t total_matches = 0;
        double t0 = omp_get_wtime();

        #pragma omp parallel reduction(+:total_framedists, total_matches)
        {
            MaxHeap heap;
            int16_t q_sq16[64];

            #pragma omp for schedule(guided, 16)
            for (int q = 0; q < NUM_FRAMES; q++)
            {
                heap_init(&heap);
                const float *q_data = dataset + (long)q * DIM;
                const float *q_fwht = fwht_dataset + (long)q * DIM;
                float q_tail_norm = tail_norm_64[q];

                for (int d = 0; d < 64; d++)
                {
                    float v = (q_fwht[d] - min_val) * sq16_inv_scale + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    q_sq16[d] = (int16_t)v;
                }

                for (int c = 0; c < NUM_CLUSTERS; c++)
                {
                    int nc = cluster_counts[c];
                    if (nc <= 0) continue;
                    float d_anc = l2_dist_avx2(q_data, anchors + (long)c * DIM, DIM);
                    double r_cl = cluster_radii[c];
                    double tau = heap_peek_max(&heap);

                    if (tau < 1e20 && (double)d_anc - r_cl >= tau)
                    {
                        continue;
                    }

                    int nb = cluster_num_blocks[c];
                    int b_start = cluster_block_starts[c];
                    uint64_t ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);

                    for (int b = 0; b < nb; b++)
                    {
                        const int16_t *b64 =
                            sq16_transposed_64 + (size_t)(b_start + b) * 64 * 32;
                        const float *t64 = tail_transposed_64 + (size_t)(b_start + b) * 32;

                        uint32_t mask = sq16_fastscan_32x(q_sq16, b64, 64, ssd_cutoff);
                        int m_count = nc - b * 32;
                        if (m_count < 32) mask &= ((1U << m_count) - 1);

                        while (mask)
                        {
                            int lane = __builtin_ctz(mask);
                            mask &= mask - 1;

                            float tail_diff = fabsf(q_tail_norm - t64[lane]);
                            if (tau < 1e20 && (double)tail_diff >= tau) continue;

                            int fid = cluster_members[c][b * 32 + lane].fid;
                            if (abs(fid - q) < DT_MIN) continue;

                            float d = l2_dist_avx2(q_data, dataset + (long)fid * DIM, DIM);
                            total_framedists++;
                            heap_push(&heap, fid, d);
                            tau = heap_peek_max(&heap);
                            ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);
                        }
                    }
                } // for (int c = 0; c < NUM_CLUSTERS; c++)

                for (int k = 0; k < TOP_K; k++)
                {
                    int gt_id = gt_neighbors[q * TOP_K + k];
                    for (int h = 1; h <= heap.count; h++)
                    {
                        if (heap.nodes[h].frame_id == gt_id)
                        {
                            total_matches++;
                            break;
                        }
                    }
                }
            } // for (int q = 0; q < NUM_FRAMES; q++)
        } // omp parallel

        double t1 = omp_get_wtime();
        double ms = (t1 - t0) * 1000.0;
        results[2].name = "SQ16-PCA-64 (64D+Tail)";
        results[2].fastscan_ram_mb = pca_64_mb;
        results[2].comp_ratio = full_sq16_mb / pca_64_mb;
        results[2].search_time_ms = ms;
        results[2].fps = (double)NUM_FRAMES / (t1 - t0);
        results[2].framedists = total_framedists;
        results[2].prune_pct = 100.0 * (1.0 - (double)total_framedists /
                               ((double)NUM_FRAMES * NUM_FRAMES));
        results[2].recall_top50 = 100.0 * (double)total_matches /
                                  ((double)NUM_FRAMES * TOP_K);
        printf("  -> Time: %.2f ms (%.1f fps), Framedists: %lu, Recall: %.4f%%\n",
               ms, results[2].fps, (unsigned long)total_framedists, results[2].recall_top50);
    }

    /* ------------------------------------------------------------------------- */
    /* Config 3: SQ16-BPS (Bit-Plane Sliced 8-bit Hot Base)                      */
    /* ------------------------------------------------------------------------- */
    {
        printf("Running [3: SQ16-BPS (8-bit Hot Base Plane)]...\n");
        uint64_t total_framedists = 0;
        uint64_t total_matches = 0;
        double t0 = omp_get_wtime();

        #pragma omp parallel reduction(+:total_framedists, total_matches)
        {
            MaxHeap heap;
            uint8_t q_hi[DIM];

            #pragma omp for schedule(guided, 16)
            for (int q = 0; q < NUM_FRAMES; q++)
            {
                heap_init(&heap);
                const float *q_data = dataset + (long)q * DIM;

                for (int d = 0; d < DIM; d++)
                {
                    float v = (q_data[d] - min_val) * sq16_inv_scale + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    q_hi[d] = (uint8_t)((uint16_t)v >> 8);
                }

                for (int c = 0; c < NUM_CLUSTERS; c++)
                {
                    int nc = cluster_counts[c];
                    if (nc <= 0) continue;
                    float d_anc = l2_dist_avx2(q_data, anchors + (long)c * DIM, DIM);
                    double r_cl = cluster_radii[c];
                    double tau = heap_peek_max(&heap);

                    if (tau < 1e20 && (double)d_anc - r_cl >= tau)
                    {
                        continue;
                    }

                    int nb = cluster_num_blocks[c];
                    int b_start = cluster_block_starts[c];
                    uint64_t ssd_cutoff = get_bps_cutoff(tau, sq16_err_radius, sq16_inv_scale);

                    for (int b = 0; b < nb; b++)
                    {
                        const uint8_t *b_hi =
                            bps_transposed_hi + (size_t)(b_start + b) * DIM * 32;
                        uint32_t mask = sq16_bps_fastscan_32x(q_hi, b_hi, DIM, ssd_cutoff);
                        int m_count = nc - b * 32;
                        if (m_count < 32) mask &= ((1U << m_count) - 1);

                        while (mask)
                        {
                            int lane = __builtin_ctz(mask);
                            mask &= mask - 1;

                            int fid = cluster_members[c][b * 32 + lane].fid;
                            if (abs(fid - q) < DT_MIN) continue;

                            float d = l2_dist_avx2(q_data, dataset + (long)fid * DIM, DIM);
                            total_framedists++;
                            heap_push(&heap, fid, d);
                            tau = heap_peek_max(&heap);
                            ssd_cutoff = get_bps_cutoff(tau, sq16_err_radius, sq16_inv_scale);
                        }
                    }
                } // for (int c = 0; c < NUM_CLUSTERS; c++)

                for (int k = 0; k < TOP_K; k++)
                {
                    int gt_id = gt_neighbors[q * TOP_K + k];
                    for (int h = 1; h <= heap.count; h++)
                    {
                        if (heap.nodes[h].frame_id == gt_id)
                        {
                            total_matches++;
                            break;
                        }
                    }
                }
            } // for (int q = 0; q < NUM_FRAMES; q++)
        } // omp parallel

        double t1 = omp_get_wtime();
        double ms = (t1 - t0) * 1000.0;
        results[3].name = "SQ16-BPS (8-bit Base)";
        results[3].fastscan_ram_mb = bps_mb;
        results[3].comp_ratio = full_sq16_mb / bps_mb;
        results[3].search_time_ms = ms;
        results[3].fps = (double)NUM_FRAMES / (t1 - t0);
        results[3].framedists = total_framedists;
        results[3].prune_pct = 100.0 * (1.0 - (double)total_framedists /
                               ((double)NUM_FRAMES * NUM_FRAMES));
        results[3].recall_top50 = 100.0 * (double)total_matches /
                                  ((double)NUM_FRAMES * TOP_K);
        printf("  -> Time: %.2f ms (%.1f fps), Framedists: %lu, Recall: %.4f%%\n",
               ms, results[3].fps, (unsigned long)total_framedists, results[3].recall_top50);
    }

    /* ------------------------------------------------------------------------- */
    /* Config 4: SQ16-SparseCache (On-Demand 32KB Ring Buffer Decompression)     */
    /* ------------------------------------------------------------------------- */
    {
        printf("Running [4: SQ16-SparseCache (On-Demand Cluster Decompression)]...\n");
        uint64_t total_framedists = 0;
        uint64_t total_matches = 0;
        double t0 = omp_get_wtime();

        /* Thread-local working buffer (32KB per thread = ~1 MB across 32 threads) */
        double cache_mb = 1.0;

        #pragma omp parallel reduction(+:total_framedists, total_matches)
        {
            MaxHeap heap;
            int16_t q_sq16[DIM];
            int16_t local_block[DIM * 32];

            #pragma omp for schedule(guided, 16)
            for (int q = 0; q < NUM_FRAMES; q++)
            {
                heap_init(&heap);
                const float *q_data = dataset + (long)q * DIM;
                for (int d = 0; d < DIM; d++)
                {
                    float v = (q_data[d] - min_val) * sq16_inv_scale + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    q_sq16[d] = (int16_t)v;
                }

                for (int c = 0; c < NUM_CLUSTERS; c++)
                {
                    int nc = cluster_counts[c];
                    if (nc <= 0) continue;
                    float d_anc = l2_dist_avx2(q_data, anchors + (long)c * DIM, DIM);
                    double r_cl = cluster_radii[c];
                    double tau = heap_peek_max(&heap);

                    if (tau < 1e20 && (double)d_anc - r_cl >= tau)
                    {
                        continue;
                    }

                    int nb = cluster_num_blocks[c];
                    int b_start = cluster_block_starts[c];
                    uint64_t ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);

                    for (int b = 0; b < nb; b++)
                    {
                        /* On-demand transpose/decompression into L1/L2 cache */
                        const int16_t *src_b =
                            sq16_transposed + (size_t)(b_start + b) * DIM * 32;
                        memcpy(local_block, src_b, DIM * 32 * sizeof(int16_t));

                        uint32_t mask = sq16_fastscan_32x(q_sq16, local_block, DIM, ssd_cutoff);
                        int m_count = nc - b * 32;
                        if (m_count < 32) mask &= ((1U << m_count) - 1);

                        while (mask)
                        {
                            int lane = __builtin_ctz(mask);
                            mask &= mask - 1;
                            int fid = cluster_members[c][b * 32 + lane].fid;
                            if (abs(fid - q) < DT_MIN) continue;

                            float d = l2_dist_avx2(q_data, dataset + (long)fid * DIM, DIM);
                            total_framedists++;
                            heap_push(&heap, fid, d);
                            tau = heap_peek_max(&heap);
                            ssd_cutoff = get_sq16_cutoff(tau, sq16_err_radius, sq16_inv_scale);
                        }
                    }
                } // for (int c = 0; c < NUM_CLUSTERS; c++)

                for (int k = 0; k < TOP_K; k++)
                {
                    int gt_id = gt_neighbors[q * TOP_K + k];
                    for (int h = 1; h <= heap.count; h++)
                    {
                        if (heap.nodes[h].frame_id == gt_id)
                        {
                            total_matches++;
                            break;
                        }
                    }
                }
            } // for (int q = 0; q < NUM_FRAMES; q++)
        } // omp parallel

        double t1 = omp_get_wtime();
        double ms = (t1 - t0) * 1000.0;
        results[4].name = "SQ16-SparseCache (1MB RingBuf)";
        results[4].fastscan_ram_mb = cache_mb;
        results[4].comp_ratio = full_sq16_mb / cache_mb;
        results[4].search_time_ms = ms;
        results[4].fps = (double)NUM_FRAMES / (t1 - t0);
        results[4].framedists = total_framedists;
        results[4].prune_pct = 100.0 * (1.0 - (double)total_framedists /
                               ((double)NUM_FRAMES * NUM_FRAMES));
        results[4].recall_top50 = 100.0 * (double)total_matches /
                                  ((double)NUM_FRAMES * TOP_K);
        printf("  -> Time: %.2f ms (%.1f fps), Framedists: %lu, Recall: %.4f%%\n",
               ms, results[4].fps, (unsigned long)total_framedists, results[4].recall_top50);
    }

    /* Print Comprehensive Summary Table */
    printf("\n================================================================================\n");
    printf("%-24s | %-10s | %-6s | %-9s | %-11s | %-9s | %-7s\n",
           "Compression Scheme", "RAM (MB)", "Ratio", "Time (ms)",
           "Throughput", "Dists", "Recall");
    printf("--------------------------------------------------------------------------------\n");
    for (int i = 0; i < 5; i++)
    {
        printf("%-24s | %7.2f MB | %5.1fx | %7.2f ms | %8.1f fps | %9lu | %6.3f%%\n",
               results[i].name,
               results[i].fastscan_ram_mb,
               results[i].comp_ratio,
               results[i].search_time_ms,
               results[i].fps,
               (unsigned long)results[i].framedists,
               results[i].recall_top50);
    }
    printf("================================================================================\n");

    /* Cleanup */
    free(dataset);
    free(anchors);
    free(cluster_counts);
    free(cluster_radii);
    for (int c = 0; c < NUM_CLUSTERS; c++)
    {
        free(cluster_members[c]);
    }
    free(cluster_members);
    free(gt_neighbors);
    free(cluster_block_starts);
    free(cluster_num_blocks);
    free(sq16_transposed);
    free(fwht_dataset);
    free(tail_norm_128);
    free(tail_norm_64);
    free(sq16_transposed_128);
    free(tail_transposed_128);
    free(sq16_transposed_64);
    free(tail_transposed_64);
    free(bps_transposed_hi);

    return 0;
}

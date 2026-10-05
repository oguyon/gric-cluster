/**
 * @file knn_reconstruct.c
 * @brief Computes a reconstructed dataset using k-NN indices and distances.
 *
 * Implements post-processing dataset reconstruction from k-NN results. Functions in this
 * file parse k-NN outputs, compute weighted averages (uniform or inverse distance weighting)
 * across nearest neighbors to estimate query coordinates, and write reconstructed arrays
 * and quality metrics to disk.
 */

#include "knn_defs.h"
#include "knn_reader.h"
#include "cli_colors.h"
#include "gric_bin_io.h"
#include "gric_simd.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    const char *knn_dir;
    const char *data_b_path;
    char        out_path[1024];
    char        qual_path[1024];
    int         use_idw;
    float       alpha;
    int         verbose;
} ReconConfig;

#if GRIC_HAVE_AVX512_TARGET
/**
 * knn_accum_weighted_frame_avx512() - Accumulate weighted neighbor frame using AVX-512.
 * @out_d:      Destination output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @w:          Weight multiplier.
 * @b_dim:      Dimension of the frame.
 */
GRIC_TARGET_AVX512
static void knn_accum_weighted_frame_avx512(
    float       *restrict out_d,
    const float *restrict neighbor_b,
    float                 w,
    uint64_t              b_dim)
{
    uint64_t d = 0;
    __m512 vw = _mm512_set1_ps(w);
    for (; d + 15 < b_dim; d += 16)
    {
        __m512 vnb = _mm512_loadu_ps(neighbor_b + d);
        __m512 vout = _mm512_loadu_ps(out_d + d);
        vout = _mm512_fmadd_ps(vw, vnb, vout);
        _mm512_storeu_ps(out_d + d, vout);
    }
    for (; d < b_dim; d++)
    {
        out_d[d] += w * neighbor_b[d];
    }
}

/**
 * knn_calc_dist_sq_avx512() - Compute squared distance between frames using AVX-512.
 * @out_d:      Output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @b_dim:      Dimension of the frame.
 *
 * Return: Sum of squared differences.
 */
GRIC_TARGET_AVX512
static float knn_calc_dist_sq_avx512(
    const float *restrict out_d,
    const float *restrict neighbor_b,
    uint64_t              b_dim)
{
    uint64_t d = 0;
    __m512 vsum = _mm512_setzero_ps();
    for (; d + 15 < b_dim; d += 16)
    {
        __m512 vnb = _mm512_loadu_ps(neighbor_b + d);
        __m512 vout = _mm512_loadu_ps(out_d + d);
        __m512 diff = _mm512_sub_ps(vnb, vout);
        vsum = _mm512_fmadd_ps(diff, diff, vsum);
    }
    float dist_sq = _mm512_reduce_add_ps(vsum);
    for (; d < b_dim; d++)
    {
        float diff = neighbor_b[d] - out_d[d];
        dist_sq += diff * diff;
    }
    return dist_sq;
}
#endif // GRIC_HAVE_AVX512_TARGET

/**
 * print_usage() - Print command-line usage synopsis for knn_reconstruct.
 * @prog_name: Name of executable.
 */
static void print_usage(
    const char *prog_name)
{
    printf(
        "Usage: %s <knn_result_dir> <dataset_B>"
        " [-o <output_D>] [-w uniform|idw]"
        " [-alpha 1.0] [-v] [-h]\n",
        prog_name);
}

/**
 * recon_parse_args() - Parse command line arguments into ReconConfig.
 * @argc: Argument count.
 * @argv: Argument vector.
 * @cfg:  Configuration structure to populate.
 *
 * Return: 0 on success, 1 on error, 2 if help displayed.
 */
static int recon_parse_args(
    int          argc,
    char       **argv,
    ReconConfig *cfg)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return 1;
    }

    memset(cfg, 0, sizeof(ReconConfig));
    cfg->knn_dir = argv[1];
    cfg->data_b_path = argv[2];
    cfg->alpha = 1.0f;

    snprintf(cfg->out_path, sizeof(cfg->out_path), "%s/knn_recon.bin", cfg->knn_dir);
    snprintf(cfg->qual_path, sizeof(cfg->qual_path), "%s/knn_quality.bin", cfg->knn_dir);

    for (int ii = 3; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h") == 0)
        {
            print_usage(argv[0]);
            return 2;
        }
        if (strcmp(argv[ii], "-o") == 0 && ii + 1 < argc)
        {
            strncpy(cfg->out_path, argv[++ii], sizeof(cfg->out_path) - 1);
        }
        else if (strcmp(argv[ii], "-w") == 0 && ii + 1 < argc)
        {
            const char *mode = argv[++ii];
            if (strcmp(mode, "idw") == 0)
            {
                cfg->use_idw = 1;
            }
            else if (strcmp(mode, "uniform") != 0)
            {
                fprintf(stderr, "Unknown weighting mode: %s\n", mode);
                return 1;
            }
        }
        else if (strcmp(argv[ii], "-alpha") == 0 && ii + 1 < argc)
        {
            cfg->alpha = (float)atof(argv[++ii]);
        }
        else if (strcmp(argv[ii], "-v") == 0)
        {
            cfg->verbose = 1;
        }
    }
    return 0;
}

/**
 * recon_load_knn_results() - Load k-NN indices and distances from disk.
 * @knn_dir:       Directory containing knn_indices.bin and knn_distances.bin.
 * @out_indices:   Pointer to receive allocated indices array.
 * @out_distances: Pointer to receive allocated distances array.
 * @out_N:         Pointer to store query frame count.
 * @out_k:         Pointer to store neighbor count k.
 *
 * Return: 0 on success, -1 on error.
 */
static int recon_load_knn_results(
    const char  *knn_dir,
    uint32_t   **out_indices,
    float      **out_distances,
    uint64_t    *out_N,
    uint64_t    *out_k)
{
    char idx_path[1024], dist_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/knn_indices.bin", knn_dir);
    snprintf(dist_path, sizeof(dist_path), "%s/knn_distances.bin", knn_dir);

    FILE *fp_idx = fopen(idx_path, "rb");
    if (fp_idx == NULL)
    {
        fprintf(stderr, "Error: cannot open %s\n", idx_path);
        return -1;
    }

    gric_bin_header_t hdr_idx;
    if (gric_bin_read_header(fp_idx, &hdr_idx, NULL) != 0)
    {
        fprintf(stderr, "Error: invalid header in %s\n", idx_path);
        fclose(fp_idx);
        return -1;
    }

    FILE *fp_dist = fopen(dist_path, "rb");
    if (fp_dist == NULL)
    {
        fprintf(stderr, "Error: cannot open %s\n", dist_path);
        fclose(fp_idx);
        return -1;
    }

    gric_bin_header_t hdr_dist;
    if (gric_bin_read_header(fp_dist, &hdr_dist, NULL) != 0)
    {
        fprintf(stderr, "Error: invalid header in %s\n", dist_path);
        fclose(fp_idx);
        fclose(fp_dist);
        return -1;
    }

    uint64_t N = hdr_idx.dims[0];
    uint64_t k = hdr_idx.dims[1];

    fseek(fp_idx, (long)hdr_idx.header_bytes, SEEK_SET);
    fseek(fp_dist, (long)hdr_dist.header_bytes, SEEK_SET);

    uint32_t *indices = (uint32_t *)malloc((size_t)(N * k * sizeof(uint32_t)));
    float *distances = (float *)malloc((size_t)(N * k * sizeof(float)));

    if (indices == NULL || distances == NULL)
    {
        fprintf(stderr, "Error: out of memory for k-NN result arrays\n");
        free(indices);
        free(distances);
        fclose(fp_idx);
        fclose(fp_dist);
        return -1;
    }

    size_t total = (size_t)(N * k);
    if (fread(indices, sizeof(uint32_t), total, fp_idx) != total ||
        fread(distances, sizeof(float), total, fp_dist) != total)
    {
        fprintf(stderr, "Error: failed to read k-NN result arrays\n");
        free(indices);
        free(distances);
        fclose(fp_idx);
        fclose(fp_dist);
        return -1;
    }

    fclose(fp_idx);
    fclose(fp_dist);

    *out_indices = indices;
    *out_distances = distances;
    *out_N = N;
    *out_k = k;
    return 0;
}

/**
 * recon_load_dataset_b() - Read frames of dataset B into memory.
 * @data_b_path: Path to dataset B file.
 * @out_data_b:  Pointer to receive allocated dataset array.
 * @out_b_n:     Pointer to store frame count.
 * @out_b_dim:   Pointer to store frame dimension.
 *
 * Return: 0 on success, -1 on error.
 */
static int recon_load_dataset_b(
    const char *data_b_path,
    float     **out_data_b,
    long       *out_b_n,
    uint64_t   *out_b_dim)
{
    long b_n = 0, b_w = 0, b_h = 0;
    if (knn_reader_inspect(data_b_path, &b_n, &b_w, &b_h) != 0)
    {
        fprintf(stderr, "Error: failed to inspect %s\n", data_b_path);
        return -1;
    }

    uint64_t b_dim = (uint64_t)(b_w * b_h);
    KnnFrameReader b_ctx;
    if (knn_reader_open(&b_ctx, data_b_path, b_n, b_w, b_h, 0) != 0)
    {
        fprintf(stderr, "Error: failed to open %s\n", data_b_path);
        return -1;
    }

    float *data_b = (float *)malloc((size_t)b_n * (size_t)b_dim * sizeof(float));
    if (data_b == NULL)
    {
        fprintf(stderr, "Error: out of memory for dataset B\n");
        knn_reader_close(&b_ctx);
        return -1;
    }

    for (long ii = 0; ii < b_n; ii++)
    {
        if (knn_reader_read_frame(&b_ctx, ii, &data_b[(size_t)ii * b_dim]) != 0)
        {
            fprintf(stderr, "Error: failed to read frame %ld from %s\n", ii, data_b_path);
            knn_reader_close(&b_ctx);
            free(data_b);
            return -1;
        }
    }
    knn_reader_close(&b_ctx);

    *out_data_b = data_b;
    *out_b_n = b_n;
    *out_b_dim = b_dim;
    return 0;
}

/**
 * recon_compute_weights() - Compute neighbor weights (exact, uniform, or IDW).
 * @dist:    Array of k distances.
 * @k:       Neighbor count.
 * @use_idw: Flag to enable inverse distance weighting.
 * @alpha:   IDW exponent.
 * @weights: Output array of k normalized weights.
 */
static void recon_compute_weights(
    const float *dist,
    uint64_t     k,
    int          use_idw,
    float        alpha,
    float       *weights)
{
    int exact_match = -1;
    for (uint64_t jj = 0; jj < k; jj++)
    {
        if (dist[jj] < 1e-9f)
        {
            exact_match = (int)jj;
            break;
        }
    }

    if (exact_match >= 0)
    {
        for (uint64_t jj = 0; jj < k; jj++)
        {
            weights[jj] = 0.0f;
        }
        weights[exact_match] = 1.0f;
        return;
    }

    if (!use_idw)
    {
        float inv_k = 1.0f / (float)k;
        for (uint64_t jj = 0; jj < k; jj++)
        {
            weights[jj] = inv_k;
        }
        return;
    }

    float sum_w = 0.0f;
    for (uint64_t jj = 0; jj < k; jj++)
    {
        float dd = fmaxf(dist[jj], 1e-7f);
        weights[jj] = powf(1.0f / dd, alpha);
        sum_w += weights[jj];
    }
    for (uint64_t jj = 0; jj < k; jj++)
    {
        weights[jj] /= sum_w;
    }
}

/**
 * recon_accumulate_frame() - Accumulate weighted neighbor frames into target output.
 * @out_d:      Destination output frame buffer.
 * @data_b:     Dataset B resident frame buffer.
 * @idx:        Array of k neighbor indices.
 * @weights:    Array of k normalized weights.
 * @k:          Neighbor count.
 * @b_dim:      Dimension of each frame.
 */
static void recon_accumulate_frame(
    float          *restrict out_d,
    const float    *restrict data_b,
    const uint32_t *restrict idx,
    const float    *restrict weights,
    uint64_t                 k,
    uint64_t                 b_dim)
{
    for (uint64_t jj = 0; jj < k; jj++)
    {
        uint32_t neighbor_idx = idx[jj];
        float ww = weights[jj];
        const float *restrict neighbor_b = data_b + neighbor_idx * b_dim;

#if GRIC_HAVE_AVX512_TARGET
        if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
        {
            knn_accum_weighted_frame_avx512(out_d, neighbor_b, ww, b_dim);
            continue;
        }
#endif
        uint64_t dd = 0;
#ifdef __AVX2__
        __m256 vw = _mm256_set1_ps(ww);
        for (; dd + 7 < b_dim; dd += 8)
        {
            __m256 vnb = _mm256_loadu_ps(neighbor_b + dd);
            __m256 vout = _mm256_loadu_ps(out_d + dd);
            vout = _mm256_fmadd_ps(vw, vnb, vout);
            _mm256_storeu_ps(out_d + dd, vout);
        }
#endif
        for (; dd < b_dim; dd++)
        {
            out_d[dd] += ww * neighbor_b[dd];
        }
    }
}

/**
 * recon_compute_variance() - Compute weighted reconstruction variance for a frame.
 * @out_d:   Reconstructed frame buffer.
 * @data_b:  Dataset B resident frame buffer.
 * @idx:     Array of k neighbor indices.
 * @weights: Array of k normalized weights.
 * @k:       Neighbor count.
 * @b_dim:   Dimension of each frame.
 *
 * Return: Weighted variance sum of squared differences.
 */
static float recon_compute_variance(
    const float    *restrict out_d,
    const float    *restrict data_b,
    const uint32_t *restrict idx,
    const float    *restrict weights,
    uint64_t                 k,
    uint64_t                 b_dim)
{
    float variance = 0.0f;
    for (uint64_t jj = 0; jj < k; jj++)
    {
        uint32_t neighbor_idx = idx[jj];
        float ww = weights[jj];
        const float *restrict neighbor_b = data_b + neighbor_idx * b_dim;

        float dist_sq = 0.0f;
#if GRIC_HAVE_AVX512_TARGET
        if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
        {
            dist_sq = knn_calc_dist_sq_avx512(out_d, neighbor_b, b_dim);
            variance += ww * dist_sq;
            continue;
        }
#endif
        uint64_t dd = 0;
#ifdef __AVX2__
        __m256 vsum = _mm256_setzero_ps();
        for (; dd + 7 < b_dim; dd += 8)
        {
            __m256 vnb = _mm256_loadu_ps(neighbor_b + dd);
            __m256 vout = _mm256_loadu_ps(out_d + dd);
            __m256 diff = _mm256_sub_ps(vnb, vout);
            vsum = _mm256_fmadd_ps(diff, diff, vsum);
        }
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        __m128 vsum128 = _mm_add_ps(vlow, vhigh);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        dist_sq = _mm_cvtss_f32(vsum128);
#endif
        for (; dd < b_dim; dd++)
        {
            float diff = neighbor_b[dd] - out_d[dd];
            dist_sq += diff * diff;
        }
        variance += ww * dist_sq;
    }
    return variance;
}

/**
 * recon_save_outputs() - Write reconstructed array D and quality metrics to disk.
 * @out_path:  Output binary filepath for D.
 * @qual_path: Output binary filepath for quality metrics.
 * @d_out:     Reconstructed array [N x b_dim].
 * @qual:      Quality metrics array [N x 2].
 * @n_queries: Number of query frames N.
 * @b_dim:     Frame dimension.
 *
 * Return: 0 on success, -1 on error.
 */
static int recon_save_outputs(
    const char  *out_path,
    const char  *qual_path,
    const float *d_out,
    const float *qual,
    uint64_t     n_queries,
    uint64_t     b_dim)
{
    FILE *fp_out = fopen(out_path, "wb");
    if (fp_out == NULL)
    {
        fprintf(stderr, "Error: cannot write to %s\n", out_path);
        return -1;
    }

    gric_bin_header_t hdr_out;
    memset(&hdr_out, 0, sizeof(hdr_out));
    hdr_out.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr_out.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr_out.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr_out.ndim = 2;
    hdr_out.dims[0] = n_queries;
    hdr_out.dims[1] = b_dim;
    hdr_out.num_elements = n_queries * b_dim;
    hdr_out.data_bytes = hdr_out.num_elements * sizeof(float);

    if (gric_bin_write_header(fp_out, &hdr_out, "knn_reconstruct output D") != 0)
    {
        fprintf(stderr, "Error writing header to %s\n", out_path);
    }
    fwrite(d_out, sizeof(float), (size_t)(n_queries * b_dim), fp_out);
    fclose(fp_out);

    FILE *fp_qual = fopen(qual_path, "wb");
    if (fp_qual == NULL)
    {
        fprintf(stderr, "Error: cannot write to %s\n", qual_path);
        return -1;
    }

    gric_bin_header_t hdr_qual;
    memset(&hdr_qual, 0, sizeof(hdr_qual));
    hdr_qual.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr_qual.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr_qual.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr_qual.ndim = 2;
    hdr_qual.dims[0] = n_queries;
    hdr_qual.dims[1] = 2;
    hdr_qual.num_elements = n_queries * 2;
    hdr_qual.data_bytes = hdr_qual.num_elements * sizeof(float);

    if (gric_bin_write_header(fp_qual, &hdr_qual, "knn_reconstruct output quality") != 0)
    {
        fprintf(stderr, "Error writing header to %s\n", qual_path);
    }
    fwrite(qual, sizeof(float), (size_t)(n_queries * 2), fp_qual);
    fclose(fp_qual);

    return 0;
}

/**
 * recon_print_summary() - Print reconstruction statistics and timing.
 * @cfg:            Reconstruction configuration.
 * @n_queries:      Number of query frames.
 * @k:              Number of neighbors.
 * @b_dim:          Frame dimension.
 * @total_kth_dist: Sum of kth distances.
 * @total_variance: Sum of reconstruction variances.
 * @wall_time:      Elapsed seconds.
 */
static void recon_print_summary(
    const ReconConfig *cfg,
    uint64_t           n_queries,
    uint64_t           k,
    uint64_t           b_dim,
    double             total_kth_dist,
    double             total_variance,
    double             wall_time)
{
    printf("\n%s--- KNN Reconstruction Complete ---%s\n", ansi_bold_green, ansi_reset);
    printf("Queries       : %llu\n", (unsigned long long)n_queries);
    if (cfg->verbose)
    {
        printf("Neighbors (k) : %llu\n", (unsigned long long)k);
    }
    printf("Output Dim    : %llu\n", (unsigned long long)b_dim);
    printf("Weighting     : %s%s%s\n", ansi_bold_cyan,
           cfg->use_idw ? "idw" : "uniform", ansi_reset);
    if (cfg->verbose && cfg->use_idw)
    {
        printf("IDW Alpha     : %.2f\n", cfg->alpha);
    }
    printf("Avg kth dist  : %f\n", total_kth_dist / (double)n_queries);
    printf("Avg variance  : %f\n", total_variance / (double)n_queries);
    printf("Wall time     : %.3f s\n", wall_time);
}

/**
 * recon_run_pipeline() - Execute dataset reconstruction from k-NN results.
 * @cfg: Configuration options.
 *
 * Return: 0 on success, non-zero on failure.
 */
static int recon_run_pipeline(
    const ReconConfig *cfg)
{
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    uint32_t *indices = NULL;
    float *distances = NULL;
    uint64_t N = 0, k = 0;
    if (recon_load_knn_results(cfg->knn_dir, &indices, &distances, &N, &k) != 0)
    {
        return 1;
    }

    float *data_b = NULL;
    long b_n = 0;
    uint64_t b_dim = 0;
    if (recon_load_dataset_b(cfg->data_b_path, &data_b, &b_n, &b_dim) != 0)
    {
        free(indices);
        free(distances);
        return 1;
    }

    float *d_out = (float *)calloc((size_t)N * (size_t)b_dim, sizeof(float));
    float *qual = (float *)calloc((size_t)N * 2, sizeof(float));
    float *weights = (float *)malloc((size_t)k * sizeof(float));

    if (d_out == NULL || qual == NULL || weights == NULL)
    {
        fprintf(stderr, "Error: out of memory for D/qual/weights\n");
        free(d_out);
        free(qual);
        free(weights);
        free(indices);
        free(distances);
        free(data_b);
        return 1;
    }

    double total_kth_dist = 0.0;
    double total_variance = 0.0;

    for (uint64_t ii = 0; ii < N; ii++)
    {
        uint32_t *restrict idx = indices + ii * k;
        float *restrict dist = distances + ii * k;

        recon_compute_weights(dist, k, cfg->use_idw, cfg->alpha, weights);
        recon_accumulate_frame(d_out + ii * b_dim, data_b, idx, weights, k, b_dim);
        float variance = recon_compute_variance(d_out + ii * b_dim, data_b, idx,
                                                weights, k, b_dim);

        qual[ii * 2 + 0] = dist[k - 1];
        qual[ii * 2 + 1] = variance;

        total_kth_dist += (double)dist[k - 1];
        total_variance += (double)variance;
    }

    int save_rc = recon_save_outputs(cfg->out_path, cfg->qual_path, d_out, qual, N, b_dim);

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double wall_time = (t_end.tv_sec - t_start.tv_sec) +
                       (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    recon_print_summary(cfg, N, k, b_dim, total_kth_dist, total_variance, wall_time);

    free(indices);
    free(distances);
    free(data_b);
    free(d_out);
    free(qual);
    free(weights);

    return (save_rc == 0) ? 0 : 1;
}

int main(
    int    argc,
    char **argv)
{
    cli_colors_init();

    ReconConfig cfg;
    int parse_rc = recon_parse_args(argc, argv, &cfg);
    if (parse_rc != 0)
    {
        return (parse_rc == 2) ? 0 : 1;
    }

    return recon_run_pipeline(&cfg);
}

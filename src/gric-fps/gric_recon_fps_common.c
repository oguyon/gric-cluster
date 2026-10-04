/**
 * @file gric_recon_fps_common.c
 * @brief Common adapter implementation for GRIC dataset reconstruction Milk streaming integration.
 */

#define _POSIX_C_SOURCE 200809L
#include "gric_recon_fps_common.h"
#include "gric_recon_fps_params.h"
#include "knn_defs.h"
#include "knn_reader.h"
#include "cluster_shm.h"
#include "shared/sys/gric_rss.h"
#include "shared/sys/gric_simd.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

char     fps_recon_in_name[FUNCTION_PARAMETER_STRMAXLEN]     = "";
char     fps_recon_out_name[FUNCTION_PARAMETER_STRMAXLEN]    = "recon_D";
uint64_t fps_recon_cnt2sync                                  = 0;
uint64_t fps_recon_allow_frame_drop                          = 0;
char     fps_recon_target_data[FUNCTION_PARAMETER_STRMAXLEN] = "";
char     fps_recon_weight_mode[FUNCTION_PARAMETER_STRMAXLEN] = "uniform";
double   fps_recon_alpha                                     = 1.0;
uint32_t fps_recon_k                                         = 10;
uint64_t fps_recon_max_frames                                = 0;
char     fps_recon_out_file[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static FILE *recon_out_fp                                    = NULL;
static int   recon_out_is_binary                              = 0;
static char  recon_out_buf[65536];

uint64_t fps_recon_status_frames                             = 0;
double   fps_recon_status_latency_us                         = 0.0;
double   fps_recon_status_fps                                = 0.0;
double   fps_recon_status_variance                           = 0.0;
double   fps_recon_status_k_eff                              = 0.0;
int64_t  fps_recon_status_stream_lag                         = 0;
double   fps_recon_status_memory_rss_mb                      = 0.0;
char     fps_recon_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN] = "";

static GricClusterShmStatus *status_shm_ptr        = NULL;
static char                  active_shm_path[PATH_MAX] = "";
static struct timespec       session_start_time;
static struct timespec       prev_fps_calc_time;
static uint64_t              prev_fps_calc_frames  = 0;

/* Resident Target Dataset B */
static float                *target_b_data         = NULL;
static uint64_t              target_b_samples      = 0;
static uint64_t              target_b_dim          = 0;
static int                   engine_initialized    = 0;

/* Pre-allocated per-frame scratch buffers */
static float                *scratch_out_d         = NULL;
static float                *scratch_weights       = NULL;
static uint32_t              scratch_max_k         = 0;

#if GRIC_HAVE_AVX512_TARGET
/**
 * recon_accum_weighted_frame_avx512() - Accumulate weighted neighbor vector using AVX-512.
 * @out_d:      Destination output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @w:          Weight multiplier.
 * @b_dim:      Dimension of the frame.
 */
GRIC_TARGET_AVX512
static void recon_accum_weighted_frame_avx512(
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
 * recon_calc_dist_sq_avx512() - Compute squared distance between frames using AVX-512.
 * @out_d:      Output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @b_dim:      Dimension of the frame.
 *
 * Return: Sum of squared differences.
 */
GRIC_TARGET_AVX512
static float recon_calc_dist_sq_avx512(
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

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
/**
 * recon_accum_weighted_frame_avx2() - Accumulate weighted neighbor vector using AVX2.
 * @out_d:      Destination output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @w:          Weight multiplier.
 * @b_dim:      Dimension of the frame.
 */
GRIC_TARGET_AVX2
static void recon_accum_weighted_frame_avx2(
    float       *restrict out_d,
    const float *restrict neighbor_b,
    float                 w,
    uint64_t              b_dim)
{
    uint64_t d = 0;
    __m256 vw = _mm256_set1_ps(w);
    for (; d + 7 < b_dim; d += 8)
    {
        __m256 vnb = _mm256_loadu_ps(neighbor_b + d);
        __m256 vout = _mm256_loadu_ps(out_d + d);
        vout = _mm256_fmadd_ps(vw, vnb, vout);
        _mm256_storeu_ps(out_d + d, vout);
    }
    for (; d < b_dim; d++)
    {
        out_d[d] += w * neighbor_b[d];
    }
}

/**
 * recon_calc_dist_sq_avx2() - Compute squared distance between frames using AVX2.
 * @out_d:      Output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @b_dim:      Dimension of the frame.
 *
 * Return: Sum of squared differences.
 */
GRIC_TARGET_AVX2
static float recon_calc_dist_sq_avx2(
    const float *restrict out_d,
    const float *restrict neighbor_b,
    uint64_t              b_dim)
{
    uint64_t d = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; d + 7 < b_dim; d += 8)
    {
        __m256 vnb = _mm256_loadu_ps(neighbor_b + d);
        __m256 vout = _mm256_loadu_ps(out_d + d);
        __m256 diff = _mm256_sub_ps(vnb, vout);
        vsum = _mm256_fmadd_ps(diff, diff, vsum);
    }
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    __m128 vsum128 = _mm_add_ps(vlow, vhigh);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    vsum128 = _mm_hadd_ps(vsum128, vsum128);
    float dist_sq = _mm_cvtss_f32(vsum128);
    for (; d < b_dim; d++)
    {
        float diff = neighbor_b[d] - out_d[d];
        dist_sq += diff * diff;
    }
    return dist_sq;
}
#endif // AVX2 target

/**
 * recon_accum_weighted_frame() - Vectorized accumulation dispatcher.
 * @out_d:      Destination output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @w:          Weight multiplier.
 * @b_dim:      Dimension of the frame.
 */
static void recon_accum_weighted_frame(
    float       *restrict out_d,
    const float *restrict neighbor_b,
    float                 w,
    uint64_t              b_dim)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        recon_accum_weighted_frame_avx512(out_d, neighbor_b, w, b_dim);
        return;
    }
#endif
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        recon_accum_weighted_frame_avx2(out_d, neighbor_b, w, b_dim);
        return;
    }
#endif
    for (uint64_t d = 0; d < b_dim; d++)
    {
        out_d[d] += w * neighbor_b[d];
    }
}

/**
 * recon_calc_dist_sq() - Vectorized squared distance dispatcher.
 * @out_d:      Output frame buffer [b_dim].
 * @neighbor_b: Neighbor frame buffer [b_dim].
 * @b_dim:      Dimension of the frame.
 *
 * Return: Sum of squared differences.
 */
static float recon_calc_dist_sq(
    const float *restrict out_d,
    const float *restrict neighbor_b,
    uint64_t              b_dim)
{
#if GRIC_HAVE_AVX512_TARGET
    if (gric_get_simd_level() >= GRIC_SIMD_AVX512)
    {
        return recon_calc_dist_sq_avx512(out_d, neighbor_b, b_dim);
    }
#endif
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
    if (gric_get_simd_level() >= GRIC_SIMD_AVX2)
    {
        return recon_calc_dist_sq_avx2(out_d, neighbor_b, b_dim);
    }
#endif
    float dist_sq = 0.0f;
    for (uint64_t d = 0; d < b_dim; d++)
    {
        float diff = neighbor_b[d] - out_d[d];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

/**
 * gric_recon_fps_custom_conf_check() - Validate runtime configuration parameters.
 *
 * Return: 0 on valid configuration, non-zero if invalid.
 */
errno_t gric_recon_fps_custom_conf_check(void)
{
    if (fps_recon_in_name[0] == '\0')
    {
        return 1;
    }
    if (fps_recon_target_data[0] == '\0')
    {
        return 1;
    }
    return 0;
}

/**
 * gric_recon_fps_init_engine() - Ingest and stage resident target dataset B.
 * @out_b_dim: Output pointer returning the coordinate dimension of dataset B.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_recon_fps_init_engine(
    uint32_t *out_b_dim)
{
    if (engine_initialized)
    {
        if (out_b_dim)
        {
            *out_b_dim = (uint32_t)target_b_dim;
        }
        return 0;
    }

    if (fps_recon_target_data[0] == '\0')
    {
        fprintf(stderr, "Error: Target dataset B not specified.\n");
        return 1;
    }

    long b_n = 0;
    long b_w = 0;
    long b_h = 0;
    if (knn_reader_inspect(fps_recon_target_data, &b_n, &b_w, &b_h) != 0)
    {
        fprintf(stderr, "Error: Failed to inspect target dataset %s\n", fps_recon_target_data);
        return 1;
    }

    uint64_t b_dim = (uint64_t)(b_w * b_h);
    target_b_samples = (uint64_t)b_n;
    target_b_dim = b_dim;

    KnnFrameReader b_ctx;
    if (knn_reader_open(&b_ctx, fps_recon_target_data, b_n, b_w, b_h, 0) != 0)
    {
        fprintf(stderr, "Error: Failed to open target dataset %s\n", fps_recon_target_data);
        return 1;
    }

    target_b_data = malloc((size_t)(target_b_samples * target_b_dim * sizeof(float)));
    if (!target_b_data)
    {
        fprintf(stderr, "Error: Out of memory staging target dataset B\n");
        knn_reader_close(&b_ctx);
        return 1;
    }

    for (long ii = 0; ii < b_n; ii++)
    {
        if (knn_reader_read_frame(&b_ctx, ii, &target_b_data[ii * b_dim]) != 0)
        {
            fprintf(stderr, "Error: Failed reading frame %ld from %s\n",
                    ii, fps_recon_target_data);
            knn_reader_close(&b_ctx);
            free(target_b_data);
            target_b_data = NULL;
            return 1;
        }
    }
    knn_reader_close(&b_ctx);

    scratch_out_d = malloc((size_t)(target_b_dim * sizeof(float)));
    scratch_max_k = 4096;
    scratch_weights = malloc((size_t)(scratch_max_k * sizeof(float)));

    if (!scratch_out_d || !scratch_weights)
    {
        fprintf(stderr, "Error: Out of memory for reconstruction scratch buffers\n");
        if (scratch_out_d) { free(scratch_out_d); scratch_out_d = NULL; }
        if (scratch_weights) { free(scratch_weights); scratch_weights = NULL; }
        free(target_b_data);
        target_b_data = NULL;
        return 1;
    }

    if (out_b_dim)
    {
        *out_b_dim = (uint32_t)target_b_dim;
    }
    engine_initialized = 1;
    return 0;
}

/**
 * gric_recon_fps_cleanup_engine() - Free resident target dataset B and buffers.
 */
void gric_recon_fps_cleanup_engine(void)
{
    if (target_b_data)
    {
        free(target_b_data);
        target_b_data = NULL;
    }
    if (scratch_out_d)
    {
        free(scratch_out_d);
        scratch_out_d = NULL;
    }
    if (scratch_weights)
    {
        free(scratch_weights);
        scratch_weights = NULL;
    }
    target_b_samples = 0;
    target_b_dim = 0;
    scratch_max_k = 0;
    engine_initialized = 0;
}

/**
 * gric_recon_fps_init_output_streams() - Create output ImageStreamIO stream D.
 * @b_dim:     Coordinate dimension of dataset B.
 * @out_recon: Stream structure pointer.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_recon_fps_init_output_streams(
    uint32_t  b_dim,
    IMAGE    *out_recon)
{
    if (!out_recon || fps_recon_out_name[0] == '\0')
    {
        return 1;
    }

    uint32_t imsize[2] = { b_dim, 1 };
    if (ImageStreamIO_createIm_gpu(
            out_recon, fps_recon_out_name, 2, imsize,
            _DATATYPE_FLOAT, -1, 1, 10,
            IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER) != 0)
    {
        fprintf(stderr, "Error: Failed to create output stream %s\n", fps_recon_out_name);
        return 1;
    }

    if (fps_recon_out_file[0] != '\0')
    {
        size_t flen = strlen(fps_recon_out_file);
        recon_out_is_binary = (flen > 4 && strcmp(fps_recon_out_file + flen - 4, ".bin") == 0);
        recon_out_fp = fopen(fps_recon_out_file, recon_out_is_binary ? "wb" : "w");
        if (!recon_out_fp)
        {
            fprintf(stderr, "Warning: Unable to open output file '%s' for writing\n",
                    fps_recon_out_file);
        }
        else
        {
            setvbuf(recon_out_fp, recon_out_buf, _IOFBF, sizeof(recon_out_buf));
        }
    }

    return 0;
}

/**
 * gric_recon_fps_close_output_streams() - Safely disconnect output stream D.
 * @out_recon: Output stream handle.
 */
void gric_recon_fps_close_output_streams(
    IMAGE *out_recon)
{
    if (recon_out_fp != NULL)
    {
        fflush(recon_out_fp);
        fclose(recon_out_fp);
        recon_out_fp = NULL;
    }

    if (out_recon && out_recon->used == 1)
    {
        ImageStreamIO_closeIm(out_recon);
    }
}

/**
 * gric_recon_fps_process_frame() - Reconstruct one frame from k-NN matches.
 * @raw_matches: Pointer to input matches array [k x 2].
 * @datatype:    Data type code of input stream.
 * @match_k:     Number of nearest neighbors in input stream.
 * @frame_index: Frame count (cnt0).
 * @frame_time:  Arrival timestamp.
 * @latency_us:  Algorithm latency.
 * @out_recon:   Output reconstructed stream D handle.
 *
 * Return: 0 on success, non-zero on error.
 */
errno_t gric_recon_fps_process_frame(
    const void      *raw_matches,
    int              datatype,
    uint32_t         match_k,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_recon)
{
    (void)frame_index;
    (void)frame_time;
    (void)latency_us;

    if (!engine_initialized || !raw_matches || match_k == 0)
    {
        return 1;
    }

    if (datatype != _DATATYPE_FLOAT)
    {
        fprintf(stderr, "Error: Match stream must be float32\n");
        return 1;
    }

    uint32_t k_used = (fps_recon_k > 0 && fps_recon_k < match_k) ? fps_recon_k : match_k;
    if (k_used > scratch_max_k)
    {
        k_used = scratch_max_k;
    }

    const float *f_matches = (const float *)raw_matches;
    const float *f_indices = f_matches;
    const float *f_dists   = f_matches + match_k;

    /* 1. Calculate weights */
    int exact_match = -1;
    for (uint32_t j = 0; j < k_used; j++)
    {
        if (f_dists[j] < 1e-9f)
        {
            exact_match = (int)j;
            break;
        }
    }

    if (exact_match >= 0)
    {
        for (uint32_t j = 0; j < k_used; j++)
        {
            scratch_weights[j] = (j == (uint32_t)exact_match) ? 1.0f : 0.0f;
        }
    }
    else if (strcmp(fps_recon_weight_mode, "idw") == 0)
    {
        float sum_w = 0.0f;
        for (uint32_t j = 0; j < k_used; j++)
        {
            float d = fmaxf(f_dists[j], 1e-7f);
            scratch_weights[j] = powf(1.0f / d, (float)fps_recon_alpha);
            sum_w += scratch_weights[j];
        }
        float inv_sum = (sum_w > 0.0f) ? (1.0f / sum_w) : 0.0f;
        for (uint32_t j = 0; j < k_used; j++)
        {
            scratch_weights[j] *= inv_sum;
        }
    }
    else
    {
        float uniform_w = 1.0f / (float)k_used;
        for (uint32_t j = 0; j < k_used; j++)
        {
            scratch_weights[j] = uniform_w;
        }
    }

    /* 2. Vectorized accumulation: D = sum_j (w_j * B[idx_j]) */
    memset(scratch_out_d, 0, (size_t)(target_b_dim * sizeof(float)));

    for (uint32_t j = 0; j < k_used; j++)
    {
        uint32_t idx = (uint32_t)f_indices[j];
        if (idx < target_b_samples)
        {
            const float *neighbor_b = target_b_data + (idx * target_b_dim);
            recon_accum_weighted_frame(scratch_out_d, neighbor_b,
                                       scratch_weights[j], target_b_dim);
        }
    }

    /* 3. Compute target dispersion / variance */
    float variance = 0.0f;
    for (uint32_t j = 0; j < k_used; j++)
    {
        uint32_t idx = (uint32_t)f_indices[j];
        if (idx < target_b_samples)
        {
            const float *neighbor_b = target_b_data + (idx * target_b_dim);
            float dist_sq = recon_calc_dist_sq(scratch_out_d, neighbor_b, target_b_dim);
            variance += scratch_weights[j] * dist_sq;
        }
    }
    fps_recon_status_variance = (double)variance;

    /* 4. Compute effective number of points averaged: k_eff = 1 / sum(w_j^2) */
    float sum_w_sq = 0.0f;
    for (uint32_t j = 0; j < k_used; j++)
    {
        sum_w_sq += scratch_weights[j] * scratch_weights[j];
    }
    double frame_k_eff = (sum_w_sq > 1e-12f) ? (1.0 / (double)sum_w_sq) : 1.0;
    fps_recon_status_k_eff =
        (fps_recon_status_k_eff * (double)fps_recon_status_frames + frame_k_eff) /
        (double)(fps_recon_status_frames + 1);

    /* 5. Publish reconstructed frame D to ImageStreamIO stream */
    if (out_recon && out_recon->used == 1)
    {
        out_recon->md[0].write = 1;
        memcpy(out_recon->array.raw, scratch_out_d, (size_t)(target_b_dim * sizeof(float)));
    }

    /* 5. Optionally write reconstructed vector to output file */
    if (recon_out_fp != NULL)
    {
        if (recon_out_is_binary)
        {
            fwrite(scratch_out_d, sizeof(float), (size_t)target_b_dim, recon_out_fp);
        }
        else
        {
            for (uint64_t d = 0; d < target_b_dim; d++)
            {
                fprintf(recon_out_fp, (d == target_b_dim - 1) ? "%.6f\n" : "%.6f ",
                        (double)scratch_out_d[d]);
            }
        }
        if ((fps_recon_status_frames & 63) == 0)
        {
            fflush(recon_out_fp);
        }
    }

    fps_recon_status_frames++;
    return 0;
}

/**
 * gric_recon_fps_status_init() - Map shared-memory bridge status file.
 * @instance_name:     FPS instance name.
 * @custom_status_path: Optional explicit status file path.
 */
void gric_recon_fps_status_init(
    const char *instance_name,
    const char *custom_status_path)
{
    clock_gettime(CLOCK_MONOTONIC, &session_start_time);
    clock_gettime(CLOCK_MONOTONIC, &prev_fps_calc_time);
    prev_fps_calc_frames = 0;

    if (custom_status_path && custom_status_path[0] != '\0')
    {
        strncpy(active_shm_path, custom_status_path, sizeof(active_shm_path) - 1);
        active_shm_path[sizeof(active_shm_path) - 1] = '\0';
    }
    else
    {
        snprintf(active_shm_path, sizeof(active_shm_path), "/dev/shm/gric_recon_status_%s.shm",
                 (instance_name && instance_name[0] != '\0') ? instance_name : "default");
    }

    int fd = open(active_shm_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
        return;
    }

    if (ftruncate(fd, (off_t)sizeof(GricClusterShmStatus)) != 0)
    {
        close(fd);
        return;
    }

    void *mapped = mmap(NULL, sizeof(GricClusterShmStatus),
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED)
    {
        status_shm_ptr = NULL;
        return;
    }

    status_shm_ptr = (GricClusterShmStatus *)mapped;
    memset(status_shm_ptr, 0, sizeof(GricClusterShmStatus));
    status_shm_ptr->magic = GRIC_SHM_MAGIC;
    status_shm_ptr->version = GRIC_SHM_VERSION;
    status_shm_ptr->pid = (uint32_t)getpid();
    status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
    status_shm_ptr->total_frames = fps_recon_max_frames;
    status_shm_ptr->num_clusters = fps_recon_k;
    strncpy(status_shm_ptr->input_source, fps_recon_in_name,
            sizeof(status_shm_ptr->input_source) - 1);

    if (getcwd(status_shm_ptr->config_cwd, sizeof(status_shm_ptr->config_cwd) - 1) == NULL)
    {
        status_shm_ptr->config_cwd[0] = '\0';
    }
}

/**
 * gric_recon_fps_status_update() - Refresh real-time status parameters and bridge SHM.
 * @frame_index: Frame counter (cnt0).
 * @latency_us:  Algorithm execution latency in microseconds.
 * @stream_lag:  Input stream frame lag.
 * @write_slice: Current input ring buffer write slice.
 * @read_slice:  Current input ring buffer read slice.
 */
void gric_recon_fps_status_update(
    uint64_t frame_index,
    double   latency_us,
    int64_t  stream_lag,
    long     write_slice,
    long     read_slice)
{
    (void)frame_index;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    fps_recon_status_latency_us = latency_us;
    fps_recon_status_stream_lag = stream_lag;
    fps_recon_status_memory_rss_mb = gric_rss_mb_sampled();

    double elapsed_fps = (now.tv_sec - prev_fps_calc_time.tv_sec) +
                         (now.tv_nsec - prev_fps_calc_time.tv_nsec) * 1e-9;
    if (elapsed_fps >= 0.5)
    {
        uint64_t frames_delta = fps_recon_status_frames - prev_fps_calc_frames;
        fps_recon_status_fps = (double)frames_delta / elapsed_fps;
        prev_fps_calc_time = now;
        prev_fps_calc_frames = fps_recon_status_frames;
    }

    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
        status_shm_ptr->total_frames_processed = fps_recon_status_frames;
        status_shm_ptr->num_clusters = fps_recon_k;
        status_shm_ptr->elapsed_ms = (now.tv_sec - session_start_time.tv_sec) * 1000.0 +
                                     (now.tv_nsec - session_start_time.tv_nsec) / 1000000.0;
        status_shm_ptr->stream_read_slice = read_slice;
        status_shm_ptr->stream_write_slice = write_slice;
        status_shm_ptr->stream_lag = stream_lag;
        status_shm_ptr->last_assignment_dist = fps_recon_status_variance;
        status_shm_ptr->memory_rss_kb = (uint64_t)(fps_recon_status_memory_rss_mb * 1024.0);
        status_shm_ptr->total_frames = fps_recon_max_frames;

        struct timespec real_now;
        clock_gettime(CLOCK_REALTIME, &real_now);
        status_shm_ptr->last_update_time = (uint64_t)real_now.tv_sec * 1000000000ULL +
                                           (uint64_t)real_now.tv_nsec;
    }
}

/**
 * gric_recon_fps_status_close() - Finalize and unmap the status shared memory bridge.
 * @unlink_shm: Non-zero to remove the file from /dev/shm.
 */
void gric_recon_fps_status_close(
    int unlink_shm)
{
    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = (unlink_shm == 0) ? GRIC_STATUS_SUCCESS :
                                                           GRIC_STATUS_ERROR;
        munmap(status_shm_ptr, sizeof(GricClusterShmStatus));
        status_shm_ptr = NULL;
    }
    if (unlink_shm && active_shm_path[0] != '\0')
    {
        unlink(active_shm_path);
        active_shm_path[0] = '\0';
    }
}

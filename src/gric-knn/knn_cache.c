/**
 * @file knn_cache.c
 * @brief Memory caching and scalar quantization sidecar generation/loading for k-NN.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_cache.h"
#include "knn_reader.h"
#include "scalar_quant.h"
#include "rabit_quant.h"
#include "gric_bin_io.h"
#include "gric_hash.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * knn_calibrate_dataset_minmax() - Scan dataset frames to find global minimum and maximum.
 * @reader:     Opened KnnFrameReader.
 * @frame_buf:  Pre-allocated temporary frame buffer.
 * @num_frames: Total number of frames in dataset.
 * @dim:        Number of elements per frame.
 * @is_double:  Non-zero if double precision, zero if single precision float.
 * @out_min:    Destination pointer for global minimum.
 * @out_max:    Destination pointer for global maximum.
 *
 * Return: 0 on success, -1 on failure.
 */
static int knn_calibrate_dataset_minmax(
    KnnFrameReader *reader,
    void           *frame_buf,
    long            num_frames,
    long            dim,
    int             is_double,
    float          *out_min,
    float          *out_max)
{
    float global_min = 1e30f;
    float global_max = -1e30f;

    for (long i = 0; i < num_frames; i++)
    {
        if (knn_reader_read_frame(reader, i, frame_buf) == 0)
        {
            if (is_double)
            {
                const double *da = (const double *)frame_buf;
                for (long d = 0; d < dim; d++)
                {
                    float v = (float)da[d];
                    if (v < global_min)
                    {
                        global_min = v;
                    }
                    if (v > global_max)
                    {
                        global_max = v;
                    }
                }
            }
            else
            {
                const float *fa = (const float *)frame_buf;
                for (long d = 0; d < dim; d++)
                {
                    float v = fa[d];
                    if (v < global_min)
                    {
                        global_min = v;
                    }
                    if (v > global_max)
                    {
                        global_max = v;
                    }
                }
            }
        }
    } // for (long i = 0; i < num_frames; i++)

    *out_min = global_min;
    *out_max = global_max;
    return 0;
}

/**
 * knn_quantize_cluster_anchors_sq8() - Pre-quantize anchor vectors into SQ8 buffer.
 * @model: Pointer to initialized KnnModel.
 */
static void knn_quantize_cluster_anchors_sq8(
    KnnModel *model)
{
    if (model->clusters == NULL || model->num_clusters <= 0)
    {
        return;
    }

    long dim = model->frame_elements;
    size_t anchor_elems = (size_t)model->num_clusters * (size_t)dim;
    model->anchor_sq8_buffer = (uint8_t *)malloc(anchor_elems * sizeof(uint8_t));
    if (model->anchor_sq8_buffer == NULL)
    {
        return;
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        uint8_t *dst = model->anchor_sq8_buffer + (size_t)c * (size_t)dim;
        if (model->is_double)
        {
            sq8_quantize_double(
                (const double *)model->clusters[c].anchor_data, dst, &model->sq8_params
            );
        }
        else
        {
            sq8_quantize_float(
                (const float *)model->clusters[c].anchor_data, dst, &model->sq8_params
            );
        }
    }
}

/**
 * knn_quantize_cluster_anchors_sq16() - Pre-quantize anchor vectors into SQ16 buffer.
 * @model: Pointer to initialized KnnModel.
 */
static void knn_quantize_cluster_anchors_sq16(
    KnnModel *model)
{
    if (model->clusters == NULL || model->num_clusters <= 0)
    {
        return;
    }

    long dim = model->frame_elements;
    size_t anchor_elems = (size_t)model->num_clusters * (size_t)dim;
    model->anchor_sq16_buffer = (int16_t *)malloc(anchor_elems * sizeof(int16_t));
    if (model->anchor_sq16_buffer == NULL)
    {
        return;
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        int16_t *dst = model->anchor_sq16_buffer + (size_t)c * (size_t)dim;
        if (model->is_double)
        {
            sq16_quantize_double(
                (const double *)model->clusters[c].anchor_data, dst, &model->sq16_params
            );
        }
        else
        {
            sq16_quantize_float(
                (const float *)model->clusters[c].anchor_data, dst, &model->sq16_params
            );
        }
    }
}

static uint64_t knn_hash_bytes_u64(
    uint64_t      hash,
    const void   *data,
    size_t        len)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++)
    {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static double knn_model_max_cluster_radius(
    const KnnModel *model)
{
    double max_radius = (model != NULL) ? model->model_rlim : 0.0;
    if (model == NULL || model->clusters == NULL)
    {
        return max_radius;
    }

    for (int c = 0; c < model->num_clusters; c++)
    {
        double radius = (double)model->clusters[c].radius;
        if (radius > max_radius)
        {
            max_radius = radius;
        }
    }
    return max_radius;
}

static uint64_t knn_rq8_sidecar_fingerprint(
    const KnnModel *model)
{
    uint64_t hash = 1469598103934665603ULL;
    if (model == NULL || model->clusters == NULL || model->frame_cluster_map == NULL)
    {
        return hash;
    }

    hash = knn_hash_bytes_u64(hash, &model->num_clusters, sizeof(model->num_clusters));
    hash = knn_hash_bytes_u64(
        hash, &model->total_dataset_frames, sizeof(model->total_dataset_frames));
    hash = knn_hash_bytes_u64(hash, &model->frame_elements, sizeof(model->frame_elements));
    hash = knn_hash_bytes_u64(hash, &model->is_double, sizeof(model->is_double));

    hash = knn_hash_bytes_u64(
        hash, model->frame_cluster_map,
        (size_t)model->total_dataset_frames * sizeof(int)
    );

    size_t anchor_elem_size = model->is_double ? sizeof(double) : sizeof(float);
    for (int c = 0; c < model->num_clusters; c++)
    {
        if (model->clusters[c].anchor_data == NULL)
        {
            continue;
        }
        hash = knn_hash_bytes_u64(hash, &c, sizeof(c));
        hash = knn_hash_bytes_u64(hash, &model->clusters[c].radius, sizeof(double));
        hash = knn_hash_bytes_u64(
            hash, model->clusters[c].anchor_data,
            (size_t)model->frame_elements * anchor_elem_size
        );
    }

    return hash;
}

/**
 * knn_model_build_or_load_sq8() - Build or load quantized SQ8 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_sq8(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_sq8)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return -1;
    }

    // Path 1: Load from sidecar file if requested
    if (config->sq8_load_path != NULL)
    {
        long loaded_frames = 0;
        if (sq8_load_sidecar(config->sq8_load_path, &model->sq8_params,
                             &model->sq8_dataset_buffer, &loaded_frames) != 0)
        {
            fprintf(stderr, "Error: Failed to load SQ8 sidecar file '%s'\n",
                    config->sq8_load_path);
            return -1;
        }
        if (loaded_frames != N || model->sq8_params.dim != dim)
        {
            fprintf(stderr, "Error: SQ8 sidecar dimensions mismatch (%ldx%ld vs %ldx%ld)\n",
                    loaded_frames, model->sq8_params.dim, N, dim);
            free(model->sq8_dataset_buffer);
            model->sq8_dataset_buffer = NULL;
            return -1;
        }
        if (config->verbose_level >= 1)
        {
            printf("Loaded SQ8 sidecar: %ld frames, range [%.4f, %.4f], scale=%.6f\n",
                   N, model->sq8_params.min_val, model->sq8_params.max_val,
                   model->sq8_params.scale);
        }
        return 0;
    }

    // Path 2: Build SQ8 representation by scanning dataset frames
    KnnFrameReader reader;
    int open_res = 0;
    if (config->memory_data != NULL)
    {
        open_res = knn_reader_open_memory(
            &reader, config->memory_data, N, dim, model->is_double);
    }
    else
    {
        open_res = knn_reader_open(
            &reader, config->input_data_path, N, model->frame_width,
            model->frame_height, model->is_double);
    }
    if (open_res != 0)
    {
        fprintf(stderr, "Error: Failed to open dataset reader for SQ8 quantization\n");
        return -1;
    }

    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    void *frame_buf = malloc((size_t)dim * elem_size);
    if (frame_buf == NULL)
    {
        knn_reader_close(&reader);
        return -1;
    }

    // Calibrate min and max by scanning frames or using profile
    if (model->has_profile && model->profile.sq8_params.scale > 0.0f)
    {
        model->sq8_params = model->profile.sq8_params;
        model->sq8_params.dim = dim;
        if (config->verbose_level >= 1)
        {
            printf("  [PROFILE] Fast SQ8 init using profile range [%.4f, %.4f]\n",
                   model->profile.sq8_params.min_val, model->profile.sq8_params.max_val);
        }
    }
    else
    {
        float global_min = 0.0f;
        float global_max = 0.0f;
        if (knn_calibrate_dataset_minmax(&reader, frame_buf, N, dim, model->is_double,
                                         &global_min, &global_max) != 0)
        {
            free(frame_buf);
            knn_reader_close(&reader);
            return -1;
        }

        sq8_init_params(&model->sq8_params, global_min, global_max, dim);
    }

    // Allocate resident uint8 buffer [N x dim]
    size_t total_bytes = (size_t)N * (size_t)dim;
    model->sq8_dataset_buffer = (uint8_t *)malloc(total_bytes);
    if (model->sq8_dataset_buffer == NULL)
    {
        free(frame_buf);
        knn_reader_close(&reader);
        return -1;
    }

    // Quantize all frames
    for (long i = 0; i < N; i++)
    {
        if (knn_reader_read_frame(&reader, i, frame_buf) == 0)
        {
            uint8_t *dst = model->sq8_dataset_buffer + (size_t)i * (size_t)dim;
            if (model->is_double)
            {
                sq8_quantize_double((const double *)frame_buf, dst, &model->sq8_params);
            }
            else
            {
                sq8_quantize_float((const float *)frame_buf, dst, &model->sq8_params);
            }
        }
    } // for (long i = 0; i < N; i++)

    free(frame_buf);
    knn_reader_close(&reader);

    if (config->verbose_level >= 1)
    {
        printf("Built SQ8 dataset cache: %ld frames (%.2f MB), range [%.4f, %.4f]\n",
               N, (double)total_bytes / (1024.0 * 1024.0),
               model->sq8_params.min_val, model->sq8_params.max_val);
    }

    // Optional: Save sidecar file
    if (config->sq8_save_path != NULL)
    {
        if (sq8_save_sidecar(config->sq8_save_path, &model->sq8_params,
                             model->sq8_dataset_buffer, N) == 0)
        {
            if (config->verbose_level >= 1)
            {
                printf("Saved SQ8 sidecar file to '%s'\n", config->sq8_save_path);
            }
        }
        else
        {
            fprintf(stderr, "Warning: Failed to write SQ8 sidecar file '%s'\n",
                    config->sq8_save_path);
        }
    }

    // Pre-quantize anchor vectors for fast Level 2 anchor lower-bound pruning
    knn_quantize_cluster_anchors_sq8(model);

    return 0;
}

/**
 * knn_deduplicate_sq16_frames() - Deduplicate SQ16 frames into unique representative pool.
 * @model:  Pointer to KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
static int knn_deduplicate_sq16_frames(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || model->sq16_dataset_buffer == NULL || model->total_dataset_frames <= 0)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;

    model->frame_to_unique_map = (int *)malloc((size_t)N * sizeof(int));
    if (model->frame_to_unique_map == NULL)
    {
        return -1;
    }

    /* Size hash table as power of two >= 2*N (minimum 1024) */
    size_t cap = 1024;
    while (cap < (size_t)N * 2)
    {
        cap <<= 1;
    }
    size_t mask = cap - 1;

    typedef struct
    {
        uint64_t hash;
        int32_t  frame_id;
    } DedupSlot;

    DedupSlot *slots = (DedupSlot *)malloc(cap * sizeof(DedupSlot));
    if (slots == NULL)
    {
        for (long i = 0; i < N; i++)
        {
            model->frame_to_unique_map[i] = (int)i;
        }
        model->num_unique_frames = N;
        return 0;
    }

    for (size_t c = 0; c < cap; c++)
    {
        slots[c].frame_id = -1;
    }

    long unique_count = 0;
    for (long i = 0; i < N; i++)
    {
        const int16_t *cur_vec = model->sq16_dataset_buffer + (size_t)i * (size_t)dim;
        uint64_t h = gric_hash_i16(cur_vec, dim);
        size_t idx = (size_t)(h & mask);
        int match_id = -1;

        while (slots[idx].frame_id != -1)
        {
            if (slots[idx].hash == h)
            {
                const int16_t *match_vec =
                    model->sq16_dataset_buffer + (size_t)slots[idx].frame_id * (size_t)dim;
                if (memcmp(cur_vec, match_vec, (size_t)dim * sizeof(int16_t)) == 0)
                {
                    match_id = (int)slots[idx].frame_id;
                    break;
                }
            }
            idx = (idx + 1) & mask;
        }

        if (match_id != -1)
        {
            model->frame_to_unique_map[i] = match_id;
        }
        else
        {
            slots[idx].hash = h;
            slots[idx].frame_id = (int32_t)i;
            model->frame_to_unique_map[i] = (int)i;
            unique_count++;
        }
    } // for (long i = 0; i < N; i++)

    free(slots);
    model->num_unique_frames = unique_count;

    if (config->verbose_level >= 1)
    {
        double dup_pct = (N > 0) ? (100.0 * (1.0 - (double)unique_count / (double)N)) : 0.0;
        printf("  SQ16 Deduplication: %ld unique frames / %ld total (%.1f%% duplicate)\n",
               unique_count, N, dup_pct);
    }

    return 0;
}

/**
 * knn_model_build_or_load_sq16() - Build or load quantized SQ16 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_sq16(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (!config->use_sq16 || model == NULL || config == NULL)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return 0;
    }

    // Path 1: Load precomputed sidecar file if path specified
    if (config->sq16_load_path != NULL)
    {
        long loaded_frames = 0;
        if (sq16_load_sidecar(config->sq16_load_path, &model->sq16_params,
                              &model->sq16_dataset_buffer, &loaded_frames) != 0)
        {
            fprintf(stderr, "Error: Failed to load SQ16 sidecar file '%s'\n",
                    config->sq16_load_path);
            return -1;
        }
        if (loaded_frames != N || model->sq16_params.dim != dim)
        {
            fprintf(stderr, "Error: SQ16 sidecar dimensions mismatch (%ldx%ld vs %ldx%ld)\n",
                    loaded_frames, model->sq16_params.dim, N, dim);
            free(model->sq16_dataset_buffer);
            model->sq16_dataset_buffer = NULL;
            return -1;
        }
        if (config->verbose_level >= 1)
        {
            printf("Loaded SQ16 sidecar: %ld frames, range [%.4f, %.4f], scale=%.6f\n",
                   N, model->sq16_params.min_val, model->sq16_params.max_val,
                   model->sq16_params.scale);
        }
        goto sq16_postprocess;
    }

    // Path 2: Build SQ16 representation by scanning dataset frames
    KnnFrameReader reader;
    int open_res = 0;
    if (config->memory_data != NULL)
    {
        open_res = knn_reader_open_memory(
            &reader, config->memory_data, N, dim, model->is_double);
    }
    else
    {
        open_res = knn_reader_open(
            &reader, config->input_data_path, N, model->frame_width,
            model->frame_height, model->is_double);
    }
    if (open_res != 0)
    {
        fprintf(stderr, "Error: Failed to open dataset reader for SQ16 quantization\n");
        return -1;
    }

    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    void *frame_buf = malloc((size_t)dim * elem_size);
    if (frame_buf == NULL)
    {
        knn_reader_close(&reader);
        return -1;
    }

    // Calibrate SQ16 min/max
    if (model->has_profile && model->profile.sq16_params.scale > 0.0f)
    {
        model->sq16_params = model->profile.sq16_params;
        model->sq16_params.dim = dim;
        if (config->verbose_level >= 1)
        {
            printf("  [PROFILE] Fast SQ16 init using profile range [%.4f, %.4f]\n",
                   model->profile.sq16_params.min_val, model->profile.sq16_params.max_val);
        }
    }
    else
    {
        float global_min = 0.0f;
        float global_max = 0.0f;
        if (knn_calibrate_dataset_minmax(&reader, frame_buf, N, dim, model->is_double,
                                         &global_min, &global_max) != 0)
        {
            free(frame_buf);
            knn_reader_close(&reader);
            return -1;
        }

        sq16_init_params(&model->sq16_params, global_min, global_max, dim);

        /* Enforce --sq16-ratio bound: scale = alpha*rlim / sqrt(D) */
        double eff_rlim = (config->rlim_cutoff > 0.0) ? config->rlim_cutoff : model->model_rlim;
        if (config->sq16_ratio > 0.0 && eff_rlim > 0.0)
        {
            float target_scale = (float)((config->sq16_ratio * eff_rlim) / sqrt((double)dim));
            float span = global_max - global_min;
            float min_scale = (span > 0.0f) ? (span / 32767.0f) : 1e-6f;
            if (target_scale < min_scale)
            {
                target_scale = min_scale;
            }
            float center = 0.5f * (global_min + global_max);
            model->sq16_params.scale = target_scale;
            model->sq16_params.inv_scale = 1.0f / target_scale;
            model->sq16_params.min_val = center - 16384.0f * target_scale;
            model->sq16_params.max_val = center + 16383.0f * target_scale;
            model->sq16_params.err_radius = sqrtf((float)dim) * target_scale * 0.5f;
        }
    }

    // Allocate resident int16 buffer [N x dim]
    size_t total_elements = (size_t)N * (size_t)dim;
    model->sq16_dataset_buffer = (int16_t *)malloc(total_elements * sizeof(int16_t));
    if (model->sq16_dataset_buffer == NULL)
    {
        free(frame_buf);
        knn_reader_close(&reader);
        return -1;
    }

    // Quantize all frames
    for (long i = 0; i < N; i++)
    {
        if (knn_reader_read_frame(&reader, i, frame_buf) == 0)
        {
            int16_t *dst = model->sq16_dataset_buffer + (size_t)i * (size_t)dim;
            if (model->is_double)
            {
                sq16_quantize_double((const double *)frame_buf, dst, &model->sq16_params);
            }
            else
            {
                sq16_quantize_float((const float *)frame_buf, dst, &model->sq16_params);
            }
        }
    } // for (long i = 0; i < N; i++)

    free(frame_buf);
    knn_reader_close(&reader);

    if (config->verbose_level >= 1)
    {
        printf("Built SQ16 dataset cache: %ld frames (%.2f MB), range [%.4f, %.4f]\n",
               N, (double)(total_elements * sizeof(int16_t)) / (1024.0 * 1024.0),
               model->sq16_params.min_val, model->sq16_params.max_val);
    }

    // Optional: Save sidecar file
    if (config->sq16_save_path != NULL)
    {
        if (sq16_save_sidecar(config->sq16_save_path, &model->sq16_params,
                              model->sq16_dataset_buffer, N) == 0)
        {
            if (config->verbose_level >= 1)
            {
                printf("Saved SQ16 sidecar file to '%s'\n", config->sq16_save_path);
            }
        }
        else
        {
            fprintf(stderr, "Warning: Failed to write SQ16 sidecar file '%s'\n",
                    config->sq16_save_path);
        }
    }

sq16_postprocess:
    // Pre-quantize anchor vectors for fast Level 2 anchor lower-bound pruning
    knn_quantize_cluster_anchors_sq16(model);

    if (config->use_memo)
    {
        knn_deduplicate_sq16_frames(model, config);
    }

    // Build transposed SIMD FastScan blocks for all clusters
    if (knn_model_build_transposed_sq16(model, config) != 0)
    {
        return -1;
    }

    return 0;
}

/**
 * knn_model_build_or_load_rq8() - Build or load quantized RQ8 dataset buffer into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_rq8(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_rq8)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return 0;
    }

    double eff_rlim = knn_model_max_cluster_radius(model);
    if (eff_rlim <= 0.0)
    {
        eff_rlim = 1.0;
    }
    rq8_init_params(&model->rq8_params, (float)eff_rlim, dim);

    for (int c = 0; c < model->num_clusters; c++)
    {
        double cl_r = model->clusters[c].radius;
        if (cl_r <= 0.0)
        {
            cl_r = (model->model_rlim > 0.0) ? model->model_rlim : eff_rlim;
        }
        rq8_init_params(&model->clusters[c].rq8_params, (float)cl_r, dim);
    }

    uint64_t expected_fingerprint = knn_rq8_sidecar_fingerprint(model);

    // Path 1: Load precomputed sidecar file if path specified
    if (config->rq8_load_path != NULL)
    {
        long loaded_frames = 0;
        uint64_t loaded_fingerprint = 0;
        if (rq8_load_sidecar(config->rq8_load_path, &model->rq8_params,
                             &model->rq8_dataset_buffer, &loaded_frames,
                             &loaded_fingerprint) != 0)
        {
            fprintf(stderr, "Error: Failed to load RQ8 sidecar file '%s'\n",
                    config->rq8_load_path);
            return -1;
        }
        if (loaded_frames != N || model->rq8_params.dim != dim ||
            loaded_fingerprint != expected_fingerprint)
        {
            fprintf(stderr,
                    "Error: RQ8 sidecar mismatch (loaded %ld frames, dim %ld, "
                    "fingerprint 0x%016llx; expected %ld, %ld, 0x%016llx)\n",
                    loaded_frames, model->rq8_params.dim,
                    (unsigned long long)loaded_fingerprint, N, dim,
                    (unsigned long long)expected_fingerprint);
            free(model->rq8_dataset_buffer);
            model->rq8_dataset_buffer = NULL;
            return -1;
        }
        if (config->verbose_level >= 1)
        {
            printf("Loaded RQ8 dataset sidecar: %ld frames, rlim: %.4f, scale: %.6f\n",
                   N, model->rq8_params.rlim, model->rq8_params.scale);
        }

        if (knn_model_build_transposed_rq8(model, config) != 0)
        {
            return -1;
        }
        return 0;
    } // if (config->rq8_load_path != NULL)

    // Allocate resident int8 buffer [N x dim]
    size_t total_elements = (size_t)N * (size_t)dim;
    model->rq8_dataset_buffer = (int8_t *)malloc(total_elements * sizeof(int8_t));
    if (model->rq8_dataset_buffer == NULL)
    {
        return -1;
    }

    // Fast path: Dataset already cached in RAM
    if (model->dataset_buffer != NULL)
    {
        for (long i = 0; i < N; i++)
        {
            int c = model->frame_cluster_map ? model->frame_cluster_map[i] : 0;
            if (c < 0 || c >= model->num_clusters)
            {
                c = 0;
            }
            const void *anchor = model->clusters[c].anchor_data;
            int8_t *dst = model->rq8_dataset_buffer + (size_t)i * (size_t)dim;

            if (model->is_double)
            {
                const double *src = (const double *)model->dataset_buffer + (size_t)i * (size_t)dim;
                rq8_quantize_residual_double(src, (const double *)anchor, dst,
                                             &model->clusters[c].rq8_params);
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer + (size_t)i * (size_t)dim;
                rq8_quantize_residual_float(src, (const float *)anchor, dst,
                                            &model->clusters[c].rq8_params);
            }
        } // for (long i = 0; i < N; i++)
    }
    else
    {
        // Reader fallback for non-cached dataset
        KnnFrameReader reader;
        int open_res = -1;
        if (config->memory_data != NULL)
        {
            open_res = knn_reader_open_memory(
                &reader, config->memory_data, N, dim, model->is_double
            );
        }
        else
        {
            open_res = knn_reader_open(
                &reader, config->input_data_path, N,
                model->frame_width, model->frame_height, model->is_double
            );
        }
        if (open_res != 0)
        {
            free(model->rq8_dataset_buffer);
            model->rq8_dataset_buffer = NULL;
            return -1;
        }

        size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
        void *frame_buf = malloc((size_t)dim * elem_size);
        if (frame_buf == NULL)
        {
            knn_reader_close(&reader);
            free(model->rq8_dataset_buffer);
            model->rq8_dataset_buffer = NULL;
            return -1;
        }

        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&reader, i, frame_buf) != 0)
            {
                free(frame_buf);
                knn_reader_close(&reader);
                free(model->rq8_dataset_buffer);
                model->rq8_dataset_buffer = NULL;
                return -1;
            }

            int c = model->frame_cluster_map ? model->frame_cluster_map[i] : 0;
            if (c < 0 || c >= model->num_clusters)
            {
                c = 0;
            }
            const void *anchor = model->clusters[c].anchor_data;
            int8_t *dst = model->rq8_dataset_buffer + (size_t)i * (size_t)dim;

            if (model->is_double)
            {
                rq8_quantize_residual_double((const double *)frame_buf,
                                             (const double *)anchor, dst,
                                             &model->clusters[c].rq8_params);
            }
            else
            {
                rq8_quantize_residual_float((const float *)frame_buf,
                                            (const float *)anchor, dst,
                                            &model->clusters[c].rq8_params);
            }
        } // for (long i = 0; i < N; i++)

        free(frame_buf);
        knn_reader_close(&reader);
    }

    if (config->verbose_level >= 1)
    {
        printf("Built RQ8 dataset cache: %ld frames (%.2f MB), rlim: %.4f, scale: %.6f\n",
               N, (double)(total_elements * sizeof(int8_t)) / (1024.0 * 1024.0),
               model->rq8_params.rlim, model->rq8_params.scale);
    }

    // Optional: Save sidecar file
    if (config->rq8_save_path != NULL)
    {
        if (rq8_save_sidecar(config->rq8_save_path, &model->rq8_params,
                             model->rq8_dataset_buffer, N, expected_fingerprint) == 0)
        {
            if (config->verbose_level >= 1)
            {
                printf("Saved RQ8 sidecar file to '%s'\n", config->rq8_save_path);
            }
        }
        else
        {
            fprintf(stderr, "Warning: Failed to save RQ8 sidecar file to '%s'\n",
                    config->rq8_save_path);
        }
    }

    // Build transposed SIMD FastScan blocks for all clusters
    if (knn_model_build_transposed_rq8(model, config) != 0)
    {
        return -1;
    }

    return 0;
}

/**
 * knn_model_cache_dataset() - Preload dataset frames into resident RAM buffer if feasible.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success (cached or skipped), -1 on error.
 */
int knn_model_cache_dataset(
    KnnModel  *model,
    KnnConfig *config)
{
    if (model == NULL || config == NULL)
    {
        return -1;
    }

    if (config->memory_data != NULL || config->no_cache_dataset)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    size_t total_bytes = (size_t)N * (size_t)dim * elem_size;

    // Cache if dataset is <= 2 GB
    if (total_bytes > 2ULL * 1024 * 1024 * 1024)
    {
        return 0;
    }

    FILE *fp_bin = fopen(config->input_data_path, "rb");
    if (fp_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(fp_bin, &hdr, &comment) == 0)
        {
            if (comment != NULL)
            {
                free(comment);
            }
            int type_matches = (model->is_double &&
                                hdr.data_type == GRIC_BIN_DTYPE_FLOAT64) ||
                               (!model->is_double &&
                                hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
            if (type_matches && hdr.num_elements >= (uint64_t)N * (uint64_t)dim)
            {
                int fd = fileno(fp_bin);
                struct stat st;
                if (fstat(fd, &st) == 0 &&
                    (size_t)st.st_size >= hdr.header_bytes + total_bytes)
                {
                    void *mmap_addr = mmap(NULL, (size_t)st.st_size, PROT_READ,
                                           MAP_SHARED, fd, 0);
                    if (mmap_addr != MAP_FAILED)
                    {
                        posix_madvise(mmap_addr, (size_t)st.st_size, POSIX_MADV_WILLNEED);
                        model->dataset_mmap_addr = mmap_addr;
                        model->dataset_mmap_size = (size_t)st.st_size;
                        model->dataset_buffer = (char *)mmap_addr + hdr.header_bytes;
                        config->memory_data = model->dataset_buffer;
                        fclose(fp_bin);
                        if (config->verbose_level >= 1)
                        {
                            printf("Mmapped resident dataset in RAM: %ld frames (%.2f MB, "
                                   "zero-copy)\n",
                                   N, (double)total_bytes / (1024.0 * 1024.0));
                        }
                        knn_model_build_ivf_layout(model, config);
                        return 0;
                    }
                }
            }
        }
        else if (comment != NULL)
        {
            free(comment);
        }
        fclose(fp_bin);
    }

    void *buf = malloc(total_bytes);
    if (buf == NULL)
    {
        return 0;
    }

    KnnFrameReader reader;
    if (knn_reader_open(&reader, config->input_data_path, N,
                        model->frame_width, model->frame_height,
                        model->is_double) != 0)
    {
        free(buf);
        return -1;
    }

    for (long i = 0; i < N; i++)
    {
        char *dst = (char *)buf + (size_t)i * (size_t)dim * elem_size;
        if (knn_reader_read_frame(&reader, i, dst) != 0)
        {
            free(buf);
            knn_reader_close(&reader);
            return -1;
        }
    }

    knn_reader_close(&reader);

    model->dataset_buffer = buf;
    config->memory_data = buf;
    if (config->verbose_level >= 1)
    {
        printf("Cached resident dataset in RAM: %ld frames (%.2f MB)\n",
               N, (double)total_bytes / (1024.0 * 1024.0));
    }

    knn_model_build_ivf_layout(model, config);

    return 0;
}

/**
 * knn_model_build_or_load_pq() - Build or load PQ codebook and codes into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_pq(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_pq)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return 0;
    }

    /* Path 1: Load from sidecar */
    if (config->pq_load_path != NULL)
    {
        long loaded_frames = 0;
        if (pq_load_sidecar(config->pq_load_path, &model->pq_codebook,
                            &model->pq_transposed_buffer, &loaded_frames) != 0)
        {
            fprintf(stderr, "Error: Failed to load PQ sidecar '%s'\n", config->pq_load_path);
            return -1;
        }

        if (loaded_frames != N || model->pq_codebook->dim != dim)
        {
            fprintf(stderr, "Error: PQ sidecar dimension mismatch (%ldx%ld vs %ldx%ld)\n",
                    loaded_frames, model->pq_codebook->dim, N, dim);
            return -1;
        }

        /* Re-map transposed pointers to clusters */
        int M = model->num_clusters;
        int m = model->pq_codebook->m;
        size_t cur_offset = 0;
        for (int c = 0; c < M; c++)
        {
            int num_m = model->clusters[c].num_members;
            int n_blocks = (num_m + PQ_FASTSCAN_BLOCK_SIZE - 1) / PQ_FASTSCAN_BLOCK_SIZE;
            model->clusters[c].num_pq_blocks = n_blocks;
            model->clusters[c].pq_transposed = (n_blocks > 0) ?
                (model->pq_transposed_buffer + cur_offset) : NULL;
            cur_offset += (size_t)n_blocks * (size_t)m * PQ_FASTSCAN_BLOCK_SIZE;
        } // for (int c = 0; c < M; c++)

        if (config->verbose_level >= 1)
        {
            printf("Loaded PQ sidecar: %ld frames, m=%d, d_sub=%d, K=%d\n",
                   N, m, model->pq_codebook->d_sub, model->pq_codebook->k_centroids);
        }
        return 0;
    } // if (config->pq_load_path != NULL)

    /* Path 2: Train codebook and quantize */
    int m = config->pq_m;
    if (m <= 0)
    {
        if (dim >= 64 && (dim % 16 == 0))
        {
            m = 16;
        }
        else if (dim >= 32 && (dim % 8 == 0))
        {
            m = 8;
        }
        else if (dim >= 16 && (dim % 4 == 0))
        {
            m = (int)(dim / 4);
        }
        else if (dim % 2 == 0)
        {
            m = (int)(dim / 2);
        }
        else
        {
            m = 1;
        }
    }

    if ((dim % m) != 0)
    {
        fprintf(stderr, "Error: PQ subquantizers m=%d does not divide dim=%ld\n", m, dim);
        return -1;
    }

    int K = (config->pq_bits == 8) ? 256 : 16;
    model->pq_codebook = pq_codebook_alloc(dim, m, K);
    if (model->pq_codebook == NULL)
    {
        return -1;
    }

    /* Collect training sample */
    long num_train = (N < 1000) ? N : 1000;
    float *train_data = (float *)malloc((size_t)num_train * (size_t)dim * sizeof(float));
    if (train_data == NULL)
    {
        return -1;
    }

    if (model->dataset_buffer != NULL)
    {
        for (long i = 0; i < num_train; i++)
        {
            if (model->is_double)
            {
                const double *src = (const double *)model->dataset_buffer + i * dim;
                for (long d = 0; d < dim; d++)
                {
                    train_data[i * dim + d] = (float)src[d];
                }
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer + i * dim;
                memcpy(train_data + i * dim, src, (size_t)dim * sizeof(float));
            }
        } // for (long i = 0; i < num_train; i++)
    }
    else
    {
        KnnFrameReader reader;
        if (knn_reader_open(&reader, config->input_data_path, N,
                            model->frame_width, model->frame_height, model->is_double) != 0)
        {
            free(train_data);
            return -1;
        }
        void *tmp_frame = malloc((size_t)dim * (model->is_double ? sizeof(double) : sizeof(float)));
        for (long i = 0; i < num_train; i++)
        {
            if (knn_reader_read_frame(&reader, i, tmp_frame) == 0)
            {
                if (model->is_double)
                {
                    const double *dptr = (const double *)tmp_frame;
                    for (long d = 0; d < dim; d++)
                    {
                        train_data[i * dim + d] = (float)dptr[d];
                    }
                }
                else
                {
                    memcpy(train_data + i * dim, tmp_frame, (size_t)dim * sizeof(float));
                }
            }
        } // for (long i = 0; i < num_train; i++)
        free(tmp_frame);
        knn_reader_close(&reader);
    }

    if (config->verbose_level >= 1)
    {
        printf("Training PQ codebook on %ld sample frames (m=%d, d_sub=%d, K=%d)...\n",
               num_train, m, model->pq_codebook->d_sub, K);
    }

    if (pq_train_codebook(model->pq_codebook, train_data, num_train, 15) != 0)
    {
        free(train_data);
        return -1;
    }
    free(train_data);

    /* Allocate and quantize full dataset */
    model->pq_dataset_buffer = (uint8_t *)malloc((size_t)N * (size_t)m);
    if (model->pq_dataset_buffer == NULL)
    {
        return -1;
    }

    if (model->dataset_buffer != NULL)
    {
        for (long i = 0; i < N; i++)
        {
            uint8_t *dst = model->pq_dataset_buffer + i * m;
            if (model->is_double)
            {
                const double *src = (const double *)model->dataset_buffer + i * dim;
                pq_quantize_frame_double(src, dst, model->pq_codebook);
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer + i * dim;
                pq_quantize_frame_float(src, dst, model->pq_codebook);
            }
        } // for (long i = 0; i < N; i++)
    }
    else
    {
        KnnFrameReader reader;
        if (knn_reader_open(&reader, config->input_data_path, N,
                            model->frame_width, model->frame_height, model->is_double) != 0)
        {
            return -1;
        }
        void *tmp_frame = malloc((size_t)dim * (model->is_double ? sizeof(double) : sizeof(float)));
        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&reader, i, tmp_frame) == 0)
            {
                uint8_t *dst = model->pq_dataset_buffer + i * m;
                if (model->is_double)
                {
                    pq_quantize_frame_double((const double *)tmp_frame, dst, model->pq_codebook);
                }
                else
                {
                    pq_quantize_frame_float((const float *)tmp_frame, dst, model->pq_codebook);
                }
            }
        } // for (long i = 0; i < N; i++)
        free(tmp_frame);
        knn_reader_close(&reader);
    }

    if (knn_model_build_transposed_pq(model, config) != 0)
    {
        return -1;
    }

    if (config->pq_save_path != NULL)
    {
        if (pq_save_sidecar(config->pq_save_path, model->pq_codebook,
                            model->pq_transposed_buffer, N) != 0)
        {
            fprintf(stderr, "Warning: Failed to save PQ sidecar to '%s'\n", config->pq_save_path);
        }
        else if (config->verbose_level >= 1)
        {
            printf("Saved PQ sidecar to '%s'\n", config->pq_save_path);
        }
    }

    return 0;
}

/**
 * knn_model_build_or_load_rabitq() - Build or load RaBitQ bit codes into KnnModel.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_or_load_rabitq(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_rabitq)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return 0;
    }

    /* Path 1: Load from precomputed sidecar file */
    if (config->rabitq_load_path != NULL)
    {
        long loaded_frames = 0;
        if (rabitq_load_sidecar(config->rabitq_load_path, &model->rabitq_params,
                                &model->rabitq_meta_buffer, &model->rabitq_dataset_buffer,
                                &loaded_frames) != 0)
        {
            fprintf(stderr, "Error: Failed to load RaBitQ sidecar '%s'\n",
                    config->rabitq_load_path);
            return -1;
        }

        if (loaded_frames != N || model->rabitq_params.dim != dim)
        {
            fprintf(stderr, "Error: RaBitQ sidecar dimension mismatch (%ldx%ld vs %ldx%ld)\n",
                    loaded_frames, model->rabitq_params.dim, N, dim);
            return -1;
        }

        if (knn_model_build_transposed_rabitq(model, config) != 0)
        {
            return -1;
        }

        if (config->verbose_level >= 1)
        {
            printf("Loaded RaBitQ sidecar: %ld frames, dim=%ld (pad=%ld), bits=%d\n",
                   N, dim, model->rabitq_params.dim_pad, model->rabitq_params.bits);
        }
        return 0;
    } // if (config->rabitq_load_path != NULL)

    /* Path 2: Initialize RaBitQ params and quantize dataset */
    int bits = (config->rabitq_bits == 1) ? 1 : 2;
    if (rabitq_init_params(&model->rabitq_params, dim, bits, 0) != 0)
    {
        return -1;
    }

    size_t code_bytes = model->rabitq_params.code_bytes_per_vec;
    model->rabitq_dataset_buffer = (uint8_t *)malloc((size_t)N * code_bytes);
    model->rabitq_meta_buffer = (RaBitQMeta *)malloc((size_t)N * sizeof(RaBitQMeta));

    if (model->rabitq_dataset_buffer == NULL || model->rabitq_meta_buffer == NULL)
    {
        return -1;
    }

    if (model->dataset_buffer != NULL)
    {
        for (long i = 0; i < N; i++)
        {
            uint8_t *dst_code = model->rabitq_dataset_buffer + (size_t)i * code_bytes;
            RaBitQMeta *dst_meta = &model->rabitq_meta_buffer[i];

            if (model->is_double)
            {
                const double *src = (const double *)model->dataset_buffer + i * dim;
                rabitq_quantize_vector_double(src, dst_code, dst_meta, &model->rabitq_params);
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer + i * dim;
                rabitq_quantize_vector_float(src, dst_code, dst_meta, &model->rabitq_params);
            }
        } // for (long i = 0; i < N; i++)
    }
    else
    {
        KnnFrameReader reader;
        if (knn_reader_open(&reader, config->input_data_path, N,
                            model->frame_width, model->frame_height, model->is_double) != 0)
        {
            return -1;
        }

        size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
        void *tmp_frame = malloc((size_t)dim * elem_size);
        if (tmp_frame == NULL)
        {
            knn_reader_close(&reader);
            return -1;
        }

        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&reader, i, tmp_frame) == 0)
            {
                uint8_t *dst_code = model->rabitq_dataset_buffer + (size_t)i * code_bytes;
                RaBitQMeta *dst_meta = &model->rabitq_meta_buffer[i];

                if (model->is_double)
                {
                    rabitq_quantize_vector_double(
                        (const double *)tmp_frame, dst_code, dst_meta, &model->rabitq_params
                    );
                }
                else
                {
                    rabitq_quantize_vector_float(
                        (const float *)tmp_frame, dst_code, dst_meta, &model->rabitq_params
                    );
                }
            }
        } // for (long i = 0; i < N; i++)

        free(tmp_frame);
        knn_reader_close(&reader);
    }

    if (knn_model_build_transposed_rabitq(model, config) != 0)
    {
        return -1;
    }

    if (config->rabitq_save_path != NULL)
    {
        if (rabitq_save_sidecar(config->rabitq_save_path, &model->rabitq_params,
                                model->rabitq_meta_buffer, model->rabitq_dataset_buffer, N) != 0)
        {
            fprintf(stderr, "Warning: Failed to save RaBitQ sidecar to '%s'\n",
                    config->rabitq_save_path);
        }
        else if (config->verbose_level >= 1)
        {
            printf("Saved RaBitQ sidecar to '%s'\n", config->rabitq_save_path);
        }
    }

    return 0;
}



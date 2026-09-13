/**
 * @file knn_cache.c
 * @brief Memory caching and scalar quantization sidecar generation/loading for k-NN.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_cache.h"
#include "knn_reader.h"
#include "scalar_quant.h"
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

static int knn_size_mul(
    size_t  a,
    size_t  b,
    size_t *out)
{
    if (out == NULL)
    {
        return -1;
    }
    if (a != 0 && b > SIZE_MAX / a)
    {
        return -1;
    }
    *out = a * b;
    return 0;
}

static int knn_size_add(
    size_t  a,
    size_t  b,
    size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b)
    {
        return -1;
    }
    *out = a + b;
    return 0;
}

static int knn_alloc_sq16_transposed_buffer(
    int16_t **buffer,
    size_t    bytes)
{
    if (buffer == NULL)
    {
        return -1;
    }

#if defined(_POSIX_VERSION)
    if (posix_memalign((void **)buffer, 64, bytes) == 0)
    {
        return 0;
    }
#endif

    *buffer = (int16_t *)malloc(bytes);
    return (*buffer != NULL) ? 0 : -1;
}

static int knn_alloc_rq8_transposed_buffer(
    int8_t **buffer,
    size_t   bytes)
{
    if (buffer == NULL)
    {
        return -1;
    }

#if defined(_POSIX_VERSION)
    if (posix_memalign((void **)buffer, 64, bytes) == 0)
    {
        return 0;
    }
#endif

    *buffer = (int8_t *)malloc(bytes);
    return (*buffer != NULL) ? 0 : -1;
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
        float global_min = 1e30f;
        float global_max = -1e30f;

        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&reader, i, frame_buf) == 0)
            {
                if (model->is_double)
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
        } // for (long i = 0; i < N; i++)

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
    if (model->clusters != NULL && model->num_clusters > 0)
    {
        size_t anchor_elems = (size_t)model->num_clusters * (size_t)dim;
        model->anchor_sq8_buffer = (uint8_t *)malloc(anchor_elems * sizeof(uint8_t));
        if (model->anchor_sq8_buffer != NULL)
        {
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
    }

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
        float global_min = 1e30f;
        float global_max = -1e30f;

        for (long i = 0; i < N; i++)
        {
            if (knn_reader_read_frame(&reader, i, frame_buf) == 0)
            {
                if (model->is_double)
                {
                    const double *dptr = (const double *)frame_buf;
                    for (long d = 0; d < dim; d++)
                    {
                        float v = (float)dptr[d];
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
                    const float *fptr = (const float *)frame_buf;
                    for (long d = 0; d < dim; d++)
                    {
                        float v = fptr[d];
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
        } // for (long i = 0; i < N; i++)

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
    if (model->clusters != NULL && model->num_clusters > 0)
    {
        size_t anchor_elems = (size_t)model->num_clusters * (size_t)dim;
        model->anchor_sq16_buffer = (int16_t *)malloc(anchor_elems * sizeof(int16_t));
        if (model->anchor_sq16_buffer != NULL)
        {
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
    }

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
 * knn_model_build_transposed_sq16() - Build cluster-local transposed SQ16 FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_sq16(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_sq16)
    {
        return 0;
    }

    if (model->sq16_dataset_buffer == NULL || model->clusters == NULL ||
        model->num_clusters <= 0)
    {
        return 0;
    }

    long dim = model->frame_elements;
    int M = model->num_clusters;
    size_t transposed_budget_elems = 0;
    size_t total_transposed_elems = 0;
    int skipped_clusters = 0;

    if (model->sq16_transposed_buffer != NULL)
    {
        free(model->sq16_transposed_buffer);
        model->sq16_transposed_buffer = NULL;
    }
    for (int c = 0; c < M; c++)
    {
        model->clusters[c].sq16_transposed = NULL;
        model->clusters[c].num_sq16_blocks = 0;
    }
    if (knn_size_mul((size_t)model->total_dataset_frames, (size_t)dim,
                     &transposed_budget_elems) != 0)
    {
        return -1;
    }

    for (int c = 0; c < M; c++)
    {
        int num_m = model->clusters[c].num_members;
        int n_blocks = (num_m + SQ16_FASTSCAN_BLOCK_SIZE - 1) / SQ16_FASTSCAN_BLOCK_SIZE;
        size_t padded_members = 0;
        size_t padded_elems = 0;
        size_t next_total = 0;

        if (knn_size_mul((size_t)n_blocks, SQ16_FASTSCAN_BLOCK_SIZE, &padded_members) != 0 ||
            knn_size_mul(padded_members, (size_t)dim, &padded_elems) != 0)
        {
            return -1;
        }

        if (num_m <= 0)
        {
            model->clusters[c].sq16_transposed = NULL;
            model->clusters[c].num_sq16_blocks = 0;
            skipped_clusters++;
            continue;
        }

        if (knn_size_add(total_transposed_elems, padded_elems, &next_total) != 0)
        {
            return -1;
        }
        if (next_total > transposed_budget_elems)
        {
            skipped_clusters++;
            continue;
        }

        model->clusters[c].num_sq16_blocks = n_blocks;
        total_transposed_elems = next_total;
    } // for (int c = 0; c < M; c++)

    if (total_transposed_elems == 0)
    {
        return 0;
    }

    size_t alloc_bytes = 0;
    if (knn_size_mul(total_transposed_elems, sizeof(int16_t), &alloc_bytes) != 0)
    {
        return -1;
    }

    if (knn_alloc_sq16_transposed_buffer(&model->sq16_transposed_buffer, alloc_bytes) != 0)
    {
        return -1;
    }

    size_t cur_offset = 0;
    for (int c = 0; c < M; c++)
    {
        int n_blocks = model->clusters[c].num_sq16_blocks;
        if (n_blocks <= 0)
        {
            model->clusters[c].sq16_transposed = NULL;
            continue;
        }
        int16_t *cl_buf = model->sq16_transposed_buffer + cur_offset;
        model->clusters[c].sq16_transposed = cl_buf;
        cur_offset += (size_t)n_blocks * (size_t)dim * SQ16_FASTSCAN_BLOCK_SIZE;

        int num_m = model->clusters[c].num_members;
        for (int b = 0; b < n_blocks; b++)
        {
            int16_t *block_ptr = cl_buf + (size_t)b * (size_t)dim * SQ16_FASTSCAN_BLOCK_SIZE;
            int m_start = b * SQ16_FASTSCAN_BLOCK_SIZE;
            int m_count = num_m - m_start;
            if (m_count > SQ16_FASTSCAN_BLOCK_SIZE)
            {
                m_count = SQ16_FASTSCAN_BLOCK_SIZE;
            }

            for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
            {
                if (i < m_count)
                {
                    long cand_id = (long)model->clusters[c].members[m_start + i].frame_id;
                    if (cand_id < 0 || cand_id >= model->total_dataset_frames)
                    {
                        return -1;
                    }
                    const int16_t *cand_src = model->sq16_dataset_buffer + cand_id * dim;
                    for (long d = 0; d < dim; d++)
                    {
                        block_ptr[d * SQ16_FASTSCAN_BLOCK_SIZE + i] = cand_src[d];
                    }
                }
                else
                {
                    // Pad dummy lanes with 32767 so they never pass cutoff
                    for (long d = 0; d < dim; d++)
                    {
                        block_ptr[d * SQ16_FASTSCAN_BLOCK_SIZE + i] = 32767;
                    }
                }
            } // for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
        } // for (int b = 0; b < n_blocks; b++)
    } // for (int c = 0; c < M; c++)

    if (config->verbose_level >= 1)
    {
        double mb = (double)(total_transposed_elems * sizeof(int16_t)) / (1024.0 * 1024.0);
        printf("  [FASTSCAN] Built SIMD transposed blocks (%s): %.2f MB",
               sq16_get_simd_mode_str(), mb);
        if (skipped_clusters > 0)
        {
            printf(" (%d clusters kept on per-candidate SQ16 due to memory budget)",
                   skipped_clusters);
        }
        printf("\n");
    }

    return 0;
}

/**
 * knn_model_build_transposed_rq8() - Build cluster-local transposed RQ8 FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_rq8(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_rq8 || model->rq8_dataset_buffer == NULL)
    {
        return 0;
    }

    int M = model->num_clusters;
    long dim = model->frame_elements;
    if (M <= 0 || dim <= 0)
    {
        return 0;
    }

    size_t total_transposed_elems = 0;
    size_t transposed_budget_elems = 0;
    int skipped_clusters = 0;

    if (model->rq8_transposed_buffer != NULL)
    {
        free(model->rq8_transposed_buffer);
        model->rq8_transposed_buffer = NULL;
    }
    for (int c = 0; c < M; c++)
    {
        model->clusters[c].rq8_transposed = NULL;
        model->clusters[c].num_rq8_blocks = 0;
    }
    if (knn_size_mul((size_t)model->total_dataset_frames, (size_t)dim,
                     &transposed_budget_elems) != 0)
    {
        return -1;
    }

    for (int c = 0; c < M; c++)
    {
        int num_m = model->clusters[c].num_members;
        int n_blocks = (num_m + RQ8_FASTSCAN_BLOCK_SIZE - 1) / RQ8_FASTSCAN_BLOCK_SIZE;
        size_t padded_members = 0;
        size_t padded_elems = 0;
        size_t next_total = 0;

        if (knn_size_mul((size_t)n_blocks, RQ8_FASTSCAN_BLOCK_SIZE, &padded_members) != 0 ||
            knn_size_mul(padded_members, (size_t)dim, &padded_elems) != 0)
        {
            return -1;
        }

        if (num_m <= 0)
        {
            model->clusters[c].rq8_transposed = NULL;
            model->clusters[c].num_rq8_blocks = 0;
            skipped_clusters++;
            continue;
        }

        if (knn_size_add(total_transposed_elems, padded_elems, &next_total) != 0)
        {
            return -1;
        }
        if (next_total > transposed_budget_elems)
        {
            skipped_clusters++;
            continue;
        }

        model->clusters[c].num_rq8_blocks = n_blocks;
        total_transposed_elems = next_total;
    } // for (int c = 0; c < M; c++)

    if (total_transposed_elems == 0)
    {
        return 0;
    }

    size_t alloc_bytes = 0;
    if (knn_size_mul(total_transposed_elems, sizeof(int8_t), &alloc_bytes) != 0)
    {
        return -1;
    }

    if (knn_alloc_rq8_transposed_buffer(&model->rq8_transposed_buffer, alloc_bytes) != 0)
    {
        return -1;
    }

    size_t cur_offset = 0;
    for (int c = 0; c < M; c++)
    {
        int n_blocks = model->clusters[c].num_rq8_blocks;
        if (n_blocks <= 0)
        {
            model->clusters[c].rq8_transposed = NULL;
            continue;
        }
        int8_t *cl_buf = model->rq8_transposed_buffer + cur_offset;
        model->clusters[c].rq8_transposed = cl_buf;
        cur_offset += (size_t)n_blocks * (size_t)dim * RQ8_FASTSCAN_BLOCK_SIZE;

        int num_m = model->clusters[c].num_members;
        for (int b = 0; b < n_blocks; b++)
        {
            int8_t *block_ptr = cl_buf + (size_t)b * (size_t)dim * RQ8_FASTSCAN_BLOCK_SIZE;
            int m_start = b * RQ8_FASTSCAN_BLOCK_SIZE;
            int m_count = num_m - m_start;
            if (m_count > RQ8_FASTSCAN_BLOCK_SIZE)
            {
                m_count = RQ8_FASTSCAN_BLOCK_SIZE;
            }

            for (int i = 0; i < RQ8_FASTSCAN_BLOCK_SIZE; i++)
            {
                if (i < m_count)
                {
                    long cand_id = (long)model->clusters[c].members[m_start + i].frame_id;
                    if (cand_id < 0 || cand_id >= model->total_dataset_frames)
                    {
                        return -1;
                    }
                    const int8_t *cand_src = model->rq8_dataset_buffer + cand_id * dim;
                    for (long d = 0; d < dim; d++)
                    {
                        block_ptr[d * RQ8_FASTSCAN_BLOCK_SIZE + i] = cand_src[d];
                    }
                }
                else
                {
                    // Pad dummy lanes with alternating +-127 so they never pass cutoff
                    for (long d = 0; d < dim; d++)
                    {
                        block_ptr[d * RQ8_FASTSCAN_BLOCK_SIZE + i] = (d & 1) ? -127 : 127;
                    }
                }
            } // for (int i = 0; i < RQ8_FASTSCAN_BLOCK_SIZE; i++)
        } // for (int b = 0; b < n_blocks; b++)
    } // for (int c = 0; c < M; c++)

    if (config->verbose_level >= 1)
    {
        double mb = (double)(total_transposed_elems * sizeof(int8_t)) / (1024.0 * 1024.0);
        printf("  [FASTSCAN] Built RQ8 SIMD transposed blocks (%s): %.2f MB",
               rq8_get_simd_mode_str(), mb);
        if (skipped_clusters > 0)
        {
            printf(" (%d clusters kept on per-candidate RQ8 due to memory budget)",
                   skipped_clusters);
        }
        printf("\n");
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
    if (!config->use_rq8 || model == NULL || config == NULL)
    {
        return 0;
    }

    long N = model->total_dataset_frames;
    long dim = model->frame_elements;
    if (N <= 0 || dim <= 0)
    {
        return 0;
    }

    double eff_rlim = (config->rlim_cutoff > 0.0) ? config->rlim_cutoff : model->model_rlim;
    if (eff_rlim <= 0.0)
    {
        eff_rlim = 1.0;
    }
    rq8_init_params(&model->rq8_params, (float)eff_rlim, dim);

    // Path 1: Load precomputed sidecar file if path specified
    if (config->rq8_load_path != NULL)
    {
        long loaded_frames = 0;
        if (rq8_load_sidecar(config->rq8_load_path, &model->rq8_params,
                             &model->rq8_dataset_buffer, &loaded_frames) != 0)
        {
            fprintf(stderr, "Error: Failed to load RQ8 sidecar file '%s'\n",
                    config->rq8_load_path);
            return -1;
        }
        if (loaded_frames != N || model->rq8_params.dim != dim)
        {
            fprintf(stderr,
                    "Error: RQ8 sidecar mismatch (loaded %ld frames, dim %ld; expected %ld, %ld)\n",
                    loaded_frames, model->rq8_params.dim, N, dim);
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
                rq8_quantize_residual_double(src, (const double *)anchor, dst, &model->rq8_params);
            }
            else
            {
                const float *src = (const float *)model->dataset_buffer + (size_t)i * (size_t)dim;
                rq8_quantize_residual_float(src, (const float *)anchor, dst, &model->rq8_params);
            }
        } // for (long i = 0; i < N; i++)
    }
    else
    {
        // Reader fallback for non-cached dataset
        KnnFrameReader reader;
        if (knn_reader_open(&reader, config->input_data_path, model->is_fits_input,
                            model->frame_width, model->frame_height, model->is_double) != 0)
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
            if (knn_reader_read_frame(&reader, i, frame_buf) == 0)
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
                    rq8_quantize_residual_double((const double *)frame_buf,
                                                 (const double *)anchor, dst, &model->rq8_params);
                }
                else
                {
                    rq8_quantize_residual_float((const float *)frame_buf,
                                                (const float *)anchor, dst, &model->rq8_params);
                }
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
                             model->rq8_dataset_buffer, N) == 0)
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

    return 0;
}

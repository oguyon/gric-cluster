/**
 * @file knn_cache_layout.c
 * @brief Memory layout reorganizations for k-NN: FastScan transposed blocks and IVF buffers.
 */

#include "knn_cache_layout.h"
#include "scalar_quant.h"
#include "residual_quant.h"
#include "product_quant.h"
#include "rabit_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

/**
 * knn_size_mul() - Multiply two size_t values with overflow checking.
 * @a:   First factor.
 * @b:   Second factor.
 * @out: Pointer to destination size_t.
 *
 * Return: 0 on success, -1 on overflow.
 */
static int knn_size_mul(
    size_t  a,
    size_t  b,
    size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a)
    {
        return -1;
    }
    *out = a * b;
    return 0;
}

/**
 * knn_size_add() - Add two size_t values with overflow checking.
 * @a:   First term.
 * @b:   Second term.
 * @out: Pointer to destination size_t.
 *
 * Return: 0 on success, -1 on overflow.
 */
static int knn_size_add(
    size_t  a,
    size_t  b,
    size_t *out)
{
    if (SIZE_MAX - a < b)
    {
        return -1;
    }
    *out = a + b;
    return 0;
}

/**
 * knn_alloc_sq16_transposed_buffer() - Allocate aligned buffer for SQ16 transposed blocks.
 * @buffer: Output pointer to allocated int16_t buffer.
 * @bytes:  Number of bytes to allocate.
 *
 * Return: 0 on success, -1 on failure.
 */
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

/**
 * knn_alloc_rq8_transposed_buffer() - Allocate aligned buffer for RQ8 transposed blocks.
 * @buffer: Output pointer to allocated int8_t buffer.
 * @bytes:  Number of bytes to allocate.
 *
 * Return: 0 on success, -1 on failure.
 */
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

    if (config->use_sq16_sparse)
    {
        for (int c = 0; c < M; c++)
        {
            int num_m = model->clusters[c].num_members;
            if (num_m > 0)
            {
                model->clusters[c].num_sq16_blocks =
                    (num_m + SQ16_FASTSCAN_BLOCK_SIZE - 1) / SQ16_FASTSCAN_BLOCK_SIZE;
            }
            else
            {
                model->clusters[c].num_sq16_blocks = 0;
            }
            model->clusters[c].sq16_transposed = NULL;
        }
        if (config->verbose_level >= 1)
        {
            printf("  [FASTSCAN] SQ16 SparseCache Active: 0.00 MB resident index\n"
                   "             (On-demand 32 KB scratchpad per OpenMP thread)\n");
        }
        return 0;
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
 * knn_model_build_transposed_pq() - Build cluster-local transposed PQ FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_pq(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_pq ||
        model->pq_dataset_buffer == NULL || model->pq_codebook == NULL)
    {
        return 0;
    }

    int M = model->num_clusters;
    int m = model->pq_codebook->m;
    if (M <= 0 || m <= 0)
    {
        return 0;
    }

    size_t total_transposed_bytes = 0;
    if (model->pq_transposed_buffer != NULL)
    {
        free(model->pq_transposed_buffer);
        model->pq_transposed_buffer = NULL;
    }

    for (int c = 0; c < M; c++)
    {
        int num_m = model->clusters[c].num_members;
        int n_blocks = (num_m + PQ_FASTSCAN_BLOCK_SIZE - 1) / PQ_FASTSCAN_BLOCK_SIZE;
        model->clusters[c].pq_transposed = NULL;
        model->clusters[c].num_pq_blocks = n_blocks;
        total_transposed_bytes += (size_t)n_blocks * (size_t)m * PQ_FASTSCAN_BLOCK_SIZE;
    } // for (int c = 0; c < M; c++)

    if (total_transposed_bytes == 0)
    {
        return 0;
    }

    model->pq_transposed_buffer = (uint8_t *)malloc(total_transposed_bytes);
    if (model->pq_transposed_buffer == NULL)
    {
        return -1;
    }

    size_t cur_offset = 0;
    uint8_t *temp_row_codes = (uint8_t *)malloc((size_t)PQ_FASTSCAN_BLOCK_SIZE * (size_t)m);
    if (temp_row_codes == NULL)
    {
        free(model->pq_transposed_buffer);
        model->pq_transposed_buffer = NULL;
        return -1;
    }

    for (int c = 0; c < M; c++)
    {
        int n_blocks = model->clusters[c].num_pq_blocks;
        if (n_blocks <= 0)
        {
            continue;
        }

        uint8_t *cl_buf = model->pq_transposed_buffer + cur_offset;
        model->clusters[c].pq_transposed = cl_buf;
        cur_offset += (size_t)n_blocks * (size_t)m * PQ_FASTSCAN_BLOCK_SIZE;

        int num_m = model->clusters[c].num_members;
        for (int b = 0; b < n_blocks; b++)
        {
            int m_start = b * PQ_FASTSCAN_BLOCK_SIZE;
            int m_count = num_m - m_start;
            if (m_count > PQ_FASTSCAN_BLOCK_SIZE)
            {
                m_count = PQ_FASTSCAN_BLOCK_SIZE;
            }

            for (int i = 0; i < PQ_FASTSCAN_BLOCK_SIZE; i++)
            {
                uint8_t *dst_code = temp_row_codes + i * m;
                if (i < m_count)
                {
                    long cand_id = (long)model->clusters[c].members[m_start + i].frame_id;
                    const uint8_t *src_code = model->pq_dataset_buffer + cand_id * m;
                    memcpy(dst_code, src_code, (size_t)m);
                }
                else
                {
                    memset(dst_code, 0, (size_t)m);
                }
            } // for (int i = 0; i < PQ_FASTSCAN_BLOCK_SIZE; i++)

            uint8_t *dst_block = cl_buf + (size_t)b * (size_t)m * PQ_FASTSCAN_BLOCK_SIZE;
            pq_transpose_block_codes(temp_row_codes, dst_block, m, m_count);
        } // for (int b = 0; b < n_blocks; b++)
    } // for (int c = 0; c < M; c++)

    free(temp_row_codes);

    if (config->verbose_level >= 1)
    {
        double mb = (double)total_transposed_bytes / (1024.0 * 1024.0);
        printf("  [FASTSCAN] Built PQ SIMD transposed blocks (%s): %.2f MB\n",
               pq_get_simd_mode_str(), mb);
    }

    return 0;
}

/**
 * knn_model_build_transposed_rabitq() - Build cluster-local transposed RaBitQ FastScan blocks.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_transposed_rabitq(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || config == NULL || !config->use_rabitq ||
        model->rabitq_dataset_buffer == NULL || model->rabitq_meta_buffer == NULL)
    {
        return 0;
    }

    int M = model->num_clusters;
    long dim_pad = model->rabitq_params.dim_pad;
    int num_nibbles = (model->rabitq_params.bits == 2) ?
        (int)(dim_pad / 2) : (int)(dim_pad / 4);
    if (M <= 0 || num_nibbles <= 0)
    {
        return 0;
    }

    size_t total_transposed_bytes = 0;
    if (model->rabitq_transposed_buffer != NULL)
    {
        free(model->rabitq_transposed_buffer);
        model->rabitq_transposed_buffer = NULL;
    }

    for (int c = 0; c < M; c++)
    {
        int num_m = model->clusters[c].num_members;
        int n_blocks = (num_m + RABITQ_FASTSCAN_BLOCK_SIZE - 1) / RABITQ_FASTSCAN_BLOCK_SIZE;
        model->clusters[c].num_rabitq_blocks = n_blocks;
        total_transposed_bytes += (size_t)n_blocks * (size_t)num_nibbles * 16;
    } // for (int c = 0; c < M; c++)

    if (total_transposed_bytes == 0)
    {
        return 0;
    }

    model->rabitq_transposed_buffer = (uint8_t *)malloc(total_transposed_bytes);
    if (model->rabitq_transposed_buffer == NULL)
    {
        return -1;
    }

    size_t cur_offset = 0;
    size_t code_bytes = model->rabitq_params.code_bytes_per_vec;

    for (int c = 0; c < M; c++)
    {
        int n_blocks = model->clusters[c].num_rabitq_blocks;
        int num_m = model->clusters[c].num_members;
        if (n_blocks <= 0 || num_m <= 0)
        {
            model->clusters[c].rabitq_transposed = NULL;
            model->clusters[c].rabitq_meta = NULL;
            continue;
        }

        uint8_t *cl_buf = model->rabitq_transposed_buffer + cur_offset;
        model->clusters[c].rabitq_transposed = cl_buf;
        cur_offset += (size_t)n_blocks * (size_t)num_nibbles * 16;

        model->clusters[c].rabitq_meta = (RaBitQMeta *)malloc(
            (size_t)n_blocks * RABITQ_FASTSCAN_BLOCK_SIZE * sizeof(RaBitQMeta)
        );
        if (model->clusters[c].rabitq_meta == NULL)
        {
            return -1;
        }

        for (int b = 0; b < n_blocks; b++)
        {
            int m_start = b * RABITQ_FASTSCAN_BLOCK_SIZE;
            uint8_t *b_ptr = cl_buf + (size_t)b * (size_t)num_nibbles * 16;
            memset(b_ptr, 0, (size_t)num_nibbles * 16);

            for (int i = 0; i < RABITQ_FASTSCAN_BLOCK_SIZE; i++)
            {
                int m_idx = m_start + i;
                RaBitQMeta *dst_meta = &model->clusters[c].rabitq_meta[m_start + i];
                if (m_idx < num_m)
                {
                    long cand_id = (long)model->clusters[c].members[m_idx].frame_id;
                    *dst_meta = model->rabitq_meta_buffer[cand_id];

                    const uint8_t *src_code = model->rabitq_dataset_buffer +
                                              cand_id * code_bytes;

                    for (int nb = 0; nb < num_nibbles; nb++)
                    {
                        uint8_t byte_val = src_code[nb >> 1];
                        uint8_t nib = (nb & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);
                        if (i < 16)
                        {
                            b_ptr[nb * 16 + i] |= (uint8_t)(nib & 0x0F);
                        }
                        else
                        {
                            b_ptr[nb * 16 + (i - 16)] |= (uint8_t)((nib & 0x0F) << 4);
                        }
                    } // for (int nb = 0; nb < num_nibbles; nb++)
                }
                else
                {
                    dst_meta->norm = 1e30f;
                    dst_meta->recon_scale = 0.0f;
                    dst_meta->err_norm = 0.0f;
                }
            } // for (int i = 0; i < RABITQ_FASTSCAN_BLOCK_SIZE; i++)
        } // for (int b = 0; b < n_blocks; b++)
    } // for (int c = 0; c < M; c++)

    if (config->verbose_level >= 1)
    {
        double mb = (double)total_transposed_bytes / (1024.0 * 1024.0);
        printf("  [FASTSCAN] Built RaBitQ SIMD transposed blocks (%s): %.2f MB\n",
               rabitq_get_simd_mode_str(), mb);
    }

    return 0;
}

/**
 * knn_model_build_ivf_layout() - Reorganize dataset into contiguous per-cluster IVF layout.
 * @model:  Pointer to initialized KnnModel.
 * @config: Pointer to KnnConfig.
 *
 * Return: 0 on success, -1 on failure.
 */
int knn_model_build_ivf_layout(
    KnnModel        *model,
    const KnnConfig *config)
{
    if (model == NULL || model->dataset_buffer == NULL ||
        model->clusters == NULL || model->num_clusters <= 0)
    {
        return 0;
    }

    if (model->ivf_dataset_buffer != NULL)
    {
        free(model->ivf_dataset_buffer);
        model->ivf_dataset_buffer = NULL;
    }

    int M = model->num_clusters;
    long dim = model->frame_elements;
    size_t elem_size = model->is_double ? sizeof(double) : sizeof(float);
    size_t frame_bytes = (size_t)dim * elem_size;

    size_t total_members = 0;
    for (int c = 0; c < M; c++)
    {
        total_members += (size_t)model->clusters[c].num_members;
    }

    if (total_members == 0)
    {
        return 0;
    }

    size_t total_bytes = total_members * frame_bytes;
    void *buf = NULL;
    if (posix_memalign(&buf, 64, total_bytes) != 0 || buf == NULL)
    {
        return -1;
    }

    model->ivf_dataset_buffer = buf;
    char *cursor = (char *)buf;

    for (int c = 0; c < M; c++)
    {
        KnnCluster *cl = &model->clusters[c];
        int num_m = cl->num_members;
        if (num_m <= 0)
        {
            cl->ivf_vectors = NULL;
            continue;
        }

        cl->ivf_vectors = (void *)cursor;
        for (int m = 0; m < num_m; m++)
        {
            uint32_t fid = cl->members[m].frame_id;
            const char *src = (const char *)model->dataset_buffer + (size_t)fid * frame_bytes;
            memcpy(cursor + (size_t)m * frame_bytes, src, frame_bytes);
        }
        cursor += (size_t)num_m * frame_bytes;
    }

    if (config != NULL && config->verbose_level >= 1)
    {
        printf("Built CPU contiguous IVF layout: %zu frames (%.2f MB)\n",
               total_members, (double)total_bytes / (1024.0 * 1024.0));
    }

    return 0;
}

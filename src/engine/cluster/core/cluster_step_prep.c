/**
 * @file cluster_step_prep.c
 * @brief Implementation of frame quantization buffer preparation and prediction retrieval.
 */

#define _POSIX_C_SOURCE 200809L

#include "cluster_step_prep.h"
#include "cluster_prune.h"
#include "scalar_quant.h"
#include "eq16_quant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * prepare_frame_quantization() - Prepares scalar quantization buffers and quantizes
 *                                the current frame if EQ16, SQ16, or SQ8 is active.
 * @config:        Pointer to clustering configuration.
 * @state:         Pointer to clustering state.
 * @current_frame: Pointer to input frame to quantize.
 *
 * Automatically detects and selects optimal quantization scheme if not explicitly set:
 * - D >= 8 and D % 8 == 0: EQ16 (16-bit E8 lattice quantization)
 * - D >= 32: SQ16 (16-bit scalar quantization)
 * - Otherwise: SQ8 (8-bit scalar quantization)
 */
void prepare_frame_quantization(
    ClusterConfig *config,
    ClusterState  *state,
    const Frame   *current_frame)
{
    if (config->optim.use_sq8 < 0 && config->optim.use_sq16 < 0 && config->optim.use_eq16 < 0)
    {
        long frame_dim = current_frame->width * current_frame->height;
        if (frame_dim >= 8 && (frame_dim % 8 == 0))
        {
            config->optim.use_eq16 = 1;
            config->optim.use_sq16 = 0;
            config->optim.use_sq8 = 0;
        }
        else if (frame_dim >= 32)
        {
            config->optim.use_sq16 = 1;
            config->optim.use_eq16 = 0;
            config->optim.use_sq8 = 0;
        }
        else
        {
            config->optim.use_sq8 = 1;
            config->optim.use_sq16 = 0;
            config->optim.use_eq16 = 0;
        }
    }
    else
    {
        if (config->optim.use_sq8 < 0)
        {
            config->optim.use_sq8 = 0;
        }
        if (config->optim.use_sq16 < 0)
        {
            config->optim.use_sq16 = 0;
        }
        if (config->optim.use_eq16 < 0)
        {
            config->optim.use_eq16 = 0;
        }
    }

    if (config->optim.use_eq16)
    {
        long frame_dim = current_frame->width * current_frame->height;
        if (state->current_frame_eq16 == NULL)
        {
            if (posix_memalign((void **)&state->current_frame_eq16, 64,
                               (size_t)frame_dim * sizeof(int16_t)) != 0)
            {
                state->current_frame_eq16 = NULL;
            }
        }
        if (state->current_frame_eq16_adc == NULL)
        {
            if (posix_memalign((void **)&state->current_frame_eq16_adc, 64,
                               (size_t)frame_dim * sizeof(float)) != 0)
            {
                state->current_frame_eq16_adc = NULL;
            }
        }
        if (state->anchor_matrix_eq16 == NULL)
        {
            size_t total_eq16 = (size_t)config->algo.maxnbclust * (size_t)frame_dim;
            if (posix_memalign((void **)&state->anchor_matrix_eq16, 64,
                               total_eq16 * sizeof(int16_t)) != 0)
            {
                state->anchor_matrix_eq16 = NULL;
            }
        }
        if (state->anchor_matrix_eq16_interleaved == NULL)
        {
            size_t num_blocks = ((size_t)config->algo.maxnbclust + 7) / 8;
            size_t num_pairs = ((size_t)frame_dim + 1) / 2;
            size_t total_interleaved = num_blocks * num_pairs * 8;
            if (posix_memalign((void **)&state->anchor_matrix_eq16_interleaved, 64,
                               total_interleaved * sizeof(int32_t)) != 0)
            {
                state->anchor_matrix_eq16_interleaved = NULL;
            }
            else
            {
                memset(state->anchor_matrix_eq16_interleaved, 0,
                       total_interleaved * sizeof(int32_t));
            }
        }
        if (state->anchor_matrix_adc_interleaved == NULL && config->optim.use_eq16_adc)
        {
            size_t num_blocks_adc = ((size_t)config->algo.maxnbclust + 15) / 16;
            size_t total_adc = num_blocks_adc * (size_t)frame_dim * 16;
            if (posix_memalign((void **)&state->anchor_matrix_adc_interleaved, 64,
                               total_adc * sizeof(float)) != 0)
            {
                state->anchor_matrix_adc_interleaved = NULL;
            }
            else
            {
                memset(state->anchor_matrix_adc_interleaved, 0,
                       total_adc * sizeof(float));
            }
        }
        if (!state->eq16_calibrated)
        {
            if (current_frame->is_double)
            {
                eq16_calibrate_double(&config->optim.eq16_params,
                                      (const double *)current_frame->data,
                                      frame_dim, frame_dim);
            }
            else
            {
                eq16_calibrate_float(&config->optim.eq16_params,
                                     (const float *)current_frame->data,
                                     frame_dim, frame_dim);
            }

            /* Enforce --sq16-ratio bound: scale = alpha*rlim / sqrt(D) */
            if (config->optim.sq16_ratio > 0.0 && config->algo.rlim > 0.0)
            {
                float target_scale = (float)((config->optim.sq16_ratio * config->algo.rlim) /
                                             sqrt((double)frame_dim));
                float center = 0.5f * (config->optim.eq16_params.min_val +
                                       config->optim.eq16_params.max_val);
                config->optim.eq16_params.scale = target_scale;
                config->optim.eq16_params.inv_scale = 1.0f / target_scale;
                config->optim.eq16_params.center = center;
                long k = frame_dim / 8;
                long rem = frame_dim % 8;
                float covering_sq = (float)k * 1.0f + (float)rem * 0.25f;
                config->optim.eq16_params.err_radius = sqrtf(covering_sq) * target_scale;
            }

            state->eq16_calibrated = 1;
        }

        if (state->perm_dim != NULL)
        {
            if (current_frame->is_double)
            {
                eq16_quantize_double_perm((const double *)current_frame->data,
                                          state->current_frame_eq16,
                                          &config->optim.eq16_params,
                                          state->perm_dim);
                if (state->current_frame_eq16_adc != NULL)
                {
                    eq16_prepare_query_adc_double_perm((const double *)current_frame->data,
                                                       state->current_frame_eq16_adc,
                                                       &config->optim.eq16_params,
                                                       state->perm_dim);
                }
            }
            else
            {
                eq16_quantize_float_perm((const float *)current_frame->data,
                                         state->current_frame_eq16,
                                         &config->optim.eq16_params,
                                         state->perm_dim);
                if (state->current_frame_eq16_adc != NULL)
                {
                    eq16_prepare_query_adc_float_perm((const float *)current_frame->data,
                                                      state->current_frame_eq16_adc,
                                                      &config->optim.eq16_params,
                                                      state->perm_dim);
                }
            }
        }
        else
        {
            if (current_frame->is_double)
            {
                eq16_quantize_double((const double *)current_frame->data,
                                     state->current_frame_eq16,
                                     &config->optim.eq16_params);
                if (state->current_frame_eq16_adc != NULL)
                {
                    eq16_prepare_query_adc_double((const double *)current_frame->data,
                                                  state->current_frame_eq16_adc,
                                                  &config->optim.eq16_params);
                }
            }
            else
            {
                eq16_quantize_float((const float *)current_frame->data,
                                    state->current_frame_eq16,
                                    &config->optim.eq16_params);
                if (state->current_frame_eq16_adc != NULL)
                {
                    eq16_prepare_query_adc_float((const float *)current_frame->data,
                                                 state->current_frame_eq16_adc,
                                                 &config->optim.eq16_params);
                }
            }
        }
    }
    else if (config->optim.use_sq16)
    {
        long frame_dim = current_frame->width * current_frame->height;
        if (state->current_frame_sq16 == NULL)
        {
            if (posix_memalign((void **)&state->current_frame_sq16, 64,
                               (size_t)frame_dim * sizeof(int16_t)) != 0)
            {
                state->current_frame_sq16 = NULL;
            }
        }
        if (state->anchor_matrix_sq16 == NULL)
        {
            size_t total_sq16 = (size_t)config->algo.maxnbclust * (size_t)frame_dim;
            if (posix_memalign((void **)&state->anchor_matrix_sq16, 64,
                               total_sq16 * sizeof(int16_t)) != 0)
            {
                state->anchor_matrix_sq16 = NULL;
            }
        }
        if (state->anchor_matrix_sq16_interleaved == NULL)
        {
            size_t num_blocks = ((size_t)config->algo.maxnbclust + 7) / 8;
            size_t num_pairs = ((size_t)frame_dim + 1) / 2;
            size_t total_interleaved = num_blocks * num_pairs * 8;
            if (posix_memalign((void **)&state->anchor_matrix_sq16_interleaved, 64,
                               total_interleaved * sizeof(int32_t)) != 0)
            {
                state->anchor_matrix_sq16_interleaved = NULL;
            }
            else
            {
                memset(state->anchor_matrix_sq16_interleaved, 0,
                       total_interleaved * sizeof(int32_t));
            }
        }
        if (!state->sq16_calibrated)
        {
            if (current_frame->is_double)
            {
                sq16_calibrate_double(&config->optim.sq16_params,
                                      (const double *)current_frame->data,
                                      frame_dim, frame_dim);
            }
            else
            {
                sq16_calibrate_float(&config->optim.sq16_params,
                                     (const float *)current_frame->data,
                                     frame_dim, frame_dim);
            }

            /* Enforce --sq16-ratio bound: scale = alpha*rlim / sqrt(D) */
            if (config->optim.sq16_ratio > 0.0 && config->algo.rlim > 0.0)
            {
                float target_scale = (float)((config->optim.sq16_ratio * config->algo.rlim) /
                                             sqrt((double)frame_dim));
                float center = 0.5f * (config->optim.sq16_params.min_val +
                                       config->optim.sq16_params.max_val);
                config->optim.sq16_params.scale = target_scale;
                config->optim.sq16_params.inv_scale = 1.0f / target_scale;
                config->optim.sq16_params.min_val = center - 16384.0f * target_scale;
                config->optim.sq16_params.max_val = center + 16383.0f * target_scale;
                config->optim.sq16_params.err_radius =
                    sqrtf((float)frame_dim) * target_scale * 0.5f;
            }

            if (config->optim.use_memo)
            {
                state->scratch.memo_table.err_radius =
                    (double)config->optim.sq16_params.err_radius;
                state->scratch.memo_table.rlim = config->algo.rlim;
            }

            state->sq16_calibrated = 1;
        }
        if (config->optim.use_memo && state->scratch.memo_table.rlim == 0.0)
        {
            state->scratch.memo_table.err_radius =
                (double)config->optim.sq16_params.err_radius;
            state->scratch.memo_table.rlim = config->algo.rlim;
        }

        if (state->perm_dim != NULL)
        {
            if (current_frame->is_double)
            {
                sq16_quantize_double_perm((const double *)current_frame->data,
                                          state->current_frame_sq16,
                                          &config->optim.sq16_params,
                                          state->perm_dim);
            }
            else
            {
                sq16_quantize_float_perm((const float *)current_frame->data,
                                         state->current_frame_sq16,
                                         &config->optim.sq16_params,
                                         state->perm_dim);
            }
        }
        else
        {
            if (current_frame->is_double)
            {
                sq16_quantize_double((const double *)current_frame->data,
                                     state->current_frame_sq16,
                                     &config->optim.sq16_params);
            }
            else
            {
                sq16_quantize_float((const float *)current_frame->data,
                                    state->current_frame_sq16,
                                    &config->optim.sq16_params);
            }
        }
    }
    else if (config->optim.use_sq8)
    {
        long frame_dim = current_frame->width * current_frame->height;
        if (state->current_frame_sq8 == NULL)
        {
            if (posix_memalign((void **)&state->current_frame_sq8, 64,
                               (size_t)frame_dim) != 0)
            {
                state->current_frame_sq8 = NULL;
            }
        }
        if (state->anchor_matrix_sq8 == NULL)
        {
            size_t total_sq8 = (size_t)config->algo.maxnbclust * (size_t)frame_dim;
            if (posix_memalign((void **)&state->anchor_matrix_sq8, 64,
                               total_sq8 * sizeof(uint8_t)) != 0)
            {
                state->anchor_matrix_sq8 = NULL;
            }
        }
        if (!state->sq8_calibrated)
        {
            if (current_frame->is_double)
            {
                sq8_calibrate_double(&config->optim.sq8_params,
                                     (const double *)current_frame->data,
                                     frame_dim, frame_dim);
            }
            else
            {
                sq8_calibrate_float(&config->optim.sq8_params,
                                    (const float *)current_frame->data,
                                     frame_dim, frame_dim);
            }
            state->sq8_calibrated = 1;
        }
        if (current_frame->is_double)
        {
            sq8_quantize_double((const double *)current_frame->data,
                                state->current_frame_sq8,
                                &config->optim.sq8_params);
        }
        else
        {
            sq8_quantize_float((const float *)current_frame->data,
                               state->current_frame_sq8,
                               &config->optim.sq8_params);
        }
    }

    if (!current_frame->is_double && state->anchor_matrix_float == NULL)
    {
        long frame_dim = current_frame->width * current_frame->height;
        size_t total_float = (size_t)config->algo.maxnbclust * (size_t)frame_dim;
        if (posix_memalign((void **)&state->anchor_matrix_float, 64,
                           total_float * sizeof(float)) != 0)
        {
            state->anchor_matrix_float = NULL;
        }
    }

    if (!current_frame->is_double && state->anchor_norms_float == NULL)
    {
        size_t total_norms = (size_t)config->algo.maxnbclust * sizeof(float);
        if (posix_memalign((void **)&state->anchor_norms_float, 64, total_norms) != 0)
        {
            state->anchor_norms_float = NULL;
        }
    }
}

/**
 * retrieve_prediction_candidates() - Retrieves prediction candidates into scratch buffer.
 * @config:          Pointer to clustering configuration.
 * @state:           Pointer to clustering state with pre-allocated scratch buffers.
 * @pred_candidates: Pointer to pre-allocated output array of candidate cluster IDs.
 *
 * Combines local trajectory predictions with joint-tuple multi-tile predictions.
 *
 * Return: Number of prediction candidates populated in @pred_candidates.
 */
int retrieve_prediction_candidates(
    ClusterConfig *config,
    ClusterState  *state,
    int           *pred_candidates)
{
    int num_preds = 0;
    if (!config->optim.pred_mode ||
        state->telemetry.total_frames_processed < config->optim.pred_len)
    {
        return 0;
    }

    int *local_candidates = state->scratch.local_candidates;
    int num_local = 0;
    if (local_candidates != NULL)
    {
        num_local = get_prediction_candidates(state, config, local_candidates,
                                              config->optim.pred_n);
    }

    if (pred_candidates == NULL)
    {
        return 0;
    }

    if (num_local == 1)
    {
        /* Unambiguous local prediction: prioritize it first */
        pred_candidates[num_preds++] = local_candidates[0];

        /* Append joint predictions as fallback */
        if (state->scratch.tuple_pred_count > 0)
        {
            for (int j = 0; j < state->scratch.tuple_pred_count &&
                 num_preds < config->optim.pred_n; j++)
            {
                int jc = state->scratch.tuple_pred_candidates[j];
                if (jc != local_candidates[0])
                {
                    pred_candidates[num_preds++] = jc;
                }
            }
        }
    }
    else
    {
        /* Ambiguous or no local match: prioritize joint predictions to resolve it */
        if (state->scratch.tuple_pred_count > 0)
        {
            int n_out = (state->scratch.tuple_pred_count < config->optim.pred_n) ?
                        state->scratch.tuple_pred_count : config->optim.pred_n;
            for (int i = 0; i < n_out; i++)
            {
                pred_candidates[num_preds++] = state->scratch.tuple_pred_candidates[i];
            }
        }
        if (num_preds < config->optim.pred_n && num_local > 0)
        {
            for (int i = 0; i < num_local && num_preds < config->optim.pred_n; i++)
            {
                int lc = local_candidates[i];
                int dup = 0;
                for (int k = 0; k < num_preds; k++)
                {
                    if (pred_candidates[k] == lc)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                {
                    pred_candidates[num_preds++] = lc;
                }
            }
        }
    }

    return num_preds;
}

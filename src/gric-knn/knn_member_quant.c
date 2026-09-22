/**
 * @file knn_member_quant.c
 * @brief Quantization-accelerated member candidate search routines (RaBitQ, PQ, RQ8, EQ16, SQ16).
 */

#include "knn_member_quant.h"
#include "scalar_quant.h"
#include "residual_quant.h"
#include "product_quant.h"
#include "rabit_quant.h"
#include "eq16_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * knn_eval_members_rabitq() - Evaluate cluster members using RaBitQ FastScan.
 */
void knn_eval_members_rabitq(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    int num_b = cl->num_rabitq_blocks;
    long frame_elem = model->frame_elements;
    int num_nibbles = (model->rabitq_params.bits == 2) ?
        (int)(model->rabitq_params.dim_pad / 2) :
        (int)(model->rabitq_params.dim_pad / 4);
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, RABITQ_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * RABITQ_FASTSCAN_BLOCK_SIZE + RABITQ_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * RABITQ_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && d_left_b >= tau_thresh)
        {
            int pruned = (left_b + 1) * RABITQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && d_right_b >= tau_thresh)
        {
            int pruned = num_m - right_b * RABITQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * RABITQ_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > RABITQ_FASTSCAN_BLOCK_SIZE)
        {
            m_count = RABITQ_FASTSCAN_BLOCK_SIZE;
        }

        const uint8_t *b_codes = cl->rabitq_transposed +
            (size_t)b * (size_t)num_nibbles * 16;
        const RaBitQMeta *b_meta = &cl->rabitq_meta[m_start];

        telem->rabitq_evaluations += (uint64_t)m_count;

        uint32_t pass_mask = rabitq_fastscan_32x(
            &visited->query_rabitq_lut,
            b_codes,
            b_meta,
            tau_thresh,
            config->epsilon
        );
        if (m_count < RABITQ_FASTSCAN_BLOCK_SIZE)
        {
            pass_mask &= ((1U << m_count) - 1);
        }

        if (!pass_mask)
        {
            telem->rabitq_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->rabitq_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
            {
                continue;
            }

            void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
            const void *cand_ptr = knn_resolve_candidate_data(
                cand_id, m, cl, model, reader, slot, frame_bytes
            );
            if (cand_ptr == NULL)
            {
                continue;
            }

            knn_append_or_eval_candidate(
                cand_id, cand_ptr, query_id, query_data, model, config,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                batch, telem);
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_pq() - Evaluate cluster members using Product Quantization FastScan.
 */
void knn_eval_members_pq(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    int num_b = cl->num_pq_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, PQ_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * PQ_FASTSCAN_BLOCK_SIZE + PQ_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * PQ_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && d_left_b >= tau_thresh)
        {
            int pruned = (left_b + 1) * PQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && d_right_b >= tau_thresh)
        {
            int pruned = num_m - right_b * PQ_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * PQ_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > PQ_FASTSCAN_BLOCK_SIZE)
        {
            m_count = PQ_FASTSCAN_BLOCK_SIZE;
        }

        const uint8_t *b_codes = cl->pq_transposed +
            (size_t)b * (size_t)model->pq_codebook->m * PQ_FASTSCAN_BLOCK_SIZE;
        telem->pq_evaluations += (uint64_t)m_count;

        uint8_t cutoff_u8 = 255;
        if (current_tau > 0.0)
        {
            double eff_tau = current_tau / eps_factor;
            double tau_scaled = eff_tau * eff_tau * (double)visited->query_pq_table.scale;
            if (tau_scaled < 255.0)
            {
                cutoff_u8 = (uint8_t)tau_scaled;
            }
        }

        uint32_t pass_mask = pq_fastscan_32x(
            visited->query_pq_table.lut_u8,
            b_codes,
            model->pq_codebook->m,
            cutoff_u8
        );
        if (m_count < PQ_FASTSCAN_BLOCK_SIZE)
        {
            pass_mask &= ((1U << m_count) - 1);
        }

        if (!pass_mask)
        {
            telem->pq_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->pq_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
            {
                continue;
            }

            void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
            const void *cand_ptr = knn_resolve_candidate_data(
                cand_id, m, cl, model, reader, slot, frame_bytes
            );
            if (cand_ptr == NULL)
            {
                continue;
            }

            knn_append_or_eval_candidate(
                cand_id, cand_ptr, query_id, query_data, model, config,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                batch, telem);
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_rq8_blocks() - Evaluate cluster members using RQ8 FastScan blocks.
 */
void knn_eval_members_rq8_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    int num_b = cl->num_rq8_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;
    float cached_cutoff_adc = 1e30f;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, RQ8_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * RQ8_FASTSCAN_BLOCK_SIZE + RQ8_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * RQ8_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * RQ8_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * RQ8_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * RQ8_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > RQ8_FASTSCAN_BLOCK_SIZE)
        {
            m_count = RQ8_FASTSCAN_BLOCK_SIZE;
        }

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
            {
                cached_cutoff_adc = compute_rq8_cutoff_thresh_adc_cluster(
                    current_tau, &cl->rq8_params, config
                );
            }
            else
            {
                cached_ssd_cutoff = compute_rq8_cutoff_thresh_cluster(
                    current_tau, &cl->rq8_params, config
                );
            }
        }

        uint32_t pass_mask = 0;
        if (cl->rq8_transposed != NULL)
        {
            const int8_t *b_coords = cl->rq8_transposed +
                (size_t)b * (size_t)frame_elem * RQ8_FASTSCAN_BLOCK_SIZE;
            telem->rq8_evaluations += (uint64_t)m_count;

            if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
            {
                pass_mask = rq8_fastscan_32x_adc(
                    visited->query_rq8_adc, b_coords, frame_elem, cached_cutoff_adc
                );
            }
            else if (visited->query_rq8 != NULL)
            {
                pass_mask = rq8_fastscan_32x(
                    visited->query_rq8, b_coords, frame_elem, cached_ssd_cutoff
                );
            }

            if (m_count < RQ8_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_rq8_sparse &&
                 (cl->rq8_vectors != NULL || model->rq8_dataset_buffer != NULL))
        {
            telem->rq8_evaluations += (uint64_t)m_count;
            if (config->use_rq8_adc && visited->query_rq8_adc != NULL)
            {
                int i = 0;
                if (cl->rq8_vectors != NULL)
                {
                    const int8_t *cl_cands = cl->rq8_vectors +
                        (size_t)m_start * (size_t)frame_elem;
                    for (; i <= m_count - 4; i += 4)
                    {
                        const int8_t *cands[4];
                        const int8_t *p = cl_cands + (size_t)i * (size_t)frame_elem;
                        cands[0] = p;
                        cands[1] = p + frame_elem;
                        cands[2] = p + 2 * frame_elem;
                        cands[3] = p + 3 * frame_elem;

                        float dsq[4];
                        rq8_dist_asym_cutoff_batch_1x4(
                            visited->query_rq8_adc, cands, frame_elem,
                            cached_cutoff_adc, dsq
                        );
                        if (dsq[0] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 0));
                        }
                        if (dsq[1] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 1));
                        }
                        if (dsq[2] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 2));
                        }
                        if (dsq[3] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 3));
                        }
                    } // for (; i <= m_count - 4; i += 4)

                    for (; i < m_count; i++)
                    {
                        const int8_t *cand_rq8 = cl_cands + (size_t)i * (size_t)frame_elem;
                        float dist_sq = rq8_dist_asym_cutoff_f32(
                            visited->query_rq8_adc, cand_rq8, frame_elem,
                            cached_cutoff_adc
                        );
                        if (dist_sq <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
                else
                {
                    for (; i <= m_count - 4; i += 4)
                    {
                        const int8_t *cands[4];
                        cands[0] = model->rq8_dataset_buffer +
                            (size_t)cl->members[m_start + i + 0].frame_id * (size_t)frame_elem;
                        cands[1] = model->rq8_dataset_buffer +
                            (size_t)cl->members[m_start + i + 1].frame_id * (size_t)frame_elem;
                        cands[2] = model->rq8_dataset_buffer +
                            (size_t)cl->members[m_start + i + 2].frame_id * (size_t)frame_elem;
                        cands[3] = model->rq8_dataset_buffer +
                            (size_t)cl->members[m_start + i + 3].frame_id * (size_t)frame_elem;

                        float dsq[4];
                        rq8_dist_asym_cutoff_batch_1x4(
                            visited->query_rq8_adc, cands, frame_elem,
                            cached_cutoff_adc, dsq
                        );
                        if (dsq[0] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 0));
                        }
                        if (dsq[1] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 1));
                        }
                        if (dsq[2] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 2));
                        }
                        if (dsq[3] <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << (i + 3));
                        }
                    } // for (; i <= m_count - 4; i += 4)

                    for (; i < m_count; i++)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        const int8_t *cand_rq8 = model->rq8_dataset_buffer +
                            (size_t)cand_id * (size_t)frame_elem;
                        float dist_sq = rq8_dist_asym_cutoff_f32(
                            visited->query_rq8_adc, cand_rq8, frame_elem,
                            cached_cutoff_adc
                        );
                        if (dist_sq <= cached_cutoff_adc)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
            }
            else if (visited->query_rq8 != NULL)
            {
                if (cl->rq8_vectors != NULL)
                {
                    const int8_t *cl_cands = cl->rq8_vectors +
                        (size_t)m_start * (size_t)frame_elem;
                    for (int i = 0; i < m_count; i++)
                    {
                        const int8_t *cand_rq8 = cl_cands + (size_t)i * (size_t)frame_elem;
                        uint64_t ssd = rq8_dist_squared_cutoff_i8(
                            visited->query_rq8, cand_rq8, frame_elem, cached_ssd_cutoff
                        );
                        if (ssd <= cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < m_count; i++)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        const int8_t *cand_rq8 = model->rq8_dataset_buffer +
                            (size_t)cand_id * (size_t)frame_elem;
                        uint64_t ssd = rq8_dist_squared_cutoff_i8(
                            visited->query_rq8, cand_rq8, frame_elem, cached_ssd_cutoff
                        );
                        if (ssd <= cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
            }
        }

        if (!pass_mask)
        {
            telem->rq8_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->rq8_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
            {
                continue;
            }

            void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
            const void *cand_ptr = knn_resolve_candidate_data(
                cand_id, m, cl, model, reader, slot, frame_bytes
            );
            if (cand_ptr == NULL)
            {
                continue;
            }

            knn_append_or_eval_candidate(
                cand_id, cand_ptr, query_id, query_data, model, config,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                batch, telem);
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_eq16_blocks() - Evaluate cluster members using EQ16 FastScan blocks.
 */
void knn_eval_members_eq16_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    int num_b = cl->num_eq16_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, EQ16_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * EQ16_FASTSCAN_BLOCK_SIZE + EQ16_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * EQ16_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * EQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * EQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * EQ16_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > EQ16_FASTSCAN_BLOCK_SIZE)
        {
            m_count = EQ16_FASTSCAN_BLOCK_SIZE;
        }

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            cached_ssd_cutoff = compute_eq16_cutoff_thresh(current_tau, model, config);
        }

        uint32_t pass_mask = 0;
        if (cl->eq16_transposed != NULL)
        {
            const int16_t *b_coords = cl->eq16_transposed +
                (size_t)b * (size_t)frame_elem * EQ16_FASTSCAN_BLOCK_SIZE;
            telem->eq16_evaluations += (uint64_t)m_count;

            if (config->use_eq16_adc && visited->query_eq16_adc != NULL)
            {
                pass_mask = eq16_fastscan_32x_adc(
                    visited->query_eq16_adc, b_coords, frame_elem, (float)cached_ssd_cutoff
                );
            }
            else if (visited->query_eq16 != NULL)
            {
                pass_mask = eq16_fastscan_32x_i16(
                    visited->query_eq16, b_coords, frame_elem, cached_ssd_cutoff
                );
            }

            if (m_count < EQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_eq16_sparse &&
                 (cl->eq16_vectors != NULL || model->eq16_dataset_buffer != NULL))
        {
            telem->eq16_evaluations += (uint64_t)m_count;
            if (config->use_eq16_adc && visited->query_eq16_adc != NULL)
            {
                int i = 0;
                if (cl->eq16_vectors != NULL)
                {
                    const int16_t *cl_cands = cl->eq16_vectors +
                        (size_t)m_start * (size_t)frame_elem;

                    for (; i <= m_count - 8; i += 8)
                    {
                        const int16_t *cands[8];
                        const int16_t *p = cl_cands + (size_t)i * (size_t)frame_elem;
                        cands[0] = p;
                        cands[1] = p + frame_elem;
                        cands[2] = p + 2 * frame_elem;
                        cands[3] = p + 3 * frame_elem;
                        cands[4] = p + 4 * frame_elem;
                        cands[5] = p + 5 * frame_elem;
                        cands[6] = p + 6 * frame_elem;
                        cands[7] = p + 7 * frame_elem;

                        float dsq[8];
                        eq16_dist_asym_cutoff_batch_1x8(
                            visited->query_eq16_adc, cands, frame_elem,
                            (float)cached_ssd_cutoff, dsq
                        );
                        for (int k = 0; k < 8; k++)
                        {
                            if (dsq[k] <= (float)cached_ssd_cutoff)
                            {
                                pass_mask |= (1U << (i + k));
                            }
                        }
                    } // for (; i <= m_count - 8; i += 8)

                    for (; i <= m_count - 4; i += 4)
                    {
                        const int16_t *cands[4];
                        const int16_t *p = cl_cands + (size_t)i * (size_t)frame_elem;
                        cands[0] = p;
                        cands[1] = p + frame_elem;
                        cands[2] = p + 2 * frame_elem;
                        cands[3] = p + 3 * frame_elem;

                        float dsq[4];
                        eq16_dist_asym_cutoff_batch_1x4(
                            visited->query_eq16_adc, cands, frame_elem,
                            (float)cached_ssd_cutoff, dsq
                        );
                        if (dsq[0] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 0));
                        }
                        if (dsq[1] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 1));
                        }
                        if (dsq[2] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 2));
                        }
                        if (dsq[3] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 3));
                        }
                    } // for (; i <= m_count - 4; i += 4)

                    for (; i < m_count; i++)
                    {
                        const int16_t *cand_eq16 = cl_cands + (size_t)i * (size_t)frame_elem;
                        float dist_sq = eq16_dist_asym_cutoff_f32(
                            visited->query_eq16_adc, cand_eq16, frame_elem,
                            (float)cached_ssd_cutoff
                        );
                        if (dist_sq <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
                else
                {
                    for (; i <= m_count - 8; i += 8)
                    {
                        const int16_t *cands[8];
                        for (int k = 0; k < 8; k++)
                        {
                            long cid = (long)cl->members[m_start + i + k].frame_id;
                            cands[k] = model->eq16_dataset_buffer +
                                (size_t)cid * (size_t)frame_elem;
                        }

                        float dsq[8];
                        eq16_dist_asym_cutoff_batch_1x8(
                            visited->query_eq16_adc, cands, frame_elem,
                            (float)cached_ssd_cutoff, dsq
                        );
                        for (int k = 0; k < 8; k++)
                        {
                            if (dsq[k] <= (float)cached_ssd_cutoff)
                            {
                                pass_mask |= (1U << (i + k));
                            }
                        }
                    } // for (; i <= m_count - 8; i += 8)

                    for (; i <= m_count - 4; i += 4)
                    {
                        const int16_t *cands[4];
                        cands[0] = model->eq16_dataset_buffer +
                            (size_t)cl->members[m_start + i + 0].frame_id * (size_t)frame_elem;
                        cands[1] = model->eq16_dataset_buffer +
                            (size_t)cl->members[m_start + i + 1].frame_id * (size_t)frame_elem;
                        cands[2] = model->eq16_dataset_buffer +
                            (size_t)cl->members[m_start + i + 2].frame_id * (size_t)frame_elem;
                        cands[3] = model->eq16_dataset_buffer +
                            (size_t)cl->members[m_start + i + 3].frame_id * (size_t)frame_elem;

                        float dsq[4];
                        eq16_dist_asym_cutoff_batch_1x4(
                            visited->query_eq16_adc, cands, frame_elem,
                            (float)cached_ssd_cutoff, dsq
                        );
                        if (dsq[0] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 0));
                        }
                        if (dsq[1] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 1));
                        }
                        if (dsq[2] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 2));
                        }
                        if (dsq[3] <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << (i + 3));
                        }
                    } // for (; i <= m_count - 4; i += 4)

                    for (; i < m_count; i++)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                            (size_t)cand_id * (size_t)frame_elem;
                        float dist_sq = eq16_dist_asym_cutoff_f32(
                            visited->query_eq16_adc, cand_eq16, frame_elem,
                            (float)cached_ssd_cutoff
                        );
                        if (dist_sq <= (float)cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
            }
            else if (visited->query_eq16 != NULL)
            {
                if (cl->eq16_vectors != NULL)
                {
                    const int16_t *cl_cands = cl->eq16_vectors +
                        (size_t)m_start * (size_t)frame_elem;
                    for (int i = 0; i < m_count; i++)
                    {
                        const int16_t *cand_eq16 = cl_cands + (size_t)i * (size_t)frame_elem;
                        uint64_t ssd = eq16_dist_squared_cutoff_i16(
                            visited->query_eq16, cand_eq16, frame_elem, cached_ssd_cutoff
                        );
                        if (ssd <= cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < m_count; i++)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        const int16_t *cand_eq16 = model->eq16_dataset_buffer +
                            (size_t)cand_id * (size_t)frame_elem;
                        uint64_t ssd = eq16_dist_squared_cutoff_i16(
                            visited->query_eq16, cand_eq16, frame_elem, cached_ssd_cutoff
                        );
                        if (ssd <= cached_ssd_cutoff)
                        {
                            pass_mask |= (1U << i);
                        }
                    }
                }
            }
        }
        else
        {
            continue;
        }

        if (!pass_mask)
        {
            telem->eq16_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->eq16_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
            {
                continue;
            }

            void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
            const void *cand_ptr = knn_resolve_candidate_data(
                cand_id, m, cl, model, reader, slot, frame_bytes
            );
            if (cand_ptr == NULL)
            {
                continue;
            }

            knn_append_or_eval_candidate(
                cand_id, cand_ptr, query_id, query_data, model, config,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                batch, telem);
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

/**
 * knn_eval_members_sq16_blocks() - Evaluate cluster members using SQ16 FastScan blocks.
 */
void knn_eval_members_sq16_blocks(
    const KnnCluster       *cl,
    double                  d_anchor,
    double                  r_home,
    double                  dcc_home,
    double                  sq16_delta,
    int                     num_active_pivots,
    const double           *pivot_diffs,
    long                    query_id,
    const void *restrict    query_data,
    const KnnModel         *model,
    const KnnConfig        *config,
    KnnFrameReader         *reader,
    void          *restrict cand_buffer,
    KnnMaxHeap             *heap,
    KnnMaxHeap             *all_heaps,
#ifdef _OPENMP
    omp_lock_t             *bucket_locks,
#endif
    KnnVisitedTracker      *visited,
    KnnCandidateBatch      *batch,
    KnnTelemetry  *restrict telem)
{
    int num_m = cl->num_members;
    int num_b = cl->num_sq16_blocks;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    size_t frame_bytes = (size_t)frame_elem *
        (model->is_double ? sizeof(double) : sizeof(float));

    double last_tau = -1.0;
    uint64_t cached_ssd_cutoff = UINT64_MAX;

    int left_b = -1;
    int right_b = 0;
    knn_init_annular_block_pointers(
        cl->members, num_m, num_b, SQ16_FASTSCAN_BLOCK_SIZE, d_anchor, &left_b, &right_b
    );

    while (left_b >= 0 || right_b < num_b)
    {
        double d_left_b = 1e30;
        if (left_b >= 0)
        {
            int m_end_l = (left_b == num_b - 1) ? (num_m - 1) :
                (left_b * SQ16_FASTSCAN_BLOCK_SIZE + SQ16_FASTSCAN_BLOCK_SIZE - 1);
            float r_max_l = cl->members[m_end_l].r_anchor;
            d_left_b = (d_anchor > (double)r_max_l) ? (d_anchor - (double)r_max_l) : 0.0;
        }

        double d_right_b = 1e30;
        if (right_b < num_b)
        {
            float r_min_r = cl->members[right_b * SQ16_FASTSCAN_BLOCK_SIZE].r_anchor;
            d_right_b = ((double)r_min_r > d_anchor) ? ((double)r_min_r - d_anchor) : 0.0;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_thresh = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_thresh)
        {
            tau_thresh = config->rlim_cutoff;
        }

        if (left_b >= 0 && (d_left_b - sq16_delta >= tau_thresh))
        {
            int pruned = (left_b + 1) * SQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > num_m)
            {
                pruned = num_m;
            }
            telem->level3_annular_pruned += (uint64_t)pruned;
            left_b = -1;
            d_left_b = 1e30;
        }
        if (right_b < num_b && (d_right_b - sq16_delta >= tau_thresh))
        {
            int pruned = num_m - right_b * SQ16_FASTSCAN_BLOCK_SIZE;
            if (pruned > 0)
            {
                telem->level3_annular_pruned += (uint64_t)pruned;
            }
            right_b = num_b;
            d_right_b = 1e30;
        }
        if (left_b < 0 && right_b >= num_b)
        {
            break;
        }

        int b = (d_left_b <= d_right_b) ? left_b-- : right_b++;
        int m_start = b * SQ16_FASTSCAN_BLOCK_SIZE;
        int m_count = num_m - m_start;
        if (m_count > SQ16_FASTSCAN_BLOCK_SIZE)
        {
            m_count = SQ16_FASTSCAN_BLOCK_SIZE;
        }

        if (current_tau != last_tau)
        {
            last_tau = current_tau;
            cached_ssd_cutoff = compute_sq16_cutoff_thresh(current_tau, model, config);
        }

        uint32_t pass_mask = 0;
        if (cl->sq16_transposed != NULL)
        {
            const int16_t *b_coords = cl->sq16_transposed +
                (size_t)b * (size_t)frame_elem * SQ16_FASTSCAN_BLOCK_SIZE;
            telem->sq16_evaluations += (uint64_t)m_count;
            pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_sq16_sparse && config->use_sq16_sparse_lru &&
                 visited->sq16_lru_cache != NULL)
        {
            /* 16-block LRU Transposed FastScan in L2 cache */
            KnnSparseLruCache *lru = visited->sq16_lru_cache;
            int cl_idx = (int)(cl - model->clusters);
            int hit_slot = -1;
            int lru_slot = 0;
            uint64_t min_seq = UINT64_MAX;

            for (int s = 0; s < KNN_SPARSE_LRU_BLOCKS; s++)
            {
                if (lru->blocks[s].cl_id == cl_idx && lru->blocks[s].block_id == b)
                {
                    hit_slot = s;
                    break;
                }
                if (lru->blocks[s].access_seq < min_seq)
                {
                    min_seq = lru->blocks[s].access_seq;
                    lru_slot = s;
                }
            }

            int16_t *b_coords = NULL;
            if (hit_slot >= 0)
            {
                lru->blocks[hit_slot].access_seq = ++lru->access_clock;
                b_coords = lru->blocks[hit_slot].data;
            }
            else
            {
                b_coords = lru->blocks[lru_slot].data;
                lru->blocks[lru_slot].cl_id = cl_idx;
                lru->blocks[lru_slot].block_id = b;
                lru->blocks[lru_slot].access_seq = ++lru->access_clock;

                const int16_t *cand_ptrs[SQ16_FASTSCAN_BLOCK_SIZE];
                for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
                {
                    if (i < m_count)
                    {
                        long cand_id = (long)cl->members[m_start + i].frame_id;
                        cand_ptrs[i] = model->sq16_dataset_buffer + cand_id * frame_elem;
                    }
                    else
                    {
                        cand_ptrs[i] = NULL;
                    }
                }

                for (long d = 0; d < frame_elem; d++)
                {
                    int16_t *dst = b_coords + d * SQ16_FASTSCAN_BLOCK_SIZE;
                    for (int i = 0; i < SQ16_FASTSCAN_BLOCK_SIZE; i++)
                    {
                        dst[i] = cand_ptrs[i] ? cand_ptrs[i][d] : 32767;
                    }
                }
            }

            telem->sq16_evaluations += (uint64_t)m_count;
            pass_mask = sq16_fastscan_32x(
                visited->query_sq16, b_coords, frame_elem, cached_ssd_cutoff
            );
            if (m_count < SQ16_FASTSCAN_BLOCK_SIZE)
            {
                pass_mask &= ((1U << m_count) - 1);
            }
        }
        else if (config->use_sq16_sparse && model->sq16_dataset_buffer != NULL)
        {
            /* Direct Row-Major SIMD with early cutoff (0 MB resident index) */
            telem->sq16_evaluations += (uint64_t)m_count;
            for (int i = 0; i < m_count; i++)
            {
                long cand_id = (long)cl->members[m_start + i].frame_id;
                const int16_t *cand_sq16 = model->sq16_dataset_buffer +
                                           (size_t)cand_id * (size_t)frame_elem;
                uint64_t ssd = sq16_dist_squared_cutoff_i16(
                    visited->query_sq16, cand_sq16, frame_elem, cached_ssd_cutoff
                );
                if (ssd <= cached_ssd_cutoff)
                {
                    pass_mask |= (1U << i);
                }
            }
        }
        else
        {
            continue;
        }

        if (!pass_mask)
        {
            telem->sq16_members_pruned += (uint64_t)m_count;
            continue;
        }

        int passed_count = knn_popcount32(pass_mask);
        telem->sq16_members_pruned += (uint64_t)(m_count - passed_count);

        while (pass_mask)
        {
            int lane = knn_ctz32(pass_mask);
            pass_mask &= pass_mask - 1;

            int m = m_start + lane;
            long cand_id = (long)cl->members[m].frame_id;
            double r_cand = (double)cl->members[m].r_anchor;

            if (knn_is_candidate_pruned(
                    query_id, cand_id, r_cand, d_anchor, r_home, dcc_home,
                    tau_thresh, sq16_delta, num_active_pivots, pivot_diffs,
                    config, heap, visited, telem))
            {
                continue;
            }

            void *slot = (char *)cand_buffer + (size_t)batch->count * frame_bytes;
            const void *cand_ptr = knn_resolve_candidate_data(
                cand_id, m, cl, model, reader, slot, frame_bytes
            );
            if (cand_ptr == NULL)
            {
                continue;
            }

            knn_append_or_eval_candidate(
                cand_id, cand_ptr, query_id, query_data, model, config,
                heap, all_heaps,
#ifdef _OPENMP
                bucket_locks,
#endif
                batch, telem);
        } // while (pass_mask)
    } // while (left_b >= 0 || right_b < num_b)
}

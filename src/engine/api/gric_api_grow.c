/**
 * @file gric_api_grow.c
 * @brief Dynamic capacity expansion and buffer reallocation for libgric.
 */

#include "gric_api_internal.h"
#include "cluster_dcc.h"
#include "cluster_steps.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * grow_matrix() - Expand an N x N matrix to new dimensions preserving submatrix.
 * @old:       Previous matrix buffer (freed upon successful copy).
 * @old_n:     Previous dimension.
 * @new_n:     New dimension.
 * @elem_size: Size in bytes of each element.
 *
 * Return: Pointer to newly allocated matrix, or NULL on error.
 */
static void *grow_matrix(
    void   *old,
    size_t  old_n,
    size_t  new_n,
    size_t  elem_size)
{
    if (new_n == 0 || elem_size == 0)
    {
        return NULL;
    }

    void *new_buf = calloc(new_n * new_n, elem_size);
    if (!new_buf)
    {
        return NULL;
    }

    if (old)
    {
        for (size_t i = 0; i < old_n; i++)
        {
            memcpy(
                (char *)new_buf + (i * new_n * elem_size),
                (char *)old + (i * old_n * elem_size),
                old_n * elem_size
            );
        }
        free(old);
    }
    return new_buf;
}

/**
 * grow_scratch_buffers() - Reallocate scratch vectors during dynamic context growth.
 * @s:     Cluster scratch workspace.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_scratch_buffers(
    ClusterScratch *s,
    size_t          old_N,
    size_t          new_N)
{
    s->mixed_probs = (double *)realloc(s->mixed_probs, new_N * sizeof(double));
    s->clmembflag = (int *)realloc(s->clmembflag, new_N * sizeof(int));
    s->active_clusters = (int *)realloc(s->active_clusters, new_N * sizeof(int));
    s->probsortedclindex = (int *)realloc(s->probsortedclindex, new_N * sizeof(int));

    double *new_cp = NULL;
    if (posix_memalign((void **)&new_cp, 64, new_N * sizeof(double)) == 0)
    {
        memset(new_cp, 0, new_N * sizeof(double));
        if (s->cluster_probs)
        {
            memcpy(new_cp, s->cluster_probs, old_N * sizeof(double));
            free(s->cluster_probs);
        }
        s->cluster_probs = new_cp;
    }
    s->current_gprobs = (double *)realloc(s->current_gprobs, new_N * sizeof(double));

    s->entropy_p_current = (double *)realloc(s->entropy_p_current, new_N * sizeof(double));
    s->entropy_candidates = (Candidate *)realloc(
        s->entropy_candidates, new_N * sizeof(Candidate));
    s->entropy_prob_scores = (TargetScore *)realloc(
        s->entropy_prob_scores, new_N * sizeof(TargetScore));
    s->entropy_prune_scores = (TargetScore *)realloc(
        s->entropy_prune_scores, new_N * sizeof(TargetScore));
    s->entropy_active_indices = (int *)realloc(s->entropy_active_indices, new_N * sizeof(int));
    s->entropy_plog2p = (double *)realloc(s->entropy_plog2p, new_N * sizeof(double));
    s->entropy_visited = (uint8_t *)realloc(s->entropy_visited, new_N * sizeof(uint8_t));
    s->refine_queue = (Candidate *)realloc(s->refine_queue, new_N * sizeof(Candidate));
    s->refine_queue_capacity = (int)new_N;
    s->tuple_pred_candidates = (int *)realloc(s->tuple_pred_candidates, new_N * sizeof(int));
    s->pred_candidates = (int *)realloc(s->pred_candidates, new_N * sizeof(int));
    s->local_candidates = (int *)realloc(s->local_candidates, new_N * sizeof(int));

    memset(&s->clmembflag[old_N], 0, (new_N - old_N) * sizeof(int));
    memset(&s->mixed_probs[old_N], 0, (new_N - old_N) * sizeof(double));
    memset(&s->current_gprobs[old_N], 0, (new_N - old_N) * sizeof(double));
    memset(&s->probsortedclindex[old_N], 0, (new_N - old_N) * sizeof(int));
    if (s->entropy_visited)
    {
        memset(&s->entropy_visited[old_N], 0, (new_N - old_N) * sizeof(uint8_t));
    }
    return 0;
}

/**
 * grow_telemetry_buffers() - Reallocate telemetry tracking arrays for expanded capacity.
 * @t:     Cluster telemetry workspace.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 */
static void grow_telemetry_buffers(
    ClusterTelemetry *t,
    size_t            old_N,
    size_t            new_N)
{
    t->pruned_fraction_sum = (double *)realloc(t->pruned_fraction_sum, new_N * sizeof(double));
    t->step_counts = (long *)realloc(t->step_counts, new_N * sizeof(long));
    t->dist_counts = (long *)realloc(t->dist_counts, (new_N + 1) * sizeof(long));
    t->pruned_counts_by_dist = (long *)realloc(
        t->pruned_counts_by_dist, (new_N + 1) * sizeof(long));
    t->cluster_query_counts = (long *)realloc(t->cluster_query_counts, new_N * sizeof(long));

    if (t->pruned_fraction_sum)
    {
        memset(&t->pruned_fraction_sum[old_N], 0, (new_N - old_N) * sizeof(double));
    }
    if (t->step_counts)
    {
        memset(&t->step_counts[old_N], 0, (new_N - old_N) * sizeof(long));
    }
    if (t->dist_counts)
    {
        memset(&t->dist_counts[old_N + 1], 0, (new_N - old_N) * sizeof(long));
    }
    if (t->pruned_counts_by_dist)
    {
        memset(&t->pruned_counts_by_dist[old_N + 1], 0, (new_N - old_N) * sizeof(long));
    }
    if (t->cluster_query_counts)
    {
        memset(&t->cluster_query_counts[old_N], 0, (new_N - old_N) * sizeof(long));
    }
    t->max_steps_recorded = (int)new_N;
}

/**
 * grow_consistency_mask() - Reallocate 3D geometric consistency mask if entropy active.
 * @ctx:   Active clustering session.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_consistency_mask(
    gric_cluster_t *ctx,
    size_t          new_N)
{
    if (!ctx->config.optim.entropy_mode && !ctx->config.optim.gprob_mode)
    {
        return 0;
    }

    ClusterScratch *s = &ctx->state.scratch;
    size_t new_mask_words = new_N * new_N * ((new_N + 63) / 64);
    uint64_t *new_mask = (uint64_t *)calloc(new_mask_words, sizeof(uint64_t));
    if (!new_mask)
    {
        return -1;
    }
    if (s->consistency_mask != NULL)
    {
        free(s->consistency_mask);
    }
    s->consistency_mask = new_mask;
    recompute_consistency_mask(&ctx->config, &ctx->state);
    return 0;
}

/**
 * grow_anchor_matrices() - Reallocate contiguous anchor and norm matrices during growth.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
/**
 * grow_float_anchors() - Reallocate float anchor and norm matrices during growth.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_float_anchors(
    gric_cluster_t *ctx,
    size_t          old_N,
    size_t          new_N)
{
    ClusterState *state = &ctx->state;
    size_t dim = ctx->ndim;

    if (state->anchor_matrix_float != NULL)
    {
        float *new_mat = NULL;
        if (posix_memalign((void **)&new_mat, 64, new_N * dim * sizeof(float)) != 0)
        {
            return -1;
        }
        memcpy(new_mat, state->anchor_matrix_float, old_N * dim * sizeof(float));
        memset(new_mat + old_N * dim, 0, (new_N - old_N) * dim * sizeof(float));
        free(state->anchor_matrix_float);
        state->anchor_matrix_float = new_mat;
    }

    if (state->anchor_norms_float != NULL)
    {
        float *new_norms = NULL;
        if (posix_memalign((void **)&new_norms, 64, new_N * sizeof(float)) != 0)
        {
            return -1;
        }
        memcpy(new_norms, state->anchor_norms_float, old_N * sizeof(float));
        memset(new_norms + old_N, 0, (new_N - old_N) * sizeof(float));
        free(state->anchor_norms_float);
        state->anchor_norms_float = new_norms;
    }

    return 0;
}

/**
 * grow_quantized_anchors() - Reallocate quantized SQ8/SQ16/EQ16 anchor matrices.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_quantized_anchors(
    gric_cluster_t *ctx,
    size_t          old_N,
    size_t          new_N)
{
    ClusterState *state = &ctx->state;
    size_t dim = ctx->ndim;

    if (state->anchor_matrix_sq8 != NULL)
    {
        uint8_t *new_sq8 = NULL;
        if (posix_memalign((void **)&new_sq8, 64, new_N * dim * sizeof(uint8_t)) != 0)
        {
            return -1;
        }
        memcpy(new_sq8, state->anchor_matrix_sq8, old_N * dim * sizeof(uint8_t));
        memset(new_sq8 + old_N * dim, 0, (new_N - old_N) * dim * sizeof(uint8_t));
        free(state->anchor_matrix_sq8);
        state->anchor_matrix_sq8 = new_sq8;
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].anchor_sq8 = new_sq8 + (size_t)i * dim;
        }
    }

    if (state->anchor_matrix_sq16 != NULL)
    {
        int16_t *new_sq16 = NULL;
        if (posix_memalign((void **)&new_sq16, 64, new_N * dim * sizeof(int16_t)) != 0)
        {
            return -1;
        }
        memcpy(new_sq16, state->anchor_matrix_sq16, old_N * dim * sizeof(int16_t));
        memset(new_sq16 + old_N * dim, 0, (new_N - old_N) * dim * sizeof(int16_t));
        free(state->anchor_matrix_sq16);
        state->anchor_matrix_sq16 = new_sq16;
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].anchor_sq16 = new_sq16 + (size_t)i * dim;
        }
    }

    if (state->anchor_matrix_eq16 != NULL)
    {
        int16_t *new_eq16 = NULL;
        if (posix_memalign((void **)&new_eq16, 64, new_N * dim * sizeof(int16_t)) != 0)
        {
            return -1;
        }
        memcpy(new_eq16, state->anchor_matrix_eq16, old_N * dim * sizeof(int16_t));
        memset(new_eq16 + old_N * dim, 0, (new_N - old_N) * dim * sizeof(int16_t));
        free(state->anchor_matrix_eq16);
        state->anchor_matrix_eq16 = new_eq16;
        for (int i = 0; i < state->num_clusters; i++)
        {
            state->clusters[i].anchor_eq16 = new_eq16 + (size_t)i * dim;
        }
    }

    return 0;
}

/**
 * grow_interleaved_anchors() - Reallocate SIMD interleaved anchor layouts.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_interleaved_anchors(
    gric_cluster_t *ctx,
    size_t          old_N,
    size_t          new_N)
{
    ClusterState *state = &ctx->state;
    size_t dim = ctx->ndim;

    if (state->anchor_matrix_sq16_interleaved != NULL)
    {
        size_t old_blocks = (old_N + 7) / 8;
        size_t new_blocks = (new_N + 7) / 8;
        size_t num_pairs = (dim + 1) / 2;
        int32_t *new_intl = NULL;
        if (posix_memalign((void **)&new_intl, 64,
                           new_blocks * num_pairs * 8 * sizeof(int32_t)) != 0)
        {
            return -1;
        }
        memcpy(new_intl, state->anchor_matrix_sq16_interleaved,
               old_blocks * num_pairs * 8 * sizeof(int32_t));
        memset(new_intl + old_blocks * num_pairs * 8, 0,
               (new_blocks - old_blocks) * num_pairs * 8 * sizeof(int32_t));
        free(state->anchor_matrix_sq16_interleaved);
        state->anchor_matrix_sq16_interleaved = new_intl;
    }

    if (state->anchor_matrix_eq16_interleaved != NULL)
    {
        size_t old_blocks = (old_N + 7) / 8;
        size_t new_blocks = (new_N + 7) / 8;
        size_t num_pairs = (dim + 1) / 2;
        int32_t *new_intl = NULL;
        if (posix_memalign((void **)&new_intl, 64,
                           new_blocks * num_pairs * 8 * sizeof(int32_t)) != 0)
        {
            return -1;
        }
        memcpy(new_intl, state->anchor_matrix_eq16_interleaved,
               old_blocks * num_pairs * 8 * sizeof(int32_t));
        memset(new_intl + old_blocks * num_pairs * 8, 0,
               (new_blocks - old_blocks) * num_pairs * 8 * sizeof(int32_t));
        free(state->anchor_matrix_eq16_interleaved);
        state->anchor_matrix_eq16_interleaved = new_intl;
    }

    if (state->anchor_matrix_adc_interleaved != NULL)
    {
        size_t old_blocks_adc = (old_N + 15) / 16;
        size_t new_blocks_adc = (new_N + 15) / 16;
        float *new_adc = NULL;
        if (posix_memalign((void **)&new_adc, 64,
                           new_blocks_adc * dim * 16 * sizeof(float)) != 0)
        {
            return -1;
        }
        memcpy(new_adc, state->anchor_matrix_adc_interleaved,
               old_blocks_adc * dim * 16 * sizeof(float));
        memset(new_adc + old_blocks_adc * dim * 16, 0,
               (new_blocks_adc - old_blocks_adc) * dim * 16 * sizeof(float));
        free(state->anchor_matrix_adc_interleaved);
        state->anchor_matrix_adc_interleaved = new_adc;
    }

    return 0;
}

/**
 * grow_anchor_matrices() - Reallocate contiguous anchor and norm matrices during growth.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_anchor_matrices(
    gric_cluster_t *ctx,
    size_t          old_N,
    size_t          new_N)
{
    if (grow_float_anchors(ctx, old_N, new_N) != 0)
    {
        return -1;
    }
    if (grow_quantized_anchors(ctx, old_N, new_N) != 0)
    {
        return -1;
    }
    if (grow_interleaved_anchors(ctx, old_N, new_N) != 0)
    {
        return -1;
    }
    return 0;
}

/**
 * grow_clusters_and_visitors() - Double cluster structures and visitor arrays.
 * @ctx:   Active clustering session.
 * @old_N: Previous cluster capacity.
 * @new_N: New cluster capacity.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int grow_clusters_and_visitors(
    gric_cluster_t *ctx,
    size_t          old_N,
    size_t          new_N)
{
    Cluster *new_clusters = (Cluster *)realloc(
        ctx->state.clusters, new_N * sizeof(Cluster)
    );
    if (!new_clusters)
    {
        return -1;
    }
    ctx->state.clusters = new_clusters;
    memset(&ctx->state.clusters[old_N], 0, (new_N - old_N) * sizeof(Cluster));

    VisitorList *new_visitors = (VisitorList *)realloc(
        ctx->state.cluster_visitors, new_N * sizeof(VisitorList)
    );
    if (!new_visitors)
    {
        return -1;
    }
    ctx->state.cluster_visitors = new_visitors;
    memset(&ctx->state.cluster_visitors[old_N], 0, (new_N - old_N) * sizeof(VisitorList));
    return 0;
}

int grow_context_capacity(
    gric_cluster_t *ctx)
{
    size_t old_N = (size_t)ctx->config.algo.maxnbclust;
    size_t new_N = old_N * 2;
    if (new_N < 16)
    {
        new_N = 16;
    }

    if (grow_clusters_and_visitors(ctx, old_N, new_N) != 0)
    {
        return -1;
    }

    if (ctx->state.transition_matrix != NULL)
    {
        long *new_tm = (long *)grow_matrix(
            ctx->state.transition_matrix, old_N, new_N, sizeof(long)
        );
        if (!new_tm)
        {
            return -1;
        }
        ctx->state.transition_matrix = new_tm;
    }

    if (grow_scratch_buffers(&ctx->state.scratch, old_N, new_N) != 0)
    {
        return -1;
    }

    if (dcc_grow_capacity(&ctx->state, new_N, ctx->config.optim.sparse_dcc_mode) != 0)
    {
        return -1;
    }

    if (grow_consistency_mask(ctx, new_N) != 0)
    {
        return -1;
    }

    if (grow_anchor_matrices(ctx, old_N, new_N) != 0)
    {
        return -1;
    }

    ctx->temp_indices = (int *)realloc(ctx->temp_indices, new_N * sizeof(int));
    ctx->temp_dists = (double *)realloc(ctx->temp_dists, new_N * sizeof(double));
    ctx->sorting_candidates = (Candidate *)realloc(
        ctx->sorting_candidates, new_N * sizeof(Candidate)
    );

    grow_telemetry_buffers(&ctx->state.telemetry, old_N, new_N);

    ctx->config.algo.maxnbclust = (int)new_N;
    return 0;
}

/**
 * @file gric_api.c
 * @brief Implementation of the public C library API for GRIC.
 */

#include "gric/gric.h"
#include "cluster_defs.h"
#include "cluster_step.h"
#include "cluster_steps.h"
#include "framedistance.h"
#include "cluster_math.h"
#include "cluster_bounds.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_CLUSTERS 256
#define DEFAULT_MAX_FRAMES   100000

/**
 * struct gric_cluster_ctx - Concrete clustering session state.
 */
struct gric_cluster_ctx
{
    ClusterConfig  config;
    ClusterState   state;
    Frame          frame;
    size_t         ndim;
    size_t         maxnbfr;
    int            current_frame_id;
    int            prev_assigned;
    int            user_maxcl;
    int           *temp_indices;
    double        *temp_dists;
    Candidate     *sorting_candidates;
};

const char *gric_version(
    void)
{
    return "1.0.0";
}

gric_status_t gric_cluster_config_default(
    gric_cluster_config_t *cfg)
{
    if (cfg == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    memset(cfg, 0, sizeof(gric_cluster_config_t));
    cfg->rlim = 0.5;
    cfg->maxnbclust = DEFAULT_MAX_CLUSTERS;
    cfg->maxnbfr = DEFAULT_MAX_FRAMES;
    cfg->tm_mixing_coeff = 0.1;
    cfg->use_double = 1;
    cfg->use_sq16 = 0;
    cfg->use_eq16 = 0;
    cfg->te4_mode = 1;
    cfg->te5_mode = 0;
    cfg->entropy_mode = 0;
    cfg->entropy_gate_bits = 0.0;
    cfg->pred_mode = 0;
    cfg->gprob_mode = 0;
    cfg->soft_bayesian_mode = 0;
    cfg->sparse_dcc_mode = 0;
    cfg->sparse_dcc_extra_evals = 0;
    cfg->maxcl_strategy = 0;
    cfg->discard_fraction = 0.0;
    cfg->ncpu = 1;

    return GRIC_SUCCESS;
}

static void *grow_matrix(
    void   *old,
    int     old_n,
    int     new_n,
    size_t  elem_size)
{
    void *new_buf = calloc((size_t)new_n * new_n, elem_size);
    if (!new_buf)
    {
        return NULL;
    }

    if (old)
    {
        for (int i = 0; i < old_n; i++)
        {
            memcpy(
                (char *)new_buf + (i * new_n * elem_size),
                (char *)old + (i * old_n * elem_size),
                (size_t)old_n * elem_size
            );
        }
        free(old);
    }
    return new_buf;
}

static int grow_context_capacity(
    gric_cluster_t *ctx)
{
    int old_N = ctx->config.algo.maxnbclust;
    int new_N = old_N * 2;
    if (new_N < 16)
    {
        new_N = 16;
    }

    Cluster *new_clusters = (Cluster *)realloc(
        ctx->state.clusters, (size_t)new_N * sizeof(Cluster)
    );
    if (!new_clusters)
    {
        return -1;
    }
    ctx->state.clusters = new_clusters;
    memset(&ctx->state.clusters[old_N], 0, (size_t)(new_N - old_N) * sizeof(Cluster));

    VisitorList *new_visitors = (VisitorList *)realloc(
        ctx->state.cluster_visitors, (size_t)new_N * sizeof(VisitorList)
    );
    if (!new_visitors)
    {
        return -1;
    }
    ctx->state.cluster_visitors = new_visitors;
    memset(&ctx->state.cluster_visitors[old_N], 0, (size_t)(new_N - old_N) * sizeof(VisitorList));

    long *new_tm = (long *)grow_matrix(ctx->state.transition_matrix, old_N, new_N, sizeof(long));
    if (!new_tm)
    {
        return -1;
    }
    ctx->state.transition_matrix = new_tm;

    ClusterScratch *s = &ctx->state.scratch;
    s->mixed_probs = (double *)realloc(s->mixed_probs, (size_t)new_N * sizeof(double));
    s->clmembflag = (int *)realloc(s->clmembflag, (size_t)new_N * sizeof(int));
    s->active_clusters = (int *)realloc(s->active_clusters, (size_t)new_N * sizeof(int));
    s->probsortedclindex = (int *)realloc(s->probsortedclindex, (size_t)new_N * sizeof(int));

    double *new_cp = NULL;
    if (posix_memalign((void **)&new_cp, 64, (size_t)new_N * sizeof(double)) == 0)
    {
        memset(new_cp, 0, (size_t)new_N * sizeof(double));
        if (s->cluster_probs)
        {
            memcpy(new_cp, s->cluster_probs, (size_t)old_N * sizeof(double));
            free(s->cluster_probs);
        }
        s->cluster_probs = new_cp;
    }
    s->current_gprobs = (double *)realloc(s->current_gprobs, (size_t)new_N * sizeof(double));

    s->dcc_min = (double *)grow_matrix(s->dcc_min, old_N, new_N, sizeof(double));
    s->dcc_max = (double *)grow_matrix(s->dcc_max, old_N, new_N, sizeof(double));
    s->dcc_measured = (char *)grow_matrix(s->dcc_measured, old_N, new_N, sizeof(char));

    size_t new_mask_words = (size_t)new_N * new_N * ((new_N + 63) / 64);
    uint64_t *new_mask = (uint64_t *)calloc(new_mask_words, sizeof(uint64_t));
    if (new_mask)
    {
        free(s->consistency_mask);
        s->consistency_mask = new_mask;
    }

    ctx->temp_indices = (int *)realloc(ctx->temp_indices, (size_t)new_N * sizeof(int));
    ctx->temp_dists = (double *)realloc(ctx->temp_dists, (size_t)new_N * sizeof(double));
    ctx->sorting_candidates = (Candidate *)realloc(
        ctx->sorting_candidates, (size_t)new_N * sizeof(Candidate)
    );

    ctx->config.algo.maxnbclust = new_N;
    ctx->state.telemetry.max_steps_recorded = new_N;

    return 0;
}

static int ensure_frame_buffer(
    gric_cluster_t *ctx)
{
    if (ctx->frame.data == NULL)
    {
        ctx->frame.data = (double *)malloc(ctx->ndim * sizeof(double));
        if (!ctx->frame.data)
        {
            return -1;
        }
        ctx->frame.is_double = 1;
        ctx->frame.width = (int)ctx->ndim;
        ctx->frame.height = 1;
    }
    return 0;
}

gric_cluster_t *gric_cluster_create(
    const gric_cluster_config_t *cfg,
    size_t                       ndim)
{
    if (ndim == 0)
    {
        return NULL;
    }

    gric_cluster_config_t local_cfg;
    if (cfg == NULL)
    {
        gric_cluster_config_default(&local_cfg);
        cfg = &local_cfg;
    }

    gric_cluster_t *ctx = (gric_cluster_t *)calloc(1, sizeof(gric_cluster_t));
    if (!ctx)
    {
        return NULL;
    }

    ctx->ndim = ndim;
    ctx->user_maxcl = cfg->maxnbclust;
    int N = (cfg->maxnbclust > 0) ? cfg->maxnbclust : DEFAULT_MAX_CLUSTERS;
    long maxfr = (cfg->maxnbfr > 0) ? cfg->maxnbfr : DEFAULT_MAX_FRAMES;
    ctx->maxnbfr = (size_t)maxfr;
    ctx->prev_assigned = -1;

    ctx->config.algo.rlim = cfg->rlim;
    ctx->config.algo.maxnbclust = N;
    ctx->config.algo.deltaprob = 0.0;
    ctx->config.algo.tm_mixing_coeff = cfg->tm_mixing_coeff;
    ctx->config.algo.maxcl_strategy = (MaxClustStrategy)cfg->maxcl_strategy;
    ctx->config.algo.discard_fraction = cfg->discard_fraction;
    ctx->config.algo.use_double = cfg->use_double;

    ctx->config.input.maxnbfr = maxfr;

    ctx->config.optim.entropy_mode = cfg->entropy_mode;
    ctx->config.optim.te4_mode = cfg->te4_mode;
    ctx->config.optim.te5_mode = cfg->te5_mode;
    ctx->config.optim.pred_mode = cfg->pred_mode;
    ctx->config.optim.pred_len = 4;
    ctx->config.optim.pred_h = 10;
    ctx->config.optim.pred_n = 3;
    ctx->config.optim.gprob_mode = cfg->gprob_mode;
    ctx->config.optim.fmatch_a = 2.0;
    ctx->config.optim.fmatch_b = 0.5;
    ctx->config.optim.max_gprob_visitors = 20;
    ctx->config.optim.entropy_gate_bits = cfg->entropy_gate_bits;
    ctx->config.optim.sparse_dcc_mode = cfg->sparse_dcc_mode;
    ctx->config.optim.sparse_dcc_extra_evals = cfg->sparse_dcc_extra_evals;
    ctx->config.optim.soft_bayesian_mode = cfg->soft_bayesian_mode;
    ctx->config.optim.soft_bayesian_sigma_coeff = 0.5;
    ctx->config.optim.use_sq16 = cfg->use_sq16;
    ctx->config.optim.use_eq16 = cfg->use_eq16;
    ctx->config.optim.use_batch_dist = 1;
    ctx->config.optim.ncpu = cfg->ncpu > 0 ? cfg->ncpu : 1;

    ctx->state.clusters = (Cluster *)calloc((size_t)N, sizeof(Cluster));
    ctx->state.cluster_visitors = (VisitorList *)calloc((size_t)N, sizeof(VisitorList));
    ctx->state.assignments = (int *)malloc((size_t)maxfr * sizeof(int));
    ctx->state.frame_infos = (FrameInfo *)calloc((size_t)maxfr, sizeof(FrameInfo));
    ctx->state.transition_matrix = (long *)calloc((size_t)N * N, sizeof(long));

    ClusterScratch *s = &ctx->state.scratch;
    s->mixed_probs = (double *)calloc((size_t)N, sizeof(double));
    s->clmembflag = (int *)calloc((size_t)N, sizeof(int));
    s->active_clusters = (int *)malloc((size_t)N * sizeof(int));
    s->num_active_clusters = 0;
    s->probsortedclindex = (int *)calloc((size_t)N, sizeof(int));
    if (posix_memalign((void **)&s->cluster_probs, 64, (size_t)N * sizeof(double)) != 0)
    {
        s->cluster_probs = NULL;
    }
    else
    {
        memset(s->cluster_probs, 0, (size_t)N * sizeof(double));
    }
    s->current_gprobs = (double *)calloc((size_t)N, sizeof(double));
    s->dcc_min = (double *)calloc((size_t)N * N, sizeof(double));
    s->dcc_max = (double *)calloc((size_t)N * N, sizeof(double));
    s->dcc_measured = (char *)calloc((size_t)N * N, sizeof(char));

    size_t mask_words = (size_t)N * N * ((N + 63) / 64);
    s->consistency_mask = (uint64_t *)calloc(mask_words, sizeof(uint64_t));

    s->entropy_p_current = (double *)calloc((size_t)N, sizeof(double));
    s->entropy_candidates = (Candidate *)calloc((size_t)N, sizeof(Candidate));
    s->entropy_prob_scores = (TargetScore *)calloc((size_t)N, sizeof(TargetScore));
    s->entropy_prune_scores = (TargetScore *)calloc((size_t)N, sizeof(TargetScore));
    s->entropy_active_indices = (int *)calloc((size_t)N, sizeof(int));
    s->entropy_plog2p = (double *)calloc((size_t)N, sizeof(double));
    s->entropy_visited = (uint8_t *)calloc((size_t)N, sizeof(uint8_t));
    s->refine_queue = (Candidate *)calloc((size_t)N, sizeof(Candidate));
    s->refine_queue_capacity = N;
    s->tuple_pred_candidates = (int *)calloc((size_t)N, sizeof(int));
    s->pred_candidates = (int *)calloc((size_t)N, sizeof(int));
    s->local_candidates = (int *)calloc((size_t)N, sizeof(int));

    ClusterTelemetry *t = &ctx->state.telemetry;
    t->max_steps_recorded = N;
    t->pruned_fraction_sum = (double *)calloc((size_t)N, sizeof(double));
    t->step_counts = (long *)calloc((size_t)N, sizeof(long));
    t->dist_counts = (long *)calloc((size_t)(N + 1), sizeof(long));
    t->pruned_counts_by_dist = (long *)calloc((size_t)(N + 1), sizeof(long));
    t->cluster_query_counts = (long *)calloc((size_t)N, sizeof(long));

    ctx->temp_indices = (int *)calloc((size_t)N, sizeof(int));
    ctx->temp_dists = (double *)calloc((size_t)N, sizeof(double));
    ctx->sorting_candidates = (Candidate *)calloc((size_t)N, sizeof(Candidate));

    ctx->frame.data = (double *)calloc(ndim, sizeof(double));
    ctx->frame.is_double = 1;
    ctx->frame.width = (int)ndim;
    ctx->frame.height = 1;
    ctx->frame.id = 0;

    for (long i = 0; i < maxfr; i++)
    {
        ctx->state.assignments[i] = -1;
    }

    return ctx;
}

gric_cluster_t *gric_cluster_create_simple(
    size_t ndim,
    double rlim)
{
    gric_cluster_config_t cfg;
    gric_cluster_config_default(&cfg);
    cfg.rlim = rlim;
    return gric_cluster_create(&cfg, ndim);
}

gric_status_t gric_cluster_feed_frame(
    gric_cluster_t *ctx,
    const double   *coords,
    int64_t        *out_cluster_id)
{
    if (ctx == NULL || coords == NULL || out_cluster_id == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    if (ensure_frame_buffer(ctx) != 0)
    {
        return GRIC_ERR_OUT_OF_MEMORY;
    }

    memcpy(ctx->frame.data, coords, ctx->ndim * sizeof(double));
    ctx->frame.id = ctx->current_frame_id;
    ctx->frame.width = (int)ctx->ndim;
    ctx->frame.height = 1;

    if (ctx->user_maxcl == 0 &&
        ctx->state.num_clusters >= ctx->config.algo.maxnbclust - 1)
    {
        if (grow_context_capacity(ctx) != 0)
        {
            return GRIC_ERR_OUT_OF_MEMORY;
        }
    }

    int cid = cluster_frame(
        &ctx->config,
        &ctx->state,
        &ctx->frame,
        &ctx->prev_assigned,
        NULL,
        ctx->temp_indices,
        ctx->temp_dists,
        ctx->sorting_candidates,
        NULL
    );

    if (cid < 0)
    {
        return GRIC_ERR_CAPACITY;
    }

    *out_cluster_id = cid;
    ctx->current_frame_id++;
    return GRIC_SUCCESS;
}

gric_status_t gric_cluster_feed_batch(
    gric_cluster_t *ctx,
    const double   *coords_flat,
    size_t          num_frames,
    int64_t        *out_cluster_ids)
{
    if (ctx == NULL || coords_flat == NULL || out_cluster_ids == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    for (size_t i = 0; i < num_frames; i++)
    {
        const double *frame_ptr = coords_flat + (i * ctx->ndim);
        gric_status_t st = gric_cluster_feed_frame(ctx, frame_ptr, &out_cluster_ids[i]);
        if (st != GRIC_SUCCESS)
        {
            return st;
        }
    }

    return GRIC_SUCCESS;
}

int64_t gric_cluster_get_num_clusters(
    const gric_cluster_t *ctx)
{
    if (ctx == NULL)
    {
        return -1;
    }
    return ctx->state.num_clusters;
}

int64_t gric_cluster_get_anchors(
    const gric_cluster_t *ctx,
    double               *out_coords,
    int                  *out_members,
    size_t                max_anchors)
{
    if (ctx == NULL || out_coords == NULL)
    {
        return -1;
    }

    size_t nc = (size_t)ctx->state.num_clusters;
    size_t limit = (max_anchors < nc) ? max_anchors : nc;

    for (size_t i = 0; i < limit; i++)
    {
        const Cluster *cl = &ctx->state.clusters[i];
        if (cl->anchor.data)
        {
            if (cl->anchor.is_double)
            {
                memcpy(
                    out_coords + (i * ctx->ndim),
                    cl->anchor.data,
                    ctx->ndim * sizeof(double)
                );
            }
            else
            {
                const float *fdata = (const float *)cl->anchor.data;
                for (size_t d = 0; d < ctx->ndim; d++)
                {
                    out_coords[i * ctx->ndim + d] = (double)fdata[d];
                }
            }
        }
        if (out_members)
        {
            out_members[i] = ctx->state.cluster_visitors[i].count;
        }
    }

    return (int64_t)limit;
}

gric_status_t gric_cluster_get_dcc(
    const gric_cluster_t *ctx,
    double               *out_dcc,
    size_t                K)
{
    if (ctx == NULL || out_dcc == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    int N = ctx->config.algo.maxnbclust;
    const double *dcc_min = ctx->state.scratch.dcc_min;
    if (!dcc_min)
    {
        return GRIC_ERR_GENERIC;
    }

    for (size_t i = 0; i < K; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            if (i < (size_t)N && j < (size_t)N)
            {
                out_dcc[i * K + j] = dcc_min[i * N + j];
            }
            else
            {
                out_dcc[i * K + j] = 0.0;
            }
        }
    }

    return GRIC_SUCCESS;
}

gric_status_t gric_cluster_reset(
    gric_cluster_t *ctx)
{
    if (ctx == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    int N = ctx->config.algo.maxnbclust;
    for (int i = 0; i < ctx->state.num_clusters; i++)
    {
        if (ctx->state.clusters[i].anchor.data)
        {
            free(ctx->state.clusters[i].anchor.data);
            ctx->state.clusters[i].anchor.data = NULL;
        }
    }
    memset(ctx->state.clusters, 0, (size_t)N * sizeof(Cluster));
    ctx->state.num_clusters = 0;
    ctx->current_frame_id = 0;
    ctx->prev_assigned = -1;

    for (size_t i = 0; i < ctx->maxnbfr; i++)
    {
        ctx->state.assignments[i] = -1;
    }
    memset(ctx->state.frame_infos, 0, ctx->maxnbfr * sizeof(FrameInfo));
    memset(ctx->state.transition_matrix, 0, (size_t)N * N * sizeof(long));

    return GRIC_SUCCESS;
}

void gric_cluster_destroy(
    gric_cluster_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }

    int N = ctx->config.algo.maxnbclust;
    if (ctx->state.clusters)
    {
        for (int i = 0; i < N; i++)
        {
            if (ctx->state.clusters[i].anchor.data)
            {
                free(ctx->state.clusters[i].anchor.data);
                ctx->state.clusters[i].anchor.data = NULL;
            }
        }
        free(ctx->state.clusters);
    }

    if (ctx->state.cluster_visitors)
    {
        for (int i = 0; i < N; i++)
        {
            if (ctx->state.cluster_visitors[i].frames)
            {
                free(ctx->state.cluster_visitors[i].frames);
            }
        }
        free(ctx->state.cluster_visitors);
    }

    free(ctx->state.assignments);
    free(ctx->state.frame_infos);
    free(ctx->state.transition_matrix);

    ClusterScratch *s = &ctx->state.scratch;
    free(s->mixed_probs);
    free(s->clmembflag);
    free(s->active_clusters);
    free(s->probsortedclindex);
    free(s->cluster_probs);
    free(s->current_gprobs);
    free(s->dcc_min);
    free(s->dcc_max);
    free(s->dcc_measured);
    free(s->consistency_mask);
    free(s->entropy_p_current);
    free(s->entropy_candidates);
    free(s->entropy_prob_scores);
    free(s->entropy_prune_scores);
    free(s->entropy_active_indices);
    free(s->entropy_plog2p);
    free(s->entropy_visited);
    free(s->refine_queue);
    free(s->tuple_pred_candidates);
    free(s->pred_candidates);
    free(s->local_candidates);

    ClusterTelemetry *t = &ctx->state.telemetry;
    free(t->pruned_fraction_sum);
    free(t->step_counts);
    free(t->dist_counts);
    free(t->pruned_counts_by_dist);
    free(t->cluster_query_counts);

    free(ctx->temp_indices);
    free(ctx->temp_dists);
    free(ctx->sorting_candidates);

    if (ctx->frame.data)
    {
        free(ctx->frame.data);
    }

    free(ctx);
}

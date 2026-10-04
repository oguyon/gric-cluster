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
#include "gric_bin_io.h"
#include "scalar_quant.h"

#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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
    cfg->query_mode = 0;
    cfg->ncpu = 1;

    return GRIC_SUCCESS;
}

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

static int grow_context_capacity(
    gric_cluster_t *ctx)
{
    size_t old_N = (size_t)ctx->config.algo.maxnbclust;
    size_t new_N = old_N * 2;
    if (new_N < 16)
    {
        new_N = 16;
    }

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

    long *new_tm = (long *)grow_matrix(ctx->state.transition_matrix, old_N, new_N, sizeof(long));
    if (!new_tm)
    {
        return -1;
    }
    ctx->state.transition_matrix = new_tm;

    ClusterScratch *s = &ctx->state.scratch;
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

    s->dcc_min = (double *)grow_matrix(s->dcc_min, old_N, new_N, sizeof(double));
    s->dcc_max = (double *)grow_matrix(s->dcc_max, old_N, new_N, sizeof(double));
    s->dcc_measured = (char *)grow_matrix(s->dcc_measured, old_N, new_N, sizeof(char));

    size_t new_mask_words = new_N * new_N * ((new_N + 63) / 64);
    uint64_t *new_mask = (uint64_t *)calloc(new_mask_words, sizeof(uint64_t));
    if (new_mask)
    {
        free(s->consistency_mask);
        s->consistency_mask = new_mask;
    }

    ctx->temp_indices = (int *)realloc(ctx->temp_indices, new_N * sizeof(int));
    ctx->temp_dists = (double *)realloc(ctx->temp_dists, new_N * sizeof(double));
    ctx->sorting_candidates = (Candidate *)realloc(
        ctx->sorting_candidates, new_N * sizeof(Candidate)
    );

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

    ClusterTelemetry *t = &ctx->state.telemetry;
    t->pruned_fraction_sum = (double *)realloc(t->pruned_fraction_sum, new_N * sizeof(double));
    t->step_counts = (long *)realloc(t->step_counts, new_N * sizeof(long));
    t->dist_counts = (long *)realloc(
        t->dist_counts, (new_N + 1) * sizeof(long));
    t->pruned_counts_by_dist = (long *)realloc(
        t->pruned_counts_by_dist, (new_N + 1) * sizeof(long));
    t->cluster_query_counts = (long *)realloc(t->cluster_query_counts, new_N * sizeof(long));

    memset(&s->clmembflag[old_N], 0, (new_N - old_N) * sizeof(int));
    memset(&s->mixed_probs[old_N], 0, (new_N - old_N) * sizeof(double));
    memset(&s->current_gprobs[old_N], 0, (new_N - old_N) * sizeof(double));
    memset(&s->probsortedclindex[old_N], 0, (new_N - old_N) * sizeof(int));
    if (s->entropy_visited)
    {
        memset(&s->entropy_visited[old_N], 0, (new_N - old_N) * sizeof(uint8_t));
    }
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

    ctx->config.algo.maxnbclust = (int)new_N;
    ctx->state.telemetry.max_steps_recorded = (int)new_N;

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
    ctx->config.algo.query_mode = cfg->query_mode;
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

    size_t sz_N = (size_t)N;
    ctx->state.clusters = (Cluster *)calloc(sz_N, sizeof(Cluster));
    ctx->state.cluster_visitors = (VisitorList *)calloc(sz_N, sizeof(VisitorList));
    ctx->state.assignments = (int *)malloc((size_t)maxfr * sizeof(int));
    ctx->state.frame_infos = (FrameInfo *)calloc((size_t)maxfr, sizeof(FrameInfo));
    ctx->state.transition_matrix = (long *)calloc(sz_N * sz_N, sizeof(long));

    ClusterScratch *s = &ctx->state.scratch;
    s->mixed_probs = (double *)calloc(sz_N, sizeof(double));
    s->clmembflag = (int *)calloc(sz_N, sizeof(int));
    s->active_clusters = (int *)malloc(sz_N * sizeof(int));
    s->num_active_clusters = 0;
    s->probsortedclindex = (int *)calloc(sz_N, sizeof(int));
    if (posix_memalign((void **)&s->cluster_probs, 64, sz_N * sizeof(double)) != 0)
    {
        s->cluster_probs = NULL;
    }
    else
    {
        memset(s->cluster_probs, 0, sz_N * sizeof(double));
    }
    s->current_gprobs = (double *)calloc(sz_N, sizeof(double));
    s->dcc_min = (double *)calloc(sz_N * sz_N, sizeof(double));
    s->dcc_max = (double *)calloc(sz_N * sz_N, sizeof(double));
    s->dcc_measured = (char *)calloc(sz_N * sz_N, sizeof(char));

    size_t mask_words = sz_N * sz_N * ((sz_N + 63) / 64);
    s->consistency_mask = (uint64_t *)calloc(mask_words, sizeof(uint64_t));

    s->entropy_p_current = (double *)calloc(sz_N, sizeof(double));
    s->entropy_candidates = (Candidate *)calloc(sz_N, sizeof(Candidate));
    s->entropy_prob_scores = (TargetScore *)calloc(sz_N, sizeof(TargetScore));
    s->entropy_prune_scores = (TargetScore *)calloc(sz_N, sizeof(TargetScore));
    s->entropy_active_indices = (int *)calloc(sz_N, sizeof(int));
    s->entropy_plog2p = (double *)calloc(sz_N, sizeof(double));
    s->entropy_visited = (uint8_t *)calloc(sz_N, sizeof(uint8_t));
    s->refine_queue = (Candidate *)calloc(sz_N, sizeof(Candidate));
    s->refine_queue_capacity = N;
    s->tuple_pred_candidates = (int *)calloc(sz_N, sizeof(int));
    s->pred_candidates = (int *)calloc(sz_N, sizeof(int));
    s->local_candidates = (int *)calloc(sz_N, sizeof(int));

    ClusterTelemetry *t = &ctx->state.telemetry;
    t->max_steps_recorded = N;
    t->pruned_fraction_sum = (double *)calloc(sz_N, sizeof(double));
    t->step_counts = (long *)calloc(sz_N, sizeof(long));
    t->dist_counts = (long *)calloc(sz_N + 1, sizeof(long));
    t->pruned_counts_by_dist = (long *)calloc(sz_N + 1, sizeof(long));
    t->cluster_query_counts = (long *)calloc(sz_N, sizeof(long));

    ctx->temp_indices = (int *)calloc(sz_N, sizeof(int));
    ctx->temp_dists = (double *)calloc(sz_N, sizeof(double));
    ctx->sorting_candidates = (Candidate *)calloc(sz_N, sizeof(Candidate));

    ctx->frame.data = NULL;
    ctx->frame.is_double = 1;
    ctx->frame.width = (int)ndim;
    ctx->frame.height = 1;
    ctx->frame.id = 0;
    ctx->frame.is_mmap = 0;
    ctx->frame.is_borrowed = 1;

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

    ctx->frame.data = (void *)coords;
    ctx->frame.is_double = 1;
    ctx->frame.id = ctx->current_frame_id;
    ctx->frame.width = (int)ctx->ndim;
    ctx->frame.height = 1;
    ctx->frame.is_mmap = 0;
    ctx->frame.is_borrowed = 1;

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
    size_t sz_N = (size_t)N;
    for (int i = 0; i < ctx->state.num_clusters; i++)
    {
        if (ctx->state.clusters[i].anchor.data)
        {
            free(ctx->state.clusters[i].anchor.data);
            ctx->state.clusters[i].anchor.data = NULL;
        }
    }
    memset(ctx->state.clusters, 0, sz_N * sizeof(Cluster));
    ctx->state.num_clusters = 0;
    ctx->state.scratch.probsorted_count = 0;
    ctx->current_frame_id = 0;
    ctx->prev_assigned = -1;

    for (size_t i = 0; i < ctx->maxnbfr; i++)
    {
        ctx->state.assignments[i] = -1;
    }
    memset(ctx->state.frame_infos, 0, ctx->maxnbfr * sizeof(FrameInfo));
    memset(ctx->state.transition_matrix, 0, sz_N * sz_N * sizeof(long));

    return GRIC_SUCCESS;
}

gric_status_t gric_cluster_set_query_mode(
    gric_cluster_t *ctx,
    int             enabled)
{
    if (ctx == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    ctx->config.algo.query_mode = enabled ? 1 : 0;
    return GRIC_SUCCESS;
}

double gric_cluster_get_last_dist(
    const gric_cluster_t *ctx)
{
    if (ctx == NULL)
    {
        return -1.0;
    }

    return ctx->state.telemetry.last_assignment_dist;
}

int64_t gric_cluster_load_anchors(
    gric_cluster_t *ctx,
    const char     *anchors_path)
{
    if (ctx == NULL || anchors_path == NULL || anchors_path[0] == '\0')
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    char file_path[2048];
    struct stat st;
    if (stat(anchors_path, &st) == 0 && S_ISDIR(st.st_mode))
    {
        snprintf(file_path, sizeof(file_path), "%s/anchors.bin", anchors_path);
        if (stat(file_path, &st) != 0)
        {
            snprintf(file_path, sizeof(file_path), "%s/anchors.txt", anchors_path);
            if (stat(file_path, &st) != 0)
            {
                return GRIC_ERR_INVALID_PARAM;
            }
        }
    }
    else
    {
        strncpy(file_path, anchors_path, sizeof(file_path) - 1);
        file_path[sizeof(file_path) - 1] = '\0';
    }

    if (ctx->state.num_clusters > 0)
    {
        gric_cluster_reset(ctx);
    }

    uint64_t K = 0;
    size_t ndim = ctx->ndim;
    int is_bin = 0;

    FILE *fp = fopen(file_path, "rb");
    if (fp == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    gric_bin_header_t hdr;
    char *comment = NULL;
    if (gric_bin_read_header(fp, &hdr, &comment) == 0)
    {
        if (hdr.ndim >= 2 && hdr.dims[1] == (uint64_t)ndim)
        {
            is_bin = 1;
            K = hdr.dims[0];
        }
        if (comment)
        {
            free(comment);
        }
    }

    if (!is_bin)
    {
        fclose(fp);
        fp = fopen(file_path, "r");
        if (fp == NULL)
        {
            return GRIC_ERR_INVALID_PARAM;
        }

        char line[65536];
        uint64_t line_count = 0;
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
            {
                continue;
            }
            line_count++;
        }
        if (line_count == 0)
        {
            fclose(fp);
            return GRIC_ERR_INVALID_PARAM;
        }
        K = line_count;
        rewind(fp);
    }

    while ((size_t)K >= (size_t)ctx->config.algo.maxnbclust)
    {
        if (grow_context_capacity(ctx) != 0)
        {
            fclose(fp);
            return GRIC_ERR_OUT_OF_MEMORY;
        }
    }

    size_t elem_size = ctx->config.algo.use_double ? sizeof(double) : sizeof(float);

    for (uint64_t c = 0; c < K; c++)
    {
        Cluster *cl = &ctx->state.clusters[c];
        cl->id = (int)c;
        cl->anchor.width = (int)ndim;
        cl->anchor.height = 1;
        cl->anchor.is_double = ctx->config.algo.use_double;
        cl->anchor.is_mmap = 0;
        cl->anchor.is_borrowed = 0;
        if (posix_memalign(&cl->anchor.data, 64, ndim * elem_size) != 0)
        {
            fclose(fp);
            return GRIC_ERR_OUT_OF_MEMORY;
        }

        if (is_bin)
        {
            if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
            {
                if (cl->anchor.is_double)
                {
                    float *fbuf = (float *)malloc(ndim * sizeof(float));
                    if (!fbuf || fread(fbuf, sizeof(float), ndim, fp) != ndim)
                    {
                        free(fbuf);
                        fclose(fp);
                        return GRIC_ERR_GENERIC;
                    }
                    double *ddata = (double *)cl->anchor.data;
                    for (size_t d = 0; d < ndim; d++)
                    {
                        ddata[d] = (double)fbuf[d];
                    }
                    free(fbuf);
                }
                else
                {
                    if (fread(cl->anchor.data, sizeof(float), ndim, fp) != ndim)
                    {
                        fclose(fp);
                        return GRIC_ERR_GENERIC;
                    }
                }
            }
            else
            {
                if (cl->anchor.is_double)
                {
                    if (fread(cl->anchor.data, sizeof(double), ndim, fp) != ndim)
                    {
                        fclose(fp);
                        return GRIC_ERR_GENERIC;
                    }
                }
                else
                {
                    double *dbuf = (double *)malloc(ndim * sizeof(double));
                    if (!dbuf || fread(dbuf, sizeof(double), ndim, fp) != ndim)
                    {
                        free(dbuf);
                        fclose(fp);
                        return GRIC_ERR_GENERIC;
                    }
                    float *fdata = (float *)cl->anchor.data;
                    for (size_t d = 0; d < ndim; d++)
                    {
                        fdata[d] = (float)dbuf[d];
                    }
                    free(dbuf);
                }
            }
        }
        else
        {
            char line[65536];
            while (fgets(line, sizeof(line), fp) != NULL)
            {
                if (line[0] != '#' && line[0] != '\n' && line[0] != '\0')
                {
                    break;
                }
            }
            char *ptr = line;
            if (cl->anchor.is_double)
            {
                double *ddata = (double *)cl->anchor.data;
                for (size_t d = 0; d < ndim; d++)
                {
                    ddata[d] = strtod(ptr, &ptr);
                }
            }
            else
            {
                float *fdata = (float *)cl->anchor.data;
                for (size_t d = 0; d < ndim; d++)
                {
                    fdata[d] = strtof(ptr, &ptr);
                }
            }
        }

        cl->prob = 1.0 / (double)K;
        if (ctx->state.scratch.cluster_probs)
        {
            ctx->state.scratch.cluster_probs[c] = 1.0 / (double)K;
        }

        if (ctx->config.optim.use_sq16)
        {
            if (posix_memalign((void **)&cl->anchor_sq16, 64, ndim * sizeof(int16_t)) == 0)
            {
                if (cl->anchor.is_double)
                {
                    sq16_quantize_double(
                        (const double *)cl->anchor.data,
                        cl->anchor_sq16,
                        &ctx->config.optim.sq16_params);
                }
                else
                {
                    sq16_quantize_float(
                        (const float *)cl->anchor.data,
                        cl->anchor_sq16,
                        &ctx->config.optim.sq16_params);
                }
            }
        }
    } // for (uint64_t c = 0; c < K; c++)
    fclose(fp);

    ctx->state.num_clusters = (int)K;

    /* Compute DCC distances between anchors */
    int N = ctx->config.algo.maxnbclust;
    for (int i = 0; i < (int)K; i++)
    {
        ctx->state.scratch.dcc_min[i * N + i] = 0.0;
        ctx->state.scratch.dcc_max[i * N + i] = 0.0;
        ctx->state.scratch.dcc_measured[i * N + i] = 1;
        for (int j = i + 1; j < (int)K; j++)
        {
            double d = 0.0;
            if (ctx->config.algo.use_double)
            {
                d = framedist_double(
                    (const double *)ctx->state.clusters[i].anchor.data,
                    (const double *)ctx->state.clusters[j].anchor.data,
                    (long)ndim);
            }
            else
            {
                d = framedist_float(
                    (const float *)ctx->state.clusters[i].anchor.data,
                    (const float *)ctx->state.clusters[j].anchor.data,
                    (long)ndim);
            }
            ctx->state.scratch.dcc_min[i * N + j] = d;
            ctx->state.scratch.dcc_min[j * N + i] = d;
            ctx->state.scratch.dcc_max[i * N + j] = d;
            ctx->state.scratch.dcc_max[j * N + i] = d;
            ctx->state.scratch.dcc_measured[i * N + j] = 1;
            ctx->state.scratch.dcc_measured[j * N + i] = 1;
        }
    }

    for (int c = 0; c < (int)K; c++)
    {
        update_consistency_mask_for_new_cluster(&ctx->config, &ctx->state, c);
    }

    ctx->config.algo.query_mode = 1;
    return (int64_t)K;
}

/**
 * gric_cluster_get_stats() - Retrieve current telemetry and computation metrics.
 * @ctx:   Active clustering handle.
 * @stats: Destination statistics struct pointer.
 *
 * Return: GRIC_SUCCESS on success, or negative error code on invalid parameters.
 */
gric_status_t gric_cluster_get_stats(
    const gric_cluster_t *ctx,
    gric_cluster_stats_t *stats)
{
    if (ctx == NULL || stats == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    memset(stats, 0, sizeof(gric_cluster_stats_t));

    const ClusterState     *s = &ctx->state;
    const ClusterTelemetry *t = &s->telemetry;

    stats->total_frames_processed = (uint64_t)t->total_frames_processed;
    stats->num_clusters = (uint32_t)s->num_clusters;
    stats->num_new_clusters = t->num_new_clusters;
    stats->framedist_calls = (uint64_t)t->framedist_calls;
    stats->framedist_calls_sample = (uint64_t)t->framedist_calls_sample;
    stats->framedist_calls_intercluster = (uint64_t)t->framedist_calls_intercluster;
    stats->clusters_pruned = (uint64_t)t->clusters_pruned;
    stats->last_assignment_dist = t->last_assignment_dist;
    stats->last_frame_dists = t->last_frame_dists;
    stats->last_frame_dfc = t->last_frame_dfc;
    stats->last_frame_dcc = t->last_frame_dcc;
    stats->time_io_ms = t->time_io_ms;
    stats->time_step_1 = t->time_step_1;
    stats->time_step_2 = t->time_step_2;
    stats->time_step_3a = t->time_step_3a;
    stats->time_step_3b = t->time_step_3b;
    stats->time_step_3c = t->time_step_3c;
    stats->time_step_4 = t->time_step_4;
    stats->time_step_5 = t->time_step_5;
    stats->time_step_refine = t->time_step_refine;
    stats->entropy_last_initial = t->entropy_last_initial;

    if (t->total_frames_processed > 0)
    {
        stats->entropy_avg_initial = t->entropy_sum_initial / (double)t->total_frames_processed;
    }
    else
    {
        stats->entropy_avg_initial = 0.0;
    }

    uint64_t total_ecalls = t->entropy_frames_gated + t->entropy_frames_evaluated;
    if (total_ecalls > 0)
    {
        stats->entropy_gate_ratio = (double)t->entropy_frames_gated / (double)total_ecalls;
    }
    else
    {
        stats->entropy_gate_ratio = 0.0;
    }

    stats->dcc_entries_populated = t->dcc_entries_populated;
    stats->dcc_pairs_total = t->dcc_pairs_total;
    stats->memo_hits = t->memo_hits;
    stats->memo_lookups = t->memo_lookups;
    stats->memo_cache_entries = t->memo_cache_entries;
    stats->memo_cache_capacity = t->memo_cache_capacity;

    return GRIC_SUCCESS;
}

/**
 * gric_cluster_save_results() - Export clustering results to disk in GRIC binary format.
 * @ctx:     Active clustering handle.
 * @out_dir: Destination directory path (must exist or will be created).
 *
 * Return: GRIC_SUCCESS on success, or negative error code on failure.
 */
gric_status_t gric_cluster_save_results(
    const gric_cluster_t *ctx,
    const char           *out_dir)
{
    if (ctx == NULL)
    {
        return GRIC_ERR_INVALID_PARAM;
    }

    const char *target_dir = (out_dir != NULL && out_dir[0] != '\0') ? out_dir : ".";

    /* Ensure destination directory exists */
    struct stat st;
    if (stat(target_dir, &st) == -1)
    {
        if (mkdir(target_dir, 0777) != 0)
        {
            return GRIC_ERR_GENERIC;
        }
    }

    const ClusterState  *state = &ctx->state;
    const ClusterConfig *config = &ctx->config;
    int                  k = state->num_clusters;
    size_t               ndim = ctx->ndim;

    if (k <= 0)
    {
        return GRIC_SUCCESS;
    }

    char path[1024];

    /* 1. Export anchors.bin (Centroids matrix [K x ndim]) */
    snprintf(path, sizeof(path), "%s/anchors.bin", target_dir);
    FILE *fp = fopen(path, "wb");
    if (fp != NULL)
    {
        gric_bin_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.file_type = GRIC_BIN_TYPE_ANCHORS;
        hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        hdr.ndim = (ndim > 1) ? 2 : 1;
        hdr.dims[0] = (uint64_t)k;
        hdr.dims[1] = (uint64_t)ndim;
        hdr.num_elements = (uint64_t)k * (uint64_t)ndim;

        if (config->algo.use_double)
        {
            hdr.data_type = GRIC_BIN_DTYPE_FLOAT64;
            hdr.data_bytes = hdr.num_elements * sizeof(double);
            if (gric_bin_write_header(fp, &hdr, "Cluster centroids") == 0)
            {
                for (int i = 0; i < k; i++)
                {
                    if (state->clusters[i].anchor.is_double)
                    {
                        fwrite(state->clusters[i].anchor.data, sizeof(double), ndim, fp);
                    }
                    else
                    {
                        const float *fdata = (const float *)state->clusters[i].anchor.data;
                        for (size_t d = 0; d < ndim; d++)
                        {
                            double v = (double)fdata[d];
                            fwrite(&v, sizeof(double), 1, fp);
                        }
                    }
                }
            }
        }
        else
        {
            hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
            hdr.data_bytes = hdr.num_elements * sizeof(float);
            if (gric_bin_write_header(fp, &hdr, "Cluster centroids") == 0)
            {
                if (state->anchor_matrix_float != NULL)
                {
                    fwrite(state->anchor_matrix_float, sizeof(float), hdr.num_elements, fp);
                }
                else
                {
                    for (int i = 0; i < k; i++)
                    {
                        if (state->clusters[i].anchor.is_double)
                        {
                            const double *ddata = (const double *)state->clusters[i].anchor.data;
                            for (size_t d = 0; d < ndim; d++)
                            {
                                float v = (float)ddata[d];
                                fwrite(&v, sizeof(float), 1, fp);
                            }
                        }
                        else
                        {
                            fwrite(state->clusters[i].anchor.data, sizeof(float), ndim, fp);
                        }
                    }
                }
            }
        }
        fclose(fp);
    }

    /* 2. Export dcc.bin (Pairwise inter-cluster distance matrix [K x K]) */
    snprintf(path, sizeof(path), "%s/dcc.bin", target_dir);
    fp = fopen(path, "wb");
    if (fp != NULL)
    {
        gric_bin_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.file_type = GRIC_BIN_TYPE_DCC;
        hdr.data_type = GRIC_BIN_DTYPE_FLOAT64;
        hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR | GRIC_BIN_FLAG_SYMMETRIC;
        hdr.ndim = 2;
        hdr.dims[0] = (uint64_t)k;
        hdr.dims[1] = (uint64_t)k;
        hdr.num_elements = (uint64_t)k * (uint64_t)k;
        hdr.data_bytes = hdr.num_elements * sizeof(double);

        if (gric_bin_write_header(fp, &hdr, "Pairwise DCC matrix") == 0)
        {
            int maxnbc = config->algo.maxnbclust;
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double d = (state->scratch.dcc_min != NULL)
                                   ? state->scratch.dcc_min[i * maxnbc + j]
                                   : 0.0;
                    fwrite(&d, sizeof(double), 1, fp);
                }
            }
        }
        fclose(fp);
    }

    /* 3. Export cluster_counts.bin (Member counts [K]) */
    snprintf(path, sizeof(path), "%s/cluster_counts.bin", target_dir);
    fp = fopen(path, "wb");
    if (fp != NULL)
    {
        gric_bin_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.file_type = GRIC_BIN_TYPE_COUNTS;
        hdr.data_type = GRIC_BIN_DTYPE_UINT32;
        hdr.flags = 0;
        hdr.ndim = 1;
        hdr.dims[0] = (uint64_t)k;
        hdr.num_elements = (uint64_t)k;
        hdr.data_bytes = hdr.num_elements * sizeof(uint32_t);

        if (gric_bin_write_header(fp, &hdr, "Cluster member counts") == 0)
        {
            for (int i = 0; i < k; i++)
            {
                uint32_t cnt = (state->cluster_visitors != NULL)
                                   ? (uint32_t)state->cluster_visitors[i].count
                                   : 0;
                fwrite(&cnt, sizeof(uint32_t), 1, fp);
            }
        }
        fclose(fp);
    }

    /* 4. Export cluster_radii.bin (Maximum cluster radii [K]) */
    snprintf(path, sizeof(path), "%s/cluster_radii.bin", target_dir);
    fp = fopen(path, "wb");
    if (fp != NULL)
    {
        gric_bin_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.file_type = GRIC_BIN_TYPE_GENERIC;
        hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
        hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        hdr.ndim = 1;
        hdr.dims[0] = (uint64_t)k;
        hdr.num_elements = (uint64_t)k;
        hdr.data_bytes = hdr.num_elements * sizeof(float);

        if (gric_bin_write_header(fp, &hdr, "Cluster max radii") == 0)
        {
            for (int i = 0; i < k; i++)
            {
                float r = (float)config->algo.rlim;
                fwrite(&r, sizeof(float), 1, fp);
            }
        }
        fclose(fp);
    }

    /* 5. Export frame_membership.bin (Frame-to-cluster assignments [N]) */
    long total_frames = state->telemetry.total_frames_processed;
    long nframes = (total_frames > (long)ctx->maxnbfr) ? (long)ctx->maxnbfr : total_frames;
    if (nframes > 0 && state->assignments != NULL)
    {
        snprintf(path, sizeof(path), "%s/frame_membership.bin", target_dir);
        fp = fopen(path, "wb");
        if (fp != NULL)
        {
            gric_bin_header_t hdr;
            memset(&hdr, 0, sizeof(hdr));
            hdr.file_type = GRIC_BIN_TYPE_MEMBERSHIP;
            hdr.data_type = GRIC_BIN_DTYPE_UINT32;
            hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
            hdr.ndim = 1;
            hdr.dims[0] = (uint64_t)nframes;
            hdr.num_elements = (uint64_t)nframes;
            hdr.data_bytes = hdr.num_elements * sizeof(uint32_t);

            if (gric_bin_write_header(fp, &hdr, "Frame membership") == 0)
            {
                uint32_t *mem_buf = (uint32_t *)malloc(hdr.num_elements * sizeof(uint32_t));
                if (mem_buf != NULL)
                {
                    for (long f = 0; f < nframes; f++)
                    {
                        mem_buf[f] = (state->assignments[f] >= 0)
                                         ? (uint32_t)state->assignments[f]
                                         : 0;
                    }
                    fwrite(mem_buf, sizeof(uint32_t), hdr.num_elements, fp);
                    free(mem_buf);
                }
            }
            fclose(fp);
        }
    }

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


    free(ctx);
}

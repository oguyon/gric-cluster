/**
 * @file wasm_cluster.c
 * @brief Single-tile clustering WebAssembly bridge API.
 */

#include "wasm_internal.h"

/* -------------------------------------------------------
 * Global required by cluster_core.h / cluster_step.c
 * ------------------------------------------------------- */
volatile sig_atomic_t stop_requested = 0;

/* -------------------------------------------------------
 * get_dist() — must match cluster_core.h signature.
 * Called by measure_distance_to_cluster.c and
 * handle_new_cluster_creation.c.
 *
 * Stripped of file I/O and verbose printf for WASM.
 * ------------------------------------------------------- */

/**
 * get_dist() - Evaluate Euclidean distance and update
 *              telemetry counters.
 * @a:             First frame (current sample).
 * @b:             Second frame (cluster anchor).
 * @cluster_idx:   Cluster index (>=0 for sample-cluster,
 *                 <0 for inter-cluster).
 * @cluster_prob:  Prior probability (unused in WASM).
 * @current_gprob: Geometric probability (unused).
 * @config:        Clustering configuration.
 * @state:         Clustering state.
 *
 * Return: Euclidean distance, or -1.0 on dimension mismatch.
 */
double get_dist(
    Frame         *a,
    Frame         *b,
    int            cluster_idx,
    double         cluster_prob,
    double         current_gprob,
    ClusterConfig *config,
    ClusterState  *state)
{
    (void)cluster_prob;
    (void)current_gprob;
    (void)config;

    state->telemetry.framedist_calls++;
    if (cluster_idx >= 0)
    {
        state->telemetry.framedist_calls_sample++;
    }
    else
    {
        state->telemetry.framedist_calls_intercluster++;
    }

    return framedist(a, b);
}

/**
 * print_clustering_metrics() - Stub for WASM
 *    (referenced by some step files but not needed).
 */
void print_clustering_metrics(
    const ClusterState *state,
    int                 tile_id)
{
    (void)state;
    (void)tile_id;
}

/**
 * is_ascii_input_mode() - Stub returning 0.
 */
int is_ascii_input_mode(void)
{
    return 0;
}

/**
 * free_frame() - WASM version of frame cleanup.
 *
 * In native code, free_frame() frees both the Frame
 * struct and its data buffer. In WASM, the Frame struct
 * is embedded in WasmHandle (must NOT be freed), but
 * we DO free the data buffer since processFrame()
 * allocates a fresh one per call.
 *
 * If the step code stole the data pointer (set it to
 * NULL for cluster anchor ownership), this is a no-op.
 */
void free_frame(Frame *frame_ptr)
{
    if (frame_ptr != NULL && frame_ptr->data != NULL)
    {
        free(frame_ptr->data);
        frame_ptr->data = NULL;
    }
    /* Do NOT free(frame_ptr) — it's embedded in WasmHandle */
}

/* -------------------------------------------------------
 * Exported WASM API functions
 * ------------------------------------------------------- */

/**
 * wasm_cluster_init() - Allocate and initialize a single-tile clustering instance.
 * @rlim:                       Clustering radius limit for candidate neighborhood.
 * @maxnbclust:                 Maximum number of clusters (or 0 for dynamic growth).
 * @maxnbfr:                    Maximum number of frames expected in session.
 * @ndim:                       Feature dimensionality per frame.
 * @entropy_mode:               Entropy gating and early-exit mode flag.
 * @te4_mode:                   Triangle-inequality lower bound test 4 flag.
 * @te5_mode:                   Triangle-inequality lower bound test 5 flag.
 * @pred_mode:                  Temporal Markov prediction mode flag.
 * @pred_h:                     Prediction history horizon.
 * @gprob_mode:                 Geometric prior probability mode flag.
 * @tm_mixing_coeff:            Mixing coefficient between transition matrix and priors.
 * @soft_bayesian_mode:         Soft Bayesian likelihood weighting flag.
 * @xtile_mode:                 Cross-tile tuple tracking flag.
 * @sparse_dcc_mode:            Sparse inter-cluster distance cache mode.
 * @sparse_dcc_extra_evals:     Extra DCC evaluation count limit.
 * @entropy_gate_bits:          Information entropy gating threshold in bits.
 * @entropy_first_gate_bits:    First-stage entropy threshold.
 * @entropy_fast_mode:          Accelerated approximation mode for entropy log2.
 * @soft_bayesian_sigma_coeff:  Standard deviation scaling for soft likelihood.
 * @maxcl_strategy:             Cluster eviction or limit strategy.
 * @discard_fraction:           Fraction of lowest-weight clusters to discard when full.
 * @max_gprob_visitors:         Maximum visitor depth for geometric probability.
 *
 * Purpose & Context ("What is this used for?"):
 * Serves as the primary constructor for the WebAssembly single-tile clustering
 * engine. Allocates the opaque WasmHandle struct, initializes cluster tables,
 * distance caches, scratch buffers, and telemetry structures, exposing the
 * core C clustering algorithm to browser-based interactive tools and Web Workers.
 *
 * Return: Opaque pointer to initialized WasmHandle on success, or NULL on failure.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_cluster_init(
    double rlim,
    int    maxnbclust,
    long   maxnbfr,
    int    ndim,
    int    entropy_mode,
    int    te4_mode,
    int    te5_mode,
    int    pred_mode,
    int    pred_h,
    int    gprob_mode,
    double tm_mixing_coeff,
    int    soft_bayesian_mode,
    int    xtile_mode,
    int    sparse_dcc_mode,
    int    sparse_dcc_extra_evals,
    double entropy_gate_bits,
    double entropy_first_gate_bits,
    int    entropy_fast_mode,
    double soft_bayesian_sigma_coeff,
    int    maxcl_strategy,
    double discard_fraction,
    int    max_gprob_visitors)
{
    WasmHandle *h = (WasmHandle *)calloc(
        1, sizeof(WasmHandle)
    );
    if (!h)
    {
        return NULL;
    }

    h->ndim = ndim;
    h->maxnbfr = maxnbfr;
    h->current_frame_id = 0;
    h->prev_assigned = -1;
    h->user_maxcl = maxnbclust;

    /* --- ConfigAlgorithm --- */
    h->config.algo.rlim = rlim;
    h->config.algo.maxnbclust = maxnbclust;
    h->config.algo.deltaprob = 0.0;
    h->config.algo.tm_mixing_coeff = tm_mixing_coeff;
    h->config.algo.maxcl_strategy =
        (MaxClustStrategy)maxcl_strategy;
    h->config.algo.discard_fraction = discard_fraction;
    h->config.algo.use_double = 1;

    /* --- ConfigInput --- */
    h->config.input.maxnbfr = maxnbfr;

    /* --- ConfigOptim --- */
    h->config.optim.entropy_mode = entropy_mode;
    h->config.optim.te4_mode = te4_mode;
    h->config.optim.te5_mode = te5_mode;
    h->config.optim.pred_mode = pred_mode;
    h->config.optim.pred_len = 4;
    h->config.optim.pred_h = pred_h;
    h->config.optim.pred_n = 3;
    h->config.optim.gprob_mode = gprob_mode;
    h->config.optim.fmatch_a = 2.0;
    h->config.optim.fmatch_b = 0.5;
    h->config.optim.max_gprob_visitors =
        max_gprob_visitors;
    h->config.optim.entropy_max_targets = 24;
    h->config.optim.entropy_min_prob = 0.001;
    h->config.optim.entropy_gate_bits =
        entropy_gate_bits;
    h->config.optim.entropy_first_gate_bits =
        entropy_first_gate_bits;
    h->config.optim.entropy_fast_mode =
        entropy_fast_mode;
    h->config.optim.entropy_leader_shortcut = 0;
    h->config.optim.entropy_leader_cutoff = 0.50;
    h->config.optim.sparse_dcc_mode = sparse_dcc_mode;
    h->config.optim.sparse_dcc_extra_evals =
        sparse_dcc_extra_evals;
    h->config.optim.soft_bayesian_mode =
        soft_bayesian_mode;
    h->config.optim.soft_bayesian_sigma_coeff =
        soft_bayesian_sigma_coeff;
    h->config.optim.xtile_mode = xtile_mode;
    h->config.optim.xtile_decay = 0.70;
    h->config.optim.use_batch_dist = 1;

    /* --- ConfigOutput (all disabled for WASM) --- */
    h->config.output.verbose_level = 0;
    h->config.output.progress_mode = 0;

    /* --- Allocate ClusterState arrays --- */
    int N = maxnbclust;

    h->state.clusters =
        (Cluster *)calloc(N, sizeof(Cluster));
    h->state.cluster_visitors =
        (VisitorList *)calloc(N, sizeof(VisitorList));
    h->state.assignments =
        (int *)malloc((size_t)maxnbfr * sizeof(int));
    h->state.frame_infos =
        (FrameInfo *)calloc(maxnbfr, sizeof(FrameInfo));
    h->state.num_clusters = 0;
    h->state.distall_out = NULL;
    h->state.shm_ptr = NULL;
    h->state.cross_tile_hook = NULL;
    h->state.cross_tile_ctx = NULL;

    /* Transition matrix */
    h->state.transition_matrix =
        (long *)calloc((size_t)N * N, sizeof(long));

    /* Scratch buffers */
    {
        ClusterScratch *s = &h->state.scratch;

        s->mixed_probs =
            (double *)calloc(N, sizeof(double));
        s->clmembflag =
            (int *)calloc(N, sizeof(int));
        s->probsortedclindex =
            (int *)calloc(N, sizeof(int));
        s->cluster_probs =
            (double *)calloc(N, sizeof(double));
        s->current_gprobs =
            (double *)calloc(N, sizeof(double));

        s->dcc_min =
            (double *)calloc((size_t)N * N,
                             sizeof(double));
        s->dcc_max =
            (double *)calloc((size_t)N * N,
                             sizeof(double));
        s->dcc_measured =
            (char *)calloc((size_t)N * N,
                           sizeof(char));

        size_t mask_words =
            (size_t)N * N * ((N + 63) / 64);
        s->consistency_mask =
            (uint64_t *)calloc(mask_words,
                               sizeof(uint64_t));

        s->entropy_p_current =
            (double *)calloc(N, sizeof(double));
        s->entropy_candidates =
            (Candidate *)calloc(N, sizeof(Candidate));
        s->entropy_prob_scores =
            (TargetScore *)calloc(N,
                                  sizeof(TargetScore));
        s->entropy_prune_scores =
            (TargetScore *)calloc(N,
                                  sizeof(TargetScore));
        s->entropy_active_indices =
            (int *)calloc(N, sizeof(int));
        s->entropy_plog2p =
            (double *)calloc(N, sizeof(double));
        s->entropy_visited =
            (uint8_t *)calloc(N, sizeof(uint8_t));
        s->refine_queue =
            (Candidate *)calloc(N, sizeof(Candidate));
        s->refine_queue_size = 0;
        s->refine_queue_idx = 0;
        s->refine_queue_capacity = N;
        s->refine_queue_last_num_clusters = 0;
        s->tuple_pred_candidates =
            (int *)calloc(N, sizeof(int));
        s->tuple_pred_count = 0;
        s->pred_candidates =
            (int *)calloc(N, sizeof(int));
        s->local_candidates =
            (int *)calloc(N, sizeof(int));
    } // Scratch buffers

    /* Telemetry arrays */
    {
        ClusterTelemetry *t = &h->state.telemetry;
        t->max_steps_recorded = N;
        t->pruned_fraction_sum =
            (double *)calloc(N, sizeof(double));
        t->step_counts =
            (long *)calloc(N, sizeof(long));
        t->dist_counts =
            (long *)calloc(N + 1, sizeof(long));
        t->pruned_counts_by_dist =
            (long *)calloc(N + 1, sizeof(long));
        t->cluster_query_counts =
            (long *)calloc(N, sizeof(long));
    } // Telemetry arrays

    /* Per-frame scratch arrays */
    h->temp_indices =
        (int *)calloc(N, sizeof(int));
    h->temp_dists =
        (double *)calloc(N, sizeof(double));
    h->sorting_candidates =
        (Candidate *)calloc(N, sizeof(Candidate));

    /* Reusable input frame */
    h->frame.data =
        (double *)calloc(ndim, sizeof(double));
    h->frame.is_double = 1;
    h->frame.width = ndim;
    h->frame.height = 1;
    h->frame.id = 0;
    h->frame.cnt0 = 0;

    if (h->state.clusters == NULL ||
        h->state.cluster_visitors == NULL ||
        h->state.assignments == NULL ||
        h->state.frame_infos == NULL ||
        h->state.transition_matrix == NULL ||
        h->state.scratch.mixed_probs == NULL ||
        h->state.scratch.clmembflag == NULL ||
        h->state.scratch.probsortedclindex == NULL ||
        h->state.scratch.cluster_probs == NULL ||
        h->state.scratch.current_gprobs == NULL ||
        h->state.scratch.dcc_min == NULL ||
        h->state.scratch.dcc_max == NULL ||
        h->state.scratch.dcc_measured == NULL ||
        h->state.scratch.consistency_mask == NULL ||
        h->state.scratch.entropy_p_current == NULL ||
        h->state.scratch.entropy_candidates == NULL ||
        h->state.scratch.entropy_prob_scores == NULL ||
        h->state.scratch.entropy_prune_scores == NULL ||
        h->state.scratch.entropy_active_indices == NULL ||
        h->state.scratch.entropy_plog2p == NULL ||
        h->state.scratch.entropy_visited == NULL ||
        h->state.scratch.refine_queue == NULL ||
        h->state.scratch.tuple_pred_candidates == NULL ||
        h->state.scratch.pred_candidates == NULL ||
        h->state.scratch.local_candidates == NULL ||
        h->state.telemetry.pruned_fraction_sum == NULL ||
        h->state.telemetry.step_counts == NULL ||
        h->state.telemetry.dist_counts == NULL ||
        h->state.telemetry.pruned_counts_by_dist == NULL ||
        h->state.telemetry.cluster_query_counts == NULL ||
        h->temp_indices == NULL ||
        h->temp_dists == NULL ||
        h->sorting_candidates == NULL ||
        h->frame.data == NULL)
    {
        wasm_cluster_free(h);
        return NULL;
    }

    /* Initialize assignments to -1 */
    for (long i = 0; i < maxnbfr; i++)
    {
        h->state.assignments[i] = -1;
    }

    return h;
}

/* -------------------------------------------------------
 * Dynamic Growth Helpers
 * ------------------------------------------------------- */

/**
 * grow_nxn_matrix() - Reallocate and re-stride a 2D N-by-N square matrix.
 * @old:       Existing square matrix buffer to expand (freed on success).
 * @old_n:     Current row and column dimension.
 * @new_n:     New target row and column dimension (must be >= old_n).
 * @elem_size: Size in bytes of each matrix element.
 *
 * Purpose & Context ("What is this used for?"):
 * Used during dynamic capacity expansion when maxnbclust is exceeded in
 * unlimited-cluster WASM mode. Expands 2D inter-cluster matrices (such as DCC
 * distance bounds and transition matrices) while properly copying existing rows
 * to match the wider stride of the enlarged matrix.
 *
 * Return: Pointer to newly allocated, re-strided buffer, or NULL on allocation error.
 */
static void *grow_nxn_matrix(
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

/**
 * grow_capacity() - Double the maximum cluster capacity of a WASM clustering instance.
 * @h: Opaque WasmHandle instance pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Enables dynamic cluster growth in WebAssembly execution when maxnbclust was
 * set to 0 (unlimited mode). Doubles array allocations for clusters, scratch
 * buffers, telemetry tables, and re-strides 2D matrices (DCC, transition matrix)
 * without losing previously computed cluster anchors or membership statistics.
 *
 * Return: 0 on success, or -1 on memory allocation error.
 */
static int grow_capacity(WasmHandle *h)
{
    int old_N = h->config.algo.maxnbclust;
    int new_N = old_N * 2;

    Cluster *new_clusters = (Cluster *)realloc(
        h->state.clusters,
        (size_t)new_N * sizeof(Cluster)
    );
    if (!new_clusters)
    {
        return -1;
    }
    memset(new_clusters + old_N, 0, (size_t)(new_N - old_N) * sizeof(Cluster));
    h->state.clusters = new_clusters;

    VisitorList *new_visitors = (VisitorList *)realloc(
        h->state.cluster_visitors,
        (size_t)new_N * sizeof(VisitorList)
    );
    if (!new_visitors)
    {
        return -1;
    }
    memset(new_visitors + old_N, 0, (size_t)(new_N - old_N) * sizeof(VisitorList));
    h->state.cluster_visitors = new_visitors;

    ClusterScratch *s = &h->state.scratch;
#define GROW_LINEAR(ptr, type)                                             \
    do                                                                     \
    {                                                                      \
        type *tmp = (type *)realloc(ptr, (size_t)new_N * sizeof(type));    \
        if (!tmp)                                                          \
        {                                                                  \
            return -1;                                                     \
        }                                                                  \
        memset(tmp + old_N, 0, (size_t)(new_N - old_N) * sizeof(type));     \
        ptr = tmp;                                                         \
    } while (0)

    GROW_LINEAR(s->mixed_probs, double);
    GROW_LINEAR(s->clmembflag, int);
    GROW_LINEAR(s->probsortedclindex, int);
    GROW_LINEAR(s->cluster_probs, double);
    GROW_LINEAR(s->current_gprobs, double);
    GROW_LINEAR(s->entropy_p_current, double);
    GROW_LINEAR(s->entropy_candidates, Candidate);
    GROW_LINEAR(s->entropy_prob_scores, TargetScore);
    GROW_LINEAR(s->entropy_prune_scores, TargetScore);
    GROW_LINEAR(s->entropy_active_indices, int);
    GROW_LINEAR(s->entropy_plog2p, double);
    GROW_LINEAR(s->entropy_visited, uint8_t);
    GROW_LINEAR(s->refine_queue, Candidate);
    GROW_LINEAR(s->tuple_pred_candidates, int);

    s->refine_queue_capacity = new_N;

    ClusterTelemetry *t = &h->state.telemetry;
    GROW_LINEAR(t->pruned_fraction_sum, double);
    GROW_LINEAR(t->step_counts, long);
    GROW_LINEAR(t->cluster_query_counts, long);

#define GROW_LINEAR_N1(ptr, type)                                          \
    do                                                                     \
    {                                                                      \
        type *tmp = (type *)realloc(ptr, (size_t)(new_N + 1) * sizeof(type)); \
        if (!tmp)                                                          \
        {                                                                  \
            return -1;                                                     \
        }                                                                  \
        memset(tmp + (old_N + 1), 0, (size_t)(new_N - old_N) * sizeof(type)); \
        ptr = tmp;                                                         \
    } while (0)

    GROW_LINEAR_N1(t->dist_counts, long);
    GROW_LINEAR_N1(t->pruned_counts_by_dist, long);

    GROW_LINEAR(h->temp_indices, int);
    GROW_LINEAR(h->temp_dists, double);
    GROW_LINEAR(h->sorting_candidates, Candidate);

#undef GROW_LINEAR
#undef GROW_LINEAR_N1

    long *new_tm = (long *)grow_nxn_matrix(
        h->state.transition_matrix, old_N, new_N, sizeof(long)
    );
    if (!new_tm)
    {
        return -1;
    }
    h->state.transition_matrix = new_tm;

    double *new_dcc_min = (double *)grow_nxn_matrix(
        s->dcc_min, old_N, new_N, sizeof(double)
    );
    if (!new_dcc_min)
    {
        return -1;
    }
    s->dcc_min = new_dcc_min;

    double *new_dcc_max = (double *)grow_nxn_matrix(
        s->dcc_max, old_N, new_N, sizeof(double)
    );
    if (!new_dcc_max)
    {
        return -1;
    }
    s->dcc_max = new_dcc_max;

    char *new_dcc_measured = (char *)grow_nxn_matrix(
        s->dcc_measured, old_N, new_N, sizeof(char)
    );
    if (!new_dcc_measured)
    {
        return -1;
    }
    s->dcc_measured = new_dcc_measured;

    size_t new_mask_words = (size_t)new_N * new_N * ((new_N + 63) / 64);
    uint64_t *new_mask = (uint64_t *)calloc(new_mask_words, sizeof(uint64_t));
    if (!new_mask)
    {
        return -1;
    }
    free(s->consistency_mask);
    s->consistency_mask = new_mask;

    h->config.algo.maxnbclust = new_N;
    t->max_steps_recorded = new_N;

    return 0;
}

/**
 * wasm_cluster_process_frame() - Cluster a single incoming coordinate frame.
 * @ptr:    Opaque WasmHandle pointer.
 * @coords: Array of double-precision coordinates of length ndim.
 * @ndim:   Dimensionality of the input coordinate vector.
 *
 * Purpose & Context ("What is this used for?"):
 * Serves as the real-time frame ingestion entry point for WebAssembly clients.
 * Feeds a newly arrived coordinate vector into the incremental clustering engine,
 * executing triangle-inequality pruning, entropy gating, and temporal prediction
 * to assign the frame to an existing cluster or create a new cluster anchor.
 *
 * Return: Assigned cluster index (>= 0), or negative value on error (-1: invalid,
 *         -2: maximum cluster limit reached).
 */
/**
 * wasm_cluster_ensure_frame_buffer() - Ensure scratch frame buffer is allocated.
 * @h:    Opaque WasmHandle pointer.
 * @ndim: Dimensionality of frame.
 *
 * Purpose & Context ("What is this used for?"):
 * Replenishes h->frame.data after ownership of the buffer is transferred to a newly
 * created cluster anchor in cluster_frame(). Reuses existing heap buffer when not stolen.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static int wasm_cluster_ensure_frame_buffer(
    WasmHandle *h,
    int         ndim)
{
    if (h->frame.data == NULL)
    {
        h->frame.data = (double *)malloc((size_t)ndim * sizeof(double));
        if (!h->frame.data)
        {
            return -1;
        }
        h->frame.is_double = 1;
    }
    return 0;
}

/**
 * wasm_cluster_process_frame() - Cluster a single incoming coordinate frame.
 * @ptr:    Opaque WasmHandle pointer.
 * @coords: Array of double-precision coordinates of length ndim.
 * @ndim:   Dimensionality of the input coordinate vector.
 *
 * Purpose & Context ("What is this used for?"):
 * Serves as the real-time frame ingestion entry point for WebAssembly clients.
 * Feeds a newly arrived coordinate vector into the incremental clustering engine,
 * executing triangle-inequality pruning, entropy gating, and temporal prediction
 * to assign the frame to an existing cluster or create a new cluster anchor.
 *
 * Return: Assigned cluster index (>= 0), or negative value on error (-1: invalid,
 *         -2: maximum cluster limit reached).
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_frame(
    void   *ptr,
    double *coords,
    int     ndim)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !coords || ndim != h->ndim)
    {
        return -1;
    }

    /*
     * The C step functions (initialize_initial_cluster,
     * handle_new_cluster_creation) take ownership of
     * current_frame->data via shallow copy + NULL:
     *
     *   state->clusters[k].anchor = *current_frame;
     *   current_frame->data = NULL;
     *
     * This means each call must provide a FRESH heap
     * allocation, just like native getframe() does.
     * After cluster_frame() returns, h->frame.data
     * will be NULL if the step code stole it.
     */
    if (wasm_cluster_ensure_frame_buffer(h, ndim) != 0)
    {
        return -1;
    }

    memcpy(h->frame.data, coords,
           (size_t)ndim * sizeof(double));
    h->frame.id = h->current_frame_id;
    h->frame.width = ndim;
    h->frame.height = 1;

    if (h->user_maxcl == 0 &&
        h->state.num_clusters >= h->config.algo.maxnbclust - 1)
    {
        if (grow_capacity(h) != 0)
        {
            return -1; /* OOM */
        }
    }

    if (h->state.trace)
    {
        trace_buffer_begin_frame(h->state.trace);
    }

    int result = cluster_frame(
        &h->config,
        &h->state,
        &h->frame,
        &h->prev_assigned,
        NULL, /* ascii_out */
        h->temp_indices,
        h->temp_dists,
        h->sorting_candidates,
        NULL  /* verbose_candidates */
    );

    h->current_frame_id++;
    return result;
}

/**
 * wasm_cluster_process_batch() - Process a contiguous batch of coordinate frames.
 * @ptr:             Opaque WasmHandle pointer.
 * @coords_flat:     Contiguous array of input coordinates [num_frames * ndim].
 * @out_assignments: Output array to store assigned cluster indices [num_frames] (may be NULL).
 * @num_frames:      Number of consecutive frames in the batch.
 * @ndim:            Dimensionality of each frame vector.
 *
 * Purpose & Context ("What is this used for?"):
 * Maximizes throughput across the JavaScript-WebAssembly boundary by batching
 * multiple frames into a single function call, minimizing boundary crossing
 * overhead. Iterates through all frames and records per-frame cluster assignments.
 *
 * Return: Number of successfully processed frames (0..num_frames), or -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_process_batch(
    void   *ptr,
    double *coords_flat,
    int    *out_assignments,
    int     num_frames,
    int     ndim)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !coords_flat || num_frames <= 0 ||
        ndim != h->ndim)
    {
        return -1;
    }

    for (int f = 0; f < num_frames; f++)
    {
        if (wasm_cluster_ensure_frame_buffer(h, ndim) != 0)
        {
            return f;
        }

        memcpy(h->frame.data, &coords_flat[f * ndim], (size_t)ndim * sizeof(double));
        h->frame.id = h->current_frame_id;
        h->frame.width = ndim;
        h->frame.height = 1;

        if (h->user_maxcl == 0 &&
            h->state.num_clusters >= h->config.algo.maxnbclust - 1)
        {
            if (grow_capacity(h) != 0)
            {
                /* Return count of frames processed successfully before OOM */
                return f;
            }
        }

        int assigned = cluster_frame(
            &h->config,
            &h->state,
            &h->frame,
            &h->prev_assigned,
            NULL,
            h->temp_indices,
            h->temp_dists,
            h->sorting_candidates,
            NULL
        );

        if (assigned == -2)
        {
            if (out_assignments)
            {
                out_assignments[f] = -2;
            }
            return f;
        }

        if (out_assignments)
        {
            out_assignments[f] = assigned;
        }

        h->current_frame_id++;
    }

    return num_frames;
}

/**
 * wasm_cluster_get_num_clusters() - Query the current number of active clusters.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Allows JavaScript/browser callers to inspect the count of active clusters
 * formed so far, enabling UI components to adjust visualization scaling and
 * allocate output buffers of appropriate size.
 *
 * Return: Count of active clusters, or 0 if handle is invalid.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_num_clusters(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return 0;
    }
    return h->state.num_clusters;
}

/**
 * wasm_cluster_get_anchors() - Retrieve cluster anchor coordinates and visitor member counts.
 * @ptr:         Opaque WasmHandle pointer.
 * @out_coords:  Output buffer for anchor coordinates [num_clusters * ndim].
 * @out_members: Output buffer for member visit counts [num_clusters].
 * @ndim:        Dimensionality of coordinates.
 *
 * Purpose & Context ("What is this used for?"):
 * Exports the cluster centroid/anchor vectors and their member counts across the
 * WebAssembly linear memory boundary, allowing browser renderers (Canvas, WebGL,
 * Three.js) to display cluster centroids and sizes.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_anchors(
    void   *ptr,
    double *out_coords,
    int    *out_members,
    int     ndim)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_coords || !out_members || ndim != h->ndim)
    {
        return;
    }

    int K = h->state.num_clusters;

    for (int i = 0; i < K; i++)
    {
        const double *data =
            h->state.clusters[i].anchor.data;
        if (data)
        {
            memcpy(&out_coords[i * ndim], data,
                   (size_t)ndim * sizeof(double));
        }
        /* Member count from visitor list */
        out_members[i] =
            h->state.cluster_visitors[i].count;
    }
}

/**
 * wasm_cluster_get_dcc() - Retrieve the inter-cluster distance (DCC) matrix.
 * @ptr:     Opaque WasmHandle pointer.
 * @out_dcc: Output buffer for square distance matrix [K * K].
 * @K:       Requested matrix dimension (clamped to active cluster count).
 *
 * Purpose & Context ("What is this used for?"):
 * Exports pairwise Euclidean distances between cluster centroids to the WebAssembly
 * host environment, enabling heatmap visualizations, dendrograms, and inter-cluster
 * geometry analysis in the browser.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_dcc(
    void   *ptr,
    double *out_dcc,
    int     K)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_dcc || K < 0)
    {
        return;
    }

    int nc = h->state.num_clusters;
    int lim = (K < nc) ? K : nc;
    int stride = h->config.algo.maxnbclust;

    /* Zero the output */
    memset(out_dcc, 0,
           (size_t)K * K * sizeof(double));

    for (int i = 0; i < lim; i++)
    {
        for (int j = 0; j < lim; j++)
        {
            out_dcc[i * K + j] =
                h->state.scratch.dcc_min[
                    i * stride + j];
        }
    }
}

/**
 * wasm_cluster_get_transition_matrix() - Retrieve the Markov transition matrix between clusters.
 * @ptr:    Opaque WasmHandle pointer.
 * @out_tm: Output buffer for square transition count matrix [K * K].
 * @K:      Requested matrix dimension.
 *
 * Purpose & Context ("What is this used for?"):
 * Exports empirical state transition frequencies between consecutive cluster
 * assignments, enabling Markov chain diagram visualizations, temporal transition
 * graph generation, and sequence predictability evaluation in client UIs.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_transition_matrix(
    void *ptr,
    long *out_tm,
    int   K)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_tm)
    {
        return;
    }

    int nc = h->state.num_clusters;
    int lim = (K < nc) ? K : nc;
    int stride = h->config.algo.maxnbclust;

    memset(out_tm, 0,
           (size_t)K * K * sizeof(long));

    for (int i = 0; i < lim; i++)
    {
        for (int j = 0; j < lim; j++)
        {
            out_tm[i * K + j] =
                h->state.transition_matrix[
                    i * stride + j];
        }
    }
}

/**
 * wasm_cluster_get_telemetry() - Query performance metrics and pruning telemetry counters.
 * @ptr:       Opaque WasmHandle pointer.
 * @out_stats: Output array receiving TELEM_* statistics values.
 * @out_len:   Pointer receiving the count of telemetry statistics written.
 *
 * Purpose & Context ("What is this used for?"):
 * Exposes internal algorithmic telemetry (distance calculation counts, pruning
 * efficiency, prediction hit rates, entropy gating rates) to browser dashboards
 * and benchmark runners for real-time monitoring and parameter tuning.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_telemetry(
    void   *ptr,
    double *out_stats,
    int    *out_len)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_stats || !out_len)
    {
        return;
    }

    const ClusterTelemetry *t = &h->state.telemetry;

    out_stats[TELEM_FRAMEDIST_CALLS] =
        (double)t->framedist_calls;
    out_stats[TELEM_FRAMEDIST_SAMPLE] =
        (double)t->framedist_calls_sample;
    out_stats[TELEM_FRAMEDIST_INTERCLUSTER] =
        (double)t->framedist_calls_intercluster;
    out_stats[TELEM_CLUSTERS_PRUNED] =
        (double)t->clusters_pruned;
    out_stats[TELEM_TOTAL_FRAMES] =
        (double)t->total_frames_processed;
    out_stats[TELEM_LAST_FRAME_DISTS] =
        (double)t->last_frame_dists;
    out_stats[TELEM_LAST_FRAME_DFC] =
        (double)t->last_frame_dfc;
    out_stats[TELEM_LAST_FRAME_DCC] =
        (double)t->last_frame_dcc;
    out_stats[TELEM_LAST_ASSIGNMENT_DIST] =
        t->last_assignment_dist;
    out_stats[TELEM_NUM_NEW_CLUSTERS] =
        (double)t->num_new_clusters;
    out_stats[TELEM_PRED_ATTEMPTS] =
        (double)t->pred_attempts;
    out_stats[TELEM_PRED_HITS] =
        (double)t->pred_hits;
    out_stats[TELEM_ENTROPY_GATED] =
        (double)t->entropy_frames_gated;
    out_stats[TELEM_ENTROPY_EVALUATED] =
        (double)t->entropy_frames_evaluated;
    out_stats[TELEM_ENTROPY_SUM_INITIAL] =
        t->entropy_sum_initial;
    out_stats[TELEM_ENTROPY_MAX_INITIAL] =
        t->entropy_max_initial;
    out_stats[TELEM_ENTROPY_LAST_INITIAL] =
        t->entropy_last_initial;
    out_stats[TELEM_DCC_ENTRIES_POPULATED] =
        (double)t->dcc_entries_populated;
    out_stats[TELEM_DCC_PAIRS_TOTAL] =
        (double)t->dcc_pairs_total;

    *out_len = TELEM_COUNT;
}

/**
 * wasm_cluster_get_probs() - Retrieve normalized prior probabilities for active clusters.
 * @ptr:       Opaque WasmHandle pointer.
 * @out_probs: Output double array for cluster probabilities [K].
 * @K:         Requested cluster array dimension.
 *
 * Purpose & Context ("What is this used for?"):
 * Exports the prior assignment probability distribution of active clusters to
 * browser tools for Bayesian probability monitoring and cluster distribution
 * plotting.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_get_probs(
    void   *ptr,
    double *out_probs,
    int     K)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_probs)
    {
        return;
    }

    int nc = h->state.num_clusters;
    int lim = (K < nc) ? K : nc;

    memset(out_probs, 0,
           (size_t)K * sizeof(double));

    for (int i = 0; i < lim; i++)
    {
        out_probs[i] = h->state.clusters[i].prob;
    }
}

/**
 * wasm_cluster_get_evaluations() - Retrieve distance evaluations for the last frame.
 * @ptr:         Opaque WasmHandle pointer.
 * @out_indices: Output array for evaluated cluster indices.
 * @out_dists:   Output array for evaluated Euclidean distances.
 * @max_evals:   Maximum capacity of output arrays.
 *
 * Purpose & Context ("What is this used for?"):
 * Copies the sequence of cluster indices and measured distances that were
 * evaluated against the current frame during the search loop. Used by diagnostic
 * tools and visualizers to inspect candidate pruning and search paths.
 *
 * Return: Number of evaluations recorded for the most recent frame.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_evaluations(
    void   *ptr,
    int    *out_indices,
    double *out_dists,
    int     max_evals)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !out_indices || !out_dists || max_evals <= 0)
    {
        return 0;
    }

    int count = (int)h->state.telemetry.last_frame_dfc;
    if (count > max_evals)
    {
        count = max_evals;
    }
    if (count > h->config.algo.maxnbclust)
    {
        count = h->config.algo.maxnbclust;
    }

    if (count > 0 && h->temp_indices && h->temp_dists)
    {
        memcpy(out_indices, h->temp_indices, (size_t)count * sizeof(int));
        memcpy(out_dists, h->temp_dists, (size_t)count * sizeof(double));
    }
    else
    {
        count = 0;
    }

    return count;
}

/**
 * wasm_cluster_reset() - Reset all clustering state while retaining memory allocations.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Reinitializes cluster assignments, transition matrices, scratch tables, and
 * telemetry counters to allow reprocessing a new dataset or restarting a streaming
 * simulation without incurring reallocation and deallocation overhead.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_reset(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return;
    }

    int N = h->config.algo.maxnbclust;
    long maxnbfr = h->maxnbfr;

    h->current_frame_id = 0;
    h->prev_assigned = -1;
    h->state.num_clusters = 0;

    if (h->state.trace)
    {
        trace_buffer_clear(h->state.trace);
    }

    /* Free per-cluster anchor data */
    for (int i = 0; i < N; i++)
    {
        if (h->state.clusters[i].anchor.data)
        {
            free(h->state.clusters[i].anchor.data);
            h->state.clusters[i].anchor.data = NULL;
        }
    }

    memset(h->state.clusters, 0,
           (size_t)N * sizeof(Cluster));

    /* Free visitor list arrays */
    for (int i = 0; i < N; i++)
    {
        if (h->state.cluster_visitors[i].frames)
        {
            free(
                h->state.cluster_visitors[i].frames
            );
        }
    }
    memset(h->state.cluster_visitors, 0,
           (size_t)N * sizeof(VisitorList));

    /* Reset assignments */
    for (long i = 0; i < maxnbfr; i++)
    {
        h->state.assignments[i] = -1;
    }

    memset(h->state.frame_infos, 0,
           (size_t)maxnbfr * sizeof(FrameInfo));
    memset(h->state.transition_matrix, 0,
           (size_t)N * N * sizeof(long));

    /* Reset scratch */
    {
        ClusterScratch *s = &h->state.scratch;

        memset(s->dcc_min, 0,
               (size_t)N * N * sizeof(double));
        memset(s->dcc_max, 0,
               (size_t)N * N * sizeof(double));
        memset(s->dcc_measured, 0,
               (size_t)N * N * sizeof(char));

        size_t mask_words =
            (size_t)N * N * ((N + 63) / 64);
        memset(s->consistency_mask, 0,
               mask_words * sizeof(uint64_t));

        s->refine_queue_size = 0;
        s->refine_queue_idx = 0;
        s->refine_queue_last_num_clusters = 0;
        s->tuple_pred_count = 0;
    } // Reset scratch

    /* Reset telemetry (preserve array pointers) */
    {
        ClusterTelemetry *t = &h->state.telemetry;
        double *pfsum = t->pruned_fraction_sum;
        long *scnt = t->step_counts;
        long *dcnt = t->dist_counts;
        long *pcnt = t->pruned_counts_by_dist;
        long *qcnt = t->cluster_query_counts;
        int max_steps = t->max_steps_recorded;

        memset(t, 0, sizeof(ClusterTelemetry));

        t->pruned_fraction_sum = pfsum;
        t->step_counts = scnt;
        t->dist_counts = dcnt;
        t->pruned_counts_by_dist = pcnt;
        t->cluster_query_counts = qcnt;
        t->max_steps_recorded = max_steps;

        if (pfsum)
        {
            memset(pfsum, 0,
                   (size_t)N * sizeof(double));
        }
        if (scnt)
        {
            memset(scnt, 0,
                   (size_t)N * sizeof(long));
        }
        if (dcnt)
        {
            memset(dcnt, 0,
                   (size_t)(N + 1) * sizeof(long));
        }
        if (pcnt)
        {
            memset(pcnt, 0,
                   (size_t)(N + 1) * sizeof(long));
        }
        if (qcnt)
        {
            memset(qcnt, 0,
                   (size_t)N * sizeof(long));
        }
    } // Reset telemetry
}

/**
 * wasm_cluster_reassign_nearest() - Reassign all frames to their nearest cluster anchor.
 * @ptr:             Opaque WasmHandle pointer.
 * @coords_flat:     Contiguous array of frame coordinates [num_frames * ndim].
 * @num_frames:      Total number of frames to reassign.
 * @ndim:            Dimensionality of coordinates.
 * @out_assignments: Output array to store refined assignments (may be NULL).
 *
 * Purpose & Context ("What is this used for?"):
 * Implements Pass 2 nearest-centroid reassignment in WebAssembly. Because online
 * incremental clustering assigns earlier frames before later clusters exist,
 * Pass 2 provides an optimal nearest-neighbor partition across all discovered
 * anchors.
 *
 * Return: Total number of frames whose cluster assignment changed during reassignment.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_reassign_nearest(
    void         *ptr,
    const double *coords_flat,
    int           num_frames,
    int           ndim,
    int          *out_assignments)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (h == NULL || h->state.num_clusters <= 0 ||
        num_frames <= 0 || ndim <= 0 || ndim != h->ndim ||
        coords_flat == NULL)
    {
        return 0;
    }

    int K = h->state.num_clusters;
    int reassigned = 0;

    for (int t = 0; t < num_frames; t++)
    {
        const double *p = &coords_flat[(size_t)t * (size_t)ndim];
        int best_k = -1;
        double min_dist_sq = 1e30;

        for (int k = 0; k < K; k++)
        {
            const double *anchor = h->state.clusters[k].anchor.data;
            if (anchor == NULL)
            {
                continue;
            }

            double dist_sq = 0.0;
            for (int d = 0; d < ndim; d++)
            {
                double diff = p[d] - anchor[d];
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                best_k = k;
            }
        }

        if (best_k >= 0)
        {
            if (out_assignments != NULL)
            {
                out_assignments[t] = best_k;
            }
            if (h->state.assignments != NULL && (long)t < h->maxnbfr)
            {
                if (h->state.assignments[t] != best_k)
                {
                    reassigned++;
                }
                h->state.assignments[t] = best_k;
            }
            if (h->state.frame_infos != NULL && (long)t < h->maxnbfr)
            {
                h->state.frame_infos[t].assignment = best_k;
            }
        }
    }

    return reassigned;
}

/**
 * wasm_cluster_free() - Free all resources and memory associated with a WASM cluster instance.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Serves as the destructor for WebAssembly single-tile clustering instances.
 * Frees cluster anchors, visitor lists, scratch buffers, telemetry structures,
 * and the top-level WasmHandle to prevent memory leaks in the WASM heap.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_free(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return;
    }

    int N = h->config.algo.maxnbclust;

    if (h->state.trace)
    {
        trace_buffer_destroy(h->state.trace);
        h->state.trace = NULL;
    }

    /* Free per-cluster anchor data */
    if (h->state.clusters != NULL)
    {
        for (int i = 0; i < N; i++)
        {
            if (h->state.clusters[i].anchor.data)
            {
                free(h->state.clusters[i].anchor.data);
            }
        }
    }

    /* Free visitor list arrays */
    if (h->state.cluster_visitors != NULL)
    {
        for (int i = 0; i < N; i++)
        {
            if (h->state.cluster_visitors[i].frames)
            {
                free(
                    h->state.cluster_visitors[i].frames
                );
            }
        }
    }

    free(h->state.clusters);
    free(h->state.cluster_visitors);
    free(h->state.assignments);
    free(h->state.frame_infos);
    free(h->state.transition_matrix);

    /* Free scratch */
    {
        ClusterScratch *s = &h->state.scratch;
        free(s->mixed_probs);
        free(s->clmembflag);
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
    } // Free scratch

    /* Free telemetry arrays */
    {
        ClusterTelemetry *t = &h->state.telemetry;
        free(t->pruned_fraction_sum);
        free(t->step_counts);
        free(t->dist_counts);
        free(t->pruned_counts_by_dist);
        free(t->cluster_query_counts);
    } // Free telemetry arrays

    free(h->temp_indices);
    free(h->temp_dists);
    free(h->sorting_candidates);

    if (h->frame.data)
    {
        free(h->frame.data);
    }

    free(h);
}

/**
 * wasm_cluster_set_trace() - Enable or disable detailed execution event tracing.
 * @ptr:      Opaque WasmHandle pointer.
 * @enabled:  Non-zero to enable trace ring buffer, 0 to disable.
 * @capacity: Maximum event ring buffer capacity (number of TraceEvent items).
 *
 * Purpose & Context ("What is this used for?"):
 * Controls the algorithmic event tracer in WebAssembly. When enabled, records
 * fine-grained candidate evaluations, distance calculations, and pruning steps
 * into a circular trace buffer for step-by-step playback in the simulator UI.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_set_trace(
    void *ptr,
    int   enabled,
    int   capacity)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return;
    }

    if (enabled && h->state.trace == NULL)
    {
        h->state.trace = trace_buffer_create(capacity);
    }
    else if (!enabled && h->state.trace != NULL)
    {
        trace_buffer_destroy(h->state.trace);
        h->state.trace = NULL;
    }
}

/**
 * wasm_cluster_get_trace_count() - Query the number of events recorded in the trace buffer.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Returns the current fill level of the event trace buffer for UI timeline widgets.
 *
 * Return: Number of recorded trace events, or 0 if tracing is disabled.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_count(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !h->state.trace)
    {
        return 0;
    }
    return h->state.trace->count;
}

/**
 * wasm_cluster_get_trace_events() - Get pointer to trace event ring buffer in WASM memory.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Provides direct WebAssembly linear memory access to the TraceEvent array so JavaScript
 * can construct typed views without copying.
 *
 * Return: Pointer to trace events buffer, or NULL if trace disabled.
 */
EMSCRIPTEN_KEEPALIVE
void *wasm_cluster_get_trace_events(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !h->state.trace)
    {
        return NULL;
    }
    return h->state.trace->events;
}

/**
 * wasm_cluster_get_trace_event_size() - Return size in bytes of a single TraceEvent struct.
 *
 * Purpose & Context ("What is this used for?"):
 * Informs JavaScript client code of the C struct stride so typed views can correctly
 * deserialize event records across different WebAssembly compiler targets.
 *
 * Return: Stride of TraceEvent in bytes.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_event_size(void)
{
    return sizeof(TraceEvent);
}

/**
 * wasm_cluster_get_trace_head() - Return current head position in trace ring buffer.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Supplies the circular buffer write index so JavaScript can display a sliding window
 * of the most recent clustering decisions.
 *
 * Return: Head index within the circular buffer, or 0 if disabled.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_head(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !h->state.trace)
    {
        return 0;
    }
    return h->state.trace->head;
}

/**
 * wasm_cluster_get_trace_frame_start() - Query event buffer index where current frame began.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Demarcates frame boundaries in the trace buffer so UI visualizers can replay all
 * evaluations for the current frame in isolation.
 *
 * Return: Starting index of the current frame in the trace buffer, or 0.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_trace_frame_start(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !h->state.trace)
    {
        return 0;
    }
    return h->state.trace->frame_start;
}

/**
 * wasm_cluster_clear_trace() - Reset the trace buffer without altering clustering state.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Empties recorded trace history when starting a new visualization recording or
 * clearing UI event queues.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_clear_trace(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h || !h->state.trace)
    {
        return;
    }
    trace_buffer_clear(h->state.trace);
}

/**
 * wasm_cluster_set_unlimited() - Toggle unlimited cluster capacity mode.
 * @ptr:       Opaque WasmHandle pointer.
 * @unlimited: 1 to allow dynamic growth beyond initial maxnbclust, 0 to enforce limit.
 *
 * Purpose & Context ("What is this used for?"):
 * Allows browser clients to remove cluster limits dynamically when exploring unknown
 * datasets without restarting the session.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_cluster_set_unlimited(
    void *ptr,
    int   unlimited)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return;
    }

    if (unlimited)
    {
        h->user_maxcl = 0;
    }
    else
    {
        h->user_maxcl = h->config.algo.maxnbclust;
    }
}

/**
 * wasm_cluster_get_capacity() - Query current allocated cluster capacity.
 * @ptr: Opaque WasmHandle pointer.
 *
 * Purpose & Context ("What is this used for?"):
 * Returns the currently allocated cluster buffer size, reflecting any dynamic
 * doublings performed during runtime.
 *
 * Return: Current maximum cluster capacity, or 0 on invalid handle.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_cluster_get_capacity(void *ptr)
{
    WasmHandle *h = (WasmHandle *)ptr;
    if (!h)
    {
        return 0;
    }
    return h->config.algo.maxnbclust;
}

/**
 * wasm_cluster_get_version() - Retrieve version and build metadata string.
 *
 * Purpose & Context ("What is this used for?"):
 * Provides git commit hash and build timestamp to browser clients and diagnostic logs
 * to ensure client-side JS and compiled WASM binary versions match.
 *
 * Return: Null-terminated version string.
 */
EMSCRIPTEN_KEEPALIVE
const char *wasm_cluster_get_version(void)
{
    static char buf[128];
    static int init = 0;
    if (!init)
    {
#if defined(GRIC_BUILD_DATE) && defined(GRIC_GIT_HASH)
        snprintf(buf, sizeof(buf), "%s | %s",
                 GRIC_GIT_HASH, GRIC_BUILD_DATE);
#elif defined(GRIC_GIT_HASH)
        snprintf(buf, sizeof(buf), "%s | %s %s",
                 GRIC_GIT_HASH, __DATE__, __TIME__);
#else
        snprintf(buf, sizeof(buf), "dev | %s %s",
                 __DATE__, __TIME__);
#endif
        init = 1;
    }
    return buf;
}

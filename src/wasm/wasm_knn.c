/**
 * @file wasm_knn.c
 * @brief k-NN search WebAssembly bridge API.
 */

#include "wasm_internal.h"
#include "knn_parser.h"


/* -------------------------------------------------------
 * k-NN WASM API
 * ------------------------------------------------------- */

static inline double wasm_knn_calc_dist(
    const double *a,
    const double *b,
    int           dim)
{
    double sum = 0.0;
    for (int d = 0; d < dim; d++)
    {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * wasm_knn_run_search() - Execute metric-pruned k-NN solver on dataset in WASM.
 * @handle:           Active WasmHandle containing cluster state.
 * @dataset_points:   Contiguous array of frame coordinates [total_frames * ndim].
 * @total_frames:     Total number of dataset frames (N).
 * @ndim:             Dimensionality of coordinate vectors.
 * @k:                Requested nearest neighbors count.
 * @min_temporal_sep: Minimum temporal frame separation (|i - j| >= dtmin).
 * @past_only:        1 to search only preceding frames (j < i).
 * @future_only:      1 to search only subsequent frames (j > i).
 * @epsilon:          Slack factor for (1+eps)-ANN pruning.
 * @rlim_cutoff:      Optional maximum distance cutoff (0.0 = disabled).
 * @out_indices:      Output array for neighbor frame IDs [N * k].
 * @out_distances:    Output array for neighbor distances [N * k].
 * @out_telemetry:    Output array for telemetry statistics [8 doubles].
 *
 * Return: 0 on success, -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_knn_run_search(
    void         *handle,
    const double *dataset_points,
    long          total_frames,
    int           ndim,
    int           k,
    int           min_temporal_sep,
    int           past_only,
    int           future_only,
    double        epsilon,
    double        rlim_cutoff,
    int           use_multi_pivot,
    int          *out_indices,
    double       *out_distances,
    double       *out_telemetry)
{
    WasmHandle *h = (WasmHandle *)handle;
    if (h == NULL || dataset_points == NULL || out_indices == NULL ||
        out_distances == NULL || out_telemetry == NULL || total_frames <= 0 ||
        ndim <= 0 || k <= 0 || ndim != h->ndim)
    {
        return -1;
    }

    int M = h->state.num_clusters;
    if (M <= 0)
    {
        return -1;
    }

    /* 1. Build resident KnnModel from active cluster state */
    KnnModel model;
    memset(&model, 0, sizeof(KnnModel));
    model.frame_width = ndim;
    model.frame_height = 1;
    model.frame_elements = ndim;
    model.num_clusters = M;
    model.total_dataset_frames = total_frames;
    model.is_fits_input = 0;
    model.is_double = 1;

    model.clusters = (KnnCluster *)calloc((size_t)M, sizeof(KnnCluster));
    model.frame_cluster_map = (int *)malloc((size_t)total_frames * sizeof(int));
    model.frame_r_anchor = (float *)malloc((size_t)total_frames * sizeof(float));
    model.dcc_matrix = (double *)malloc((size_t)M * (size_t)M * sizeof(double));

    if (model.clusters == NULL || model.frame_cluster_map == NULL ||
        model.frame_r_anchor == NULL || model.dcc_matrix == NULL)
    {
        goto cleanup_model_alloc;
    }

    /* Initialize cluster anchors and capacities */
    for (int c = 0; c < M; c++)
    {
        model.clusters[c].cluster_id = c;
        model.clusters[c].anchor_data = (double *)malloc((size_t)ndim * sizeof(double));
        if (model.clusters[c].anchor_data == NULL)
        {
            goto cleanup_model_alloc;
        }

        if (h->state.clusters[c].anchor.data != NULL)
        {
            memcpy(model.clusters[c].anchor_data,
                   h->state.clusters[c].anchor.data,
                   (size_t)ndim * sizeof(double));
        }
        else
        {
            memset(model.clusters[c].anchor_data, 0, (size_t)ndim * sizeof(double));
        }

        int est_members = (h->state.cluster_visitors != NULL) ?
                          h->state.cluster_visitors[c].count : 16;
        model.clusters[c].capacity = (est_members > 16) ? est_members + 16 : 16;
        model.clusters[c].num_members = 0;
        model.clusters[c].radius = 0.0;
        model.clusters[c].members =
            (MemberMeta *)malloc((size_t)model.clusters[c].capacity * sizeof(MemberMeta));
        if (model.clusters[c].members == NULL)
        {
            goto cleanup_model_alloc;
        }
    } // for (int c = 0; c < M; c++)

    /* Compute exact pairwise inter-cluster anchor distances */
    for (int i = 0; i < M; i++)
    {
        model.dcc_matrix[i * M + i] = 0.0;
        for (int j = i + 1; j < M; j++)
        {
            double d = wasm_knn_calc_dist(
                model.clusters[i].anchor_data,
                model.clusters[j].anchor_data,
                ndim);
            model.dcc_matrix[i * M + j] = d;
            model.dcc_matrix[j * M + i] = d;
        }
    }

    /* Assign each dataset frame to its home cluster and compute anchor distances */
    for (long i = 0; i < total_frames; i++)
    {
        const double *f_coords = &dataset_points[i * ndim];
        int home_c = -1;

        if (h->state.assignments != NULL &&
            i < (long)h->current_frame_id && i < h->maxnbfr)
        {
            home_c = h->state.assignments[i];
        }

        /* Fallback: find nearest cluster if assignment is missing or out of range */
        if (home_c < 0 || home_c >= M)
        {
            double min_d = 1e30;
            int best_c = 0;
            for (int c = 0; c < M; c++)
            {
                double d = wasm_knn_calc_dist(f_coords, model.clusters[c].anchor_data, ndim);
                if (d < min_d)
                {
                    min_d = d;
                    best_c = c;
                }
            }
            home_c = best_c;
        }

        double r_anchor = wasm_knn_calc_dist(f_coords, model.clusters[home_c].anchor_data, ndim);
        model.frame_cluster_map[i] = home_c;
        model.frame_r_anchor[i] = (float)r_anchor;

        KnnCluster *cl = &model.clusters[home_c];
        if (cl->num_members >= cl->capacity)
        {
            cl->capacity *= 2;
            MemberMeta *new_mem = (MemberMeta *)realloc(
                cl->members, (size_t)cl->capacity * sizeof(MemberMeta));
            if (new_mem == NULL)
            {
                goto cleanup_model_alloc;
            }
            cl->members = new_mem;
        }

        cl->members[cl->num_members].frame_id = (uint32_t)i;
        cl->members[cl->num_members].r_anchor = (float)r_anchor;
        cl->num_members++;

        if (r_anchor > cl->radius)
        {
            cl->radius = r_anchor;
        }
    } // for (long i = 0; i < total_frames; i++)

    /* Sort each cluster's members array by ascending r_anchor for O(log N) binary search */
    for (int c = 0; c < M; c++)
    {
        if (model.clusters[c].num_members > 1)
        {
            qsort(model.clusters[c].members,
                  (size_t)model.clusters[c].num_members,
                  sizeof(MemberMeta),
                  knn_compare_member_meta_radii);
        }
    }

    model.anchor_ptrs = (const void **)malloc((size_t)M * sizeof(const void *));
    model.cluster_radii = (double *)malloc((size_t)M * sizeof(double));
    if (model.anchor_ptrs == NULL || model.cluster_radii == NULL)
    {
        goto cleanup_model_alloc;
    }
    for (int c = 0; c < M; c++)
    {
        model.anchor_ptrs[c] = model.clusters[c].anchor_data;
        model.cluster_radii[c] = model.clusters[c].radius;
    }

    /* 2. Configure KnnConfig for in-memory execution */
    KnnConfig config;
    memset(&config, 0, sizeof(KnnConfig));
    config.k = k;
    config.min_temporal_sep = min_temporal_sep;
    config.past_only = past_only;
    config.future_only = future_only;
    config.epsilon = epsilon;
    config.rlim_cutoff = rlim_cutoff;
    config.memory_data = dataset_points;
    config.use_multi_pivot = use_multi_pivot;
    config.use_reciprocal = (!past_only && !future_only) ? 1 : 0;
    config.use_double = 1;
    config.use_batch_dist = 1;
    config.nthreads = 1;
    config.progress_mode = 0;
    config.verbose_level = 0;

    /* 3. Run the exact C Knn search engine */
    KnnResults results;
    memset(&results, 0, sizeof(KnnResults));
    KnnTelemetry telemetry;
    memset(&telemetry, 0, sizeof(KnnTelemetry));

    int status = knn_run_search(&config, &model, &results, &telemetry);
    if (status != 0)
    {
        knn_results_free(&results);
        goto cleanup_model_alloc;
    }

    /* 4. Copy results back into exported WASM memory buffers */
    if (results.indices != NULL)
    {
        memcpy(out_indices, results.indices, (size_t)total_frames * (size_t)k * sizeof(int));
    }
    if (results.distances != NULL)
    {
        memcpy(out_distances, results.distances,
               (size_t)total_frames * (size_t)k * sizeof(double));
    }

    out_telemetry[0] = (double)telemetry.total_queries;
    out_telemetry[1] = (double)telemetry.framedist_calls;
    out_telemetry[2] = (double)telemetry.level1_clusters_pruned;
    out_telemetry[3] = (double)telemetry.level2_anchors_pruned;
    out_telemetry[4] = (double)telemetry.level3_annular_pruned;
    out_telemetry[5] = (double)telemetry.temporal_pruned;
    out_telemetry[6] = (double)telemetry.total_candidates_considered;
    out_telemetry[7] = telemetry.time_search_ms;

    knn_results_free(&results);

    /* Free model buffers */
    for (int c = 0; c < M; c++)
    {
        if (model.clusters[c].anchor_data != NULL)
        {
            free(model.clusters[c].anchor_data);
        }
        if (model.clusters[c].members != NULL)
        {
            free(model.clusters[c].members);
        }
    }
    free(model.clusters);
    free(model.frame_cluster_map);
    free(model.frame_r_anchor);
    free(model.dcc_matrix);
    if (model.anchor_ptrs != NULL)
    {
        free((void *)model.anchor_ptrs);
    }
    if (model.cluster_radii != NULL)
    {
        free(model.cluster_radii);
    }

    return 0;

cleanup_model_alloc:
    if (model.clusters != NULL)
    {
        for (int c = 0; c < M; c++)
        {
            if (model.clusters[c].anchor_data != NULL)
            {
                free(model.clusters[c].anchor_data);
            }
            if (model.clusters[c].members != NULL)
            {
                free(model.clusters[c].members);
            }
        }
        free(model.clusters);
    }
    if (model.frame_cluster_map != NULL)
    {
        free(model.frame_cluster_map);
    }
    if (model.frame_r_anchor != NULL)
    {
        free(model.frame_r_anchor);
    }
    if (model.dcc_matrix != NULL)
    {
        free(model.dcc_matrix);
    }
    if (model.anchor_ptrs != NULL)
    {
        free((void *)model.anchor_ptrs);
    }
    if (model.cluster_radii != NULL)
    {
        free(model.cluster_radii);
    }
    return -1;
}


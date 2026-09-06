/**
 * @file probe_engine.c
 * @brief Analysis engine for dataset geometry, variance, and parameter tuning.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#include "probe_engine.h"
#include "gric_bin_io.h"
#include "gric_bin_header.h"
#include "scalar_quant.h"
#include "cluster_locator.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#endif

/** Context for sorting dimension indices by descending variance */
static const double *s_dim_vars = NULL;

static int compare_dim_indices(
    const void *a,
    const void *b)
{
    long ia = *(const long *)a;
    long ib = *(const long *)b;
    double va = s_dim_vars[ia];
    double vb = s_dim_vars[ib];

    if (va > vb)
    {
        return -1;
    }
    if (va < vb)
    {
        return 1;
    }
    return (ia < ib) ? -1 : 1;
}

static int compare_doubles(
    const void *a,
    const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
    {
        return -1;
    }
    if (da > db)
    {
        return 1;
    }
    return 0;
}

/**
 * load_ascii_data() - Ingest ASCII vector coordinates.
 */
static float *load_ascii_data(
    const char *filepath,
    long        max_samples,
    long       *out_frames,
    long       *out_dim)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        return NULL;
    }

    /* Step 1: determine dimension D from first non-comment line */
    char line[65536];
    long dim = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }

        char *endptr = NULL;
        while (*p)
        {
            strtod(p, &endptr);
            if (endptr == p) break;
            dim++;
            p = endptr;
            while (*p == ' ' || *p == '\t' || *p == ',') p++;
        }
        break;
    }

    if (dim <= 0)
    {
        fclose(fp);
        return NULL;
    }

    /* Step 2: count total lines */
    fseek(fp, 0, SEEK_SET);
    long total_lines = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }
        total_lines++;
        if (max_samples > 0 && total_lines >= max_samples)
        {
            break;
        }
    }

    if (total_lines <= 0)
    {
        fclose(fp);
        return NULL;
    }

    float *buffer = (float *)malloc((size_t)(total_lines * dim) * sizeof(float));
    if (buffer == NULL)
    {
        fclose(fp);
        return NULL;
    }

    /* Step 3: read coordinates */
    fseek(fp, 0, SEEK_SET);
    long frame_idx = 0;
    while (fgets(line, sizeof(line), fp) && frame_idx < total_lines)
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
        {
            continue;
        }

        float *row = buffer + frame_idx * dim;
        char *endptr = NULL;
        for (long d = 0; d < dim; d++)
        {
            row[d] = (float)strtod(p, &endptr);
            if (endptr == p)
            {
                row[d] = 0.0f;
            }
            p = endptr;
            while (*p == ' ' || *p == '\t' || *p == ',') p++;
        }
        frame_idx++;
    }

    fclose(fp);
    *out_frames = frame_idx;
    *out_dim = dim;
    return buffer;
}

/**
 * load_bin_data() - Ingest self-describing GRIC binary datasets.
 */
static float *load_bin_data(
    const char *filepath,
    long        max_samples,
    long       *out_frames,
    long       *out_width,
    long       *out_height,
    long       *out_dim,
    int        *out_is_double)
{
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    gric_bin_header_t hdr;
    char *comment = NULL;
    if (gric_bin_read_header(fp, &hdr, &comment) != 0)
    {
        fclose(fp);
        return NULL;
    }
    if (comment != NULL)
    {
        free(comment);
    }

    long width = 1;
    long height = 1;
    long total_frames = 1;

    if (hdr.ndim == 1)
    {
        total_frames = (long)hdr.dims[0];
        width = 1;
        height = 1;
    }
    else if (hdr.ndim == 2)
    {
        total_frames = (long)hdr.dims[0];
        width = (long)hdr.dims[1];
        height = 1;
    }
    else if (hdr.ndim == 3)
    {
        width = (long)hdr.dims[0];
        height = (long)hdr.dims[1];
        total_frames = (long)hdr.dims[2];
    }
    else if (hdr.ndim == 4)
    {
        width = (long)hdr.dims[0];
        height = (long)hdr.dims[1];
        total_frames = (long)hdr.dims[3];
    }

    long dim = width * height;
    long n_load = total_frames;
    if (max_samples > 0 && max_samples < n_load)
    {
        n_load = max_samples;
    }

    float *buffer = (float *)malloc((size_t)(n_load * dim) * sizeof(float));
    if (buffer == NULL)
    {
        fclose(fp);
        return NULL;
    }

    size_t elem_size = gric_bin_data_type_size(hdr.data_type);
    if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT32)
    {
        size_t read_n = fread(buffer, sizeof(float), (size_t)(n_load * dim), fp);
        (void)read_n;
    }
    else if (hdr.data_type == GRIC_BIN_DTYPE_FLOAT64)
    {
        *out_is_double = 1;
        double *dbuf = (double *)malloc((size_t)(n_load * dim) * sizeof(double));
        if (dbuf != NULL)
        {
            size_t read_n = fread(dbuf, sizeof(double), (size_t)(n_load * dim), fp);
            (void)read_n;
            for (long ii = 0; ii < n_load * dim; ii++)
            {
                buffer[ii] = (float)dbuf[ii];
            }
            free(dbuf);
        }
    }
    else
    {
        /* Generic byte conversion */
        uint8_t *raw = (uint8_t *)malloc(elem_size * (size_t)(n_load * dim));
        if (raw != NULL)
        {
            size_t read_n = fread(raw, elem_size, (size_t)(n_load * dim), fp);
            (void)read_n;
            for (long ii = 0; ii < n_load * dim; ii++)
            {
                buffer[ii] = (float)raw[ii];
            }
            free(raw);
        }
    }

    fclose(fp);
    *out_frames = n_load;
    *out_width = width;
    *out_height = height;
    *out_dim = dim;
    return buffer;
}

#ifdef USE_CFITSIO
/**
 * load_fits_data() - Ingest FITS 2D tables or 3D cubes.
 */
static float *load_fits_data(
    const char *filepath,
    long        max_samples,
    long       *out_frames,
    long       *out_width,
    long       *out_height,
    long       *out_dim,
    int        *out_is_double)
{
    fitsfile *fptr = NULL;
    int status = 0;

    if (fits_open_file(&fptr, filepath, READONLY, &status))
    {
        return NULL;
    }

    int bitpix = 0;
    int naxis = 0;
    long naxes[4] = {0, 0, 0, 0};

    if (fits_get_img_param(fptr, 4, &bitpix, &naxis, naxes, &status))
    {
        fits_close_file(fptr, &status);
        return NULL;
    }

    if (bitpix == DOUBLE_IMG)
    {
        *out_is_double = 1;
    }

    long width = 1;
    long height = 1;
    long total_frames = 1;

    if (naxis == 1)
    {
        total_frames = naxes[0];
        width = 1;
        height = 1;
    }
    else if (naxis == 2)
    {
        /* Either N x D (vector table) or W x H (single image) */
        width = naxes[0];
        total_frames = naxes[1];
        height = 1;
    }
    else if (naxis >= 3)
    {
        width = naxes[0];
        height = naxes[1];
        total_frames = naxes[2];
    }

    long dim = width * height;
    long n_load = total_frames;
    if (max_samples > 0 && max_samples < n_load)
    {
        n_load = max_samples;
    }

    float *buffer = (float *)malloc((size_t)(n_load * dim) * sizeof(float));
    if (buffer == NULL)
    {
        fits_close_file(fptr, &status);
        return NULL;
    }

    long fpixel[4] = {1, 1, 1, 1};
    if (fits_read_pix(fptr, TFLOAT, fpixel, n_load * dim, NULL, buffer, NULL, &status))
    {
        free(buffer);
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_close_file(fptr, &status);
    *out_frames = n_load;
    *out_width = width;
    *out_height = height;
    *out_dim = dim;
    return buffer;
}
#endif

/**
 * probe_profile_pruning() - Empirically evaluate TE3, TE4, and TE5 pruning.
 * @data:       Dataset samples array [num_frames * dim].
 * @num_frames: Total frame count.
 * @dim:        Dimension count.
 * @rlim:       Calibrated cluster radius.
 * @profile:    Pointer to GricProfile to update.
 */
static void probe_profile_pruning(
    const float *restrict data,
    long                  num_frames,
    long                  dim,
    double                rlim,
    GricProfile          *profile)
{
    if (data == NULL || profile == NULL || num_frames < 6 || dim < 1 || rlim <= 0.0)
    {
        profile->te3_prune_rate = 0.0;
        profile->te4_marginal_rate = 0.0;
        profile->te5_marginal_rate = 0.0;
        profile->te4_enabled = 0;
        profile->te5_enabled = 0;
        strcpy(profile->recommended_prune_mode, "3P");
        return;
    }

    /* Choose sample cluster anchor count K (max 32) */
    int k_anchors = (int)(num_frames / 3);
    if (k_anchors > 32)
    {
        k_anchors = 32;
    }
    if (k_anchors < 4)
    {
        k_anchors = (num_frames > 4) ? 4 : (int)num_frames;
    }

    long *anchor_indices = (long *)malloc((size_t)k_anchors * sizeof(long));
    double *dcc = (double *)malloc((size_t)(k_anchors * k_anchors) * sizeof(double));

    if (anchor_indices == NULL || dcc == NULL)
    {
        free(anchor_indices);
        free(dcc);
        profile->te4_enabled = 0;
        profile->te5_enabled = 0;
        strcpy(profile->recommended_prune_mode, "3P");
        return;
    }

    /* Evenly spaced anchors across dataset */
    for (int k = 0; k < k_anchors; k++)
    {
        double frac = (k_anchors > 1) ? (double)k / (double)(k_anchors - 1) : 0.0;
        anchor_indices[k] = (long)(frac * (double)(num_frames - 1));
    }

    /* Precompute pairwise inter-anchor distances */
    for (int i = 0; i < k_anchors; i++)
    {
        const float *ai = data + anchor_indices[i] * dim;
        dcc[i * k_anchors + i] = 0.0;
        for (int j = i + 1; j < k_anchors; j++)
        {
            const float *aj = data + anchor_indices[j] * dim;
            double sum = 0.0;
            for (long d = 0; d < dim; d++)
            {
                double diff = (double)ai[d] - (double)aj[d];
                sum += diff * diff;
            }
            double dist = sqrt(sum);
            dcc[i * k_anchors + j] = dist;
            dcc[j * k_anchors + i] = dist;
        }
    }

    /* Choose sample query frames (max 64) */
    int m_queries = (int)num_frames;
    if (m_queries > 64)
    {
        m_queries = 64;
    }

    long total_candidates = 0;
    long te3_pruned = 0;
    long te4_marginal_pruned = 0;
    long te5_marginal_pruned = 0;
    long greedy_evals = 0;
    long entropy_evals = 0;

    double *dfc = (double *)malloc((size_t)k_anchors * sizeof(double));
    int *unpruned = (int *)malloc((size_t)k_anchors * sizeof(int));
    int *order_greedy = (int *)malloc((size_t)k_anchors * sizeof(int));
    int *order_entropy = (int *)malloc((size_t)k_anchors * sizeof(int));
    double *cand_lb = (double *)malloc((size_t)k_anchors * sizeof(double));

    if (dfc == NULL || unpruned == NULL || order_greedy == NULL ||
        order_entropy == NULL || cand_lb == NULL)
    {
        free(anchor_indices);
        free(dcc);
        free(dfc);
        free(unpruned);
        free(order_greedy);
        free(order_entropy);
        free(cand_lb);
        profile->te4_enabled = 0;
        profile->te5_enabled = 0;
        profile->entropy_enabled = 0;
        strcpy(profile->recommended_prune_mode, "3P");
        return;
    }

    for (int m = 0; m < m_queries; m++)
    {
        double frac = (m_queries > 1) ? (double)m / (double)(m_queries - 1) : 0.0;
        long q_idx = (long)(frac * (double)(num_frames - 1));
        const float *fq = data + q_idx * dim;

        /* Measure distance from query to all anchors */
        for (int k = 0; k < k_anchors; k++)
        {
            const float *ak = data + anchor_indices[k] * dim;
            double sum = 0.0;
            for (long d = 0; d < dim; d++)
            {
                double diff = (double)fq[d] - (double)ak[d];
                sum += diff * diff;
            }
            dfc[k] = sqrt(sum);
        }

        /* Pick first anchor as closest */
        int a1 = 0;
        double min_dfc = dfc[0];
        for (int k = 1; k < k_anchors; k++)
        {
            if (dfc[k] < min_dfc)
            {
                min_dfc = dfc[k];
                a1 = k;
            }
        }

        /* Evaluate TE3 pruning relative to anchor a1 */
        int unpruned_count = 0;
        for (int k = 0; k < k_anchors; k++)
        {
            if (k == a1)
            {
                continue;
            }
            total_candidates++;
            double te3_bound = fabs(dfc[a1] - dcc[a1 * k_anchors + k]);
            if (te3_bound > rlim)
            {
                te3_pruned++;
            }
            else
            {
                unpruned[unpruned_count++] = k;
            }
        }

        /* If unpruned candidates remain and we have a second anchor, evaluate TE4 */
        int remaining_count = unpruned_count;
        if (unpruned_count > 1)
        {
            int a2 = unpruned[0];
            remaining_count = 0;

            for (int u = 1; u < unpruned_count; u++)
            {
                int k = unpruned[u];
                /* Check if anchor a2 also prunes k via TE3 */
                double te3_bound2 = fabs(dfc[a2] - dcc[a2 * k_anchors + k]);
                if (te3_bound2 > rlim)
                {
                    continue;
                }

                /* Check 4-point pruning (TE4) */
                double min_d4 = calc_min_dist_4pt(
                    dfc[a1], dfc[a2], dcc[a1 * k_anchors + a2],
                    dcc[a1 * k_anchors + k], dcc[a2 * k_anchors + k]);
                if (min_d4 > rlim)
                {
                    te4_marginal_pruned++;
                }
                else
                {
                    unpruned[remaining_count++] = k;
                }
            }

            /* If unpruned candidates remain and dim >= 3, evaluate TE5 */
            if (remaining_count > 1 && dim >= 3)
            {
                int a3 = unpruned[0];
                int final_count = 0;
                for (int u = 1; u < remaining_count; u++)
                {
                    int k = unpruned[u];
                    /* Check TE3 and TE4 with a3 */
                    double te3_bound3 = fabs(dfc[a3] - dcc[a3 * k_anchors + k]);
                    if (te3_bound3 > rlim)
                    {
                        continue;
                    }
                    double min_d4_3 = calc_min_dist_4pt(
                        dfc[a1], dfc[a3], dcc[a1 * k_anchors + a3],
                        dcc[a1 * k_anchors + k], dcc[a3 * k_anchors + k]);
                    if (min_d4_3 > rlim)
                    {
                        continue;
                    }

                    /* Check 5-point pruning (TE5) */
                    double min_d5 = calc_min_dist_5pt(
                        dfc[a1], dfc[a2], dfc[a3],
                        dcc[k * k_anchors + a1], dcc[k * k_anchors + a2],
                        dcc[k * k_anchors + a3],
                        dcc[a1 * k_anchors + a2], dcc[a1 * k_anchors + a3],
                        dcc[a2 * k_anchors + a3]);
                    if (min_d5 > rlim)
                    {
                        te5_marginal_pruned++;
                    }
                    else
                    {
                        unpruned[final_count++] = k;
                    }
                }
                remaining_count = final_count;
            }
        }

        /* Simulate greedy vs entropy candidate ordering for remaining candidates */
        if (remaining_count >= 3 && dim >= 8)
        {
            for (int u = 0; u < remaining_count; u++)
            {
                int k = unpruned[u];
                cand_lb[u] = fabs(dfc[a1] - dcc[a1 * k_anchors + k]);
                order_greedy[u] = u;
                order_entropy[u] = u;
            }

            /* Sort order_greedy by cand_lb ascending */
            for (int i = 0; i < remaining_count - 1; i++)
            {
                for (int j = i + 1; j < remaining_count; j++)
                {
                    if (cand_lb[order_greedy[j]] < cand_lb[order_greedy[i]])
                    {
                        int tmp = order_greedy[i];
                        order_greedy[i] = order_greedy[j];
                        order_greedy[j] = tmp;
                    }
                }
            }

            for (int u = 0; u < remaining_count; u++)
            {
                greedy_evals++;
                int k = unpruned[order_greedy[u]];
                if (dfc[k] <= rlim)
                {
                    break;
                }
            }

            /* Entropy ordering: select candidate closest to median lower bound */
            double med_lb = cand_lb[order_greedy[remaining_count / 2]];
            for (int i = 0; i < remaining_count - 1; i++)
            {
                for (int j = i + 1; j < remaining_count; j++)
                {
                    double diff_i = fabs(cand_lb[order_entropy[i]] - med_lb);
                    double diff_j = fabs(cand_lb[order_entropy[j]] - med_lb);
                    if (diff_j < diff_i)
                    {
                        int tmp = order_entropy[i];
                        order_entropy[i] = order_entropy[j];
                        order_entropy[j] = tmp;
                    }
                }
            }

            for (int u = 0; u < remaining_count; u++)
            {
                entropy_evals++;
                int k = unpruned[order_entropy[u]];
                if (dfc[k] <= rlim)
                {
                    break;
                }
            }
        }
    } // for (int m = 0; m < m_queries; m++)

    double te3_rate = (total_candidates > 0) ?
        ((double)te3_pruned / (double)total_candidates) : 0.0;
    double te4_marg = (total_candidates > 0) ?
        ((double)te4_marginal_pruned / (double)total_candidates) : 0.0;
    double te5_marg = (total_candidates > 0) ?
        ((double)te5_marginal_pruned / (double)total_candidates) : 0.0;

    profile->te3_prune_rate = te3_rate;
    profile->te4_marginal_rate = te4_marg;
    profile->te5_marginal_rate = te5_marg;

    /* Cost-benefit FLOP trade-off:
     * Distance calculation cost: ~ 2 * dim FLOPs
     * TE4 evaluation cost: ~ 40 FLOPs
     * TE5 evaluation cost: ~ 120 FLOPs
     */
    double cost_dist = 2.0 * (double)dim;
    double benefit_te4 = te4_marg * cost_dist - 40.0;
    double benefit_te5 = te5_marg * cost_dist - 120.0;

    if (dim >= 8 && benefit_te5 > 0.0 && te5_marg >= 0.03)
    {
        profile->te4_enabled = 1;
        profile->te5_enabled = 1;
        strcpy(profile->recommended_prune_mode, "5P");
    }
    else if (dim >= 4 && benefit_te4 > 0.0 && te4_marg >= 0.03)
    {
        profile->te4_enabled = 1;
        profile->te5_enabled = 0;
        strcpy(profile->recommended_prune_mode, "4P");
    }
    else
    {
        profile->te4_enabled = 0;
        profile->te5_enabled = 0;
        strcpy(profile->recommended_prune_mode, "3P");
    }

    if (dim >= 8 && greedy_evals > 20 &&
        (double)entropy_evals < (double)greedy_evals * 0.95)
    {
        profile->entropy_enabled = 1;
        profile->entropy_gate = 0.20;
    }
    else
    {
        profile->entropy_enabled = 0;
        profile->entropy_gate = 0.20;
    }

    free(anchor_indices);
    free(dcc);
    free(dfc);
    free(unpruned);
    free(order_greedy);
    free(order_entropy);
    free(cand_lb);
}

/**
 * probe_run() - Execute the complete dataset profiling pipeline.
 * @config:  Input probe configuration.
 * @results: Output results container.
 *
 * Return: 0 on success, -1 on failure.
 */
int probe_run(
    const ProbeConfig *config,
    ProbeResults      *results)
{
    if (config == NULL || results == NULL)
    {
        return -1;
    }

    memset(results, 0, sizeof(ProbeResults));

    long max_samples = 0;
    if (config->force_all)
    {
        max_samples = 0;
    }
    else if (config->sample_limit > 0)
    {
        max_samples = config->sample_limit;
    }
    else
    {
        /* Adaptive default: first 5,000 frames if larger */
        max_samples = 5000;
    }

    float *data = NULL;
    long num_frames = 0;
    long width = 1;
    long height = 1;
    long dim = 0;
    int is_double = config->use_double;

    const char *ext = strrchr(config->dataset_path, '.');

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 10%%] Ingesting dataset frames\n");
        fflush(stderr);
    }

    if (ext != NULL && (strcasecmp(ext, ".bin") == 0 || strcasecmp(ext, ".clusterdat") == 0))
    {
        data = load_bin_data(config->dataset_path, max_samples, &num_frames,
                             &width, &height, &dim, &is_double);
    }
#ifdef USE_CFITSIO
    else if (ext != NULL && (strcasecmp(ext, ".fits") == 0 || strcasecmp(ext, ".fits.gz") == 0))
    {
        data = load_fits_data(config->dataset_path, max_samples, &num_frames,
                              &width, &height, &dim, &is_double);
    }
#endif
    else
    {
        data = load_ascii_data(config->dataset_path, max_samples, &num_frames, &dim);
        width = dim;
        height = 1;
    }

    if (data == NULL || num_frames <= 0 || dim <= 0)
    {
        return -1;
    }

    /* Initialize profile container */
    gric_profile_init(&results->profile, dim);
    snprintf(results->profile.dataset_path, sizeof(results->profile.dataset_path),
             "%s", config->dataset_path);
    results->profile.num_frames = num_frames;
    results->profile.width = width;
    results->profile.height = height;
    results->profile.dim = dim;
    results->profile.is_image = (height > 1) ? 1 : 0;
    results->profile.is_double = is_double;

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 30%%] Computing coordinate statistics and ranges\n");
        fflush(stderr);
    }

    /* Step 1: Single-pass Welford coordinate statistics and ranges */
    double global_min = 1e30;
    double global_max = -1e30;
    double *mins = (double *)malloc((size_t)dim * sizeof(double));
    double *maxs = (double *)malloc((size_t)dim * sizeof(double));

    int dead_count = 0;

    for (long d = 0; d < dim; d++)
    {
        double mean = 0.0;
        double m2 = 0.0;
        double min_val = 1e30;
        double max_val = -1e30;

        for (long f = 0; f < num_frames; f++)
        {
            double val = (double)data[f * dim + d];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;

            double delta = val - mean;
            mean += delta / (double)(f + 1);
            m2 += delta * (val - mean);
        }

        mins[d] = min_val;
        maxs[d] = max_val;
        if (min_val < global_min) global_min = min_val;
        if (max_val > global_max) global_max = max_val;

        double var = (num_frames > 1) ? (m2 / (double)(num_frames - 1)) : 0.0;
        results->profile.var_dim[d] = var;
        if (var < 1e-12)
        {
            dead_count++;
        }
    } // for (long d = 0; d < dim; d++)

    results->dead_dims_count = dead_count;

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 55%%] Spectral variance analysis and SQ8 calibration\n");
        fflush(stderr);
    }

    /* Step 2: Spectral variance sorting */
    for (long d = 0; d < dim; d++)
    {
        results->profile.perm_dim[d] = d;
    }
    s_dim_vars = results->profile.var_dim;
    qsort(results->profile.perm_dim, (size_t)dim, sizeof(long), compare_dim_indices);

    /* Step 3: Compute residual tail envelopes Omega_m */
    double accum_tail_sq = 0.0;
    for (long ii = dim - 1; ii >= 0; ii--)
    {
        long d = results->profile.perm_dim[ii];
        double range = maxs[d] - mins[d];
        accum_tail_sq += range * range;
        results->profile.residual_tail[ii] = sqrt(accum_tail_sq);
    } // for (long ii = dim - 1; ii >= 0; ii--)

    /* Step 4: Calibrate SQ8 quantization */
    results->profile.use_sq8 = (dim >= 32) ? 1 : 0;
    sq8_init_params(&results->profile.sq8_params, (float)global_min, (float)global_max, dim);

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 75%%] Sampling pairwise distance spectrum\n");
        fflush(stderr);
    }

    /* Step 5: Subsampled pairwise distance spectrum */
    long s_pairs = (num_frames < 200) ? (num_frames * (num_frames - 1) / 2) : 10000;
    if (s_pairs > 10000) s_pairs = 10000;

    double *dist_samples = (double *)malloc((size_t)s_pairs * sizeof(double));
    if (dist_samples != NULL)
    {
        unsigned int seed = 42;
        for (long s = 0; s < s_pairs; s++)
        {
            long i = rand_r(&seed) % num_frames;
            long j = rand_r(&seed) % num_frames;
            while (j == i && num_frames > 1)
            {
                j = rand_r(&seed) % num_frames;
            }

            const float *fa = data + i * dim;
            const float *fb = data + j * dim;
            double sum = 0.0;
            for (long d = 0; d < dim; d++)
            {
                double diff = (double)fa[d] - (double)fb[d];
                sum += diff * diff;
            }
            dist_samples[s] = sqrt(sum);
        }

        qsort(dist_samples, (size_t)s_pairs, sizeof(double), compare_doubles);

        results->profile.dist_min = dist_samples[0];
        results->profile.dist_p01 = dist_samples[(long)(s_pairs * 0.01)];
        results->profile.dist_p05 = dist_samples[(long)(s_pairs * 0.05)];
        results->profile.dist_p10 = dist_samples[(long)(s_pairs * 0.10)];
        results->profile.dist_p25 = dist_samples[(long)(s_pairs * 0.25)];
        results->profile.dist_p50 = dist_samples[(long)(s_pairs * 0.50)];
        results->profile.dist_p75 = dist_samples[(long)(s_pairs * 0.75)];
        results->profile.dist_p90 = dist_samples[(long)(s_pairs * 0.90)];
        results->profile.dist_max = dist_samples[s_pairs - 1];

        free(dist_samples);
    }

    /* Step 6: Radius presets */
    results->profile.rlim_fine = results->profile.dist_p05;
    results->profile.rlim_balanced = results->profile.dist_p10;
    results->profile.rlim_coarse = results->profile.dist_p25;
    results->profile.rlim_recommended = results->profile.rlim_balanced;
    strcpy(results->profile.preset_name, "balanced");

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 90%%] Evaluating temporal continuity and presets\n");
        fflush(stderr);
    }

    /* Step 7: Temporal autocorrelation & continuity */
    if (num_frames >= 2)
    {
        double sum_seq = 0.0;
        long seq_pairs = (num_frames < 2000) ? (num_frames - 1) : 2000;
        for (long t = 0; t < seq_pairs; t++)
        {
            const float *fa = data + t * dim;
            const float *fb = data + (t + 1) * dim;
            double sum = 0.0;
            for (long d = 0; d < dim; d++)
            {
                double diff = (double)fa[d] - (double)fb[d];
                sum += diff * diff;
            }
            sum_seq += sqrt(sum);
        }
        double d_seq = sum_seq / (double)seq_pairs;
        results->profile.continuity_ratio =
            (results->profile.dist_p50 > 0.0) ? (d_seq / results->profile.dist_p50) : 1.0;
        results->noise_floor_est = d_seq / sqrt(2.0 * (double)dim);
        results->profile.noise_floor_est = results->noise_floor_est;

        if (results->profile.continuity_ratio < 0.20)
        {
            results->profile.tm_mixing_coeff = 0.35;
        }
        else if (results->profile.continuity_ratio < 0.50)
        {
            results->profile.tm_mixing_coeff = 0.15;
        }
        else
        {
            results->profile.tm_mixing_coeff = 0.0;
        }

        if (results->profile.continuity_ratio < 0.50)
        {
            results->profile.pred_enabled = 1;
            results->profile.pred_len = 2;
            int horizon = (int)(15.0 / (results->profile.continuity_ratio + 0.01));
            if (horizon > 2000) horizon = 2000;
            if (horizon < 100) horizon = 100;
            results->profile.pred_h = horizon;
        }
        else
        {
            results->profile.pred_enabled = 0;
        }

        /* Step 7b: Soft Bayesian Likelihood Gating */
        if (results->profile.rlim_balanced > 0.0)
        {
            double eta = results->profile.noise_floor_est / results->profile.rlim_balanced;
            if (eta > 0.25)
            {
                results->profile.soft_bayesian_enabled = 1;
                double coeff = 1.0 + (eta - 0.25);
                if (coeff > 1.5)
                {
                    coeff = 1.5;
                }
                results->profile.soft_bayesian_sigma_coeff = coeff;
            }
            else
            {
                results->profile.soft_bayesian_enabled = 0;
                results->profile.soft_bayesian_sigma_coeff = 1.0;
            }
        }
    } // if (num_frames >= 2)

    /* Step 8: Spatial tiling guidance */
    if (results->profile.is_image && width >= 16 && height >= 16)
    {
        results->profile.tiles_x = 2;
        results->profile.tiles_y = 2;
    }
    else
    {
        results->profile.tiles_x = 1;
        results->profile.tiles_y = 1;
    }

    /* Step 9: Empirical metric pruning profiling (TE3 vs TE4 vs TE5) */
    probe_profile_pruning(data, num_frames, dim, results->profile.rlim_balanced,
                          &results->profile);

#ifdef _OPENMP
    results->profile.ncpu = omp_get_max_threads();
#elif defined(_SC_NPROCESSORS_ONLN)
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    results->profile.ncpu = (ncpu > 0) ? (int)ncpu : 4;
#elif defined(_SC_NPROCESSORS_CONF)
    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    results->profile.ncpu = (ncpu > 0) ? (int)ncpu : 4;
#else
    results->profile.ncpu = 4;
#endif

    int est_cl = (int)(num_frames * 0.15);
    if (est_cl < 100)
    {
        est_cl = 100;
    }
    if (est_cl > 10000)
    {
        est_cl = 10000;
    }
    results->profile.recommended_maxcl = est_cl;
    results->profile.sparse_dcc_enabled = (est_cl >= 2000) ? 1 : 0;
    results->profile.recommend_double = ((global_max - global_min) > 1e7) ? 1 : 0;

    if (config->show_progress)
    {
        fprintf(stderr, "[PROBE: 100%%] Profile calibration complete\n");
        fflush(stderr);
    }

    free(mins);
    free(maxs);
    free(data);
    return 0;
}

/**
 * probe_results_free() - Free resources in a ProbeResults.
 * @results: Pointer to ProbeResults.
 */
void probe_results_free(
    ProbeResults *results)
{
    if (results == NULL)
    {
        return;
    }

    gric_profile_free(&results->profile);
}

/**
 * @file cluster_io_results.c
 * @brief Results serialization for the core clustering engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

#ifdef USE_PNG
#include "png_io.h"
#endif

#include "cluster_io.h"
#include "common.h"
#include "frameread.h"
#include "gric_bin_io.h"

static void write_dcc_results(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state)
{
    if (!config->output.output_dcc)
    {
        return;
    }

    char out_path[4096];
    printf("Writing dcc.bin\n");
    snprintf(out_path, sizeof(out_path), "%s/dcc.bin", out_dir);
    FILE *dcc_bin_fp = fopen(out_path, "wb");
    if (dcc_bin_fp != NULL)
    {
        gric_bin_header_t dcc_hdr;
        memset(&dcc_hdr, 0, sizeof(dcc_hdr));
        dcc_hdr.file_type = GRIC_BIN_TYPE_DCC;
        dcc_hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        dcc_hdr.ndim = 2;
        dcc_hdr.dims[0] = state->num_clusters;
        dcc_hdr.dims[1] = state->num_clusters;
        dcc_hdr.num_elements = (uint64_t)state->num_clusters * (uint64_t)state->num_clusters;

        if (config->algo.use_double)
        {
            dcc_hdr.data_type = GRIC_BIN_DTYPE_FLOAT64;
            dcc_hdr.data_bytes = dcc_hdr.num_elements * sizeof(double);
            if (gric_bin_write_header(dcc_bin_fp, &dcc_hdr, "DCC distance matrix") == 0)
            {
                double *dcc_buf = malloc(dcc_hdr.num_elements * sizeof(double));
                if (dcc_buf != NULL)
                {
                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        for (int j = 0; j < state->num_clusters; j++)
                        {
                            double d = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                            dcc_buf[i * state->num_clusters + j] = (d >= 0) ? d : 0.0;
                        }
                    }
                    fwrite(dcc_buf, sizeof(double), dcc_hdr.num_elements, dcc_bin_fp);
                    free(dcc_buf);
                }
            }
        }
        else
        {
            dcc_hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
            dcc_hdr.data_bytes = dcc_hdr.num_elements * sizeof(float);
            if (gric_bin_write_header(dcc_bin_fp, &dcc_hdr, "DCC distance matrix") == 0)
            {
                float *dcc_buf = malloc(dcc_hdr.num_elements * sizeof(float));
                if (dcc_buf != NULL)
                {
                    for (int i = 0; i < state->num_clusters; i++)
                    {
                        for (int j = 0; j < state->num_clusters; j++)
                        {
                            double d = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                            dcc_buf[i * state->num_clusters + j] = (d >= 0) ? (float)d : 0.0f;
                        }
                    }
                    fwrite(dcc_buf, sizeof(float), dcc_hdr.num_elements, dcc_bin_fp);
                    free(dcc_buf);
                }
            }
        }
        fclose(dcc_bin_fp);
    }

    printf("Writing dcc.txt\n");
    snprintf(out_path, sizeof(out_path), "%s/dcc.txt", out_dir);
    FILE *dcc_out = fopen(out_path, "w");
    if (dcc_out != NULL)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            for (int j = 0; j < state->num_clusters; j++)
            {
                double d = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                if (state->scratch.dcc_measured[i * config->algo.maxnbclust + j] && d >= 0)
                {
                    fprintf(dcc_out, "%d %d %.6f\n", i, j, d);
                }
            }
        }
        fclose(dcc_out);
    }

    if (config->optim.sparse_dcc_mode)
    {
        printf("Writing dccmin.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/dccmin.txt", out_dir);
        FILE *dccmin_out = fopen(out_path, "w");
        if (dccmin_out != NULL)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                for (int j = 0; j < state->num_clusters; j++)
                {
                    double d_min = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                    if (d_min > 0.0)
                    {
                        fprintf(dccmin_out, "%d %d %.6f\n", i, j, d_min);
                    }
                }
            }
            fclose(dccmin_out);
        }
    }
} // write_dcc_results

static void write_transition_matrix_results(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state)
{
    if (!config->output.output_tm || state->transition_matrix == NULL)
    {
        return;
    }

    char out_path[4096];
    printf("Writing transition_matrix.txt\n");
    snprintf(out_path, sizeof(out_path), "%s/transition_matrix.txt", out_dir);
    FILE *tm_out = fopen(out_path, "w");
    if (tm_out != NULL)
    {
        for (int i = 0; i < state->num_clusters; i++)
        {
            for (int j = 0; j < state->num_clusters; j++)
            {
                long val = state->transition_matrix[i * config->algo.maxnbclust + j];
                if (val > 0)
                {
                    fprintf(tm_out, "%d %d %ld\n", i, j, val);
                }
            }
        }
        fclose(tm_out);
    }
} // write_transition_matrix_results

static void write_anchors_results(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    long                 width,
    long                 height,
    long                 nelements)
{
    if (!config->output.output_anchors)
    {
        return;
    }

    char out_path[4096];
    printf("Writing anchors.bin\n");
    snprintf(out_path, sizeof(out_path), "%s/anchors.bin", out_dir);
    FILE *a_bin_fp = fopen(out_path, "wb");
    if (a_bin_fp != NULL)
    {
        gric_bin_header_t a_hdr;
        memset(&a_hdr, 0, sizeof(a_hdr));
        a_hdr.file_type = GRIC_BIN_TYPE_ANCHORS;
        a_hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
        a_hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        a_hdr.ndim = (nelements > 1) ? 2 : 1;
        a_hdr.dims[0] = state->num_clusters;
        a_hdr.dims[1] = nelements;
        a_hdr.num_elements = (uint64_t)state->num_clusters * (uint64_t)nelements;
        if (config->algo.use_double)
        {
            a_hdr.data_type = GRIC_BIN_DTYPE_FLOAT64;
            a_hdr.data_bytes = a_hdr.num_elements * sizeof(double);
            if (gric_bin_write_header(a_bin_fp, &a_hdr, "Cluster centroids") == 0)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    fwrite(state->clusters[i].anchor.data, sizeof(double),
                           (size_t)nelements, a_bin_fp);
                }
            }
        }
        else
        {
            a_hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
            a_hdr.data_bytes = a_hdr.num_elements * sizeof(float);
            if (gric_bin_write_header(a_bin_fp, &a_hdr, "Cluster centroids") == 0)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    fwrite(state->clusters[i].anchor.data, sizeof(float),
                           (size_t)nelements, a_bin_fp);
                }
            }
        }
        fclose(a_bin_fp);
    }

    printf("Writing anchors\n");
    if (config->output.pngout_mode)
    {
#ifdef USE_PNG
        for (int i = 0; i < state->num_clusters; i++)
        {
            snprintf(out_path, sizeof(out_path), "%s/anchor_%04d.png", out_dir, i);
            write_png_frame(out_path, state->clusters[i].anchor.data, width, height);
        }
#else
        fprintf(stderr, "Warning: PNG output requested but not compiled in.\n");
#endif
    }
    else if ((is_ascii_input_mode() || is_stream_input_mode() || height == 1) &&
             !config->output.fitsout_mode)
    {
        snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
        FILE *afptr = fopen(out_path, "w");
        if (afptr != NULL)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                for (long k = 0; k < nelements; k++)
                {
                    double v = config->algo.use_double ?
                        ((double *)state->clusters[i].anchor.data)[k] :
                        (double)((float *)state->clusters[i].anchor.data)[k];
                    fprintf(afptr, "%f ", v);
                }
                fprintf(afptr, "\n");
            }
            fclose(afptr);
        }
    }
    else
    {
#ifdef USE_CFITSIO
        int status = 0;
        fitsfile *afptr;
        snprintf(out_path, sizeof(out_path), "!%s/anchors.fits", out_dir);
        fits_create_file(&afptr, out_path, &status);
        long naxes[3] = {width, height, state->num_clusters};
        int bitpix = config->algo.use_double ? DOUBLE_IMG : FLOAT_IMG;
        int dtype = config->algo.use_double ? TDOUBLE : TFLOAT;
        fits_create_img(afptr, bitpix, 3, naxes, &status);

        for (int i = 0; i < state->num_clusters; i++)
        {
            long fpixel[3] = {1, 1, i + 1};
            fits_write_pix(afptr, dtype, fpixel, nelements,
                           state->clusters[i].anchor.data, &status);
        }
        fits_close_file(afptr, &status);
#else
        fprintf(stderr,
                "Warning: FITS output requested but not compiled in. Saving as ASCII.\n");
        snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
        FILE *afptr = fopen(out_path, "w");
        if (afptr != NULL)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                for (long k = 0; k < nelements; k++)
                {
                    double v = config->algo.use_double ?
                        ((double *)state->clusters[i].anchor.data)[k] :
                        (double)((float *)state->clusters[i].anchor.data)[k];
                    fprintf(afptr, "%f ", v);
                }
                fprintf(afptr, "\n");
            }
            fclose(afptr);
        }
#endif
    }
} // write_anchors_results

static void write_counts_results(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    const int           *cluster_counts)
{
    if (!config->output.output_counts)
    {
        return;
    }

    char out_path[4096];
    printf("Writing cluster_counts.bin\n");
    snprintf(out_path, sizeof(out_path), "%s/cluster_counts.bin", out_dir);
    FILE *cnt_bin_fp = fopen(out_path, "wb");
    if (cnt_bin_fp != NULL)
    {
        gric_bin_header_t cnt_hdr;
        memset(&cnt_hdr, 0, sizeof(cnt_hdr));
        cnt_hdr.file_type = GRIC_BIN_TYPE_COUNTS;
        cnt_hdr.data_type = GRIC_BIN_DTYPE_UINT32;
        cnt_hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        cnt_hdr.ndim = 1;
        cnt_hdr.dims[0] = state->num_clusters;
        cnt_hdr.num_elements = state->num_clusters;
        cnt_hdr.data_bytes = cnt_hdr.num_elements * sizeof(uint32_t);

        if (gric_bin_write_header(cnt_bin_fp, &cnt_hdr, "Cluster counts") == 0)
        {
            uint32_t *cnt_buf = malloc(cnt_hdr.num_elements * sizeof(uint32_t));
            if (cnt_buf != NULL)
            {
                for (int c = 0; c < state->num_clusters; c++)
                {
                    cnt_buf[c] = (uint32_t)cluster_counts[c];
                }
                fwrite(cnt_buf, sizeof(uint32_t), cnt_hdr.num_elements, cnt_bin_fp);
                free(cnt_buf);
            }
        }
        fclose(cnt_bin_fp);
    }

    printf("Writing cluster_counts.txt\n");
    snprintf(out_path, sizeof(out_path), "%s/cluster_counts.txt", out_dir);
    FILE *count_out = fopen(out_path, "w");
    if (count_out != NULL)
    {
        for (int c = 0; c < state->num_clusters; c++)
        {
            fprintf(count_out, "Cluster %d: %d frames\n", c, cluster_counts[c]);
        }
        fclose(count_out);
    }
} // write_counts_results

static void write_membership_results(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state)
{
    if (!config->output.output_membership)
    {
        return;
    }

    char out_path[4096];
    printf("Writing frame_membership.bin\n");
    snprintf(out_path, sizeof(out_path), "%s/frame_membership.bin", out_dir);
    FILE *mem_bin_fp = fopen(out_path, "wb");
    if (mem_bin_fp != NULL)
    {
        gric_bin_header_t mem_hdr;
        memset(&mem_hdr, 0, sizeof(mem_hdr));
        mem_hdr.file_type = GRIC_BIN_TYPE_MEMBERSHIP;
        mem_hdr.data_type = GRIC_BIN_DTYPE_UINT32;
        mem_hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        mem_hdr.ndim = 1;
        mem_hdr.dims[0] = state->telemetry.total_frames_processed;
        mem_hdr.num_elements = state->telemetry.total_frames_processed;
        mem_hdr.data_bytes = mem_hdr.num_elements * sizeof(uint32_t);

        if (gric_bin_write_header(mem_bin_fp, &mem_hdr, "Frame membership") == 0)
        {
            uint32_t *mem_buf = malloc(mem_hdr.num_elements * sizeof(uint32_t));
            if (mem_buf != NULL)
            {
                for (long f = 0; f < state->telemetry.total_frames_processed; f++)
                {
                    mem_buf[f] = (state->assignments[f] >= 0) ?
                                 (uint32_t)state->assignments[f] : 0;
                }
                fwrite(mem_buf, sizeof(uint32_t), mem_hdr.num_elements, mem_bin_fp);
                free(mem_buf);
            }
        }
        fclose(mem_bin_fp);
    }
} // write_membership_results

static void write_radii_results(
    const char         *out_dir,
    const ClusterState *state,
    const int          *cluster_counts)
{
    double *cluster_max_radii = calloc((size_t)state->num_clusters, sizeof(double));
    if (cluster_max_radii == NULL)
    {
        return;
    }

    for (long f = 0; f < state->telemetry.total_frames_processed; f++)
    {
        int c = state->assignments[f];
        if (c >= 0 && c < state->num_clusters && state->frame_infos &&
            state->frame_infos[f].cluster_indices && state->frame_infos[f].distances)
        {
            double d = 0.0;
            for (int i = 0; i < state->frame_infos[f].num_dists; i++)
            {
                if (state->frame_infos[f].cluster_indices[i] == c)
                {
                    d = state->frame_infos[f].distances[i];
                    break;
                }
            }
            if (d > cluster_max_radii[c])
            {
                cluster_max_radii[c] = d;
            }
        }
    }

    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/cluster_radii.bin", out_dir);
    FILE *rad_bin_fp = fopen(out_path, "wb");
    if (rad_bin_fp != NULL)
    {
        gric_bin_header_t rad_hdr;
        memset(&rad_hdr, 0, sizeof(rad_hdr));
        rad_hdr.file_type = GRIC_BIN_TYPE_GENERIC;
        rad_hdr.data_type = GRIC_BIN_DTYPE_FLOAT32;
        rad_hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
        rad_hdr.ndim = 1;
        rad_hdr.dims[0] = state->num_clusters;
        rad_hdr.num_elements = state->num_clusters;
        rad_hdr.data_bytes = rad_hdr.num_elements * sizeof(float);

        if (gric_bin_write_header(rad_bin_fp, &rad_hdr, "Cluster max radii") == 0)
        {
            float *rad_buf = malloc(rad_hdr.num_elements * sizeof(float));
            if (rad_buf != NULL)
            {
                for (int c = 0; c < state->num_clusters; c++)
                {
                    rad_buf[c] = (float)cluster_max_radii[c];
                }
                fwrite(rad_buf, sizeof(float), rad_hdr.num_elements, rad_bin_fp);
                free(rad_buf);
            }
        }
        fclose(rad_bin_fp);
    }

    snprintf(out_path, sizeof(out_path), "%s/cluster_radii.txt", out_dir);
    FILE *radii_out = fopen(out_path, "w");
    if (radii_out != NULL)
    {
        fprintf(radii_out, "# cluster_id member_count max_radius\n");
        for (int c = 0; c < state->num_clusters; c++)
        {
            fprintf(radii_out, "%d %d %.6f\n",
                    c, cluster_counts[c], cluster_max_radii[c]);
        }
        fclose(radii_out);
    }
    free(cluster_max_radii);
} // write_radii_results

#ifdef USE_PNG
/**
 * write_clusters_and_averages_png() - Export cluster member frames and averages as PNG files.
 * @out_dir:        Target output directory.
 * @config:         Active ClusterConfig.
 * @state:          Active ClusterState.
 * @cluster_counts: Frame counts per cluster.
 * @width:          Image width in pixels.
 * @height:         Image height in pixels.
 * @nelements:      Pixel count per frame.
 * @avg_buffer:     Preallocated scratch buffer for computing frame averages.
 */
static void write_clusters_and_averages_png(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    const int           *cluster_counts,
    long                 width,
    long                 height,
    long                 nelements,
    double              *avg_buffer)
{
    char out_path[4096];

    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] == 0)
        {
            continue;
        }

        if (config->output.output_clusters)
        {
            char cluster_dir[1024];
            snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
            safe_mkdir(cluster_dir);
        }

        if (config->output.average_mode && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                avg_buffer[k] = 0.0;
            }
        }

        for (long f = 0; f < state->telemetry.total_frames_processed; f++)
        {
            if (state->assignments[f] == c)
            {
                Frame *fr = getframe_at(f);
                if (fr != NULL)
                {
                    if (config->output.output_clusters)
                    {
                        char cluster_dir[1024];
                        snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                        snprintf(out_path, sizeof(out_path), "%s/frame%05ld.png", cluster_dir, f);
                        write_png_frame(out_path, fr->data, width, height);
                    }
                    if (config->output.average_mode && avg_buffer != NULL)
                    {
                        for (long k = 0; k < nelements; k++)
                        {
                            double v = fr->is_double ?
                                ((double *)fr->data)[k] : (double)((float *)fr->data)[k];
                            avg_buffer[k] += v;
                        }
                    }
                    free_frame(fr);
                }
            }
        }

        if (config->output.average_mode && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                avg_buffer[k] /= cluster_counts[c];
            }
            snprintf(out_path, sizeof(out_path), "%s/average_%04d.png", out_dir, c);
            write_png_frame(out_path, avg_buffer, width, height);
        }
    }
}
#endif // USE_PNG

/**
 * write_clusters_and_averages_ascii() - Export cluster member frames and averages to ASCII text.
 * @out_dir:        Target output directory.
 * @config:         Active ClusterConfig.
 * @state:          Active ClusterState.
 * @cluster_counts: Frame counts per cluster.
 * @nelements:      Total elements per frame.
 * @avg_buffer:     Preallocated scratch buffer for frame averages.
 */
static void write_clusters_and_averages_ascii(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    const int           *cluster_counts,
    long                 nelements,
    double              *avg_buffer)
{
    char out_path[4096];
    FILE *avg_file = NULL;

    if (config->output.average_mode)
    {
        snprintf(out_path, sizeof(out_path), "%s/average.txt", out_dir);
        avg_file = fopen(out_path, "w");
    }

    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] == 0)
        {
            if (avg_file != NULL)
            {
                for (long k = 0; k < nelements; k++)
                {
                    fprintf(avg_file, "0.0 ");
                }
                fprintf(avg_file, "\n");
            }
            continue;
        }

        FILE *cfptr = NULL;
        if (config->output.output_clusters)
        {
            char fname[1024];
            snprintf(fname, sizeof(fname), "%s/cluster_%d.txt", out_dir, c);
            cfptr = fopen(fname, "w");
        }

        if (config->output.average_mode && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                avg_buffer[k] = 0.0;
            }
        }

        for (long f = 0; f < state->telemetry.total_frames_processed; f++)
        {
            if (state->assignments[f] == c)
            {
                Frame *fr = getframe_at(f);
                if (fr != NULL)
                {
                    for (long k = 0; k < nelements; k++)
                    {
                        double v = fr->is_double ?
                            ((double *)fr->data)[k] : (double)((float *)fr->data)[k];
                        if (cfptr != NULL)
                        {
                            fprintf(cfptr, "%f ", v);
                        }
                        if (config->output.average_mode && avg_buffer != NULL)
                        {
                            avg_buffer[k] += v;
                        }
                    }
                    if (cfptr != NULL)
                    {
                        fprintf(cfptr, "\n");
                    }
                    free_frame(fr);
                }
            }
        }

        if (cfptr != NULL)
        {
            fclose(cfptr);
        }

        if (avg_file != NULL && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                fprintf(avg_file, "%f ", avg_buffer[k] / cluster_counts[c]);
            }
            fprintf(avg_file, "\n");
        }
    }

    if (avg_file != NULL)
    {
        fclose(avg_file);
    }
}

#ifdef USE_CFITSIO
/**
 * write_clusters_and_averages_fits() - Export cluster cubes and average images as FITS.
 * @out_dir:        Target output directory.
 * @config:         Active ClusterConfig.
 * @state:          Active ClusterState.
 * @cluster_counts: Frame counts per cluster.
 * @width:          Image width in pixels.
 * @height:         Image height in pixels.
 * @nelements:      Pixel count per frame.
 * @avg_buffer:     Preallocated scratch buffer for frame averages.
 */
static void write_clusters_and_averages_fits(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    const int           *cluster_counts,
    long                 width,
    long                 height,
    long                 nelements,
    double              *avg_buffer)
{
    char out_path[4096];
    int status = 0;
    fitsfile *avg_ptr = NULL;

    if (config->output.average_mode)
    {
        snprintf(out_path, sizeof(out_path), "!%s/average.fits", out_dir);
        fits_create_file(&avg_ptr, out_path, &status);
        long anaxes[3] = {width, height, state->num_clusters};
        fits_create_img(avg_ptr, DOUBLE_IMG, 3, anaxes, &status);
    }

    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] == 0)
        {
            continue;
        }

        fitsfile *cfptr = NULL;
        if (config->output.output_clusters)
        {
            char fname[1024];
            snprintf(fname, sizeof(fname), "!%s/cluster_%d.fits", out_dir, c);
            fits_create_file(&cfptr, fname, &status);
            long cnaxes[3] = {width, height, cluster_counts[c]};
            int bitpix = config->algo.use_double ? DOUBLE_IMG : FLOAT_IMG;
            fits_create_img(cfptr, bitpix, 3, cnaxes, &status);
        }

        if (config->output.average_mode && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                avg_buffer[k] = 0.0;
            }
        }

        int fr_count = 0;
        for (long f = 0; f < state->telemetry.total_frames_processed; f++)
        {
            if (state->assignments[f] == c)
            {
                Frame *fr = getframe_at(f);
                if (fr != NULL)
                {
                    if (cfptr != NULL)
                    {
                        long fpixel[3] = {1, 1, fr_count + 1};
                        int dtype = fr->is_double ? TDOUBLE : TFLOAT;
                        fits_write_pix(cfptr, dtype, fpixel, nelements, fr->data, &status);
                    }
                    if (config->output.average_mode && avg_buffer != NULL)
                    {
                        for (long k = 0; k < nelements; k++)
                        {
                            double v = fr->is_double ?
                                ((double *)fr->data)[k] : (double)((float *)fr->data)[k];
                            avg_buffer[k] += v;
                        }
                    }
                    free_frame(fr);
                    fr_count++;
                }
            }
        }

        if (cfptr != NULL)
        {
            fits_close_file(cfptr, &status);
        }

        if (config->output.average_mode && avg_ptr != NULL && avg_buffer != NULL)
        {
            for (long k = 0; k < nelements; k++)
            {
                avg_buffer[k] /= cluster_counts[c];
            }
            long fpixel[3] = {1, 1, c + 1};
            fits_write_pix(avg_ptr, TDOUBLE, fpixel, nelements, avg_buffer, &status);
        }
    }

    if (avg_ptr != NULL)
    {
        fits_close_file(avg_ptr, &status);
    }
}
#endif // USE_CFITSIO

/**
 * write_clusters_and_averages() - Export cluster data and computed averages in active formats.
 * @out_dir:        Target output directory.
 * @config:         Active ClusterConfig.
 * @state:          Active ClusterState.
 * @cluster_counts: Frame counts per cluster.
 * @width:          Image width in pixels.
 * @height:         Image height in pixels.
 * @nelements:      Total elements per frame.
 */
static void write_clusters_and_averages(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    const int           *cluster_counts,
    long                 width,
    long                 height,
    long                 nelements)
{
    double *avg_buffer = NULL;
    if (config->output.average_mode)
    {
        avg_buffer = calloc((size_t)nelements, sizeof(double));
    }

    int active_cluster_count = 0;
    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] > 0)
        {
            active_cluster_count++;
        }
    }

    if (config->output.output_clusters)
    {
        printf("Writing cluster files (%d files)\n", active_cluster_count);
    }
    if (config->output.average_mode)
    {
        printf("Writing average cluster files\n");
    }

    if (config->output.pngout_mode)
    {
#ifdef USE_PNG
        write_clusters_and_averages_png(
            out_dir, config, state, cluster_counts, width, height, nelements, avg_buffer
        );
#endif
    }
    else if ((is_ascii_input_mode() || is_stream_input_mode() || height == 1) &&
             !config->output.fitsout_mode)
    {
        write_clusters_and_averages_ascii(
            out_dir, config, state, cluster_counts, nelements, avg_buffer
        );
    }
    else
    {
#ifdef USE_CFITSIO
        write_clusters_and_averages_fits(
            out_dir, config, state, cluster_counts, width, height, nelements, avg_buffer
        );
#endif
    }

    if (avg_buffer != NULL)
    {
        free(avg_buffer);
    }
} // write_clusters_and_averages

static void write_clustered_output_file(
    const char          *out_dir,
    const ClusterConfig *config,
    const ClusterState  *state,
    long                 nelements)
{
    if (!config->output.output_clustered)
    {
        return;
    }

    printf("Writing clustered output file\n");
    const char *base_name_only = strrchr(config->input.fits_filename, '/');
    if (base_name_only != NULL)
    {
        base_name_only++;
    }
    else
    {
        base_name_only = config->input.fits_filename;
    }

    char *temp_base = strdup(base_name_only);
    char *ext = strrchr(temp_base, '.');
    if (ext != NULL && strcmp(ext, ".txt") == 0)
    {
        *ext = '\0';
    }

    char *clustered_fname = malloc(strlen(out_dir) + strlen(temp_base) + 30);
    if (clustered_fname != NULL)
    {
        sprintf(clustered_fname, "%s/%s.clustered.txt", out_dir, temp_base);
        free(temp_base);
        FILE *clustered_out = fopen(clustered_fname, "w");
        if (clustered_out != NULL)
        {
            fprintf(clustered_out, "# Parameters:\n");
            fprintf(clustered_out, "# rlim %.6f\n", config->algo.rlim);
            fprintf(clustered_out, "# dprob %.6f\n", config->algo.deltaprob);
            fprintf(clustered_out, "# maxcl %d\n", config->algo.maxnbclust);
            fprintf(clustered_out, "# maxim %ld\n", config->input.maxnbfr);
            fprintf(clustered_out, "# gprob_mode %d\n", config->optim.gprob_mode);
            fprintf(clustered_out, "# fmatcha %.2f\n", config->optim.fmatch_a);
            fprintf(clustered_out, "# fmatchb %.2f\n", config->optim.fmatch_b);

            fprintf(clustered_out, "# Stats:\n");
            fprintf(clustered_out, "# Total Clusters %d\n", state->num_clusters);
            fprintf(clustered_out,
                    "# Total Distance Computations %ld\n",
                    state->telemetry.framedist_calls);
            fprintf(clustered_out, "# Clusters Pruned %ld\n",
                    state->telemetry.clusters_pruned);

            double avg_dist = 0.0;
            if (state->telemetry.total_frames_processed > 0)
            {
                avg_dist = (double)state->telemetry.framedist_calls /
                           (double)state->telemetry.total_frames_processed;
            }
            fprintf(clustered_out, "# Avg Dist/Frame %.2f\n", avg_dist);

            if (state->telemetry.pruned_fraction_sum && state->telemetry.step_counts)
            {
                for (int k = 0; k < state->telemetry.max_steps_recorded; k++)
                {
                    if (state->telemetry.step_counts[k] > 0)
                    {
                        fprintf(clustered_out,
                                "# Pruning Step %d: %.4f\n",
                                k,
                                state->telemetry.pruned_fraction_sum[k] /
                                state->telemetry.step_counts[k]);
                    }
                    else if (k > 0 && state->telemetry.step_counts[k] == 0)
                    {
                        break;
                    }
                }
            }

            int next_new_cluster = 0;
            for (long i = 0; i < state->telemetry.total_frames_processed; i++)
            {
                int assigned = state->assignments[i];
                if (assigned == next_new_cluster)
                {
                    fprintf(clustered_out, "# NEWCLUSTER %d %ld ", assigned, i);
                    for (long k = 0; k < nelements; k++)
                    {
                        double v = config->algo.use_double ?
                            ((double *)state->clusters[assigned].anchor.data)[k] :
                            (double)((float *)state->clusters[assigned].anchor.data)[k];
                        fprintf(clustered_out, "%f ", v);
                    }
                    fprintf(clustered_out, "\n");
                    next_new_cluster++;
                }

                Frame *fr = getframe_at(i);
                if (fr != NULL)
                {
                    fprintf(clustered_out, "%ld %d ", i, assigned);
                    for (long k = 0; k < nelements; k++)
                    {
                        double v = fr->is_double ?
                            ((double *)fr->data)[k] : (double)((float *)fr->data)[k];
                        fprintf(clustered_out, "%f ", v);
                    }
                    fprintf(clustered_out, "\n");
                    free_frame(fr);
                }
            }
            fclose(clustered_out);
        }
        free(clustered_fname);
    }
    else
    {
        free(temp_base);
    }
} // write_clustered_output_file

void write_results(
    ClusterConfig *config,
    ClusterState  *state)
{
    char *out_dir = NULL;
    if (config->output.user_outdir)
    {
        out_dir = strdup(config->output.user_outdir);
    }
    else
    {
        out_dir = create_output_dir_name(config->input.fits_filename);
    }

    if (!out_dir)
    {
        return;
    }

    long width = get_frame_width();
    long height = get_frame_height();
    long nelements = width * height;

    /* Compute cluster counts */
    int *cluster_counts = calloc((size_t)state->num_clusters, sizeof(int));
    if (cluster_counts != NULL)
    {
        for (long i = 0; i < state->telemetry.total_frames_processed; i++)
        {
            if (state->assignments[i] >= 0 && state->assignments[i] < state->num_clusters)
            {
                cluster_counts[state->assignments[i]]++;
            }
        }
    }

    write_dcc_results(out_dir, config, state);
    write_transition_matrix_results(out_dir, config, state);
    write_anchors_results(out_dir, config, state, width, height, nelements);
    if (cluster_counts != NULL)
    {
        write_counts_results(out_dir, config, state, cluster_counts);
    }
    write_membership_results(out_dir, config, state);
    if (cluster_counts != NULL)
    {
        write_radii_results(out_dir, state, cluster_counts);
        write_clusters_and_averages(out_dir, config, state, cluster_counts,
                                    width, height, nelements);
        free(cluster_counts);
    }
    write_clustered_output_file(out_dir, config, state, nelements);

    free(out_dir);
} // write_results

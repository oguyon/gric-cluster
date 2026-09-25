/**
 * @file gric_fps_common.c
 * @brief Common adapter implementation for GRIC Milk streaming integration.
 */

#include "gric_fps_common.h"
#include "gric_fps_params.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char     fps_in_name[FUNCTION_PARAMETER_STRMAXLEN]          = "";
char     fps_out_assign_name[FUNCTION_PARAMETER_STRMAXLEN]  = "";
char     fps_out_anchors_name[FUNCTION_PARAMETER_STRMAXLEN] = "";
char     fps_out_counts_name[FUNCTION_PARAMETER_STRMAXLEN]  = "";
int32_t  fps_stream_anchors   = 0;
int32_t  fps_stream_counts    = 0;
int32_t  fps_allow_frame_drop = 0;
double   fps_rlim             = 0.5;
double   fps_deltaprob        = 0.01;
uint32_t fps_maxnbclust       = 256;
int64_t  fps_maxcl_strategy   = 0;
uint32_t fps_ncpu             = 0;
int32_t  fps_use_double       = 0;
int32_t  fps_use_sq16         = 0;
int32_t  fps_entropy_mode     = 0;
int32_t  fps_reset_state      = 0;

static gric_cluster_t *cluster_ctx        = NULL;
static double         *coord_conv_buf     = NULL;
static size_t          current_ndim       = 0;
static uint32_t        prev_cluster_count = 0;

/**
 * gric_fps_custom_conf_check() - Validate runtime configuration parameters.
 *
 * Checks input stream existence, valid radius limits, thread limits, and cluster limits.
 *
 * Return: 0 on valid configuration, non-zero if invalid.
 */
errno_t gric_fps_custom_conf_check(void)
{
    if (fps_in_name[0] == '\0')
    {
        return 1;
    }

    if (fps_rlim <= 0.0)
    {
        fps_rlim = 0.5;
    }

    if (fps_deltaprob <= 0.0 || fps_deltaprob > 1.0)
    {
        fps_deltaprob = 0.01;
    }

    if (fps_maxnbclust < 2)
    {
        fps_maxnbclust = 256;
    }

    if (fps_ncpu == 0)
    {
        fps_ncpu = 1;
    }

    /* Probe input stream in shared memory */
    IMAGE test_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_in_name, &test_img) == 0)
    {
        ImageStreamIO_closeIm(&test_img);
    }

    return 0;
}

/**
 * gric_fps_init_engine() - Allocate and initialize the libgric session.
 * @ndim: Total elements per frame.
 *
 * Return: 0 on success, or non-zero on failure.
 */
errno_t gric_fps_init_engine(
    uint32_t ndim)
{
    gric_fps_cleanup_engine();

    current_ndim = (size_t)ndim;
    coord_conv_buf = (double *)calloc(current_ndim, sizeof(double));
    if (!coord_conv_buf)
    {
        return 1;
    }

    gric_cluster_config_t cfg;
    gric_cluster_config_default(&cfg);

    cfg.rlim = fps_rlim;
    cfg.maxnbclust = (int)fps_maxnbclust;
    cfg.tm_mixing_coeff = fps_deltaprob;
    cfg.ncpu = (int)fps_ncpu;
    cfg.use_double = (int)fps_use_double;
    cfg.use_sq16 = (int)fps_use_sq16;
    cfg.entropy_mode = (int)fps_entropy_mode;
    cfg.maxcl_strategy = (int)fps_maxcl_strategy;

    cluster_ctx = gric_cluster_create(&cfg, current_ndim);
    if (!cluster_ctx)
    {
        free(coord_conv_buf);
        coord_conv_buf = NULL;
        return 1;
    }

    prev_cluster_count = 0;
    return 0;
}

/**
 * gric_fps_cleanup_engine() - Free the libgric clustering session and buffers.
 */
void gric_fps_cleanup_engine(void)
{
    if (cluster_ctx)
    {
        gric_cluster_destroy(cluster_ctx);
        cluster_ctx = NULL;
    }
    if (coord_conv_buf)
    {
        free(coord_conv_buf);
        coord_conv_buf = NULL;
    }
    current_ndim = 0;
    prev_cluster_count = 0;
}

/**
 * gric_fps_init_output_streams() - Create all output ImageStreamIO streams.
 * @xsize:        Frame width.
 * @ysize:        Frame height.
 * @max_clusters: Cluster capacity.
 * @out_assign:   Assignment output stream.
 * @out_anchors:  Centroid vectors output stream.
 * @out_counts:   Cluster count output stream.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_fps_init_output_streams(
    uint32_t xsize,
    uint32_t ysize,
    uint32_t max_clusters,
    IMAGE   *out_assign,
    IMAGE   *out_anchors,
    IMAGE   *out_counts)
{
    const char *assign_name = (fps_out_assign_name[0] != '\0')
                                  ? fps_out_assign_name : "gric_assign";

    /* Create 2D assignment stream [GRIC_ASSIGN_PACKET_ELEMS x 1] */
    uint32_t assign_dims[2] = { GRIC_ASSIGN_PACKET_ELEMS, 1 };
    if (ImageStreamIO_createIm_gpu(out_assign, assign_name, 2, assign_dims,
                                   _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0)
    {
        return 1;
    }

    /* Optionally create 3D anchors stream [xsize x ysize x max_clusters] */
    if (fps_stream_anchors != 0 && out_anchors)
    {
        const char *anchors_name = (fps_out_anchors_name[0] != '\0')
                                       ? fps_out_anchors_name : "gric_anchors";
        uint32_t anchor_dims[3] = { xsize, ysize, max_clusters };
        ImageStreamIO_createIm_gpu(out_anchors, anchors_name, 3, anchor_dims,
                                   _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);
    }

    /* Optionally create 1D counts stream [max_clusters] */
    if (fps_stream_counts != 0 && out_counts)
    {
        const char *counts_name = (fps_out_counts_name[0] != '\0')
                                      ? fps_out_counts_name : "gric_counts";
        uint32_t count_dims[1] = { max_clusters };
        ImageStreamIO_createIm_gpu(out_counts, counts_name, 1, count_dims,
                                   _DATATYPE_UINT32, -1, 1, 10, 0, 0, 0);
    }

    return 0;
}

/**
 * gric_fps_close_output_streams() - Safely disconnect output ImageStreamIO streams.
 * @out_assign:  Assignment stream handle.
 * @out_anchors: Centroids stream handle.
 * @out_counts:  Counts stream handle.
 */
void gric_fps_close_output_streams(
    IMAGE *out_assign,
    IMAGE *out_anchors,
    IMAGE *out_counts)
{
    if (out_assign && out_assign->used == 1)
    {
        ImageStreamIO_closeIm(out_assign);
    }
    if (out_anchors && out_anchors->used == 1)
    {
        ImageStreamIO_closeIm(out_anchors);
    }
    if (out_counts && out_counts->used == 1)
    {
        ImageStreamIO_closeIm(out_counts);
    }
}

/**
 * gric_fps_process_frame() - Cluster one incoming frame and broadcast output.
 * @raw_pixels:  Source pixel buffer pointer.
 * @datatype:    Input stream pixel data type code.
 * @ndim:        Dimension count.
 * @frame_index: Frame count (cnt0).
 * @frame_time:  Arrival timestamp.
 * @latency_us:  Algorithm execution latency in microseconds.
 * @out_assign:  Assignment stream handle.
 * @out_anchors: Centroids stream handle.
 * @out_counts:  Counts stream handle.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_fps_process_frame(
    const void      *raw_pixels,
    int              datatype,
    uint32_t         ndim,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_assign,
    IMAGE           *out_anchors,
    IMAGE           *out_counts)
{
    (void)frame_time;
    if (!cluster_ctx || !coord_conv_buf || ndim != current_ndim)
    {
        return 1;
    }

    /* Check dynamic state reset trigger */
    if (fps_reset_state != 0)
    {
        fps_reset_state = 0;
        gric_cluster_reset(cluster_ctx);
        prev_cluster_count = 0;
    }

    /* Convert input pixels to coordinate buffer */
    {
        switch (datatype)
        {
        case _DATATYPE_FLOAT:
        {
            const float *f_in = (const float *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)f_in[ii];
            }
            break;
        }
        case _DATATYPE_DOUBLE:
        {
            memcpy(coord_conv_buf, raw_pixels, ndim * sizeof(double));
            break;
        }
        case _DATATYPE_UINT16:
        {
            const uint16_t *u16_in = (const uint16_t *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)u16_in[ii];
            }
            break;
        }
        case _DATATYPE_UINT8:
        {
            const uint8_t *u8_in = (const uint8_t *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)u8_in[ii];
            }
            break;
        }
        default:
            return 1;
        }
    }

    /* Ingest frame through libgric clustering pipeline */
    int64_t assigned_cluster_id = -1;
    gric_status_t status = gric_cluster_feed_frame(cluster_ctx, coord_conv_buf,
                                                   &assigned_cluster_id);
    if (status != GRIC_SUCCESS)
    {
        return 1;
    }

    int64_t total_clusters = gric_cluster_get_num_clusters(cluster_ctx);
    int is_new = (total_clusters > (int64_t)prev_cluster_count);

    /* Write 1D telemetry packet to out_assign */
    if (out_assign && out_assign->used == 1)
    {
        out_assign->md[0].write = 1;
        float *assign_data = (float *)out_assign->array.raw;
        assign_data[0] = (float)frame_index;
        assign_data[1] = (float)assigned_cluster_id;
        assign_data[2] = 0.0f;
        assign_data[3] = is_new ? 1.0f : 0.0f;
        assign_data[4] = (float)total_clusters;
        assign_data[5] = (float)latency_us;
        assign_data[6] = 1.0f;
        assign_data[7] = 0.0f;

        out_assign->md[0].cnt0++;
        out_assign->md[0].write = 0;
        ImageStreamIO_sempost(out_assign, -1);
    }

    /* Update anchors stream if a new cluster was spawned */
    if (is_new && out_anchors && out_anchors->used == 1 &&
        assigned_cluster_id >= 0 && assigned_cluster_id < (int64_t)out_anchors->md[0].size[2])
    {
        out_anchors->md[0].write = 1;
        float *anchor_slice = (float *)out_anchors->array.raw +
                              ((size_t)assigned_cluster_id * ndim);
        for (uint32_t ii = 0; ii < ndim; ii++)
        {
            anchor_slice[ii] = (float)coord_conv_buf[ii];
        }
        out_anchors->md[0].cnt0++;
        out_anchors->md[0].write = 0;
        ImageStreamIO_sempost(out_anchors, -1);
    }

    /* Update counts stream */
    if (out_counts && out_counts->used == 1 && assigned_cluster_id >= 0 &&
        assigned_cluster_id < (int64_t)out_counts->md[0].size[0])
    {
        out_counts->md[0].write = 1;
        uint32_t *counts_data = (uint32_t *)out_counts->array.raw;
        counts_data[assigned_cluster_id]++;
        out_counts->md[0].cnt0++;
        out_counts->md[0].write = 0;
        ImageStreamIO_sempost(out_counts, -1);
    }

    prev_cluster_count = (uint32_t)total_clusters;
    return 0;
}

/**
 * gric_fps_get_cluster_count() - Query discovered cluster count.
 *
 * Return: Cluster count, or 0 if uninitialized.
 */
int64_t gric_fps_get_cluster_count(void)
{
    return cluster_ctx ? gric_cluster_get_num_clusters(cluster_ctx) : 0;
}

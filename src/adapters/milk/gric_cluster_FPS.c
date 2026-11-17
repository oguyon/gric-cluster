/**
 * @file gric_cluster_FPS.c
 * @brief MILK_RUNNABLE streaming clustering unit (CLI + standalone fpsexec).
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "gric_fps_common.h"
#include "gric_fps_params.h"
#include "gric_milk_unit_helpers.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

MILK_PARAMS_DEFINE(gric_cluster, GRIC_FPS_PARAMS);

/**
 * cluster_process_step() - Execute one frame of clustering and update telemetry.
 * @in_im:       Input ImageStreamIO stream.
 * @ndim:        Pixels per 2D frame.
 * @frame_count: 1-based processed frame counter.
 * @processinfo: Active PROCESSINFO handle (may be NULL).
 * @out_assign:  Assignment telemetry output stream.
 * @out_anchors: Cluster anchors output stream.
 * @out_counts:  Cluster counts output stream.
 *
 * Return: 1 to continue loop, 0 if fps_max_frames has been reached.
 */
static int cluster_process_step(
    IMAGE       *in_im,
    uint32_t     ndim,
    uint64_t     frame_count,
    PROCESSINFO *processinfo,
    IMAGE       *out_assign,
    IMAGE       *out_anchors,
    IMAGE       *out_counts)
{
    uint64_t cnt0 = in_im->md[0].cnt0;
    uint32_t slice_idx = (in_im->md[0].naxis > 2) ? (uint32_t)in_im->md[0].cnt1 : 0;
    int typesize = ImageStreamIO_typesize((uint8_t)in_im->md[0].datatype);
    const void *raw_slice = (const char *)in_im->array.raw +
                            ((size_t)slice_idx * ndim * (size_t)typesize);

    struct timespec t_start;
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    gric_fps_process_frame(raw_slice, in_im->md[0].datatype, ndim,
                           cnt0, in_im->md[0].atime, 0.0,
                           out_assign, out_anchors, out_counts);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e3;
    if (out_assign->used == 1)
    {
        float *assign_data = (float *)out_assign->array.raw;
        assign_data[5] = (float)latency_us;
        out_assign->md[0].write = 0;
    }

    processinfo_update_output_stream(processinfo, out_assign, in_im);

    long write_slice = (in_im->md[0].naxis > 2) ? (long)in_im->md[0].cnt1 : 0;
    long stream_lag = (in_im->md[0].cnt0 > frame_count)
                          ? (long)(in_im->md[0].cnt0 - frame_count)
                          : 0;
    gric_fps_status_update(cnt0, latency_us, stream_lag, write_slice, (long)slice_idx);

    if (frame_count % 100 == 0)
    {
        processinfo_WriteMessage_fmt(processinfo, "K=%ld Frames=%lu",
                                     (long)gric_fps_get_cluster_count(),
                                     (unsigned long)frame_count);
    }

    if (fps_max_frames > 0 && frame_count >= fps_max_frames)
    {
        return 0;
    }

    if (fps_cnt2sync)
    {
        in_im->md[0].cnt2++;
    }
    return 1;
}

/**
 * gric_cluster_compute() - Main real-time stream clustering runnable loop.
 * @self: Pointer to runnable_gric_cluster descriptor.
 *
 * Return: RETURN_SUCCESS on clean exit, RETURN_FAILURE on setup error.
 */
static errno_t gric_cluster_compute(
    const MILK_RUNNABLE *self)
{
    if (fps_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input stream specified.\n");
        return RETURN_FAILURE;
    }

    IMGID inimg = stream_connect(fps_in_name);
    if (inimg.ID == -1 || inimg.im == NULL)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_in_name);
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    uint32_t xsize = inimg.im->md[0].size[0];
    uint32_t ysize = inimg.im->md[0].size[1];
    uint32_t ndim = xsize * ysize;
    uint32_t max_clusters = (fps_maxnbclust > 0) ? fps_maxnbclust : 256;

    if (gric_fps_init_engine(ndim) != 0)
    {
        fprintf(stderr, "Error initializing GRIC engine.\n");
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    uint32_t loaded_k = (uint32_t)gric_fps_get_cluster_count();
    if (loaded_k > max_clusters)
    {
        max_clusters = loaded_k;
    }

    IMAGE out_assign;
    IMAGE out_anchors;
    IMAGE out_counts;
    memset(&out_assign, 0, sizeof(IMAGE));
    memset(&out_anchors, 0, sizeof(IMAGE));
    memset(&out_counts, 0, sizeof(IMAGE));

    if (gric_fps_init_output_streams(xsize, ysize, max_clusters,
                                     &out_assign, &out_anchors, &out_counts) != 0)
    {
        fprintf(stderr, "Error creating output streams.\n");
        gric_fps_cleanup_engine();
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    gric_milk_configure_trigger(self, fps_in_name, fps_max_frames);

    MILK_PROCINFO_INIT(self)
    gric_milk_ensure_sem_trigger(processinfo, inimg.im);
    gric_fps_status_init(gric_milk_instance_name("gric_cluster", processinfo),
                         fps_shm_status_file);

    uint64_t frame_count = 0;
    MILK_PROCINFO_LOOPSTART(self)
    {
        frame_count++;
        if (cluster_process_step(inimg.im, ndim, frame_count, processinfo,
                                 &out_assign, &out_anchors, &out_counts) == 0)
        {
            processloopOK = 0;
        }
    }
    MILK_PROCINFO_END(self)

    gric_fps_status_close(0);
    gric_fps_close_output_streams(&out_assign, &out_anchors, &out_counts);
    if (fps_save_dir[0] != '\0')
    {
        gric_fps_save_results(fps_save_dir);
    }
    gric_fps_cleanup_engine();
    imgid_free(&inimg);

    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_gric_cluster = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "gric_cluster",
              .cmdkey      = "gric_cluster",
              .description = "GRIC real-time geometric stream clustering" },
    .srcfile = __FILE__,
    .flags   = CLICMDFLAG_PROCINFO,
    MILK_PARAMS_REF(gric_cluster),
    .compute   = gric_cluster_compute,
    .confcheck = gric_fps_custom_conf_check,
};
EXPORT_FPSEXEC(milk, "gric-cluster", runnable_gric_cluster);

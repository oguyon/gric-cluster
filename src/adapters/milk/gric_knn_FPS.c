/**
 * @file gric_knn_FPS.c
 * @brief MILK_RUNNABLE streaming k-NN search unit (CLI + standalone fpsexec).
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "gric_knn_fps_common.h"
#include "gric_knn_fps_params.h"
#include "gric_milk_unit_helpers.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

MILK_PARAMS_DEFINE(gric_knn, GRIC_KNN_FPS_PARAMS);

/**
 * knn_process_step() - Execute one frame of k-NN search and update telemetry.
 * @in_im:       Input query ImageStreamIO stream.
 * @ndim:        Pixels per query frame.
 * @frame_count: 1-based processed query counter.
 * @processinfo: Active PROCESSINFO handle (may be NULL).
 * @out_knn:     Output matches stream [k x 2].
 *
 * Return: 1 to continue loop, 0 if fps_knn_max_frames has been reached.
 */
static int knn_process_step(
    IMAGE       *in_im,
    uint32_t     ndim,
    uint64_t     frame_count,
    PROCESSINFO *processinfo,
    IMAGE       *out_knn)
{
    uint64_t cnt0 = in_im->md[0].cnt0;
    uint32_t slice_idx = (in_im->md[0].naxis > 2) ? (uint32_t)in_im->md[0].cnt1 : 0;
    int typesize = ImageStreamIO_typesize((uint8_t)in_im->md[0].datatype);
    const void *raw_slice = (const char *)in_im->array.raw +
                            ((size_t)slice_idx * ndim * (size_t)typesize);

    struct timespec t_start;
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    gric_knn_fps_process_frame(raw_slice, in_im->md[0].datatype, ndim,
                               cnt0, in_im->md[0].atime, 0.0, out_knn);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e3;

    processinfo_update_output_stream(processinfo, out_knn, in_im);

    long write_slice = (in_im->md[0].naxis > 2) ? (long)in_im->md[0].cnt1 : 0;
    long stream_lag = (in_im->md[0].cnt0 > frame_count)
                          ? (long)(in_im->md[0].cnt0 - frame_count)
                          : 0;
    gric_knn_fps_status_update(cnt0, latency_us, stream_lag, write_slice, (long)slice_idx);

    if (frame_count % 100 == 0)
    {
        processinfo_WriteMessage_fmt(processinfo, "Queries=%lu Latency=%.1fus",
                                     (unsigned long)frame_count, latency_us);
    }

    if (fps_knn_max_frames > 0 && frame_count >= fps_knn_max_frames)
    {
        return 0;
    }

    if (fps_knn_cnt2sync)
    {
        in_im->md[0].cnt2++;
    }
    return 1;
}

/**
 * gric_knn_compute() - Main real-time streaming k-NN runnable loop.
 * @self: Pointer to runnable_gric_knn descriptor.
 *
 * Return: RETURN_SUCCESS on clean exit, RETURN_FAILURE on setup error.
 */
static errno_t gric_knn_compute(
    const MILK_RUNNABLE *self)
{
    if (fps_knn_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input stream specified.\n");
        return RETURN_FAILURE;
    }

    IMGID inimg = stream_connect(fps_knn_in_name);
    if (inimg.ID == -1 || inimg.im == NULL)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_knn_in_name);
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    uint32_t ndim = inimg.im->md[0].size[0] * inimg.im->md[0].size[1];
    if (gric_knn_fps_init_engine(ndim) != 0)
    {
        fprintf(stderr, "Error initializing k-NN engine.\n");
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    IMAGE out_knn;
    memset(&out_knn, 0, sizeof(IMAGE));
    if (gric_knn_fps_init_output_streams(fps_knn_k, &out_knn) != 0)
    {
        fprintf(stderr, "Error creating output stream.\n");
        gric_knn_fps_cleanup_engine();
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    gric_milk_configure_trigger(self, fps_knn_in_name, fps_knn_max_frames);

    MILK_PROCINFO_INIT(self)
    gric_milk_ensure_sem_trigger(processinfo, inimg.im);
    gric_knn_fps_status_init(gric_milk_instance_name("gric_knn", processinfo),
                             fps_knn_shm_status_file);

    uint64_t frame_count = 0;
    MILK_PROCINFO_LOOPSTART(self)
    {
        frame_count++;
        if (knn_process_step(inimg.im, ndim, frame_count, processinfo, &out_knn) == 0)
        {
            processloopOK = 0;
        }
    }
    MILK_PROCINFO_END(self)

    gric_knn_fps_status_close(0);
    gric_knn_fps_close_output_streams(&out_knn);
    gric_knn_fps_cleanup_engine();
    imgid_free(&inimg);

    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_gric_knn = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "gric_knn",
              .cmdkey      = "gric_knn",
              .description = "GRIC real-time streaming k-nearest neighbor search" },
    .srcfile = __FILE__,
    .flags   = CLICMDFLAG_PROCINFO,
    MILK_PARAMS_REF(gric_knn),
    .compute   = gric_knn_compute,
    .confcheck = gric_knn_fps_custom_conf_check,
};
EXPORT_FPSEXEC(milk, "gric-knn", runnable_gric_knn);

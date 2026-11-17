/**
 * @file gric_recon_FPS.c
 * @brief MILK_RUNNABLE streaming k-NN reconstruction unit (CLI + standalone fpsexec).
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "gric_recon_fps_common.h"
#include "gric_recon_fps_params.h"
#include "gric_milk_unit_helpers.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

MILK_PARAMS_DEFINE(gric_reconstruct, GRIC_RECON_FPS_PARAMS);

/**
 * recon_process_step() - Reconstruct one frame from k-NN matches and update telemetry.
 * @in_im:       Input matches ImageStreamIO stream [k x 2].
 * @match_k:     Number of neighbor entries per frame.
 * @match_rows:  Number of rows in matches stream (>= 2).
 * @frame_count: 1-based processed frame counter.
 * @processinfo: Active PROCESSINFO handle (may be NULL).
 * @out_recon:   Output reconstructed stream [b_dim].
 *
 * Return: 1 to continue loop, 0 if fps_recon_max_frames has been reached.
 */
static int recon_process_step(
    IMAGE       *in_im,
    uint32_t     match_k,
    uint32_t     match_rows,
    uint64_t     frame_count,
    PROCESSINFO *processinfo,
    IMAGE       *out_recon)
{
    uint64_t cnt0 = in_im->md[0].cnt0;
    uint32_t slice_idx = (in_im->md[0].naxis > 2) ? (uint32_t)in_im->md[0].cnt1 : 0;
    int typesize = ImageStreamIO_typesize((uint8_t)in_im->md[0].datatype);
    const void *raw_slice = (const char *)in_im->array.raw +
                            ((size_t)slice_idx * match_k * match_rows * (size_t)typesize);

    struct timespec t_start;
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    gric_recon_fps_process_frame(raw_slice, in_im->md[0].datatype, match_k,
                                 cnt0, in_im->md[0].atime, 0.0, out_recon);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e3;

    processinfo_update_output_stream(processinfo, out_recon, in_im);

    long write_slice = (in_im->md[0].naxis > 2) ? (long)in_im->md[0].cnt1 : 0;
    long stream_lag = (in_im->md[0].cnt0 > frame_count)
                          ? (long)(in_im->md[0].cnt0 - frame_count)
                          : 0;
    gric_recon_fps_status_update(cnt0, latency_us, stream_lag, write_slice, (long)slice_idx);

    if (frame_count % 100 == 0)
    {
        processinfo_WriteMessage_fmt(processinfo, "Frames=%lu Latency=%.1fus Var=%.4f",
                                     (unsigned long)frame_count, latency_us,
                                     fps_recon_status_variance);
    }

    if (fps_recon_max_frames > 0 && frame_count >= fps_recon_max_frames)
    {
        return 0;
    }

    if (fps_recon_cnt2sync)
    {
        in_im->md[0].cnt2++;
    }
    return 1;
}

/**
 * gric_recon_compute() - Main real-time streaming reconstruction runnable loop.
 * @self: Pointer to runnable_gric_reconstruct descriptor.
 *
 * Return: RETURN_SUCCESS on clean exit, RETURN_FAILURE on setup error.
 */
static errno_t gric_recon_compute(
    const MILK_RUNNABLE *self)
{
    if (fps_recon_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input matches stream specified.\n");
        return RETURN_FAILURE;
    }

    IMGID inimg = stream_connect(fps_recon_in_name);
    if (inimg.ID == -1 || inimg.im == NULL)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_recon_in_name);
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    uint32_t match_k = inimg.im->md[0].size[0];
    uint32_t match_rows = inimg.im->md[0].size[1];
    if (match_k == 0 || match_rows < 2)
    {
        fprintf(stderr, "Error: Input stream must have dimension [k x 2] (got %u x %u)\n",
                match_k, match_rows);
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    uint32_t b_dim = 0;
    if (gric_recon_fps_init_engine(&b_dim) != 0)
    {
        fprintf(stderr, "Error initializing reconstruction engine with target dataset.\n");
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    IMAGE out_recon;
    memset(&out_recon, 0, sizeof(IMAGE));
    if (gric_recon_fps_init_output_streams(b_dim, &out_recon) != 0)
    {
        fprintf(stderr, "Error creating reconstructed output stream.\n");
        gric_recon_fps_cleanup_engine();
        imgid_free(&inimg);
        return RETURN_FAILURE;
    }

    gric_milk_configure_trigger(self, fps_recon_in_name, fps_recon_max_frames);

    MILK_PROCINFO_INIT(self)
    gric_milk_ensure_sem_trigger(processinfo, inimg.im);
    gric_recon_fps_status_init(gric_milk_instance_name("gric_reconstruct", processinfo),
                               fps_recon_shm_status_file);

    uint64_t frame_count = 0;
    MILK_PROCINFO_LOOPSTART(self)
    {
        frame_count++;
        if (recon_process_step(inimg.im, match_k, match_rows, frame_count,
                               processinfo, &out_recon) == 0)
        {
            processloopOK = 0;
        }
    }
    MILK_PROCINFO_END(self)

    gric_recon_fps_status_close(0);
    gric_recon_fps_close_output_streams(&out_recon);
    gric_recon_fps_cleanup_engine();
    imgid_free(&inimg);

    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_gric_reconstruct = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "gric_reconstruct",
              .cmdkey      = "gric_reconstruct",
              .description = "GRIC real-time streaming k-NN dataset reconstruction" },
    .srcfile = __FILE__,
    .flags   = CLICMDFLAG_PROCINFO,
    MILK_PARAMS_REF(gric_reconstruct),
    .compute   = gric_recon_compute,
    .confcheck = gric_recon_fps_custom_conf_check,
};
EXPORT_FPSEXEC(milk, "gric-reconstruct", runnable_gric_reconstruct);

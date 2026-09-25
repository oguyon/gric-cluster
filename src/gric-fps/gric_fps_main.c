/**
 * @file gric_fps_main.c
 * @brief Standalone FPS daemon executable for GRIC real-time stream clustering.
 */

#include "gric_fps_common.h"
#include "gric_fps_params.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "gric_cluster",
    .cmdkey      = "gric_cluster",
    .description = "GRIC real-time geometric stream clustering"
};

static FPS_CLI_BINDING my_bindings[] = { GRIC_FPS_PARAMS(FPS_X_BINDING) };
static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { GRIC_FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)

/**
 * compute_function() - Main real-time stream clustering loop.
 *
 * Connects to input ImageStreamIO stream, sets up process monitoring via libprocessinfo,
 * creates output streams, and ingests frames sequentially via semaphore triggers.
 *
 * Return: RETURN_SUCCESS on clean exit, non-zero on failure.
 */
static errno_t compute_function(void)
{
    if (fps_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input stream specified.\n");
        return 1;
    }

    IMAGE in_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_in_name, &in_img) != 0)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_in_name);
        return 1;
    }

    uint32_t xsize = in_img.md[0].size[0];
    uint32_t ysize = in_img.md[0].size[1];
    uint32_t ndim = xsize * ysize;
    uint32_t max_clusters = (fps_maxnbclust > 0) ? fps_maxnbclust : 256;

    if (gric_fps_init_engine(ndim) != 0)
    {
        fprintf(stderr, "Error initializing GRIC engine.\n");
        ImageStreamIO_closeIm(&in_img);
        return 1;
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
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_in_name,
            sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
    CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        uint64_t cnt0 = in_img.md[0].cnt0;
        uint32_t slice_idx = (in_img.md[0].naxis > 2) ? (uint32_t)in_img.md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)in_img.md[0].datatype);
        const void *raw_slice = (const char *)in_img.array.raw +
                                ((size_t)slice_idx * ndim * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        gric_fps_process_frame(raw_slice, in_img.md[0].datatype, ndim,
                               cnt0, in_img.md[0].atime, 0.0,
                               &out_assign, &out_anchors, &out_counts);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;
        if (out_assign.used == 1)
        {
            float *assign_data = (float *)out_assign.array.raw;
            assign_data[5] = (float)latency_us;
        }

        processinfo_update_output_stream(processinfo, &out_assign, &in_img);

        static uint64_t msg_cnt = 0;
        if (++msg_cnt % 100 == 0)
        {
            processinfo_WriteMessage_fmt(processinfo, "K=%ld Frames=%lu",
                                         (long)gric_fps_get_cluster_count(),
                                         (unsigned long)msg_cnt);
        }
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    gric_fps_close_output_streams(&out_assign, &out_anchors, &out_counts);
    ImageStreamIO_closeIm(&in_img);
    gric_fps_cleanup_engine();

    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    GRIC_FPS_PARAMS,
    compute_function,
    gric_fps_custom_conf_check)
#endif

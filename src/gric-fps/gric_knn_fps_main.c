/**
 * @file gric_knn_fps_main.c
 * @brief Standalone FPS daemon executable for GRIC real-time streaming k-NN.
 */

#include "gric_knn_fps_common.h"
#include "gric_knn_fps_params.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "gric_knn",
    .cmdkey      = "gric_knn",
    .description = "GRIC real-time streaming k-nearest neighbor search"
};

static FPS_CLI_BINDING my_bindings[] = { GRIC_KNN_FPS_PARAMS(FPS_X_BINDING) };
static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { GRIC_KNN_FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
/**
 * init_procinfo_env - Enable procinfo mode by default for standalone FPS.
 *
 * Sets MILK_FPSPROCINFO=1 in the environment before standalone CLI options are parsed,
 * ensuring all fpsexec commands initialize and run with libprocessinfo enabled.
 */
__attribute__((constructor))
static void init_procinfo_env(void)
{
    setenv("MILK_FPSPROCINFO", "1", 0);
}

CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(knn, CLIcmddata, FPS_app_info)

/**
 * compute_function() - Main real-time streaming k-NN loop.
 *
 * Connects to query ImageStreamIO stream, loads/caches resident reference dataset,
 * creates output k-NN stream [k x 2], and processes frames via semaphore triggers.
 *
 * Return: RETURN_SUCCESS on clean exit, non-zero on failure.
 */
static errno_t compute_function(void)
{
    if (fps_knn_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input stream specified.\n");
        return 1;
    }

    IMAGE in_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_knn_in_name, &in_img) != 0)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_knn_in_name);
        return 1;
    }

    uint32_t xsize = in_img.md[0].size[0];
    uint32_t ysize = in_img.md[0].size[1];
    uint32_t ndim = xsize * ysize;

    if (gric_knn_fps_init_engine(ndim) != 0)
    {
        fprintf(stderr, "Error initializing k-NN engine.\n");
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    IMAGE out_knn;
    memset(&out_knn, 0, sizeof(IMAGE));
    uint64_t frame_count = 0;

    if (gric_knn_fps_init_output_streams(fps_knn_k, &out_knn) != 0)
    {
        fprintf(stderr, "Error creating output stream.\n");
        gric_knn_fps_cleanup_engine();
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_knn_in_name,
            sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
    CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    if (processinfo != NULL)
    {
        processinfo->triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE;
        processinfo_waitoninputstream_init(processinfo, &in_img,
                                           PROCESSINFO_TRIGGERMODE_SEMAPHORE,
                                           -1);
    }

    const char *fps_instance_name = "gric_knn";
    if (dcfpsptr != NULL && dcfpsptr->md != NULL && dcfpsptr->md->name[0] != '\0')
    {
        fps_instance_name = dcfpsptr->md->name;
    }
    else if (processinfo != NULL && processinfo->name[0] != '\0')
    {
        fps_instance_name = processinfo->name;
    }
    gric_knn_fps_status_init(fps_instance_name, fps_knn_shm_status_file);

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        uint64_t cnt0 = in_img.md[0].cnt0;
        uint32_t slice_idx = (in_img.md[0].naxis > 2) ? (uint32_t)in_img.md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)in_img.md[0].datatype);
        const void *raw_slice = (const char *)in_img.array.raw +
                                ((size_t)slice_idx * ndim * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        frame_count++;
        gric_knn_fps_process_frame(raw_slice, in_img.md[0].datatype, ndim,
                                   cnt0, in_img.md[0].atime, 0.0,
                                   &out_knn);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;

        processinfo_update_output_stream(processinfo, &out_knn, &in_img);

        long write_slice = (in_img.md[0].naxis > 2) ? (long)in_img.md[0].cnt1 : 0;
        long read_slice = (long)slice_idx;
        long stream_lag = (in_img.md[0].cnt0 > frame_count)
                              ? (long)(in_img.md[0].cnt0 - frame_count)
                              : 0;
        gric_knn_fps_status_update(cnt0, latency_us, stream_lag, write_slice, read_slice);

        if (frame_count % 100 == 0)
        {
            processinfo_WriteMessage_fmt(processinfo, "Queries=%lu Latency=%.1fus",
                                         (unsigned long)frame_count, latency_us);
        }

        if (fps_knn_max_frames > 0 && frame_count >= fps_knn_max_frames)
        {
            processloopOK = 0;
        }

        if (fps_knn_cnt2sync && processloopOK == 1)
        {
            in_img.md[0].cnt2++;
        }
    } // INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    gric_knn_fps_status_close(0);
    gric_knn_fps_close_output_streams(&out_knn);
    ImageStreamIO_closeIm(&in_img);
    gric_knn_fps_cleanup_engine();

    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    GRIC_KNN_FPS_PARAMS,
    compute_function,
    gric_knn_fps_custom_conf_check)
#endif

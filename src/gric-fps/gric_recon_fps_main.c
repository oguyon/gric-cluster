/**
 * @file gric_recon_fps_main.c
 * @brief Standalone FPS daemon executable for GRIC real-time streaming reconstruction.
 */

#include "gric_recon_fps_common.h"
#include "gric_recon_fps_params.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "gric_reconstruct",
    .cmdkey      = "gric_reconstruct",
    .description = "GRIC real-time streaming k-NN dataset reconstruction"
};

static FPS_CLI_BINDING my_bindings[] = { GRIC_RECON_FPS_PARAMS(FPS_X_BINDING) };
static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { GRIC_RECON_FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
/**
 * init_procinfo_env() - Enable procinfo mode by default for standalone FPS.
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

FPS_CMDSETTINGS_INIT(recon, CLIcmddata, FPS_app_info)

/**
 * compute_function() - Main real-time streaming reconstruction loop.
 *
 * Connects to matches ImageStreamIO stream [k x 2], stages target dataset B,
 * creates output stream D [b_dim], and processes frames via semaphore triggers.
 *
 * Return: RETURN_SUCCESS on clean exit, non-zero on failure.
 */
static errno_t compute_function(void)
{
    if (fps_recon_in_name[0] == '\0')
    {
        fprintf(stderr, "Error: No input matches stream specified.\n");
        return 1;
    }

    IMAGE in_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_recon_in_name, &in_img) != 0)
    {
        fprintf(stderr, "Error connecting to input stream: %s\n", fps_recon_in_name);
        return 1;
    }

    uint32_t match_k = in_img.md[0].size[0];
    uint32_t match_rows = in_img.md[0].size[1];

    if (match_k == 0 || match_rows < 2)
    {
        fprintf(stderr, "Error: Input stream must have dimension [k x 2] (got %u x %u)\n",
                match_k, match_rows);
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    uint32_t b_dim = 0;
    if (gric_recon_fps_init_engine(&b_dim) != 0)
    {
        fprintf(stderr, "Error initializing reconstruction engine with target dataset.\n");
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    IMAGE out_recon;
    memset(&out_recon, 0, sizeof(IMAGE));
    uint64_t frame_count = 0;

    if (gric_recon_fps_init_output_streams(b_dim, &out_recon) != 0)
    {
        fprintf(stderr, "Error creating reconstructed output stream.\n");
        gric_recon_fps_cleanup_engine();
        ImageStreamIO_closeIm(&in_img);
        return 1;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_recon_in_name,
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

    const char *fps_instance_name = "gric_reconstruct";
    if (dcfpsptr != NULL && dcfpsptr->md != NULL && dcfpsptr->md->name[0] != '\0')
    {
        fps_instance_name = dcfpsptr->md->name;
    }
    else if (processinfo != NULL && processinfo->name[0] != '\0')
    {
        fps_instance_name = processinfo->name;
    }
    gric_recon_fps_status_init(fps_instance_name, fps_recon_shm_status_file);

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        uint64_t cnt0 = in_img.md[0].cnt0;
        uint32_t slice_idx = (in_img.md[0].naxis > 2) ? (uint32_t)in_img.md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)in_img.md[0].datatype);
        const void *raw_slice = (const char *)in_img.array.raw +
                                ((size_t)slice_idx * match_k * match_rows * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        frame_count++;
        gric_recon_fps_process_frame(raw_slice, in_img.md[0].datatype, match_k,
                                     cnt0, in_img.md[0].atime, 0.0,
                                     &out_recon);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;

        processinfo_update_output_stream(processinfo, &out_recon, &in_img);

        long write_slice = (in_img.md[0].naxis > 2) ? (long)in_img.md[0].cnt1 : 0;
        long read_slice = (long)slice_idx;
        long stream_lag = (in_img.md[0].cnt0 > frame_count)
                              ? (long)(in_img.md[0].cnt0 - frame_count)
                              : 0;
        gric_recon_fps_status_update(cnt0, latency_us, stream_lag, write_slice, read_slice);

        if (frame_count % 100 == 0)
        {
            processinfo_WriteMessage_fmt(processinfo, "Frames=%lu Latency=%.1fus Var=%.4f",
                                         (unsigned long)frame_count, latency_us,
                                         fps_recon_status_variance);
        }

        if (fps_recon_max_frames > 0 && frame_count >= fps_recon_max_frames)
        {
            if (dcfpsptr != NULL)
            {
                fpsresync_modvar_to_fps(dcfpsptr, dcfpsbindings, dcfpsnbindings);
            }
            processloopOK = 0;
            break;
        }

        if (fps_recon_cnt2sync && processloopOK == 1)
        {
            in_img.md[0].cnt2++;
        }
    } // INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    gric_recon_fps_status_close(0);
    gric_recon_fps_close_output_streams(&out_recon);
    ImageStreamIO_closeIm(&in_img);
    gric_recon_fps_cleanup_engine();

    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2_CONFCHECK(
    FPS_app_info,
    GRIC_RECON_FPS_PARAMS,
    compute_function,
    gric_recon_fps_custom_conf_check)
#endif

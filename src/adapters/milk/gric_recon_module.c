/**
 * @file gric_recon_module.c
 * @brief Milk CLI shared module implementation for GRIC streaming reconstruction.
 */

#include "milk_config.h"
#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
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

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(recon, CLIcmddata, FPS_app_info)

/**
 * compute_function() - CLI compute wrapper for gric_reconstruct.
 *
 * Resolves input image from Milk data directory, initializes output stream,
 * and enters the standard ProcessInfo execution loop.
 *
 * Return: RETURN_SUCCESS on success, RETURN_FAILURE on error.
 */
static errno_t compute_function(void)
{
    if (fps_recon_in_name[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    IMAGE local_in_img;
    memset(&local_in_img, 0, sizeof(IMAGE));
    int local_shm_attached = 0;
    IMAGE *in_im_ptr = NULL;

    IMGID inimg = imgid_make_from_name(fps_recon_in_name);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID != -1 && inimg.im != NULL)
    {
        in_im_ptr = inimg.im;
    }
    else
    {
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_recon_in_name, &local_in_img) == 0)
        {
            in_im_ptr = &local_in_img;
            local_shm_attached = 1;
        }
        else
        {
            return RETURN_FAILURE;
        }
    }

    uint32_t match_k = in_im_ptr->md[0].size[0];
    uint32_t match_rows = in_im_ptr->md[0].size[1];

    if (match_k == 0 || match_rows < 2)
    {
        if (local_shm_attached != 0)
        {
            ImageStreamIO_closeIm(&local_in_img);
        }
        return RETURN_FAILURE;
    }

    uint32_t b_dim = 0;
    if (gric_recon_fps_init_engine(&b_dim) != 0)
    {
        if (local_shm_attached != 0)
        {
            ImageStreamIO_closeIm(&local_in_img);
        }
        return RETURN_FAILURE;
    }

    IMAGE out_recon;
    memset(&out_recon, 0, sizeof(IMAGE));
    uint64_t frame_count = 0;

    if (gric_recon_fps_init_output_streams(b_dim, &out_recon) != 0)
    {
        gric_recon_fps_cleanup_engine();
        if (local_shm_attached != 0)
        {
            ImageStreamIO_closeIm(&local_in_img);
        }
        return RETURN_FAILURE;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_recon_in_name,
            sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
    CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    if (processinfo != NULL && in_im_ptr != NULL)
    {
        processinfo_waitoninputstream_init(processinfo, in_im_ptr,
                                           CLIcmddata.cmdsettings->triggermode,
                                           CLIcmddata.cmdsettings->semindexrequested);
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
        uint64_t cnt0 = in_im_ptr->md[0].cnt0;
        uint32_t slice_idx = (in_im_ptr->md[0].naxis > 2) ? (uint32_t)in_im_ptr->md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)in_im_ptr->md[0].datatype);
        const void *raw_slice = (const char *)in_im_ptr->array.raw +
                                ((size_t)slice_idx * match_k * match_rows * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        frame_count++;
        gric_recon_fps_process_frame(raw_slice, in_im_ptr->md[0].datatype, match_k,
                                     cnt0, in_im_ptr->md[0].atime, 0.0,
                                     &out_recon);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;

        processinfo_update_output_stream(processinfo, &out_recon, in_im_ptr);

        long write_slice = (in_im_ptr->md[0].naxis > 2) ? (long)in_im_ptr->md[0].cnt1 : 0;
        long read_slice = (long)slice_idx;
        long stream_lag = (in_im_ptr->md[0].cnt0 > frame_count)
                              ? (long)(in_im_ptr->md[0].cnt0 - frame_count)
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
            processloopOK = 0;
            break;
        }

        if (fps_recon_cnt2sync && processloopOK == 1)
        {
            in_im_ptr->md[0].cnt2++;
        }
    } // INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    gric_recon_fps_status_close(0);
    gric_recon_fps_close_output_streams(&out_recon);
    gric_recon_fps_cleanup_engine();
    if (local_shm_attached != 0)
    {
        ImageStreamIO_closeIm(&local_in_img);
    }

    return RETURN_SUCCESS;
}

/**
 * CLIfunction() - Standard FPS CLI wrapper function.
 */
static errno_t CLIfunction(void)
{
    if (data.cmdNBarg >= 2 && data.cmdargtoken[1].val.string[0] == '.')
    {
        char fpsname[200];
        snprintf(fpsname, sizeof(fpsname), "_%s", FPS_app_info.fps_name);
        FPS *lfps = fps_local_get_or_create(fpsname, FUNCTION_PARAMETER_NBPARAM_DEFAULT);
        if (lfps != NULL)
        {
            if (lfps->NBparam == 0)
            {
                fps_generic_init(fpsname, &FPS_app_info, my_bindings, nb_bindings, 0);
            }
            dcfpsptr = lfps;
            dcfpsbindings = my_bindings;
            dcfpsnbindings = nb_bindings;
        }

        CLI_checkarg_array(farg, CLIcmddata.nbarg);
        if (lfps != NULL)
        {
            fps_process_cli_and_sync(lfps, farg, my_bindings, nb_bindings);
        }
        return RETURN_SUCCESS;
    }

    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

/**
 * CLIADDCMD_gric_reconstruct() - Register the gric_reconstruct command in Milk CLI.
 *
 * Return: RETURN_SUCCESS on success.
 */
errno_t CLIADDCMD_gric_reconstruct(void)
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    CLIcmddata.FPS_customCONFcheck = gric_recon_fps_custom_conf_check;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

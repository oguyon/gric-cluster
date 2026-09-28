/**
 * @file gric_module.c
 * @brief Milk CLI shared module implementation for GRIC clustering.
 */

#define MODULE_SHORTNAME_DEFAULT "gric"
#define MODULE_DESCRIPTION       "GRIC clustering module"

#include "milk_config.h"
#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
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

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)

/**
 * compute_function() - CLI compute wrapper for gric_cluster.
 *
 * Resolves input image from Milk data directory, initializes output streams,
 * and enters the standard ProcessInfo execution loop.
 *
 * Return: RETURN_SUCCESS on success, RETURN_FAILURE on error.
 */
static errno_t compute_function(void)
{
    if (fps_in_name[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    IMAGE local_in_img;
    memset(&local_in_img, 0, sizeof(IMAGE));
    int local_shm_attached = 0;
    IMAGE *in_im_ptr = NULL;

    IMGID inimg = imgid_make_from_name(fps_in_name);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID != -1 && inimg.im != NULL)
    {
        in_im_ptr = inimg.im;
    }
    else
    {
        if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_in_name, &local_in_img) == 0)
        {
            in_im_ptr = &local_in_img;
            local_shm_attached = 1;
        }
        else
        {
            return RETURN_FAILURE;
        }
    }

    uint32_t xsize = in_im_ptr->md[0].size[0];
    uint32_t ysize = in_im_ptr->md[0].size[1];
    uint32_t ndim = xsize * ysize;
    uint32_t max_clusters = (fps_maxnbclust > 0) ? fps_maxnbclust : 256;

    if (gric_fps_init_engine(ndim) != 0)
    {
        if (local_shm_attached != 0)
        {
            ImageStreamIO_closeIm(&local_in_img);
        }
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
    uint64_t frame_count = 0;

    if (gric_fps_init_output_streams(xsize, ysize, max_clusters,
                                     &out_assign, &out_anchors, &out_counts) != 0)
    {
        gric_fps_cleanup_engine();
        if (local_shm_attached != 0)
        {
            ImageStreamIO_closeIm(&local_in_img);
        }
        return RETURN_FAILURE;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_in_name,
            sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
    CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    if (processinfo != NULL && in_im_ptr != NULL)
    {
        processinfo_waitoninputstream_init(processinfo, in_im_ptr,
                                           CLIcmddata.cmdsettings->triggermode,
                                           CLIcmddata.cmdsettings->semindexrequested);
    }

    const char *fps_instance_name = "gric_cluster";
    if (dcfpsptr != NULL && dcfpsptr->md != NULL && dcfpsptr->md->name[0] != '\0')
    {
        fps_instance_name = dcfpsptr->md->name;
    }
    else if (processinfo != NULL && processinfo->name[0] != '\0')
    {
        fps_instance_name = processinfo->name;
    }
    gric_fps_status_init(fps_instance_name, fps_shm_status_file);

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        uint64_t cnt0 = in_im_ptr->md[0].cnt0;
        uint32_t slice_idx = (in_im_ptr->md[0].naxis > 2) ? (uint32_t)in_im_ptr->md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)in_im_ptr->md[0].datatype);
        const void *raw_slice = (const char *)in_im_ptr->array.raw +
                                ((size_t)slice_idx * ndim * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        frame_count++;
        gric_fps_process_frame(raw_slice, in_im_ptr->md[0].datatype, ndim,
                               cnt0, in_im_ptr->md[0].atime, 0.0,
                               &out_assign, &out_anchors, &out_counts);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;
        if (out_assign.used == 1)
        {
            float *assign_data = (float *)out_assign.array.raw;
            assign_data[5] = (float)latency_us;
        }

        processinfo_update_output_stream(processinfo, &out_assign, in_im_ptr);

        long write_slice = (in_im_ptr->md[0].naxis > 2) ? (long)in_im_ptr->md[0].cnt1 : 0;
        long read_slice = (long)slice_idx;
        long stream_lag = (in_im_ptr->md[0].cnt0 > frame_count)
                              ? (long)(in_im_ptr->md[0].cnt0 - frame_count)
                              : 0;
        gric_fps_status_update(cnt0, latency_us, stream_lag, write_slice, read_slice);

        if (frame_count % 100 == 0)
        {
            processinfo_WriteMessage_fmt(processinfo, "K=%ld Frames=%lu",
                                         (long)gric_fps_get_cluster_count(),
                                         (unsigned long)frame_count);
        }

        if (fps_max_frames > 0 && frame_count >= fps_max_frames)
        {
            processloopOK = 0;
        }

        if (fps_cnt2sync && processloopOK == 1)
        {
            in_im_ptr->md[0].cnt2++;
        }
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    gric_fps_status_close(0);
    gric_fps_close_output_streams(&out_assign, &out_anchors, &out_counts);
    if (fps_save_dir[0] != '\0')
    {
        gric_fps_save_results(fps_save_dir);
    }
    gric_fps_cleanup_engine();
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
 * CLIADDCMD_gric_cluster() - Register the gric_cluster command in Milk CLI.
 *
 * Return: RETURN_SUCCESS on success.
 */
errno_t CLIADDCMD_gric_cluster(void)
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    CLIcmddata.FPS_customCONFcheck = gric_fps_custom_conf_check;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

errno_t CLIADDCMD_gric_knn(void);
errno_t CLIADDCMD_gric_reconstruct(void);

/**
 * init_module_CLI() - Module initializer function called by MILK_MODULE.
 *
 * Return: RETURN_SUCCESS on success.
 */
static errno_t init_module_CLI(void)
{
    CLIADDCMD_gric_cluster();
    CLIADDCMD_gric_knn();
    CLIADDCMD_gric_reconstruct();
    return RETURN_SUCCESS;
}

MILK_MODULE(milkgric, init_module_CLI, NULL);

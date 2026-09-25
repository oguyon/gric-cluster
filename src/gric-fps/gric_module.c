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

    IMGID inimg = imgid_make_from_name(fps_in_name);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t xsize = inimg.md->size[0];
    uint32_t ysize = inimg.md->size[1];
    uint32_t ndim = xsize * ysize;
    uint32_t max_clusters = (fps_maxnbclust > 0) ? fps_maxnbclust : 256;

    if (gric_fps_init_engine(ndim) != 0)
    {
        return RETURN_FAILURE;
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
        gric_fps_cleanup_engine();
        return RETURN_FAILURE;
    }

    strncpy(CLIcmddata.cmdsettings->triggerstreamname, fps_in_name,
            sizeof(CLIcmddata.cmdsettings->triggerstreamname) - 1);
    CLIcmddata.cmdsettings->flags |= CLICMDFLAG_PROCINFO;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    if (processinfo != NULL && inimg.im != NULL)
    {
        processinfo_waitoninputstream_init(processinfo, inimg.im,
                                           CLIcmddata.cmdsettings->triggermode,
                                           CLIcmddata.cmdsettings->semindexrequested);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        uint64_t cnt0 = inimg.im->md[0].cnt0;
        uint32_t slice_idx = (inimg.im->md[0].naxis > 2) ? (uint32_t)inimg.im->md[0].cnt1 : 0;
        int typesize = ImageStreamIO_typesize((uint8_t)inimg.im->md[0].datatype);
        const void *raw_slice = (const char *)inimg.im->array.raw +
                                ((size_t)slice_idx * ndim * (size_t)typesize);

        struct timespec t_start;
        struct timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        gric_fps_process_frame(raw_slice, inimg.im->md[0].datatype, ndim,
                               cnt0, inimg.im->md[0].atime, 0.0,
                               &out_assign, &out_anchors, &out_counts);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double latency_us = (t_end.tv_sec - t_start.tv_sec) * 1e6 +
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e3;
        if (out_assign.used == 1)
        {
            float *assign_data = (float *)out_assign.array.raw;
            assign_data[5] = (float)latency_us;
        }

        processinfo_update_output_stream(processinfo, &out_assign, inimg.im);

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
    gric_fps_cleanup_engine();

    return RETURN_SUCCESS;
}

/**
 * CLIfunction() - Standard FPS CLI wrapper function.
 */
static errno_t CLIfunction(void)
{
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

/**
 * init_module_CLI() - Module initializer function called by MILK_MODULE.
 *
 * Return: RETURN_SUCCESS on success.
 */
static errno_t init_module_CLI(void)
{
    CLIADDCMD_gric_cluster();
    return RETURN_SUCCESS;
}

MILK_MODULE(milkgric, init_module_CLI, NULL);

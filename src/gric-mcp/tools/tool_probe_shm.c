/**
 * @file tool_probe_shm.c
 * @brief Non-blocking ImageStreamIO shared-memory stream prober and telemetry inspector.
 */

#include "mcp_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

int mcp_tool_probe_shm(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *name_item = cJSON_GetObjectItemCaseSensitive(args, "stream_name");
    if (name_item == NULL || !cJSON_IsString(name_item))
    {
        cJSON_AddStringToObject(res, "error", "stream_name is required");
        return -1;
    }
    const char *stream_name = name_item->valuestring;

#ifdef USE_IMAGESTREAMIO
    IMAGE image;
    memset(&image, 0, sizeof(IMAGE));

    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &image) != 0)
    {
        cJSON_AddStringToObject(res, "stream_name", stream_name);
        cJSON_AddStringToObject(res, "status", "NOT_FOUND");
        cJSON_AddStringToObject(res, "error", "Could not connect to shared memory stream");
        return -1;
    }

    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "status", "CONNECTED");
    cJSON_AddNumberToObject(res, "naxis", (double)image.md[0].naxis);
    cJSON_AddNumberToObject(res, "size0", (double)image.md[0].size[0]);
    cJSON_AddNumberToObject(res, "size1", (double)image.md[0].size[1]);
    cJSON_AddNumberToObject(
        res, "size2",
        (image.md[0].naxis > 2) ? (double)image.md[0].size[2] : 1.0);
    cJSON_AddNumberToObject(res, "datatype", (double)image.md[0].datatype);
    cJSON_AddNumberToObject(res, "write_cnt0", (double)image.md[0].cnt0);
    cJSON_AddNumberToObject(res, "write_slice_cnt1", (double)image.md[0].cnt1);
    cJSON_AddNumberToObject(res, "active_semaphores", (double)image.md[0].sem);

    ImageStreamIO_closeIm(&image);
    return 0;
#else
    cJSON_AddStringToObject(res, "stream_name", stream_name);
    cJSON_AddStringToObject(res, "status", "DISABLED");
    cJSON_AddStringToObject(res, "error", "ImageStreamIO support not compiled into this build");
    return -1;
#endif
} // mcp_tool_probe_shm

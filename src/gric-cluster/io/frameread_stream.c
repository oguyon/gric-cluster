/**
 * @file frameread_stream.c
 * @brief ImageStreamIO format reader implementation.
 */

#include "frameread_internal.h"

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#include <errno.h>
#include <limits.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * init_stream() - Initialize the ImageStreamIO shared memory frame reader.
 * @stream_name: Name of the shared memory stream to connect to.
 *
 * Connects to the shared memory stream, retrieves its dimensions, initializes the stream
 * counters, and sets up real-time wait tracking variables.
 *
 * Return: 0 on success, or -1 on failure.
 */
int init_stream(
    char *stream_name)
{
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &stream_image) != 0)
    {
        fprintf(stderr, "Error connecting to stream %s\n", stream_name);
        return -1;
    }

    frame_width = stream_image.md[0].size[0];
    frame_height = stream_image.md[0].size[1];

    if (stream_image.md[0].naxis > 2)
    {
        stream_depth = stream_image.md[0].size[2];
        is_3d = 1;
    }
    else
    {
        stream_depth = 1;
        is_3d = 0;
    }

    num_frames = LONG_MAX; /* Stream is effectively infinite */
    is_stream_mode = 1;

    /* Initialize state to current stream head */
    last_cnt0 = stream_image.md[0].cnt0;
    current_read_slice = stream_image.md[0].cnt1;
    current_write_slice = stream_image.md[0].cnt1;
    first_stream_frame = 1;
    stream_read_counter = 0;

    printf("Connected to stream %s (%ld x %ld x %ld)\n", stream_name, frame_width,
           frame_height, stream_depth);

    return 0;
}

/**
 * getframe_stream() - Retrieve a frame from the shared memory stream.
 * @frame_struct: Pointer to the Frame struct to populate.
 * @index:        Zero-based index of the frame to retrieve.
 *
 * Checks stream synchronization, waits via sem_timedwait if the stream has no new frames,
 * updates stream lag statistics, checks for circular buffer overrun, and parses data into
 * the frame buffer based on the stream data type.
 *
 * Return: 0 on success, or -1 on failure.
 */
int getframe_stream(
    Frame *frame_struct,
    long   index)
{
    long nelements = frame_width * frame_height;

    /* Prevent random access / rewinding in stream mode */
    if (index != stream_read_counter)
    {
        return -1;
    }

    if (cnt2sync_enabled)
    {
        stream_image.md[0].cnt2++;
    }

    /* Wait for new data if we caught up */
    while (stream_image.md[0].cnt0 <= last_cnt0)
    {
        struct timespec t0;
        struct timespec t1;
        struct timespec ts;
        int ret;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;

        ret = sem_timedwait(stream_image.semptr[0], &ts);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        cumulative_wait_time_sec += (t1.tv_sec - t0.tv_sec) +
                                    (t1.tv_nsec - t0.tv_nsec) / 1e9;

        if (ret == -1)
        {
            if (errno == ETIMEDOUT)
            {
                fprintf(stderr, "Stream timeout (1s). Ending.\n");
                return -1;
            }
            if (errno == EINTR)
            {
                continue;
            }
            perror("sem_timedwait");
            return -1;
        }
    }

    /* We process one frame forward */
    last_cnt0++;
    stream_read_counter++;

    frame_struct->cnt0 = stream_image.md[0].cnt0;
    frame_struct->atime = stream_image.md[0].atime;

    if (is_3d)
    {
        current_read_slice = (current_read_slice + 1) % stream_depth;
    }
    else
    {
        /* In 2D stream, data is always at 0 (or updated in place) */
        current_read_slice = 0;
    }

    /* Update stats */
    {
        uint64_t actual_stream_cnt0 = stream_image.md[0].cnt0;
        current_write_slice = stream_image.md[0].cnt1;

        /* Check for circular buffer overrun */
        if (is_3d)
        {
            long lag = (long)(actual_stream_cnt0 - last_cnt0);
            if (lag >= stream_depth)
            {
                fprintf(stderr,
                        "\nError: Circular buffer overrun. Lag (%ld) exceeds depth (%ld). "
                        "Stopping.\n", lag, stream_depth);
                return -1;
            }
        }
    }

    {
        /* Pointer offset */
        long offset = current_read_slice * nelements;
        int dtype = stream_image.md[0].datatype;

#define _DATATYPE_UINT8 1
#define _DATATYPE_INT8 2
#define _DATATYPE_UINT16 3
#define _DATATYPE_INT16 4
#define _DATATYPE_UINT32 5
#define _DATATYPE_INT32 6
#define _DATATYPE_UINT64 7
#define _DATATYPE_INT64 8
#define _DATATYPE_FLOAT 9
#define _DATATYPE_DOUBLE 10

        switch (dtype)
        {
        case _DATATYPE_FLOAT:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((float *)stream_image.array.F)[offset + ii];
            }
            break;
        case _DATATYPE_DOUBLE:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    ((double *)stream_image.array.D)[offset + ii];
            }
            break;
        case _DATATYPE_UINT8:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((uint8_t *)stream_image.array.UI8)[offset + ii];
            }
            break;
        case _DATATYPE_UINT16:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((uint16_t *)stream_image.array.UI16)[offset + ii];
            }
            break;
        case _DATATYPE_INT16:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((int16_t *)stream_image.array.SI16)[offset + ii];
            }
            break;
        case _DATATYPE_UINT32:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((uint32_t *)stream_image.array.UI32)[offset + ii];
            }
            break;
        case _DATATYPE_INT32:
            for (long ii = 0; ii < nelements; ii++)
            {
                frame_struct->data[ii] =
                    (double)((int32_t *)stream_image.array.SI32)[offset + ii];
            }
            break;
        default:
            fprintf(stderr, "Unsupported stream datatype: %d\n", dtype);
            return -1;
        }
    }

    return 0;
}

/**
 * close_stream() - Close resources associated with the ImageStreamIO frame reader.
 */
void close_stream(void)
{
    is_stream_mode = 0;
}

/**
 * reset_stream() - Reset the ImageStreamIO frame reader.
 */
void reset_stream(void)
{
    /* Stream mode is real-time and doesn't support rewinding */
}

#endif // USE_IMAGESTREAMIO

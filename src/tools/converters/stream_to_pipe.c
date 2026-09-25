/**
 * @file stream_to_pipe.c
 * @brief ImageStreamIO tap and stdout redirection utility.
 *
 * Connects to a live ImageStreamIO shared memory buffer and pipes its raw image data
 * frames directly to stdout.
 *
 * Main Functions:
 * - main: Entry point of the stream piping tool.
 */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "cli_colors.h"

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

static volatile sig_atomic_t stop = 0;

/**
 * handle_signal - Signal handler for graceful termination
 * @sig: Signal number received
 *
 * Sets the stop flag to notify main processing loop to exit cleanly.
 */
static void handle_signal(
    int sig)
{
    (void)sig;
    stop = 1;
}



static void print_usage(const char *progname)
{
    fprintf(stderr, "Usage: %s <stream_name> [max_frames]\n", progname);
} // print_usage



static void print_help(const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-stream-to-pipe%s - Pipes raw ImageStreamIO stream data to stdout\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s<stream_name>%s %s[max_frames]%s\n\n", ansi_bold_green, progname, ansi_reset,
           ansi_color_magenta, ansi_reset, ansi_color_grey, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Pipes raw floating-point data from an ImageStreamIO stream directly to stdout.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-h, --help%s           Show this help message\n\n", ansi_color_green, ansi_reset);
    printf("  Arguments:\n");
    printf("    %s<stream_name>%s      Name of the ImageStreamIO stream\n", ansi_color_magenta,
           ansi_reset);
    printf("    %s[max_frames]%s       Optional: Limit output to N frames\n\n", ansi_color_grey,
           ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s mystream 500\n", ansi_color_grey, ansi_reset, ansi_bold_green, progname,
           ansi_reset);
    cli_print_color_mode();
} // print_help

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    // Check for help option early
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

#ifndef USE_IMAGESTREAMIO
    fprintf(stderr, "Error: ImageStreamIO support not compiled in.\n");
    return 1;
#else
    if (argc < 2)
    {
        fprintf(stderr, "Error: Missing required arguments.\n");
        print_usage(argv[0]);
        return 1;
    }

    char *stream_name = argv[1];
    long max_frames = -1;
    if (argc > 2)
    {
        max_frames = atol(argv[2]);
    }

    IMAGE stream_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &stream_image) != 0)
    {
        fprintf(stderr, "Error connecting to stream %s\n", stream_name);
        return 1;
    }

    long width = stream_image.md[0].size[0];
    long height = (stream_image.md[0].naxis > 1) ? (long)stream_image.md[0].size[1] : 1L;
    long nelements = width * height;

    if (nelements <= 0)
    {
        fprintf(stderr, "Error: invalid stream dimensions %ldx%ld\n", width, height);
        ImageStreamIO_closeIm(&stream_image);
        return 1;
    }

    // Output metadata to stderr for user/script info
    fprintf(stderr, "Connected: %s (%ldx%ld), Type: %d\n", stream_name, width, height,
            stream_image.md[0].datatype);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* Clear SA_RESTART so blocking syscalls return EINTR */

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGPIPE, &sa, NULL);

    int sem_idx = ImageStreamIO_getsemwaitindex(&stream_image, -1);
    if (sem_idx < 0)
    {
        sem_idx = 0;
    }

    // Buffer for one frame (doubles)
    double *buffer = (double *)malloc((size_t)nelements * sizeof(double));
    if (!buffer)
    {
        if (sem_idx >= 0 && stream_image.semReadPID && sem_idx < stream_image.md[0].sem)
        {
            stream_image.semReadPID[sem_idx] = 0;
        }
        ImageStreamIO_closeIm(&stream_image);
        return 1;
    }

    uint64_t last_cnt0 = stream_image.md[0].cnt0;
    long processed = 0;

    while (!stop && (max_frames < 0 || processed < max_frames))
    {
        // Wait for next frame
        while (stream_image.md[0].cnt0 <= last_cnt0 && !stop)
        {
            struct timespec ts;

            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100000000; /* 100 ms timeout */
            if (ts.tv_nsec >= 1000000000)
            {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000;
            }

            int ret = ImageStreamIO_semtimedwait(&stream_image, sem_idx, &ts);
            if (ret != 0)
            {
                if (errno == ETIMEDOUT || errno == EINTR)
                {
                    continue;
                }
                if (!stop)
                {
                    fprintf(stderr, "Semwait failed: %s\n", strerror(errno));
                }
                break;
            }
        } // while (waiting for next frame)
        if (stop)
        {
            break;
        }

        last_cnt0++;

        long current_read_slice = 0;
        if (stream_image.md[0].naxis > 2)
        {
            long depth = stream_image.md[0].size[2];
            static long slice_idx = -1;
            if (slice_idx == -1)
            {
                slice_idx = stream_image.md[0].cnt1;
            }
            else
            {
                slice_idx = (slice_idx + 1) % depth;
            }

            current_read_slice = slice_idx;
        }

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
                buffer[ii] = (double)((float *)stream_image.array.F)[offset + ii];
            }
            break;
        case _DATATYPE_DOUBLE:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = ((double *)stream_image.array.D)[offset + ii];
            }
            break;
        case _DATATYPE_UINT8:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = (double)((uint8_t *)stream_image.array.UI8)[offset + ii];
            }
            break;
        case _DATATYPE_UINT16:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = (double)((uint16_t *)stream_image.array.UI16)[offset + ii];
            }
            break;
        case _DATATYPE_INT16:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = (double)((int16_t *)stream_image.array.SI16)[offset + ii];
            }
            break;
        case _DATATYPE_UINT32:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = (double)((uint32_t *)stream_image.array.UI32)[offset + ii];
            }
            break;
        case _DATATYPE_INT32:
            for (long ii = 0; ii < nelements; ii++)
            {
                buffer[ii] = (double)((int32_t *)stream_image.array.SI32)[offset + ii];
            }
            break;
        default:
            break;
        }

        // Write raw bytes to stdout
        size_t written = fwrite(buffer, sizeof(double), (size_t)nelements, stdout);
        if (written != (size_t)nelements)
        {
            break;
        }
        fflush(stdout);

        processed++;
    } // while (!stop && ...)

    free(buffer);
    if (sem_idx >= 0 && stream_image.semReadPID && sem_idx < stream_image.md[0].sem)
    {
        stream_image.semReadPID[sem_idx] = 0;
    }
    ImageStreamIO_closeIm(&stream_image);
    return 0;
#endif
}

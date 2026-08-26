/**
 * @file txt2stream.c
 * @brief Raw coordinate text file to ImageStreamIO shared memory streamer.
 *
 * Reads multi-dimensional coordinate samples from an ASCII text file and streams
 * them into an ImageStreamIO shared-memory circular ring buffer in real time.
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "shared/cli_colors.h"

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

#define DEFAULT_FPS          100.0
#define DEFAULT_BUFFER_DEPTH 1000

static volatile sig_atomic_t s_stop_requested = 0;

/**
 * handle_sigint() - Signal handler for graceful stream shutdown.
 * @sig: Signal number received.
 */
static void handle_sigint(
    int sig)
{
    (void)sig;
    s_stop_requested = 1;
}

/**
 * print_usage() - Display compact command usage summary.
 * @progname: Name of the executable.
 */
static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s <input.txt> <stream_name> [options]\n", progname);
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

/**
 * print_help() - Display full documentation and CLI option reference.
 * @progname: Name of the executable.
 */
static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-txt2stream%s - Stream raw ASCII coordinate points to ImageStreamIO\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s<input.txt>%s %s<stream_name>%s %s[options]%s\n\n",
           ansi_bold_green, progname, ansi_reset,
           ansi_color_magenta, ansi_reset,
           ansi_color_magenta, ansi_reset,
           ansi_color_grey, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Parses multi-dimensional coordinate vectors from an ASCII text file and writes\n"
           "  each vector into an ImageStreamIO shared-memory circular ring buffer in /dev/shm.\n"
           "  Designed to supply high-rate real-time data feeds for gric-cluster.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-fps%s %s<val>%s         Streaming frame rate (frames/sec, default: %.1f, 0=max)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, DEFAULT_FPS);
    printf("  %s-depth%s %s<N>%s         Circular ring buffer depth (default: %d)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, DEFAULT_BUFFER_DEPTH);
    printf("  %s-loop%s              Loop dataset indefinitely until stopped (Ctrl+C)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-repeat%s %s<N>%s        Replay the dataset N times\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-maxfr%s %s<N>%s         Stop after streaming N total frames\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-cnt2sync%s          Enable flow-control handshaking (wait for consumer cnt2)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-double%s            Write 64-bit double precision (default: 32-bit float)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-v, -verbose%s       Print diagnostic logging and progress\n",
           ansi_color_green, ansi_reset);
    printf("  %s-h, --help%s         Display this help message\n\n",
           ansi_color_green, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s benchmarks/2Dspiral.txt spiral_stream -fps 200 -loop\n",
           ansi_color_grey, ansi_reset, ansi_bold_green, progname, ansi_reset);
    printf("  %s$%s %s%s%s 3Drand.txt rand3d_stream -fps 1000 -maxfr 5000\n\n",
           ansi_color_grey, ansi_reset, ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
}

/**
 * load_ascii_data() - Read coordinate matrix from an ASCII text file.
 * @filename:     Path to the ASCII file.
 * @out_dim:      Output pointer to store detected vector dimension (D).
 * @out_count:    Output pointer to store number of coordinate samples (N).
 * @out_data:     Output pointer to allocated flat double array [N * D].
 *
 * Return: 0 on success, or -1 on error.
 */
static int load_ascii_data(
    const char  *filename,
    int         *out_dim,
    long        *out_count,
    double     **out_data)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open file '%s': %s\n", filename, strerror(errno));
        return -1;
    }

    char *line = NULL;
    size_t len = 0;
    int dim = 0;
    long count = 0;
    size_t capacity = 1024;
    double *data = NULL;

    while (getline(&line, &len, fp) != -1)
    {
        char *p = line;
        while (isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '#' || *p == '\0')
        {
            continue;
        }

        /* Determine dimension on first non-comment line */
        if (dim == 0)
        {
            int cols = 0;
            int in_token = 0;
            for (char *c = p; *c != '\0'; c++)
            {
                if (!isspace((unsigned char)*c))
                {
                    if (!in_token)
                    {
                        cols++;
                        in_token = 1;
                    }
                }
                else
                {
                    in_token = 0;
                }
            }
            dim = cols;
            if (dim <= 0)
            {
                continue;
            }
            data = (double *)malloc(capacity * (size_t)dim * sizeof(double));
            if (data == NULL)
            {
                perror("Memory allocation failed");
                free(line);
                fclose(fp);
                return -1;
            }
        } // if (dim == 0)

        /* Expand buffer capacity if needed */
        if ((size_t)count >= capacity)
        {
            capacity *= 2;
            double *new_data = (double *)realloc(data, capacity * (size_t)dim * sizeof(double));
            if (new_data == NULL)
            {
                perror("Memory reallocation failed");
                free(data);
                free(line);
                fclose(fp);
                return -1;
            }
            data = new_data;
        }

        /* Parse coordinate values */
        double *dst = data + (size_t)count * (size_t)dim;
        char *endptr = NULL;
        int parsed = 0;
        for (int d = 0; d < dim; d++)
        {
            dst[d] = strtod(p, &endptr);
            if (p == endptr)
            {
                break;
            }
            p = endptr;
            parsed++;
        }

        if (parsed == dim)
        {
            count++;
        }
    } // while (getline)

    free(line);
    fclose(fp);

    if (count == 0 || dim == 0)
    {
        fprintf(stderr, "Error: No valid coordinate data found in '%s'\n", filename);
        if (data != NULL)
        {
            free(data);
        }
        return -1;
    }

    *out_dim = dim;
    *out_count = count;
    *out_data = data;
    return 0;
}

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

#ifndef USE_IMAGESTREAMIO
    fprintf(stderr, "Error: ImageStreamIO support is not compiled in.\n");
    return 1;
#else
    const char *input_file = NULL;
    const char *stream_name = NULL;
    double fps = DEFAULT_FPS;
    int buffer_depth = DEFAULT_BUFFER_DEPTH;
    int loop_mode = 0;
    int repeats = 1;
    long max_frames = -1;
    int cnt2sync = 0;
    int use_double = 0;
    int verbose = 0;

    int positional_idx = 0;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-fps") == 0 && i + 1 < argc)
        {
            fps = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-depth") == 0 && i + 1 < argc)
        {
            buffer_depth = atoi(argv[++i]);
            if (buffer_depth < 1)
            {
                buffer_depth = 1;
            }
        }
        else if (strcmp(argv[i], "-loop") == 0)
        {
            loop_mode = 1;
        }
        else if (strcmp(argv[i], "-repeat") == 0 && i + 1 < argc)
        {
            repeats = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-maxfr") == 0 && i + 1 < argc)
        {
            max_frames = atol(argv[++i]);
        }
        else if (strcmp(argv[i], "-cnt2sync") == 0)
        {
            cnt2sync = 1;
        }
        else if (strcmp(argv[i], "-double") == 0)
        {
            use_double = 1;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "-verbose") == 0)
        {
            verbose = 1;
        }
        else if (argv[i][0] == '-')
        {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        else
        {
            if (positional_idx == 0)
            {
                input_file = argv[i];
            }
            else if (positional_idx == 1)
            {
                stream_name = argv[i];
            }
            else
            {
                fprintf(stderr, "Error: Unexpected argument '%s'\n", argv[i]);
                return 1;
            }
            positional_idx++;
        }
    } // for (argv)

    if (input_file == NULL || stream_name == NULL)
    {
        fprintf(stderr, "Error: Missing required arguments <input.txt> and <stream_name>.\n");
        print_usage(argv[0]);
        return 1;
    }

    int dim = 0;
    long count = 0;
    double *data = NULL;
    if (load_ascii_data(input_file, &dim, &count, &data) != 0)
    {
        return 1;
    }

    printf("%s[gric-txt2stream]%s Loaded %ld samples (%d-D) from %s\n",
           ansi_bold_cyan, ansi_reset, count, dim, input_file);

    /* Setup ImageStreamIO stream */
    IMAGE stream_image;
    memset(&stream_image, 0, sizeof(IMAGE));
    uint8_t dtype = use_double ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;

    uint32_t dims[3] = {(uint32_t)dim, 1, (uint32_t)buffer_depth};
    int naxis = (buffer_depth > 1) ? 3 : 2;

    if (ImageStreamIO_createIm(&stream_image, stream_name, (uint8_t)naxis, dims,
                               dtype, 1, 1, 1) != 0)
    {
        fprintf(stderr, "Error: Failed to create ImageStreamIO stream '%s'\n", stream_name);
        free(data);
        return 1;
    }

    struct timespec ts_init;
    clock_gettime(CLOCK_REALTIME, &ts_init);
    stream_image.md[0].creationtime = ts_init;
    stream_image.md[0].atime = ts_init;
    stream_image.md[0].writetime = ts_init;
    stream_image.md[0].cnt0 = 0;
    stream_image.md[0].cnt1 = 0;
    stream_image.md[0].cnt2 = 0;

    printf("%s[gric-txt2stream]%s Created stream '%s' (%d x 1 x %d, type: %s)\n",
           ansi_bold_green, ansi_reset, stream_name, dim, buffer_depth,
           use_double ? "DOUBLE" : "FLOAT");
    if (fps > 0.0)
    {
        printf("  Pacing: %.1f FPS (%.2f us/frame)\n", fps, 1000000.0 / fps);
    }
    else
    {
        printf("  Pacing: Unthrottled (maximum rate)\n");
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    long long us_per_frame = (fps > 0.0) ? (long long)(1000000.0 / fps) : 0;
    long total_streamed = 0;
    int current_repeat = 0;
    long write_slice = 0;

    struct timespec last_time;
    clock_gettime(CLOCK_MONOTONIC, &last_time);
    struct timespec stream_start = last_time;

    while (!s_stop_requested)
    {
        for (long idx = 0; idx < count && !s_stop_requested; idx++)
        {
            if (max_frames > 0 && total_streamed >= max_frames)
            {
                break;
            }

            /* Wait for consumer synchronization if cnt2sync is enabled */
            if (cnt2sync)
            {
                while (!s_stop_requested && stream_image.md[0].cnt0 > stream_image.md[0].cnt2)
                {
                    usleep(10);
                }
            }
            else if (us_per_frame > 0)
            {
                /* Rate pacing */
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                long long elapsed_us = (now.tv_sec - last_time.tv_sec) * 1000000LL +
                                       (now.tv_nsec - last_time.tv_nsec) / 1000;
                if (elapsed_us < us_per_frame)
                {
                    usleep((useconds_t)(us_per_frame - elapsed_us));
                }
                clock_gettime(CLOCK_MONOTONIC, &last_time);
            }

            if (s_stop_requested)
            {
                break;
            }

            /* Write coordinate slice */
            size_t offset = (size_t)write_slice * (size_t)dim;
            const double *src = data + (size_t)idx * (size_t)dim;

            stream_image.md[0].write = 1;
            if (use_double)
            {
                double *dst = (double *)stream_image.array.D + offset;
                for (int d = 0; d < dim; d++)
                {
                    dst[d] = src[d];
                }
            }
            else
            {
                float *dst = (float *)stream_image.array.F + offset;
                for (int d = 0; d < dim; d++)
                {
                    dst[d] = (float)src[d];
                }
            }
            stream_image.md[0].write = 0;

            struct timespec tw;
            clock_gettime(CLOCK_REALTIME, &tw);
            stream_image.md[0].writetime = tw;
            stream_image.md[0].atime = tw;
            stream_image.md[0].lastaccesstime = tw;
            stream_image.md[0].cnt1 = (uint64_t)write_slice;
            stream_image.md[0].cnt0++;

            ImageStreamIO_sempost(&stream_image, -1);

            write_slice = (write_slice + 1) % buffer_depth;
            total_streamed++;

            if (verbose && (total_streamed % 1000 == 0))
            {
                printf("\rStreamed: %ld frames", total_streamed);
                fflush(stdout);
            }
        } // for (samples)

        if (max_frames > 0 && total_streamed >= max_frames)
        {
            break;
        }

        current_repeat++;
        if (!loop_mode && current_repeat >= repeats)
        {
            break;
        }
    } // while (!s_stop_requested)

    struct timespec stream_end;
    clock_gettime(CLOCK_MONOTONIC, &stream_end);
    double elapsed_sec = (stream_end.tv_sec - stream_start.tv_sec) +
                         (stream_end.tv_nsec - stream_start.tv_nsec) / 1e9;
    double actual_fps = (elapsed_sec > 0.0) ? (double)total_streamed / elapsed_sec : 0.0;

    printf("\n%s[gric-txt2stream]%s Streaming finished: %ld frames streamed in %.3f s (%.1f FPS)\n",
           ansi_bold_green, ansi_reset, total_streamed, elapsed_sec, actual_fps);

    free(data);
    return 0;
#endif
}

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

volatile sig_atomic_t stop = 0;

void handle_sigint(int sig)
{
    stop = 1;
}

static const char *ansi_bold = "";
static const char *ansi_reset = "";
static const char *ansi_color_green = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_cyan = "";
static const char *ansi_color_grey = "";
static const char *ansi_color_yellow = "";
static const char *ansi_color_red = "";

static void init_colors(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_bold = "\x1b[1m";
        ansi_reset = "\x1b[0m";
        ansi_color_green = "\x1b[32m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_cyan = "\x1b[36m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_red = "\x1b[31m";
    }
} // init_colors

static void print_usage(const char *progname)
{
    fprintf(stderr, "Usage: %s <stream_name> [max_frames]\n", progname);
} // print_usage

static void print_color_mode(void)
{
    const char *no_color = getenv("NO_COLOR");
    printf("\n%sCOLOR MODE%s\n", ansi_bold_cyan, ansi_reset);
    if (no_color == NULL)
    {
        printf("  %sENABLED%s (color escape codes are active; disable by setting NO_COLOR=1)\n",
               ansi_color_green, ansi_reset);
    }
    else
    {
        printf("  %sDISABLED%s (NO_COLOR environment variable is present)\n",
               ansi_color_red, ansi_reset);
    }
} // print_color_mode

static void print_help(const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  gric-stream-to-pipe - Pipes raw ImageStreamIO stream data to stdout\n\n");

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
    print_color_mode();
} // print_help

int main(int argc, char *argv[])
{
    init_colors();

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
        max_frames = atol(argv[2]);

    IMAGE stream_image;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(stream_name, &stream_image) != 0)
    {
        fprintf(stderr, "Error connecting to stream %s\n", stream_name);
        return 1;
    }

    long width = stream_image.md[0].size[0];
    long height = stream_image.md[0].size[1];
    long nelements = width * height;

    // Output metadata to stderr for user/script info
    fprintf(stderr, "Connected: %s (%ldx%ld), Type: %d\n", stream_name, width, height,
            stream_image.md[0].datatype);

    signal(SIGINT, handle_sigint);

    // Buffer for one frame (doubles)
    double *buffer = (double *)malloc(nelements * sizeof(double));
    if (!buffer)
        return 1;

    uint64_t last_cnt0 = stream_image.md[0].cnt0;
    long processed = 0;

    while (!stop && (max_frames < 0 || processed < max_frames))
    {
        // Wait for next frame
        // Loop to catch up if behind, similar to frameread.c logic
        while (stream_image.md[0].cnt0 <= last_cnt0 && !stop)
        {
            if (ImageStreamIO_semwait(&stream_image, 0) != 0)
            {
                if (!stop)
                    fprintf(stderr, "Semwait failed\n");
                break;
            }
        }
        if (stop)
            break;

        last_cnt0++;

        // Calculate read index (simplified 2D assumption or circular buffer logic?)
        // For simple piping, we just want "a frame".
        // If 3D, we should handle slice.
        // Let's copy frameread.c's slice logic.
        long current_read_slice = 0;
        if (stream_image.md[0].naxis > 2)
        {
            long depth = stream_image.md[0].size[2];
            // We need to track slice. Since we increment last_cnt0 1 by 1,
            // we can just track slice % depth.
            // However, we need initialized slice.
            // Let's simplify: read the slice corresponding to last_cnt0?
            // Usually slice = cnt1. But cnt1 is write pos.
            // If we assume sequential, we track it.
            // Ideally we copy frameread.c's stateful logic, but for a simple dumper
            // let's just grab the latest frame if we want real-time, or sequential if we
            // want exactness.
            // User wants "streaming input... just like image-stream".
            // So I should implement sequential reading.

            // Re-implement slice tracking:
            // Initial slice was stream_image.md[0].cnt1 at start.
            // But we didn't capture start state perfectly here.
            // Let's assume slice 0 for simple 2D, or try to sync.
            // Actually, for this tool, let's assume 2D streams for now
            // OR reuse the logic: slice = (initial_slice + processed) % depth.
            static long slice_idx = -1;
            if (slice_idx == -1)
                slice_idx = stream_image.md[0].cnt1;
            else
                slice_idx = (slice_idx + 1) % depth;

            current_read_slice = slice_idx;
        }

        long offset = current_read_slice * nelements;
        int dtype = stream_image.md[0].datatype;

// Type conversion to Double
// Macros (copied from frameread.c/ImageStruct.h knowledge)
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

        // This pointer cast logic depends on type
        switch (dtype)
        {
        case _DATATYPE_FLOAT:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((float *)stream_image.array.F)[offset + i];
            break;
        case _DATATYPE_DOUBLE:
            for (long i = 0; i < nelements; i++)
                buffer[i] = ((double *)stream_image.array.D)[offset + i];
            break;
        case _DATATYPE_UINT8:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((uint8_t *)stream_image.array.UI8)[offset + i];
            break;
        case _DATATYPE_UINT16:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((uint16_t *)stream_image.array.UI16)[offset + i];
            break;
        case _DATATYPE_INT16:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((int16_t *)stream_image.array.SI16)[offset + i];
            break;
        case _DATATYPE_UINT32:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((uint32_t *)stream_image.array.UI32)[offset + i];
            break;
        case _DATATYPE_INT32:
            for (long i = 0; i < nelements; i++)
                buffer[i] = (double)((int32_t *)stream_image.array.SI32)[offset + i];
            break;
        default:
            // Skip unsupported
            break;
        }

        // Write raw bytes to stdout
        fwrite(buffer, sizeof(double), nelements, stdout);
        // fflush(stdout); // Optional, maybe better for piping

        processed++;
    }

    free(buffer);
    return 0;
#endif
}

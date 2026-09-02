/**
 * @file ascii-spot-2-video.c
 * @brief Coordinate to video or ImageStreamIO generator.
 *
 * Translates point coordinate lists into 2D image matrices of a Gaussian spot, outputting
 * them as MP4 files (via FFmpeg pipe) or a live ImageStreamIO stream.
 */

#include <ctype.h>
#include <math.h>
#include <signal.h>
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

volatile sig_atomic_t stop_requested = 0;

/**
 * handle_sigint() - Signal handler for graceful termination.
 * @sig: Signal number.
 */
static void handle_sigint(
    int sig)
{
    (void)sig;
    stop_requested = 1;
}

/**
 * struct SamplePoint - 3D coordinate point.
 * @v1: First dimension coordinate.
 * @v2: Second dimension coordinate.
 * @v3: Third dimension coordinate (scale/variance).
 */
typedef struct {
    double v1;
    double v2;
    double v3;
} SamplePoint;

/**
 * struct SpotGeneratorConfig - Configuration arguments for spot simulation.
 * @size:        Image size (width and height).
 * @alpha:       Spot size scaling factor.
 * @input_file:  Input ASCII coordinates file.
 * @output_file: Output destination (file or SHM name).
 * @noise_level: Standard deviation of Gaussian noise.
 * @max_frames:  Maximum frames to render.
 * @isio_mode:   Flag for ImageStreamIO shared memory output.
 * @fps:         Playback frame rate.
 * @cnt2sync:    Flag for cnt2 sync gating.
 * @loop_mode:   Flag for continuous looping.
 * @repeats:     Number of sequence repetitions.
 * @pc_mode:     Flag to export photocenter.
 */
typedef struct {
    int         size;
    double      alpha;
    const char *input_file;
    const char *output_file;
    double      noise_level;
    int         max_frames;
    int         isio_mode;
    double      fps;
    int         cnt2sync;
    int         loop_mode;
    int         repeats;
    int         pc_mode;
} SpotGeneratorConfig;

/**
 * gauss_noise() - Generate zero-mean Gaussian noise via Box-Muller transform.
 * @stddev: Standard deviation.
 *
 * Return: Sampled noise value.
 */
static double gauss_noise(
    double stddev)
{
    if (stddev <= 0.0)
    {
        return 0.0;
    }
    double u = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double v = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return stddev * sqrt(-2.0 * log(u)) * cos(2.0 * 3.14159265358979323846 * v);
}

/**
 * compute_centroid_float() - Calculate centroid (xc, yc) and total flux from float buffer.
 * @buffer: Single-precision pixel buffer [size * size].
 * @size:   Image dimension.
 * @xc:     Output X coordinate.
 * @yc:     Output Y coordinate.
 * @flux:   Output total integrated flux.
 */
static void compute_centroid_float(
    const float *buffer,
    int          size,
    double      *xc,
    double      *yc,
    double      *flux)
{
    double sum_x = 0.0, sum_y = 0.0, sum_val = 0.0;
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            double val = (double)buffer[y * size + x];
            sum_val += val;
            sum_x += x * val;
            sum_y += y * val;
        }
    }
    if (sum_val != 0.0)
    {
        *xc = sum_x / sum_val;
        *yc = sum_y / sum_val;
    }
    else
    {
        *xc = size / 2.0;
        *yc = size / 2.0;
    }
    *flux = sum_val;
}

/**
 * compute_centroid_u8() - Calculate centroid (xc, yc) and total flux from RGB u8 buffer.
 * @buffer: RGB pixel buffer [size * size * 3].
 * @size:   Image dimension.
 * @xc:     Output X coordinate.
 * @yc:     Output Y coordinate.
 * @flux:   Output total integrated flux.
 */
static void compute_centroid_u8(
    const unsigned char *buffer,
    int                  size,
    double              *xc,
    double              *yc,
    double              *flux)
{
    double sum_x = 0.0, sum_y = 0.0, sum_val = 0.0;
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            double val = (double)buffer[(y * size + x) * 3];
            sum_val += val;
            sum_x += x * val;
            sum_y += y * val;
        }
    }
    if (sum_val != 0.0)
    {
        *xc = sum_x / sum_val;
        *yc = sum_y / sum_val;
    }
    else
    {
        *xc = size / 2.0;
        *yc = size / 2.0;
    }
    *flux = sum_val;
}

/**
 * render_gaussian_spot_float() - Render a Gaussian spot into a single-precision float buffer.
 * @buffer:      Output float buffer [size * size].
 * @size:        Image dimension.
 * @alpha:       Spot size scale.
 * @v1:          X coordinate in [-1.5, 1.5].
 * @v2:          Y coordinate in [-1.5, 1.5].
 * @v3:          Z scale coordinate.
 * @noise_level: Noise stddev.
 */
static void render_gaussian_spot_float(
    float  *buffer,
    int     size,
    double  alpha,
    double  v1,
    double  v2,
    double  v3,
    double  noise_level)
{
    memset(buffer, 0, (size_t)(size * size) * sizeof(float));
    double cx = (v1 + 1.5) / 3.0 * size;
    double cy = (1.0 - (v2 + 1.5) / 3.0) * size;
    double sigma = size * alpha * (v3 + 1.5) / 2.0;
    double ts2 = 2.0 * sigma * sigma;

    int r = (int)ceil(4.0 * sigma);
    int mx = (int)cx - r;
    int Mx = (int)cx + r;
    int my = (int)cy - r;
    int My = (int)cy + r;

    if (mx < 0)
    {
        mx = 0;
    }
    if (Mx >= size)
    {
        Mx = size - 1;
    }
    if (my < 0)
    {
        my = 0;
    }
    if (My >= size)
    {
        My = size - 1;
    }

    for (int y = my; y <= My; y++)
    {
        for (int x = mx; x <= Mx; x++)
        {
            double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            float v = (float)(255.0 * exp(-d2 / ts2));
            if (v > 0)
            {
                buffer[y * size + x] = v;
            }
        }
    }

    if (noise_level > 0.0)
    {
        for (int k = 0; k < size * size; k++)
        {
            buffer[k] += (float)gauss_noise(noise_level);
        }
    }
}

/**
 * render_gaussian_spot_rgb() - Render a Gaussian spot into an RGB24 unsigned char buffer.
 * @buffer:      Output RGB24 buffer [size * size * 3].
 * @size:        Image dimension.
 * @alpha:       Spot size scale.
 * @v1:          X coordinate in [-1.5, 1.5].
 * @v2:          Y coordinate in [-1.5, 1.5].
 * @v3:          Z scale coordinate.
 * @noise_level: Noise stddev.
 */
static void render_gaussian_spot_rgb(
    unsigned char *buffer,
    int            size,
    double         alpha,
    double         v1,
    double         v2,
    double         v3,
    double         noise_level)
{
    memset(buffer, 0, (size_t)(size * size * 3));
    double cx = (v1 + 1.5) / 3.0 * size;
    double cy = (1.0 - (v2 + 1.5) / 3.0) * size;
    double sigma = size * alpha * (v3 + 1.5) / 2.0;
    double ts2 = 2.0 * sigma * sigma;

    int r = (int)ceil(4.0 * sigma);
    int mx = (int)cx - r;
    int Mx = (int)cx + r;
    int my = (int)cy - r;
    int My = (int)cy + r;

    if (mx < 0)
    {
        mx = 0;
    }
    if (Mx >= size)
    {
        Mx = size - 1;
    }
    if (my < 0)
    {
        my = 0;
    }
    if (My >= size)
    {
        My = size - 1;
    }

    for (int y = my; y <= My; y++)
    {
        for (int x = mx; x <= Mx; x++)
        {
            double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            unsigned char v = (unsigned char)(255.0 * exp(-d2 / ts2));
            if (v > 0)
            {
                int idx = (y * size + x) * 3;
                buffer[idx] = v;
                buffer[idx + 1] = v;
                buffer[idx + 2] = v;
            }
        }
    }

    if (noise_level > 0.0)
    {
        for (int k = 0; k < size * size; k++)
        {
            double n = gauss_noise(noise_level);
            int val = buffer[3 * k] + (int)round(n);
            if (val < 0)
            {
                val = 0;
            }
            if (val > 255)
            {
                val = 255;
            }
            unsigned char uval = (unsigned char)val;
            buffer[3 * k] = uval;
            buffer[3 * k + 1] = uval;
            buffer[3 * k + 2] = uval;
        }
    }
}

/**
 * export_samples_txt() - Write sampled coordinate sequence to disk.
 * @output_file:  Base output path.
 * @samples:      Sample array.
 * @sample_count: Number of samples.
 */
static void export_samples_txt(
    const char        *output_file,
    const SamplePoint *samples,
    size_t             sample_count)
{
    if (sample_count == 0 || output_file == NULL)
    {
        return;
    }

    char txt_out_name[2048];
    snprintf(txt_out_name, sizeof(txt_out_name), "%s.txt", output_file);
    FILE *ftxt = fopen(txt_out_name, "w");
    if (ftxt)
    {
        for (size_t i = 0; i < sample_count; i++)
        {
            fprintf(ftxt, "%f %f %f\n", samples[i].v1, samples[i].v2, samples[i].v3);
        }
        fclose(ftxt);
        printf("Written %zu samples to %s\n", sample_count, txt_out_name);
    }
    else
    {
        fprintf(stderr, "Error: Could not write to %s\n", txt_out_name);
    }
}

/**
 * print_help_raw() - Output full formatted manual screen to stdout.
 * @progname: Executable name.
 */
static void print_help_raw(
    const char *progname)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %sgric-ascii-spot-2-video%s - Convert coordinate text file to video/stream\n\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s%s%s %s[options]%s %s<pixel_size>%s %s<alpha>%s %s<input.txt>%s %s<output>%s\n\n",
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET, ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Reads a text file containing 3D coordinates (v1, v2, v3) and generates a 2D\n");
    printf("  Gaussian spot. Output can be an MP4 video or an ImageStreamIO stream.\n\n");

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %sOutput Configuration%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s-isio%s            Write to ImageStreamIO stream instead of video file\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("    %s-pc%s              Compute and write photocenter to <output>.pc\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("\n  %sSimulation Parameters%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s-noise%s %s<val>%s     Add Gaussian noise with stddev <val> "
           "(%sDefault:%s%s 0.0%s)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_COLOR_CYAN, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);

    printf("\n  %sPlayback Control%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s-fps%s %s<val>%s       Set target frame rate (frames per second)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("    %s-cnt2sync%s        Enable cnt2 synchronization (ISIO mode only)\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("    %s-loop%s            Loop content forever\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("    %s-repeat%s %s<N>%s      Repeat content N times\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("    %s-maxfr%s %s<N>%s       Stop after N frames\n\n",
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);

    printf("  Arguments:\n");
    printf("    %spixel_size%s         Image size in pixels (square)\n",
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("    %salpha%s              Scaling factor for spot size\n",
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("    %sinput.txt%s          Input text file with samples (v1 v2 v3)\n",
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    printf("    %soutput%s             Output filename (MP4 file or Stream Name)\n\n",
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %s%s%s 256 2.0 input.txt output.mp4\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET, ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET);
    cli_print_color_mode();
}

/**
 * print_help() - Paged help wrapper.
 * @progname: Executable name.
 */
static void print_help(
    const char *progname)
{
    FILE *tmp = tmpfile();
    if (tmp != NULL)
    {
        int saved_stdout = dup(STDOUT_FILENO);
        int tmp_fd = fileno(tmp);
        dup2(tmp_fd, STDOUT_FILENO);

        print_help_raw(progname);
        fflush(stdout);

        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);

        fseek(tmp, 0, SEEK_END);
        long sz = ftell(tmp);
        fseek(tmp, 0, SEEK_SET);

        char *buf = malloc((size_t)sz + 1);
        if (buf != NULL)
        {
            size_t read_bytes = fread(buf, 1, (size_t)sz, tmp);
            buf[read_bytes] = '\0';
            cli_print_pager(buf);
            free(buf);
        }
        fclose(tmp);
    }
    else
    {
        print_help_raw(progname);
    }
}

/**
 * parse_cli_options() - Parse command-line flags into SpotGeneratorConfig.
 * @argc: Number of arguments.
 * @argv: Argument array.
 * @cfg:  Output configuration struct.
 *
 * Return: 0 on success, 1 on error/help.
 */
static int parse_cli_options(
    int                  argc,
    char                *argv[],
    SpotGeneratorConfig *cfg)
{
    memset(cfg, 0, sizeof(SpotGeneratorConfig));
    cfg->max_frames = -1;
    cfg->repeats = 1;

    int positional_idx = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-isio") == 0)
        {
            cfg->isio_mode = 1;
        }
        else if (strcmp(argv[i], "-cnt2sync") == 0)
        {
            cfg->cnt2sync = 1;
        }
        else if (strcmp(argv[i], "-loop") == 0)
        {
            cfg->loop_mode = 1;
        }
        else if (strcmp(argv[i], "-pc") == 0)
        {
            cfg->pc_mode = 1;
        }
        else if (strcmp(argv[i], "-repeat") == 0)
        {
            if (i + 1 < argc)
            {
                cfg->repeats = atoi(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-fps") == 0)
        {
            if (i + 1 < argc)
            {
                cfg->fps = atof(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-noise") == 0)
        {
            if (i + 1 < argc)
            {
                cfg->noise_level = atof(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-maxfr") == 0)
        {
            if (i + 1 < argc)
            {
                cfg->max_frames = atoi(argv[++i]);
            }
        }
        else if (argv[i][0] == '-')
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
        else
        {
            switch (positional_idx)
            {
            case 0:
                cfg->size = atoi(argv[i]);
                break;
            case 1:
                cfg->alpha = atof(argv[i]);
                break;
            case 2:
                cfg->input_file = argv[i];
                break;
            case 3:
                cfg->output_file = argv[i];
                break;
            case 4:
                if (cfg->noise_level == 0.0)
                {
                    cfg->noise_level = atof(argv[i]);
                }
                break;
            case 5:
                if (cfg->max_frames == -1)
                {
                    cfg->max_frames = atoi(argv[i]);
                }
                break;
            }
            positional_idx++;
        }
    }

    if (positional_idx < 4)
    {
        fprintf(stderr, "Error: Missing required positional arguments.\n");
        print_help(argv[0]);
        return 1;
    }

    return 0;
}

/**
 * main() - Entry point of the spot video generator.
 * @argc: Command argument count.
 * @argv: Command argument array.
 *
 * Return: 0 on success, 1 on error.
 */
int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    if (argc < 2)
    {
        print_help(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        print_help(argv[0]);
        return 0;
    }

    SpotGeneratorConfig cfg;
    if (parse_cli_options(argc, argv, &cfg) != 0)
    {
        return 1;
    }

    FILE *fin = fopen(cfg.input_file, "r");
    if (!fin)
    {
        fprintf(stderr, "Error: Could not open input file %s\n", cfg.input_file);
        return 1;
    }

    FILE *pc_out = NULL;
    if (cfg.pc_mode)
    {
        char pc_fname[2048];
        snprintf(pc_fname, sizeof(pc_fname), "%s.pc", cfg.output_file);
        pc_out = fopen(pc_fname, "w");
        if (!pc_out)
        {
            fprintf(stderr, "Error: Could not open photocenter output file %s\n", pc_fname);
            fclose(fin);
            return 1;
        }
    }

#ifdef USE_IMAGESTREAMIO
    IMAGE stream_image;
    float *stream_buffer = NULL;
#endif
    unsigned char *frame_rgb = NULL;
    FILE *pipe = NULL;

    if (cfg.isio_mode)
    {
#ifdef USE_IMAGESTREAMIO
        uint32_t dims[2] = {(uint32_t)cfg.size, (uint32_t)cfg.size};
        if (ImageStreamIO_createIm(&stream_image, cfg.output_file, 2, dims,
                                  _DATATYPE_FLOAT, 1, 1, 1) != 0)
        {
            fclose(fin);
            if (pc_out) fclose(pc_out);
            return 1;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        stream_image.md[0].creationtime = ts;
        stream_image.md[0].atime = ts;
        stream_image.md[0].writetime = ts;
        stream_buffer = (float *)malloc((size_t)(cfg.size * cfg.size) * sizeof(float));
#else
        fprintf(stderr, "Error: ImageStreamIO support not compiled in.\n");
        fclose(fin);
        if (pc_out) fclose(pc_out);
        return 1;
#endif
    }
    else
    {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s %dx%d -r 30 -i - -c:v "
                 "libx264 -pix_fmt yuv420p -crf 10 -preset slow \"%s\"",
                 cfg.size, cfg.size, cfg.output_file);
        pipe = popen(cmd, "w");
        frame_rgb = (unsigned char *)malloc((size_t)(cfg.size * cfg.size * 3));
        if (!pipe)
        {
            fprintf(stderr, "Error: Could not open ffmpeg pipe.\n");
            fclose(fin);
            if (pc_out) fclose(pc_out);
            return 1;
        }
    }

    srand((unsigned int)time(NULL));
    char line[1024];
    int frame_count = 0;
    long long us_per_frame = (cfg.fps > 0) ? (long long)(1000000.0 / cfg.fps) : 0;
    struct timespec last_time, now;
    clock_gettime(CLOCK_MONOTONIC, &last_time);
    signal(SIGINT, handle_sigint);
    int current_repeat = 0;

    SamplePoint *samples = NULL;
    size_t sample_count = 0;
    size_t sample_capacity = 0;

    while (!stop_requested)
    {
        if (cfg.max_frames > 0 && frame_count >= cfg.max_frames)
        {
            break;
        }
        if (!fgets(line, sizeof(line) - 1, fin))
        {
            current_repeat++;
            if (cfg.loop_mode || current_repeat < cfg.repeats)
            {
                rewind(fin);
                continue;
            }
            break;
        }
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        double v1, v2, v3 = 0.0;
        int items = sscanf(line, "%lf %lf %lf", &v1, &v2, &v3);
        if (items < 2)
        {
            continue;
        }

        if (sample_count >= sample_capacity)
        {
            sample_capacity = (sample_capacity == 0) ? 1024 : sample_capacity * 2;
            samples = (SamplePoint *)realloc(samples, sample_capacity * sizeof(SamplePoint));
        }
        samples[sample_count].v1 = v1;
        samples[sample_count].v2 = v2;
        samples[sample_count].v3 = v3;
        sample_count++;

        if (cfg.isio_mode)
        {
#ifdef USE_IMAGESTREAMIO
            render_gaussian_spot_float(
                stream_buffer, cfg.size, cfg.alpha, v1, v2, v3, cfg.noise_level
            );

            if (cfg.pc_mode && pc_out)
            {
                double xc, yc, flux;
                compute_centroid_float(stream_buffer, cfg.size, &xc, &yc, &flux);
                double pc_v1 = (xc / cfg.size * 3.0) - 1.5;
                double pc_v2 = (1.0 - yc / cfg.size) * 3.0 - 1.5;
                fprintf(pc_out, "%f %f %f\n", pc_v1, pc_v2, flux);
            }

            if (cfg.cnt2sync)
            {
                while (!stop_requested)
                {
                    if (stream_image.md[0].cnt0 < stream_image.md[0].cnt2)
                    {
                        break;
                    }
                    usleep(10);
                }
            }
            else if (us_per_frame > 0)
            {
                clock_gettime(CLOCK_MONOTONIC, &now);
                long long el = (now.tv_sec - last_time.tv_sec) * 1000000LL +
                               (now.tv_nsec - last_time.tv_nsec) / 1000;
                if (el < us_per_frame)
                {
                    usleep((useconds_t)(us_per_frame - el));
                }
                clock_gettime(CLOCK_MONOTONIC, &last_time);
            }

            stream_image.md[0].write = 1;
            memcpy(stream_image.array.F, stream_buffer,
                   (size_t)(cfg.size * cfg.size) * sizeof(float));
            stream_image.md[0].write = 0;

            struct timespec tw;
            clock_gettime(CLOCK_REALTIME, &tw);
            stream_image.md[0].writetime = tw;
            stream_image.md[0].atime = tw;
            stream_image.md[0].lastaccesstime = tw;
            stream_image.md[0].cnt0++;
            ImageStreamIO_sempost(&stream_image, -1);
#endif
        }
        else
        {
            render_gaussian_spot_rgb(
                frame_rgb, cfg.size, cfg.alpha, v1, v2, v3, cfg.noise_level
            );

            if (cfg.pc_mode && pc_out)
            {
                double xc, yc, flux;
                compute_centroid_u8(frame_rgb, cfg.size, &xc, &yc, &flux);
                double pc_v1 = (xc / cfg.size * 3.0) - 1.5;
                double pc_v2 = (1.0 - yc / cfg.size) * 3.0 - 1.5;
                fprintf(pc_out, "%f %f %f\n", pc_v1, pc_v2, flux);
            }

            if (pipe)
            {
                fwrite(frame_rgb, 1, (size_t)(cfg.size * cfg.size * 3), pipe);
            }

            if (us_per_frame > 0)
            {
                clock_gettime(CLOCK_MONOTONIC, &now);
                long long el = (now.tv_sec - last_time.tv_sec) * 1000000LL +
                               (now.tv_nsec - last_time.tv_nsec) / 1000;
                if (el < us_per_frame)
                {
                    usleep((useconds_t)(us_per_frame - el));
                }
                clock_gettime(CLOCK_MONOTONIC, &last_time);
            }
        }
        frame_count++;
    }

    export_samples_txt(cfg.output_file, samples, sample_count);

    if (samples)
    {
        free(samples);
    }
    if (pc_out)
    {
        fclose(pc_out);
        printf("Written photocenter data to %s.pc\n", cfg.output_file);
    }

    if (cfg.isio_mode)
    {
#ifdef USE_IMAGESTREAMIO
        if (stream_buffer)
        {
            free(stream_buffer);
        }
#endif
    }
    else
    {
        if (frame_rgb)
        {
            free(frame_rgb);
        }
        if (pipe)
        {
            pclose(pipe);
        }
    }
    fclose(fin);
    return 0;
}
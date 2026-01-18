#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>
#include <signal.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>
#endif

#define ANSI_COLOR_ORANGE  "\x1b[38;5;208m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_BG_GREEN      "\x1b[42m"
#define ANSI_COLOR_BLACK   "\x1b[30m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"
#define ANSI_UNDERLINE     "\x1b[4m"

volatile sig_atomic_t stop_requested = 0;

void handle_sigint(int sig) {
    stop_requested = 1;
}

void clamp(int *val) {
    if (*val < 0) *val = 0;
    if (*val > 255) *val = 255;
}

void print_help(const char *progname) {
    printf("%sNAME%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  gric-ascii-spot-2-video - Convert coordinate text file to video/stream\n\n");

    printf("%sSYNOPSIS%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  %s [options] <pixel_size> <alpha> <input.txt> <output>\n\n", progname);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  Reads a text file containing 3D coordinates (v1, v2, v3) and generates a 2D Gaussian spot.\n");
    printf("  Output can be an MP4 video (via ffmpeg) or an ImageStreamIO shared memory stream.\n");

    printf("\n%sOPTIONS%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  %sOutput Configuration%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-isio%s            Write to ImageStreamIO stream instead of video file\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-pc%s              Compute and write photocenter to <output>.pc\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);

    printf("\n  %sSimulation Parameters%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-noise <val>%s     Add Gaussian noise with stddev <val> (Default: 0.0)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    
    printf("\n  %sPlayback Control%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-fps <val>%s       Set target frame rate (frames per second)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-cnt2sync%s        Enable cnt2 synchronization (ISIO mode only)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-loop%s            Loop content forever\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-repeat <N>%s      Repeat content N times\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-maxfr <N>%s       Stop after N frames\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    
    printf("\n%sARGUMENTS%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  pixel_size      Image size in pixels (square)\n");
    printf("  alpha           Scaling factor for spot size\n");
    printf("  input.txt       Input text file with samples (v1 v2 v3)\n");
    printf("  output          Output filename (MP4 file or Stream Name)\n");
    printf("\n");
}

typedef struct {
    double v1;
    double v2;
    double v3;
} SamplePoint;

double gauss_noise(double stddev) {
    if (stddev <= 0.0) return 0.0;
    double u = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double v = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return stddev * sqrt(-2.0 * log(u)) * cos(2.0 * 3.14159265359 * v);
}

// Compute centroid and flux from float buffer
void compute_centroid_float(float *buffer, int size, double *xc, double *yc, double *flux) {
    double sum_x = 0.0, sum_y = 0.0, sum_val = 0.0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            double val = (double)buffer[y * size + x];
            sum_val += val;
            sum_x += x * val;
            sum_y += y * val;
        }
    }
    if (sum_val != 0.0) {
        *xc = sum_x / sum_val;
        *yc = sum_y / sum_val;
    } else {
        *xc = size / 2.0;
        *yc = size / 2.0;
    }
    *flux = sum_val;
}

// Compute centroid and flux from RGB u8 buffer
void compute_centroid_u8(unsigned char *buffer, int size, double *xc, double *yc, double *flux) {
    double sum_x = 0.0, sum_y = 0.0, sum_val = 0.0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            // Use one channel (R) as they are identical
            double val = (double)buffer[(y * size + x) * 3];
            sum_val += val;
            sum_x += x * val;
            sum_y += y * val;
        }
    }
    if (sum_val != 0.0) {
        *xc = sum_x / sum_val;
        *yc = sum_y / sum_val;
    } else {
        *xc = size / 2.0;
        *yc = size / 2.0;
    }
    *flux = sum_val;
}

int main(int argc, char *argv[]) {
    // Basic check before parsing
    if (argc < 2) { print_help(argv[0]); return 1; }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) { print_help(argv[0]); return 0; }

    int size = 0; 
    double alpha = 0.0; 
    char *input_file = NULL; 
    char *output_file = NULL; 
    
    double noise_level = 0.0; 
    int max_frames = -1;
    int isio_mode = 0; 
    double fps = 0.0; 
    int cnt2sync = 0;
    int loop_mode = 0; 
    int repeats = 1;
    int pc_mode = 0;

    int positional_idx = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-isio") == 0) { isio_mode = 1; }
        else if (strcmp(argv[i], "-cnt2sync") == 0) { cnt2sync = 1; }
        else if (strcmp(argv[i], "-loop") == 0) { loop_mode = 1; }
        else if (strcmp(argv[i], "-pc") == 0) { pc_mode = 1; }
        else if (strcmp(argv[i], "-repeat") == 0) { if (i+1 < argc) repeats = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-fps") == 0) { if (i+1 < argc) fps = atof(argv[++i]); }
        else if (strcmp(argv[i], "-noise") == 0) { if (i+1 < argc) noise_level = atof(argv[++i]); }
        else if (strcmp(argv[i], "-maxfr") == 0) { if (i+1 < argc) max_frames = atoi(argv[++i]); }
        else if (argv[i][0] == '-') { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
        else {
            switch (positional_idx) {
                case 0: size = atoi(argv[i]); break; 
                case 1: alpha = atof(argv[i]); break;
                case 2: input_file = argv[i]; break; 
                case 3: output_file = argv[i]; break;
                // Legacy support for positional args (optional)
                case 4: 
                    // Only accept if not already set by -noise
                    if (noise_level == 0.0) noise_level = atof(argv[i]); 
                    break; 
                case 5: 
                    // Only accept if not already set by -maxfr
                    if (max_frames == -1) max_frames = atoi(argv[i]); 
                    break;
            }
            positional_idx++;
        }
    }
    
    if (positional_idx < 4) { 
        fprintf(stderr, "Error: Missing required positional arguments.\n");
        print_help(argv[0]); 
        return 1; 
    }

    FILE *fin = fopen(input_file, "r"); 
    if (!fin) {
        fprintf(stderr, "Error: Could not open input file %s\n", input_file);
        return 1;
    }
    
    // Setup PC output
    FILE *pc_out = NULL;
    if (pc_mode) {
        char pc_fname[2048];
        snprintf(pc_fname, sizeof(pc_fname), "%s.pc", output_file);
        pc_out = fopen(pc_fname, "w");
        if (!pc_out) {
            fprintf(stderr, "Error: Could not open photocenter output file %s\n", pc_fname);
            return 1;
        }
    }

    #ifdef USE_IMAGESTREAMIO
    IMAGE stream_image; float *stream_buffer = NULL;
    #endif
    unsigned char *frame_rgb = NULL; FILE *pipe = NULL;
    
    if (isio_mode) {
        #ifdef USE_IMAGESTREAMIO
        uint32_t dims[2] = {(uint32_t)size, (uint32_t)size};
        if (ImageStreamIO_createIm(&stream_image, output_file, 2, dims, _DATATYPE_FLOAT, 1, 1, 1) != 0) return 1;
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        stream_image.md[0].creationtime = ts; stream_image.md[0].atime = ts; stream_image.md[0].writetime = ts;
        stream_buffer = (float *)malloc(size * size * sizeof(float));
        #else
        fprintf(stderr, "Error: ImageStreamIO support not compiled in.\n");
        return 1;
        #endif
    } else {
        char cmd[1024]; snprintf(cmd, 1023, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s %dx%d -r 30 -i - -c:v libx264 -pix_fmt yuv420p -crf 10 -preset slow \"%s\"", size, size, output_file);
        pipe = popen(cmd, "w"); frame_rgb = (unsigned char *)malloc(size * size * 3);
        if (!pipe) {
            fprintf(stderr, "Error: Could not open ffmpeg pipe.\n");
            return 1;
        }
    }
    
    srand(time(NULL)); char line[1024]; int frame_count = 0; long long us_per_frame = (fps > 0) ? (long long)(1000000.0 / fps) : 0;
    struct timespec last_time, now; clock_gettime(CLOCK_MONOTONIC, &last_time);
    signal(SIGINT, handle_sigint); int current_repeat = 0;
    
    // Sample collection
    SamplePoint *samples = NULL;
    size_t sample_count = 0;
    size_t sample_capacity = 0;

    while (!stop_requested) {
        if (max_frames > 0 && frame_count >= max_frames) break;
        if (!fgets(line, 1023, fin)) {
            current_repeat++;
            if (loop_mode || current_repeat < repeats) { rewind(fin); continue; }
            break;
        }
        if (line[0] == '#' || line[0] == '\n') continue;
        
        double v1, v2, v3 = 0.0; 
        int items = sscanf(line, "%lf %lf %lf", &v1, &v2, &v3);
        if (items < 2) continue;
        
        // Store sample
        if (sample_count >= sample_capacity) {
            sample_capacity = (sample_capacity == 0) ? 1024 : sample_capacity * 2;
            samples = (SamplePoint *)realloc(samples, sample_capacity * sizeof(SamplePoint));
        }
        samples[sample_count].v1 = v1;
        samples[sample_count].v2 = v2;
        samples[sample_count].v3 = v3;
        sample_count++;

        if (isio_mode) {
            #ifdef USE_IMAGESTREAMIO
            memset(stream_buffer, 0, size * size * sizeof(float));
            double cx = (v1+1.5)/3.0*size, cy = (1.0-(v2+1.5)/3.0)*size, sigma = size*alpha*(v3+1.5)/2.0, ts2 = 2.0*sigma*sigma;
            int r = (int)ceil(4.0*sigma), mx = (int)cx-r, Mx = (int)cx+r, my = (int)cy-r, My = (int)cy+r;
            if (mx<0) mx=0; if (Mx>=size) Mx=size-1; if (my<0) my=0; if (My>=size) My=size-1;
            for (int y=my; y<=My; y++) for (int x=mx; x<=Mx; x++) { double d2 = (x-cx)*(x-cx)+(y-cy)*(y-cy); float v = (float)(255.0*exp(-d2/ts2)); if (v>0) stream_buffer[y*size+x] = v; }
            
            if (noise_level > 0.0) {
                for (int k = 0; k < size * size; k++) {
                    stream_buffer[k] += (float)gauss_noise(noise_level);
                }
            }

            if (pc_mode && pc_out) {
                double xc, yc, flux;
                compute_centroid_float(stream_buffer, size, &xc, &yc, &flux);
                // Convert back to v1, v2
                double pc_v1 = (xc / size * 3.0) - 1.5;
                double pc_v2 = (1.0 - yc / size) * 3.0 - 1.5;
                fprintf(pc_out, "%f %f %f\n", pc_v1, pc_v2, flux);
            }

            if (cnt2sync) { while (!stop_requested) { if (stream_image.md[0].cnt0 < stream_image.md[0].cnt2) break; usleep(10); } } 
            else if (us_per_frame > 0) { clock_gettime(CLOCK_MONOTONIC, &now); long long el = (now.tv_sec-last_time.tv_sec)*1000000LL+(now.tv_nsec-last_time.tv_nsec)/1000; if (el<us_per_frame) usleep(us_per_frame-el); clock_gettime(CLOCK_MONOTONIC, &last_time); } 
            memcpy(stream_image.array.F, stream_buffer, size*size*sizeof(float));
            struct timespec tw; clock_gettime(CLOCK_REALTIME, &tw);
            stream_image.md[0].writetime = tw; stream_image.md[0].atime = tw; stream_image.md[0].lastaccesstime = tw;
            stream_image.md[0].cnt0++; ImageStreamIO_sempost(&stream_image, -1);
            #endif
        } else {
            memset(frame_rgb, 0, size*size*3);
            double cx = (v1+1.5)/3.0*size, cy = (1.0-(v2+1.5)/3.0)*size, sigma = size*alpha*(v3+1.5)/2.0, ts2 = 2.0*sigma*sigma;
            int r = (int)ceil(4.0*sigma), mx = (int)cx-r, Mx = (int)cx+r, my = (int)cy-r, My = (int)cy+r;
            if (mx<0) mx=0; if (Mx>=size) Mx=size-1; if (my<0) my=0; if (My>=size) My=size-1;
            for (int y=my; y<=My; y++) {
                for (int x=mx; x<=Mx; x++) {
                    double d2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                    unsigned char v = (unsigned char)(255.0*exp(-d2/ts2));
                    if (v > 0) {
                        int idx = (y*size + x) * 3;
                        frame_rgb[idx] = v;
                        frame_rgb[idx+1] = v;
                        frame_rgb[idx+2] = v;
                    }
                }
            }

            if (noise_level > 0.0) {
                for (int k = 0; k < size * size; k++) {
                    double n = gauss_noise(noise_level);
                    int val = frame_rgb[3*k] + (int)round(n);
                    if (val < 0) val = 0;
                    if (val > 255) val = 255;
                    unsigned char uval = (unsigned char)val;
                    frame_rgb[3*k] = uval;
                    frame_rgb[3*k+1] = uval;
                    frame_rgb[3*k+2] = uval;
                }
            }

            if (pc_mode && pc_out) {
                double xc, yc, flux;
                compute_centroid_u8(frame_rgb, size, &xc, &yc, &flux);
                // Convert back to v1, v2
                double pc_v1 = (xc / size * 3.0) - 1.5;
                double pc_v2 = (1.0 - yc / size) * 3.0 - 1.5;
                fprintf(pc_out, "%f %f %f\n", pc_v1, pc_v2, flux);
            }

            if (pipe) fwrite(frame_rgb, 1, size*size*3, pipe);
            
            if (us_per_frame > 0) { 
                 clock_gettime(CLOCK_MONOTONIC, &now); 
                 long long el = (now.tv_sec-last_time.tv_sec)*1000000LL+(now.tv_nsec-last_time.tv_nsec)/1000; 
                 if (el<us_per_frame) usleep(us_per_frame-el); 
                 clock_gettime(CLOCK_MONOTONIC, &last_time); 
            }
        }
        frame_count++;
    }
    
    // Write samples to text file
    if (sample_count > 0 && output_file) {
        char txt_out_name[2048];
        snprintf(txt_out_name, sizeof(txt_out_name), "%s.txt", output_file);
        FILE *ftxt = fopen(txt_out_name, "w");
        if (ftxt) {
            for (size_t i = 0; i < sample_count; i++) {
                fprintf(ftxt, "%f %f %f\n", samples[i].v1, samples[i].v2, samples[i].v3);
            }
            fclose(ftxt);
            printf("Written %zu samples to %s\n", sample_count, txt_out_name);
        } else {
            fprintf(stderr, "Error: Could not write to %s\n", txt_out_name);
        }
    }
    if (samples) free(samples);
    if (pc_out) { fclose(pc_out); printf("Written photocenter data to %s.pc\n", output_file); }

    if (isio_mode) {
#ifdef USE_IMAGESTREAMIO
        if (stream_buffer) free(stream_buffer);
#endif
    } else { if (frame_rgb) free(frame_rgb); if (pipe) pclose(pipe); }
    fclose(fin); return 0;
}
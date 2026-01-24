#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_PNG
#include <png.h>
#include "simple_font.h"
#else
void print_help(const char *progname) {
    fprintf(stderr, "Usage: %s <log_file> [output_file.png]\n", progname);
    fprintf(stderr, "This tool requires libpng to be enabled at compile time.\n");
}
int main(int argc, char *argv[]) {
    print_help(argv[0]);
    return 1;
}
#endif // USE_PNG

#ifdef USE_PNG

#define PLOT_WIDTH 600
#define PLOT_HEIGHT 400

typedef struct {
    unsigned char r, g, b;
} ColorRGB;

typedef struct {
    int width;
    int height;
    unsigned char *data; 
} Canvas;


Canvas* init_canvas(int w, int h) {
    Canvas *c = (Canvas*)malloc(sizeof(Canvas));
    if (!c) return NULL;
    c->width = w;
    c->height = h;
    c->data = (unsigned char*)malloc(w * h * 3);
    if (!c->data) { free(c); return NULL; }
    memset(c->data, 255, w * h * 3);
    return c;
}

void free_canvas(Canvas *c) {
    if (c) { free(c->data); free(c); }
}

void set_pixel_opaque(Canvas *c, int x, int y, ColorRGB col) {
    if (x < 0 || x >= c->width || y < 0 || y >= c->height) return;
    int idx = (y * c->width + x) * 3;
    c->data[idx]   = col.r;
    c->data[idx+1] = col.g;
    c->data[idx+2] = col.b;
}

void draw_filled_rect(Canvas *c, int x, int y, int w, int h, ColorRGB col) {
    for (int j = y; j < y + h; j++) {
        for (int i = x; i < x + w; i++) {
            set_pixel_opaque(c, i, j, col);
        }
    }
}

void draw_line(Canvas *c, int x0, int y0, int x1, int y1, ColorRGB col) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (1) {
        set_pixel_opaque(c, x0, y0, col);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_char(Canvas *c, int x, int y, char ch, ColorRGB col, int scale, int bold) {
    if (ch < 32 || ch > 126) return;
    int idx = ch - 32;
    for (int col_idx = 0; col_idx < 5; col_idx++) {
        unsigned char bits = font5x7[idx][col_idx];
        for (int row = 0; row < 7; row++) {
            if (bits & (1 << row)) {
                for (int sy=0; sy<scale; sy++) {
                    for (int sx=0; sx<scale; sx++) {
                        set_pixel_opaque(c, x + col_idx*scale + sx, y + row*scale + sy, col);
                        if (bold) set_pixel_opaque(c, x + col_idx*scale + sx + 1, y + row*scale + sy, col);
                    }
                }
            }
        }
    }
}

void draw_string(Canvas *c, int x, int y, const char *str, ColorRGB col, int align, int scale, int bold) {
    int len = strlen(str);
    int total_width = len * 6 * scale;
    int start_x = (align == 1) ? x - total_width : x;
    for (int i = 0; i < len; i++) {
        draw_char(c, start_x + i * 6 * scale, y, str[i], col, scale, bold);
    }
}

void draw_histogram(Canvas *c, int x, int y, int w, int h, long *data, int count, const char* title) {
    ColorRGB bg = {250, 250, 250}, border = {0, 0, 0}, grid = {200, 200, 200}, bar_col = {100, 100, 255};
    draw_filled_rect(c, x, y, w, h, bg);
    draw_line(c, x, y, x+w, y, border); draw_line(c, x+w, y, x+w, y+h, border);
    draw_line(c, x+w, y+h, x, y+h, border); draw_line(c, x, y+h, x, y, border);
    long max_val = 0; int max_idx = 0;
    for (int i=0; i<count; i++) { if (data[i] > 0) max_idx = i; if (data[i] > max_val) max_val = data[i]; }
    if (max_val == 0) return;
    int display_count = max_idx + 2;
    if (display_count > count) display_count = count;
    double log_max = log10((double)max_val); if (log_max < 1.0) log_max = 1.0;
    
    // Y-axis (log scale)
    for (int p = 0; p <= (int)log_max + 1; p++) {
        double val = pow(10, p); if (val > max_val * 2.0) break;
        double norm_h = p / (log_max * 1.1); int y_pos = y + h - 10 - (int)(norm_h * (h - 40));
        if (y_pos >= y && y_pos <= y + h - 10) {
            draw_line(c, x, y_pos, x + w, y_pos, grid);
            char label[32]; snprintf(label, sizeof(label), p == 0 ? "1" : "10^%d", p);
            draw_string(c, x + 2, y_pos - 8, label, border, 0, 1, 0);
        }
    }

    // X-axis and bars
    double bar_w = (double)(w - 40) / display_count; if (bar_w < 1.0) bar_w = 1.0;
    for (int i=0; i<display_count; i++) {
        if (data[i] > 0) {
            double lg = log10((double)data[i]); double norm_h = lg / (log_max * 1.1);
            int bar_h = (int)(norm_h * (h - 40));
            if (bar_h < 1) bar_h = 1;
            int bar_x = x + 35 + (int)(i * bar_w);
            int bar_y = y + h - 10 - bar_h;
            draw_filled_rect(c, bar_x, bar_y, (int)bar_w + 1, bar_h, bar_col);
            
            // Only draw values for significant bars to avoid clutter
            if (bar_h > 10) {
              char val_str[32]; snprintf(val_str, sizeof(val_str), "%ld", data[i]);
              draw_string(c, bar_x, bar_y - 10, val_str, border, 0, 1, 0);
            }
        }
        // Draw x-axis tick/label
        char bin_str[32]; snprintf(bin_str, sizeof(bin_str), "%d", i);
        draw_string(c, x + 35 + (int)(i * bar_w), y + h - 8, bin_str, border, 0, 1, 0);
    }
    draw_string(c, x + w/2 - 80, y + 5, title, border, 0, 2, 1);
    draw_string(c, x + w/2 - 100, y + h + 15, "Number of Distance Computations", border, 0, 1, 1);
}


int save_png(Canvas *c, const char *filename) {
    FILE *fp = fopen(filename, "wb"); if (!fp) return 1;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return 1; }
    png_infop info = png_create_info_struct(png); if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return 1; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return 1; }
    png_init_io(png, fp);
    png_set_IHDR(png, info, c->width, c->height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    png_bytep row_pointers[c->height];
    for (int y = 0; y < c->height; y++) row_pointers[y] = &c->data[y * c->width * 3];
    png_write_image(png, row_pointers); png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info); fclose(fp);
    return 0;
}

void print_help(const char *progname) {
    printf("Usage: %s <locate_run.log> [output_file.png]\n\n", progname);
    printf("Description:\n");
    printf("  gric-locate-plot is a visualization tool for gric-locate results.\n");
    printf("  It reads a 'locate_run.log' file and generates a bar plot showing the\n");
    printf("  distribution of distance calculations performed per frame.\n\n");
    printf("Arguments:\n");
    printf("  <locate_run.log>  The log file generated by gric-locate.\n");
    printf("  [output_file.png] Optional: Path to save the resulting PNG image. Defaults to\n");
    printf("                    'locate_histogram.png' in the same directory as the log file.\n\n");
    printf("Example:\n");
    printf("  %s my_output/locate_run.log my_output/locate_dist_plot.png\n", progname);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        print_help(argv[0]);
        return 1;
    }

    char *log_filename = argv[1];
    char output_filename[4096] = {0};

    if (argc == 3) {
        strncpy(output_filename, argv[2], sizeof(output_filename) - 1);
    } else {
        char *last_slash = strrchr(log_filename, '/');
        if (last_slash) {
            int dir_len = last_slash - log_filename + 1;
            strncpy(output_filename, log_filename, dir_len);
        }
        strcat(output_filename, "locate_histogram.png");
    }

    // Max number of bins in histogram, should be coordinated with gric-locate
    const int MAX_BINS = 10000; 
    long *hist_data = (long *)calloc(MAX_BINS, sizeof(long));
    if (!hist_data) {
        fprintf(stderr, "Error: Failed to allocate memory for histogram data.\n");
        return 1;
    }

    int hist_parsing = 0;
    long total_frames = 0;

    printf("Reading log file: %s\n", log_filename);
    FILE *flog = fopen(log_filename, "r");
    if (!flog) {
        fprintf(stderr, "Error: Could not open log file %s\n", log_filename);
        free(hist_data);
        return 1;
    }

    char line[4096];
    while (fgets(line, sizeof(line), flog)) {
        if (hist_parsing) {
            if (strncmp(line, "STATS_DIST_HIST_END", 19) == 0) {
                hist_parsing = 0;
            } else {
                int k;
                long c;
                if (sscanf(line, "%d %ld", &k, &c) == 2 && k < MAX_BINS) {
                    hist_data[k] = c;
                }
            }
            continue;
        }
        
        if (strncmp(line, "STATS_TOTAL_FRAMES_PROCESSED: ", 30) == 0) {
            sscanf(line + 30, "%ld", &total_frames);
        } else if (strncmp(line, "STATS_DIST_HIST_START", 21) == 0) {
            hist_parsing = 1;
        }
    }
    fclose(flog);
    
    printf("Log loaded: %ld frames processed.\n", total_frames);

    Canvas *canvas = init_canvas(PLOT_WIDTH, PLOT_HEIGHT);
    if (!canvas) {
        fprintf(stderr, "Error: Failed to create canvas.\n");
        free(hist_data);
        return 1;
    }
    
    draw_histogram(canvas, 50, 50, PLOT_WIDTH - 100, PLOT_HEIGHT - 100, hist_data, MAX_BINS, "Distance Computations per Frame");

    printf("Saving PNG output: %s\n", output_filename);
    if (save_png(canvas, output_filename) != 0) {
        fprintf(stderr, "Error saving PNG file.\n");
    }

    free_canvas(canvas);
    free(hist_data);
    return 0;
}
#endif // USE_PNG

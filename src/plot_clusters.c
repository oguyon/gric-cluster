#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_PNG
#include <png.h>
#include "simple_font.h"
#endif

#define SVG_WIDTH 1200
#define PLOT_WIDTH 800
#define SVG_HEIGHT 800
#define VIEW_MIN -1.1
#define VIEW_MAX 1.1
#define VIEW_RANGE (VIEW_MAX - VIEW_MIN)

// Palette of colors for clusters
const char *colors[] = {
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#ffffff", "#000000"
};
#define NUM_COLORS (sizeof(colors) / sizeof(colors[0]))

typedef struct {
    int id;
    double x;
    double y;
} Anchor;

typedef struct {
    char text[256];
} HeaderLine;

#ifdef USE_PNG
typedef struct {
    unsigned char r, g, b;
} ColorRGB;

typedef struct {
    int width;
    int height;
    unsigned char *data; // RGB buffer
} Canvas;
#endif

double map_x(double x) {
    return (x - VIEW_MIN) / VIEW_RANGE * PLOT_WIDTH;
}

double map_y(double y) {
    return (VIEW_MAX - y) / VIEW_RANGE * SVG_HEIGHT; // SVG y is down
}

void print_args_on_error(int argc, char *argv[]) {
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++) {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
}

#ifdef USE_PNG
ColorRGB parse_color(const char *hex) {
    ColorRGB c = {0, 0, 0};
    if (hex[0] == '#') hex++;
    if (strcmp(hex, "black") == 0) return c;
    if (strcmp(hex, "white") == 0) { c.r=255; c.g=255; c.b=255; return c; }

    unsigned int r, g, b;
    if (sscanf(hex, "%02x%02x%02x", &r, &g, &b) == 3) {
        c.r = r; c.g = g; c.b = b;
    }
    return c;
}

Canvas* init_canvas(int w, int h) {
    Canvas *c = (Canvas*)malloc(sizeof(Canvas));
    c->width = w;
    c->height = h;
    c->data = (unsigned char*)calloc(w * h * 3, 1);
    // Fill white
    memset(c->data, 255, w * h * 3);
    return c;
}

void free_canvas(Canvas *c) {
    if (c) {
        free(c->data);
        free(c);
    }
}

void set_pixel(Canvas *c, int x, int y, ColorRGB col) {
    if (x < 0 || x >= c->width || y < 0 || y >= c->height) return;
    int idx = (y * c->width + x) * 3;
    c->data[idx]   = (unsigned char)(0.7 * col.r + 0.3 * 255);
    c->data[idx+1] = (unsigned char)(0.7 * col.g + 0.3 * 255);
    c->data[idx+2] = (unsigned char)(0.7 * col.b + 0.3 * 255);
}

void set_pixel_opaque(Canvas *c, int x, int y, ColorRGB col) {
    if (x < 0 || x >= c->width || y < 0 || y >= c->height) return;
    int idx = (y * c->width + x) * 3;
    c->data[idx]   = col.r;
    c->data[idx+1] = col.g;
    c->data[idx+2] = col.b;
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

void draw_circle(Canvas *c, int cx, int cy, int r, ColorRGB col) {
    int x = r, y = 0;
    int err = 0;
    while (x >= y) {
        set_pixel_opaque(c, cx + x, cy + y, col);
        set_pixel_opaque(c, cx + y, cy + x, col);
        set_pixel_opaque(c, cx - y, cy + x, col);
        set_pixel_opaque(c, cx - x, cy + y, col);
        set_pixel_opaque(c, cx - x, cy - y, col);
        set_pixel_opaque(c, cx - y, cy - x, col);
        set_pixel_opaque(c, cx + y, cy - x, col);
        set_pixel_opaque(c, cx + x, cy - y, col);
        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

void draw_filled_circle(Canvas *c, int cx, int cy, int r, ColorRGB col) {
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            if (x*x + y*y <= r*r) {
                set_pixel(c, cx+x, cy+y, col);
            }
        }
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
                        if (bold) {
                            set_pixel_opaque(c, x + col_idx*scale + sx + 1, y + row*scale + sy, col);
                        }
                    }
                }
            }
        }
    }
}

void draw_string(Canvas *c, int x, int y, const char *str, ColorRGB col, int align, int scale, int bold) {
    int len = strlen(str);
    int char_width = 5 * scale;
    int space_width = 1 * scale;
    int total_width = len * (char_width + space_width);

    int start_x = x;
    if (align == 1) start_x = x - total_width;

    for (int i = 0; i < len; i++) {
        draw_char(c, start_x + i * (char_width + space_width), y, str[i], col, scale, bold);
    }
}

int save_png(Canvas *c, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) return 1;

    png_infop info = png_create_info_struct(png);
    if (!info) return 1;

    if (setjmp(png_jmpbuf(png))) return 1;

    png_init_io(png, fp);
    png_set_IHDR(png, info, c->width, c->height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep row_pointers[c->height];
    for (int y = 0; y < c->height; y++) {
        row_pointers[y] = &c->data[y * c->width * 3];
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 0;
}

void draw_filled_rect(Canvas *c, int x, int y, int w, int h, ColorRGB col) {
    for (int j = y; j < y + h; j++) {
        for (int i = x; i < x + w; i++) {
            set_pixel_opaque(c, i, j, col);
        }
    }
}

void draw_histogram(Canvas *c, int x, int y, int w, int h, long *data, int count) {
    ColorRGB bg = {250, 250, 250};
    draw_filled_rect(c, x, y, w, h, bg);
    ColorRGB border = {0, 0, 0};
    draw_line(c, x, y, x+w, y, border);
    draw_line(c, x+w, y, x+w, y+h, border);
    draw_line(c, x+w, y+h, x, y+h, border);
    draw_line(c, x, y+h, x, y, border);

    long max_val = 0;
    int max_idx = 0;
    for (int i=0; i<count; i++) {
        if (data[i] > 0) max_idx = i;
        if (data[i] > max_val) max_val = data[i];
    }
    if (max_val == 0) return;
    
    int display_count = max_idx + 2;
    double log_max = log10((double)max_val);
    if (log_max < 1.0) log_max = 1.0;

    ColorRGB grid = {200, 200, 200};
    for (int p = 0; p <= (int)log_max + 1; p++) {
        double val = pow(10, p);
        if (val > max_val * 2.0) break;
        double norm_h = p / (log_max * 1.1);
        int y_pos = y + h - 10 - (int)(norm_h * (h - 20));
        if (y_pos >= y && y_pos <= y + h - 10) {
            draw_line(c, x, y_pos, x + w, y_pos, grid);
            char label[32];
            if (p == 0) snprintf(label, sizeof(label), "1");
            else snprintf(label, sizeof(label), "10^%d", p);
            draw_string(c, x + 2, y_pos - 8, label, border, 0, 1, 0);
        }
    }

    double bar_w = (double)(w - 30) / display_count;
    if (bar_w < 1.0) bar_w = 1.0;

    ColorRGB bar_col = {100, 100, 255};
    ColorRGB text_col = {0, 0, 0};
    
    for (int i=0; i<display_count; i++) {
        if (data[i] > 0) {
            double lg = log10((double)data[i]);
            double norm_h = lg / (log_max * 1.1);
            int bar_h = (int)(norm_h * (h - 20));
            int bar_x = x + 25 + (int)(i * bar_w);
            int bar_y = y + h - 10 - bar_h;
            draw_filled_rect(c, bar_x, bar_y, (int)bar_w + 1, bar_h, bar_col);
            
            char val_str[32];
            snprintf(val_str, sizeof(val_str), "%ld", data[i]);
            draw_string(c, bar_x, bar_y - 10, val_str, text_col, 0, 1, 0);
            
            char bin_str[32];
            snprintf(bin_str, sizeof(bin_str), "%d", i);
            draw_string(c, bar_x, y + h - 8, bin_str, text_col, 0, 1, 0);
        }
    }
    
    char title[] = "Samples / Dist Count";
    draw_string(c, x + w/2 - (strlen(title)*6)/2, y + 2, title, border, 0, 1, 1);
}

void draw_cluster_histogram(Canvas *c, int x, int y, int w, int h, long *data, int count) {
    ColorRGB bg = {250, 250, 250};
    draw_filled_rect(c, x, y, w, h, bg);
    ColorRGB border = {0, 0, 0};
    draw_line(c, x, y, x+w, y, border);
    draw_line(c, x+w, y, x+w, y+h, border);
    draw_line(c, x+w, y+h, x, y+h, border);
    draw_line(c, x, y+h, x, y, border);

    long max_val = 0;
    for (int i=0; i<count; i++) if (data[i] > max_val) max_val = data[i];
    if (max_val == 0) return;
    
    double log_max = log10((double)max_val);
    if (log_max < 1.0) log_max = 1.0;

    // Grid (Y axis)
    ColorRGB grid = {200, 200, 200};
    for (int p = 0; p <= (int)log_max + 1; p++) {
        double val = pow(10, p);
        if (val > max_val * 2.0) break;
        double norm_h = p / (log_max * 1.1);
        int y_pos = y + h - (int)(norm_h * (h - 10));
        if (y_pos >= y && y_pos <= y + h) {
            draw_line(c, x, y_pos, x + w, y_pos, grid);
            char label[32];
            if (p == 0) snprintf(label, sizeof(label), "1");
            else snprintf(label, sizeof(label), "10^%d", p);
            draw_string(c, x - 5, y_pos - 8, label, border, 1, 1, 0);
        }
    }

    double bar_w = (double)w / count;
    ColorRGB bar_col = {100, 200, 100};
    
    for (int i=0; i<count; i++) {
        if (data[i] > 0) {
            double lg = log10((double)data[i]);
            double norm_h = lg / (log_max * 1.1);
            int bar_h = (int)(norm_h * (h - 10));
            int bar_x = x + (int)(i * bar_w);
            int bar_y = y + h - bar_h;
            int draw_w = (int)bar_w;
            if (draw_w < 1) draw_w = 1;
            draw_filled_rect(c, bar_x, bar_y, draw_w, bar_h, bar_col);
        }
    }
    
    char title[] = "Samples / Cluster";
    draw_string(c, x + w/2 - (strlen(title)*6)/2, y + 2, title, border, 0, 1, 1);
}

void draw_dcc_matrix(Canvas *c, int x, int y, int w, int h, const char *dcc_file, int num_clusters) {
    if (num_clusters <= 0 || !dcc_file[0]) return;
    
    double *matrix = (double *)calloc(num_clusters * num_clusters, sizeof(double));
    if (!matrix) return;
    for(long k=0; k<num_clusters*num_clusters; k++) matrix[k] = -1.0;

    FILE *f = fopen(dcc_file, "r");
    double max_dist = 0.0;
    if (f) {
        char line[1024];
        while(fgets(line, sizeof(line), f)) {
            int i, j;
            double d;
            if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3) {
                if (i>=0 && i<num_clusters && j>=0 && j<num_clusters) {
                    matrix[i*num_clusters + j] = d;
                    matrix[j*num_clusters + i] = d;
                    if (d > max_dist) max_dist = d;
                }
            }
        }
        fclose(f);
    }

    double cell_w = (double)w / num_clusters;
    double cell_h = (double)h / num_clusters;
    
    for (int i=0; i<num_clusters; i++) {
        for (int j=0; j<num_clusters; j++) {
            double d = matrix[i*num_clusters + j];
            int px = x + (int)(j * cell_w);
            // Invert Y: i=0 at bottom
            int py = y + h - (int)((i+1) * cell_h);
            
            int pw = (int)((j+1)*cell_w) - (int)(j*cell_w);
            int ph = (int)((i+1)*cell_h) - (int)(i*cell_h);
            if (pw < 1) pw = 1; 
            if (ph < 1) ph = 1;

            ColorRGB col;
            if (d < 0) { 
                if (i==j) { col.r=255; col.g=255; col.b=255; }
                else { col.r=255; col.g=0; col.b=0; }
            } else {
                unsigned char val = (unsigned char)(255.0 * (d / (max_dist > 0 ? max_dist : 1.0)));
                col.r = val; col.g = val; col.b = val;
            }
            draw_filled_rect(c, px, py, pw, ph, col);
            
            if (num_clusters < 25 && d >= 0) {
                char txt[32];
                snprintf(txt, sizeof(txt), "%.2f", d);
                ColorRGB txt_col = (col.r > 128) ? (ColorRGB){0,0,0} : (ColorRGB){255,255,255};
                draw_string(c, px + pw/2 - 10, py + ph/2 - 3, txt, txt_col, 0, 1, 0);
            }
        }
    }
    
    // Axes
    ColorRGB black = {0,0,0};
    draw_line(c, x, y, x, y+h, black);
    draw_line(c, x, y+h, x+w, y+h, black);
    draw_string(c, x-15, y+h-5, "0", black, 0, 1, 0);
    draw_string(c, x, y+h+15, "0", black, 0, 1, 0);
    char n_str[32];
    snprintf(n_str, sizeof(n_str), "%d", num_clusters);
    draw_string(c, x-25, y+5, n_str, black, 0, 1, 0);
    draw_string(c, x+w-10, y+h+15, n_str, black, 0, 1, 0);
    
    // Colorbar (Right side, vertical)
    int cb_x = x + w + 10;
    int cb_w = 10;
    for (int j=0; j<h; j++) {
        // 0 (black) at bottom, Max (white) at top
        // Screen Y increases down. y+h is bottom.
        unsigned char val = (unsigned char)(255.0 * (1.0 - (double)j / h));
        ColorRGB col = {val, val, val};
        draw_filled_rect(c, cb_x, y+j, cb_w, 1, col);
    }
    draw_string(c, cb_x + 15, y + h, "0", black, 0, 1, 0);
    char max_s[32];
    snprintf(max_s, sizeof(max_s), "%.2f", max_dist);
    draw_string(c, cb_x + 15, y + 10, max_s, black, 0, 1, 0);

    free(matrix);
}

void draw_scale(Canvas *c, int x, int y) {
    double units = 0.5;
    int len_px = (int)(units / VIEW_RANGE * 800); // PLOT_WIDTH
    ColorRGB black = {0,0,0};
    draw_line(c, x, y, x + len_px, y, black);
    draw_line(c, x, y-5, x, y+5, black);
    draw_line(c, x + len_px, y-5, x + len_px, y+5, black);
    draw_string(c, x + len_px/2 - 10, y + 15, "0.5", black, 0, 1, 1);
}
#endif

void print_help(const char *progname) {
    printf("Usage: %s [options] <points_file> <log_file> [output_file]\n", progname);
    printf("Description:\n");
    printf("  Visualizes clustering results by combining original points with membership info from log.\n");
    printf("Arguments:\n");
    printf("  <points_file>     Original input text file (coordinates).\n");
    printf("  <log_file>        Log file created by image-cluster (contains stats and output dir).\n");
    printf("  [output_file]     Optional output filename.\n\n");
    printf("Options:\n");
    printf("  -h, --help        Show this help message.\n");
    printf("  -svg              Output SVG image instead of PNG (default: PNG).\n");
    printf("  -fs <size>        Set font size for text labels (default: 18.0).\n");
}

int main(int argc, char *argv[]) {
    char *points_filename = NULL;
    char *log_filename = NULL;
    char output_filename[1024] = {0};
    int png_mode = 1;
    double font_size = 18.0;

    int arg_idx = 1;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[arg_idx], "-svg") == 0) {
            png_mode = 0;
        } else if (strcmp(argv[arg_idx], "-fs") == 0) {
            if (arg_idx + 1 < argc) {
                font_size = atof(argv[++arg_idx]);
                if (font_size < 1.0) font_size = 24.0;
            } else {
                fprintf(stderr, "Error: -fs requires an argument.\n");
                return 1;
            }
        } else if (argv[arg_idx][0] == '-') {
             fprintf(stderr, "Unknown option: %s\n", argv[arg_idx]);
             print_help(argv[0]);
             return 1;
        } else {
             if (!points_filename) {
                 points_filename = argv[arg_idx];
             } else if (!log_filename) {
                 log_filename = argv[arg_idx];
             } else if (strlen(output_filename) == 0) {
                 strncpy(output_filename, argv[arg_idx], sizeof(output_filename)-1);
             } else {
                 fprintf(stderr, "Error: Too many arguments.\n");
                 print_help(argv[0]);
                 return 1;
             }
        }
        arg_idx++;
    }

    if (!points_filename || !log_filename) {
        if (argc > 1) fprintf(stderr, "Error: Missing input files.\n");
        print_help(argv[0]);
        return 1;
    }

    HeaderLine stats[100];
    int num_stats = 0;
    double rlim = 0.0;
    double dprob = 0.01;
    int maxcl = 1000;
    long maxim = 100000;
    int gprob = 0, te4 = 0, te5 = 0;
    double fmatcha = 2.0, fmatchb = 0.5;

    char output_dir[4096] = {0};
    char dcc_filename[4096] = {0};
    long total_frames = 0, total_clusters = 0, total_dists = 0;
    long *hist_data = (long *)calloc(10000, sizeof(long));
    int hist_parsing = 0;

    printf("Parsing log file: %s\n", log_filename);
    FILE *flog = fopen(log_filename, "r");
    if (!flog) { perror("Error opening log file"); return 1; }
    
    char line[4096];
    while (fgets(line, sizeof(line), flog)) {
        if (hist_parsing) {
            if (strncmp(line, "STATS_DIST_HIST_END", 19) == 0) {
                hist_parsing = 0;
            } else {
                int k;
                long count, pruned;
                if (sscanf(line, "%d %ld %ld", &k, &count, &pruned) >= 2) {
                    if (k >= 0 && k < 10000) hist_data[k] = count;
                }
            }
            continue;
        }

        if (strncmp(line, "OUTPUT_DIR: ", 12) == 0) {
            char *p = line + 12;
            if (p[strlen(p)-1] == '\n') p[strlen(p)-1] = '\0';
            strcpy(output_dir, p);
        } else if (strncmp(line, "OUTPUT_FILE: ", 13) == 0) {
            char *p = line + 13;
            if (p[strlen(p)-1] == '\n') p[strlen(p)-1] = '\0';
            if (strstr(p, "dcc.txt")) strcpy(dcc_filename, p);
        } else if (strncmp(line, "PARAM_RLIM: ", 12) == 0) { sscanf(line + 12, "%lf", &rlim);
        } else if (strncmp(line, "PARAM_DPROB: ", 13) == 0) { sscanf(line + 13, "%lf", &dprob);
        } else if (strncmp(line, "PARAM_MAXCL: ", 13) == 0) { sscanf(line + 13, "%d", &maxcl);
        } else if (strncmp(line, "PARAM_MAXIM: ", 13) == 0) { sscanf(line + 13, "%ld", &maxim);
        } else if (strncmp(line, "PARAM_GPROB: ", 13) == 0) { sscanf(line + 13, "%d", &gprob);
        } else if (strncmp(line, "PARAM_FMATCHA: ", 15) == 0) { sscanf(line + 15, "%lf", &fmatcha);
        } else if (strncmp(line, "PARAM_FMATCHB: ", 15) == 0) { sscanf(line + 15, "%lf", &fmatchb);
        } else if (strncmp(line, "PARAM_TE4: ", 11) == 0) { sscanf(line + 11, "%d", &te4);
        } else if (strncmp(line, "PARAM_TE5: ", 11) == 0) { sscanf(line + 11, "%d", &te5);
        } else if (strncmp(line, "STATS_DIST_HIST_START", 21) == 0) {
            hist_parsing = 1;
        } else if (strncmp(line, "STATS_", 6) == 0) {
            char *key = line + 6;
            char *val = strchr(key, ':');
            if (val) {
                *val = '\0';
                val++;
                while (*val == ' ') val++;
                if (val[strlen(val)-1] == '\n') val[strlen(val)-1] = '\0';
                
                if (strcmp(key, "CLUSTERS") == 0) total_clusters = atol(val);
                else if (strcmp(key, "FRAMES") == 0) total_frames = atol(val);
                else if (strcmp(key, "DISTS") == 0) total_dists = atol(val);
            }
        }
    }
    fclose(flog);
    
    snprintf(stats[0].text, 255, "%ld fr -> %ld cl (%ld dist)", total_frames, total_clusters, total_dists);
    
    char param_str[1024];
    int p_off = snprintf(param_str, sizeof(param_str), "Params: R=%.3f", rlim);
    if (dprob != 0.01) p_off += snprintf(param_str + p_off, sizeof(param_str)-p_off, ", dprob=%.3f", dprob);
    if (gprob) p_off += snprintf(param_str + p_off, sizeof(param_str)-p_off, ", gprob=ON");
    if (te4) p_off += snprintf(param_str + p_off, sizeof(param_str)-p_off, ", te4=ON");
    if (te5) p_off += snprintf(param_str + p_off, sizeof(param_str)-p_off, ", te5=ON");
    strncpy(stats[1].text, param_str, 255);
    num_stats = 2;

    if (strlen(output_dir) == 0) {
        fprintf(stderr, "Error: Could not find OUTPUT_DIR in log.\n");
        return 1;
    }

    if (strlen(output_filename) == 0) {
        strncpy(output_filename, points_filename, sizeof(output_filename) - 5);
        char *ext = strrchr(output_filename, '.');
        if (ext) *ext = '\0';
        strcat(output_filename, png_mode ? ".png" : ".svg");
    }

    FILE *f_pts = fopen(points_filename, "r");
    if (!f_pts) { perror("Error opening points file"); return 1; }

    char memb_filename[4096];
    snprintf(memb_filename, sizeof(memb_filename), "%s/frame_membership.txt", output_dir);
    FILE *f_memb = fopen(memb_filename, "r");
    if (!f_memb) { perror("Error opening membership file"); fclose(f_pts); return 1; }

    // Init output buffers
    FILE *svg_out = NULL;
    #ifdef USE_PNG
    Canvas *canvas = NULL;
    #endif

    if (png_mode) {
        #ifdef USE_PNG
        canvas = init_canvas(SVG_WIDTH, SVG_HEIGHT);
        ColorRGB grid_col = {0, 0, 0};
        int cx = (int)map_x(0);
        int cy = (int)map_y(0);
        draw_line(canvas, 0, cy, SVG_WIDTH, cy, grid_col);
        draw_line(canvas, cx, 0, cx, SVG_HEIGHT, grid_col);

        ColorRGB gray = {128, 128, 128};
        int bx1 = (int)map_x(-1), by1 = (int)map_y(1);
        int bx2 = (int)map_x(1), by2 = (int)map_y(-1);
        draw_line(canvas, bx1, by1, bx2, by1, gray);
        draw_line(canvas, bx2, by1, bx2, by2, gray);
        draw_line(canvas, bx2, by2, bx1, by2, gray);
        draw_line(canvas, bx1, by2, bx1, by1, gray);
        #endif
    } else {
        svg_out = fopen(output_filename, "w");
        if (!svg_out) {
            perror("Error opening output file");
            fclose(f_pts);
            fclose(f_memb);
            return 1;
        }
        fprintf(svg_out, "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
        fprintf(svg_out, "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">\n", SVG_WIDTH, SVG_HEIGHT);
        fprintf(svg_out, "<rect width=\"100%%\" height=\"100%%\" fill=\"white\" />\n");

        double cx = map_x(0);
        double cy = map_y(0);
        fprintf(svg_out, "<line x1=\"0\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" stroke=\"black\" stroke-width=\"1\" />\n", cy, SVG_WIDTH, cy);
        fprintf(svg_out, "<line x1=\"%.2f\" y1=\"0\" x2=\"%.2f\" y2=\"%d\" stroke=\"black\" stroke-width=\"1\" />\n", cx, cx, SVG_HEIGHT);

        double bx1 = map_x(-1);
        double by1 = map_y(1);
        double bx2 = map_x(1);
        double by2 = map_y(-1);
        fprintf(svg_out, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"none\" stroke=\"gray\" stroke-dasharray=\"5,5\" />\n",
                bx1, by1, bx2 - bx1, by2 - by1);
    }

    Anchor anchors[10000];
    int num_anchors = 0;
    
    char *cluster_seen = (char *)calloc(10000, 1);
    long *samples_per_cluster = (long *)calloc(10000, sizeof(long));

    char line_pts[4096];
    char line_memb[1024];
    
    while (1) {
        if (!fgets(line_pts, sizeof(line_pts), f_pts)) break;
        
        char *p = line_pts;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '#') continue;

        if (!fgets(line_memb, sizeof(line_memb), f_memb)) break;

        long m_idx;
        int cluster_id;
        if (sscanf(line_memb, "%ld %d", &m_idx, &cluster_id) != 2) continue;

        if (cluster_id >= 0 && cluster_id < 10000) samples_per_cluster[cluster_id]++;

        double x, y;
        if (sscanf(line_pts, "%lf %lf", &x, &y) < 2) continue;

        double sx = map_x(x);
        double sy = map_y(y);

        const char *hex = colors[abs(cluster_id) % NUM_COLORS];
        if (cluster_id < 0) hex = "#000000";

        if (png_mode) {
            #ifdef USE_PNG
            ColorRGB col = parse_color(hex);
            draw_filled_circle(canvas, (int)sx, (int)sy, 3, col);
            #endif
        } else {
            fprintf(svg_out, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"3\" fill=\"%s\" opacity=\"0.7\" />\n", sx, sy, hex);
        }

        if (cluster_id >= 0 && cluster_id < 10000) {
            if (!cluster_seen[cluster_id]) {
                if (num_anchors < 10000) {
                    anchors[num_anchors].id = cluster_id;
                    anchors[num_anchors].x = x;
                    anchors[num_anchors].y = y;
                    num_anchors++;
                }
                cluster_seen[cluster_id] = 1;
            }
        }
    }
    free(cluster_seen);

    double r_px = (rlim / VIEW_RANGE) * PLOT_WIDTH;
    for (int i = 0; i < num_anchors; i++) {
        double ax = map_x(anchors[i].x);
        double ay = map_y(anchors[i].y);

        if (png_mode) {
            #ifdef USE_PNG
            ColorRGB black = {0,0,0};
            draw_circle(canvas, (int)ax, (int)ay, (int)r_px, black);

            draw_line(canvas, (int)ax-5, (int)ay, (int)ax+5, (int)ay, black);
            draw_line(canvas, (int)ax, (int)ay-5, (int)ax, (int)ay+5, black);
            #endif
        } else {
            fprintf(svg_out, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" stroke=\"black\" fill=\"none\" stroke-width=\"1.5\" />\n", ax, ay, r_px);
            fprintf(svg_out, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"black\" stroke-width=\"2\" />\n",
                    ax - 5, ay, ax + 5, ay);
            fprintf(svg_out, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"black\" stroke-width=\"2\" />\n",
                    ax, ay - 5, ax, ay + 5);
        }
    }

    int text_y = 20;
    int line_height = (int)(font_size * 1.5);
    int scale = (int)(font_size / 10.0);
    if (scale < 1) scale = 1;

    if (png_mode) {
        #ifdef USE_PNG
        ColorRGB black = {0,0,0};
        int text_x = 810; // Right panel start
        for (int i = 0; i < num_stats; i++) {
            int bold = (strstr(stats[i].text, "Avg Dist/Frame") != NULL);
            draw_string(canvas, text_x, text_y, stats[i].text, black, 0, scale, bold);
            text_y += line_height;
        }
        
        draw_histogram(canvas, 850, 100, 300, 150, hist_data, 10000);
        draw_cluster_histogram(canvas, 850, 300, 300, 130, samples_per_cluster, (int)total_clusters);
        if (dcc_filename[0]) draw_dcc_matrix(canvas, 850, 450, 300, 300, dcc_filename, (int)total_clusters);
        draw_scale(canvas, 50, 750);

        save_png(canvas, output_filename);
        free_canvas(canvas);
        #endif
    } else {
        // ... (SVG logic unchanged, or I should free samples_per_cluster here too? No, it's freed at end of main usually, but I put free inside main scope?)
        // Wait, where is free(hist_data)? It's at end of main.
        // So I should free samples_per_cluster at end of main.
    }

    fclose(f_pts);
    fclose(f_memb);
    printf("Generated: %s\n", output_filename);
    free(hist_data);
    free(samples_per_cluster);

    return 0;
}

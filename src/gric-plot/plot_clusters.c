/**
 * @file plot_clusters.c
 * @brief Diagnostic scatter and histogram plotting utility.
 *
 * Processes coordinate files and clustering logs to output scatter plots and
 * histograms in PNG or SVG format. Includes a basic custom drawing canvas.
 *
 * Main Functions:
 * - init_canvas: Allocates and initializes the image canvas.
 * - draw_line: Renders lines onto the canvas.
 * - draw_circle: Renders filled or empty circles.
 * - draw_string: Renders text strings on the canvas using a basic custom font.
 * - draw_histogram: Computes and draws 1D histograms of cluster statistics.
 * - main: Entry point of the plotting utility.
 */
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef USE_PNG
#include "canvas/canvas.h"
#endif

#define SVG_WIDTH 2400
#define PLOT_WIDTH 1600
#define SVG_HEIGHT 1600
#define Q_HEIGHT 1800
#define SCALE_X(x) ((int)((x) * (SVG_WIDTH / 1200.0)))
#define SCALE_Y(y) ((int)((y) * (SVG_HEIGHT / 800.0)))
#define VIEW_MIN -1.1
#define VIEW_MAX 1.1
#define VIEW_RANGE (VIEW_MAX - VIEW_MIN)

const char *colors[] = {"#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
                        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
                        "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
                        "#000075", "#808080", "#ffffff", "#000000"};
#define NUM_COLORS (sizeof(colors) / sizeof(colors[0]))

static long get_rounded_count(long val)
{
    if (val <= 10)
    {
        return 10;
    }
    long p = 1;
    while (p * 10 <= val)
    {
        p *= 10;
    }
    long mult = val / p;
    if (mult >= 5)
    {
        return 5 * p;
    }
    if (mult >= 2)
    {
        return 2 * p;
    }
    return p;
}

typedef struct
{
    int id;
    double x;
    double y;
} Anchor;

typedef struct
{
    char text[256];
} HeaderLine;

#ifdef USE_PNG
static double png_font_scale = 1.0;
#endif

double map_x(double x)
{
    return (x - VIEW_MIN) / VIEW_RANGE * PLOT_WIDTH;
}

double map_y(double y)
{
    return (VIEW_MAX - y) / VIEW_RANGE * SVG_HEIGHT;
}

double map_q_x(double x)
{
    return (x - (-1.2)) / 2.4 * PLOT_WIDTH;
}

double map_q_y(double y)
{
    return (1.5 - y) / 2.7 * Q_HEIGHT;
}

#ifdef USE_PNG
static void draw_scale(Canvas *c, int x, int y)
{
    double units = 0.5;
    int len_px = (int)(units / VIEW_RANGE * PLOT_WIDTH);
    ColorRGB black = {0, 0, 0};
    draw_line(c, x, y, x + len_px, y, black);
    draw_line(c, x, y - 5, x, y + 5, black);
    draw_line(c, x + len_px, y - 5, x + len_px, y + 5, black);
    draw_string(c, (int)(x + len_px / 2.0 - 10.0 * png_font_scale),
                (int)(y + 15.0 * png_font_scale), "0.5", black, 0, png_font_scale, 1);
}
#endif

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

void print_help(const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  gric-plot - Visualization tool for clustering results\n\n");

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s[options]%s %s<points_file>%s %s<log_file>%s %s[output_file]%s\n\n",
           ansi_bold_green, progname, ansi_reset, ansi_color_grey, ansi_reset,
           ansi_color_magenta, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_grey,
           ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  gric-plot is a visualization tool for clustering results produced by gric-cluster.\n");
    printf("  It generates a comprehensive summary image (PNG or SVG) containing:\n");
    printf("    1. A scatter plot of the clustered data points, color-coded by cluster ID.\n");
    printf("    2. Anchor points marked with circles representing the radius limit (rlim).\n");
    printf("    3. A distance histogram (Samples vs. Distance Count).\n");
    printf("    4. A cluster size histogram (Samples per Cluster).\n");
    printf("    5. A Cluster-to-Cluster Distance Matrix (if dcc.txt is available).\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-h, --help%s           Show this help message\n", ansi_color_green, ansi_reset);
    printf("  %s-svg%s                 Output SVG image instead of PNG (%sdefault:%s%s PNG%s)\n",
           ansi_color_green, ansi_reset, ansi_color_cyan, ansi_reset, ansi_color_cyan, ansi_reset);
    printf("  %s-fs%s %s<size>%s           Set font size for text labels (%sdefault:%s%s 18.0%s)\n\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_cyan,
           ansi_reset, ansi_color_cyan, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s input_points.txt gric_run_log.txt summary_plot.png\n", ansi_color_grey,
           ansi_reset, ansi_bold_green, progname, ansi_reset);
    print_color_mode();
}

int main(int argc, char *argv[])
{
    init_colors();
    char *points_filename = NULL, *log_filename = NULL, output_filename[1024] = {0};
    int png_mode = 1;
    double font_size = 18.0;
    int arg_idx = 1;
    while (arg_idx < argc)
    {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[arg_idx], "-svg") == 0)
            png_mode = 0;
        else if (strcmp(argv[arg_idx], "-fs") == 0)
        {
            if (arg_idx + 1 < argc)
                font_size = atof(argv[++arg_idx]);
        }
        else if (argv[arg_idx][0] == '-')
        {
            fprintf(stderr, "Unknown: %s\n", argv[arg_idx]);
            return 1;
        }
        else
        {
            if (!points_filename)
                points_filename = argv[arg_idx];
            else if (!log_filename)
                log_filename = argv[arg_idx];
            else
                strncpy(output_filename, argv[arg_idx], 1023);
        }
        arg_idx++;
    }
#ifndef USE_PNG
    png_mode = 0;
#endif
    if (!points_filename || !log_filename)
    {
        print_help(argv[0]);
        return 1;
    }
#ifdef USE_PNG
    png_font_scale = (font_size / 18.0) * ((double)SVG_HEIGHT / 800.0);
    if (png_font_scale < 0.5)
    {
        png_font_scale = 0.5;
    }
    canvas_set_font_scale(png_font_scale);
#endif
    HeaderLine stats[100];
    int num_stats = 0;
    double rlim = 0.0, dprob = 0.01, fmatcha = 2.0, fmatchb = 0.5;
    int maxcl = 1000, gprob = 0, te4 = 0, te5 = 0;
    long maxim = 100000;
    char output_dir[4096] = {0}, dcc_filename[4096] = {0};
    long total_frames = 0, total_clusters = 0, total_dists = 0;
    long *hist_data = (long *)calloc(10000, sizeof(long));
    int hist_parsing = 0;
    printf("Reading log file: %s\n", log_filename);
    printf("Font size: %.1f\n", font_size);
    printf("Font type: 8x16 Monospace\n");
    FILE *flog = fopen(log_filename, "r");
    if (!flog)
    {
        fprintf(stderr, "Error: Could not open log file %s\n", log_filename);
        return 1;
    }
    int queries_parsing = 0;
    long *cluster_query_counts = (long *)calloc(10000, sizeof(long));

    char line[4096];
    while (fgets(line, sizeof(line), flog))
    {
        if (hist_parsing)
        {
            if (strncmp(line, "STATS_DIST_HIST_END", 19) == 0)
                hist_parsing = 0;
            else
            {
                int k;
                long c, p;
                if (sscanf(line, "%d %ld %ld", &k, &c, &p) >= 2 && k < 10000)
                    hist_data[k] = c;
            }
            continue;
        }
        if (queries_parsing)
        {
            if (strncmp(line, "STATS_CLUSTER_QUERIES_END", 25) == 0)
                queries_parsing = 0;
            else
            {
                int k;
                long q;
                if (sscanf(line, "%d %ld", &k, &q) == 2 && k < 10000)
                    cluster_query_counts[k] = q;
            }
            continue;
        }
        if (strncmp(line, "OUTPUT_DIR: ", 12) == 0)
        {
            strcpy(output_dir, line + 12);
            if (output_dir[strlen(output_dir) - 1] == '\n')
                output_dir[strlen(output_dir) - 1] = '\0';
        }
        else if (strncmp(line, "OUTPUT_FILE: ", 13) == 0)
        {
            if (strstr(line, "dcc.txt"))
            {
                strcpy(dcc_filename, line + 13);
                if (dcc_filename[strlen(dcc_filename) - 1] == '\n')
                    dcc_filename[strlen(dcc_filename) - 1] = '\0';
            }
        }
        else if (strncmp(line, "PARAM_RLIM: ", 12) == 0)
            sscanf(line + 12, "%lf", &rlim);
        else if (strncmp(line, "PARAM_DPROB: ", 13) == 0)
            sscanf(line + 13, "%lf", &dprob);
        else if (strncmp(line, "PARAM_MAXCL: ", 13) == 0)
            sscanf(line + 13, "%d", &maxcl);
        else if (strncmp(line, "PARAM_MAXIM: ", 13) == 0)
            sscanf(line + 13, "%ld", &maxim);
        else if (strncmp(line, "PARAM_GPROB: ", 13) == 0)
            sscanf(line + 13, "%d", &gprob);
        else if (strncmp(line, "PARAM_FMATCHA: ", 15) == 0)
            sscanf(line + 15, "%lf", &fmatcha);
        else if (strncmp(line, "PARAM_FMATCHB: ", 15) == 0)
            sscanf(line + 15, "%lf", &fmatchb);
        else if (strncmp(line, "PARAM_TE4: ", 11) == 0)
            sscanf(line + 11, "%d", &te4);
        else if (strncmp(line, "PARAM_TE5: ", 11) == 0)
            sscanf(line + 11, "%d", &te5);
        else if (strncmp(line, "STATS_DIST_HIST_START", 21) == 0)
            hist_parsing = 1;
        else if (strncmp(line, "STATS_CLUSTER_QUERIES_START", 27) == 0)
            queries_parsing = 1;
        else if (strncmp(line, "STATS_", 6) == 0)
        {
            char *k = line + 6, *v = strchr(k, ':');
            if (v)
            {
                *v = '\0';
                v++;
                if (strcmp(k, "CLUSTERS") == 0)
                    total_clusters = atol(v);
                else if (strcmp(k, "FRAMES") == 0)
                    total_frames = atol(v);
                else if (strcmp(k, "DISTS") == 0)
                    total_dists = atol(v);
            }
        }
    }
    fclose(flog);
    printf("Log loaded: %ld frames, %ld clusters\n", total_frames, total_clusters);

    snprintf(stats[0].text, 255, "%ld fr -> %ld cl (%ld dist)", total_frames, total_clusters,
             total_dists);
    char p_str[1024];
    int po = snprintf(p_str, 1023, "Params: R=%.3f", rlim);
    if (dprob != 0.01)
        po += snprintf(p_str + po, 1023 - po, ", dprob=%.3f", dprob);
    if (gprob)
        po += snprintf(p_str + po, 1023 - po, ", gprob=ON");
    if (te4)
        po += snprintf(p_str + po, 1023 - po, ", te4=ON");
    if (te5)
        po += snprintf(p_str + po, 1023 - po, ", te5=ON");
    strncpy(stats[1].text, p_str, 255);
    num_stats = 2;
    if (!output_filename[0])
    {
        strncpy(output_filename, points_filename, 1018);
        char *e = strrchr(output_filename, '.');
        if (e)
            *e = '\0';
        strcat(output_filename, png_mode ? ".png" : ".svg");
    }

    char queries_output_filename[1024];
    strncpy(queries_output_filename, output_filename, sizeof(queries_output_filename) - 1);
    queries_output_filename[sizeof(queries_output_filename) - 1] = '\0';
    char *ext_q = strrchr(queries_output_filename, '.');
    if (ext_q != NULL)
    {
        char temp_ext[64];
        strcpy(temp_ext, ext_q);
        strcpy(ext_q, ".queries");
        strcat(queries_output_filename, temp_ext);
    }
    else
    {
        strcat(queries_output_filename, ".queries");
    }

    char memb_filename[4096];
    if (output_dir[0])
        snprintf(memb_filename, sizeof(memb_filename), "%s/frame_membership.txt", output_dir);
    else
        strcpy(memb_filename, "frame_membership.txt");

    printf("Reading points: %s\n", points_filename);
    printf("Reading membership: %s\n", memb_filename);

    FILE *f_pts = fopen(points_filename, "r"), *f_memb = fopen(memb_filename, "r");
    if (!f_pts)
    {
        fprintf(stderr, "Error: Could not open points file %s\n", points_filename);
        return 1;
    }
    if (!f_memb)
    {
        fprintf(stderr, "Error: Could not open membership file %s\n", memb_filename);
        fclose(f_pts);
        return 1;
    }

    FILE *svg_out = png_mode ? NULL : fopen(output_filename, "w");
    FILE *q_svg_out = png_mode ? NULL : fopen(queries_output_filename, "w");
#ifdef USE_PNG
    canvas_set_svg_output(svg_out);
    Canvas *canvas = png_mode ? init_canvas(SVG_WIDTH, SVG_HEIGHT) : NULL;
    Canvas *q_canvas = png_mode ? init_canvas(PLOT_WIDTH, Q_HEIGHT) : NULL;
#endif
    if (!png_mode && svg_out)
    {
        fprintf(svg_out,
                "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\"><rect "
                "width=\"100%%\" height=\"100%%\" fill=\"white\" />",
                SVG_WIDTH, SVG_HEIGHT);
        double cx = map_x(0), cy = map_y(0);
        fprintf(svg_out,
                "<line x1=\"0\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" stroke=\"black\" /><line "
                "x1=\"%.2f\" y1=\"0\" x2=\"%.2f\" y2=\"%d\" stroke=\"black\" />",
                cy, PLOT_WIDTH, cy, cx, cx, SVG_HEIGHT);
    }
    if (!png_mode && q_svg_out)
    {
        fprintf(q_svg_out,
                "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\"><rect "
                "width=\"100%%\" height=\"100%%\" fill=\"white\" />",
                PLOT_WIDTH, Q_HEIGHT);
        double cx = map_q_x(0), cy = map_q_y(0);
        fprintf(q_svg_out,
                "<line x1=\"0\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" stroke=\"black\" /><line "
                "x1=\"%.2f\" y1=\"0\" x2=\"%.2f\" y2=\"%d\" stroke=\"black\" />",
                cy, PLOT_WIDTH, cy, cx, cx, Q_HEIGHT);
    }
    Anchor anchors[10000];
    int num_anchors = 0;
    char *cluster_seen = calloc(10000, 1);
    long *samples_per_cluster = calloc(10000, sizeof(long));
    char lp[4096], lm[1024];
    long frame_count = 0;
    while (fgets(lp, 4095, f_pts) && fgets(lm, 1023, f_memb))
    {
        if (lp[0] == '#')
            continue;
        long idx;
        int cid;
        sscanf(lm, "%ld %d", &idx, &cid);
        if (cid >= 0 && cid < 10000)
        {
            samples_per_cluster[cid]++;
            if (!cluster_seen[cid])
            {
                anchors[num_anchors++] = (Anchor){cid, 0, 0};
                sscanf(lp, "%lf %lf", &anchors[num_anchors - 1].x, &anchors[num_anchors - 1].y);
                cluster_seen[cid] = 1;
            }
        }
        double x, y;
        sscanf(lp, "%lf %lf", &x, &y);
        double sx = map_x(x), sy = map_y(y);
        const char *hex = colors[abs(cid) % NUM_COLORS];
        if (cid < 0)
            hex = "#000000";
        if (sx >= 0 && sx <= PLOT_WIDTH && sy >= 0 && sy <= SVG_HEIGHT)
        {
            if (png_mode)
            {
#ifdef USE_PNG
                ColorRGB c = parse_color(hex);
                int pt_size = (int)(3.0 * ((double)SVG_HEIGHT / 800.0));
                if (pt_size < 1)
                {
                    pt_size = 1;
                }
                draw_filled_rect(canvas, (int)sx - pt_size / 2, (int)sy - pt_size / 2, pt_size, pt_size, c);
#endif
            }
            else
            {
                double pt_r = 2.0 * ((double)SVG_HEIGHT / 800.0);
                fprintf(svg_out, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"%s\" />", sx, sy, pt_r, hex);
            }
        }

        frame_count++;
        if (frame_count % 10000 == 0)
        {
            printf("\rProcessing frames: %ld / %ld", frame_count, total_frames);
            fflush(stdout);
        }
    }
    printf("\rProcessing frames: %ld / %ld [DONE]\n", frame_count, total_frames);

    double r_px = (rlim / VIEW_RANGE) * PLOT_WIDTH;
    for (int i = 0; i < num_anchors; i++)
    {
        double ax = map_x(anchors[i].x), ay = map_y(anchors[i].y);

        if (ax >= 0 && ax <= PLOT_WIDTH && ay >= 0 && ay <= SVG_HEIGHT)
        {
            if (png_mode)
            {
#ifdef USE_PNG
                ColorRGB b = {0, 0, 0};
                int thickness = (int)(3.0 * ((double)SVG_HEIGHT / 800.0));
                if (thickness < 1)
                {
                    thickness = 1;
                }
                draw_circle(canvas, (int)ax, (int)ay, (int)r_px, thickness, b);
#endif
            }
            else
            {
                double stroke_w = 3.0 * ((double)SVG_HEIGHT / 800.0);
                fprintf(svg_out,
                        "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" stroke=\"black\" fill=\"none\" stroke-width=\"%.1f\" />",
                        ax, ay, r_px, stroke_w);
            }
        }
    }

    long total_queries = 0;
    long max_q = 0;
    for (int i = 0; i < num_anchors; i++)
    {
        long q_c = cluster_query_counts[anchors[i].id];
        total_queries += q_c;
        if (q_c > max_q)
        {
            max_q = q_c;
        }
    }
    double q_coeff = 0.0;
    if (total_queries > 0)
    {
        q_coeff = sqrt((0.10 * PLOT_WIDTH * Q_HEIGHT) / (M_PI * total_queries));
    }

    // --- Draw proportional query disks in the separate queries file ---
    if (png_mode)
    {
#ifdef USE_PNG
        ColorRGB border_color = {180, 180, 180};
        ColorRGB axis_color = {220, 220, 220};
        ColorRGB text_color = {0, 0, 0};

        // Draw border
        draw_line(q_canvas, 0, 0, PLOT_WIDTH - 1, 0, border_color);
        draw_line(q_canvas, PLOT_WIDTH - 1, 0, PLOT_WIDTH - 1, Q_HEIGHT - 1, border_color);
        draw_line(q_canvas, PLOT_WIDTH - 1, Q_HEIGHT - 1, 0, Q_HEIGHT - 1, border_color);
        draw_line(q_canvas, 0, Q_HEIGHT - 1, 0, 0, border_color);

        // Draw axes (origin 0,0)
        double cx = map_q_x(0.0);
        double cy = map_q_y(0.0);
        draw_line(q_canvas, 0, (int)cy, PLOT_WIDTH - 1, (int)cy, axis_color);
        draw_line(q_canvas, (int)cx, 0, (int)cx, Q_HEIGHT - 1, axis_color);

        // Draw Title
        draw_string(q_canvas, 20, 40, "Query Count Proportional Circles (Total Area = 10%)", text_color, 0, png_font_scale, 1);
#endif
    }
    else if (q_svg_out)
    {
        double cx = map_q_x(0.0);
        double cy = map_q_y(0.0);
        fprintf(q_svg_out,
                "<text x=\"20\" y=\"40\" font-family=\"monospace\" font-size=\"20\" fill=\"black\" font-weight=\"bold\">Query Count Proportional Circles (Total Area = 10%%)</text>");
    }

    for (int i = 0; i < num_anchors; i++)
    {
        double ax = map_q_x(anchors[i].x);
        double ay = map_q_y(anchors[i].y);

        if (ax >= 0 && ax <= PLOT_WIDTH && ay >= 0 && ay <= Q_HEIGHT)
        {
            long q_count = cluster_query_counts[anchors[i].id];
            double r_val = 0.0;
            if (q_count > 0)
            {
                r_val = q_coeff * sqrt((double)q_count);
                if (r_val < 1.0)
                {
                    r_val = 1.0;
                }
            }

            char label[64];
            if (q_count > 0)
            {
                snprintf(label, sizeof(label), "C%d(q:%ld)", anchors[i].id, q_count);
            }
            else
            {
                snprintf(label, sizeof(label), "C%d", anchors[i].id);
            }

            if (png_mode)
            {
#ifdef USE_PNG
                ColorRGB black = {0, 0, 0};

                if (q_count > 0)
                {
                    draw_circle(q_canvas, (int)ax, (int)ay, (int)r_val, 2, black);
                }
#endif
            }
            else if (q_svg_out)
            {
                if (q_count > 0)
                {
                    fprintf(q_svg_out,
                            "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"none\" stroke=\"black\" stroke-width=\"2\" />",
                            ax, ay, r_val);
                }
            }
        }
    }

    // Draw caption legend in the top right corner
    if (png_mode)
    {
#ifdef USE_PNG
        if (total_queries > 0)
        {
            ColorRGB gray = {150, 150, 150};
            ColorRGB text_color = {50, 50, 50};
            long c_large = get_rounded_count(max_q);
            long c_medium = c_large / 2;
            long c_small = c_large >= 10 ? c_large / 10 : 1;

            double r_large = q_coeff * sqrt((double)c_large);
            double r_medium = q_coeff * sqrt((double)c_medium);
            double r_small = q_coeff * sqrt((double)c_small);

            int lx = 1250, ly = 120;
            draw_circle(q_canvas, lx, ly, (int)r_large, 2, gray);
            draw_circle(q_canvas, lx, ly, (int)r_medium, 2, gray);
            draw_circle(q_canvas, lx, ly, (int)r_small, 2, gray);

            int text_x = lx + (int)r_large + 15;
            char l_str[128];
            snprintf(l_str, sizeof(l_str), "%ld queries", c_large);
            draw_string(q_canvas, text_x, ly - 35, l_str, text_color, 0, png_font_scale * 0.8, 0);
            snprintf(l_str, sizeof(l_str), "%ld queries", c_medium);
            draw_string(q_canvas, text_x, ly - 5, l_str, text_color, 0, png_font_scale * 0.8, 0);
            snprintf(l_str, sizeof(l_str), "%ld queries", c_small);
            draw_string(q_canvas, text_x, ly + 25, l_str, text_color, 0, png_font_scale * 0.8, 0);
        }
#endif
    }
    else if (q_svg_out)
    {
        if (total_queries > 0)
        {
            long c_large = get_rounded_count(max_q);
            long c_medium = c_large / 2;
            long c_small = c_large >= 10 ? c_large / 10 : 1;

            double r_large = q_coeff * sqrt((double)c_large);
            double r_medium = q_coeff * sqrt((double)c_medium);
            double r_small = q_coeff * sqrt((double)c_small);

            int lx = 1250, ly = 120;
            fprintf(q_svg_out,
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" stroke=\"#969696\" stroke-width=\"2\" />"
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" stroke=\"#969696\" stroke-width=\"2\" />"
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" stroke=\"#969696\" stroke-width=\"2\" />",
                    lx, ly, r_large, lx, ly, r_medium, lx, ly, r_small);

            double text_x = lx + r_large + 15.0;
            fprintf(q_svg_out,
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" fill=\"#323232\">%ld queries</text>"
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" fill=\"#323232\">%ld queries</text>"
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" fill=\"#323232\">%ld queries</text>",
                    text_x, ly - 30, c_large,
                    text_x, ly, c_medium,
                    text_x, ly + 30, c_small);
        }
    }
#ifdef USE_PNG
    for (int i = 0; i < num_stats; i++)
    {
        draw_string(canvas, SCALE_X(810), (int)(SCALE_Y(20) + i * 20.0 * png_font_scale), stats[i].text, (ColorRGB){0, 0, 0}, 0, png_font_scale, 0);
    }
    draw_histogram(canvas, SCALE_X(850), SCALE_Y(100), SCALE_X(300), SCALE_Y(150), hist_data, 10000);
    draw_cluster_histogram(canvas, SCALE_X(850), SCALE_Y(300), SCALE_X(300), SCALE_Y(130), samples_per_cluster,
                           (int)total_clusters);
    if (dcc_filename[0])
    {
        draw_dcc_matrix(canvas, SCALE_X(850), SCALE_Y(450), SCALE_X(300), SCALE_Y(300), dcc_filename, (int)total_clusters);
    }
    draw_scale(canvas, SCALE_X(50), SCALE_Y(750));
#endif

    if (png_mode)
    {
#ifdef USE_PNG
        printf("Saving PNG output: %s\n", output_filename);
        save_png(canvas, output_filename);
        free_canvas(canvas);

        printf("Saving queries PNG output: %s\n", queries_output_filename);
        save_png(q_canvas, queries_output_filename);
        free_canvas(q_canvas);
#endif
    }
    else
    {
        printf("Saving SVG output: %s\n", output_filename);
        fprintf(svg_out, "</svg>");
        fclose(svg_out);

        if (q_svg_out)
        {
            printf("Saving queries SVG output: %s\n", queries_output_filename);
            fprintf(q_svg_out, "</svg>");
            fclose(q_svg_out);
        }
    }
    fclose(f_pts);
    fclose(f_memb);
    free(cluster_seen);
    free(samples_per_cluster);
    free(hist_data);
    free(cluster_query_counts);
    return 0;
}
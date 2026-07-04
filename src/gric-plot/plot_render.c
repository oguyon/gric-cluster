/**
 * @file plot_render.c
 * @brief Implementation of SVG and PNG plotting and rendering functions.
 */

#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plot_internal.h"

#ifdef USE_PNG
#include "canvas/canvas.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
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

static const char *colors[] = {
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ffffff", "#000000"
};
#define NUM_COLORS (sizeof(colors) / sizeof(colors[0]))

#ifdef USE_PNG
static double png_font_scale = 1.0;
#endif

static double map_x(double x)
{
    return (x - VIEW_MIN) / VIEW_RANGE * PLOT_WIDTH;
}

static double map_y(double y)
{
    return (VIEW_MAX - y) / VIEW_RANGE * SVG_HEIGHT;
}

static double map_q_x(double x)
{
    return (x - (-1.2)) / 2.4 * PLOT_WIDTH;
}

static double map_q_y(double y)
{
    return (1.5 - y) / 2.7 * Q_HEIGHT;
}

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

#ifdef USE_PNG
/**
 * @brief Draw scale onto the canvas.
 *
 * @param c Canvas pointer.
 * @param x x-coordinate.
 * @param y y-coordinate.
 */
static void draw_scale(
    Canvas *c,
    int     x,
    int     y)
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

/**
 * @brief Render SVG or PNG plots from the parsed data.
 *
 * This function handles all coordinate mappings, color palettes, and SVG/PNG rendering.
 *
 * @param points_filename         Path to the points coordinate file.
 * @param output_filename         Path to the output image file.
 * @param queries_output_filename Path to the queries output image file.
 * @param data                    Pointer to the parsed PlotData.
 * @param png_mode                1 for PNG output, 0 for SVG output.
 * @param font_size               Font size for text rendering.
 *
 * @return 0 on success, 1 on failure.
 */
int render_plots(
    const char     *points_filename,
    const char     *output_filename,
    const char     *queries_output_filename,
    const PlotData *data,
    int             png_mode,
    double          font_size)
{
#ifdef USE_PNG
    png_font_scale = (font_size / 18.0) * ((double)SVG_HEIGHT / 800.0);
    if (png_font_scale < 0.5)
    {
        png_font_scale = 0.5;
    }
    canvas_set_font_scale(png_font_scale);
#endif

    char memb_filename[8192];
    if (data->output_dir[0] != '\0')
    {
        snprintf(memb_filename, sizeof(memb_filename), "%s/frame_membership.txt",
                 data->output_dir);
    }
    else
    {
        strcpy(memb_filename, "frame_membership.txt");
    }

    FILE *f_pts = fopen(points_filename, "r");
    if (f_pts == NULL)
    {
        fprintf(stderr, "Error: Could not open points file %s\n", points_filename);
        return 1;
    }

    FILE *f_memb = fopen(memb_filename, "r");
    if (f_memb == NULL)
    {
        fprintf(stderr, "Error: Could not open membership file %s\n", memb_filename);
        fclose(f_pts);
        return 1;
    }

    FILE *svg_out = png_mode ? NULL : fopen(output_filename, "w");
    if (png_mode == 0 && svg_out == NULL)
    {
        fprintf(stderr, "Error: Could not open output file %s\n", output_filename);
        fclose(f_pts);
        fclose(f_memb);
        return 1;
    }

    FILE *q_svg_out = png_mode ? NULL : fopen(queries_output_filename, "w");
    if (png_mode == 0 && q_svg_out == NULL)
    {
        fprintf(stderr, "Error: Could not open output file %s\n", queries_output_filename);
        if (svg_out != NULL)
        {
            fclose(svg_out);
        }
        fclose(f_pts);
        fclose(f_memb);
        return 1;
    }

#ifdef USE_PNG
    canvas_set_svg_output(svg_out);
    Canvas *canvas = png_mode ? init_canvas(SVG_WIDTH, SVG_HEIGHT) : NULL;
    Canvas *q_canvas = png_mode ? init_canvas(PLOT_WIDTH, Q_HEIGHT) : NULL;
#endif

    if (png_mode == 0 && svg_out != NULL)
    {
        fprintf(svg_out,
                "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\"><rect "
                "width=\"100%%\" height=\"100%%\" fill=\"white\" />",
                SVG_WIDTH, SVG_HEIGHT);
        double cx = map_x(0);
        double cy = map_y(0);
        fprintf(svg_out,
                "<line x1=\"0\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" stroke=\"black\" /><line "
                "x1=\"%.2f\" y1=\"0\" x2=\"%.2f\" y2=\"%d\" stroke=\"black\" />",
                cy, PLOT_WIDTH, cy, cx, cx, SVG_HEIGHT);
    }

    if (png_mode == 0 && q_svg_out != NULL)
    {
        fprintf(q_svg_out,
                "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\"><rect "
                "width=\"100%%\" height=\"100%%\" fill=\"white\" />",
                PLOT_WIDTH, Q_HEIGHT);
        double cx = map_q_x(0);
        double cy = map_q_y(0);
        fprintf(q_svg_out,
                "<line x1=\"0\" y1=\"%.2f\" x2=\"%d\" y2=\"%.2f\" stroke=\"black\" /><line "
                "x1=\"%.2f\" y1=\"0\" x2=\"%.2f\" y2=\"%d\" stroke=\"black\" />",
                cy, PLOT_WIDTH, cy, cx, cx, Q_HEIGHT);
    }

    /* First pass: read files to plot coordinate points */
    {
        char lp[4096];
        char lm[1024];
        long frame_count = 0;

        while (fgets(lp, sizeof(lp), f_pts) != NULL && fgets(lm, sizeof(lm), f_memb) != NULL)
        {
            if (lp[0] == '#')
            {
                continue;
            }
            long idx;
            int cid;
            if (sscanf(lm, "%ld %d", &idx, &cid) == 2)
            {
                double x, y;
                if (sscanf(lp, "%lf %lf", &x, &y) == 2)
                {
                    double sx = map_x(x);
                    double sy = map_y(y);
                    const char *hex = colors[abs(cid) % NUM_COLORS];
                    if (cid < 0)
                    {
                        hex = "#000000";
                    }
                    if (sx >= 0 && sx <= PLOT_WIDTH && sy >= 0 && sy <= SVG_HEIGHT)
                    {
                        if (png_mode != 0)
                        {
#ifdef USE_PNG
                            ColorRGB c = parse_color(hex);
                            int pt_size = (int)(3.0 * ((double)SVG_HEIGHT / 800.0));
                            if (pt_size < 1)
                            {
                                pt_size = 1;
                            }
                            draw_filled_rect(canvas, (int)sx - pt_size / 2,
                                             (int)sy - pt_size / 2, pt_size, pt_size, c);
#endif
                        }
                        else
                        {
                            double pt_r = 2.0 * ((double)SVG_HEIGHT / 800.0);
                            fprintf(svg_out,
                                    "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"%s\" />",
                                    sx, sy, pt_r, hex);
                        }
                    }
                }
            } // if sscanf
            frame_count++;
            if (frame_count % 10000 == 0)
            {
                printf("\rProcessing frames: %ld / %ld", frame_count, data->total_frames);
                fflush(stdout);
            }
        } // while fgets
        printf("\rProcessing frames: %ld / %ld [DONE]\n", frame_count, data->total_frames);
    } // First pass

    /* Draw anchors and circles representing the radius limit */
    {
        double r_px = (data->rlim / VIEW_RANGE) * PLOT_WIDTH;
        for (int i = 0; i < data->num_anchors; i++)
        {
            double ax = map_x(data->anchors[i].x);
            double ay = map_y(data->anchors[i].y);

            if (ax >= 0 && ax <= PLOT_WIDTH && ay >= 0 && ay <= SVG_HEIGHT)
            {
                if (png_mode != 0)
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
                            "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" stroke=\"black\" "
                            "fill=\"none\" stroke-width=\"%.1f\" />",
                            ax, ay, r_px, stroke_w);
                }
            }
        }
    } // Draw anchors

    /* Calculate metrics for proportional query count circles */
    long total_queries = 0;
    long max_q = 0;
    {
        for (int i = 0; i < data->num_anchors; i++)
        {
            long q_c = data->cluster_query_counts[data->anchors[i].id];
            total_queries += q_c;
            if (q_c > max_q)
            {
                max_q = q_c;
            }
        }
    } // Calculate queries metrics

    double q_coeff = 0.0;
    if (total_queries > 0)
    {
        q_coeff = sqrt((0.10 * PLOT_WIDTH * Q_HEIGHT) / (M_PI * total_queries));
    }

    /* Initialize separate queries files layout */
    if (png_mode != 0)
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
        draw_string(q_canvas, 20, 40,
                    "Query Count Proportional Circles (Total Area = 10%)",
                    text_color, 0, png_font_scale, 1);
#endif
    }
    else if (q_svg_out != NULL)
    {
        fprintf(q_svg_out,
                "<text x=\"20\" y=\"40\" font-family=\"monospace\" font-size=\"20\" "
                "fill=\"black\" font-weight=\"bold\">Query Count Proportional Circles "
                "(Total Area = 10%%)</text>");
    }

    /* Draw proportional query count circles */
    {
        for (int i = 0; i < data->num_anchors; i++)
        {
            double ax = map_q_x(data->anchors[i].x);
            double ay = map_q_y(data->anchors[i].y);

            if (ax >= 0 && ax <= PLOT_WIDTH && ay >= 0 && ay <= Q_HEIGHT)
            {
                long q_count = data->cluster_query_counts[data->anchors[i].id];
                double r_val = 0.0;
                if (q_count > 0)
                {
                    r_val = q_coeff * sqrt((double)q_count);
                    if (r_val < 1.0)
                    {
                        r_val = 1.0;
                    }
                }

                if (png_mode != 0)
                {
#ifdef USE_PNG
                    ColorRGB black = {0, 0, 0};
                    if (q_count > 0)
                    {
                        draw_circle(q_canvas, (int)ax, (int)ay, (int)r_val, 2, black);
                    }
#endif
                }
                else if (q_svg_out != NULL)
                {
                    if (q_count > 0)
                    {
                        fprintf(q_svg_out,
                                "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" fill=\"none\" "
                                "stroke=\"black\" stroke-width=\"2\" />",
                                ax, ay, r_val);
                    }
                }
            }
        }
    } // Draw proportional query circles

    /* Draw caption legend in the top right corner */
    if (png_mode != 0)
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

            int lx = 1250;
            int ly = 120;
            draw_circle(q_canvas, lx, ly, (int)r_large, 2, gray);
            draw_circle(q_canvas, lx, ly, (int)r_medium, 2, gray);
            draw_circle(q_canvas, lx, ly, (int)r_small, 2, gray);

            int text_x = lx + (int)r_large + 15;
            char l_str[128];
            snprintf(l_str, sizeof(l_str), "%ld queries", c_large);
            draw_string(q_canvas, text_x, ly - 35, l_str, text_color, 0,
                        png_font_scale * 0.8, 0);
            snprintf(l_str, sizeof(l_str), "%ld queries", c_medium);
            draw_string(q_canvas, text_x, ly - 5, l_str, text_color, 0,
                        png_font_scale * 0.8, 0);
            snprintf(l_str, sizeof(l_str), "%ld queries", c_small);
            draw_string(q_canvas, text_x, ly + 25, l_str, text_color, 0,
                        png_font_scale * 0.8, 0);
        }
#endif
    }
    else if (q_svg_out != NULL)
    {
        if (total_queries > 0)
        {
            long c_large = get_rounded_count(max_q);
            long c_medium = c_large / 2;
            long c_small = c_large >= 10 ? c_large / 10 : 1;

            double r_large = q_coeff * sqrt((double)c_large);
            double r_medium = q_coeff * sqrt((double)c_medium);
            double r_small = q_coeff * sqrt((double)c_small);

            int lx = 1250;
            int ly = 120;
            fprintf(q_svg_out,
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" "
                    "stroke=\"#969696\" stroke-width=\"2\" />"
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" "
                    "stroke=\"#969696\" stroke-width=\"2\" />"
                    "<circle cx=\"%d\" cy=\"%d\" r=\"%.2f\" fill=\"none\" "
                    "stroke=\"#969696\" stroke-width=\"2\" />",
                    lx, ly, r_large,
                    lx, ly, r_medium,
                    lx, ly, r_small);

            double text_x = lx + r_large + 15.0;
            fprintf(q_svg_out,
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" "
                    "fill=\"#323232\">%ld queries</text>"
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" "
                    "fill=\"#323232\">%ld queries</text>"
                    "<text x=\"%.2f\" y=\"%d\" font-family=\"monospace\" font-size=\"14\" "
                    "fill=\"#323232\">%ld queries</text>",
                    text_x, ly - 30, c_large,
                    text_x, ly, c_medium,
                    text_x, ly + 30, c_small);
        }
    }

#ifdef USE_PNG
    /* Draw stats texts and histograms onto main canvas */
    {
        for (int i = 0; i < data->num_stats; i++)
        {
            draw_string(canvas, SCALE_X(810),
                        (int)(SCALE_Y(20) + i * 20.0 * png_font_scale),
                        data->stats[i].text, (ColorRGB){0, 0, 0}, 0, png_font_scale, 0);
        }
        draw_histogram(canvas, SCALE_X(850), SCALE_Y(100), SCALE_X(300), SCALE_Y(150),
                       (long *)data->hist_data, 10000);
        draw_cluster_histogram(canvas, SCALE_X(850), SCALE_Y(300), SCALE_X(300), SCALE_Y(130),
                               (long *)data->samples_per_cluster, (int)data->total_clusters);
        if (data->dcc_filename[0] != '\0')
        {
            draw_dcc_matrix(canvas, SCALE_X(850), SCALE_Y(450), SCALE_X(300), SCALE_Y(300),
                            data->dcc_filename, (int)data->total_clusters);
        }
        draw_scale(canvas, SCALE_X(50), SCALE_Y(750));
    } // Draw stats and histograms
#endif

    /* Save outputs and cleanup canvas */
    if (png_mode != 0)
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

        if (q_svg_out != NULL)
        {
            printf("Saving queries SVG output: %s\n", queries_output_filename);
            fprintf(q_svg_out, "</svg>");
            fclose(q_svg_out);
        }
    }

    fclose(f_pts);
    fclose(f_memb);
    return 0;
}

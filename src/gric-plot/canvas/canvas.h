/**
 * @file canvas.h
 * @brief Declarations for the custom graphics drawing canvas.
 *
 * Provides basic pixel manipulation, line, circle, text, histogram,
 * and DCC matrix drawing functions for PNG and SVG output.
 */

#ifndef PLOT_CANVAS_H
#define PLOT_CANVAS_H

#include <stdio.h>

#ifdef USE_PNG
typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} ColorRGB;

typedef struct
{
    int            width;
    int            height;
    unsigned char *data;
} Canvas;

Canvas *init_canvas(
    int w,
    int h);

void free_canvas(
    Canvas *c);

void set_pixel_opaque(
    Canvas   *c,
    int       x,
    int       y,
    ColorRGB  col);

void draw_filled_rect(
    Canvas   *c,
    int       x,
    int       y,
    int       w,
    int       h,
    ColorRGB  col);

void draw_line(
    Canvas   *c,
    int       x0,
    int       y0,
    int       x1,
    int       y1,
    ColorRGB  col);

void draw_circle(
    Canvas   *c,
    int       cx,
    int       cy,
    int       r,
    int       thickness,
    ColorRGB  col);

void draw_char(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale,
    int       bold);

void draw_string(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    int         align,
    double      scale,
    int         bold);

void draw_char_rotated90(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale);

void draw_string_rotated90(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    double      scale);

void draw_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count);

void draw_cluster_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count);

void draw_dcc_matrix(
    Canvas     *c,
    int         x,
    int         y,
    int         w,
    int         h,
    const char *dcc_file,
    int         num_clusters);

int save_png(
    Canvas     *c,
    const char *filename);

void canvas_set_svg_output(
    FILE *fp);

void canvas_set_font_scale(
    double scale);

ColorRGB parse_color(
    const char *hex);
#endif // USE_PNG

#endif // PLOT_CANVAS_H

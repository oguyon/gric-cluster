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

/**
 * init_canvas() - Allocate and initialize an RGB raster canvas.
 * @w: Canvas width in pixels.
 * @h: Canvas height in pixels.
 *
 * Return: Pointer to initialized Canvas, or NULL on allocation failure.
 */
Canvas *init_canvas(
    int w,
    int h);

/**
 * free_canvas() - Free pixel memory and canvas structure.
 * @c: Canvas pointer.
 */
void free_canvas(
    Canvas *c);

/**
 * set_pixel_opaque() - Set a single canvas pixel color.
 * @c:   Canvas pointer.
 * @x:   Horizontal pixel coordinate.
 * @y:   Vertical pixel coordinate.
 * @col: RGB color value.
 */
void set_pixel_opaque(
    Canvas   *c,
    int       x,
    int       y,
    ColorRGB  col);

/**
 * draw_filled_rect() - Draw solid color filled rectangle.
 * @c:   Canvas pointer.
 * @x:   Left coordinate.
 * @y:   Top coordinate.
 * @w:   Width in pixels.
 * @h:   Height in pixels.
 * @col: Fill RGB color.
 */
void draw_filled_rect(
    Canvas   *c,
    int       x,
    int       y,
    int       w,
    int       h,
    ColorRGB  col);

/**
 * draw_line() - Draw line segment using Bresenham's algorithm.
 * @c:   Canvas pointer.
 * @x0:  Start X coordinate.
 * @y0:  Start Y coordinate.
 * @x1:  End X coordinate.
 * @y1:  End Y coordinate.
 * @col: Line RGB color.
 */
void draw_line(
    Canvas   *c,
    int       x0,
    int       y0,
    int       x1,
    int       y1,
    ColorRGB  col);

/**
 * draw_circle() - Draw circle outline or filled ring.
 * @c:         Canvas pointer.
 * @cx:        Center X coordinate.
 * @cy:        Center Y coordinate.
 * @r:         Radius in pixels.
 * @thickness: Line thickness.
 * @col:       Circle RGB color.
 */
void draw_circle(
    Canvas   *c,
    int       cx,
    int       cy,
    int       r,
    int       thickness,
    ColorRGB  col);

/**
 * draw_char() - Render a single ASCII character onto canvas.
 * @c:     Canvas pointer.
 * @x:     Top-left X position.
 * @y:     Top-left Y position.
 * @ch:    Character byte.
 * @col:   Text RGB color.
 * @scale: Scaling multiplier.
 * @bold:  Non-zero for bold glyph rendering.
 */
void draw_char(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale,
    int       bold);

/**
 * draw_string() - Render null-terminated ASCII string onto canvas.
 * @c:     Canvas pointer.
 * @x:     Text origin X.
 * @y:     Text origin Y.
 * @str:   Text string.
 * @col:   Text RGB color.
 * @align: Alignment (-1: left, 0: center, 1: right).
 * @scale: Scaling multiplier.
 * @bold:  Non-zero for bold rendering.
 */
void draw_string(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    int         align,
    double      scale,
    int         bold);

/**
 * draw_char_rotated90() - Render single character rotated 90 degrees counter-clockwise.
 * @c:     Canvas pointer.
 * @x:     Origin X.
 * @y:     Origin Y.
 * @ch:    Character byte.
 * @col:   Text RGB color.
 * @scale: Scaling multiplier.
 */
void draw_char_rotated90(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale);

/**
 * draw_string_rotated90() - Render string rotated 90 degrees counter-clockwise.
 * @c:     Canvas pointer.
 * @x:     Origin X.
 * @y:     Origin Y.
 * @str:   Text string.
 * @col:   Text RGB color.
 * @scale: Scaling multiplier.
 */
void draw_string_rotated90(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    double      scale);

/**
 * draw_histogram() - Render a bar histogram on the canvas.
 * @c:     Canvas pointer.
 * @x:     Left coordinate.
 * @y:     Top coordinate.
 * @w:     Plot width.
 * @h:     Plot height.
 * @data:  Array of bin counts.
 * @count: Number of bins.
 */
void draw_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count);

/**
 * draw_cluster_histogram() - Render cluster population histogram.
 * @c:     Canvas pointer.
 * @x:     Left coordinate.
 * @y:     Top coordinate.
 * @w:     Plot width.
 * @h:     Plot height.
 * @data:  Array of cluster counts.
 * @count: Number of clusters.
 */
void draw_cluster_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count);

/**
 * draw_dcc_matrix() - Render inter-cluster distance heatmap matrix.
 * @c:            Canvas pointer.
 * @x:            Left coordinate.
 * @y:            Top coordinate.
 * @w:            Matrix plot width.
 * @h:            Matrix plot height.
 * @dcc_file:     Path to DCC matrix text file.
 * @num_clusters: Number of clusters in matrix.
 */
void draw_dcc_matrix(
    Canvas     *c,
    int         x,
    int         y,
    int         w,
    int         h,
    const char *dcc_file,
    int         num_clusters);

/**
 * save_png() - Encode canvas to PNG file on disk.
 * @c:        Canvas pointer.
 * @filename: Output PNG file path.
 *
 * Return: 0 on success, -1 on failure.
 */
int save_png(
    Canvas     *c,
    const char *filename);

/**
 * canvas_set_svg_output() - Enable companion SVG stream generation.
 * @fp: Open writable file pointer for SVG output, or NULL to disable.
 */
void canvas_set_svg_output(
    FILE *fp);

/**
 * canvas_set_font_scale() - Set global font scale factor for text drawing.
 * @scale: Scaling multiplier (1.0 = default).
 */
void canvas_set_font_scale(
    double scale);

/**
 * parse_color() - Parse hex RGB string ("#RRGGBB" or "RRGGBB") into ColorRGB.
 * @hex: Hex string.
 *
 * Return: Decoded ColorRGB structure.
 */
ColorRGB parse_color(
    const char *hex);
#endif // USE_PNG

#endif // PLOT_CANVAS_H

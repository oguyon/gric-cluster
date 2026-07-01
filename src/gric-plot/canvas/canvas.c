/**
 * @file canvas.c
 * @brief Implementation of custom graphics drawing canvas.
 *
 * Implements basic canvas drawing functions such as drawing rectangles, lines,
 * circles, rendering text, histograms, and saving to PNG format.
 */

#include "canvas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_PNG
#define FONT8x16_IMPLEMENTATION
#include "../font8x16.h"
#include <png.h>

static double png_font_scale = 1.0;
static FILE  *svg_out_file = NULL;

void canvas_set_svg_output(
    FILE *fp)
{
    svg_out_file = fp;
}

void canvas_set_font_scale(
    double scale)
{
    png_font_scale = scale;
}

ColorRGB parse_color(
    const char *hex)
{
    ColorRGB     c = {0, 0, 0};
    unsigned int r, g, b;

    if (hex[0] == '#')
    {
        hex++;
    }

    if (sscanf(hex, "%02x%02x%02x", &r, &g, &b) == 3)
    {
        c.r = r;
        c.g = g;
        c.b = b;
    }

    return c;
}

Canvas *init_canvas(
    int w,
    int h)
{
    Canvas *c = (Canvas *)malloc(sizeof(Canvas));

    c->width = w;
    c->height = h;
    c->data = (unsigned char *)malloc(w * h * 3);
    memset(c->data, 255, w * h * 3);

    return c;
}

void free_canvas(
    Canvas *c)
{
    if (c)
    {
        free(c->data);
        free(c);
    }
}

void set_pixel_opaque(
    Canvas   *c,
    int       x,
    int       y,
    ColorRGB  col)
{
    if (x < 0 || x >= c->width || y < 0 || y >= c->height)
    {
        return;
    }

    int idx = (y * c->width + x) * 3;
    c->data[idx] = col.r;
    c->data[idx + 1] = col.g;
    c->data[idx + 2] = col.b;
}

void draw_filled_rect(
    Canvas   *c,
    int       x,
    int       y,
    int       w,
    int       h,
    ColorRGB  col)
{
    if (c == NULL)
    {
        if (svg_out_file != NULL)
        {
            fprintf(svg_out_file,
                    "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
                    "fill=\"#%02x%02x%02x\" />\n",
                    x, y, w, h, col.r, col.g, col.b);
        }
        return;
    }

    for (int j = y; j < y + h; j++)
    {
        for (int i = x; i < x + w; i++)
        {
            set_pixel_opaque(c, i, j, col);
        }
    }
}

void draw_line(
    Canvas   *c,
    int       x0,
    int       y0,
    int       x1,
    int       y1,
    ColorRGB  col)
{
    if (c == NULL)
    {
        if (svg_out_file != NULL)
        {
            fprintf(svg_out_file,
                    "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" "
                    "stroke=\"#%02x%02x%02x\" stroke-width=\"1\" />\n",
                    x0, y0, x1, y1, col.r, col.g, col.b);
        }
        return;
    }

    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int e2;

    while (1)
    {
        set_pixel_opaque(c, x0, y0, col);
        if (x0 == x1 && y0 == y1)
        {
            break;
        }
        e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_circle(
    Canvas   *c,
    int       cx,
    int       cy,
    int       r,
    int       thickness,
    ColorRGB  col)
{
    if (c == NULL)
    {
        return;
    }

    for (int t = -thickness / 2; t <= thickness / 2; t++)
    {
        int cur_r = r + t;
        if (cur_r < 0)
        {
            continue;
        }

        int x = cur_r;
        int y = 0;
        int err = 0;

        while (x >= y)
        {
            set_pixel_opaque(c, cx + x, cy + y, col);
            set_pixel_opaque(c, cx + y, cy + x, col);
            set_pixel_opaque(c, cx - y, cy + x, col);
            set_pixel_opaque(c, cx - x, cy + y, col);
            set_pixel_opaque(c, cx - x, cy - y, col);
            set_pixel_opaque(c, cx - y, cy - x, col);
            set_pixel_opaque(c, cx + y, cy - x, col);
            set_pixel_opaque(c, cx + x, cy - y, col);
            if (err <= 0)
            {
                y += 1;
                err += 2 * y + 1;
            }
            if (err > 0)
            {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }
}

void draw_char(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale,
    int       bold)
{
    if (c == NULL)
    {
        return;
    }
    if (ch < 32 || ch > 126)
    {
        return;
    }

    int idx = (unsigned char)ch;

    // Bounding box for 8x16 character at scale
    int min_cx = x;
    int max_cx = x + (int)((bold ? 9.0 : 8.0) * scale) - 1;
    int min_cy = y;
    int max_cy = y + (int)(16.0 * scale) - 1;

    const int    N = 4;
    const double inv_N = 1.0 / N;

    // Sub-pixel loop for anti-aliasing
    for (int cy = min_cy; cy <= max_cy; cy++)
    {
        for (int cx = min_cx; cx <= max_cx; cx++)
        {
            if (cx < 0 || cx >= c->width || cy < 0 || cy >= c->height)
            {
                continue;
            }

            int coverage = 0;
            for (int sy = 0; sy < N; sy++)
            {
                double local_y = (cy - y + (sy + 0.5) * inv_N) / scale;
                int    row = (int)local_y;
                if (row < 0 || row >= 16)
                {
                    continue;
                }

                for (int sx = 0; sx < N; sx++)
                {
                    double local_x = (cx - x + (sx + 0.5) * inv_N) / scale;
                    int    col_idx = (int)local_x;

                    int active = 0;
                    if (col_idx >= 0 && col_idx < 8)
                    {
                        unsigned char bits = font8x16[idx][row];
                        if (bits & (1 << (7 - col_idx)))
                        {
                            active = 1;
                        }
                    }
                    if (!active && bold && col_idx > 0 && col_idx <= 8)
                    {
                        unsigned char bits = font8x16[idx][row];
                        if (bits & (1 << (7 - (col_idx - 1))))
                        {
                            active = 1;
                        }
                    }

                    if (active)
                    {
                        coverage++;
                    }
                }
            }

            if (coverage > 0)
            {
                double alpha = (double)coverage / (N * N);
                int    pixel_idx = (cy * c->width + cx) * 3;
                c->data[pixel_idx] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx] + alpha * col.r);
                c->data[pixel_idx + 1] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx + 1] + alpha * col.g);
                c->data[pixel_idx + 2] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx + 2] + alpha * col.b);
            }
        }
    }
}

void draw_string(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    int         align,
    double      scale,
    int         bold)
{
    if (c == NULL)
    {
        if (svg_out_file != NULL)
        {
            const char *anchor = (align == 1) ? "end" : "start";
            double      font_sz = 12.0 * scale;
            double      baseline_y = y + 13.0 * scale;
            fprintf(svg_out_file,
                    "<text x=\"%d\" y=\"%.2f\" font-family=\"monospace\" font-size=\"%.1f\" "
                    "text-anchor=\"%s\" fill=\"#%02x%02x%02x\" %s>%s</text>\n",
                    x, baseline_y, font_sz, anchor, col.r, col.g, col.b,
                    bold ? "font-weight=\"bold\"" : "", str);
        }
        return;
    }

    int len = strlen(str);
    int total_width = (int)(len * 8 * scale);
    int start_x = (align == 1) ? x - total_width : x;

    for (int ii = 0; ii < len; ii++)
    {
        draw_char(c, start_x + (int)(ii * 8 * scale), y, str[ii], col, scale, bold);
    }
}

void draw_char_rotated90(
    Canvas   *c,
    int       x,
    int       y,
    char      ch,
    ColorRGB  col,
    double    scale)
{
    if (c == NULL)
    {
        return;
    }
    if (ch < 32 || ch > 126)
    {
        return;
    }

    int idx = (unsigned char)ch;

    // Bounding box in rotated coordinates
    int min_cx = x - (int)(16.0 * scale) + 1;
    int max_cx = x;
    int min_cy = y;
    int max_cy = y + (int)(8.0 * scale) - 1;

    const int N = 4;
    const double inv_N = 1.0 / N;

    for (int cy = min_cy; cy <= max_cy; cy++)
    {
        for (int cx = min_cx; cx <= max_cx; cx++)
        {
            if (cx < 0 || cx >= c->width || cy < 0 || cy >= c->height)
            {
                continue;
            }

            int coverage = 0;
            for (int sy = 0; sy < N; sy++)
            {
                for (int sx = 0; sx < N; sx++)
                {
                    double local_y = (x - (cx + (sx + 0.5) * inv_N)) / scale;
                    double local_x = (cy + (sy + 0.5) * inv_N - y) / scale;
                    int    row = (int)local_y;
                    int    col_idx = (int)local_x;

                    if (row >= 0 && row < 16 && col_idx >= 0 && col_idx < 8)
                    {
                        unsigned char bits = font8x16[idx][row];
                        if (bits & (1 << (7 - col_idx)))
                        {
                            coverage++;
                        }
                    }
                }
            }

            if (coverage > 0)
            {
                double alpha = (double)coverage / (N * N);
                int    pixel_idx = (cy * c->width + cx) * 3;
                c->data[pixel_idx] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx] + alpha * col.r);
                c->data[pixel_idx + 1] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx + 1] + alpha * col.g);
                c->data[pixel_idx + 2] =
                    (unsigned char)((1.0 - alpha) * c->data[pixel_idx + 2] + alpha * col.b);
            }
        }
    }
}

void draw_string_rotated90(
    Canvas     *c,
    int         x,
    int         y,
    const char *str,
    ColorRGB    col,
    double      scale)
{
    if (c == NULL)
    {
        if (svg_out_file != NULL)
        {
            double font_sz = 12.0 * scale;
            fprintf(svg_out_file,
                    "<text x=\"%d\" y=\"%d\" font-family=\"monospace\" font-size=\"%.1f\" "
                    "text-anchor=\"start\" fill=\"#%02x%02x%02x\" "
                    "transform=\"rotate(90, %d, %d)\">%s</text>\n",
                    x, y, font_sz, col.r, col.g, col.b, x, y, str);
        }
        return;
    }

    int len = strlen(str);

    for (int ii = 0; ii < len; ii++)
    {
        draw_char_rotated90(c, x, y + (int)(ii * 8 * scale), str[ii], col, scale);
    }
}

void draw_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count)
{
    ColorRGB bg = {250, 250, 250};
    ColorRGB border = {0, 0, 0};
    ColorRGB grid = {200, 200, 200};
    ColorRGB bar_col = {100, 100, 255};

    draw_filled_rect(c, x, y, w, h, bg);
    draw_line(c, x, y, x + w, y, border);
    draw_line(c, x + w, y, x + w, y + h, border);
    draw_line(c, x + w, y + h, x, y + h, border);
    draw_line(c, x, y + h, x, y, border);

    long max_val = 0;
    int  max_idx = 0;

    for (int i = 0; i < count; i++)
    {
        if (data[i] > 0)
        {
            max_idx = i;
        }
        if (data[i] > max_val)
        {
            max_val = data[i];
        }
    }

    if (max_val == 0)
    {
        return;
    }

    int    display_count = max_idx + 2;
    double log_max = log10((double)max_val);

    if (log_max < 1.0)
    {
        log_max = 1.0;
    }

    for (int p = 0; p <= (int)log_max + 1; p++)
    {
        double val = pow(10, p);
        if (val > max_val * 2.0)
        {
            break;
        }

        double norm_h = p / (log_max * 1.1);
        int    y_pos = y + h - 10 - (int)(norm_h * (h - 20));

        if (y_pos >= y && y_pos <= y + h - 10)
        {
            draw_line(c, x, y_pos, x + w, y_pos, grid);
            char label[32];
            snprintf(label, sizeof(label), p == 0 ? "1" : "10^%d", p);
            draw_string(c, x + 2, (int)(y_pos - 8.0 * png_font_scale), label, border, 0,
                        png_font_scale, 0);
        }
    }

    double bar_w = (double)(w - 30) / display_count;
    if (bar_w < 1.0)
    {
        bar_w = 1.0;
    }

    for (int i = 0; i < display_count; i++)
    {
        if (data[i] > 0)
        {
            double lg = log10((double)data[i]);
            double norm_h = lg / (log_max * 1.1);
            int    bar_h = (int)(norm_h * (h - 20));
            int    bar_x = x + 25 + (int)(i * bar_w);
            int    bar_y = y + h - 10 - bar_h;

            draw_filled_rect(c, bar_x, bar_y, (int)bar_w + 1, bar_h, bar_col);

            char val_str[32];
            snprintf(val_str, sizeof(val_str), "%ld", data[i]);
            draw_string(c, bar_x, (int)(bar_y - 10.0 * png_font_scale), val_str, border, 0,
                        png_font_scale, 0);

            char bin_str[32];
            snprintf(bin_str, sizeof(bin_str), "%d", i);

            /* Center the bin label horizontally with the corresponding bar */
            {
                int    bin_len = strlen(bin_str);
                double label_w = bin_len * 6.0 * png_font_scale;
                int    center_x = (int)(bar_x + bar_w / 2.0 - label_w / 2.0);
                draw_string(c, center_x, (int)(y + h - 10.0 + 2.0 * png_font_scale),
                            bin_str, border, 0, png_font_scale, 0);
            }
        }
    }

    draw_string(c, (int)(x + w / 2.0 - 60.0 * png_font_scale),
                (int)(y - 15.0 * png_font_scale), "Samples / Dist Count", border, 0,
                png_font_scale, 1);
}

void draw_cluster_histogram(
    Canvas *c,
    int     x,
    int     y,
    int     w,
    int     h,
    long   *data,
    int     count)
{
    ColorRGB bg = {250, 250, 250};
    ColorRGB border = {0, 0, 0};
    ColorRGB grid = {200, 200, 200};
    ColorRGB bar_col = {100, 200, 100};

    draw_filled_rect(c, x, y, w, h, bg);
    draw_line(c, x, y, x + w, y, border);
    draw_line(c, x + w, y, x + w, y + h, border);
    draw_line(c, x + w, y + h, x, y + h, border);
    draw_line(c, x, y + h, x, y, border);

    long max_val = 0;
    for (int i = 0; i < count; i++)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
        }
    }

    if (max_val == 0)
    {
        return;
    }

    double log_max = log10((double)max_val);
    if (log_max < 1.0)
    {
        log_max = 1.0;
    }

    for (int p = 0; p <= (int)log_max + 1; p++)
    {
        double val = pow(10, p);
        if (val > max_val * 2.0)
        {
            break;
        }

        double norm_h = p / (log_max * 1.1);
        int    y_pos = y + h - (int)(norm_h * (h - 10));

        if (y_pos >= y && y_pos <= y + h)
        {
            draw_line(c, x, y_pos, x + w, y_pos, grid);
            char label[32];
            snprintf(label, sizeof(label), p == 0 ? "1" : "10^%d", p);
            draw_string(c, x - 5, (int)(y_pos - 8.0 * png_font_scale), label, border, 1,
                        png_font_scale, 0);
        }
    }

    double bar_w = (double)w / count;
    for (int i = 0; i < count; i++)
    {
        if (data[i] > 0)
        {
            double lg = log10((double)data[i]);
            double norm_h = lg / (log_max * 1.1);
            int    bar_h = (int)(norm_h * (h - 10));
            int    bar_x = x + (int)(i * bar_w);
            int    bar_y = y + h - bar_h;
            int    draw_w = (int)bar_w;

            if (draw_w < 1)
            {
                draw_w = 1;
            }
            draw_filled_rect(c, bar_x, bar_y, draw_w, bar_h, bar_col);
        }
    }

    draw_string(c, (int)(x + w / 2.0 - 51.0 * png_font_scale),
                (int)(y - 15.0 * png_font_scale), "Samples / Cluster", border, 0,
                png_font_scale, 1);
}

void draw_dcc_matrix(
    Canvas     *c,
    int         x,
    int         y,
    int         w,
    int         h,
    const char *dcc_file,
    int         num_clusters)
{
    if (num_clusters <= 0 || !dcc_file[0])
    {
        return;
    }

    double *matrix = (double *)malloc(num_clusters * num_clusters * sizeof(double));
    for (long k = 0; k < num_clusters * num_clusters; k++)
    {
        matrix[k] = -1.0;
    }

    FILE   *f = fopen(dcc_file, "r");
    double  max_dist = 0.0;

    if (f)
    {
        char line[1024];
        while (fgets(line, sizeof(line), f))
        {
            int    i, j;
            double d;
            if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3)
            {
                if (i >= 0 && i < num_clusters && j >= 0 && j < num_clusters)
                {
                    matrix[i * num_clusters + j] = d;
                    matrix[j * num_clusters + i] = d;
                    if (d > max_dist)
                    {
                        max_dist = d;
                    }
                }
            }
        }
        fclose(f);
    }

    double cell_w = (double)w / num_clusters;
    double cell_h = (double)h / num_clusters;

    for (int i = 0; i < num_clusters; i++)
    {
        for (int j = 0; j < num_clusters; j++)
        {
            double   d = matrix[i * num_clusters + j];
            int      px = x + (int)(j * cell_w);
            int      py = y + h - (int)((i + 1) * cell_h);
            int      pw = (int)((j + 1) * cell_w) - (int)(j * cell_w);
            int      ph = (int)((i + 1) * cell_h) - (int)(i * cell_h);
            ColorRGB col;

            if (pw < 1)
            {
                pw = 1;
            }
            if (ph < 1)
            {
                ph = 1;
            }

            if (d < 0)
            {
                if (i == j)
                {
                    col = (ColorRGB){255, 255, 255};
                }
                else
                {
                    col = (ColorRGB){255, 0, 0};
                }
            }
            else
            {
                unsigned char val =
                    (unsigned char)(255.0 * (d / (max_dist > 0 ? max_dist : 1.0)));
                col = (ColorRGB){val, val, val};
            }

            draw_filled_rect(c, px, py, pw, ph, col);

            if (num_clusters < 25 && d >= 0)
            {
                char     txt[32];
                ColorRGB txt_col;

                snprintf(txt, sizeof(txt), "%.2f", d);
                txt_col = (col.r > 128) ? (ColorRGB){0, 0, 0} : (ColorRGB){255, 255, 255};
                draw_string(c, (int)(px + pw / 2.0 - 10.0 * png_font_scale),
                            (int)(py + ph / 2.0 - 3.0 * png_font_scale), txt, txt_col, 0,
                            png_font_scale, 0);
            }
        }
    }

    ColorRGB black = {0, 0, 0};
    draw_line(c, x, y, x, y + h, black);
    draw_line(c, x, y + h, x + w, y + h, black);
    draw_string(c, (int)(x - 15.0 * png_font_scale), (int)(y + h - 5.0 * png_font_scale),
                "0", black, 0, png_font_scale, 0);
    draw_string(c, x, (int)(y + h + 15.0 * png_font_scale), "0", black, 0, png_font_scale, 0);

    char n_str[32];
    snprintf(n_str, sizeof(n_str), "%d", num_clusters);
    draw_string(c, (int)(x - 25.0 * png_font_scale), (int)(y + 5.0 * png_font_scale),
                n_str, black, 0, png_font_scale, 0);
    draw_string(c, (int)(x + w - 10.0 * png_font_scale),
                (int)(y + h + 15.0 * png_font_scale), n_str, black, 0, png_font_scale, 0);

    int cb_x = x + w + 10;
    int cb_w = 10;

    for (int j = 0; j < h; j++)
    {
        unsigned char val = (unsigned char)(255.0 * (1.0 - (double)j / h));
        draw_filled_rect(c, cb_x, y + j, cb_w, 1, (ColorRGB){val, val, val});
    }

    draw_string(c, (int)(cb_x + 25.0 * png_font_scale), y + h, "0", black, 0, png_font_scale, 0);

    char max_s[32];
    snprintf(max_s, sizeof(max_s), "%.2f", max_dist);
    draw_string_rotated90(c, (int)(cb_x + 25.0 * png_font_scale),
                          (int)(y + 10.0 * png_font_scale), max_s, black, png_font_scale);
    free(matrix);
}

int save_png(
    Canvas     *c,
    const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        return 1;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        return 1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 1;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, c->width, c->height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep row_pointers[c->height];
    for (int y = 0; y < c->height; y++)
    {
        row_pointers[y] = &c->data[y * c->width * 3];
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return 0;
}
#endif // USE_PNG

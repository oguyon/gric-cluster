#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_PNG
#include <png.h>
#endif
#include <sys/stat.h>
#include "cluster_defs.h"

#ifdef USE_PNG
void write_png_frame(const char *filename, double *data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open PNG output file");
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return; }

    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return; }

    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep row_pointers[height];
    unsigned char *row_data = (unsigned char*)malloc(width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double val = data[y * width + x];
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            row_data[x] = (unsigned char)val;
        }
        row_pointers[y] = row_data;
        png_write_row(png, row_pointers[y]);
    }

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    free(row_data);
    fclose(fp);
}
#endif

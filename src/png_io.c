#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_PNG
#include <png.h>
#endif
#include <sys/stat.h>
#include "cluster_defs.h"
#include "png_io.h"

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

double* read_png_frame(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open PNG input file");
        return NULL;
    }

    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        fclose(fp);
        return NULL;
    }
    
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "File %s is not recognized as a PNG file\n", filename);
        fclose(fp);
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return NULL; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);

    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    
    // Convert RGB to Gray
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA) {
        png_set_rgb_to_gray_fixed(png, 1, -1, -1);
    }

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    // Ensure rowbytes matches width (8-bit gray)
    if (rowbytes != *width) {
        // This might happen if there are extra channels (Alpha) that rgb_to_gray didn't strip?
        // png_set_rgb_to_gray should handle it, but let's be safe.
        // Actually, if we have alpha, we should strip it or compose it. 
        // png_set_strip_alpha(png);
    }

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++)
        row_pointers[y] = (png_byte*)malloc(rowbytes);

    png_read_image(png, row_pointers);

    double *data = (double*)malloc((*width) * (*height) * sizeof(double));
    if (!data) {
        for (int y = 0; y < *height; y++) free(row_pointers[y]);
        free(row_pointers);
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
             // Assuming 8-bit gray now
             data[y * (*width) + x] = (double)row_pointers[y][x];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return data;
}
#else
// stubs if no png support
void write_png_frame(const char *filename, double *data, int width, int height) {
    fprintf(stderr, "PNG support not compiled in.\n");
}
double* read_png_frame(const char *filename, int *width, int *height) {
    fprintf(stderr, "PNG support not compiled in.\n");
    return NULL;
}
#endif
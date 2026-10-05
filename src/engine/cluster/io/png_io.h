#ifndef PNG_IO_H
#define PNG_IO_H

/**
 * write_png_frame() - Save 2D image matrix to grayscale PNG file.
 * @filename: Output PNG file path.
 * @data:     Image pixel data array.
 * @width:    Image width in pixels.
 * @height:   Image height in pixels.
 */
void write_png_frame(
    const char *filename,
    double     *data,
    int         width,
    int         height);

/**
 * read_png_frame() - Read grayscale pixel data from PNG image.
 * @filename:   Input PNG file path.
 * @width:      Output pointer for image width.
 * @height:     Output pointer for image height.
 *
 * Return: Dynamically allocated double array of pixel values, or NULL on error.
 */
double *read_png_frame(
    const char *filename,
    int        *width,
    int        *height);

#endif // PNG_IO_H

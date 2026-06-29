#ifndef PNG_IO_H
#define PNG_IO_H

void write_png_frame(const char *filename, double *data, int width, int height);
double* read_png_frame(const char *filename, int *width, int *height);

#endif // PNG_IO_H

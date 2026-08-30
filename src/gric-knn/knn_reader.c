/**
 * @file knn_reader.c
 * @brief Out-of-core random-access frame reader for FITS and ASCII datasets.
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "knn_reader.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/**
 * check_is_fits_path() - Check if path has FITS extension.
 * @path: Input path.
 *
 * Return: 1 if FITS, 0 otherwise.
 */
static int check_is_fits_path(
    const char *path)
{
    if (path == NULL)
    {
        return 0;
    }

    size_t len = strlen(path);
    if (len >= 5 && strcasecmp(path + len - 5, ".fits") == 0)
    {
        return 1;
    }
    if (len >= 8 && strcasecmp(path + len - 8, ".fits.gz") == 0)
    {
        return 1;
    }

    return 0;
}

/**
 * knn_reader_inspect() - Discover dataset sample count and coordinate dimensions.
 * @path:         Path to dataset file.
 * @total_frames: Output pointer for sample/frame count.
 * @frame_width:  Output pointer for frame width / coordinate count.
 * @frame_height: Output pointer for frame height.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_inspect(
    const char *path,
    long       *total_frames,
    long       *frame_width,
    long       *frame_height)
{
    if (path == NULL || total_frames == NULL || frame_width == NULL || frame_height == NULL)
    {
        return -1;
    }

    *total_frames = 0;
    *frame_width = 0;
    *frame_height = 1;

    FILE *fp_bin = fopen(path, "rb");
    if (fp_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(fp_bin, &hdr, &comment) == 0)
        {
            *total_frames = (long)hdr.dims[0];
            *frame_width = (hdr.ndim >= 2) ? (long)hdr.dims[1] : 1;
            *frame_height = (hdr.ndim >= 3) ? (long)hdr.dims[2] : 1;
            if (comment != NULL)
            {
                free(comment);
            }
            fclose(fp_bin);
            return 0;
        }
        if (comment != NULL)
        {
            free(comment);
        }
        fclose(fp_bin);
    }

    if (check_is_fits_path(path))
    {
#ifdef USE_CFITSIO
        int status = 0;
        fitsfile *fptr = NULL;
        fits_open_file(&fptr, path, READONLY, &status);
        if (status == 0 && fptr != NULL)
        {
            int naxis = 0;
            long naxes[3] = {0, 0, 0};
            fits_get_img_dim(fptr, &naxis, &status);
            fits_get_img_size(fptr, 3, naxes, &status);
            if (status == 0)
            {
                if (naxis >= 3)
                {
                    *frame_width = naxes[0];
                    *frame_height = naxes[1];
                    *total_frames = naxes[2];
                }
                else if (naxis == 2)
                {
                    *frame_width = naxes[0];
                    *frame_height = naxes[1];
                    *total_frames = 1;
                }
                fits_close_file(fptr, &status);
                return 0;
            }
            fits_close_file(fptr, &status);
        }
        return -1;
#else
        return -1;
#endif
    }

    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        return -1;
    }

    char line_buf[65536];
    long count = 0;
    long elements_detected = 0;

    while (fgets(line_buf, sizeof(line_buf), f) != NULL)
    {
        if (line_buf[0] == '#' || line_buf[0] == '\n' || line_buf[0] == '\0')
        {
            continue;
        }

        if (elements_detected == 0)
        {
            char *ptr = line_buf;
            while (*ptr != '\0')
            {
                while (isspace((unsigned char)*ptr))
                {
                    ptr++;
                }
                if (*ptr == '\0')
                {
                    break;
                }
                elements_detected++;
                while (*ptr != '\0' && !isspace((unsigned char)*ptr))
                {
                    ptr++;
                }
            } // while parsing first line
        }

        count++;
    } // while reading ASCII

    fclose(f);

    if (count == 0 || elements_detected == 0)
    {
        return -1;
    }

    *total_frames = count;
    *frame_width = elements_detected;
    *frame_height = 1;

    return 0;
}

/**
 * build_ascii_index() - Build 64-bit line offset seek table for ASCII files.
 * @reader: Pointer to KnnFrameReader.
 *
 * Return: 0 on success, -1 on error.
 */
static int build_ascii_index(
    KnnFrameReader *reader)
{
    FILE *f = fopen(reader->input_path, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Error: Could not open ASCII dataset '%s'\n", reader->input_path);
        return -1;
    }

    reader->line_offsets = (uint64_t *)malloc((size_t)reader->total_frames * sizeof(uint64_t));
    if (reader->line_offsets == NULL)
    {
        fclose(f);
        return -1;
    }

    char line_buf[65536];
    long frame_idx = 0;
    off_t offset = ftello(f);

    while (fgets(line_buf, sizeof(line_buf), f) != NULL && frame_idx < reader->total_frames)
    {
        if (line_buf[0] != '#' && line_buf[0] != '\n' && line_buf[0] != '\0')
        {
            reader->line_offsets[frame_idx++] = (uint64_t)offset;
        }
        offset = ftello(f);
    } // while indexing

    fclose(f);

    if (frame_idx < reader->total_frames)
    {
        fprintf(stderr, "Warning: Expected %ld ASCII frames, indexed %ld\n",
                reader->total_frames, frame_idx);
    }

    reader->ascii_file = fopen(reader->input_path, "r");
    if (reader->ascii_file == NULL)
    {
        return -1;
    }

    return 0;
}

/**
 * knn_reader_open() - Open dataset and prepare out-of-core access structures.
 * @reader:       Pointer to KnnFrameReader.
 * @input_path:   Path to the input file.
 * @total_frames: Total number of frames in dataset.
 * @frame_width:  Frame width in pixels/coordinates.
 * @frame_height: Frame height in pixels.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_open(
    KnnFrameReader *reader,
    const char     *input_path,
    long            total_frames,
    long            frame_width,
    long            frame_height)
{
    if (reader == NULL || input_path == NULL)
    {
        return -1;
    }

    memset(reader, 0, sizeof(KnnFrameReader));
    reader->input_path = strdup(input_path);
    reader->total_frames = total_frames;
    reader->frame_width = frame_width;
    reader->frame_height = frame_height;
    reader->frame_elements = frame_width * frame_height;

    FILE *fp_bin = fopen(input_path, "rb");
    if (fp_bin != NULL)
    {
        gric_bin_header_t hdr;
        char *comment = NULL;
        if (gric_bin_read_header(fp_bin, &hdr, &comment) == 0)
        {
            reader->is_bin = 1;
            reader->bin_data_type = hdr.data_type;
            reader->bin_header_bytes = hdr.header_bytes;
            reader->bin_file = fp_bin;
            if (comment != NULL)
            {
                free(comment);
            }
            return 0;
        }
        if (comment != NULL)
        {
            free(comment);
        }
        fclose(fp_bin);
    }

    reader->is_fits = check_is_fits_path(input_path);

    if (reader->is_fits)
    {
#ifdef USE_CFITSIO
        int status = 0;
        fits_open_file(&reader->fits_ptr, reader->input_path, READONLY, &status);
        if (status != 0 || reader->fits_ptr == NULL)
        {
            fprintf(stderr, "Error: Could not open FITS dataset '%s' (CFITSIO error %d)\n",
                    reader->input_path, status);
            return -1;
        }
        return 0;
#else
        fprintf(stderr, "Error: FITS support not compiled in\n");
        return -1;
#endif
    }
    else
    {
        return build_ascii_index(reader);
    }
}

/**
 * knn_reader_open_memory() - Prepare in-memory dataset access.
 * @reader:         Pointer to KnnFrameReader.
 * @memory_data:    Contiguous buffer of frame vectors [total_frames * frame_elements].
 * @total_frames:   Total number of frames in dataset.
 * @frame_elements: Number of elements per frame.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_open_memory(
    KnnFrameReader *reader,
    const double   *memory_data,
    long            total_frames,
    long            frame_elements)
{
    if (reader == NULL || memory_data == NULL || total_frames <= 0 || frame_elements <= 0)
    {
        return -1;
    }

    memset(reader, 0, sizeof(KnnFrameReader));
    reader->memory_data = memory_data;
    reader->total_frames = total_frames;
    reader->frame_width = frame_elements;
    reader->frame_height = 1;
    reader->frame_elements = frame_elements;

    return 0;
}

/**
 * knn_reader_clone_thread() - Create a thread-local reader handle.
 * @src: Master KnnFrameReader structure.
 * @dst: Destination thread-local KnnFrameReader structure.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_clone_thread(
    const KnnFrameReader *src,
    KnnFrameReader       *dst)
{
    if (src == NULL || dst == NULL)
    {
        return -1;
    }

    memcpy(dst, src, sizeof(KnnFrameReader));

    if (src->memory_data != NULL)
    {
        dst->memory_data = src->memory_data;
        dst->input_path = NULL;
        dst->ascii_file = NULL;
        dst->bin_file = NULL;
        return 0;
    }

    dst->input_path = (src->input_path != NULL) ? strdup(src->input_path) : NULL;
    dst->ascii_file = NULL;
    dst->bin_file = NULL;

    if (src->is_bin)
    {
        dst->bin_file = fopen(dst->input_path, "rb");
        if (dst->bin_file == NULL)
        {
            return -1;
        }
        return 0;
    }

    if (src->is_fits)
    {
#ifdef USE_CFITSIO
        dst->fits_ptr = NULL;
        int status = 0;
#ifdef _OPENMP
#pragma omp critical(fits_io_lock)
#endif
        {
            fits_open_file(&dst->fits_ptr, dst->input_path, READONLY, &status);
        }
        if (status != 0 || dst->fits_ptr == NULL)
        {
            return -1;
        }
#endif
    }
    else
    {
        dst->ascii_file = fopen(dst->input_path, "r");
        if (dst->ascii_file == NULL)
        {
            return -1;
        }
    }

    return 0;
}

/**
 * knn_reader_read_frame() - Read single frame slice into buffer.
 * @reader:   Pointer to KnnFrameReader.
 * @frame_id: Zero-indexed frame ID to retrieve.
 * @out_data: Output array (size frame_elements).
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_read_frame(
    KnnFrameReader *reader,
    long            frame_id,
    double         *out_data)
{
    if (reader == NULL || out_data == NULL || frame_id < 0 || frame_id >= reader->total_frames)
    {
        return -1;
    }

    if (reader->memory_data != NULL)
    {
        memcpy(out_data, &reader->memory_data[frame_id * reader->frame_elements],
               (size_t)reader->frame_elements * sizeof(double));
        return 0;
    }

    if (reader->is_bin)
    {
        if (reader->bin_file == NULL)
        {
            return -1;
        }
        size_t elem_size = gric_bin_data_type_size((gric_bin_data_type_t)reader->bin_data_type);
        if (elem_size == 0)
        {
            return -1;
        }
        off_t offset = (off_t)reader->bin_header_bytes +
                       (off_t)frame_id * (off_t)reader->frame_elements * (off_t)elem_size;
        if (fseeko(reader->bin_file, offset, SEEK_SET) != 0)
        {
            return -1;
        }
        if (reader->bin_data_type == GRIC_BIN_DTYPE_FLOAT64)
        {
            if (fread(out_data, sizeof(double), (size_t)reader->frame_elements,
                      reader->bin_file) != (size_t)reader->frame_elements)
            {
                return -1;
            }
        }
        else if (reader->bin_data_type == GRIC_BIN_DTYPE_FLOAT32)
        {
            float *fbuf = (float *)malloc((size_t)reader->frame_elements * sizeof(float));
            if (fbuf == NULL)
            {
                return -1;
            }
            if (fread(fbuf, sizeof(float), (size_t)reader->frame_elements,
                      reader->bin_file) != (size_t)reader->frame_elements)
            {
                free(fbuf);
                return -1;
            }
            for (long k = 0; k < reader->frame_elements; k++)
            {
                out_data[k] = (double)fbuf[k];
            }
            free(fbuf);
        }
        else if (reader->bin_data_type == GRIC_BIN_DTYPE_UINT32)
        {
            uint32_t *ubuf = (uint32_t *)malloc((size_t)reader->frame_elements * sizeof(uint32_t));
            if (ubuf == NULL)
            {
                return -1;
            }
            if (fread(ubuf, sizeof(uint32_t), (size_t)reader->frame_elements,
                      reader->bin_file) != (size_t)reader->frame_elements)
            {
                free(ubuf);
                return -1;
            }
            for (long k = 0; k < reader->frame_elements; k++)
            {
                out_data[k] = (double)ubuf[k];
            }
            free(ubuf);
        }
        return 0;
    }

    if (reader->is_fits)
    {
#ifdef USE_CFITSIO
        int status = 0;
        long fpixel[3] = {1, 1, frame_id + 1};
        fits_read_pix(reader->fits_ptr, TDOUBLE, fpixel, reader->frame_elements, NULL,
                      out_data, NULL, &status);
        return (status == 0) ? 0 : -1;
#else
        return -1;
#endif
    }
    else
    {
        if (reader->ascii_file == NULL || reader->line_offsets == NULL)
        {
            return -1;
        }

        off_t offset = (off_t)reader->line_offsets[frame_id];
        if (fseeko(reader->ascii_file, offset, SEEK_SET) != 0)
        {
            return -1;
        }

        for (long k = 0; k < reader->frame_elements; k++)
        {
            if (fscanf(reader->ascii_file, "%lf", &out_data[k]) != 1)
            {
                out_data[k] = 0.0;
            }
        }
        return 0;
    }
}

/**
 * knn_reader_close_thread() - Close thread-local file handles.
 * @reader: Pointer to thread-local KnnFrameReader.
 */
void knn_reader_close_thread(
    KnnFrameReader *reader)
{
    if (reader == NULL)
    {
        return;
    }

    if (reader->bin_file != NULL)
    {
        fclose(reader->bin_file);
        reader->bin_file = NULL;
    }

    if (reader->is_fits)
    {
#ifdef USE_CFITSIO
        if (reader->fits_ptr != NULL)
        {
            int status = 0;
#ifdef _OPENMP
#pragma omp critical(fits_io_lock)
#endif
            {
                fits_close_file(reader->fits_ptr, &status);
            }
            reader->fits_ptr = NULL;
        }
#endif
    }
    else
    {
        if (reader->ascii_file != NULL)
        {
            fclose(reader->ascii_file);
            reader->ascii_file = NULL;
        }
    }

    if (reader->input_path != NULL)
    {
        free(reader->input_path);
        reader->input_path = NULL;
    }
}

/**
 * knn_reader_close() - Close master reader and free shared index structures.
 * @reader: Pointer to KnnFrameReader.
 */
void knn_reader_close(
    KnnFrameReader *reader)
{
    if (reader == NULL)
    {
        return;
    }

    knn_reader_close_thread(reader);

    if (reader->line_offsets != NULL)
    {
        free(reader->line_offsets);
        reader->line_offsets = NULL;
    }
}

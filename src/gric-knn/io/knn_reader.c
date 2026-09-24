/**
 * @file knn_reader.c
 * @brief Out-of-core random-access frame reader for FITS and ASCII datasets.
 *
 * Implements low-overhead random-access frame retrieval from out-of-core storage.
 * Functions in this file inspect dataset file dimensions, build 64-bit line offset tables
 * for ASCII files via zero-copy mmap, memory-map binary files, and read individual coordinate
 * frames across OpenMP threads with thread-safe file descriptors and zero-copy memory access.
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "knn_reader.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * parse_ascii_row_float() - Parse an in-memory row of floating-point values.
 * @p:         Pointer to start of the row data.
 * @end:       Pointer to end of the mapped memory buffer.
 * @out:       Destination array for float values.
 * @nelements: Number of elements to parse.
 *
 * Return: 0 on success, or -1 on parse failure.
 */
static inline int parse_ascii_row_float(
    const char *p,
    const char *end,
    float      *out,
    long        nelements)
{
    for (long ii = 0; ii < nelements; ii++)
    {
        char *next = NULL;
        if (end - p < 64)
        {
            char tmp[64];
            size_t rem = (size_t)(end - p);
            memcpy(tmp, p, rem);
            tmp[rem] = '\0';
            out[ii] = strtof(tmp, &next);
            if (next == tmp)
            {
                return -1;
            }
            p += (next - tmp);
        }
        else
        {
            out[ii] = strtof(p, &next);
            if (next == p)
            {
                return -1;
            }
            p = next;
        }
    }
    return 0;
}

/**
 * parse_ascii_row_double() - Parse an in-memory row of double-precision values.
 * @p:         Pointer to start of the row data.
 * @end:       Pointer to end of the mapped memory buffer.
 * @out:       Destination array for double values.
 * @nelements: Number of elements to parse.
 *
 * Return: 0 on success, or -1 on parse failure.
 */
static inline int parse_ascii_row_double(
    const char *p,
    const char *end,
    double     *out,
    long        nelements)
{
    for (long ii = 0; ii < nelements; ii++)
    {
        char *next = NULL;
        if (end - p < 64)
        {
            char tmp[64];
            size_t rem = (size_t)(end - p);
            memcpy(tmp, p, rem);
            tmp[rem] = '\0';
            out[ii] = strtod(tmp, &next);
            if (next == tmp)
            {
                return -1;
            }
            p += (next - tmp);
        }
        else
        {
            out[ii] = strtod(p, &next);
            if (next == p)
            {
                return -1;
            }
            p = next;
        }
    }
    return 0;
}

/**
 * probe_binary_dataset_path() - Probe if a corresponding .bin dataset exists.
 * @orig_path: Original input file path.
 * @bin_path:  Output buffer for probed .bin path.
 * @max_len:   Size of @bin_path buffer.
 *
 * Return: 1 if matching .bin exists and has valid header, 0 otherwise.
 */
static int probe_binary_dataset_path(
    const char *orig_path,
    char       *bin_path,
    size_t      max_len)
{
    if (orig_path == NULL || bin_path == NULL)
    {
        return 0;
    }

    size_t len = strlen(orig_path);
    if (len >= 4 && strcasecmp(orig_path + len - 4, ".bin") == 0)
    {
        return 0;
    }

    /* If .txt, check replacing .txt with .bin */
    if (len >= 4 && strcasecmp(orig_path + len - 4, ".txt") == 0)
    {
        if (len + 1 >= max_len)
        {
            return 0;
        }
        snprintf(bin_path, max_len, "%.*s.bin", (int)(len - 4), orig_path);
        struct stat st;
        if (stat(bin_path, &st) == 0 && st.st_size > 64)
        {
            return 1;
        }
    }

    /* If extensionless, check orig_path.bin */
    if (strrchr(orig_path, '.') == NULL)
    {
        if (len + 5 >= max_len)
        {
            return 0;
        }
        snprintf(bin_path, max_len, "%s.bin", orig_path);
        struct stat st;
        if (stat(bin_path, &st) == 0 && st.st_size > 64)
        {
            return 1;
        }
    }

    return 0;
}

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

    char auto_bin[1024];
    const char *effective_path = path;
    if (probe_binary_dataset_path(path, auto_bin, sizeof(auto_bin)))
    {
        effective_path = auto_bin;
    }

    FILE *fp_bin = fopen(effective_path, "rb");
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

    if (check_is_fits_path(effective_path))
    {
#ifdef USE_CFITSIO
        int status = 0;
        fitsfile *fptr = NULL;
        fits_open_file(&fptr, effective_path, READONLY, &status);
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

    FILE *f = fopen(effective_path, "r");
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
 * build_ascii_index() - Build 64-bit line offset table via zero-copy mmap.
 * @reader: Pointer to KnnFrameReader.
 *
 * Return: 0 on success, -1 on error.
 */
static int build_ascii_index(
    KnnFrameReader *reader)
{
    reader->ascii_fd = open(reader->input_path, O_RDONLY);
    if (reader->ascii_fd < 0)
    {
        fprintf(stderr, "Error: Could not open ASCII dataset '%s'\n", reader->input_path);
        return -1;
    }

    struct stat st;
    if (fstat(reader->ascii_fd, &st) != 0 || st.st_size <= 0)
    {
        close(reader->ascii_fd);
        reader->ascii_fd = -1;
        return -1;
    }

    reader->ascii_mmap_size = (size_t)st.st_size;
    reader->ascii_mmap_addr = mmap(NULL, reader->ascii_mmap_size, PROT_READ, MAP_PRIVATE,
                                   reader->ascii_fd, 0);
    if (reader->ascii_mmap_addr == MAP_FAILED)
    {
        reader->ascii_mmap_addr = NULL;
        reader->ascii_mmap_size = 0;
        close(reader->ascii_fd);
        reader->ascii_fd = -1;
        return -1;
    }

    posix_madvise(reader->ascii_mmap_addr, reader->ascii_mmap_size,
                  POSIX_MADV_SEQUENTIAL | POSIX_MADV_WILLNEED);

    size_t line_count = 0;
    const char *buf = (const char *)reader->ascii_mmap_addr;
    const char *buf_end = buf + reader->ascii_mmap_size;
    const char *scan = buf;

    while (scan < buf_end)
    {
        const char *nl = (const char *)memchr(scan, '\n', (size_t)(buf_end - scan));
        if (nl == NULL)
        {
            line_count++;
            break;
        }
        line_count++;
        scan = nl + 1;
    }

    if (line_count == 0)
    {
        line_count = 1;
    }

    reader->line_offsets = (uint64_t *)malloc(line_count * sizeof(uint64_t));
    if (reader->line_offsets == NULL)
    {
        munmap(reader->ascii_mmap_addr, reader->ascii_mmap_size);
        reader->ascii_mmap_addr = NULL;
        reader->ascii_mmap_size = 0;
        close(reader->ascii_fd);
        reader->ascii_fd = -1;
        return -1;
    }

    {
        /* Scan mapped buffer to index lines */
        size_t pos = 0;
        long frame_idx = 0;

        while (pos < reader->ascii_mmap_size && frame_idx < reader->total_frames)
        {
            size_t line_end = pos;
            while (line_end < reader->ascii_mmap_size &&
                   buf[line_end] != '\n' && buf[line_end] != '\r')
            {
                line_end++;
            }

            size_t p = pos;
            while (p < line_end && isspace((unsigned char)buf[p]))
            {
                p++;
            }

            if (p < line_end && buf[p] != '#')
            {
                reader->line_offsets[frame_idx++] = (uint64_t)p;
            }

            pos = line_end;
            if (pos < reader->ascii_mmap_size && buf[pos] == '\r')
            {
                pos++;
            }
            if (pos < reader->ascii_mmap_size && buf[pos] == '\n')
            {
                pos++;
            }
        } // while pos < ascii_mmap_size

        if (frame_idx < reader->total_frames)
        {
            fprintf(stderr, "Warning: Expected %ld ASCII frames, indexed %ld\n",
                    reader->total_frames, frame_idx);
        }
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
 * @use_double:   1 for double precision, 0 for single-precision float.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_open(
    KnnFrameReader *reader,
    const char     *input_path,
    long            total_frames,
    long            frame_width,
    long            frame_height,
    int             use_double)
{
    if (reader == NULL || input_path == NULL)
    {
        return -1;
    }

    memset(reader, 0, sizeof(KnnFrameReader));
    reader->ascii_fd = -1;

    char auto_bin[1024];
    const char *effective_path = input_path;
    if (probe_binary_dataset_path(input_path, auto_bin, sizeof(auto_bin)))
    {
        printf("[READER] Adopting binary dataset by default: %s\n", auto_bin);
        effective_path = auto_bin;
    }

    reader->input_path = strdup(effective_path);
    reader->total_frames = total_frames;
    reader->frame_width = frame_width;
    reader->frame_height = frame_height;
    reader->frame_elements = frame_width * frame_height;
    reader->use_double = use_double;

    FILE *fp_bin = fopen(effective_path, "rb");
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

            int type_matches = (use_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT64) ||
                               (!use_double && hdr.data_type == GRIC_BIN_DTYPE_FLOAT32);
            if (type_matches)
            {
                int fd = fileno(fp_bin);
                struct stat st;
                if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > (off_t)hdr.header_bytes)
                {
                    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                    if (addr != MAP_FAILED)
                    {
                        posix_madvise(addr, (size_t)st.st_size, POSIX_MADV_WILLNEED);
                        reader->bin_mmap_addr = addr;
                        reader->bin_mmap_size = (size_t)st.st_size;
                        reader->memory_data = (const char *)addr + hdr.header_bytes;
                    }
                }
            }
            return 0;
        }
        if (comment != NULL)
        {
            free(comment);
        }
        fclose(fp_bin);
    }

    reader->is_fits = check_is_fits_path(effective_path);

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

    return build_ascii_index(reader);
}

/**
 * knn_reader_open_memory() - Prepare in-memory dataset access.
 * @reader:         Pointer to KnnFrameReader.
 * @memory_data:    Contiguous buffer of frame vectors [total_frames * frame_elements].
 * @total_frames:   Total number of frames in dataset.
 * @frame_elements: Number of elements per frame.
 * @use_double:     1 for double precision, 0 for float.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_reader_open_memory(
    KnnFrameReader *reader,
    const void     *memory_data,
    long            total_frames,
    long            frame_elements,
    int             use_double)
{
    if (reader == NULL || memory_data == NULL || total_frames <= 0 || frame_elements <= 0)
    {
        return -1;
    }

    memset(reader, 0, sizeof(KnnFrameReader));
    reader->ascii_fd = -1;
    reader->memory_data = memory_data;
    reader->total_frames = total_frames;
    reader->frame_width = frame_elements;
    reader->frame_height = 1;
    reader->frame_elements = frame_elements;
    reader->use_double = use_double;

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
    dst->bin_mmap_addr = NULL;
    dst->bin_mmap_size = 0;
    dst->ascii_fd = -1;

    if (src->memory_data != NULL)
    {
        dst->memory_data = src->memory_data;
        dst->input_path = NULL;
        dst->ascii_file = NULL;
        dst->bin_file = NULL;
        return 0;
    }

    if (src->ascii_mmap_addr != NULL)
    {
        dst->ascii_mmap_addr = src->ascii_mmap_addr;
        dst->ascii_mmap_size = src->ascii_mmap_size;
        dst->line_offsets = src->line_offsets;
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
    void           *out_data)
{
    if (reader == NULL || out_data == NULL || frame_id < 0 || frame_id >= reader->total_frames)
    {
        return -1;
    }

    size_t elem_size = reader->use_double ? sizeof(double) : sizeof(float);

    if (reader->memory_data != NULL)
    {
        const char *src_ptr = (const char *)reader->memory_data;
        size_t offset = (size_t)frame_id * (size_t)reader->frame_elements * elem_size;
        memcpy(out_data, src_ptr + offset, (size_t)reader->frame_elements * elem_size);
        return 0;
    }

    if (reader->ascii_mmap_addr != NULL && reader->line_offsets != NULL)
    {
        const char *p = (const char *)reader->ascii_mmap_addr + reader->line_offsets[frame_id];
        const char *end = (const char *)reader->ascii_mmap_addr + reader->ascii_mmap_size;
        if (reader->use_double)
        {
            return parse_ascii_row_double(p, end, (double *)out_data, reader->frame_elements);
        }
        return parse_ascii_row_float(p, end, (float *)out_data, reader->frame_elements);
    }

    if (reader->is_bin)
    {
        if (reader->bin_file == NULL)
        {
            return -1;
        }
        size_t bin_elem_size = gric_bin_data_type_size(
            (gric_bin_data_type_t)reader->bin_data_type);
        if (bin_elem_size == 0)
        {
            return -1;
        }
        off_t offset = (off_t)reader->bin_header_bytes +
                       (off_t)frame_id * (off_t)reader->frame_elements * (off_t)bin_elem_size;
        if (fseeko(reader->bin_file, offset, SEEK_SET) != 0)
        {
            return -1;
        }
        if (reader->bin_data_type == GRIC_BIN_DTYPE_FLOAT64)
        {
            if (reader->use_double)
            {
                if (fread(out_data, sizeof(double), (size_t)reader->frame_elements,
                          reader->bin_file) != (size_t)reader->frame_elements)
                {
                    return -1;
                }
            }
            else
            {
                double *dbuf = (double *)malloc((size_t)reader->frame_elements * sizeof(double));
                if (dbuf == NULL)
                {
                    return -1;
                }
                if (fread(dbuf, sizeof(double), (size_t)reader->frame_elements,
                          reader->bin_file) != (size_t)reader->frame_elements)
                {
                    free(dbuf);
                    return -1;
                }
                float *fptr = (float *)out_data;
                for (long k = 0; k < reader->frame_elements; k++)
                {
                    fptr[k] = (float)dbuf[k];
                }
                free(dbuf);
            }
        }
        else if (reader->bin_data_type == GRIC_BIN_DTYPE_FLOAT32)
        {
            if (!reader->use_double)
            {
                if (fread(out_data, sizeof(float), (size_t)reader->frame_elements,
                          reader->bin_file) != (size_t)reader->frame_elements)
                {
                    return -1;
                }
            }
            else
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
                double *dptr = (double *)out_data;
                for (long k = 0; k < reader->frame_elements; k++)
                {
                    dptr[k] = (double)fbuf[k];
                }
                free(fbuf);
            }
        }
        else if (reader->bin_data_type == GRIC_BIN_DTYPE_UINT32)
        {
            size_t ubytes = (size_t)reader->frame_elements * sizeof(uint32_t);
            uint32_t *ubuf = (uint32_t *)malloc(ubytes);
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
            if (reader->use_double)
            {
                double *dptr = (double *)out_data;
                for (long k = 0; k < reader->frame_elements; k++)
                {
                    dptr[k] = (double)ubuf[k];
                }
            }
            else
            {
                float *fptr = (float *)out_data;
                for (long k = 0; k < reader->frame_elements; k++)
                {
                    fptr[k] = (float)ubuf[k];
                }
            }
            free(ubuf);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (reader->is_fits)
    {
#ifdef USE_CFITSIO
        int status = 0;
        long fpixel[3] = {1, 1, frame_id + 1};
        int datatype = reader->use_double ? TDOUBLE : TFLOAT;
        fits_read_pix(reader->fits_ptr, datatype, fpixel, reader->frame_elements, NULL,
                      out_data, NULL, &status);
        return (status == 0) ? 0 : -1;
#else
        return -1;
#endif
    }

    if (reader->ascii_file == NULL || reader->line_offsets == NULL)
    {
        return -1;
    }

    off_t offset = (off_t)reader->line_offsets[frame_id];
    if (fseeko(reader->ascii_file, offset, SEEK_SET) != 0)
    {
        return -1;
    }

    if (reader->use_double)
    {
        double *dptr = (double *)out_data;
        for (long k = 0; k < reader->frame_elements; k++)
        {
            if (fscanf(reader->ascii_file, "%lf", &dptr[k]) != 1)
            {
                dptr[k] = 0.0;
            }
        }
    }
    else
    {
        float *fptr = (float *)out_data;
        for (long k = 0; k < reader->frame_elements; k++)
        {
            if (fscanf(reader->ascii_file, "%f", &fptr[k]) != 1)
            {
                fptr[k] = 0.0f;
            }
        }
    }
    return 0;
}

/**
 * knn_reader_close_thread() - Close thread-local file handles
 * @reader: Pointer to thread-local KnnFrameReader.
 *
 * Closes thread-local file descriptors (binary file handle, FITS file pointer with
 * critical section lock, or ASCII file handle) and frees thread-local input path string.
 * Leaves shared memory mappings intact for other threads.
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
 * knn_reader_close() - Close master reader and free shared index structures
 * @reader: Pointer to KnnFrameReader context.
 *
 * Unmaps memory-mapped binary and ASCII files, releases thread-local file handles,
 * frees ASCII line offset tables, and resets reader fields to NULL or zero.
 */
void knn_reader_close(
    KnnFrameReader *reader)
{
    if (reader == NULL)
    {
        return;
    }

    if (reader->bin_mmap_addr != NULL)
    {
        munmap(reader->bin_mmap_addr, reader->bin_mmap_size);
        reader->bin_mmap_addr = NULL;
        reader->bin_mmap_size = 0;
        reader->memory_data = NULL;
    }

    if (reader->ascii_mmap_addr != NULL && reader->ascii_mmap_addr != MAP_FAILED)
    {
        munmap(reader->ascii_mmap_addr, reader->ascii_mmap_size);
        reader->ascii_mmap_addr = NULL;
        reader->ascii_mmap_size = 0;
    }

    if (reader->ascii_fd >= 0)
    {
        close(reader->ascii_fd);
        reader->ascii_fd = -1;
    }

    knn_reader_close_thread(reader);

    if (reader->line_offsets != NULL)
    {
        free(reader->line_offsets);
        reader->line_offsets = NULL;
    }
}

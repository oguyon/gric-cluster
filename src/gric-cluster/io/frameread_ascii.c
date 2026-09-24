/**
 * @file frameread_ascii.c
 * @brief Memory-mapped ASCII dataset reader implementation.
 */

#define _POSIX_C_SOURCE 200809L
#include "frameread_internal.h"
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const double pow10_table[31] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
    1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30
};

/**
 * fast_parse_float_val() - High-speed ASCII float parser for standard decimal tokens.
 * @p:       Pointer to start of token in memory.
 * @end:     Pointer to buffer end.
 * @out_val: Output float receiving parsed value.
 *
 * Return: Pointer to next unparsed character on success, or NULL if fallback required.
 */
static inline const char *fast_parse_float_val(
    const char *p,
    const char *end,
    float      *out_val)
{
    while (p < end && (*p == ' ' || *p == '\t'))
    {
        p++;
    }
    if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
    {
        return NULL;
    }

    float sign = 1.0f;
    if (*p == '-')
    {
        sign = -1.0f;
        p++;
    }
    else if (*p == '+')
    {
        p++;
    }

    if (p >= end || (!(*p >= '0' && *p <= '9') && *p != '.'))
    {
        return NULL;
    }

    uint64_t int_part = 0;
    int digits = 0;
    while (p < end && *p >= '0' && *p <= '9')
    {
        int_part = int_part * 10 + (uint64_t)(*p - '0');
        digits++;
        p++;
    }

    double frac_part = 0.0;
    double scale = 1.0;
    if (p < end && *p == '.')
    {
        p++;
        while (p < end && *p >= '0' && *p <= '9')
        {
            frac_part = frac_part * 10.0 + (double)(*p - '0');
            scale *= 10.0;
            digits++;
            p++;
        }
    }

    if (digits == 0)
    {
        return NULL;
    }

    double val = (double)int_part + (frac_part / scale);

    if (p < end && (*p == 'e' || *p == 'E'))
    {
        p++;
        int exp_sign = 1;
        if (p < end && *p == '-')
        {
            exp_sign = -1;
            p++;
        }
        else if (p < end && *p == '+')
        {
            p++;
        }
        int exp = 0;
        int exp_digits = 0;
        while (p < end && *p >= '0' && *p <= '9')
        {
            exp = exp * 10 + (*p - '0');
            exp_digits++;
            p++;
        }
        if (exp_digits == 0)
        {
            return NULL;
        }
        if (exp <= 30)
        {
            val = (exp_sign < 0) ? (val / pow10_table[exp]) : (val * pow10_table[exp]);
        }
        else
        {
            val *= pow(10.0, exp_sign * exp);
        }
    }

    *out_val = (float)(sign * val);
    return p;
}

/**
 * fast_parse_double_val() - High-speed ASCII double parser for standard decimal tokens.
 * @p:       Pointer to start of token in memory.
 * @end:     Pointer to buffer end.
 * @out_val: Output double receiving parsed value.
 *
 * Return: Pointer to next unparsed character on success, or NULL if fallback required.
 */
static inline const char *fast_parse_double_val(
    const char *p,
    const char *end,
    double     *out_val)
{
    while (p < end && (*p == ' ' || *p == '\t'))
    {
        p++;
    }
    if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
    {
        return NULL;
    }

    double sign = 1.0;
    if (*p == '-')
    {
        sign = -1.0;
        p++;
    }
    else if (*p == '+')
    {
        p++;
    }

    if (p >= end || (!(*p >= '0' && *p <= '9') && *p != '.'))
    {
        return NULL;
    }

    uint64_t int_part = 0;
    int digits = 0;
    while (p < end && *p >= '0' && *p <= '9')
    {
        int_part = int_part * 10 + (uint64_t)(*p - '0');
        digits++;
        p++;
    }

    double frac_part = 0.0;
    double scale = 1.0;
    if (p < end && *p == '.')
    {
        p++;
        while (p < end && *p >= '0' && *p <= '9')
        {
            frac_part = frac_part * 10.0 + (double)(*p - '0');
            scale *= 10.0;
            digits++;
            p++;
        }
    }

    if (digits == 0)
    {
        return NULL;
    }

    double val = (double)int_part + (frac_part / scale);

    if (p < end && (*p == 'e' || *p == 'E'))
    {
        p++;
        int exp_sign = 1;
        if (p < end && *p == '-')
        {
            exp_sign = -1;
            p++;
        }
        else if (p < end && *p == '+')
        {
            p++;
        }
        int exp = 0;
        int exp_digits = 0;
        while (p < end && *p >= '0' && *p <= '9')
        {
            exp = exp * 10 + (*p - '0');
            exp_digits++;
            p++;
        }
        if (exp_digits == 0)
        {
            return NULL;
        }
        if (exp <= 30)
        {
            val = (exp_sign < 0) ? (val / pow10_table[exp]) : (val * pow10_table[exp]);
        }
        else
        {
            val *= pow(10.0, exp_sign * exp);
        }
    }

    *out_val = sign * val;
    return p;
}

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
        const char *next = fast_parse_float_val(p, end, &out[ii]);
        if (next != NULL)
        {
            p = next;
            continue;
        }

        /* Fallback for edge cases (e.g. hex floats, NaN, Inf) */
        char *strto_next = NULL;
        if (end - p < 64)
        {
            char tmp[64];
            size_t rem = (size_t)(end - p);
            memcpy(tmp, p, rem);
            tmp[rem] = '\0';
            out[ii] = strtof(tmp, &strto_next);
            if (strto_next == tmp)
            {
                return -1;
            }
            p += (strto_next - tmp);
        }
        else
        {
            out[ii] = strtof(p, &strto_next);
            if (strto_next == p)
            {
                return -1;
            }
            p = strto_next;
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
        const char *next = fast_parse_double_val(p, end, &out[ii]);
        if (next != NULL)
        {
            p = next;
            continue;
        }

        /* Fallback for edge cases (e.g. hex floats, NaN, Inf) */
        char *strto_next = NULL;
        if (end - p < 64)
        {
            char tmp[64];
            size_t rem = (size_t)(end - p);
            memcpy(tmp, p, rem);
            tmp[rem] = '\0';
            out[ii] = strtod(tmp, &strto_next);
            if (strto_next == tmp)
            {
                return -1;
            }
            p += (strto_next - tmp);
        }
        else
        {
            out[ii] = strtod(p, &strto_next);
            if (strto_next == p)
            {
                return -1;
            }
            p = strto_next;
        }
    }
    return 0;
}

/**
 * init_ascii() - Initialize the ASCII format frame reader via memory mapping.
 * @filename: Path to the ASCII (.txt) file containing coordinates.
 *
 * Maps the file into memory using mmap, indexes all line offsets in a single scan,
 * and determines the frame width (number of columns) from the first data line.
 *
 * Return: 0 on success, or -1 on failure.
 */
int init_ascii(
    char *filename)
{
    ascii_fd = open(filename, O_RDONLY);
    if (ascii_fd < 0)
    {
        perror("Failed to open ASCII file");
        return -1;
    }

    struct stat st;
    if (fstat(ascii_fd, &st) != 0)
    {
        perror("Failed to stat ASCII file");
        close(ascii_fd);
        ascii_fd = -1;
        return -1;
    }

    if (st.st_size <= 0)
    {
        fprintf(stderr, "Error: Empty ASCII file.\n");
        close(ascii_fd);
        ascii_fd = -1;
        return -1;
    }

    ascii_mmap_size = (size_t)st.st_size;
    ascii_mmap_addr = mmap(NULL, ascii_mmap_size, PROT_READ, MAP_PRIVATE, ascii_fd, 0);
    if (ascii_mmap_addr == MAP_FAILED)
    {
        perror("Failed to mmap ASCII file");
        ascii_mmap_addr = NULL;
        ascii_mmap_size = 0;
        close(ascii_fd);
        ascii_fd = -1;
        return -1;
    }

    posix_madvise(ascii_mmap_addr, ascii_mmap_size, POSIX_MADV_SEQUENTIAL | POSIX_MADV_WILLNEED);

    is_ascii_mode = 1;
    num_frames = 0;

    size_t line_count = 0;
    const char *buf = (const char *)ascii_mmap_addr;
    const char *buf_end = buf + ascii_mmap_size;
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

    ascii_line_offsets = (long *)malloc(line_count * sizeof(long));
    if (ascii_line_offsets == NULL)
    {
        perror("Memory allocation failed");
        close_ascii();
        return -1;
    }

    {
        /* Scan mapped buffer to index lines and determine column count */
        size_t pos = 0;
        int first_line = 1;

        while (pos < ascii_mmap_size)
        {
            /* Find end of line */
            size_t line_end = pos;
            while (line_end < ascii_mmap_size && buf[line_end] != '\n' && buf[line_end] != '\r')
            {
                line_end++;
            }

            /* Find first non-whitespace character in this line */
            size_t p = pos;
            while (p < line_end && isspace((unsigned char)buf[p]))
            {
                p++;
            }

            if (p < line_end && buf[p] != '#')
            {
                /* Valid data line */
                ascii_line_offsets[num_frames++] = (long)p;

                if (first_line)
                {
                    int cols = 0;
                    int in_num = 0;
                    for (size_t k = p; k < line_end; k++)
                    {
                        if (buf[k] == '#')
                        {
                            break;
                        }
                        if (!isspace((unsigned char)buf[k]))
                        {
                            if (!in_num)
                            {
                                cols++;
                                in_num = 1;
                            }
                        }
                        else
                        {
                            in_num = 0;
                        }
                    }
                    frame_width = cols;
                    frame_height = 1;
                    first_line = 0;
                }
            } // if (p < line_end && buf[p] != '#')

            /* Advance pos past line ending */
            pos = line_end;
            if (pos < ascii_mmap_size && buf[pos] == '\r')
            {
                pos++;
            }
            if (pos < ascii_mmap_size && buf[pos] == '\n')
            {
                pos++;
            }
        } // while (pos < ascii_mmap_size)
    }

    if (num_frames == 0)
    {
        fprintf(stderr, "Error: Empty ASCII file.\n");
        close_ascii();
        return -1;
    }

    return 0;
}

/**
 * getframe_ascii() - Read a frame from the mapped ASCII file buffer.
 * @frame_struct: Pointer to the Frame struct to populate.
 * @index:        Zero-based index of the frame to retrieve.
 *
 * Direct memory lookup at the indexed offset of the mapped ASCII buffer,
 * avoiding per-frame kernel seek syscalls.
 *
 * Return: 0 on success, or -1 on failure.
 */
int getframe_ascii(
    Frame *frame_struct,
    long   index)
{
    if (ascii_mmap_addr == NULL || index < 0 || index >= num_frames)
    {
        return -1;
    }

    long nelements = frame_width * frame_height;
    const char *p = (const char *)ascii_mmap_addr + ascii_line_offsets[index];
    const char *end = (const char *)ascii_mmap_addr + ascii_mmap_size;

    if (frame_struct->is_double)
    {
        double *dptr = (double *)frame_struct->data;
        return parse_ascii_row_double(p, end, dptr, nelements);
    }

    float *fptr = (float *)frame_struct->data;
    return parse_ascii_row_float(p, end, fptr, nelements);
}

/**
 * close_ascii() - Unmap ASCII memory mapping and free resources.
 */
void close_ascii(void)
{
    if (ascii_mmap_addr != NULL && ascii_mmap_addr != MAP_FAILED)
    {
        munmap(ascii_mmap_addr, ascii_mmap_size);
        ascii_mmap_addr = NULL;
    }
    ascii_mmap_size = 0;

    if (ascii_fd >= 0)
    {
        close(ascii_fd);
        ascii_fd = -1;
    }

    if (ascii_ptr != NULL)
    {
        fclose(ascii_ptr);
        ascii_ptr = NULL;
    }

    if (ascii_line_offsets != NULL)
    {
        free(ascii_line_offsets);
        ascii_line_offsets = NULL;
    }

    is_ascii_mode = 0;
}

/**
 * reset_ascii() - Reset the ASCII reader to the beginning.
 */
void reset_ascii(void)
{
    /* Mapped ASCII frames use random-access offsets; index is reset by caller */
}

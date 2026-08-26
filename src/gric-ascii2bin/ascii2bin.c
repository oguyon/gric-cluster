/**
 * @file ascii2bin.c
 * @brief Utility to encode ASCII tables and coordinates into self-describing GRIC binary format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <stdint.h>
#include "gric_bin_io.h"

/**
 * print_usage() - Print command-line help for gric-ascii2bin.
 * @prog: Executable name.
 */
static void print_usage(
    const char *prog)
{
    printf("\033[1;36mGRIC ASCII-to-Binary Converter (gric-ascii2bin)\033[0m\n\n");
    printf("Usage:\n");
    printf("  %s <input.txt> <output.bin> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -type <type>        Semantic file type: anchors, dcc, membership, counts, coords, generic\n");
    printf("  -double             Encode floating-point as float64 (default: float32)\n");
    printf("  -uint32             Encode as unsigned 32-bit integers\n");
    printf("  -int32              Encode as signed 32-bit integers\n");
    printf("  -dim <D>            Explicit number of columns/dimensions (default: auto-detected)\n");
    printf("  -comment <text>     Embed description string in header\n");
    printf("  -v, --verbose       Print verbose encoding details\n");
    printf("  -h, --help          Display this help message\n\n");
    printf("Examples:\n");
    printf("  %s 2Dspiral.txt spiral.bin -type coords\n", prog);
    printf("  %s dcc.txt dcc.bin -type dcc -double\n", prog);
    printf("  %s frame_membership.txt membership.bin -type membership -uint32\n", prog);
}

/**
 * count_tokens_in_line() - Count whitespace-separated numbers in a line.
 * @line: Input string.
 *
 * Return: Number of numeric tokens.
 */
static size_t count_tokens_in_line(
    const char *line)
{
    size_t count = 0;
    const char *p = line;

    while (*p != '\0')
    {
        while (*p != '\0' && isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '\0' || *p == '#' || (p[0] == '/' && p[1] == '/'))
        {
            break;
        }
        count++;
        while (*p != '\0' && !isspace((unsigned char)*p))
        {
            p++;
        }
    }

    return count;
}

int main(
    int   argc,
    char *argv[])
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return (argc == 1) ? 0 : 1;
    }

    const char *input_path = NULL;
    const char *output_path = NULL;
    const char *type_str = NULL;
    const char *comment_str = NULL;
    gric_bin_data_type_t dtype = GRIC_BIN_DTYPE_FLOAT32;
    int explicit_dim = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
            verbose = 1;
        }
        else if (strcmp(argv[i], "-double") == 0)
        {
            dtype = GRIC_BIN_DTYPE_FLOAT64;
        }
        else if (strcmp(argv[i], "-uint32") == 0)
        {
            dtype = GRIC_BIN_DTYPE_UINT32;
        }
        else if (strcmp(argv[i], "-int32") == 0)
        {
            dtype = GRIC_BIN_DTYPE_INT32;
        }
        else if (strcmp(argv[i], "-type") == 0 && i + 1 < argc)
        {
            type_str = argv[++i];
        }
        else if (strcmp(argv[i], "-comment") == 0 && i + 1 < argc)
        {
            comment_str = argv[++i];
        }
        else if (strcmp(argv[i], "-dim") == 0 && i + 1 < argc)
        {
            explicit_dim = atoi(argv[++i]);
        }
        else if (argv[i][0] != '-')
        {
            if (input_path == NULL)
            {
                input_path = argv[i];
            }
            else if (output_path == NULL)
            {
                output_path = argv[i];
            }
        }
    }

    if (input_path == NULL || output_path == NULL)
    {
        fprintf(stderr, "Error: Both <input.txt> and <output.bin> are required.\n");
        return 1;
    }

    FILE *in_fp = fopen(input_path, "r");
    if (in_fp == NULL)
    {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return 1;
    }

    char line_buf[65536];
    size_t ncols = (explicit_dim > 0) ? (size_t)explicit_dim : 0;
    size_t nrows = 0;
    size_t cap_elements = 1024;
    size_t num_elements = 0;
    double *raw_data = (double *)malloc(cap_elements * sizeof(double));

    if (raw_data == NULL)
    {
        fclose(in_fp);
        fprintf(stderr, "Error: Out of memory\n");
        return 1;
    }

    while (fgets(line_buf, sizeof(line_buf), in_fp) != NULL)
    {
        char *p = line_buf;
        while (*p != '\0' && isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '\0' || *p == '#' || (p[0] == '/' && p[1] == '/'))
        {
            continue;
        }

        size_t tokens_in_line = count_tokens_in_line(p);
        if (tokens_in_line == 0)
        {
            continue;
        }

        if (ncols == 0)
        {
            ncols = tokens_in_line;
        }

        char *endptr = NULL;
        for (size_t c = 0; c < tokens_in_line; c++)
        {
            double val = strtod(p, &endptr);
            if (p == endptr)
            {
                break;
            }
            p = endptr;

            if (num_elements >= cap_elements)
            {
                cap_elements *= 2;
                double *new_data = (double *)realloc(raw_data, cap_elements * sizeof(double));
                if (new_data == NULL)
                {
                    free(raw_data);
                    fclose(in_fp);
                    fprintf(stderr, "Error: Reallocation failed during parse\n");
                    return 1;
                }
                raw_data = new_data;
            }

            raw_data[num_elements++] = val;
        }

        nrows++;
    }

    fclose(in_fp);

    if (num_elements == 0 || nrows == 0)
    {
        free(raw_data);
        fprintf(stderr, "Error: No data records found in '%s'\n", input_path);
        return 1;
    }

    // Auto-adjust ncols if single column or variable
    if (ncols == 0 || num_elements % nrows != 0)
    {
        ncols = num_elements / nrows;
    }

    gric_bin_file_type_t ftype = gric_bin_file_type_from_str(type_str);
    if (ftype == GRIC_BIN_TYPE_GENERIC)
    {
        if (strstr(input_path, "dcc") != NULL)
        {
            ftype = GRIC_BIN_TYPE_DCC;
        }
        else if (strstr(input_path, "anchor") != NULL || strstr(input_path, "centroid") != NULL)
        {
            ftype = GRIC_BIN_TYPE_ANCHORS;
        }
        else if (strstr(input_path, "membership") != NULL || strstr(input_path, "assign") != NULL)
        {
            ftype = GRIC_BIN_TYPE_MEMBERSHIP;
        }
        else if (strstr(input_path, "count") != NULL)
        {
            ftype = GRIC_BIN_TYPE_COUNTS;
        }
        else if (ncols >= 2)
        {
            ftype = GRIC_BIN_TYPE_COORDINATES;
        }
    }

    gric_bin_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, GRIC_BIN_MAGIC, 4);
    hdr.version = GRIC_BIN_VERSION;
    hdr.endian = GRIC_BIN_ENDIAN_LITTLE;
    size_t comment_len = (comment_str != NULL) ? strlen(comment_str) : 0;
    hdr.header_bytes = (uint16_t)(GRIC_BIN_HEADER_DEFAULT_SIZE + comment_len);
    hdr.file_type = (uint8_t)ftype;
    hdr.data_type = (uint8_t)dtype;
    hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr.num_elements = num_elements;
    hdr.data_bytes = num_elements * gric_bin_data_type_size(dtype);

    if (ncols == 1)
    {
        hdr.ndim = 1;
        hdr.dims[0] = nrows;
    }
    else
    {
        hdr.ndim = 2;
        hdr.dims[0] = nrows;
        hdr.dims[1] = ncols;
    }

    FILE *out_fp = fopen(output_path, "wb");
    if (out_fp == NULL)
    {
        free(raw_data);
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_path);
        return 1;
    }

    if (gric_bin_write_header(out_fp, &hdr, comment_str) != 0)
    {
        free(raw_data);
        fclose(out_fp);
        fprintf(stderr, "Error: Failed to write binary header\n");
        return 1;
    }

    int write_ok = 1;
    if (dtype == GRIC_BIN_DTYPE_FLOAT32)
    {
        float *fbuf = (float *)malloc(num_elements * sizeof(float));
        if (fbuf != NULL)
        {
            for (size_t i = 0; i < num_elements; i++)
            {
                fbuf[i] = (float)raw_data[i];
            }
            if (fwrite(fbuf, sizeof(float), num_elements, out_fp) != num_elements)
            {
                write_ok = 0;
            }
            free(fbuf);
        }
        else
        {
            write_ok = 0;
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_FLOAT64)
    {
        if (fwrite(raw_data, sizeof(double), num_elements, out_fp) != num_elements)
        {
            write_ok = 0;
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_UINT32)
    {
        uint32_t *u32buf = (uint32_t *)malloc(num_elements * sizeof(uint32_t));
        if (u32buf != NULL)
        {
            for (size_t i = 0; i < num_elements; i++)
            {
                u32buf[i] = (uint32_t)raw_data[i];
            }
            if (fwrite(u32buf, sizeof(uint32_t), num_elements, out_fp) != num_elements)
            {
                write_ok = 0;
            }
            free(u32buf);
        }
        else
        {
            write_ok = 0;
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_INT32)
    {
        int32_t *i32buf = (int32_t *)malloc(num_elements * sizeof(int32_t));
        if (i32buf != NULL)
        {
            for (size_t i = 0; i < num_elements; i++)
            {
                i32buf[i] = (int32_t)raw_data[i];
            }
            if (fwrite(i32buf, sizeof(int32_t), num_elements, out_fp) != num_elements)
            {
                write_ok = 0;
            }
            free(i32buf);
        }
        else
        {
            write_ok = 0;
        }
    }

    free(raw_data);
    fclose(out_fp);

    if (!write_ok)
    {
        fprintf(stderr, "Error: Failed to write data payload to '%s'\n", output_path);
        return 1;
    }

    if (verbose)
    {
        printf("Successfully encoded '%s' -> '%s'\n", input_path, output_path);
        gric_bin_print_header_info(stdout, &hdr, comment_str);
    }

    return 0;
}

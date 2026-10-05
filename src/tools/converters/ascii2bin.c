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

typedef struct
{
    const char           *input_path;
    const char           *output_path;
    char                  auto_output_path[1024];
    const char           *type_str;
    const char           *comment_str;
    gric_bin_data_type_t  dtype;
    int                   explicit_dim;
    int                   verbose;
    int                   has_index;
} Ascii2BinConfig;

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
    printf("  -type <type>        Semantic type: anchors, dcc, membership, counts, coords\n");
    printf("  -has-index          Skip first column (e.g. anchor_idx or row index)\n");
    printf("  -no-index           Do not skip first column\n");
    printf("  -double             Encode floating-point as float64 (default: float32)\n");
    printf("  -uint32             Encode as unsigned 32-bit integers\n");
    printf("  -int32              Encode as signed 32-bit integers\n");
    printf("  -uint16, -sq16      Encode as unsigned 16-bit integers (SQ16)\n");
    printf("  -int16, -eq16       Encode as signed 16-bit integers (EQ16)\n");
    printf("  -dim <D>            Explicit column count (default: auto-detected)\n");
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

/**
 * parse_cli_args() - Parse CLI options into Ascii2BinConfig.
 * @argc: Argument count.
 * @argv: Argument vector.
 * @cfg:  Config structure to populate.
 *
 * Return: 0 on success, 1 on error, 2 if help displayed.
 */
static int parse_cli_args(
    int              argc,
    char            *argv[],
    Ascii2BinConfig *cfg)
{
    memset(cfg, 0, sizeof(Ascii2BinConfig));
    cfg->dtype = GRIC_BIN_DTYPE_FLOAT32;
    cfg->has_index = -1;

    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h") == 0 || strcmp(argv[ii], "--help") == 0)
        {
            print_usage(argv[0]);
            return 2;
        }
        if (strcmp(argv[ii], "-v") == 0 || strcmp(argv[ii], "--verbose") == 0)
        {
            cfg->verbose = 1;
        }
        else if (strcmp(argv[ii], "-has-index") == 0 || strcmp(argv[ii], "--has-index") == 0)
        {
            cfg->has_index = 1;
        }
        else if (strcmp(argv[ii], "-no-index") == 0)
        {
            cfg->has_index = 0;
        }
        else if (strcmp(argv[ii], "-double") == 0)
        {
            cfg->dtype = GRIC_BIN_DTYPE_FLOAT64;
        }
        else if (strcmp(argv[ii], "-uint32") == 0)
        {
            cfg->dtype = GRIC_BIN_DTYPE_UINT32;
        }
        else if (strcmp(argv[ii], "-int32") == 0)
        {
            cfg->dtype = GRIC_BIN_DTYPE_INT32;
        }
        else if (strcmp(argv[ii], "-uint16") == 0 || strcmp(argv[ii], "-sq16") == 0)
        {
            cfg->dtype = GRIC_BIN_DTYPE_UINT16;
        }
        else if (strcmp(argv[ii], "-int16") == 0 || strcmp(argv[ii], "-eq16") == 0)
        {
            cfg->dtype = GRIC_BIN_DTYPE_INT16;
        }
        else if (strcmp(argv[ii], "-type") == 0 && ii + 1 < argc)
        {
            cfg->type_str = argv[++ii];
        }
        else if (strcmp(argv[ii], "-comment") == 0 && ii + 1 < argc)
        {
            cfg->comment_str = argv[++ii];
        }
        else if (strcmp(argv[ii], "-dim") == 0 && ii + 1 < argc)
        {
            cfg->explicit_dim = atoi(argv[++ii]);
        }
        else if (argv[ii][0] != '-')
        {
            if (cfg->input_path == NULL)
            {
                cfg->input_path = argv[ii];
            }
            else if (cfg->output_path == NULL)
            {
                cfg->output_path = argv[ii];
            }
        }
    }

    if (cfg->input_path != NULL && cfg->output_path == NULL)
    {
        snprintf(cfg->auto_output_path, sizeof(cfg->auto_output_path), "%s", cfg->input_path);
        char *dot = strrchr(cfg->auto_output_path, '.');
        if (dot != NULL)
        {
            strcpy(dot, ".bin");
        }
        else
        {
            strncat(cfg->auto_output_path, ".bin",
                    sizeof(cfg->auto_output_path) - strlen(cfg->auto_output_path) - 1);
        }
        cfg->output_path = cfg->auto_output_path;
    }
    else if (cfg->output_path != NULL && strrchr(cfg->output_path, '.') == NULL)
    {
        snprintf(cfg->auto_output_path, sizeof(cfg->auto_output_path), "%s.bin", cfg->output_path);
        cfg->output_path = cfg->auto_output_path;
    }

    if (cfg->input_path == NULL || cfg->output_path == NULL)
    {
        fprintf(stderr, "Error: <input.txt> is required.\n");
        return 1;
    }
    return 0;
}

/**
 * parse_line_tokens() - Parse numeric tokens from line and append to raw_data buffer.
 * @p:            Start of numeric text on line.
 * @tokens_line:  Number of tokens to parse.
 * @has_index:    Whether to skip first token.
 * @raw_data_ptr: Pointer to allocated double array.
 * @num_elem_ptr: Pointer to element count.
 * @cap_elem_ptr: Pointer to buffer capacity.
 *
 * Return: 0 on success, -1 on allocation failure.
 */
static int parse_line_tokens(
    char    *p,
    size_t   tokens_line,
    int      has_index,
    double **raw_data_ptr,
    size_t  *num_elem_ptr,
    size_t  *cap_elem_ptr)
{
    char *endptr = NULL;
    if (has_index)
    {
        strtod(p, &endptr);
        p = endptr;
    }

    for (size_t cc = 0; cc < tokens_line; cc++)
    {
        double val = strtod(p, &endptr);
        if (p == endptr)
        {
            break;
        }
        p = endptr;

        if (*num_elem_ptr >= *cap_elem_ptr)
        {
            *cap_elem_ptr *= 2;
            double *new_data = (double *)realloc(*raw_data_ptr, *cap_elem_ptr * sizeof(double));
            if (new_data == NULL)
            {
                return -1;
            }
            *raw_data_ptr = new_data;
        }

        (*raw_data_ptr)[(*num_elem_ptr)++] = val;
    }
    return 0;
}

/**
 * parse_ascii_table() - Ingest numbers from an ASCII file into a contiguous double array.
 * @cfg:          Converter configuration.
 * @out_data:     Pointer to store allocated double array.
 * @out_nrows:    Pointer to store row count.
 * @out_ncols:    Pointer to store column count.
 * @out_nelem:    Pointer to store total element count.
 *
 * Return: 0 on success, -1 on failure.
 */
static int parse_ascii_table(
    Ascii2BinConfig  *cfg,
    double          **out_data,
    size_t           *out_nrows,
    size_t           *out_ncols,
    size_t           *out_nelem)
{
    FILE *in_fp = fopen(cfg->input_path, "r");
    if (in_fp == NULL)
    {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", cfg->input_path);
        return -1;
    }

    char line_buf[65536];
    size_t ncols = (cfg->explicit_dim > 0) ? (size_t)cfg->explicit_dim : 0;
    size_t nrows = 0, num_elements = 0, cap_elements = 1024;
    double *raw_data = (double *)malloc(cap_elements * sizeof(double));
    if (raw_data == NULL)
    {
        fclose(in_fp);
        return -1;
    }

    while (fgets(line_buf, sizeof(line_buf), in_fp) != NULL)
    {
        char *p = line_buf;
        while (*p != '\0' && isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '\0')
        {
            continue;
        }
        if (*p == '#' || (p[0] == '/' && p[1] == '/'))
        {
            if (cfg->has_index == -1 && (strstr(p, "anchor_idx") != NULL ||
                                         strstr(p, "cluster_idx") != NULL ||
                                         strstr(p, "row_idx") != NULL ||
                                         strstr(p, "sample_idx") != NULL))
            {
                cfg->has_index = 1;
            }
            continue;
        }
        if (cfg->has_index == -1)
        {
            cfg->has_index = 0;
        }

        size_t total = count_tokens_in_line(p);
        if (total == 0 || (cfg->has_index && total <= 1))
        {
            continue;
        }
        size_t tokens_line = cfg->has_index ? (total - 1) : total;
        if (ncols == 0)
        {
            ncols = tokens_line;
        }

        if (parse_line_tokens(p, tokens_line, cfg->has_index, &raw_data,
                              &num_elements, &cap_elements) != 0)
        {
            free(raw_data);
            fclose(in_fp);
            return -1;
        }
        nrows++;
    }
    fclose(in_fp);

    if (num_elements == 0 || nrows == 0)
    {
        free(raw_data);
        return -1;
    }
    if (ncols == 0 || num_elements % nrows != 0)
    {
        ncols = num_elements / nrows;
    }

    *out_data = raw_data;
    *out_nrows = nrows;
    *out_ncols = ncols;
    *out_nelem = num_elements;
    return 0;
}

/**
 * infer_file_type() - Deduce semantic file type from type argument or filename.
 * @cfg:   Configuration options.
 * @ncols: Detected column count.
 *
 * Return: Inferred gric_bin_file_type_t enum value.
 */
static gric_bin_file_type_t infer_file_type(
    const Ascii2BinConfig *cfg,
    size_t                 ncols)
{
    gric_bin_file_type_t ftype = gric_bin_file_type_from_str(cfg->type_str);
    if (ftype != GRIC_BIN_TYPE_GENERIC)
    {
        return ftype;
    }

    if (strstr(cfg->input_path, "dcc") != NULL)
    {
        return GRIC_BIN_TYPE_DCC;
    }
    if (strstr(cfg->input_path, "anchor") != NULL ||
        strstr(cfg->input_path, "centroid") != NULL)
    {
        return GRIC_BIN_TYPE_ANCHORS;
    }
    if (strstr(cfg->input_path, "membership") != NULL ||
        strstr(cfg->input_path, "assign") != NULL)
    {
        return GRIC_BIN_TYPE_MEMBERSHIP;
    }
    if (strstr(cfg->input_path, "count") != NULL)
    {
        return GRIC_BIN_TYPE_COUNTS;
    }
    if (ncols >= 2)
    {
        return GRIC_BIN_TYPE_COORDINATES;
    }
    return GRIC_BIN_TYPE_GENERIC;
}

/**
 * write_binary_payload() - Convert double values to target dtype and write to disk.
 * @out_fp:       Open destination file handle.
 * @dtype:        Target data type.
 * @raw_data:     Input double array.
 * @num_elements: Total number of values to write.
 *
 * Return: 0 on success, -1 on write failure.
 */
static int write_binary_payload(
    FILE                 *out_fp,
    gric_bin_data_type_t  dtype,
    const double         *raw_data,
    size_t                num_elements)
{
    if (dtype == GRIC_BIN_DTYPE_FLOAT64)
    {
        return (fwrite(raw_data, sizeof(double), num_elements, out_fp) == num_elements)
               ? 0 : -1;
    }

    void *buf = malloc(num_elements * gric_bin_data_type_size(dtype));
    if (buf == NULL)
    {
        return -1;
    }

    switch (dtype)
    {
        case GRIC_BIN_DTYPE_FLOAT32:
        {
            float *b = (float *)buf;
            for (size_t ii = 0; ii < num_elements; ii++)
            {
                b[ii] = (float)raw_data[ii];
            }
            break;
        }
        case GRIC_BIN_DTYPE_UINT32:
        {
            uint32_t *b = (uint32_t *)buf;
            for (size_t ii = 0; ii < num_elements; ii++)
            {
                b[ii] = (uint32_t)raw_data[ii];
            }
            break;
        }
        case GRIC_BIN_DTYPE_INT32:
        {
            int32_t *b = (int32_t *)buf;
            for (size_t ii = 0; ii < num_elements; ii++)
            {
                b[ii] = (int32_t)raw_data[ii];
            }
            break;
        }
        case GRIC_BIN_DTYPE_UINT16:
        {
            uint16_t *b = (uint16_t *)buf;
            for (size_t ii = 0; ii < num_elements; ii++)
            {
                b[ii] = (uint16_t)raw_data[ii];
            }
            break;
        }
        case GRIC_BIN_DTYPE_INT16:
        {
            int16_t *b = (int16_t *)buf;
            for (size_t ii = 0; ii < num_elements; ii++)
            {
                b[ii] = (int16_t)raw_data[ii];
            }
            break;
        }
        default:
            free(buf);
            return -1;
    }

    size_t written = fwrite(buf, gric_bin_data_type_size(dtype), num_elements, out_fp);
    free(buf);
    return (written == num_elements) ? 0 : -1;
}

/**
 * convert_ascii_to_bin() - Orchestrate reading ASCII table and writing self-describing binary.
 * @cfg: Converter options.
 *
 * Return: 0 on success, non-zero on error.
 */
static int convert_ascii_to_bin(
    Ascii2BinConfig *cfg)
{
    double *raw_data = NULL;
    size_t nrows = 0, ncols = 0, num_elements = 0;
    if (parse_ascii_table(cfg, &raw_data, &nrows, &ncols, &num_elements) != 0)
    {
        fprintf(stderr, "Error: Failed to parse input file '%s'\n", cfg->input_path);
        return 1;
    }

    gric_bin_file_type_t ftype = infer_file_type(cfg, ncols);
    gric_bin_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, GRIC_BIN_MAGIC, 4);
    hdr.version = GRIC_BIN_VERSION;
    hdr.endian = GRIC_BIN_ENDIAN_LITTLE;
    size_t comment_len = (cfg->comment_str != NULL) ? strlen(cfg->comment_str) : 0;
    hdr.header_bytes = (uint16_t)(GRIC_BIN_HEADER_DEFAULT_SIZE + comment_len);
    hdr.file_type = (uint8_t)ftype;
    hdr.data_type = (uint8_t)cfg->dtype;
    hdr.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr.num_elements = num_elements;
    hdr.data_bytes = num_elements * gric_bin_data_type_size(cfg->dtype);

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

    FILE *out_fp = fopen(cfg->output_path, "wb");
    if (out_fp == NULL)
    {
        free(raw_data);
        fprintf(stderr, "Error: Cannot create output file '%s'\n", cfg->output_path);
        return 1;
    }

    if (gric_bin_write_header(out_fp, &hdr, cfg->comment_str) != 0 ||
        write_binary_payload(out_fp, cfg->dtype, raw_data, num_elements) != 0)
    {
        free(raw_data);
        fclose(out_fp);
        fprintf(stderr, "Error: Failed to write binary data to '%s'\n", cfg->output_path);
        return 1;
    }

    free(raw_data);
    fclose(out_fp);

    if (cfg->verbose)
    {
        printf("Successfully encoded '%s' -> '%s'\n", cfg->input_path, cfg->output_path);
        gric_bin_print_header_info(stdout, &hdr, cfg->comment_str);
    }
    return 0;
}

int main(
    int   argc,
    char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 0;
    }

    Ascii2BinConfig cfg;
    int parse_rc = parse_cli_args(argc, argv, &cfg);
    if (parse_rc != 0)
    {
        return (parse_rc == 2) ? 0 : 1;
    }

    return convert_ascii_to_bin(&cfg);
}

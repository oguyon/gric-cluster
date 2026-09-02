/**
 * @file bin2ascii.c
 * @brief Utility to decode self-describing GRIC binary files into ASCII tables or stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include "gric_bin_io.h"

/**
 * print_usage() - Print command-line help for gric-bin2ascii.
 * @prog: Executable name.
 */
static void print_usage(
    const char *prog)
{
    printf("\033[1;36mGRIC Binary-to-ASCII Decoder (gric-bin2ascii)\033[0m\n\n");
    printf("Usage:\n");
    printf("  %s <input.bin> [output.txt] [options]\n\n", prog);
    printf("Options:\n");
    printf("  -info, -i           Display header metadata summary without decoding payload\n");
    printf("  -fmt <specifier>    Custom printf formatting specifier (e.g. '%%.8f', '%%g')\n");
    printf("  -v, --verbose       Print decoding summary to stderr\n");
    printf("  -h, --help          Display this help message\n\n");
    printf("Notes:\n");
    printf("  If [output.txt] is omitted or '-', decoded ASCII is piped directly to stdout.\n\n");
    printf("Examples:\n");
    printf("  %s spiral.bin -info\n", prog);
    printf("  %s spiral.bin spiral_reconstructed.txt\n", prog);
    printf("  %s dcc.bin - | head -n 10\n", prog);
}

int main(
    int   argc,
    char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return (argc == 1) ? 0 : 1;
    }

    const char *input_path = NULL;
    const char *output_path = NULL;
    const char *custom_fmt = NULL;
    int info_only = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-info") == 0 || strcmp(argv[i], "-i") == 0)
        {
            info_only = 1;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
            verbose = 1;
        }
        else if (strcmp(argv[i], "-fmt") == 0 && i + 1 < argc)
        {
            custom_fmt = argv[++i];
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

    if (input_path == NULL)
    {
        fprintf(stderr, "Error: Input binary file <input.bin> is required.\n");
        return 1;
    }

    FILE *in_fp = fopen(input_path, "rb");
    if (in_fp == NULL)
    {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_path);
        return 1;
    }

    gric_bin_header_t hdr;
    char *comment = NULL;
    if (gric_bin_read_header(in_fp, &hdr, &comment) != 0)
    {
        fclose(in_fp);
        fprintf(stderr, "Error: '%s' is not a valid GRIC binary file.\n", input_path);
        return 1;
    }

    if (info_only)
    {
        gric_bin_print_header_info(stdout, &hdr, comment);
        if (comment != NULL)
        {
            free(comment);
        }
        fclose(in_fp);
        return 0;
    }

    FILE *out_fp = stdout;
    int close_out = 0;
    if (output_path != NULL && strcmp(output_path, "-") != 0)
    {
        out_fp = fopen(output_path, "w");
        if (out_fp == NULL)
        {
            if (comment != NULL) free(comment);
            fclose(in_fp);
            fprintf(stderr, "Error: Cannot open output file '%s' for writing.\n", output_path);
            return 1;
        }
        close_out = 1;
    }

    size_t nrows = (hdr.dims[0] > 0) ? (size_t)hdr.dims[0] : 1;
    size_t ncols = (hdr.ndim > 1 && hdr.dims[1] > 0) ? (size_t)hdr.dims[1] : 1;
    size_t total_elements = (size_t)hdr.num_elements;

    if (nrows * ncols != total_elements && hdr.ndim <= 2)
    {
        if (ncols == 0) ncols = 1;
        nrows = total_elements / ncols;
    }

    gric_bin_data_type_t dtype = (gric_bin_data_type_t)hdr.data_type;

    if (dtype == GRIC_BIN_DTYPE_FLOAT32)
    {
        const char *fmt = (custom_fmt != NULL) ? custom_fmt : "%.6f";
        float *fbuf = (float *)malloc(ncols * sizeof(float));
        if (fbuf != NULL)
        {
            for (size_t r = 0; r < nrows; r++)
            {
                if (fread(fbuf, sizeof(float), ncols, in_fp) != ncols)
                {
                    break;
                }
                for (size_t c = 0; c < ncols; c++)
                {
                    fprintf(out_fp, fmt, fbuf[c]);
                    if (c + 1 < ncols) fprintf(out_fp, " ");
                }
                fprintf(out_fp, "\n");
            }
            free(fbuf);
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_FLOAT64)
    {
        const char *fmt = (custom_fmt != NULL) ? custom_fmt : "%.10f";
        double *dbuf = (double *)malloc(ncols * sizeof(double));
        if (dbuf != NULL)
        {
            for (size_t r = 0; r < nrows; r++)
            {
                if (fread(dbuf, sizeof(double), ncols, in_fp) != ncols)
                {
                    break;
                }
                for (size_t c = 0; c < ncols; c++)
                {
                    fprintf(out_fp, fmt, dbuf[c]);
                    if (c + 1 < ncols) fprintf(out_fp, " ");
                }
                fprintf(out_fp, "\n");
            }
            free(dbuf);
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_UINT32)
    {
        const char *fmt = (custom_fmt != NULL) ? custom_fmt : "%u";
        uint32_t *u32buf = (uint32_t *)malloc(ncols * sizeof(uint32_t));
        if (u32buf != NULL)
        {
            for (size_t r = 0; r < nrows; r++)
            {
                if (fread(u32buf, sizeof(uint32_t), ncols, in_fp) != ncols)
                {
                    break;
                }
                for (size_t c = 0; c < ncols; c++)
                {
                    fprintf(out_fp, fmt, u32buf[c]);
                    if (c + 1 < ncols) fprintf(out_fp, " ");
                }
                fprintf(out_fp, "\n");
            }
            free(u32buf);
        }
    }
    else if (dtype == GRIC_BIN_DTYPE_INT32)
    {
        const char *fmt = (custom_fmt != NULL) ? custom_fmt : "%d";
        int32_t *i32buf = (int32_t *)malloc(ncols * sizeof(int32_t));
        if (i32buf != NULL)
        {
            for (size_t r = 0; r < nrows; r++)
            {
                if (fread(i32buf, sizeof(int32_t), ncols, in_fp) != ncols)
                {
                    break;
                }
                for (size_t c = 0; c < ncols; c++)
                {
                    fprintf(out_fp, fmt, i32buf[c]);
                    if (c + 1 < ncols) fprintf(out_fp, " ");
                }
                fprintf(out_fp, "\n");
            }
            free(i32buf);
        }
    }
    else
    {
        fprintf(stderr, "Error: Unsupported binary data type code %u\n", dtype);
    }

    if (comment != NULL)
    {
        free(comment);
    }

    fclose(in_fp);
    if (close_out)
    {
        fclose(out_fp);
    }

    if (verbose)
    {
        fprintf(stderr, "Decoded %lu rows x %lu cols (%lu elements) from '%s'\n",
                (unsigned long)nrows, (unsigned long)ncols,
                (unsigned long)total_elements, input_path);
    }

    return 0;
}

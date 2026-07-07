/**
 * @file plot_clusters.c
 * @brief Diagnostic scatter and histogram plotting utility.
 *
 * Processes coordinate files and clustering logs to output scatter plots and
 * histograms in PNG or SVG format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shared/cli_colors.h"
#include "plot_internal.h"

/**
 * @brief Print the CLI help message.
 *
 * @param progname The name of the program.
 */
void print_help(const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-plot%s - Visualization tool for clustering results\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s[options]%s %s<points_file>%s %s<log_file>%s %s[output_file]%s\n\n",
           ansi_bold_green, progname, ansi_reset, ansi_color_grey, ansi_reset,
           ansi_color_magenta, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_grey,
           ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  gric-plot is a visualization tool for clustering results "
           "produced by gric-cluster.\n");
    printf("  It generates a comprehensive summary image (PNG or SVG) containing:\n");
    printf("    1. A scatter plot of the clustered data points, color-coded by cluster ID.\n");
    printf("    2. Anchor points marked with circles representing the radius limit (rlim).\n");
    printf("    3. A distance histogram (Samples vs. Distance Count).\n");
    printf("    4. A cluster size histogram (Samples per Cluster).\n");
    printf("    5. A Cluster-to-Cluster Distance Matrix (if dcc.txt is available).\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-h, --help%s           Show this help message\n", ansi_color_green, ansi_reset);
    printf("  %s-svg%s                 Output SVG image instead of PNG (%sdefault:%s%s PNG%s)\n",
           ansi_color_green, ansi_reset, ansi_color_cyan, ansi_reset, ansi_color_cyan, ansi_reset);
    printf("  %s-fs%s %s<size>%s           Set font size for text labels "
           "(%sdefault:%s%s 18.0%s)\n\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_cyan,
           ansi_reset, ansi_color_cyan, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s input_points.txt gric_run_log.txt summary_plot.png\n", ansi_color_grey,
           ansi_reset, ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
} // print_help

int main(
    int     argc,
    char  **argv)
{
    cli_colors_init();
    char *points_filename = NULL;
    char *log_filename = NULL;
    char output_filename[1024] = {0};
    int png_mode = 1;
    double font_size = 18.0;
    int arg_idx = 1;

    while (arg_idx < argc)
    {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[arg_idx], "-svg") == 0)
        {
            png_mode = 0;
        }
        else if (strcmp(argv[arg_idx], "-fs") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                font_size = atof(argv[++arg_idx]);
            }
        }
        else if (argv[arg_idx][0] == '-')
        {
            fprintf(stderr, "Unknown: %s\n", argv[arg_idx]);
            return 1;
        }
        else
        {
            if (points_filename == NULL)
            {
                points_filename = argv[arg_idx];
            }
            else if (log_filename == NULL)
            {
                log_filename = argv[arg_idx];
            }
            else
            {
                strncpy(output_filename, argv[arg_idx], 1023);
                output_filename[1023] = '\0';
            }
        }
        arg_idx++;
    } // while (arg_idx < argc)

#ifndef USE_PNG
    png_mode = 0;
#endif

    if (points_filename == NULL || log_filename == NULL)
    {
        print_help(argv[0]);
        return 1;
    }

    PlotData *data = (PlotData *)calloc(1, sizeof(PlotData));
    if (data == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for PlotData\n");
        return 1;
    }

    if (parse_input_data(points_filename, log_filename, data) != 0)
    {
        free(data);
        return 1;
    }

    if (output_filename[0] == '\0')
    {
        strncpy(output_filename, points_filename, sizeof(output_filename) - 1);
        output_filename[sizeof(output_filename) - 1] = '\0';
        char *e = strrchr(output_filename, '.');
        if (e != NULL)
        {
            *e = '\0';
        }
        strncat(output_filename, png_mode != 0 ? ".png" : ".svg",
                sizeof(output_filename) - strlen(output_filename) - 1);
    }

    char queries_output_filename[1024];
    strncpy(queries_output_filename, output_filename, sizeof(queries_output_filename) - 1);
    queries_output_filename[sizeof(queries_output_filename) - 1] = '\0';
    char *ext_q = strrchr(queries_output_filename, '.');
    if (ext_q != NULL)
    {
        char temp_ext[64];
        strncpy(temp_ext, ext_q, sizeof(temp_ext) - 1);
        temp_ext[sizeof(temp_ext) - 1] = '\0';
        strcpy(ext_q, ".queries");
        strncat(queries_output_filename, temp_ext,
                sizeof(queries_output_filename) - strlen(queries_output_filename) - 1);
    }
    else
    {
        strncat(queries_output_filename, ".queries",
                sizeof(queries_output_filename) - strlen(queries_output_filename) - 1);
    }

    int ret = render_plots(points_filename, output_filename, queries_output_filename,
                           data, png_mode, font_size);

    free(data);
    return ret;
} // main
/**
 * @file main.c
 * @brief Entry point for gric-dimdensity: MLE Local Intrinsic Dimension & Density Estimator.
 */

#define _POSIX_C_SOURCE 200809L
#include "dimdensity_defs.h"
#include "dimdensity_loader.h"
#include "dimdensity_mack.h"
#include "dimdensity_mle.h"
#include "dimdensity_stats.h"
#include "dimdensity_writer.h"
#include "shared/cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s <knn_distances_or_dir> [options]\n", progname);
}

static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-dimdensity%s - Local Intrinsic Dimension (MLE) & Mack-Rosenblatt Density\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s<knn_input_or_dir>%s %s[options]%s\n\n",
           ansi_bold_green, progname, ansi_reset,
           ansi_color_magenta, ansi_reset,
           ansi_color_grey, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Estimates the local intrinsic dimension (LID) around each query sample using\n");
    printf("  the Levina-Bickel Maximum Likelihood Estimator (MLE) with finite-sample\n");
    printf("  unbiased correction, and computes the Mack-Rosenblatt k-NN probability density\n");
    printf("  adapted to the local manifold geometry.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-k%s %s<int>%s              Number of nearest neighbors to evaluate "
           "(%sdefault:%s all available)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset,
           ansi_color_cyan, ansi_reset);
    printf("  %s-kmin%s %s<int>%s           Minimum k for multi-scale range averaging\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-kmax%s %s<int>%s           Maximum k for multi-scale range averaging\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-range%s                Enable multi-scale range averaging across [kmin, kmax]\n",
           ansi_color_green, ansi_reset);
    printf("  %s-classic%s, %s-mle-classic%s Use classic (k-1) MLE instead of (k-2) unbiased\n",
           ansi_color_green, ansi_reset, ansi_color_green, ansi_reset);
    printf("  %s-kernel%s %s<type>%s        Density kernel: uniform (default), "
           "epanechnikov, gaussian\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-o, --output%s %s<path>%s   Output file path (default: dimdensity.txt + .bin)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-bin%s                  Force GRIC binary output format\n",
           ansi_color_green, ansi_reset);
    printf("  %s-fits%s                 Force FITS output format\n",
           ansi_color_green, ansi_reset);
    printf("  %s-txt%s                  Force ASCII text output format\n",
           ansi_color_green, ansi_reset);
    printf("  %s-json%s                 Output structured JSON summary report\n",
           ansi_color_green, ansi_reset);
    printf("  %s-nthreads%s %s<int>%s       Number of OpenMP worker threads\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-v, -vv%s               Verbosity level\n",
           ansi_color_green, ansi_reset);
    printf("  %s-h, --help%s            Show this help message\n\n",
           ansi_color_green, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s cluster_out/ -k 15\n",
           ansi_color_grey, ansi_reset, ansi_bold_green, progname, ansi_reset);
    printf("  %s$%s %s%s%s knn_distances.bin -kmin 5 -kmax 20 -range -o results.txt\n\n",
           ansi_color_grey, ansi_reset, ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
}

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    DimDensityConfig config;
    memset(&config, 0, sizeof(DimDensityConfig));
    config.k = 0;              // 0 = use all available
    config.unbiased_mle = 1;   // Default: (k-2) unbiased MLE
    config.kernel_type = DIMDENSITY_KERNEL_UNIFORM;
    config.output_format = DIMDENSITY_FORMAT_AUTO;
    config.verbose_level = 1;

    int arg_idx = 1;
    while (arg_idx < argc)
    {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[arg_idx], "-k") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                config.k = atoi(argv[++arg_idx]);
            }
            else
            {
                fprintf(stderr, "Error: -k requires an integer argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-kmin") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                config.k_min = atoi(argv[++arg_idx]);
                config.use_range_avg = 1;
            }
            else
            {
                fprintf(stderr, "Error: -kmin requires an integer argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-kmax") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                config.k_max = atoi(argv[++arg_idx]);
                config.use_range_avg = 1;
            }
            else
            {
                fprintf(stderr, "Error: -kmax requires an integer argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-range") == 0 ||
                 strcmp(argv[arg_idx], "--range") == 0 ||
                 strcmp(argv[arg_idx], "-R") == 0)
        {
            config.use_range_avg = 1;
            if (arg_idx + 2 < argc && argv[arg_idx + 1][0] != '-' && argv[arg_idx + 2][0] != '-')
            {
                config.k_min = atoi(argv[++arg_idx]);
                config.k_max = atoi(argv[++arg_idx]);
            }
        }
        else if (strcmp(argv[arg_idx], "-classic") == 0 ||
                 strcmp(argv[arg_idx], "-mle-classic") == 0 ||
                 strcmp(argv[arg_idx], "--classic") == 0)
        {
            config.unbiased_mle = 0;
        }
        else if (strcmp(argv[arg_idx], "-kernel") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                const char *k_str = argv[++arg_idx];
                if (strcasecmp(k_str, "uniform") == 0)
                {
                    config.kernel_type = DIMDENSITY_KERNEL_UNIFORM;
                }
                else if (strcasecmp(k_str, "epanechnikov") == 0 || strcasecmp(k_str, "epan") == 0)
                {
                    config.kernel_type = DIMDENSITY_KERNEL_EPANECHNIKOV;
                }
                else if (strcasecmp(k_str, "gaussian") == 0 || strcasecmp(k_str, "gauss") == 0)
                {
                    config.kernel_type = DIMDENSITY_KERNEL_GAUSSIAN;
                }
                else
                {
                    fprintf(stderr, "Error: Unknown kernel '%s'\n", k_str);
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "Error: -kernel requires a type argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-o") == 0 || strcmp(argv[arg_idx], "--output") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                config.output_path = argv[++arg_idx];
            }
            else
            {
                fprintf(stderr, "Error: -o requires a path argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-bin") == 0)
        {
            config.output_format = DIMDENSITY_FORMAT_BIN;
        }
        else if (strcmp(argv[arg_idx], "-fits") == 0)
        {
            config.output_format = DIMDENSITY_FORMAT_FITS;
        }
        else if (strcmp(argv[arg_idx], "-txt") == 0)
        {
            config.output_format = DIMDENSITY_FORMAT_TXT;
        }
        else if (strcmp(argv[arg_idx], "-json") == 0)
        {
            config.json_mode = 1;
        }
        else if (strcmp(argv[arg_idx], "-nthreads") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                config.nthreads = atoi(argv[++arg_idx]);
            }
            else
            {
                fprintf(stderr, "Error: -nthreads requires an integer argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[arg_idx], "-v") == 0)
        {
            config.verbose_level = 2;
        }
        else if (strcmp(argv[arg_idx], "-vv") == 0)
        {
            config.verbose_level = 3;
        }
        else if (argv[arg_idx][0] == '-')
        {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[arg_idx]);
            print_usage(argv[0]);
            return 1;
        }
        else
        {
            if (config.input_path == NULL)
            {
                config.input_path = argv[arg_idx];
            }
            else
            {
                fprintf(stderr, "Error: Too many positional arguments\n");
                print_usage(argv[0]);
                return 1;
            }
        }
        arg_idx++;
    } // while parsing arguments

    if (config.input_path == NULL)
    {
        fprintf(stderr, "Error: Missing required input path or directory\n");
        print_usage(argv[0]);
        return 1;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // 1. Load distance dataset
    KnnDistanceData dist_data;
    if (dimdensity_load_distances(config.input_path, &dist_data, config.verbose_level) != 0)
    {
        fprintf(stderr, "Error: Failed to load k-NN distance data\n");
        return 1;
    }

    // 2. Compute MLE Intrinsic Dimensions
    DimDensityResults results;
    memset(&results, 0, sizeof(DimDensityResults));

    if (dimdensity_compute_mle_dimensions(&dist_data, &config, &results) != 0)
    {
        fprintf(stderr, "Error: MLE intrinsic dimension estimation failed\n");
        dimdensity_free_distances(&dist_data);
        return 1;
    }

    // 3. Compute Mack-Rosenblatt Local Density
    if (dimdensity_compute_mack_density(&dist_data, &config, &results) != 0)
    {
        fprintf(stderr, "Error: Mack-Rosenblatt density estimation failed\n");
        if (results.local_dim != NULL)
        {
            free(results.local_dim);
        }
        if (results.rk_dist != NULL)
        {
            free(results.rk_dist);
        }
        dimdensity_free_distances(&dist_data);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    // 4. Compute Statistical Summary & Histograms
    DimDensityStats stats;
    if (dimdensity_compute_stats(&results, &stats, 15) != 0)
    {
        fprintf(stderr, "Error: Failed to compute summary statistics\n");
    }

    // 5. Output Results
    if (config.json_mode)
    {
        dimdensity_write_json_report(
            stdout, &config, &dist_data, &results, &stats, elapsed_ms);
        if (config.output_path != NULL)
        {
            dimdensity_write_results(&config, &dist_data, &results, &stats);
        }
    }
    else
    {
        dimdensity_print_dashboard(&config, &dist_data, &results, &stats, elapsed_ms);
        dimdensity_write_results(&config, &dist_data, &results, &stats);
    }

    // Cleanup
    dimdensity_free_stats(&stats);
    if (results.local_dim != NULL)
    {
        free(results.local_dim);
    }
    if (results.density != NULL)
    {
        free(results.density);
    }
    if (results.log_density != NULL)
    {
        free(results.log_density);
    }
    if (results.rk_dist != NULL)
    {
        free(results.rk_dist);
    }
    dimdensity_free_distances(&dist_data);

    return 0;
}

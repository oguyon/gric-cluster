/**
 * @file mktestseq.c
 * @brief Synthetic sequence coordinate generator tool.
 */

#define _POSIX_C_SOURCE 200809L
#include "shared/gric_gen_patterns.h"
#include "shared/cli_colors.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void print_args_on_error(
    int   argc,
    char *argv[])
{
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++)
    {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
} // print_args_on_error

static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s <N> <output_file> <pattern> [options]\n", progname);
} // print_usage

static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-mktxtseq%s - Synthetic sequence generator for testing\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s<N>%s %s<output_file>%s %s<pattern>%s %s[options]%s\n\n", ansi_bold_green,
           progname, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_magenta, ansi_reset,
           ansi_color_magenta, ansi_reset, ansi_color_grey, ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Generates synthetic coordinate sequences (walk, spiral, circle, etc.) for testing.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-repeat%s %s<M>%s          Repeat the pattern M times\n", ansi_color_green,
           ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-noise%s %s<R>%s           Add random noise with radius R to each point\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset);
    printf("  %s-shuffle%s             Shuffle the order of generated points\n\n",
           ansi_color_green, ansi_reset);
    printf("  Patterns:\n");
    printf("    %s[ND]random%s         Uniform random in unit hypercube/sphere (%sdefault:%s%s 2D%s)\n",
           ansi_color_green, ansi_reset, ansi_color_cyan, ansi_reset, ansi_color_cyan, ansi_reset);
    printf("    %s[ND]sphere%s         Random points on unit hypersphere surface\n",
           ansi_color_green, ansi_reset);
    printf("    %s[ND]walk[S]%s        Random walk (%sS = step size%s, %sdefault:%s%s 0.1%s)\n",
           ansi_color_green, ansi_reset, ansi_color_grey, ansi_reset, ansi_color_cyan, ansi_reset,
           ansi_color_cyan, ansi_reset);
    printf("    %s[ND]spiral[L]%s      Spiral (%sL = loops%s, %sdefault:%s%s 2.0%s)\n",
           ansi_color_green, ansi_reset, ansi_color_grey, ansi_reset, ansi_color_cyan, ansi_reset,
           ansi_color_cyan, ansi_reset);
    printf("    %s[ND]circle[P]%s      Circle (%sP = period%s)\n\n",
           ansi_color_green, ansi_reset, ansi_color_grey, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s 1000 test_walk.txt 2Dwalk\n", ansi_color_grey, ansi_reset,
           ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
} // print_help

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    // Check for help option early
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

    if (argc < 4)
    {
        fprintf(stderr, "Error: Missing required arguments.\n");
        print_usage(argv[0]);
        print_args_on_error(argc, argv);
        return 1;
    }

    long n_points = atol(argv[1]);
    char *filename = argv[2];
    char *pattern_str = "2Drandom";

    long repeats = 1;
    double noise_radius = 0.0;
    int shuffle = 0;

    // Parse arguments
    for (int i = 3; i < argc; i++)
    {
        if (strcmp(argv[i], "-repeat") == 0)
        {
            repeats = atol(argv[++i]);
        }
        else if (strcmp(argv[i], "-noise") == 0)
        {
            noise_radius = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-shuffle") == 0)
        {
            shuffle = 1;
        }
        else if (argv[i][0] == '-')
        {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return 1;
        }
        else
        {
            pattern_str = argv[i];
        }
    }

    GeneratorConfig config;
    config.type = GEN_RANDOM;
    config.dim = 2;
    config.param = 0.0;

    // Parse dimension [N]D
    char *dim_end = strchr(pattern_str, 'D');
    if (dim_end)
    {
        *dim_end = '\0';
        config.dim = atoi(pattern_str);
        pattern_str = dim_end + 1; // Advance past 'D'
    }
    else if (strncmp(pattern_str, "2D", 2) == 0)
    {
        config.dim = 2;
        pattern_str += 2;
    }
    else if (strncmp(pattern_str, "3D", 2) == 0)
    {
        config.dim = 3;
        pattern_str += 2;
    }

    if (config.dim < 1)
    {
        config.dim = 2;
    }

    // Parse pattern name
    if (strncmp(pattern_str, "randexp", 7) == 0 || strncmp(pattern_str, "randExp", 7) == 0)
    {
        config.type = GEN_RAND_EXP;
    }
    else if (strncmp(pattern_str, "random", 6) == 0)
    {
        config.type = GEN_RANDOM;
    }
    else if (strncmp(pattern_str, "sphere", 6) == 0)
    {
        config.type = GEN_SPHERE;
    }
    else if (strncmp(pattern_str, "walk", 4) == 0)
    {
        config.type = GEN_WALK;
        char *p = pattern_str + 4;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 0.1;
        }
    }
    else if (strncmp(pattern_str, "circle", 6) == 0)
    {
        config.type = GEN_CIRCLE;
        char *p = pattern_str + 6;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = (double)n_points;
        }
    }
    else if (strncmp(pattern_str, "spiral", 6) == 0)
    {
        config.type = GEN_SPIRAL;
        char *p = pattern_str + 6;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 2.0;
        }
    }
    else if (strncmp(pattern_str, "star", 4) == 0)
    {
        config.type = GEN_STAR;
        char *p = pattern_str + 4;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 20.0;
        }
    }
    else if (strncmp(pattern_str, "concentric_dense", 16) == 0)
    {
        config.type = GEN_CONCENTRIC_DENSE;
        char *p = pattern_str + 16;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 10.0;
        }
    }
    else if (strncmp(pattern_str, "concentric", 10) == 0)
    {
        config.type = GEN_CONCENTRIC;
        char *p = pattern_str + 10;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 5.0;
        }
    }
    else if (strncmp(pattern_str, "tree", 4) == 0)
    {
        config.type = GEN_TREE;
        char *p = pattern_str + 4;
        if (*p)
        {
            config.param = atof(p);
        }
        else
        {
            config.param = 0.0;
        }
        // The tree pattern requires at least 5 dimensions
        if (config.dim < 5)
        {
            config.dim = 5;
        }
    }

    FILE *f = fopen(filename, "w");
    if (!f)
    {
        perror("Failed to open output file");
        return 1;
    }

    srand((unsigned int)time(NULL));

    long total_points = n_points * repeats;

    double *base_buffer = (double *)malloc((size_t)(n_points * config.dim) * sizeof(double));
    double *current_walk = (double *)calloc((size_t)config.dim, sizeof(double));

    for (long i = 0; i < n_points; i++)
    {
        double *pt = &base_buffer[i * config.dim];
        switch (config.type)
        {
            case GEN_WALK:
                gen_walk_point(current_walk, config.param, config.dim);
                memcpy(pt, current_walk, (size_t)config.dim * sizeof(double));
                break;
            case GEN_CIRCLE:
                gen_circle_point(pt, i, config.param, config.dim);
                break;
            case GEN_SPIRAL:
                gen_spiral_point(pt, i, n_points, config.param, config.dim);
                break;
            case GEN_SPHERE:
                gen_sphere_point(pt, config.dim);
                break;
            case GEN_STAR:
                gen_star_point(pt, i, n_points, config.param, config.dim);
                break;
            case GEN_CONCENTRIC:
                gen_concentric_point(pt, i, n_points, config.param, config.dim);
                break;
            case GEN_TREE:
                gen_tree_point(pt, i, n_points, config.param, config.dim);
                break;
            case GEN_CONCENTRIC_DENSE:
                gen_concentric_dense_point(pt, i, n_points, config.param, config.dim);
                break;
            case GEN_RAND_EXP:
                gen_randexp_point(pt, config.dim);
                break;
            case GEN_RANDOM:
            default:
                gen_random_point(pt, config.dim);
                break;
        }
    }
    free(current_walk);

    double *final_buffer = (double *)malloc((size_t)(total_points * config.dim) * sizeof(double));

    for (long r = 0; r < repeats; r++)
    {
        for (long i = 0; i < n_points; i++)
        {
            long dest_idx = r * n_points + i;
            for (int d = 0; d < config.dim; d++)
            {
                double val = base_buffer[i * config.dim + d];
                if (noise_radius > 0.0)
                {
                    val += (2.0 * rand_double() - 1.0) * noise_radius; // Uniform noise
                }
                final_buffer[dest_idx * config.dim + d] = val;
            }
        }
    }
    free(base_buffer);

    if (shuffle)
    {
        for (long i = total_points - 1; i > 0; i--)
        {
            long j = (long)(rand_double() * (double)(i + 1));
            for (int d = 0; d < config.dim; d++)
            {
                double temp = final_buffer[i * config.dim + d];
                final_buffer[i * config.dim + d] = final_buffer[j * config.dim + d];
                final_buffer[j * config.dim + d] = temp;
            }
        }
    }

    for (long i = 0; i < total_points; i++)
    {
        for (int d = 0; d < config.dim; d++)
        {
            fprintf(f, "%.6f%s", final_buffer[i * config.dim + d],
                    (d == config.dim - 1) ? "" : " ");
        }
        fprintf(f, "\n");
    }

    free(final_buffer);
    fclose(f);
    return 0;
}

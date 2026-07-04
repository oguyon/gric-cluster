/**
 * @file mktestseq.c
 * @brief Synthetic sequence coordinate generator.
 *
 * Generates structured 2D/3D coordinate walks (spiral, circle, walk, sphere, unit random)
 * to produce text test datasets.
 *
 * Main Functions:
 * - gen_random_point: Generates a uniform random coordinate.
 * - gen_sphere_point: Generates a point on the surface of a hypersphere.
 * - gen_circle_point: Generates a circular path coordinate.
 * - gen_spiral_point: Generates a spiral path coordinate.
 * - gen_walk_point: Generates a random walk step.
 * - main: Entry point of the coordinate generator.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "shared/cli_colors.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generator types
typedef enum
{
    GEN_RANDOM,
    GEN_CIRCLE,
    GEN_WALK,
    GEN_SPIRAL,
    GEN_SPHERE,
    GEN_STAR,
    GEN_CONCENTRIC,
    GEN_TREE,
    GEN_CONCENTRIC_DENSE,
    GEN_RAND_EXP
} GenType;

typedef struct
{
    GenType type;
    int dim;
    double param; // generic parameter (period, step, loops)
} GeneratorConfig;

// Random double in [0, 1]
double rand_double()
{
    return (double)rand() / (double)RAND_MAX;
}

// Generate random point
void gen_random_point(double *out, int dim)
{
    if (dim == 3)
    {
        double r = cbrt(rand_double());
        double costheta = 1.0 - 2.0 * rand_double();
        double phi = 2.0 * M_PI * rand_double();
        double sintheta = sqrt(1.0 - costheta * costheta);
        out[0] = r * sintheta * cos(phi);
        out[1] = r * sintheta * sin(phi);
        out[2] = r * costheta;
    }
    else if (dim == 2)
    {
        double r = sqrt(rand_double());
        double theta = 2.0 * M_PI * rand_double();
        out[0] = r * cos(theta);
        out[1] = r * sin(theta);
    }
    else
    {
        for (int d = 0; d < dim; d++)
        {
            out[d] = 2.0 * rand_double() - 1.0;
        }
    }
}

void gen_randexp_point(
    double *out,
    int     dim)
{
    if (dim == 3)
    {
        double sigma = 1.0 / sqrt(-2.0 * log(0.01));
        double r;
        do
        {
            double u1, u2, u3, u4;
            do
            {
                u1 = rand_double();
            } while (u1 <= 1e-15);
            u2 = rand_double();
            do
            {
                u3 = rand_double();
            } while (u3 <= 1e-15);
            u4 = rand_double();

            double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
            double z2 = sqrt(-2.0 * log(u3)) * cos(2.0 * M_PI * u4);

            out[0] = z0 * sigma;
            out[1] = z1 * sigma;
            out[2] = z2 * sigma;

            r = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        } while (r > 1.0);
    }
    else
    {
        gen_random_point(out, dim);
    }
}

// Generate random point ON unit sphere
void gen_sphere_point(double *out, int dim)
{
    if (dim == 3)
    {
        double costheta = 1.0 - 2.0 * rand_double();
        double phi = 2.0 * M_PI * rand_double();
        double sintheta = sqrt(1.0 - costheta * costheta);
        out[0] = sintheta * cos(phi);
        out[1] = sintheta * sin(phi);
        out[2] = costheta;
    }
    else if (dim == 2)
    {
        double theta = 2.0 * M_PI * rand_double();
        out[0] = cos(theta);
        out[1] = sin(theta);
    }
    else
    {
        double sum_sq = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double u1 = rand_double();
            double u2 = rand_double();
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            out[d] = z;
            sum_sq += z * z;
        }
        double norm = sqrt(sum_sq);
        for (int d = 0; d < dim; d++)
        {
            out[d] /= norm;
        }
    }
}

void gen_circle_point(double *out, long index, double period, int dim)
{
    if (period <= 0.0)
        period = 1.0;
    double theta = 2.0 * M_PI * index / period;
    out[0] = cos(theta);
    out[1] = sin(theta);
    for (int d = 2; d < dim; d++)
        out[d] = 0.0;
}

void gen_spiral_point(double *out, long index, long total_points, double loops, int dim)
{
    double t = (double)index / (double)total_points;
    if (dim == 3)
    {
        // Conical raw coordinates scaled to fit in unit bounds
        double raw_x = 0.15 * t * cos(2.0 * M_PI * loops * t);
        double raw_y = 0.15 * t * sin(2.0 * M_PI * loops * t);
        double raw_z = 2.0 * t - 1.0;

        // Rotation angles: 60 degrees around Y, 30 degrees around X
        double cos60 = 0.5000000000000000;
        double sin60 = 0.8660254037844386;
        double cos30 = 0.8660254037844386;
        double sin30 = 0.5000000000000000;

        // Y-axis rotation
        double x1 = raw_x * cos60 + raw_z * sin60;
        double z1 = -raw_x * sin60 + raw_z * cos60;
        double y1 = raw_y;

        // X-axis rotation
        out[0] = x1;
        out[1] = y1 * cos30 - z1 * sin30;
        out[2] = y1 * sin30 + z1 * cos30;

        for (int d = 3; d < dim; d++)
            out[d] = 0.0;
    }
    else
    {
        double r = t;
        double theta = 2.0 * M_PI * loops * t;
        out[0] = r * cos(theta);
        out[1] = r * sin(theta);
        for (int d = 2; d < dim; d++)
            out[d] = 0.0;
    }
}
void gen_star_point(double *out, long index, long total_points, double spokes, int dim)
{
    int num_spokes = (spokes > 0.0) ? (int)spokes : 20;
    int spoke_idx = index % num_spokes;

    if (spoke_idx == 0)
    {
        for (int d = 0; d < dim; d++)
            out[d] = 0.0;
        return;
    }

    double phi = acos(1.0 - 2.0 * (double)spoke_idx / (double)num_spokes);
    double theta = M_PI * (1.0 + sqrt(5.0)) * (double)spoke_idx;

    double ux = sin(phi) * cos(theta);
    double uy = sin(phi) * sin(theta);
    double uz = cos(phi);

    double R = 1.0 + 0.5 * (double)spoke_idx;

    out[0] = R * ux;
    out[1] = R * uy;
    if (dim >= 3)
    {
        out[2] = R * uz;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
}
void gen_concentric_point(double *out, long index, long total_points, double shells, int dim)
{
    int num_shells = (shells > 0.0) ? (int)shells : 5;
    int clusters_per_shell = 10;
    int total_clusters = num_shells * clusters_per_shell + 1;

    int cluster_idx = index % total_clusters;

    if (cluster_idx == 0)
    {
        for (int d = 0; d < dim; d++)
            out[d] = 0.0;
        return;
    }

    int shell_idx = (cluster_idx - 1) / clusters_per_shell;
    int in_shell_idx = (cluster_idx - 1) % clusters_per_shell;

    double R = 1.5 + 1.0 * (double)shell_idx;
    double theta = 2.0 * M_PI * (double)in_shell_idx / (double)clusters_per_shell;

    out[0] = R * cos(theta);
    out[1] = R * sin(theta);
    if (dim >= 3)
    {
        out[2] = 0.0;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
}
void gen_tree_point(double *out, long index, long total_points, double unused_param, int dim)
{
    if (index % 100 == 0)
    {
        int parent_idx = (index / 100) % 31;
        int L = 0;
        int level_start = 0;
        while (L < 4 && parent_idx >= level_start + (1 << L))
        {
            level_start += (1 << L);
            L++;
        }
        int path_val = parent_idx - level_start;

        for (int d = 0; d < 5; d++)
        {
            if (d < L)
            {
                int bit = (path_val >> (L - 1 - d)) & 1;
                out[d] = (bit == 0 ? -2.0 : 2.0);
            }
            else if (d == L)
            {
                out[d] = -1.0;
            }
            else
            {
                out[d] = 0.0;
            }
        }
    }
    else
    {
        int leaf_idx = index % 32;
        for (int d = 0; d < 5; d++)
        {
            int bit = (leaf_idx >> (4 - d)) & 1;
            out[d] = (bit == 0 ? -2.0 : 2.0);
        }
    }

    for (int d = 5; d < dim; d++)
    {
        out[d] = 0.0;
    }
}
void gen_concentric_dense_point(double *out, long index, long total_points, double shells, int dim)
{
    int num_shells = (shells > 0.0) ? (int)shells : 10;
    int clusters_per_shell = 30;
    int total_clusters = num_shells * clusters_per_shell + 1;

    int cluster_idx = index % total_clusters;

    if (cluster_idx == 0)
    {
        for (int d = 0; d < dim; d++)
            out[d] = 0.0;
        return;
    }

    int shell_idx = (cluster_idx - 1) / clusters_per_shell;
    int in_shell_idx = (cluster_idx - 1) % clusters_per_shell;

    double R = 1.5 + 1.0 * (double)shell_idx;
    double theta = 2.0 * M_PI * (double)in_shell_idx / (double)clusters_per_shell;

    out[0] = R * cos(theta);
    out[1] = R * sin(theta);
    if (dim >= 3)
    {
        out[2] = 0.0;
    }
    for (int d = (dim >= 3 ? 3 : 2); d < dim; d++)
    {
        out[d] = 0.0;
    }
}
void gen_walk_point(double *current, double step_size, int dim)
{
    double *next = (double *)malloc(dim * sizeof(double));
    int attempts = 0;

    while (1)
    {
        if (dim == 3)
        {
            double costheta = 1.0 - 2.0 * rand_double();
            double phi = 2.0 * M_PI * rand_double();
            double sintheta = sqrt(1.0 - costheta * costheta);
            double dx = step_size * sintheta * cos(phi);
            double dy = step_size * sintheta * sin(phi);
            double dz = step_size * costheta;
            next[0] = current[0] + dx;
            next[1] = current[1] + dy;
            next[2] = current[2] + dz;
        }
        else if (dim == 2)
        {
            double angle = 2.0 * M_PI * rand_double();
            next[0] = current[0] + step_size * cos(angle);
            next[1] = current[1] + step_size * sin(angle);
        }
        else
        {
            // Random direction in N-dim
            double sum_sq = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double u1 = rand_double();
                double u2 = rand_double();
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                next[d] = z;
                sum_sq += z * z;
            }
            double norm = sqrt(sum_sq);
            for (int d = 0; d < dim; d++)
            {
                next[d] = current[d] + (next[d] / norm) * step_size;
            }
        }

        // Check bounds (unit sphere)
        double r2 = 0.0;
        for (int d = 0; d < dim; d++)
            r2 += next[d] * next[d];

        if (r2 <= 1.0)
            break;

        attempts++;
        if (attempts > 100)
        {
            memcpy(next, current, dim * sizeof(double));
            break;
        }
    }

    memcpy(current, next, dim * sizeof(double));
    free(next);
}

void print_args_on_error(int argc, char *argv[])
{
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++)
    {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
}static void print_usage(const char *progname)
{
    fprintf(stderr, "Usage: %s <N> <output_file> <pattern> [options]\n", progname);
} // print_usage

static void print_help(const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  gric-mktxtseq - Synthetic sequence generator for testing\n\n");

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

int main(int argc, char *argv[])
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
        config.dim = 2;

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
            config.param = atof(p);
        else
            config.param = 0.1;
    }
    else if (strncmp(pattern_str, "circle", 6) == 0)
    {
        config.type = GEN_CIRCLE;
        char *p = pattern_str + 6;
        if (*p)
            config.param = atof(p);
        else
            config.param = (double)n_points;
    }
    else if (strncmp(pattern_str, "spiral", 6) == 0)
    {
        config.type = GEN_SPIRAL;
        char *p = pattern_str + 6;
        if (*p)
            config.param = atof(p);
        else
            config.param = 2.0;
    }
    else if (strncmp(pattern_str, "star", 4) == 0)
    {
        config.type = GEN_STAR;
        char *p = pattern_str + 4;
        if (*p)
            config.param = atof(p);
        else
            config.param = 20.0;
    }
    else if (strncmp(pattern_str, "concentric_dense", 16) == 0)
    {
        config.type = GEN_CONCENTRIC_DENSE;
        char *p = pattern_str + 16;
        if (*p)
            config.param = atof(p);
        else
            config.param = 10.0;
    }
    else if (strncmp(pattern_str, "concentric", 10) == 0)
    {
        config.type = GEN_CONCENTRIC;
        char *p = pattern_str + 10;
        if (*p)
            config.param = atof(p);
        else
            config.param = 5.0;
    }
    else if (strncmp(pattern_str, "tree", 4) == 0)
    {
        config.type = GEN_TREE;
        char *p = pattern_str + 4;
        if (*p)
            config.param = atof(p);
        else
            config.param = 0.0;
        // The tree pattern requires at least 5 dimensions!
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

    srand(time(NULL));

    long total_points = n_points * repeats;

    double *base_buffer = (double *)malloc(n_points * config.dim * sizeof(double));
    double *current_walk = (double *)calloc(config.dim, sizeof(double));

    for (long i = 0; i < n_points; i++)
    {
        double *pt = &base_buffer[i * config.dim];
        switch (config.type)
        {
        case GEN_WALK:
            gen_walk_point(current_walk, config.param, config.dim);
            memcpy(pt, current_walk, config.dim * sizeof(double));
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

    double *final_buffer = (double *)malloc(total_points * config.dim * sizeof(double));

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
            long j = (long)(rand_double() * (i + 1));
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

/**
 * @file model_nd.c
 * @brief N-Dimensional coordinate reconstruction utility.
 *
 * Implements a simulated annealing optimizer to reconstruct N-dimensional coordinate
 * matrices from pairwise distance matrices (dcc.txt).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "shared/cli_colors.h"

#define MAX_CLUSTERS 2000

/**
 * struct PointND - An N-dimensional coordinate point.
 * @coords: Array of dimension coordinates [dim].
 * @dim:    Number of dimensions.
 */
typedef struct {
    double *coords;
    int     dim;
} PointND;

/**
 * dist_nd() - Euclidean distance in N-dimensional space between two points.
 * @p1: First point.
 * @p2: Second point.
 *
 * Return: Euclidean distance.
 */
static double dist_nd(
    PointND p1,
    PointND p2)
{
    double sum = 0.0;
    for (int k = 0; k < p1.dim; k++)
    {
        double d = p1.coords[k] - p2.coords[k];
        sum += d * d;
    }
    return sqrt(sum);
}

/**
 * rand_double() - Generate a pseudo-random double in [0, 1).
 *
 * Return: Random floating-point value.
 */
static double rand_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

/**
 * print_args_on_error() - Print argument list to stderr for diagnostics.
 * @argc: Number of arguments.
 * @argv: Argument array.
 */
static void print_args_on_error(
    int   argc,
    char *argv[])
{
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++)
    {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
}

/**
 * print_usage() - Print short command synopsis to stderr.
 * @progname: Executable name.
 */
static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s <dcc_file> <dimensions> <output_file> [options]\n", progname);
}

/**
 * print_help_raw() - Output full formatted manual screen to stdout.
 * @progname: Executable name.
 */
static void print_help_raw(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-NDmodel%s - N-Dimensional space reconstruction from distance matrix\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s<dcc_file>%s %s<dimensions>%s %s<output_file>%s %s[options]%s\n\n",
           ansi_bold_green, progname, ansi_reset, ansi_color_magenta, ansi_reset,
           ansi_color_magenta, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_grey,
           ansi_reset);

    printf("%sDESCRIPTION%s\n", ansi_bold_cyan, ansi_reset);
    printf("  Reconstructs N-dimensional coordinates from a cluster distance matrix\n");
    printf("  (dcc.txt) using Simulated Annealing optimization.\n\n");

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-temp%s %s<val>%s          Initial temperature (%sdefault:%s%s 10.0%s)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_cyan,
           ansi_reset, ansi_color_cyan, ansi_reset);
    printf("  %s-rate%s %s<val>%s          Cooling rate (%sdefault:%s%s 0.995%s)\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_cyan,
           ansi_reset, ansi_color_cyan, ansi_reset);
    printf("  %s-iter%s %s<val>%s          Number of iterations (%sdefault:%s%s 100000%s)\n\n",
           ansi_color_green, ansi_reset, ansi_color_magenta, ansi_reset, ansi_color_cyan,
           ansi_reset, ansi_color_cyan, ansi_reset);
    printf("  Arguments:\n");
    printf("    %s<dcc_file>%s         Input distance matrix file (dcc.txt)\n",
           ansi_color_magenta, ansi_reset);
    printf("    %s<dimensions>%s       Target dimensionality (N)\n",
           ansi_color_magenta, ansi_reset);
    printf("    %s<output_file>%s      Output filename for coordinates\n\n",
           ansi_color_magenta, ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s dcc.txt 3 coordinates.txt\n",
           ansi_color_grey, ansi_reset, ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
}

/**
 * print_help() - Paged help wrapper.
 * @progname: Executable name.
 */
static void print_help(
    const char *progname)
{
    FILE *tmp = tmpfile();
    if (tmp != NULL)
    {
        int saved_stdout = dup(STDOUT_FILENO);
        int tmp_fd = fileno(tmp);
        dup2(tmp_fd, STDOUT_FILENO);

        print_help_raw(progname);
        fflush(stdout);

        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);

        fseek(tmp, 0, SEEK_END);
        long sz = ftell(tmp);
        fseek(tmp, 0, SEEK_SET);

        char *buf = malloc((size_t)sz + 1);
        if (buf != NULL)
        {
            size_t read_bytes = fread(buf, 1, (size_t)sz, tmp);
            buf[read_bytes] = '\0';
            cli_print_pager(buf);
            free(buf);
        }
        fclose(tmp);
    }
    else
    {
        print_help_raw(progname);
    }
}

/**
 * load_distance_matrix() - Parse dcc.txt distance matrix.
 * @input_file:       Path to dcc file.
 * @out_num_clusters: Output cluster count.
 * @out_matrix:       Output allocated pairwise distance array.
 *
 * Return: 0 on success, -1 on error.
 */
static int load_distance_matrix(
    const char  *input_file,
    int         *out_num_clusters,
    double     **out_matrix)
{
    FILE *fin = fopen(input_file, "r");
    if (!fin)
    {
        perror("Error opening dcc file");
        return -1;
    }

    int max_id = -1;
    char line[1024];
    while (fgets(line, sizeof(line), fin))
    {
        int i, j;
        double d;
        if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3)
        {
            if (i > max_id)
            {
                max_id = i;
            }
            if (j > max_id)
            {
                max_id = j;
            }
        }
    }

    int num_clusters = max_id + 1;
    if (num_clusters <= 0)
    {
        fprintf(stderr, "No valid data in dcc file\n");
        fclose(fin);
        return -1;
    }
    if (num_clusters > MAX_CLUSTERS)
    {
        fprintf(stderr, "Too many clusters (%d), max is %d\n", num_clusters, MAX_CLUSTERS);
        fclose(fin);
        return -1;
    }

    double *D = (double *)malloc((size_t)(num_clusters * num_clusters) * sizeof(double));
    if (D == NULL)
    {
        fclose(fin);
        return -1;
    }

    for (int i = 0; i < num_clusters * num_clusters; i++)
    {
        D[i] = -1.0;
    }

    rewind(fin);
    while (fgets(line, sizeof(line), fin))
    {
        int i, j;
        double d;
        if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3)
        {
            D[i * num_clusters + j] = d;
            D[j * num_clusters + i] = d;
        }
    }
    fclose(fin);

    *out_num_clusters = num_clusters;
    *out_matrix = D;
    return 0;
}

/**
 * init_random_points() - Allocate and randomly initialize N-dimensional points.
 * @num_clusters: Number of points to allocate.
 * @dimensions:   Dimensionality.
 *
 * Return: Allocated PointND array or NULL.
 */
static PointND *init_random_points(
    int num_clusters,
    int dimensions)
{
    PointND *P = (PointND *)malloc((size_t)num_clusters * sizeof(PointND));
    if (P == NULL)
    {
        return NULL;
    }

    srand((unsigned int)time(NULL));
    for (int i = 0; i < num_clusters; i++)
    {
        P[i].dim = dimensions;
        P[i].coords = (double *)malloc((size_t)dimensions * sizeof(double));
        if (P[i].coords != NULL)
        {
            for (int k = 0; k < dimensions; k++)
            {
                P[i].coords[k] = (rand_double() - 0.5) * 20.0;
            }
        }
    }

    return P;
}

/**
 * free_points() - Deallocate PointND array and coordinate buffers.
 * @P:            Points array.
 * @num_clusters: Point count.
 */
static void free_points(
    PointND *P,
    int      num_clusters)
{
    if (P == NULL)
    {
        return;
    }

    for (int i = 0; i < num_clusters; i++)
    {
        if (P[i].coords != NULL)
        {
            free(P[i].coords);
            P[i].coords = NULL;
        }
    }
    free(P);
}

/**
 * run_simulated_annealing() - Execute simulated annealing to align points to distance matrix.
 * @P:            Points array to optimize in-place.
 * @D:            Pairwise target distance matrix.
 * @num_clusters: Number of points.
 * @dimensions:   Dimensionality.
 * @initial_T:    Initial temperature.
 * @cooling_rate: Multiplicative cooling factor.
 * @iterations:   Max iteration count.
 */
static void run_simulated_annealing(
    PointND      *P,
    const double *D,
    int           num_clusters,
    int           dimensions,
    double        initial_T,
    double        cooling_rate,
    int           iterations)
{
    double E = 0.0;
    int pair_count = 0;
    for (int i = 0; i < num_clusters; i++)
    {
        for (int j = i + 1; j < num_clusters; j++)
        {
            double target = D[i * num_clusters + j];
            if (target >= 0.0)
            {
                double curr = dist_nd(P[i], P[j]);
                E += (curr - target) * (curr - target);
                pair_count++;
            }
        }
    }

    if (pair_count == 0)
    {
        fprintf(stderr, "No pairs to optimize\n");
        return;
    }

    printf("Initial Energy: %.6f\n", E);

    PointND new_p;
    new_p.dim = dimensions;
    new_p.coords = (double *)malloc((size_t)dimensions * sizeof(double));
    if (new_p.coords == NULL)
    {
        return;
    }

    double T = initial_T;
    for (int k = 0; k < iterations; k++)
    {
        int idx = rand() % num_clusters;
        memcpy(new_p.coords, P[idx].coords, (size_t)dimensions * sizeof(double));

        for (int d = 0; d < dimensions; d++)
        {
            new_p.coords[d] += (rand_double() - 0.5) * T;
        }

        double dE = 0.0;
        for (int j = 0; j < num_clusters; j++)
        {
            if (idx == j)
            {
                continue;
            }
            double target = D[idx * num_clusters + j];
            if (target >= 0.0)
            {
                double old_dist = dist_nd(P[idx], P[j]);
                double new_dist = dist_nd(new_p, P[j]);
                dE += (new_dist - target) * (new_dist - target) -
                      (old_dist - target) * (old_dist - target);
            }
        }

        if (dE < 0.0 || exp(-dE / T) > rand_double())
        {
            memcpy(P[idx].coords, new_p.coords, (size_t)dimensions * sizeof(double));
            E += dE;
        }

        T *= cooling_rate;
        if (T < 1e-5)
        {
            break;
        }
    }

    printf("Final Energy: %.6f\n", E);
    free(new_p.coords);
}

/**
 * write_coordinates_file() - Export reconstructed coordinates to text file.
 * @output_file:  Output file path.
 * @P:            Points array.
 * @num_clusters: Point count.
 * @dimensions:   Dimensionality.
 */
static void write_coordinates_file(
    const char    *output_file,
    const PointND *P,
    int            num_clusters,
    int            dimensions)
{
    FILE *fout = fopen(output_file, "w");
    if (!fout)
    {
        perror("Error opening output file");
        return;
    }

    fprintf(fout, "# ID");
    for (int d = 0; d < dimensions; d++)
    {
        fprintf(fout, " Dim%d", d);
    }
    fprintf(fout, "\n");

    for (int i = 0; i < num_clusters; i++)
    {
        fprintf(fout, "%d", i);
        for (int d = 0; d < dimensions; d++)
        {
            fprintf(fout, " %.6f", P[i].coords[d]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    printf("Saved ND model to %s\n", output_file);
}

/**
 * main() - Entry point of the NDmodel utility.
 * @argc: Command argument count.
 * @argv: Command argument array.
 *
 * Return: 0 on success, 1 on error.
 */
int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

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

    char *input_file = argv[1];
    int dimensions = atoi(argv[2]);
    char *output_file = argv[3];

    double T = 10.0;
    double cooling_rate = 0.995;
    int iterations = 100000;

    for (int i = 4; i < argc; i++)
    {
        if (strcmp(argv[i], "-temp") == 0)
        {
            if (i + 1 < argc)
            {
                T = atof(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-rate") == 0)
        {
            if (i + 1 < argc)
            {
                cooling_rate = atof(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-iter") == 0)
        {
            if (i + 1 < argc)
            {
                iterations = atoi(argv[++i]);
            }
        }
    }

    if (dimensions < 1)
    {
        fprintf(stderr, "Invalid dimensions: %d\n", dimensions);
        print_args_on_error(argc, argv);
        return 1;
    }

    int num_clusters = 0;
    double *D = NULL;
    if (load_distance_matrix(input_file, &num_clusters, &D) != 0)
    {
        print_args_on_error(argc, argv);
        return 1;
    }

    PointND *P = init_random_points(num_clusters, dimensions);
    if (P == NULL)
    {
        free(D);
        print_args_on_error(argc, argv);
        return 1;
    }

    run_simulated_annealing(P, D, num_clusters, dimensions, T, cooling_rate, iterations);
    write_coordinates_file(output_file, P, num_clusters, dimensions);

    free_points(P, num_clusters);
    free(D);
    return 0;
}

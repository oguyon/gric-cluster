/**
 * @file model_nd.c
 * @brief N-Dimensional coordinate reconstruction utility.
 *
 * Implements a simulated annealing optimizer to reconstruct N-dimensional coordinate
 * matrices from pairwise distance matrices (dcc.txt).
 *
 * Main Functions:
 * - dist_nd: Computes distance in N-dimensional space between two points.
 * - rand_double: Helper to generate a random double-precision floating-point number.
 * - main: Entry point of the reconstruction utility.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "shared/cli_colors.h"
#define MAX_CLUSTERS 2000

typedef struct
{
    double *coords;
    int dim;
} PointND;

double dist_nd(PointND p1, PointND p2)
{
    double sum = 0.0;
    for (int k = 0; k < p1.dim; k++)
    {
        double d = p1.coords[k] - p2.coords[k];
        sum += d * d;
    }
    return sqrt(sum);
}

double rand_double()
{
    return (double)rand() / (double)RAND_MAX;
}

void print_args_on_error(int argc, char *argv[])
{
    fprintf(stderr, "\nProgram arguments:\n");
    for (int i = 0; i < argc; i++)
    {
        fprintf(stderr, "  argv[%d] = \"%s\"\n", i, argv[i]);
    }
    fprintf(stderr, "\n");
}



static void print_usage(const char *progname)
{
    fprintf(stderr, "Usage: %s <dcc_file> <dimensions> <output_file> [options]\n", progname);
} // print_usage



static void print_help(const char *progname)
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
    printf("    %s<dcc_file>%s         Input distance matrix file (dcc.txt)\n", ansi_color_magenta,
           ansi_reset);
    printf("    %s<dimensions>%s       Target dimensionality (N)\n", ansi_color_magenta, ansi_reset);
    printf("    %s<output_file>%s      Output filename for coordinates\n\n", ansi_color_magenta,
           ansi_reset);

    printf("%sEXAMPLES%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s$%s %s%s%s dcc.txt 3 coordinates.txt\n", ansi_color_grey, ansi_reset,
           ansi_bold_green, progname, ansi_reset);
    cli_print_color_mode();
} // print_help

int main(int argc, char *argv[])
{
    cli_colors_init();

    // Check help option early
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

    // Defaults
    double T = 10.0;
    double cooling_rate = 0.995;
    int iterations = 100000;

    // Parse options
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

    // Load distance matrix
    FILE *fin = fopen(input_file, "r");
    if (!fin)
    {
        perror("Error opening dcc file");
        print_args_on_error(argc, argv);
        return 1;
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
                max_id = i;
            if (j > max_id)
                max_id = j;
        }
    }

    int num_clusters = max_id + 1;
    if (num_clusters <= 0)
    {
        fprintf(stderr, "No valid data in dcc file\n");
        fclose(fin);
        print_args_on_error(argc, argv);
        return 1;
    }
    if (num_clusters > MAX_CLUSTERS)
    {
        fprintf(stderr, "Too many clusters (%d), max is %d\n", num_clusters, MAX_CLUSTERS);
        fclose(fin);
        print_args_on_error(argc, argv);
        return 1;
    }

    // Allocate matrix
    double *D = (double *)malloc(num_clusters * num_clusters * sizeof(double));
    for (int i = 0; i < num_clusters * num_clusters; i++)
        D[i] = -1.0;

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

    // Initialize random positions
    PointND *P = (PointND *)malloc(num_clusters * sizeof(PointND));
    srand(time(NULL));
    for (int i = 0; i < num_clusters; i++)
    {
        P[i].dim = dimensions;
        P[i].coords = (double *)malloc(dimensions * sizeof(double));
        for (int k = 0; k < dimensions; k++)
        {
            P[i].coords[k] = (rand_double() - 0.5) * 20.0;
        }
    }

    // Simulated Annealing
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
                E += pow(curr - target, 2);
                pair_count++;
            }
        }
    }

    if (pair_count == 0)
    {
        fprintf(stderr, "No pairs to optimize\n");
        // Cleanup
        free(D);
        for (int i = 0; i < num_clusters; i++)
            free(P[i].coords);
        free(P);
        print_args_on_error(argc, argv);
        return 0;
    }

    printf("Initial Energy: %.6f\n", E);

    // Temporary point for perturbations
    PointND new_p;
    new_p.dim = dimensions;
    new_p.coords = (double *)malloc(dimensions * sizeof(double));

    for (int k = 0; k < iterations; k++)
    {
        int idx = rand() % num_clusters;

        // Copy current to new
        memcpy(new_p.coords, P[idx].coords, dimensions * sizeof(double));

        // Perturb
        for (int d = 0; d < dimensions; d++)
        {
            new_p.coords[d] += (rand_double() - 0.5) * T;
        }

        // Calculate delta E
        double dE = 0.0;
        for (int j = 0; j < num_clusters; j++)
        {
            if (idx == j)
                continue;
            double target = D[idx * num_clusters + j];
            if (target >= 0.0)
            {
                double old_dist = dist_nd(P[idx], P[j]);
                double new_dist = dist_nd(new_p, P[j]);
                dE += pow(new_dist - target, 2) - pow(old_dist - target, 2);
            }
        }

        // Accept?
        if (dE < 0 || exp(-dE / T) > rand_double())
        {
            memcpy(P[idx].coords, new_p.coords, dimensions * sizeof(double));
            E += dE;
        }

        T *= cooling_rate;
        if (T < 1e-5)
            break;
    }

    printf("Final Energy: %.6f\n", E);

    FILE *fout = fopen(output_file, "w");
    if (!fout)
    {
        perror("Error opening output file");
    }
    else
    {
        fprintf(fout, "# ID");
        for (int d = 0; d < dimensions; d++)
            fprintf(fout, " Dim%d", d);
        fprintf(fout, "\n");

        for (int i = 0; i < num_clusters; i++)
        {
            fprintf(fout, "%d", i);
            for (int d = 0; d < dimensions; d++)
                fprintf(fout, " %.6f", P[i].coords[d]);
            fprintf(fout, "\n");
        }
        fclose(fout);
        printf("Saved ND model to %s\n", output_file);
    }

    free(new_p.coords);
    free(D);
    for (int i = 0; i < num_clusters; i++)
        free(P[i].coords);
    free(P);

    return 0;
}

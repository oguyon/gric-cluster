/**
 * @file gric-tune.c
 * @brief Auto-tuning and parameter calibration utility for GRIC.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#define ANSI_BOLD_CYAN    "\x1b[1;36m"
#define ANSI_BOLD_GREEN   "\x1b[1;32m"
#define ANSI_BOLD_YELLOW  "\x1b[1;33m"
#define ANSI_BOLD_RED     "\x1b[1;31m"
#define ANSI_COLOR_RESET  "\x1b[0m"

typedef struct
{
    char   name[32];
    double rlim;
    double wall_time;
    int    clusters;
    double rms;
    double max_dist;
    double entropy;
} TuneResult;

/**
 * run_profiler() - Run gric-cluster with specific settings and extract metrics.
 */
static int run_profiler(
    const char *bin_path,
    const char *input_file,
    const char *tiles,
    double      rlim,
    int         nsamples,
    int         maxcl,
    const char *extra_flags,
    TuneResult *res)
{
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s %.4f %s -maxim %d -ncpu 4 -maxcl %d -tiles %s -no_dcc %s 2>&1",
             bin_path, rlim, input_file, nsamples, maxcl, tiles, extra_flags);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    FILE *fp = popen(cmd, "r");
    if (!fp)
    {
        return -1;
    }

    strcpy(res->name, tiles);
    res->rlim = rlim;
    res->clusters = 0;
    res->rms = 0.0;
    res->max_dist = 0.0;
    res->entropy = 0.0;

    char line[1024];
    while (fgets(line, sizeof(line), fp))
    {
        if (strcmp(tiles, "1x1") == 0)
        {
            if (strstr(line, "Clusters:"))
            {
                sscanf(line, " Clusters: %d", &res->clusters);
            }
            else if (strstr(line, "RMS Distance:"))
            {
                double avg_rms = 0.0;
                sscanf(line, " RMS Distance: %lf (Avg cluster RMS: %lf, Max: %lf)",
                       &res->rms, &avg_rms, &res->max_dist);
            }
            else if (strstr(line, "Cluster Entropy:"))
            {
                sscanf(line, " Cluster Entropy: %lf bits", &res->entropy);
            }
        }
        else
        {
            if (strstr(line, "Unique Tuples (states):"))
            {
                sscanf(line, " Unique Tuples (states): %d", &res->clusters);
            }
            else if (strstr(line, "Global Joint RMS Dist:"))
            {
                double avg_tuple_rms = 0.0;
                sscanf(line, " Global Joint RMS Dist: %lf (Avg tuple RMS: %lf, Max: %lf)",
                       &res->rms, &avg_tuple_rms, &res->max_dist);
            }
            else if (strstr(line, "Joint Tuple Entropy:"))
            {
                sscanf(line, " Joint Tuple Entropy: %lf bits", &res->entropy);
            }
        }
    }
    pclose(fp);
    clock_gettime(CLOCK_MONOTONIC, &end);

    res->wall_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;
    return 0;
}

int main(
    int   argc,
    char *argv[])
{
    int nsamples = 2000;
    int maxcl = 2000;
    char *input_file = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            nsamples = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
        {
            maxcl = atoi(argv[++i]);
        }
        else if (argv[i][0] != '-')
        {
            input_file = argv[i];
        }
    }

    if (!input_file)
    {
        printf("Usage: %s <input_file> [-n <nsamples>] [-k <maxcl>]\n", argv[0]);
        return 1;
    }

    char bin_path[256] = "build/gric-cluster";
    struct stat st;
    if (stat(bin_path, &st) != 0)
    {
        strcpy(bin_path, "./gric-cluster");
        if (stat(bin_path, &st) != 0)
        {
            strcpy(bin_path, "gric-cluster");
        }
    }

    printf(ANSI_BOLD_CYAN "Starting Parameter Calibration Utility (gric-tune)...\n"
           ANSI_COLOR_RESET);
    printf("Target binary: %s\n", bin_path);
    printf("Input file:    %s\n", input_file);
    printf("Samples:       %d frames\n\n", nsamples);

    printf("Step 1: Running distance statistics scan...\n");
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s -scandist %s -maxim %d 2>&1", bin_path, input_file, nsamples);
    FILE *fp = popen(cmd, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Failed to run command: %s\n", cmd);
        return 1;
    }
    double median_dist = -1.0;
    char line[1024];
    while (fgets(line, sizeof(line), fp))
    {
        if (strstr(line, "Median:"))
        {
            sscanf(line, "Median: %lf", &median_dist);
        }
    }
    pclose(fp);

    if (median_dist < 0.0)
    {
        fprintf(stderr, "Error: Failed to extract median distance from scan.\n");
        return 1;
    }
    printf("  Found Median Distance: %.4f\n\n", median_dist);

    /* Sparse signals (moving objects) do not scale down with tile area.
     * We scale rlim slightly (0.92 for 2x2, 0.75 for 3x3) to account
     * for reduced background noise, rather than strictly by area ratio. */
    double rlim_1x1 = 1.5 * median_dist;
    double rlim_2x2 = 0.92 * rlim_1x1;
    double rlim_3x3 = 0.75 * rlim_1x1;

    TuneResult res_1x1, res_2x2, res_3x3;
    TuneResult res_2x2_h1k, res_2x2_h10k;

    printf("Step 2: Profiling 1x1 Baseline (No Tiling) at rlim = %.4f...\n", rlim_1x1);
    if (run_profiler(bin_path, input_file, "1x1", rlim_1x1, nsamples, maxcl, "", &res_1x1) != 0)
    {
        fprintf(stderr, "Error profiling 1x1 configuration.\n");
        return 1;
    }

    printf("Step 3: Profiling 2x2 Tiling at rlim = %.4f...\n", rlim_2x2);
    if (run_profiler(bin_path, input_file, "2x2", rlim_2x2, nsamples, maxcl, "", &res_2x2) != 0)
    {
        fprintf(stderr, "Error profiling 2x2 configuration.\n");
        return 1;
    }

    printf("Step 4: Profiling 3x3 Tiling at rlim = %.4f...\n", rlim_3x3);
    if (run_profiler(bin_path, input_file, "3x3", rlim_3x3, nsamples, maxcl, "", &res_3x3) != 0)
    {
        fprintf(stderr, "Error profiling 3x3 configuration.\n");
        return 1;
    }

    printf("Step 5: Profiling Trajectory Fusion Lookback (2x2, H=1000)...\n");
    if (run_profiler(bin_path, input_file, "2x2", rlim_2x2, nsamples, maxcl,
                     "\"-pred[2,1000,2]\"", &res_2x2_h1k) != 0)
    {
        fprintf(stderr, "Error profiling H=1000 configuration.\n");
        return 1;
    }

    printf("Step 6: Profiling Trajectory Fusion Lookback (2x2, H=10000)...\n");
    if (run_profiler(bin_path, input_file, "2x2", rlim_2x2, nsamples, maxcl,
                     "\"-pred[2,10000,2]\"", &res_2x2_h10k) != 0)
    {
        fprintf(stderr, "Error profiling H=10000 configuration.\n");
        return 1;
    }

    printf("\n" ANSI_BOLD_CYAN "=== CALIBRATION SUMMARY REPORT ===" ANSI_COLOR_RESET "\n");
    printf("| Grid Size | Horizon (H) | rlim | Wall Time | Unique States/Tuples | Global RMS | Max Dist | Entropy |\n");
    printf("|:---|:---|---:|---:|---:|---:|---:|---:|\n");
    printf("| 1x1 Baseline | H=0  | %.4f | %.1f ms | %4d clusters | %.4f | %.4f | %.4f bits |\n",
           res_1x1.rlim, res_1x1.wall_time, res_1x1.clusters, res_1x1.rms, res_1x1.max_dist, res_1x1.entropy);
    printf("| 2x2 Tiled    | H=0  | %.4f | %.1f ms | %4d tuples   | %.4f | %.4f | %.4f bits |\n",
           res_2x2.rlim, res_2x2.wall_time, res_2x2.clusters, res_2x2.rms, res_2x2.max_dist, res_2x2.entropy);
    printf("| 2x2 Tiled    | H=1k | %.4f | %.1f ms | %4d tuples   | %.4f | %.4f | %.4f bits |\n",
           res_2x2_h1k.rlim, res_2x2_h1k.wall_time, res_2x2_h1k.clusters, res_2x2_h1k.rms, res_2x2_h1k.max_dist, res_2x2_h1k.entropy);
    printf("| 2x2 Tiled    | H=10k| %.4f | %.1f ms | %4d tuples   | %.4f | %.4f | %.4f bits |\n",
           res_2x2_h10k.rlim, res_2x2_h10k.wall_time, res_2x2_h10k.clusters, res_2x2_h10k.rms, res_2x2_h10k.max_dist, res_2x2_h10k.entropy);
    printf("| 3x3 Tiled    | H=0  | %.4f | %.1f ms | %4d tuples   | %.4f | %.4f | %.4f bits |\n",
           res_3x3.rlim, res_3x3.wall_time, res_3x3.clusters, res_3x3.rms, res_3x3.max_dist, res_3x3.entropy);

    printf("\n" ANSI_BOLD_YELLOW "=== DIAGNOSTICS & RECOMMENDATIONS ===" ANSI_COLOR_RESET "\n");

    if (res_1x1.clusters >= maxcl)
    {
        printf(ANSI_BOLD_YELLOW "  [WARNING] Monolithic baseline maxed out the maximum cluster capacity (%d clusters)!\n" ANSI_COLOR_RESET
               "            This truncates new cluster creation and degrades quality/RMS accuracy.\n"
               "            * RECOMMENDATION: Increase maximum cluster limit using -k flag (e.g. -k %d).\n\n",
               maxcl, maxcl * 2);
    }

    if (res_2x2.clusters > nsamples * 0.15)
    {
        printf(ANSI_BOLD_RED "  [ALERT] Combinatorial State Explosion detected on 2x2 grid!\n" ANSI_COLOR_RESET
               "          The 2x2 grid created %d unique states for %d frames.\n"
               "          Tiling is too fine for the spatial correlation of this data.\n"
               "          * RECOMMENDATION: Use No Tiling (1x1).\n",
               res_2x2.clusters, nsamples);
        printf("\nRecommended command for full run:\n"
               "  gric-cluster %.4f %s -maxcl %d\n",
               rlim_1x1, input_file, maxcl);
    }
    else
    {
        int comp_1k = res_2x2.clusters - res_2x2_h1k.clusters;
        int comp_10k = res_2x2.clusters - res_2x2_h10k.clusters;

        printf("  [INFO] Tiling is stable and yields optimal state compression.\n");
        if (comp_10k > res_2x2.clusters * 0.05 && res_2x2_h10k.wall_time < 3.0 * res_2x2_h1k.wall_time)
        {
            printf("         Deep Trajectory Fusion (H=10,000) yields significant state-space\n"
                   "         compression gains (%d tuples eliminated).\n"
                   "         * RECOMMENDATION: Use 2x2 Tiling with Deep Lookback (H=10,000).\n",
                   comp_10k);
            printf("\nRecommended command for full run:\n"
                   "  gric-cluster %.4f %s -tiles 2x2 -maxcl %d -retrieval_window 10000 \"-pred[2,10000,2]\"\n",
                   rlim_2x2, input_file, maxcl);
        }
        else if (comp_1k > res_2x2.clusters * 0.02)
        {
            printf("         Moderate Trajectory Fusion (H=1,000) balances lookback memory\n"
                   "         compression with fast execution speed.\n"
                   "         * RECOMMENDATION: Use 2x2 Tiling with Moderate Lookback (H=1,000, %d tuples eliminated).\n",
                   comp_1k);
            printf("\nRecommended command for full run:\n"
                   "  gric-cluster %.4f %s -tiles 2x2 -maxcl %d -retrieval_window 1000 \"-pred[2,1000,2]\"\n",
                   rlim_2x2, input_file, maxcl);
        }
        else
        {
            printf("         Trajectory fusion lookback yields minimal compression gains.\n"
                   "         * RECOMMENDATION: Use 2x2 Tiling (H=0 Baseline) for maximum speed.\n");
            printf("\nRecommended command for full run:\n"
                   "  gric-cluster %.4f %s -tiles 2x2 -maxcl %d\n",
                   rlim_2x2, input_file, maxcl);
        }
    }

    return 0;
}

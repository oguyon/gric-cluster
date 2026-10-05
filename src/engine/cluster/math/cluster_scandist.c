/**
 * @file cluster_scandist.c
 * @brief Implementing distance-scanning logic.
 *
 * Scans sequential/pairs of frames and computes distance statistics
 * (Min, Median, Max, percentiles) for threshold selection.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_scandist.h"
#include "framedistance.h"
#include "frameread.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ANSI_COLOR_RESET "\x1b[0m"

static int compare_doubles(
    const void *a,
    const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

void run_scandist(
    ClusterConfig *config,
    char          *out_dir)
{
    long nframes = get_num_frames();

    if (nframes < 2)
    {
        printf("Not enough frames to calculate distances.\n");
        return;
    }

    long process_limit = (nframes > config->input.maxnbfr) ? config->input.maxnbfr : nframes;
    double *distances = (double *)malloc((process_limit - 1) * sizeof(double));

    if (!distances)
    {
        perror("Memory allocation failed");
        return;
    }

    Frame *prev = getframe();

    if (!prev)
    {
        free(distances);
        return;
    }

    FILE *scan_out = NULL;
    char  scan_path[1024];

    if (out_dir)
    {
        snprintf(scan_path, sizeof(scan_path), "%s/dist-scan.txt", out_dir);
        scan_out = fopen(scan_path, "w");
        if (scan_out)
        {
            fprintf(scan_out, "# Frame1 Frame2 Distance\n");
        }
    }

    printf("Scanning distances\n");

    long count = 0;

    for (long i = 1; i < process_limit; i++)
    {
        Frame *curr = getframe();

        if (!curr)
        {
            break;
        }

        double d = framedist(prev, curr);
        distances[count++] = d;

        if (scan_out)
        {
            fprintf(scan_out, "%d %d %.6f\n", prev->id, curr->id, d);
        }

        if (config->output.progress_mode && (i % 10 == 0 || i == process_limit - 1))
        {
            printf("\rScanning frame %ld / %ld", i, process_limit);
            fflush(stdout);
        }

        free_frame(prev);
        prev = curr;
    }
    free_frame(prev);

    if (scan_out)
    {
        fclose(scan_out);
    }

    if (config->output.progress_mode)
    {
        printf("\n");
    }

    if (count > 0)
    {
        qsort(distances, count, sizeof(double), compare_doubles);

        double min_val = distances[0];
        double max_val = distances[count - 1];
        double median_val;
        double p20_val;
        double p80_val;

        if (count % 2 == 1)
        {
            median_val = distances[count / 2];
        }
        else
        {
            median_val = (distances[count / 2 - 1] + distances[count / 2]) / 2.0;
        }

        double p20_idx = (count - 1) * 0.2;
        int    p20_i = (int)p20_idx;
        double p20_f = p20_idx - p20_i;

        if (p20_i + 1 < count)
        {
            p20_val = distances[p20_i] * (1.0 - p20_f) + distances[p20_i + 1] * p20_f;
        }
        else
        {
            p20_val = distances[p20_i];
        }

        double p80_idx = (count - 1) * 0.8;
        int    p80_i = (int)p80_idx;
        double p80_f = p80_idx - p80_i;

        if (p80_i + 1 < count)
        {
            p80_val = distances[p80_i] * (1.0 - p80_f) + distances[p80_i + 1] * p80_f;
        }
        else
        {
            p80_val = distances[p80_i];
        }

        if (config->input.scandist_mode)
        {
            printf("Distance statistics (%ld intervals):\n", count);
            printf("%-10s %.6f\n", "Min:", min_val);
            printf("%-10s %.6f\n", "20%:", p20_val);
            printf("%-10s %.6f\n", "Median:", median_val);
            printf("%-10s %.6f\n", "80%:", p80_val);
            printf("%-10s %.6f\n", "Max:", max_val);
        }
        else if (config->algo.auto_rlim_mode)
        {
            config->algo.rlim = config->algo.auto_rlim_factor * median_val;
            printf("Auto-rlim: Median distance = %.6f, Multiplier = %.6f -> rlim = %.6f\n",
                   median_val, config->algo.auto_rlim_factor, config->algo.rlim);
        }
    }
    else
    {
        printf("No distances calculated.\n");
    }

    free(distances);
}

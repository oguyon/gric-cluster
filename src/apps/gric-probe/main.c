/**
 * @file main.c
 * @brief Main entrypoint for gric-probe dataset characterization utility.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#include "probe_engine.h"
#include "probe_report.h"
#include "gric_profile.h"
#include "cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(
    const char *prog_name)
{
    printf("Usage: %s <dataset_path> [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  -o <path>       Output path for .gricprof file (default: <dataset>.gricprof)\n");
    printf("  -n <count>      Number of frames to sample (default: adaptive, max 5,000)\n");
    printf("  -all            Scan 100%% of dataset without subsampling\n");
    printf("  -warmup <count> Warmup frames for shared memory streaming\n");
    printf("  -double         Force double-precision (float64) processing\n");
    printf("  -q, --quiet     Quiet mode (suppress terminal dashboard)\n");
    printf("  -progress       Emit real-time progress tags [PROBE: XX%%] to stderr\n");
    printf("  -json           Output full profile JSON directly to stdout\n");
    printf("  --env           Output shell environment export commands\n");
    printf("  -h, --help      Display this help message\n\n");
    printf("Description:\n");
    printf("  Performs a single-pass geometric, statistical, and quantization scan\n");
    printf("  over raw dataset frames to determine optimal clustering radius presets\n");
    printf("  (fine, balanced, coarse), spectral variance dimension ordering, and\n");
    printf("  acceleration flags for gric-cluster and gric-knn.\n");
}

int main(
    int   argc,
    char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    ProbeConfig config;
    memset(&config, 0, sizeof(ProbeConfig));
    char out_prof_path[512] = "";
    int json_stdout = 0;
    int env_stdout = 0;
    int quiet_mode = 0;

    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h") == 0 || strcmp(argv[ii], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[ii], "-o") == 0 && ii + 1 < argc)
        {
            strncpy(out_prof_path, argv[++ii], sizeof(out_prof_path) - 1);
        }
        else if (strcmp(argv[ii], "-n") == 0 && ii + 1 < argc)
        {
            config.sample_limit = atoi(argv[++ii]);
        }
        else if (strcmp(argv[ii], "-all") == 0)
        {
            config.force_all = 1;
        }
        else if (strcmp(argv[ii], "-warmup") == 0 && ii + 1 < argc)
        {
            config.warmup_frames = atoi(argv[++ii]);
        }
        else if (strcmp(argv[ii], "-double") == 0)
        {
            config.use_double = 1;
        }
        else if (strcmp(argv[ii], "-q") == 0 || strcmp(argv[ii], "--quiet") == 0)
        {
            quiet_mode = 1;
        }
        else if (strcmp(argv[ii], "-progress") == 0 || strcmp(argv[ii], "--progress") == 0)
        {
            config.show_progress = 1;
        }
        else if (strcmp(argv[ii], "-json") == 0 || strcmp(argv[ii], "--json") == 0)
        {
            json_stdout = 1;
        }
        else if (strcmp(argv[ii], "-env") == 0 || strcmp(argv[ii], "--env") == 0)
        {
            env_stdout = 1;
        }
        else if (argv[ii][0] != '-')
        {
            strncpy(config.dataset_path, argv[ii], sizeof(config.dataset_path) - 1);
        }
    }

    if (config.dataset_path[0] == '\0')
    {
        fprintf(stderr, "Error: No dataset path specified.\n");
        print_usage(argv[0]);
        return 1;
    }

    if (out_prof_path[0] == '\0')
    {
        snprintf(out_prof_path, sizeof(out_prof_path), "%s.gricprof", config.dataset_path);
    }

    cli_colors_init();

    ProbeResults results;
    if (probe_run(&config, &results) != 0)
    {
        fprintf(stderr, "Error: Failed to probe dataset '%s'.\n", config.dataset_path);
        return 1;
    }

    /* Save profile sidecar */
    if (gric_profile_write_json(out_prof_path, &results.profile) != 0)
    {
        fprintf(stderr, "Warning: Failed to write profile file '%s'.\n", out_prof_path);
    }
    else if (!quiet_mode && !json_stdout && !env_stdout)
    {
        printf("%s[INFO]%s Generated profile sidecar: %s%s%s\n",
               ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
               ANSI_BOLD, out_prof_path, ANSI_COLOR_RESET);
    }

    if (json_stdout)
    {
        FILE *tmp = fopen(out_prof_path, "r");
        if (tmp != NULL)
        {
            char line[4096];
            while (fgets(line, sizeof(line), tmp))
            {
                fputs(line, stdout);
            }
            fclose(tmp);
        }
    }
    else if (env_stdout)
    {
        probe_report_print_env(stdout, &results);
    }
    else if (!quiet_mode)
    {
        probe_report_print_terminal(stdout, &results);
    }

    probe_results_free(&results);
    return 0;
}

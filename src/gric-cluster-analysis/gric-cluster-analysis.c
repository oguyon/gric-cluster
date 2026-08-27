/**
 * @file gric-cluster-analysis.c
 * @brief Analysis and diagnostics tool for GRIC cluster outputs and telemetry.
 */

#define _POSIX_C_SOURCE 200809L
#include "analysis_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_state(
    AnalysisState *state)
{
    memset(state, 0, sizeof(AnalysisState));
    state->time_clustering_ms = -1.0;
    state->time_output_ms = -1.0;
    state->rlim = -1.0;
    state->dprob = -1.0;
    state->maxcl = -1;
    state->maxim = -1;

    state->dist_hist = calloc(MAX_HISTOGRAM_LIMIT, sizeof(long));
    state->pruned_hist = calloc(MAX_HISTOGRAM_LIMIT, sizeof(long));
    state->query_hist = calloc(MAX_HISTOGRAM_LIMIT, sizeof(long));
} // init_state

void free_state(
    AnalysisState *state)
{
    if (state->dist_hist != NULL)
    {
        free(state->dist_hist);
    }
    if (state->pruned_hist != NULL)
    {
        free(state->pruned_hist);
    }
    if (state->query_hist != NULL)
    {
        free(state->query_hist);
    }
    if (state->assignments != NULL)
    {
        free(state->assignments);
    }
    if (state->cluster_sizes != NULL)
    {
        free(state->cluster_sizes);
    }
    if (state->birth_frames != NULL)
    {
        free(state->birth_frames);
    }
    if (state->death_frames != NULL)
    {
        free(state->death_frames);
    }
    if (state->transition_matrix != NULL)
    {
        free(state->transition_matrix);
    }
    if (state->dcc_matrix != NULL)
    {
        free(state->dcc_matrix);
    }
    if (state->dcc_measured != NULL)
    {
        free(state->dcc_measured);
    }
} // free_state

static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s -d <clusterdat_dir> [options]\n", progname);
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
} // print_usage

static void print_help(
    const char *progname)
{
    printf("%sNAME%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %sgric-cluster-analysis%s - Post-processing diagnostic utility\n\n",
           ansi_bold_green, ansi_reset);

    printf("%sUSAGE%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s%s%s %s-d <clusterdat_dir>%s %s[options]%s\n\n",
           ansi_bold_green, progname, ansi_reset, ansi_color_magenta, ansi_reset,
           ansi_color_grey, ansi_reset);

    printf("%sOPTIONS%s\n", ansi_bold_cyan, ansi_reset);
    printf("  %s-d, --dir <path>%s       Run results folder containing logs and membership\n",
           ansi_color_green, ansi_reset);
    printf("  %s-log <path>%s            Path to run log file (overrides -d path)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-memb <path>%s           Path to membership file (overrides -d path)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-dcc <path>%s            Path to intercluster distances (overrides -d path)\n",
           ansi_color_green, ansi_reset);
    printf("  %s-anchors <path>%s        Path to anchor coordinates coordinates file\n",
           ansi_color_green, ansi_reset);
    printf("  %s-points <path>%s         Path to original coordinates coordinate points\n",
           ansi_color_green, ansi_reset);
    printf("  %s-json%s                  Print report formatted as raw JSON block\n",
           ansi_color_green, ansi_reset);
    printf("  %s-o, --output <path>%s    Write reports onto specified output filename\n",
           ansi_color_green, ansi_reset);
    printf("  %s-h, --help%s             Print this helper window\n",
           ansi_color_green, ansi_reset);
} // print_help

int main(
    int    argc,
    char **argv)
{
    cli_colors_init();

    char *dir_path = NULL;
    char *log_override = NULL;
    char *memb_override = NULL;
    char *dcc_override = NULL;
    char *anchors_override = NULL;
    char *points_override = NULL;
    char *output_file = NULL;
    int   json_mode = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dir") == 0)
        {
            if (i + 1 < argc)
            {
                dir_path = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-log") == 0)
        {
            if (i + 1 < argc)
            {
                log_override = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-memb") == 0)
        {
            if (i + 1 < argc)
            {
                memb_override = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-dcc") == 0)
        {
            if (i + 1 < argc)
            {
                dcc_override = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-anchors") == 0)
        {
            if (i + 1 < argc)
            {
                anchors_override = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-points") == 0)
        {
            if (i + 1 < argc)
            {
                points_override = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
        {
            if (i + 1 < argc)
            {
                output_file = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-json") == 0)
        {
            json_mode = 1;
        }
        else
        {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (dir_path == NULL && log_override == NULL && memb_override == NULL)
    {
        fprintf(stderr, "Error: Missing required directory path (-d) or input overrides.\n");
        print_usage(argv[0]);
        return 1;
    }

    char log_path[4096] = {0};
    char memb_path[4096] = {0};
    char dcc_path[4096] = {0};
    char anchors_path[4096] = {0};

    if (dir_path != NULL)
    {
        snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", dir_path);
        snprintf(memb_path, sizeof(memb_path), "%s/frame_membership.txt", dir_path);
        snprintf(dcc_path, sizeof(dcc_path), "%s/dcc.txt", dir_path);
        snprintf(anchors_path, sizeof(anchors_path), "%s/anchors.txt", dir_path);
    }

    if (log_override != NULL)
    {
        strncpy(log_path, log_override, sizeof(log_path) - 1);
    }
    if (memb_override != NULL)
    {
        strncpy(memb_path, memb_override, sizeof(memb_path) - 1);
    }
    if (dcc_override != NULL)
    {
        strncpy(dcc_path, dcc_override, sizeof(dcc_path) - 1);
    }
    if (anchors_override != NULL)
    {
        strncpy(anchors_path, anchors_override, sizeof(anchors_path) - 1);
    }

    AnalysisState state;
    init_state(&state);

    if (log_path[0] != '\0')
    {
        if (parse_log_file(log_path, &state) != 0)
        {
            fprintf(stderr, "Error: Failed to parse log file '%s'\n", log_path);
            free_state(&state);
            return 1;
        }
    }

    if (memb_path[0] != '\0')
    {
        if (parse_membership_file(memb_path, &state) != 0)
        {
            fprintf(stderr, "Error: Failed to parse membership file '%s'\n", memb_path);
            free_state(&state);
            return 1;
        }
    }

    if (dcc_path[0] != '\0')
    {
        parse_dcc_file(dcc_path, &state);
    }

    compute_derived_stats(&state);

    FILE *out_stream = stdout;
    if (output_file != NULL)
    {
        out_stream = fopen(output_file, "w");
        if (out_stream == NULL)
        {
            fprintf(stderr, "Error: Could not open output file '%s'\n", output_file);
            free_state(&state);
            return 1;
        }
    }

    if (json_mode != 0)
    {
        write_json_report(out_stream, &state);
    }
    else
    {
        write_text_report(out_stream, &state, output_file == NULL ? 1 : 0);
    }

    if (output_file != NULL)
    {
        fclose(out_stream);
    }

    if (json_mode == 0 && output_file == NULL && state.num_frames > 0)
    {
        printf("\n%s--- Distance Evaluation Histogram ---%s\n", ansi_bold_cyan, ansi_reset);
        print_ascii_histogram(state.dist_hist, state.max_hist_val, state.num_frames);
    }

    if (points_override != NULL && anchors_path[0] != '\0')
    {
        analyze_spatial_spread(points_override, anchors_path, &state);
    }

    free_state(&state);
    return 0;
} // main

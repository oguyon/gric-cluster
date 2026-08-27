/**
 * @file analysis_parser.c
 * @brief Parsers for cluster run logs, memberships, and inter-cluster distance files.
 */

#include "analysis_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_log_file(
    const char    *filename,
    AnalysisState *state)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL)
    {
        return -1;
    }

    char line[4096];
    int  in_dist_hist = 0;
    int  in_query_hist = 0;

    while (fgets(line, sizeof(line), f) != NULL)
    {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }

        if (in_dist_hist != 0)
        {
            if (strcmp(line, "STATS_DIST_HIST_END") == 0)
            {
                in_dist_hist = 0;
            }
            else
            {
                int  k;
                long c;
                long p;
                if (sscanf(line, "%d %ld %ld", &k, &c, &p) >= 2 && k >= 0 &&
                    k < MAX_HISTOGRAM_LIMIT)
                {
                    state->dist_hist[k] = c;
                    state->pruned_hist[k] = p;
                    if (k > state->max_hist_val)
                    {
                        state->max_hist_val = k;
                    }
                }
            }
            continue;
        }

        if (in_query_hist != 0)
        {
            if (strcmp(line, "STATS_CLUSTER_QUERIES_END") == 0)
            {
                in_query_hist = 0;
            }
            else
            {
                int  k;
                long q;
                if (sscanf(line, "%d %ld", &k, &q) == 2 && k >= 0 &&
                    k < MAX_HISTOGRAM_LIMIT)
                {
                    state->query_hist[k] = q;
                }
            }
            continue;
        }

        if (strncmp(line, "CMD: ", 5) == 0)
        {
            snprintf(state->cmdline, sizeof(state->cmdline), "%.2047s", line + 5);
        }
        else if (strncmp(line, "START_TIME: ", 12) == 0)
        {
            snprintf(state->start_time, sizeof(state->start_time), "%.127s", line + 12);
        }
        else if (strncmp(line, "TIME_CLUSTERING_MS: ", 20) == 0)
        {
            state->time_clustering_ms = atof(line + 20);
        }
        else if (strncmp(line, "TIME_OUTPUT_MS: ", 16) == 0)
        {
            state->time_output_ms = atof(line + 16);
        }
        else if (strncmp(line, "PARAM_RLIM: ", 12) == 0)
        {
            state->rlim = atof(line + 12);
        }
        else if (strncmp(line, "PARAM_DPROB: ", 13) == 0)
        {
            state->dprob = atof(line + 13);
        }
        else if (strncmp(line, "PARAM_MAXCL: ", 13) == 0)
        {
            state->maxcl = atoi(line + 13);
        }
        else if (strncmp(line, "PARAM_MAXIM: ", 13) == 0)
        {
            state->maxim = atol(line + 13);
        }
        else if (strncmp(line, "STATS_CLUSTERS: ", 16) == 0)
        {
            state->num_clusters = atoi(line + 16);
        }
        else if (strncmp(line, "STATS_FRAMES: ", 14) == 0)
        {
            state->num_frames = atol(line + 14);
        }
        else if (strncmp(line, "STATS_DISTS: ", 13) == 0)
        {
            state->num_dists = atol(line + 13);
        }
        else if (strncmp(line, "STATS_DISTS_SAMPLE: ", 20) == 0)
        {
            state->num_dists_sample = atol(line + 20);
        }
        else if (strncmp(line, "STATS_DISTS_INTERCLUSTER: ", 26) == 0)
        {
            state->num_dists_intercluster = atol(line + 26);
        }
        else if (strncmp(line, "STATS_PRUNED: ", 14) == 0)
        {
            state->num_pruned = atol(line + 14);
        }
        else if (strncmp(line, "STATS_MAX_RSS_KB: ", 18) == 0)
        {
            state->max_rss = atol(line + 18);
        }
        else if (strcmp(line, "STATS_DIST_HIST_START") == 0)
        {
            in_dist_hist = 1;
        }
        else if (strcmp(line, "STATS_CLUSTER_QUERIES_START") == 0)
        {
            in_query_hist = 1;
        }
    }

    fclose(f);
    return 0;
} // parse_log_file

int parse_membership_file(
    const char    *filename,
    AnalysisState *state)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL)
    {
        return -1;
    }

    long count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] != '\n' && line[0] != '\0')
        {
            count++;
        }
    }
    rewind(f);

    if (count == 0)
    {
        fclose(f);
        return 0;
    }

    state->assignments = malloc((size_t)count * sizeof(int));
    if (state->assignments == NULL)
    {
        fclose(f);
        return -1;
    }
    state->assignments_count = count;

    long idx = 0;
    while (fgets(line, sizeof(line), f) != NULL)
    {
        long f_idx;
        int  c_idx;
        if (sscanf(line, "%ld %d", &f_idx, &c_idx) == 2)
        {
            state->assignments[idx++] = c_idx;
        }
    }

    fclose(f);
    return 0;
} // parse_membership_file

int parse_dcc_file(
    const char    *filename,
    AnalysisState *state)
{
    if (state->num_clusters <= 0)
    {
        return -1;
    }

    FILE *f = fopen(filename, "r");
    if (f == NULL)
    {
        return -1;
    }

    int n = state->num_clusters;
    state->dcc_matrix = calloc((size_t)(n * n), sizeof(double));
    state->dcc_measured = calloc((size_t)(n * n), sizeof(int));
    if (state->dcc_matrix == NULL || state->dcc_measured == NULL)
    {
        fclose(f);
        return -1;
    }

    char line[512];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        int    i;
        int    j;
        double d;
        if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3)
        {
            if (i >= 0 && i < n && j >= 0 && j < n)
            {
                state->dcc_matrix[i * n + j] = d;
                state->dcc_measured[i * n + j] = 1;
            }
        }
    }

    fclose(f);
    return 0;
} // parse_dcc_file

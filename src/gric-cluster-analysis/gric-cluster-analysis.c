/**
 * @file gric-cluster-analysis.c
 * @brief Analysis and diagnostics tool for GRIC cluster outputs and telemetry.
 */

#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include "shared/cli_colors.h"

#define MAX_HISTOGRAM_LIMIT 10000

typedef struct
{
    char   cmdline[2048];
    char   start_time[128];
    double time_clustering_ms;
    double time_output_ms;
    double rlim;
    double dprob;
    int    maxcl;
    long   maxim;
    int    num_clusters;
    long   num_frames;
    long   num_dists;
    long   num_dists_sample;
    long   num_dists_intercluster;
    long   num_pruned;
    long   max_rss;

    /* Histograms */
    long  *dist_hist;
    long  *pruned_hist;
    long  *query_hist;
    int    max_hist_val;

    /* Membership data */
    int   *assignments;
    long   assignments_count;

    /* Derived statistics */
    long  *cluster_sizes;
    double shannon_entropy;
    double normalized_entropy;

    /* Temporal Lifetimes */
    long  *birth_frames;
    long  *death_frames;

    /* Transition Matrix */
    long  *transition_matrix;

    /* DCC Matrix */
    double *dcc_matrix;
    int    *dcc_measured;
} AnalysisState;

/* Function Declarations */
static void init_state(
    AnalysisState *state);

static void free_state(
    AnalysisState *state);

static int parse_log_file(
    const char    *filename,
    AnalysisState *state);

static int parse_membership_file(
    const char    *filename,
    AnalysisState *state);

static int parse_dcc_file(
    const char    *filename,
    AnalysisState *state);

static void compute_derived_stats(
    AnalysisState *state);

static void print_ascii_histogram(
    const long *hist,
    int         max_val,
    long        total_count);

static double compute_euclidean_distance(
    const double *restrict p1,
    const double *restrict p2,
    int                    dim);

static int analyze_spatial_spread(
    const char    *points_file,
    const char    *anchors_file,
    AnalysisState *state);

static void write_text_report(
    FILE          *out,
    AnalysisState *state,
    int            color_enabled);

static void write_json_report(
    FILE          *out,
    AnalysisState *state);

static void print_usage(
    const char *progname);

static void print_help(
    const char *progname);

/**
 * init_state() - Initialize state struct pointers and values.
 * @state: The analysis state to initialize.
 */
static void init_state(
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

/**
 * free_state() - Free dynamically allocated members.
 * @state: The analysis state to free.
 */
static void free_state(
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

/**
 * parse_log_file() - Read cluster_run.log and extract parameters, stats, and histograms.
 * @filename: The path of the log file to parse.
 * @state:    The state structure to fill.
 *
 * Return: 0 on success, -1 on error.
 */
static int parse_log_file(
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
        /* Remove trailing newline */
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
                long c, p;
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
        } // if (in_dist_hist != 0)

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
        } // if (in_query_hist != 0)

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
    } // while reading lines

    fclose(f);
    return 0;
} // parse_log_file

/**
 * parse_membership_file() - Parse frame_membership.txt and populate assignments.
 * @filename: Path to membership file.
 * @state:    The state structure to fill.
 *
 * Return: 0 on success, -1 on error.
 */
static int parse_membership_file(
    const char    *filename,
    AnalysisState *state)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL)
    {
        return -1;
    }

    /* Count number of entries first */
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

    state->assignments = malloc(count * sizeof(int));
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
    } // while loading assignments

    fclose(f);
    return 0;
} // parse_membership_file

/**
 * parse_dcc_file() - Read and parse the inter-cluster distances.
 * @filename: DCC text file.
 * @state:    State with num_clusters configured.
 *
 * Return: 0 on success, -1 on error.
 */
static int parse_dcc_file(
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
    state->dcc_matrix = calloc(n * n, sizeof(double));
    state->dcc_measured = calloc(n * n, sizeof(int));
    if (state->dcc_matrix == NULL || state->dcc_measured == NULL)
    {
        fclose(f);
        return -1;
    }

    char line[512];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        int    i, j;
        double d;
        if (sscanf(line, "%d %d %lf", &i, &j, &d) == 3)
        {
            if (i >= 0 && i < n && j >= 0 && j < n)
            {
                state->dcc_matrix[i * n + j] = d;
                state->dcc_measured[i * n + j] = 1;
            }
        }
    } // while reading dcc

    fclose(f);
    return 0;
} // parse_dcc_file

/**
 * compute_derived_stats() - Compute cluster sizes, entropy, transition matrix, and lifetimes.
 * @state: AnalysisState configuration and assignments populated.
 */
static void compute_derived_stats(
    AnalysisState *state)
{
    if (state->num_clusters <= 0)
    {
        return;
    }

    int n = state->num_clusters;
    state->cluster_sizes = calloc(n, sizeof(long));
    state->birth_frames = malloc(n * sizeof(long));
    state->death_frames = malloc(n * sizeof(long));
    state->transition_matrix = calloc(n * n, sizeof(long));

    for (int i = 0; i < n; i++)
    {
        state->birth_frames[i] = -1;
        state->death_frames[i] = -1;
    }

    /* Compute cluster sizes and lifetimes */
    for (long t = 0; t < state->assignments_count; t++)
    {
        int c = state->assignments[t];
        if (c >= 0 && c < n)
        {
            state->cluster_sizes[c]++;
            if (state->birth_frames[c] == -1)
            {
                state->birth_frames[c] = t;
            }
            state->death_frames[c] = t;

            if (t > 0)
            {
                int prev_c = state->assignments[t - 1];
                if (prev_c >= 0 && prev_c < n)
                {
                    state->transition_matrix[prev_c * n + c]++;
                }
            }
        }
    } // for (long t = 0; ...)

    /* Shannon Entropy calculation */
    double entropy = 0.0;
    double log_2 = log(2.0);
    for (int i = 0; i < n; i++)
    {
        if (state->cluster_sizes[i] > 0 && state->assignments_count > 0)
        {
            double p = (double)state->cluster_sizes[i] / (double)state->assignments_count;
            entropy -= p * (log(p) / log_2);
        }
    } // for (int i = 0; ...)

    state->shannon_entropy = entropy;
    if (n > 1)
    {
        state->normalized_entropy = entropy / (log((double)n) / log_2);
    }
    else
    {
        state->normalized_entropy = 0.0;
    }
} // compute_derived_stats

/**
 * print_ascii_histogram() - Renders a simple, clean, visual ASCII histogram bar chart.
 * @hist:        Histogram data values.
 * @max_val:     Maximum evaluation value recorded.
 * @total_count: Total frames.
 */
static void print_ascii_histogram(
    const long *hist,
    int         max_val,
    long        total_count)
{
    long max_bar_val = 0;
    for (int k = 0; k <= max_val; k++)
    {
        if (hist[k] > max_bar_val)
        {
            max_bar_val = hist[k];
        }
    }

    if (max_bar_val == 0 || total_count == 0)
    {
        return;
    }

    int max_width = 40;
    for (int k = 0; k <= max_val; k++)
    {
        if (hist[k] > 0)
        {
            double pct = (double)hist[k] * 100.0 / (double)total_count;
            int    bar_len = (int)((double)hist[k] * max_width / max_bar_val);
            printf("  %4d eval: [", k);
            for (int j = 0; j < bar_len; j++)
            {
                printf("#");
            }
            for (int j = bar_len; j < max_width; j++)
            {
                printf(" ");
            }
            printf("] %8ld frames (%5.1f%%)\n", hist[k], pct);
        }
    } // for (int k = 0; ...)
} // print_ascii_histogram

/**
 * compute_euclidean_distance() - Internal fast Euclidean distance function.
 * @p1:  First vector pointer.
 * @p2:  Second vector pointer.
 * @dim: Vector dimension.
 *
 * Return: Euclidean distance.
 */
static double compute_euclidean_distance(
    const double *restrict p1,
    const double *restrict p2,
    int                    dim)
{
    double sum = 0.0;
    for (int k = 0; k < dim; k++)
    {
        double diff = p1[k] - p2[k];
        sum += diff * diff;
    }
    return sqrt(sum);
} // compute_euclidean_distance

/**
 * analyze_spatial_spread() - Optional analysis of points spatial spread from anchors.
 * @points_file:  Original coordinates text file.
 * @anchors_file: Cluster anchors text file.
 * @state:        Mutable state populated with assignments.
 *
 * Return: 0 on success, -1 on error.
 */
static int analyze_spatial_spread(
    const char    *points_file,
    const char    *anchors_file,
    AnalysisState *state)
{
    FILE *f_pts = fopen(points_file, "r");
    FILE *f_anc = fopen(anchors_file, "r");
    if (f_pts == NULL || f_anc == NULL)
    {
        if (f_pts != NULL)
        {
            fclose(f_pts);
        }
        if (f_anc != NULL)
        {
            fclose(f_anc);
        }
        return -1;
    }

    int n = state->num_clusters;

    /* Detect dimensions from first anchor line */
    int    dim = 0;
    char   first_line[65536];
    if (fgets(first_line, sizeof(first_line), f_anc) == NULL)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }
    rewind(f_anc);

    {
        char *p = first_line;
        int   in_num = 0;
        while (*p != '\0')
        {
            if (!isspace((unsigned char)*p))
            {
                if (!in_num)
                {
                    dim++;
                    in_num = 1;
                }
            }
            else
            {
                in_num = 0;
            }
            p++;
        }
    }

    if (dim == 0)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }

    /* Load all anchors */
    double *anchors = malloc(n * dim * sizeof(double));
    if (anchors == NULL)
    {
        fclose(f_pts);
        fclose(f_anc);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        char anchor_line[65536];
        if (fgets(anchor_line, sizeof(anchor_line), f_anc) == NULL)
        {
            break;
        }
        char *token = strtok(anchor_line, " \t\r\n");
        for (int k = 0; k < dim && token != NULL; k++)
        {
            anchors[i * dim + k] = atof(token);
            token = strtok(NULL, " \t\r\n");
        }
    } // for (int i = 0; ...)
    fclose(f_anc);

    /* Allocate stats arrays for points */
    double *cluster_dist_sum = calloc(n, sizeof(double));
    double *cluster_dist_sq_sum = calloc(n, sizeof(double));
    double *cluster_dist_max = calloc(n, sizeof(double));
    long   *cluster_counts_computed = calloc(n, sizeof(long));

    double *point_coords = malloc(dim * sizeof(double));
    if (cluster_dist_sum == NULL || cluster_dist_sq_sum == NULL ||
        cluster_dist_max == NULL || cluster_counts_computed == NULL ||
        point_coords == NULL)
    {
        free(anchors);
        free(cluster_dist_sum);
        free(cluster_dist_sq_sum);
        free(cluster_dist_max);
        free(cluster_counts_computed);
        free(point_coords);
        fclose(f_pts);
        return -1;
    }

    long frame_idx = 0;
    char point_line[65536];
    while (fgets(point_line, sizeof(point_line), f_pts) != NULL)
    {
        if (point_line[0] == '#')
        {
            continue;
        }

        if (frame_idx >= state->assignments_count)
        {
            break;
        }

        char *token = strtok(point_line, " \t\r\n");
        for (int k = 0; k < dim && token != NULL; k++)
        {
            point_coords[k] = atof(token);
            token = strtok(NULL, " \t\r\n");
        }

        int assigned_c = state->assignments[frame_idx];
        if (assigned_c >= 0 && assigned_c < n)
        {
            double d = compute_euclidean_distance(point_coords,
                                                  &anchors[assigned_c * dim],
                                                  dim);
            cluster_dist_sum[assigned_c] += d;
            cluster_dist_sq_sum[assigned_c] += (d * d);
            if (d > cluster_dist_max[assigned_c])
            {
                cluster_dist_max[assigned_c] = d;
            }
            cluster_counts_computed[assigned_c]++;
        }
        frame_idx++;
    } // while reading points
    fclose(f_pts);

    printf("\n%s--- Cluster Spatial Spread Analysis ---%s\n", ansi_bold_cyan, ansi_reset);
    printf("Dimensions: %d\n", dim);
    printf("  %5s | %10s | %10s | %10s | %10s\n",
           "ID", "Count", "Mean Dist", "Max Dist", "Std Dev");
    for (int i = 0; i < n; i++)
    {
        long cnt = cluster_counts_computed[i];
        if (cnt > 0)
        {
            double mean = cluster_dist_sum[i] / cnt;
            double variance = (cluster_dist_sq_sum[i] / cnt) - (mean * mean);
            if (variance < 0.0)
            {
                variance = 0.0;
            }
            double std_dev = sqrt(variance);
            printf("  %5d | %10ld | %10.5f | %10.5f | %10.5f\n",
                   i, cnt, mean, cluster_dist_max[i], std_dev);
        }
    } // for (int i = 0; ...)

    free(anchors);
    free(cluster_dist_sum);
    free(cluster_dist_sq_sum);
    free(cluster_dist_max);
    free(cluster_counts_computed);
    free(point_coords);
    return 0;
} // analyze_spatial_spread

/**
 * write_text_report() - Generates clean terminal or file textual report.
 * @out:           Output file stream.
 * @state:         Configured AnalysisState.
 * @color_enabled: Color toggling value (0 to disable).
 */
static void write_text_report(
    FILE          *out,
    AnalysisState *state,
    int            color_enabled)
{
    const char *bc = color_enabled != 0 ? ansi_bold_cyan : "";
    const char *br = color_enabled != 0 ? ansi_color_red : "";
    const char *reset = color_enabled != 0 ? ansi_reset : "";

    fprintf(out, "%s=== GRIC Clustering Run Analysis ===%s\n\n", bc, reset);

    if (strlen(state->cmdline) > 0)
    {
        fprintf(out, "Command:    %s\n", state->cmdline);
    }
    if (strlen(state->start_time) > 0)
    {
        fprintf(out, "Start Time: %s\n", state->start_time);
    }

    fprintf(out, "\n%s--- Performance Summary ---%s\n", bc, reset);
    if (state->time_clustering_ms >= 0)
    {
        fprintf(out, "Clustering Execution Time: %.3f ms\n", state->time_clustering_ms);
    }
    if (state->time_output_ms >= 0)
    {
        fprintf(out, "Serialization Time:        %.3f ms\n", state->time_output_ms);
    }
    if (state->max_rss > 0)
    {
        fprintf(out, "Max RAM usage (RSS):       %.2f MB\n", (double)state->max_rss / 1024.0);
    }

    fprintf(out, "\n%s--- Clustering Parameters ---%s\n", bc, reset);
    fprintf(out, "Radius limit (rlim):       %.6f\n", state->rlim);
    fprintf(out, "Prior discount (deltaprob): %.6f\n", state->dprob);
    fprintf(out, "Max cluster capacity:      %d\n", state->maxcl);

    fprintf(out, "\n%s--- Clustering Statistics ---%s\n", bc, reset);
    fprintf(out, "Total Clusters Incurred:   %d\n", state->num_clusters);
    fprintf(out, "Total Frames Ingested:     %ld\n", state->num_frames);
    fprintf(out, "Total Distance Computations: %ld\n", state->num_dists);
    if (state->num_frames > 0)
    {
        double avg_calls = (double)state->num_dists / state->num_frames;
        double ratio = (state->num_clusters > 0) ? avg_calls / state->num_clusters : 0.0;
        fprintf(out, "Avg Distance evaluations:  %.2f per frame (%.1f%% of full scan)\n",
                avg_calls, ratio * 100.0);
    }
    fprintf(out, "Triangle Pruning Actions:  %ld\n", state->num_pruned);

    if (state->assignments_count > 0)
    {
        fprintf(out, "\n%s--- Cluster Balance & Entropy ---%s\n", bc, reset);
        fprintf(out, "Shannon Entropy:           %.4f bits\n", state->shannon_entropy);
        fprintf(out, "Evenness (Normalized H):   %.4f (closer to 1.0 is more balanced)\n",
                state->normalized_entropy);

        /* Find largest and smallest non-empty clusters */
        int  largest_id = -1;
        long largest_sz = -1;
        int  smallest_id = -1;
        long smallest_sz = -1;
        int  empty_count = 0;

        for (int i = 0; i < state->num_clusters; i++)
        {
            long sz = state->cluster_sizes[i];
            if (sz == 0)
            {
                empty_count++;
                continue;
            }

            if (sz > largest_sz)
            {
                largest_sz = sz;
                largest_id = i;
            }

            if (smallest_sz == -1 || sz < smallest_sz)
            {
                smallest_sz = sz;
                smallest_id = i;
            }
        } // for (int i = 0; ...)

        fprintf(out, "Largest Cluster:           %d (%ld frames)\n", largest_id, largest_sz);
        fprintf(out, "Smallest Cluster:          %d (%ld frames)\n", smallest_id, smallest_sz);
        fprintf(out, "Empty/Pruned Clusters:     %d\n", empty_count);
    } // if (state->assignments_count > 0)

    if (state->assignments_count > 0)
    {
        fprintf(out, "\n%s--- Temporal Dynamics & Lifetimes ---%s\n", bc, reset);
        fprintf(out, "  %5s | %10s | %10s | %10s | %10s | %8s\n",
                "ID", "Count", "Birth Frame", "Death Frame", "Span", "Duty %");
        for (int i = 0; i < state->num_clusters; i++)
        {
            long sz = state->cluster_sizes[i];
            if (sz > 0)
            {
                long birth = state->birth_frames[i];
                long death = state->death_frames[i];
                long span = death - birth + 1;
                double duty = (double)sz * 100.0 / span;
                fprintf(out, "  %5d | %10ld | %10ld | %10ld | %10ld | %7.1f%%\n",
                        i, sz, birth, death, span, duty);
            }
        } // for (int i = 0; ...)
    } // if (state->assignments_count > 0)

    /* Transition Dynamics */
    if (state->assignments_count > 0 && state->transition_matrix != NULL)
    {
        int n = state->num_clusters;
        fprintf(out, "\n%s--- State Transition Dynamics ---%s\n", bc, reset);

        /* Print dwell statistics */
        fprintf(out, "  %5s | %12s | %10s\n", "ID", "Self-Prob", "Dwell (fr)");
        for (int i = 0; i < n; i++)
        {
            long self_trans = state->transition_matrix[i * n + i];
            long total_out = 0;
            for (int j = 0; j < n; j++)
            {
                total_out += state->transition_matrix[i * n + j];
            }

            if (total_out > 0)
            {
                double self_prob = (double)self_trans / (double)total_out;
                double dwell_time = (self_prob < 1.0) ? 1.0 / (1.0 - self_prob) : INFINITY;
                if (dwell_time == INFINITY)
                {
                    fprintf(out, "  %5d | %11.2f%% | %10s\n", i, self_prob * 100.0, "Stable");
                }
                else
                {
                    fprintf(out, "  %5d | %11.2f%% | %10.2f\n", i, self_prob * 100.0, dwell_time);
                }
            }
        } // for (int i = 0; ...)

        /* List Top 5 transitions between distinct clusters */
        typedef struct
        {
            int  src;
            int  dst;
            long count;
        } TransitionRecord;

        TransitionRecord top_trans[5] = { {0} };
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    long count = state->transition_matrix[i * n + j];
                    if (count > 0)
                    {
                        /* Insert into sorted list of top 5 */
                        for (int k = 0; k < 5; k++)
                        {
                            if (count > top_trans[k].count)
                            {
                                for (int m = 4; m > k; m--)
                                {
                                    top_trans[m] = top_trans[m - 1];
                                }
                                top_trans[k].src = i;
                                top_trans[k].dst = j;
                                top_trans[k].count = count;
                                break;
                            }
                        }
                    }
                }
            }
        } // for (int i = 0; ...)

        fprintf(out, "\nTop Distinct-State Transitions:\n");
        for (int k = 0; k < 5; k++)
        {
            if (top_trans[k].count > 0)
            {
                double pct = (double)top_trans[k].count * 100.0 / state->assignments_count;
                fprintf(out, "  Cluster %3d -> Cluster %3d: %6ld times (%5.2f%% of run)\n",
                        top_trans[k].src, top_trans[k].dst, top_trans[k].count, pct);
            }
        }
    } // if (state->assignments_count > 0 && ...)

    /* Inter-cluster Distance Topology */
    if (state->dcc_matrix != NULL)
    {
        int    n = state->num_clusters;
        double sum = 0.0;
        long   cnt = 0;
        double min_dist = -1.0;
        double max_dist = -1.0;
        int    min_i = -1, min_j = -1;
        int    max_i = -1, max_j = -1;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (state->dcc_measured[i * n + j] != 0)
                {
                    double d = state->dcc_matrix[i * n + j];
                    sum += d;
                    cnt++;
                    if (min_dist < 0.0 || d < min_dist)
                    {
                        min_dist = d;
                        min_i = i;
                        min_j = j;
                    }
                    if (max_dist < 0.0 || d > max_dist)
                    {
                        max_dist = d;
                        max_i = i;
                        max_j = j;
                    }
                }
            }
        } // for (int i = 0; ...)

        fprintf(out, "\n%s--- Cluster Distance Topology ---%s\n", bc, reset);
        if (cnt > 0)
        {
            fprintf(out, "Average distance: %.5f\n", sum / cnt);
            fprintf(out, "Minimum distance: %.5f (between C%d and C%d)\n",
                    min_dist, min_i, min_j);
            fprintf(out, "Maximum distance: %.5f (between C%d and C%d)\n",
                    max_dist, max_i, max_j);

            /* Check boundary violations: closest pair vs rlim */
            if (min_dist < state->rlim)
            {
                fprintf(out, "  %s[WARNING] closest pair distance (%.5f) < radius limit (%.5f)%s\n",
                        br, min_dist, state->rlim, reset);
            }
        }
        else
        {
            fprintf(out, "No inter-cluster distances recorded.\n");
        }
    } // if (state->dcc_matrix != NULL)
} // write_text_report

/**
 * write_json_report() - Outputs analysis stats structured in JSON format.
 * @out:   Output file stream.
 * @state: Configured AnalysisState.
 */
static void write_json_report(
    FILE          *out,
    AnalysisState *state)
{
    fprintf(out, "{\n");
    fprintf(out, "  \"cmdline\": \"%s\",\n", state->cmdline);
    fprintf(out, "  \"start_time\": \"%s\",\n", state->start_time);
    fprintf(out, "  \"time_clustering_ms\": %.3f,\n", state->time_clustering_ms);
    fprintf(out, "  \"time_serialization_ms\": %.3f,\n", state->time_output_ms);
    fprintf(out, "  \"max_rss_mb\": %.2f,\n", (double)state->max_rss / 1024.0);
    fprintf(out, "  \"params\": {\n");
    fprintf(out, "    \"rlim\": %.6f,\n", state->rlim);
    fprintf(out, "    \"dprob\": %.6f,\n", state->dprob);
    fprintf(out, "    \"maxcl\": %d\n", state->maxcl);
    fprintf(out, "  },\n");
    fprintf(out, "  \"stats\": {\n");
    fprintf(out, "    \"num_clusters\": %d,\n", state->num_clusters);
    fprintf(out, "    \"num_frames\": %ld,\n", state->num_frames);
    fprintf(out, "    \"num_dists\": %ld,\n", state->num_dists);
    fprintf(out, "    \"num_pruned\": %ld\n", state->num_pruned);
    fprintf(out, "  },\n");
    fprintf(out, "  \"entropy\": {\n");
    fprintf(out, "    \"shannon_entropy_bits\": %.4f,\n", state->shannon_entropy);
    fprintf(out, "    \"normalized_entropy\": %.4f\n", state->normalized_entropy);
    fprintf(out, "  },\n");

    fprintf(out, "  \"clusters\": [\n");
    for (int i = 0; i < state->num_clusters; i++)
    {
        long sz = (state->cluster_sizes != NULL) ? state->cluster_sizes[i] : 0;
        long birth = (state->birth_frames != NULL) ? state->birth_frames[i] : -1;
        long death = (state->death_frames != NULL) ? state->death_frames[i] : -1;
        fprintf(out, "    {\n");
        fprintf(out, "      \"id\": %d,\n", i);
        fprintf(out, "      \"size\": %ld,\n", sz);
        fprintf(out, "      \"birth_frame\": %ld,\n", birth);
        fprintf(out, "      \"death_frame\": %ld\n", death);
        fprintf(out, "    }%s\n", (i == state->num_clusters - 1) ? "" : ",");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
} // write_json_report

/**
 * print_usage() - Print small command syntax block.
 * @progname: Program binary name.
 */
static void print_usage(
    const char *progname)
{
    fprintf(stderr, "Usage: %s -d <clusterdat_dir> [options]\n", progname);
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
} // print_usage

/**
 * print_help() - Print styled full command instruction.
 * @progname: Program binary name.
 */
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
    } // for (int i = 1; ...)

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

    /* Read run log */
    if (log_path[0] != '\0')
    {
        if (parse_log_file(log_path, &state) != 0)
        {
            fprintf(stderr, "Error: Failed to parse log file '%s'\n", log_path);
            free_state(&state);
            return 1;
        }
    }

    /* Read membership log */
    if (memb_path[0] != '\0')
    {
        if (parse_membership_file(memb_path, &state) != 0)
        {
            fprintf(stderr, "Error: Failed to parse membership file '%s'\n", memb_path);
            free_state(&state);
            return 1;
        }
    }

    /* Read DCC distance matrix */
    if (dcc_path[0] != '\0')
    {
        parse_dcc_file(dcc_path, &state);
    }

    /* Compute secondary metrics */
    compute_derived_stats(&state);

    /* Write target report outputs */
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

    /* Visual terminal histogram details */
    if (json_mode == 0 && output_file == NULL && state.num_frames > 0)
    {
        printf("\n%s--- Distance Evaluation Histogram ---%s\n", ansi_bold_cyan, ansi_reset);
        print_ascii_histogram(state.dist_hist, state.max_hist_val, state.num_frames);
    }

    /* Analyze point spread if requested */
    if (points_override != NULL && anchors_path[0] != '\0')
    {
        analyze_spatial_spread(points_override, anchors_path, &state);
    }

    free_state(&state);
    return 0;
}

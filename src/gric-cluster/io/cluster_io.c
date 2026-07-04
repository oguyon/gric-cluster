/**
 * @file cluster_io.c
 * @brief Input/output management for the core clustering engine.
 *
 * Manages formatted output of clustering results, generation of runs logs, and
 * printing of colored command line help screens.
 *
 * Main Functions:
 * - print_help: Prints the detailed, colored clustering CLI help.
 * - write_results: Outputs the clustered coordinate and membership files.
 * - write_run_log: Dumps step-by-step diagnostic information of the clustering execution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef USE_CFITSIO
#include <fitsio.h>
#endif
#include "cluster_io.h"
#include "frameread.h"

// Forward decl for PNG writing
#ifdef USE_PNG
void write_png_frame(const char *filename, double *data, int width, int height);
#endif

#include <unistd.h>

static const char *ansi_color_orange = "";
static const char *ansi_color_green = "";
static const char *ansi_color_red = "";
static const char *ansi_color_blue = "";
static const char *ansi_bg_green = "";
static const char *ansi_color_black = "";
static const char *ansi_color_reset = "";
static const char *ansi_bold = "";
static const char *ansi_underline = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_yellow = "";
static const char *ansi_color_grey = "";
static const char *ansi_color_cyan = "";

#define ANSI_COLOR_ORANGE  ansi_color_orange
#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RED     ansi_color_red
#define ANSI_COLOR_BLUE    ansi_color_blue
#define ANSI_BG_GREEN      ansi_bg_green
#define ANSI_COLOR_BLACK   ansi_color_black
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_UNDERLINE     ansi_underline
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_YELLOW  ansi_color_yellow
#define ANSI_COLOR_GREY    ansi_color_grey
#define ANSI_COLOR_CYAN    ansi_color_cyan
void init_colors_io(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_color_orange = "\x1b[38;5;208m";
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_blue = "\x1b[34m";
        ansi_bg_green = "\x1b[42m";
        ansi_color_black = "\x1b[30m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_underline = "\x1b[4m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_cyan = "\x1b[36m";
    }
} // init_colors_io

/**
 * create_output_dir_name() - Build output directory path from input filename.
 * @input_file: Path to the input file (FITS, MP4, or text).
 *
 * Strips the directory prefix and recognized extensions
 * (.fits.fz, .fits, .mp4, .txt) from @input_file, then
 * appends ".clusterdat" to form the output directory name.
 *
 * Return: Heap-allocated directory name string, or NULL on
 *         allocation failure.  Caller must free().
 */
char *create_output_dir_name(const char *input_file)
{
    const char *base = strrchr(input_file, '/');
    if (base)
    {
        base++;
    }
    else
    {
        base = input_file;
    }
    char *name = strdup(base);
    if (!name)
        return NULL;
    size_t len = strlen(name);
    if (len > 8 && strcmp(name + len - 8, ".fits.fz") == 0)
        name[len - 8] = '\0';
    else if (len > 5 && strcmp(name + len - 5, ".fits") == 0)
        name[len - 5] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".mp4") == 0)
        name[len - 4] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".txt") == 0)
        name[len - 4] = '\0';

    size_t new_len = strlen(name) + strlen(".clusterdat") + 1;
    char *out_dir = (char *)malloc(new_len);
    if (out_dir)
        sprintf(out_dir, "%s.clusterdat", name);
    free(name);
    return out_dir;
}

/**
 * write_results() - Outputs the clustered coordinate and membership files.
 * @config: Pointer to the active ClusterConfig.
 * @state:  Pointer to the active ClusterState.
 *
 * Saves final clusters configuration and visitor tracking data onto files
 * in the configured results directory.
 */
void write_results(
    ClusterConfig *config,
    ClusterState  *state)
{
    char *out_dir = NULL;
    if (config->output.user_outdir)
        out_dir = strdup(config->output.user_outdir);
    else
        out_dir = create_output_dir_name(config->input.fits_filename);

    if (!out_dir)
        return;

    char out_path[4096];

    // Write dcc.txt
    if (config->output.output_dcc)
    {
        printf("Writing dcc.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/dcc.txt", out_dir);
        FILE *dcc_out = fopen(out_path, "w");
        if (dcc_out)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                for (int j = 0; j < state->num_clusters; j++)
                {
                    double d = state->scratch.dcc_min[i * config->algo.maxnbclust + j];
                    if (state->scratch.dcc_measured[i * config->algo.maxnbclust + j] && d >= 0)
                        fprintf(dcc_out, "%d %d %.6f\n", i, j, d);
                }
            }
            fclose(dcc_out);
        }
    }

    // Write Transition Matrix
    if (config->output.output_tm && state->transition_matrix)
    {
        printf("Writing transition_matrix.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/transition_matrix.txt", out_dir);
        FILE *tm_out = fopen(out_path, "w");
        if (tm_out)
        {
            for (int i = 0; i < state->num_clusters; i++)
            {
                for (int j = 0; j < state->num_clusters; j++)
                {
                    long val = state->transition_matrix[i * config->algo.maxnbclust + j];
                    if (val > 0)
                    {
                        fprintf(tm_out, "%d %d %ld\n", i, j, val);
                    }
                }
            }
            fclose(tm_out);
        }
    }

    // Write Anchors
    long width = get_frame_width();
    long height = get_frame_height();
    long nelements = width * height;

    if (config->output.output_anchors)
    {
        printf("Writing anchors\n");
        if (config->output.pngout_mode)
        {
#ifdef USE_PNG
            for (int i = 0; i < state->num_clusters; i++)
            {
                snprintf(out_path, sizeof(out_path), "%s/anchor_%04d.png", out_dir, i);
                write_png_frame(out_path, state->clusters[i].anchor.data, width, height);
            }
#else
            fprintf(stderr, "Warning: PNG output requested but not compiled in.\n");
#endif
        }
        else if (is_ascii_input_mode() && !config->output.fitsout_mode)
        {
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    for (long k = 0; k < nelements; k++)
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
        }
        else
        {
#ifdef USE_CFITSIO
            int status = 0;
            fitsfile *afptr;
            snprintf(out_path, sizeof(out_path), "!%s/anchors.fits", out_dir);
            fits_create_file(&afptr, out_path, &status);
            long naxes[3] = {width, height, state->num_clusters};
            fits_create_img(afptr, DOUBLE_IMG, 3, naxes, &status);
            for (int i = 0; i < state->num_clusters; i++)
            {
                long fpixel[3] = {1, 1, i + 1};
                fits_write_pix(afptr, TDOUBLE, fpixel, nelements, state->clusters[i].anchor.data,
                               &status);
            }
            fits_close_file(afptr, &status);
#else
            // Fallback to text if fits disabled but requested?
            fprintf(stderr,
                    "Warning: FITS output requested but not compiled in. Saving as ASCII.\n");
            // Reuse ASCII logic
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    for (long k = 0; k < nelements; k++)
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
#endif
        }
    }

    // Cluster Counts
    int *cluster_counts = (int *)calloc(state->num_clusters, sizeof(int));
    for (long i = 0; i < state->telemetry.total_frames_processed; i++)
    {
        if (state->assignments[i] >= 0 && state->assignments[i] < state->num_clusters)
            cluster_counts[state->assignments[i]]++;
    }
    if (config->output.output_counts)
    {
        printf("Writing cluster_counts.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/cluster_counts.txt", out_dir);
        FILE *count_out = fopen(out_path, "w");
        if (count_out)
        {
            for (int c = 0; c < state->num_clusters; c++)
                fprintf(count_out, "Cluster %d: %d frames\n", c, cluster_counts[c]);
            fclose(count_out);
        }
    }

    // Average buffer
    double *avg_buffer = NULL;
    if (config->output.average_mode)
        avg_buffer = (double *)calloc(nelements, sizeof(double));

    int active_cluster_count = 0;
    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] > 0)
            active_cluster_count++;
    }

    if (config->output.output_clusters)
    {
        printf("Writing cluster files (%d files)\n", active_cluster_count);
    }

    if (config->output.average_mode)
    {
        printf("Writing average cluster files\n");
    }

    if (config->output.pngout_mode)
    {
#ifdef USE_PNG
        for (int c = 0; c < state->num_clusters; c++)
        {
            if (cluster_counts[c] == 0)
                continue;

            if (config->output.output_clusters)
            {
                char cluster_dir[1024];
                snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                mkdir(cluster_dir, 0777);
            }

            if (config->output.average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;

            for (long f = 0; f < state->telemetry.total_frames_processed; f++)
            {
                if (state->assignments[f] == c)
                {
                    Frame *fr = getframe_at(f);
                    if (fr)
                    {
                        if (config->output.output_clusters)
                        {
                            char cluster_dir[1024];
                            snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir,
                                     c);
                            snprintf(out_path, sizeof(out_path), "%s/frame%05ld.png", cluster_dir,
                                     f);
                            write_png_frame(out_path, fr->data, width, height);
                        }
                        if (config->output.average_mode)
                            for (long k = 0; k < nelements; k++)
                                avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                    }
                }
            }

            if (config->output.average_mode)
            {
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] /= cluster_counts[c];
                snprintf(out_path, sizeof(out_path), "%s/average_%04d.png", out_dir, c);
                write_png_frame(out_path, avg_buffer, width, height);
            }
        }
#endif
    }
    else if (is_ascii_input_mode() && !config->output.fitsout_mode)
    {
        FILE *avg_file = NULL;
        if (config->output.average_mode)
        {
            snprintf(out_path, sizeof(out_path), "%s/average.txt", out_dir);
            avg_file = fopen(out_path, "w");
        }
        for (int c = 0; c < state->num_clusters; c++)
        {
            if (cluster_counts[c] == 0)
            {
                if (avg_file)
                {
                    for (long k = 0; k < nelements; k++)
                        fprintf(avg_file, "0.0 ");
                    fprintf(avg_file, "\n");
                }
                continue;
            }

            FILE *cfptr = NULL;
            if (config->output.output_clusters)
            {
                char fname[1024];
                snprintf(fname, sizeof(fname), "%s/cluster_%d.txt", out_dir, c);
                cfptr = fopen(fname, "w");
            }

            if (config->output.average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;
            for (long f = 0; f < state->telemetry.total_frames_processed; f++)
            {
                if (state->assignments[f] == c)
                {
                    Frame *fr = getframe_at(f);
                    if (fr)
                    {
                        for (long k = 0; k < nelements; k++)
                        {
                            if (cfptr)
                                fprintf(cfptr, "%f ", fr->data[k]);
                            if (config->output.average_mode)
                                avg_buffer[k] += fr->data[k];
                        }
                        if (cfptr)
                            fprintf(cfptr, "\n");
                        free_frame(fr);
                    }
                }
            }
            if (cfptr)
                fclose(cfptr);
            if (avg_file)
            {
                for (long k = 0; k < nelements; k++)
                    fprintf(avg_file, "%f ", avg_buffer[k] / cluster_counts[c]);
                fprintf(avg_file, "\n");
            }
        }
        if (avg_file)
            fclose(avg_file);
    }
    else
    {
#ifdef USE_CFITSIO
        int status = 0;
        fitsfile *avg_ptr = NULL;
        if (config->output.average_mode)
        {
            snprintf(out_path, sizeof(out_path), "!%s/average.fits", out_dir);
            fits_create_file(&avg_ptr, out_path, &status);
            long anaxes[3] = {width, height, state->num_clusters};
            fits_create_img(avg_ptr, DOUBLE_IMG, 3, anaxes, &status);
        }
        for (int c = 0; c < state->num_clusters; c++)
        {
            if (cluster_counts[c] == 0)
                continue;

            fitsfile *cfptr = NULL;
            if (config->output.output_clusters)
            {
                char fname[1024];
                snprintf(fname, sizeof(fname), "!%s/cluster_%d.fits", out_dir, c);
                fits_create_file(&cfptr, fname, &status);
                long cnaxes[3] = {width, height, cluster_counts[c]};
                fits_create_img(cfptr, DOUBLE_IMG, 3, cnaxes, &status);
            }

            if (config->output.average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;
            int fr_count = 0;
            for (long f = 0; f < state->telemetry.total_frames_processed; f++)
            {
                if (state->assignments[f] == c)
                {
                    Frame *fr = getframe_at(f);
                    if (fr)
                    {
                        if (cfptr)
                        {
                            long fpixel[3] = {1, 1, fr_count + 1};
                            fits_write_pix(cfptr, TDOUBLE, fpixel, nelements, fr->data, &status);
                        }
                        if (config->output.average_mode)
                            for (long k = 0; k < nelements; k++)
                                avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                        fr_count++;
                    }
                }
            }
            if (cfptr)
                fits_close_file(cfptr, &status);
            if (config->output.average_mode && avg_ptr)
            {
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] /= cluster_counts[c];
                long fpixel[3] = {1, 1, c + 1};
                fits_write_pix(avg_ptr, TDOUBLE, fpixel, nelements, avg_buffer, &status);
            }
        }
        if (avg_ptr)
            fits_close_file(avg_ptr, &status);
#else
// Fallback ASCII logic if FITS disabled but we reached here
// (Similar to block above)
#endif
    }

    if (avg_buffer)
        free(avg_buffer);

    if (config->output.output_clustered)
    {
        printf("Writing clustered output file\n");

        const char *base_name_only = strrchr(config->input.fits_filename, '/');
        if (base_name_only)
            base_name_only++;
        else
            base_name_only = config->input.fits_filename;

        char *temp_base = strdup(base_name_only);
        char *ext = strrchr(temp_base, '.');
        if (ext && strcmp(ext, ".txt") == 0)
            *ext = '\0';

        char *clustered_fname = (char *)malloc(strlen(out_dir) + strlen(temp_base) + 30);
        sprintf(clustered_fname, "%s/%s.clustered.txt", out_dir, temp_base);
        free(temp_base);

        FILE *clustered_out = fopen(clustered_fname, "w");
        if (clustered_out)
        {
            fprintf(clustered_out, "# Parameters:\n");
            fprintf(clustered_out, "# rlim %.6f\n", config->algo.rlim);
            fprintf(clustered_out, "# dprob %.6f\n", config->algo.deltaprob);
            fprintf(clustered_out, "# maxcl %d\n", config->algo.maxnbclust);
            fprintf(clustered_out, "# maxim %ld\n", config->input.maxnbfr);
            fprintf(clustered_out, "# gprob_mode %d\n", config->optim.gprob_mode);
            fprintf(clustered_out, "# fmatcha %.2f\n", config->optim.fmatch_a);
            fprintf(clustered_out, "# fmatchb %.2f\n", config->optim.fmatch_b);

            fprintf(clustered_out, "# Stats:\n");
            fprintf(clustered_out, "# Total Clusters %d\n", state->num_clusters);
            fprintf(clustered_out, "# Total Distance Computations %ld\n", state->telemetry.framedist_calls);
            fprintf(clustered_out, "# Clusters Pruned %ld\n", state->telemetry.clusters_pruned);
            double avg_dist = (state->telemetry.total_frames_processed > 0)
                                  ? (double)state->telemetry.framedist_calls / state->telemetry.total_frames_processed
                                  : 0.0;
            fprintf(clustered_out, "# Avg Dist/Frame %.2f\n", avg_dist);

            if (state->telemetry.pruned_fraction_sum && state->telemetry.step_counts)
            {
                for (int k = 0; k < state->telemetry.max_steps_recorded; k++)
                {
                    if (state->telemetry.step_counts[k] > 0)
                    {
                        fprintf(clustered_out, "# Pruning Step %d: %.4f\n", k,
                                state->telemetry.pruned_fraction_sum[k] / state->telemetry.step_counts[k]);
                    }
                    else if (k > 0 && state->telemetry.step_counts[k] == 0)
                    {
                        break;
                    }
                }
            }

            int next_new_cluster = 0;
            for (long i = 0; i < state->telemetry.total_frames_processed; i++)
            {
                int assigned = state->assignments[i];
                if (assigned == next_new_cluster)
                {
                    fprintf(clustered_out, "# NEWCLUSTER %d %ld ", assigned, i);
                    for (long k = 0; k < nelements; k++)
                        fprintf(clustered_out, "%f ", state->clusters[assigned].anchor.data[k]);
                    fprintf(clustered_out, "\n");
                    next_new_cluster++;
                }
                Frame *fr = getframe_at(i);
                if (fr)
                {
                    fprintf(clustered_out, "%ld %d ", i, assigned);
                    for (long k = 0; k < nelements; k++)
                        fprintf(clustered_out, "%f ", fr->data[k]);
                    fprintf(clustered_out, "\n");
                    free_frame(fr);
                }
            }
            fclose(clustered_out);
        }
        free(clustered_fname);
    }
    free(cluster_counts);
    free(out_dir);
}

/**
 * write_run_log() - Dumps step-by-step diagnostic information of the execution.
 * @config:   Pointer to the active ClusterConfig.
 * @state:    Pointer to the active ClusterState.
 * @cmdline:  The command line string used to invoke the process.
 * @start_ts: Real-time clock timestamp of program launch.
 * @clust_ms: Duration of the clustering loop in milliseconds.
 * @out_ms:   Duration of results serialization in milliseconds.
 * @max_rss:  Maximum resident set size in KB.
 */
void write_run_log(
    ClusterConfig  *config,
    ClusterState   *state,
    const char     *cmdline,
    struct timespec start_ts,
    double          clust_ms,
    double          out_ms,
    long            max_rss)
{
    char *out_dir = NULL;
    if (config->output.user_outdir)
        out_dir = strdup(config->output.user_outdir);
    else
        out_dir = create_output_dir_name(config->input.fits_filename);

    if (!out_dir)
        return;

    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", out_dir);
    FILE *f = fopen(log_path, "w");
    if (f)
    {
        char time_buf[64];
        struct tm *tm_info = localtime(&start_ts.tv_sec);
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

        fprintf(f, "CMD: %s\n", cmdline);
        fprintf(f, "START_TIME: %s.%09ld\n", time_buf, start_ts.tv_nsec);
        fprintf(f, "TIME_CLUSTERING_MS: %.3f\n", clust_ms);
        fprintf(f, "TIME_OUTPUT_MS: %.3f\n", out_ms);
        fprintf(f, "OUTPUT_DIR: %s\n", out_dir);
        fprintf(f, "PARAM_RLIM: %f\n", config->algo.rlim);
        fprintf(f, "PARAM_DPROB: %f\n", config->algo.deltaprob);
        fprintf(f, "PARAM_MAXCL: %d\n", config->algo.maxnbclust);
        fprintf(f, "PARAM_MAXIM: %ld\n", config->input.maxnbfr);
        fprintf(f, "PARAM_GPROB: %d\n", config->optim.gprob_mode);
        fprintf(f, "PARAM_FMATCHA: %f\n", config->optim.fmatch_a);
        fprintf(f, "PARAM_FMATCHB: %f\n", config->optim.fmatch_b);
        fprintf(f, "PARAM_TE4: %d\n", config->optim.te4_mode);
        fprintf(f, "PARAM_TE5: %d\n", config->optim.te5_mode);
        fprintf(f, "PARAM_ENTROPY: %d\n",
                config->optim.entropy_mode);
        fprintf(f, "PARAM_ENTROPY_FAST: %d\n",
                config->optim.entropy_fast_mode);
        fprintf(f, "PARAM_ENTROPY_GATE: %f\n",
                config->optim.entropy_gate_bits);
        fprintf(f, "PARAM_ENTROPY_FIRST_GATE: %f\n",
                config->optim.entropy_first_gate_bits);
        fprintf(f, "PARAM_ENTROPY_MAX_TARGETS: %d\n",
                config->optim.entropy_max_targets);
        fprintf(f, "PARAM_ENTROPY_MIN_PROB: %f\n",
                config->optim.entropy_min_prob);

        if (config->output.output_dcc)
            fprintf(f, "OUTPUT_FILE: %s/dcc.txt\n", out_dir);
        if (config->output.output_tm)
            fprintf(f, "OUTPUT_FILE: %s/transition_matrix.txt\n", out_dir);
        if (config->output.output_anchors)
            fprintf(f, "OUTPUT_FILE: %s/anchors.txt\n", out_dir);
        if (config->output.output_counts)
            fprintf(f, "OUTPUT_FILE: %s/cluster_counts.txt\n", out_dir);
        if (config->output.output_membership)
            fprintf(f, "OUTPUT_FILE: %s/frame_membership.txt\n", out_dir);

        if (config->output.output_clustered)
        {
            const char *base_name_only = strrchr(config->input.fits_filename, '/');
            if (base_name_only)
                base_name_only++;
            else
                base_name_only = config->input.fits_filename;
            char *temp_base = strdup(base_name_only);
            char *ext = strrchr(temp_base, '.');
            if (ext && strcmp(ext, ".txt") == 0)
                *ext = '\0';
            fprintf(f, "CLUSTERED_FILE: %s/%s.clustered.txt\n", out_dir, temp_base);
            free(temp_base);
        }

        fprintf(f, "STATS_CLUSTERS: %d\n", state->num_clusters);
        fprintf(f, "STATS_FRAMES: %ld\n", state->telemetry.total_frames_processed);
        fprintf(f, "STATS_DISTS: %ld\n", state->telemetry.framedist_calls);
        fprintf(f, "STATS_DISTS_SAMPLE: %ld\n", state->telemetry.framedist_calls_sample);
        fprintf(f, "STATS_DISTS_INTERCLUSTER: %ld\n", state->telemetry.framedist_calls_intercluster);
        fprintf(f, "STATS_PRUNED: %ld\n", state->telemetry.clusters_pruned);
        fprintf(f, "STATS_MAX_RSS_KB: %ld\n", max_rss);
        fprintf(f, "STATS_TIME_STEP_1_MS: %.3f\n", state->telemetry.time_step_1);
        fprintf(f, "STATS_TIME_STEP_2_MS: %.3f\n", state->telemetry.time_step_2);
        fprintf(f, "STATS_TIME_STEP_3A_MS: %.3f\n", state->telemetry.time_step_3a);
        fprintf(f, "STATS_TIME_STEP_3B_MS: %.3f\n", state->telemetry.time_step_3b);
        fprintf(f, "STATS_TIME_STEP_3B_SCORE_MS: %.3f\n", state->telemetry.time_step_3b_score);
        fprintf(f, "STATS_TIME_STEP_3B_FILTER_MS: %.3f\n", state->telemetry.time_step_3b_filter);
        fprintf(f, "STATS_TIME_STEP_3B_EVAL_MS: %.3f\n", state->telemetry.time_step_3b_eval);
        fprintf(f, "STATS_TIME_STEP_3C_MS: %.3f\n", state->telemetry.time_step_3c);
        fprintf(f, "STATS_TIME_STEP_4_MS: %.3f\n", state->telemetry.time_step_4);
        fprintf(f, "STATS_TIME_STEP_5_MS: %.3f\n", state->telemetry.time_step_5);
        fprintf(f, "STATS_TIME_STEP_REFINE_MS: %.3f\n", state->telemetry.time_step_refine);

        fprintf(f, "STATS_DIST_HIST_START\n");
        for (int k = 0; k <= config->algo.maxnbclust; k++)
        {
            if (state->telemetry.dist_counts && state->telemetry.dist_counts[k] > 0)
            {
                fprintf(f, "%d %ld %ld\n", k, state->telemetry.dist_counts[k],
                        state->telemetry.pruned_counts_by_dist[k]);
            }
        }
        fprintf(f, "STATS_DIST_HIST_END\n");

        fprintf(f, "STATS_CLUSTER_QUERIES_START\n");
        for (int k = 0; k < state->num_clusters; k++)
        {
            if (state->telemetry.cluster_query_counts && state->telemetry.cluster_query_counts[k] > 0)
            {
                fprintf(f, "%d %ld\n", k, state->telemetry.cluster_query_counts[k]);
            }
        }
        fprintf(f, "STATS_CLUSTER_QUERIES_END\n");

        fclose(f);
        printf("Log written to %s\n", log_path);
    }
    free(out_dir);
}
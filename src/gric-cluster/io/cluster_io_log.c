/**
 * @file cluster_io_log.c
 * @brief Run logging implementation for the core clustering engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cluster_io.h"
#include "common.h"

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
    {
        out_dir = strdup(config->output.user_outdir);
    }
    else
    {
        out_dir = create_output_dir_name(config->input.fits_filename);
    }

    if (!out_dir)
    {
        return;
    }

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
        fprintf(f, "PARAM_ENTROPY: %d\n", config->optim.entropy_mode);
        fprintf(f, "PARAM_ENTROPY_FAST: %d\n", config->optim.entropy_fast_mode);
        fprintf(f, "PARAM_ENTROPY_GATE: %f\n", config->optim.entropy_gate_bits);
        fprintf(f, "PARAM_ENTROPY_FIRST_GATE: %f\n", config->optim.entropy_first_gate_bits);
        fprintf(f, "PARAM_ENTROPY_MAX_TARGETS: %d\n", config->optim.entropy_max_targets);
        fprintf(f, "PARAM_ENTROPY_MIN_PROB: %f\n", config->optim.entropy_min_prob);

        if (config->output.output_dcc)
        {
            fprintf(f, "OUTPUT_FILE: %s/dcc.txt\n", out_dir);
        }
        if (config->output.output_tm)
        {
            fprintf(f, "OUTPUT_FILE: %s/transition_matrix.txt\n", out_dir);
        }
        if (config->output.output_anchors)
        {
            fprintf(f, "OUTPUT_FILE: %s/anchors.txt\n", out_dir);
        }
        if (config->output.output_counts)
        {
            fprintf(f, "OUTPUT_FILE: %s/cluster_counts.txt\n", out_dir);
        }
        if (config->output.output_membership)
        {
            fprintf(f, "OUTPUT_FILE: %s/frame_membership.txt\n", out_dir);
        }

        if (config->output.output_clustered)
        {
            const char *base_name_only = strrchr(config->input.fits_filename, '/');
            if (base_name_only)
            {
                base_name_only++;
            }
            else
            {
                base_name_only = config->input.fits_filename;
            }
            char *temp_base = strdup(base_name_only);
            char *ext = strrchr(temp_base, '.');
            if (ext && strcmp(ext, ".txt") == 0)
            {
                *ext = '\0';
            }
            fprintf(f, "CLUSTERED_FILE: %s/%s.clustered.txt\n", out_dir, temp_base);
            free(temp_base);
        }

        fprintf(f, "STATS_CLUSTERS: %d\n", state->num_clusters);
        fprintf(f, "STATS_FRAMES: %ld\n", state->telemetry.total_frames_processed);
        fprintf(f, "STATS_DISTS: %ld\n", state->telemetry.framedist_calls);
        fprintf(f, "STATS_DISTS_SAMPLE: %ld\n", state->telemetry.framedist_calls_sample);
        fprintf(f, "STATS_DISTS_INTERCLUSTER: %ld\n",
                state->telemetry.framedist_calls_intercluster);
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
            if (state->telemetry.cluster_query_counts
                && state->telemetry.cluster_query_counts[k] > 0)
            {
                fprintf(f, "%d %ld\n", k, state->telemetry.cluster_query_counts[k]);
            }
        }
        fprintf(f, "STATS_CLUSTER_QUERIES_END\n");

        fclose(f);
        printf("Log written to %s\n", log_path);
    } // if (f)

    free(out_dir);
}

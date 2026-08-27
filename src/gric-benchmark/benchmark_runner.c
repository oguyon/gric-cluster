/**
 * @file benchmark_runner.c
 * @brief Benchmark test suite runner and individual pattern execution.
 */

#include "benchmark.h"
#include <math.h>
#include <string.h>
#include <unistd.h>

static int generate_pattern_data(
    const char           *pattern,
    const char           *effective_type,
    const BenchmarkConfig *config,
    const BenchmarkPaths  *paths,
    const char           *txt_file,
    const char           *fits_file,
    const char           *nbsample_str)
{
    int is_balls = (strncmp(pattern, "balls", 5) == 0 ||
                    strstr(pattern, "bouncing") != NULL);

    if (strcmp(effective_type, "fits") == 0 || is_balls)
    {
        if (config->reuse_mp4 && access(fits_file, F_OK) == 0)
        {
            printf("Re-using existing data file: %s\n", fits_file);
            return 0;
        }

        if (access(paths->genballs_path, X_OK) != 0)
        {
            fprintf(stderr,
                    "Error: %s not found or executable. Skipping pattern '%s'.\n",
                    paths->genballs_path, pattern);
            return -1;
        }

        printf("Generating bouncing balls data for pattern: %s\n", pattern);
        char *gen_args[16];
        int gen_argc = 0;
        gen_args[gen_argc++] = (char *)paths->genballs_path;

        if (strcmp(pattern, "balls_single") == 0)
        {
            gen_args[gen_argc++] = "-n";
            gen_args[gen_argc++] = "1";
            gen_args[gen_argc++] = "-r";
            gen_args[gen_argc++] = "5.0";
            gen_args[gen_argc++] = "-W";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-H";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-f";
            gen_args[gen_argc++] = (char *)nbsample_str;
            gen_args[gen_argc++] = "-s";
            gen_args[gen_argc++] = "42";
        }
        else if (strcmp(pattern, "balls_coll") == 0 ||
                 strcmp(pattern, "balls_3_collisions") == 0)
        {
            gen_args[gen_argc++] = "-n";
            gen_args[gen_argc++] = "3";
            gen_args[gen_argc++] = "-r";
            gen_args[gen_argc++] = "5.0";
            gen_args[gen_argc++] = "-W";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-H";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-f";
            gen_args[gen_argc++] = (char *)nbsample_str;
            gen_args[gen_argc++] = "-s";
            gen_args[gen_argc++] = "42";
        }
        else if (strcmp(pattern, "balls_nocoll") == 0 ||
                 strcmp(pattern, "balls_3_nocoll") == 0)
        {
            gen_args[gen_argc++] = "-n";
            gen_args[gen_argc++] = "3";
            gen_args[gen_argc++] = "-r";
            gen_args[gen_argc++] = "5.0";
            gen_args[gen_argc++] = "-W";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-H";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-f";
            gen_args[gen_argc++] = (char *)nbsample_str;
            gen_args[gen_argc++] = "-s";
            gen_args[gen_argc++] = "42";
            gen_args[gen_argc++] = "-no-ball-collision";
        }
        else
        {
            gen_args[gen_argc++] = "-n";
            gen_args[gen_argc++] = "1";
            gen_args[gen_argc++] = "-r";
            gen_args[gen_argc++] = "5.0";
            gen_args[gen_argc++] = "-W";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-H";
            gen_args[gen_argc++] = "32";
            gen_args[gen_argc++] = "-f";
            gen_args[gen_argc++] = (char *)nbsample_str;
            gen_args[gen_argc++] = "-s";
            gen_args[gen_argc++] = "42";
        }
        gen_args[gen_argc++] = (char *)fits_file;
        gen_args[gen_argc] = NULL;

        int gen_status = run_command_redirect(paths->genballs_path, gen_args, "/dev/null");
        if (gen_status != 0)
        {
            fprintf(stderr, "Error: FITS generation failed (exit code %d)\n", gen_status);
            return -1;
        }
    }
    else
    {
        if (config->reuse_mp4 && access(txt_file, F_OK) == 0)
        {
            printf("Re-using existing data file: %s\n", txt_file);
            return 0;
        }

        printf("Generating data for pattern: %s\n", pattern);
        char *gen_args[16];
        int gen_argc = 0;
        gen_args[gen_argc++] = (char *)paths->mkseq_path;
        gen_args[gen_argc++] = (char *)nbsample_str;
        gen_args[gen_argc++] = (char *)txt_file;

        if (strcmp(pattern, "2Dspiral") == 0)
        {
            gen_args[gen_argc++] = "2Dspiral";
        }
        else if (strcmp(pattern, "2Dcircle-shuffle") == 0)
        {
            gen_args[gen_argc++] = "2Dcircle";
            gen_args[gen_argc++] = "-shuffle";
        }
        else if (strcmp(pattern, "2Dspiral-shuffle") == 0)
        {
            gen_args[gen_argc++] = "2Dspiral";
            gen_args[gen_argc++] = "-shuffle";
        }
        else if (strcmp(pattern, "2Drand") == 0)
        {
            gen_args[gen_argc++] = "2Drand";
        }
        else if (strcmp(pattern, "3Drand") == 0)
        {
            gen_args[gen_argc++] = "3Drand";
        }
        else if (strcmp(pattern, "2DcircleP10n") == 0)
        {
            gen_args[gen_argc++] = "2Dcircle10";
            gen_args[gen_argc++] = "-noise";
            gen_args[gen_argc++] = "0.04";
        }
        else if (strcmp(pattern, "3Dspiral") == 0)
        {
            gen_args[gen_argc++] = "3Dspiral";
        }
        else if (strcmp(pattern, "3Dstar") == 0)
        {
            gen_args[gen_argc++] = "3Dstar30";
            gen_args[gen_argc++] = "-noise";
            gen_args[gen_argc++] = "0.02";
            gen_args[gen_argc++] = "-shuffle";
        }
        else if (strcmp(pattern, "3Dconcentric") == 0)
        {
            gen_args[gen_argc++] = "3Dconcentric5";
            gen_args[gen_argc++] = "-noise";
            gen_args[gen_argc++] = "0.02";
            gen_args[gen_argc++] = "-shuffle";
        }
        else if (strcmp(pattern, "5Dtree") == 0)
        {
            gen_args[gen_argc++] = "5Dtree";
            gen_args[gen_argc++] = "-noise";
            gen_args[gen_argc++] = "0.02";
            gen_args[gen_argc++] = "-shuffle";
        }
        else if (strcmp(pattern, "3Dconcentric_dense") == 0)
        {
            gen_args[gen_argc++] = "3Dconcentric_dense10";
            gen_args[gen_argc++] = "-noise";
            gen_args[gen_argc++] = "0.05";
            gen_args[gen_argc++] = "-shuffle";
        }
        else
        {
            fprintf(stderr, "Error: Unknown pattern '%s'\n", pattern);
            return -1;
        }
        gen_args[gen_argc] = NULL;

        int gen_status = run_command_redirect(paths->mkseq_path, gen_args, "/dev/null");
        if (gen_status != 0)
        {
            fprintf(stderr, "Error: Data generation failed (exit code %d)\n", gen_status);
            return -1;
        }
    }
    return 0;
}

static void determine_radius_limit(
    const char           *pattern,
    const char           *effective_type,
    int                   is_balls,
    int                   has_tiles_opt,
    const BenchmarkConfig *config,
    char                 *out_rlim,
    size_t                out_rlim_size)
{
    if (strcmp(effective_type, "mp4") == 0 || strcmp(effective_type, "stream") == 0)
    {
        if (!config->rlim_set)
        {
            snprintf(out_rlim, out_rlim_size, "1000.0");
        }
        else
        {
            snprintf(out_rlim, out_rlim_size, "%s", config->rlim);
        }
    }
    else if (strcmp(effective_type, "fits") == 0 || is_balls)
    {
        if (!config->rlim_set)
        {
            if (strcmp(pattern, "balls_single") == 0 ||
                strcmp(pattern, "balls_1") == 0)
            {
                snprintf(out_rlim, out_rlim_size, "%s", has_tiles_opt ? "1.5" : "1.5");
            }
            else
            {
                snprintf(out_rlim, out_rlim_size, "%s", has_tiles_opt ? "7.0" : "7.0");
            }
        }
        else
        {
            snprintf(out_rlim, out_rlim_size, "%s", config->rlim);
        }
    }
    else
    {
        if (!config->rlim_set && strcmp(pattern, "3Dconcentric_dense") == 0)
        {
            snprintf(out_rlim, out_rlim_size, "0.40");
        }
        else if (!config->rlim_set && strcmp(pattern, "3Dspiral") == 0)
        {
            snprintf(out_rlim, out_rlim_size, "0.02");
        }
        else if (!config->rlim_set && (strcmp(pattern, "3Drand") == 0 ||
                                       strcmp(pattern, "3Dconcentric") == 0 ||
                                       strcmp(pattern, "5Dtree") == 0))
        {
            snprintf(out_rlim, out_rlim_size, "0.20");
        }
        else
        {
            snprintf(out_rlim, out_rlim_size, "%s", config->rlim);
        }
    }
    out_rlim[out_rlim_size - 1] = '\0';
}

int run_benchmark_suite(
    const BenchmarkConfig *config,
    const BenchmarkPaths  *paths,
    TestResult           **out_results,
    int                   *out_result_count)
{
    char nbsample_str[32];
    char maxcl_str[32];
    char maxim_str[32];
    snprintf(nbsample_str, sizeof(nbsample_str), "%d", config->nsamples);
    snprintf(maxcl_str, sizeof(maxcl_str), "%d", config->maxcl);
    snprintf(maxim_str, sizeof(maxim_str), "%d", config->maxim);

    TestResult *results = calloc((size_t)config->pattern_count, sizeof(TestResult));
    int result_count = 0;

    for (int ii = 0; ii < config->pattern_count; ii++)
    {
        const char *pattern = config->patterns[ii];
        int is_entropy = config->use_entropy;
        int has_tiles_opt = 0;
        for (int jj = 0; jj < config->extra_options_count; jj++)
        {
            if (config->extra_options[jj] != NULL)
            {
                if (strstr(config->extra_options[jj], "-entropy") != NULL)
                {
                    is_entropy = 1;
                }
                if (strstr(config->extra_options[jj], "-tiles") != NULL)
                {
                    has_tiles_opt = 1;
                }
            }
        }

        int is_balls = (strncmp(pattern, "balls", 5) == 0 ||
                        strstr(pattern, "bouncing") != NULL);
        const char *effective_type = (is_balls || strcmp(config->type, "fits") == 0)
                                     ? "fits" : config->type;

        printf("========================================================\n");
        printf("Benchmark: Pattern=%s Type=%s Algo=%s\n",
               pattern, effective_type, is_entropy ? "gric-entropy" : "gric-greedy");

        char txt_file[512];
        char fits_file[512];
        snprintf(txt_file, sizeof(txt_file), "%s%s.txt", paths->read_prefix, pattern);
        snprintf(fits_file, sizeof(fits_file), "%s%s.fits", paths->read_prefix, pattern);

        if (generate_pattern_data(pattern, effective_type, config, paths,
                                  txt_file, fits_file, nbsample_str) != 0)
        {
            continue;
        }

        char input_file[512];
        if (strcmp(effective_type, "fits") == 0)
        {
            strcpy(input_file, fits_file);
        }
        else if (strcmp(effective_type, "txt") == 0)
        {
            strcpy(input_file, txt_file);
        }
        else if (strcmp(effective_type, "mp4") == 0)
        {
            snprintf(input_file, sizeof(input_file), "%s%s.mp4", paths->read_prefix, pattern);
            int skip_vid = 0;
            if (config->reuse_mp4 && access(input_file, F_OK) == 0)
            {
                printf("Re-using existing video file: %s\n", input_file);
                skip_vid = 1;
            }

            if (!skip_vid)
            {
                printf("Converting %s to %s...\n", txt_file, input_file);
                char *vid_args[] =
                {
                    (char *)paths->txt2mp4_path,
                    "64",
                    "0.1",
                    txt_file,
                    input_file,
                    "0.0",
                    nbsample_str,
                    NULL
                };
                int vid_status = run_command_redirect(paths->txt2mp4_path, vid_args, "/dev/null");
                if (vid_status != 0)
                {
                    fprintf(stderr, "Error: Video conversion failed (exit code %d)\n", vid_status);
                    continue;
                }
            }
        }
        else if (strcmp(effective_type, "stream") == 0)
        {
            strcpy(input_file, pattern);
        }

        char cur_rlim[32];
        determine_radius_limit(pattern, effective_type, is_balls, has_tiles_opt,
                               config, cur_rlim, sizeof(cur_rlim));

        char log_file[512];
        snprintf(log_file, sizeof(log_file),
                 "%sbenchmark_out/%s_%s_gric.log",
                 paths->write_prefix, pattern, effective_type);

        char out_dir[512];
        snprintf(out_dir, sizeof(out_dir), "%s%s.cluster.out", paths->write_prefix, pattern);

        printf("Running gric-cluster on %s (rlim=%s)...\n", input_file, cur_rlim);

        char *cluster_args[256];
        int cluster_argc = 0;

        cluster_args[cluster_argc++] = "/usr/bin/time";
        cluster_args[cluster_argc++] = "-v";
        cluster_args[cluster_argc++] = (char *)paths->rnuc_path;
        cluster_args[cluster_argc++] = cur_rlim;
        cluster_args[cluster_argc++] = "-maxcl";
        cluster_args[cluster_argc++] = maxcl_str;
        cluster_args[cluster_argc++] = "-maxim";
        cluster_args[cluster_argc++] = maxim_str;
        cluster_args[cluster_argc++] = "-outdir";
        cluster_args[cluster_argc++] = out_dir;
        cluster_args[cluster_argc++] = "-clustered";
        if (config->use_entropy)
        {
            cluster_args[cluster_argc++] = "-entropy";
        }

        if ((strcmp(effective_type, "fits") == 0 || is_balls) && !has_tiles_opt)
        {
            cluster_args[cluster_argc++] = "-tiles";
            cluster_args[cluster_argc++] = "2x2";
            cluster_args[cluster_argc++] = "-ncpu";
            cluster_args[cluster_argc++] = "4";
        }

        int first_extra_arg_idx = cluster_argc;
        for (int jj = 0; jj < config->extra_options_count; jj++)
        {
            split_args(config->extra_options[jj], cluster_args, &cluster_argc, 256 - 3);
        }

        if (strcmp(effective_type, "stream") == 0)
        {
            cluster_args[cluster_argc++] = "-stream";
        }

        cluster_args[cluster_argc++] = input_file;
        cluster_args[cluster_argc] = NULL;

        int run_status = run_command_redirect("/usr/bin/time", cluster_args, log_file);
        if (run_status != 0)
        {
            fprintf(stderr, "Warning: gric-cluster exited with status %d\n", run_status);
        }

        for (int jj = first_extra_arg_idx; jj < cluster_argc; jj++)
        {
            if (cluster_args[jj] != NULL &&
                strcmp(cluster_args[jj], "-stream") != 0 &&
                cluster_args[jj] != input_file)
            {
                free(cluster_args[jj]);
            }
        }

        if (strcmp(effective_type, "txt") == 0 && access(paths->clplot_path, X_OK) == 0)
        {
            char cluster_log[1024];
            snprintf(cluster_log, sizeof(cluster_log), "%s/cluster_run.log", out_dir);
            char *plot_args[] = {(char *)paths->clplot_path, input_file, cluster_log, NULL};
            run_command_redirect(paths->clplot_path, plot_args, "/dev/null");
        }

        char m_time[64];
        char m_dists[64];
        char m_dists_sample[64];
        char m_dists_inter[64];
        char m_clusters[64];
        char m_mem[64];
        parse_metrics(
            log_file, m_time, m_dists,
            m_dists_sample, m_dists_inter,
            m_clusters, m_mem);

        printf("Result: Time=%sms, Clusters=%s, Mem=%sKB\n",
               m_time, m_clusters, m_mem);

        double total_dists = atof(m_dists);
        double sample_dists = atof(m_dists_sample);
        double inter_dists = atof(m_dists_inter);

        double avg_dists = (config->nsamples > 0) ? (total_dists / config->nsamples) : 0.0;
        double avg_sample_dists = (config->nsamples > 0) ? (sample_dists / config->nsamples) : 0.0;
        double avg_inter_dists = (config->nsamples > 0) ? (inter_dists / config->nsamples) : 0.0;

        if (strcmp(m_dists_sample, "N/A") != 0)
        {
            printf("%sDistances (sum): %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists, avg_dists, ANSI_COLOR_RESET);
            printf("%s  -> Sample-to-cluster: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN,
                   m_dists_sample, avg_sample_dists, ANSI_COLOR_RESET);
            printf("%s  -> Cluster-to-cluster: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN,
                   m_dists_inter, avg_inter_dists, ANSI_COLOR_RESET);
        }
        else
        {
            printf("%sDistances: %s%s (%.3f per sample)%s\n",
                   ANSI_BOLD_CYAN, ANSI_BOLD_GREEN, m_dists, avg_dists, ANSI_COLOR_RESET);
        }

        char dist_str[256];
        if (strcmp(m_dists_sample, "N/A") != 0)
        {
            snprintf(dist_str, sizeof(dist_str),
                     "%s (S:%s, C:%s)",
                     m_dists, m_dists_sample,
                     m_dists_inter);
        }
        else
        {
            snprintf(dist_str, sizeof(dist_str), "%s", m_dists);
        }

        append_summary_row(paths->summary_path, pattern, effective_type,
                           is_entropy ? "gric-entropy" : "gric-greedy",
                           config->nsamples, m_time, dist_str, m_clusters, m_mem);

        if (results != NULL && result_count < config->pattern_count)
        {
            TestResult *r = &results[result_count];
            snprintf(r->pattern, sizeof(r->pattern), "%s", pattern);
            snprintf(r->algo, sizeof(r->algo), "%s", is_entropy ? "entropy" : "greedy");
            snprintf(r->time_ms, sizeof(r->time_ms), "%s", m_time);
            r->dist_total = total_dists;
            r->dist_sample = sample_dists;
            r->dist_inter = inter_dists;
            r->avg_dist = avg_dists;
            r->clusters = atoi(m_clusters);
            snprintf(r->mem_kb, sizeof(r->mem_kb), "%s", m_mem);
            r->nsamples = config->nsamples;
            result_count++;
        }
    }

    *out_results = results;
    *out_result_count = result_count;
    return 0;
} // run_benchmark_suite

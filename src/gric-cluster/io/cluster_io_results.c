/**
 * @file cluster_io_results.c
 * @brief Results serialization for the core clustering engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

#ifdef USE_PNG
#include "png_io.h"
#endif

#include "cluster_io.h"
#include "common.h"
#include "frameread.h"

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
                    {
                        fprintf(dcc_out, "%d %d %.6f\n", i, j, d);
                    }
                }
            } // for (int i = 0; ...)
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
            } // for (int i = 0; ...)
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
                    {
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    }
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            } // if (afptr)
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
                fits_write_pix(afptr, TDOUBLE, fpixel, nelements,
                               state->clusters[i].anchor.data, &status);
            }
            fits_close_file(afptr, &status);
#else
            fprintf(stderr,
                    "Warning: FITS output requested but not compiled in. Saving as ASCII.\n");
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");

            if (afptr)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    for (long k = 0; k < nelements; k++)
                    {
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    }
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            } // if (afptr)
#endif
        }
    } // if (config->output.output_anchors)

    // Cluster Counts
    int *cluster_counts = (int *)calloc(state->num_clusters, sizeof(int));

    for (long i = 0; i < state->telemetry.total_frames_processed; i++)
    {
        if (state->assignments[i] >= 0 && state->assignments[i] < state->num_clusters)
        {
            cluster_counts[state->assignments[i]]++;
        }
    }

    if (config->output.output_counts)
    {
        printf("Writing cluster_counts.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/cluster_counts.txt", out_dir);
        FILE *count_out = fopen(out_path, "w");

        if (count_out)
        {
            for (int c = 0; c < state->num_clusters; c++)
            {
                fprintf(count_out, "Cluster %d: %d frames\n", c, cluster_counts[c]);
            }
            fclose(count_out);
        } // if (count_out)
    }

    // Average buffer
    double *avg_buffer = NULL;

    if (config->output.average_mode)
    {
        avg_buffer = (double *)calloc(nelements, sizeof(double));
    }

    int active_cluster_count = 0;

    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] > 0)
        {
            active_cluster_count++;
        }
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
            {
                continue;
            }

            if (config->output.output_clusters)
            {
                char cluster_dir[1024];
                snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                safe_mkdir(cluster_dir);
            }

            if (config->output.average_mode)
            {
                for (long k = 0; k < nelements; k++)
                {
                    avg_buffer[k] = 0.0;
                }
            }

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
                            snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d",
                                     out_dir, c);
                            snprintf(out_path, sizeof(out_path), "%s/frame%05ld.png",
                                     cluster_dir, f);
                            write_png_frame(out_path, fr->data, width, height);
                        }
                        if (config->output.average_mode)
                        {
                            for (long k = 0; k < nelements; k++)
                            {
                                avg_buffer[k] += fr->data[k];
                            }
                        }
                        free_frame(fr);
                    }
                }
            } // for (long f = 0; ...)

            if (config->output.average_mode)
            {
                for (long k = 0; k < nelements; k++)
                {
                    avg_buffer[k] /= cluster_counts[c];
                }
                snprintf(out_path, sizeof(out_path), "%s/average_%04d.png", out_dir, c);
                write_png_frame(out_path, avg_buffer, width, height);
            }
        } // for (int c = 0; ...)
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
                    {
                        fprintf(avg_file, "0.0 ");
                    }
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
            {
                for (long k = 0; k < nelements; k++)
                {
                    avg_buffer[k] = 0.0;
                }
            }

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
                            {
                                fprintf(cfptr, "%f ", fr->data[k]);
                            }
                            if (config->output.average_mode)
                            {
                                avg_buffer[k] += fr->data[k];
                            }
                        }
                        if (cfptr)
                        {
                            fprintf(cfptr, "\n");
                        }
                        free_frame(fr);
                    }
                }
            } // for (long f = 0; ...)

            if (cfptr)
            {
                fclose(cfptr);
            }

            if (avg_file)
            {
                for (long k = 0; k < nelements; k++)
                {
                    fprintf(avg_file, "%f ", avg_buffer[k] / cluster_counts[c]);
                }
                fprintf(avg_file, "\n");
            }
        } // for (int c = 0; ...)

        if (avg_file)
        {
            fclose(avg_file);
        }
    } // else if (is_ascii_input_mode() ...)
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
            {
                continue;
            }

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
            {
                for (long k = 0; k < nelements; k++)
                {
                    avg_buffer[k] = 0.0;
                }
            }

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
                        {
                            for (long k = 0; k < nelements; k++)
                            {
                                avg_buffer[k] += fr->data[k];
                            }
                        }
                        free_frame(fr);
                        fr_count++;
                    }
                }
            } // for (long f = 0; ...)

            if (cfptr)
            {
                fits_close_file(cfptr, &status);
            }

            if (config->output.average_mode && avg_ptr)
            {
                for (long k = 0; k < nelements; k++)
                {
                    avg_buffer[k] /= cluster_counts[c];
                }
                long fpixel[3] = {1, 1, c + 1};
                fits_write_pix(avg_ptr, TDOUBLE, fpixel, nelements, avg_buffer, &status);
            }
        } // for (int c = 0; ...)

        if (avg_ptr)
        {
            fits_close_file(avg_ptr, &status);
        }
#else
        // Fallback ASCII logic if FITS disabled but we reached here
        // (Similar to block above)
#endif
    } // else

    if (avg_buffer)
    {
        free(avg_buffer);
    }

    if (config->output.output_clustered)
    {
        printf("Writing clustered output file\n");
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

        char *clustered_fname =
            (char *)malloc(strlen(out_dir) + strlen(temp_base) + 30);

        if (clustered_fname)
        {
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
                fprintf(clustered_out,
                        "# Total Distance Computations %ld\n",
                        state->telemetry.framedist_calls);
                fprintf(clustered_out, "# Clusters Pruned %ld\n",
                        state->telemetry.clusters_pruned);

                double avg_dist = 0.0;
                if (state->telemetry.total_frames_processed > 0)
                {
                    avg_dist = (double)state->telemetry.framedist_calls /
                               state->telemetry.total_frames_processed;
                }
                fprintf(clustered_out, "# Avg Dist/Frame %.2f\n", avg_dist);

                if (state->telemetry.pruned_fraction_sum && state->telemetry.step_counts)
                {
                    for (int k = 0; k < state->telemetry.max_steps_recorded; k++)
                    {
                        if (state->telemetry.step_counts[k] > 0)
                        {
                            fprintf(clustered_out,
                                    "# Pruning Step %d: %.4f\n",
                                    k,
                                    state->telemetry.pruned_fraction_sum[k] /
                                    state->telemetry.step_counts[k]);
                        }
                        else if (k > 0 && state->telemetry.step_counts[k] == 0)
                        {
                            break;
                        }
                    } // for (int k = 0; ...)
                }

                int next_new_cluster = 0;

                for (long i = 0; i < state->telemetry.total_frames_processed; i++)
                {
                    int assigned = state->assignments[i];

                    if (assigned == next_new_cluster)
                    {
                        fprintf(clustered_out, "# NEWCLUSTER %d %ld ", assigned, i);
                        for (long k = 0; k < nelements; k++)
                        {
                            fprintf(clustered_out, "%f ",
                                    state->clusters[assigned].anchor.data[k]);
                        }
                        fprintf(clustered_out, "\n");
                        next_new_cluster++;
                    }

                    Frame *fr = getframe_at(i);

                    if (fr)
                    {
                        fprintf(clustered_out, "%ld %d ", i, assigned);
                        for (long k = 0; k < nelements; k++)
                        {
                            fprintf(clustered_out, "%f ", fr->data[k]);
                        }
                        fprintf(clustered_out, "\n");
                        free_frame(fr);
                    }
                } // for (long i = 0; ...)
                fclose(clustered_out);
            } // if (clustered_out)
            free(clustered_fname);
        } // if (clustered_fname)
        else
        {
            free(temp_base);
        }
    } // if (config->output.output_clustered)

    free(cluster_counts);
    free(out_dir);
}

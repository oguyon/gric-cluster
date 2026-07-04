/**
 * @file plot_log_parser.c
 * @brief Parser implementation for coordinate clustering log and membership files.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plot_internal.h"

/**
 * @brief Parse the coordinate clustering log and membership files.
 *
 * This function opens the log file to extract clustering parameters and statistics.
 * It also opens the membership and points files to load anchors and compute the number
 * of samples per cluster.
 *
 * @param points_filename Path to the points coordinate file.
 * @param log_filename    Path to the clustering log file.
 * @param data            Pointer to the PlotData structure to populate.
 *
 * @return 0 on success, 1 on failure.
 */
int parse_input_data(
    const char *points_filename,
    const char *log_filename,
    PlotData   *data)
{
    data->rlim = 0.0;
    data->dprob = 0.01;
    data->maxcl = 1000;
    data->maxim = 100000;
    data->gprob = 0;
    data->te4 = 0;
    data->te5 = 0;
    data->output_dir[0] = '\0';
    data->dcc_filename[0] = '\0';
    data->total_frames = 0;
    data->total_clusters = 0;
    data->total_dists = 0;
    data->num_anchors = 0;
    data->num_stats = 0;

    memset(data->hist_data, 0, sizeof(data->hist_data));
    memset(data->cluster_query_counts, 0, sizeof(data->cluster_query_counts));
    memset(data->samples_per_cluster, 0, sizeof(data->samples_per_cluster));

    printf("Reading log file: %s\n", log_filename);

    /* Open and parse the log file */
    {
        FILE *flog = fopen(log_filename, "r");
        if (flog == NULL)
        {
            fprintf(stderr, "Error: Could not open log file %s\n", log_filename);
            return 1;
        }

        int hist_parsing = 0;
        int queries_parsing = 0;
        char line[4096];

        while (fgets(line, sizeof(line), flog) != NULL)
        {
            if (hist_parsing != 0)
            {
                if (strncmp(line, "STATS_DIST_HIST_END", 19) == 0)
                {
                    hist_parsing = 0;
                }
                else
                {
                    int k;
                    long c, p;
                    if (sscanf(line, "%d %ld %ld", &k, &c, &p) >= 2 && k >= 0 && k < 10000)
                    {
                        data->hist_data[k] = c;
                    }
                }
                continue;
            } // if (hist_parsing != 0)

            if (queries_parsing != 0)
            {
                if (strncmp(line, "STATS_CLUSTER_QUERIES_END", 25) == 0)
                {
                    queries_parsing = 0;
                }
                else
                {
                    int k;
                    long q;
                    if (sscanf(line, "%d %ld", &k, &q) == 2 && k >= 0 && k < 10000)
                    {
                        data->cluster_query_counts[k] = q;
                    }
                }
                continue;
            } // if (queries_parsing != 0)

            if (strncmp(line, "OUTPUT_DIR: ", 12) == 0)
            {
                strncpy(data->output_dir, line + 12, sizeof(data->output_dir) - 1);
                data->output_dir[sizeof(data->output_dir) - 1] = '\0';
                size_t len = strlen(data->output_dir);
                if (len > 0 && data->output_dir[len - 1] == '\n')
                {
                    data->output_dir[len - 1] = '\0';
                }
            }
            else if (strncmp(line, "OUTPUT_FILE: ", 13) == 0)
            {
                if (strstr(line, "dcc.txt") != NULL)
                {
                    strncpy(data->dcc_filename, line + 13, sizeof(data->dcc_filename) - 1);
                    data->dcc_filename[sizeof(data->dcc_filename) - 1] = '\0';
                    size_t len = strlen(data->dcc_filename);
                    if (len > 0 && data->dcc_filename[len - 1] == '\n')
                    {
                        data->dcc_filename[len - 1] = '\0';
                    }
                }
            }
            else if (strncmp(line, "PARAM_RLIM: ", 12) == 0)
            {
                sscanf(line + 12, "%lf", &data->rlim);
            }
            else if (strncmp(line, "PARAM_DPROB: ", 13) == 0)
            {
                sscanf(line + 13, "%lf", &data->dprob);
            }
            else if (strncmp(line, "PARAM_MAXCL: ", 13) == 0)
            {
                sscanf(line + 13, "%d", &data->maxcl);
            }
            else if (strncmp(line, "PARAM_MAXIM: ", 13) == 0)
            {
                sscanf(line + 13, "%ld", &data->maxim);
            }
            else if (strncmp(line, "PARAM_GPROB: ", 13) == 0)
            {
                sscanf(line + 13, "%d", &data->gprob);
            }
            else if (strncmp(line, "PARAM_TE4: ", 11) == 0)
            {
                sscanf(line + 11, "%d", &data->te4);
            }
            else if (strncmp(line, "PARAM_TE5: ", 11) == 0)
            {
                sscanf(line + 11, "%d", &data->te5);
            }
            else if (strncmp(line, "STATS_DIST_HIST_START", 21) == 0)
            {
                hist_parsing = 1;
            }
            else if (strncmp(line, "STATS_CLUSTER_QUERIES_START", 27) == 0)
            {
                queries_parsing = 1;
            }
            else if (strncmp(line, "STATS_", 6) == 0)
            {
                char *k = line + 6;
                char *v = strchr(k, ':');
                if (v != NULL)
                {
                    *v = '\0';
                    v++;
                    if (strcmp(k, "CLUSTERS") == 0)
                    {
                        data->total_clusters = atol(v);
                    }
                    else if (strcmp(k, "FRAMES") == 0)
                    {
                        data->total_frames = atol(v);
                    }
                    else if (strcmp(k, "DISTS") == 0)
                    {
                        data->total_dists = atol(v);
                    }
                }
            }
        } // while (fgets(line, sizeof(line), flog) != NULL)
        fclose(flog);
    } // Open and parse the log file

    printf("Log loaded: %ld frames, %ld clusters\n", data->total_frames, data->total_clusters);

    /* Construct the stats lines */
    {
        snprintf(data->stats[0].text, 255, "%ld fr -> %ld cl (%ld dist)",
                 data->total_frames, data->total_clusters, data->total_dists);
        char p_str[1024];
        int po = snprintf(p_str, 1023, "Params: R=%.3f", data->rlim);
        if (data->dprob != 0.01)
        {
            po += snprintf(p_str + po, 1023 - po, ", dprob=%.3f", data->dprob);
        }
        if (data->gprob != 0)
        {
            po += snprintf(p_str + po, 1023 - po, ", gprob=ON");
        }
        if (data->te4 != 0)
        {
            po += snprintf(p_str + po, 1023 - po, ", te4=ON");
        }
        if (data->te5 != 0)
        {
            po += snprintf(p_str + po, 1023 - po, ", te5=ON");
        }
        strncpy(data->stats[1].text, p_str, 255);
        data->stats[1].text[255] = '\0';
        data->num_stats = 2;
    } // Construct the stats lines

    /* Build the membership file name */
    char memb_filename[8192];
    if (data->output_dir[0] != '\0')
    {
        snprintf(memb_filename, sizeof(memb_filename), "%s/frame_membership.txt",
                 data->output_dir);
    }
    else
    {
        strcpy(memb_filename, "frame_membership.txt");
    }

    printf("Reading points: %s\n", points_filename);
    printf("Reading membership: %s\n", memb_filename);

    /* Process points and membership to populate anchors and samples per cluster */
    {
        FILE *f_pts = fopen(points_filename, "r");
        if (f_pts == NULL)
        {
            fprintf(stderr, "Error: Could not open points file %s\n", points_filename);
            return 1;
        }

        FILE *f_memb = fopen(memb_filename, "r");
        if (f_memb == NULL)
        {
            fprintf(stderr, "Error: Could not open membership file %s\n", memb_filename);
            fclose(f_pts);
            return 1;
        }

        char *cluster_seen = calloc(10000, 1);
        if (cluster_seen == NULL)
        {
            fprintf(stderr, "Error: Memory allocation failed for cluster_seen\n");
            fclose(f_pts);
            fclose(f_memb);
            return 1;
        }

        char lp[4096];
        char lm[1024];
        long frame_count = 0;

        while (fgets(lp, sizeof(lp), f_pts) != NULL && fgets(lm, sizeof(lm), f_memb) != NULL)
        {
            if (lp[0] == '#')
            {
                continue;
            }
            long idx;
            int cid;
            if (sscanf(lm, "%ld %d", &idx, &cid) == 2 && cid >= 0 && cid < 10000)
            {
                data->samples_per_cluster[cid]++;
                if (cluster_seen[cid] == 0)
                {
                    data->anchors[data->num_anchors] = (Anchor){cid, 0.0, 0.0};
                    sscanf(lp, "%lf %lf", &data->anchors[data->num_anchors].x,
                           &data->anchors[data->num_anchors].y);
                    data->num_anchors++;
                    cluster_seen[cid] = 1;
                }
            }
            frame_count++;
            if (frame_count % 10000 == 0)
            {
                printf("\rProcessing frames: %ld / %ld", frame_count, data->total_frames);
                fflush(stdout);
            }
        } // while reading files
        printf("\rProcessing frames: %ld / %ld [DONE]\n", frame_count, data->total_frames);

        free(cluster_seen);
        fclose(f_pts);
        fclose(f_memb);
    } // Process points and membership

    return 0;
}

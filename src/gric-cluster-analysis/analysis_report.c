/**
 * @file analysis_report.c
 * @brief Terminal, ASCII histogram, and JSON reporting for gric-cluster-analysis.
 */

#include "analysis_state.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

void print_ascii_histogram(
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
            int    bar_len = (int)((double)hist[k] * (double)max_width / (double)max_bar_val);
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
    }
} // print_ascii_histogram

void write_text_report(
    FILE          *out,
    AnalysisState *state,
    int            color_enabled)
{
    const char *bc = (color_enabled != 0) ? ansi_bold_cyan : "";
    const char *br = (color_enabled != 0) ? ansi_color_red : "";
    const char *reset = (color_enabled != 0) ? ansi_reset : "";

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
        double avg_calls = (double)state->num_dists / (double)state->num_frames;
        double ratio = (state->num_clusters > 0)
                           ? avg_calls / (double)state->num_clusters
                           : 0.0;
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
        }

        fprintf(out, "Largest Cluster:           %d (%ld frames)\n", largest_id, largest_sz);
        fprintf(out, "Smallest Cluster:          %d (%ld frames)\n", smallest_id, smallest_sz);
        fprintf(out, "Empty/Pruned Clusters:     %d\n", empty_count);
    }

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
                double duty = (double)sz * 100.0 / (double)span;
                fprintf(out, "  %5d | %10ld | %10ld | %10ld | %10ld | %7.1f%%\n",
                        i, sz, birth, death, span, duty);
            }
        }
    }

    if (state->assignments_count > 0 && state->transition_matrix != NULL)
    {
        int n = state->num_clusters;
        fprintf(out, "\n%s--- State Transition Dynamics ---%s\n", bc, reset);

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
        }

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
        }

        fprintf(out, "\nTop Distinct-State Transitions:\n");
        for (int k = 0; k < 5; k++)
        {
            if (top_trans[k].count > 0)
            {
                double pct = (double)top_trans[k].count * 100.0 / (double)state->assignments_count;
                fprintf(out, "  Cluster %3d -> Cluster %3d: %6ld times (%5.2f%% of run)\n",
                        top_trans[k].src, top_trans[k].dst, top_trans[k].count, pct);
            }
        }
    }

    if (state->dcc_matrix != NULL)
    {
        int    n = state->num_clusters;
        double sum = 0.0;
        long   cnt = 0;
        double min_dist = -1.0;
        double max_dist = -1.0;
        int    min_i = -1;
        int    min_j = -1;
        int    max_i = -1;
        int    max_j = -1;

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
        }

        fprintf(out, "\n%s--- Cluster Distance Topology ---%s\n", bc, reset);
        if (cnt > 0)
        {
            fprintf(out, "Average distance: %.5f\n", sum / (double)cnt);
            fprintf(out, "Minimum distance: %.5f (between C%d and C%d)\n",
                    min_dist, min_i, min_j);
            fprintf(out, "Maximum distance: %.5f (between C%d and C%d)\n",
                    max_dist, max_i, max_j);

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
    }
} // write_text_report

void write_json_report(
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

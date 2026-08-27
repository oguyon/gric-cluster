/**
 * @file benchmark_report.c
 * @brief Formatting and reporting of benchmark execution summaries.
 */

#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>

void init_summary_file(
    const char *summary_path)
{
    FILE *sum_fp = fopen(summary_path, "r");
    if (sum_fp == NULL)
    {
        sum_fp = fopen(summary_path, "w");
        if (sum_fp != NULL)
        {
            fprintf(sum_fp,
                    "| Pattern | Type | Algo | Frames | Time (ms) | "
                    "Dist Calls | Clusters | Memory (KB) |\n");
            fprintf(sum_fp, "|---|---|---|---|---|---|---|---|\n");
            fclose(sum_fp);
        }
    }
    else
    {
        fclose(sum_fp);
    }
} // init_summary_file

void append_summary_row(
    const char *summary_path,
    const char *pattern,
    const char *type,
    const char *algo,
    int         nsamples,
    const char *time_ms,
    const char *dists_str,
    const char *clusters_str,
    const char *mem_kb)
{
    FILE *sum_fp = fopen(summary_path, "a");
    if (sum_fp != NULL)
    {
        fprintf(sum_fp,
                "| %s | %s | %s | %d | %s | %s | %s | %s |\n",
                pattern, type, algo, nsamples,
                time_ms, dists_str, clusters_str, mem_kb);
        fclose(sum_fp);
    }
} // append_summary_row

void print_summary_table(
    const TestResult *results,
    int               result_count)
{
    if (results == NULL || result_count <= 0)
    {
        return;
    }

    printf("\n");
    printf("========================================"
           "========================================"
           "================================\n");
    printf("%sSUMMARY%s\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("========================================"
           "========================================"
           "================================\n");

    printf("%s%-20s %-8s %7s %10s %10s %8s "
           "%8s %6s %10s%s\n",
           ANSI_BOLD,
           "Pattern", "Algo", "Frames", "Time(ms)",
           "DistTot", "d/frm",
           "dS/frm", "Clust", "Mem(KB)",
           ANSI_COLOR_RESET);
    printf("-------------------- -------- ------- "
           "---------- ---------- -------- "
           "-------- ------ ----------\n");

    for (int ii = 0; ii < result_count; ii++)
    {
        const TestResult *r = &results[ii];
        printf("%-20s %-8s %7d %10s %10.0f %8.2f "
               "%8.2f %6d %10s\n",
               r->pattern,
               r->algo,
               r->nsamples,
               r->time_ms,
               r->dist_total,
               r->avg_dist,
               (r->nsamples > 0)
                   ? (r->dist_sample / r->nsamples)
                   : 0.0,
               r->clusters,
               r->mem_kb);
    }

    printf("========================================"
           "========================================"
           "================================\n");
} // print_summary_table

/**
 * @file probe_report.c
 * @brief Formatted terminal reporting and export for gric-probe.
 */

#include "probe_report.h"
#include "cli_colors.h"
#include <stdio.h>
#include <string.h>

/**
 * probe_report_print_terminal() - Display formatted dashboard to terminal.
 * @out:     Target output stream (stdout/stderr).
 * @results: Pointer to completed ProbeResults.
 */
void probe_report_print_terminal(
    FILE               *out,
    const ProbeResults *results)
{
    if (out == NULL || results == NULL)
    {
        return;
    }

    const GricProfile *p = &results->profile;

    fprintf(out, "\n%s%s=== GRIC DATASET PROBE REPORT ===%s\n",
            ANSI_BOLD, ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    fprintf(out, "%sTarget:%s     %s\n", ANSI_BOLD, ANSI_COLOR_RESET, p->dataset_path);
    fprintf(out, "%sGeometry:%s   %ld frames, %s (%ld-dim",
            ANSI_BOLD, ANSI_COLOR_RESET, p->num_frames,
            p->is_image ? "2D Image" : "1D Vector", p->dim);
    if (p->is_image)
    {
        fprintf(out, ", %ld x %ld", p->width, p->height);
    }
    fprintf(out, "), %s\n", p->is_double ? "float64 (double)" : "float32 (single)");

    /* Dynamic Range & Diagnostics */
    fprintf(out, "%sRange:%s      [%.4f, %.4f] (span: %.4f)\n",
            ANSI_BOLD, ANSI_COLOR_RESET,
            (double)p->sq8_params.min_val, (double)p->sq8_params.max_val,
            (double)(p->sq8_params.max_val - p->sq8_params.min_val));
    if (results->dead_dims_count > 0)
    {
        fprintf(out, "%s[WARNING]%s   %d coordinates have zero variance (dead pixels/channels)!\n",
                ANSI_COLOR_YELLOW, ANSI_COLOR_RESET, results->dead_dims_count);
    }

    /* Variance Scree Spectrum */
    fprintf(out, "\n%s--- Coordinate Variance Spectrum ---%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    double total_var = 0.0;
    for (long d = 0; d < p->dim; d++)
    {
        total_var += p->var_dim[d];
    }

    double accum = 0.0;
    long top_k = (p->dim < 5) ? p->dim : 5;
    for (long k = 0; k < top_k; k++)
    {
        long dim_idx = p->perm_dim[k];
        double pct = (total_var > 0.0) ? (p->var_dim[dim_idx] / total_var * 100.0) : 0.0;
        accum += pct;

        char bar[32];
        int bar_len = (int)(pct / 4.0);
        if (bar_len > 25) bar_len = 25;
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        bar[bar_len] = '\0';

        fprintf(out, "  Top %ld (dim %4ld): %7.4f (%5.1f%%) |%-25s| (cum: %5.1f%%)\n",
                k + 1, dim_idx, p->var_dim[dim_idx], pct, bar, accum);
    } // for (long k = 0; k < top_k; k++)

    /* Distance Distribution Spectrum */
    fprintf(out, "\n%s--- Distance Percentile Spectrum ---%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    fprintf(out, "  Min:     %8.4f\n", p->dist_min);
    fprintf(out, "  P01:     %8.4f  %s(Ultra-Fine Granularity Reference)%s\n",
            p->dist_p01, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    fprintf(out, "  P03:     %8.4f  %s(Very Fine Granularity Reference)%s\n",
            p->dist_p03, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    fprintf(out, "  P05:     %8.4f  %s(Fine Granularity Reference)%s\n",
            p->dist_p05, ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    fprintf(out, "  P10:     %8.4f  %s(Balanced / Nominal Reference)%s\n",
            p->dist_p10, ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    fprintf(out, "  P25:     %8.4f  %s(Coarse Granularity Reference)%s\n",
            p->dist_p25, ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
    fprintf(out, "  Median:  %8.4f\n", p->dist_p50);
    fprintf(out, "  P90:     %8.4f\n", p->dist_p90);
    fprintf(out, "  Max:     %8.4f\n", p->dist_max);

    /* Temporal Continuity */
    fprintf(out, "\n%s--- Temporal Dynamics ---%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    fprintf(out, "  Continuity Ratio: %.4f (d_seq / d_med)\n", p->continuity_ratio);
    if (p->pred_enabled)
    {
        fprintf(out, "  Status:           %sStrong Temporal Continuity%s (auto-enabling -pred)\n",
                ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
        fprintf(out, "  Recommended H:    %d lookback frames\n", p->pred_h);
    }
    else
    {
        fprintf(out, "  Status:           %sUncorrelated / Static Sequence%s (prediction off)\n",
                ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
    }
    if (p->tm_mixing_coeff > 0.0)
    {
        fprintf(out, "  Transition Mixing:%s %.2f%s (-tm %.2f)\n",
                ANSI_BOLD_GREEN, p->tm_mixing_coeff, ANSI_COLOR_RESET, p->tm_mixing_coeff);
    }
    else
    {
        fprintf(out, "  Transition Mixing: %sOff%s\n",
                ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
    }

    /* Empirical Metric Pruning Efficiency */
    fprintf(out, "\n%s=== METRIC PRUNING EFFICIENCY (rlim = %.4f) ===%s\n",
            ANSI_BOLD, p->rlim_balanced, ANSI_COLOR_RESET);
    fprintf(out, "  * Standard 3-Point (TE3): %s%.1f%%%s pruned (Baseline, cost: ~2 FLOPs)\n",
            ANSI_BOLD_GREEN, p->te3_prune_rate * 100.0, ANSI_COLOR_RESET);

    double cost_dist = 2.0 * (double)p->dim;
    double benefit_te4 = p->te4_marginal_rate * cost_dist - 40.0;
    if (p->te4_enabled)
    {
        fprintf(out,
                "  * 4-Point Pruning (TE4):  %s+%.1f%%%s marginal (%sNet Gain: +%.1f FLOPs%s)\n",
                ANSI_BOLD_GREEN, p->te4_marginal_rate * 100.0, ANSI_COLOR_RESET,
                ANSI_BOLD_GREEN, benefit_te4, ANSI_COLOR_RESET);
    }
    else
    {
        fprintf(out,
                "  * 4-Point Pruning (TE4):  +%.1f%% marginal "
                "(%sNot recommended: net %.1f FLOPs%s)\n",
                p->te4_marginal_rate * 100.0,
                ANSI_COLOR_YELLOW, benefit_te4, ANSI_COLOR_RESET);
    }

    if (p->dim >= 3)
    {
        double benefit_te5 = p->te5_marginal_rate * cost_dist - 120.0;
        if (p->te5_enabled)
        {
            fprintf(out,
                    "  * 5-Point Pruning (TE5):  %s+%.1f%%%s marginal "
                    "(%sNet Gain: +%.1f FLOPs%s)\n",
                    ANSI_BOLD_GREEN, p->te5_marginal_rate * 100.0, ANSI_COLOR_RESET,
                    ANSI_BOLD_GREEN, benefit_te5, ANSI_COLOR_RESET);
        }
        else
        {
            fprintf(out,
                    "  * 5-Point Pruning (TE5):  +%.1f%% marginal (%sNot recommended%s)\n",
                    p->te5_marginal_rate * 100.0,
                    ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
        }
    }
    fprintf(out, "  * Recommended Prune Mode: %s%s%s\n",
            ANSI_BOLD_CYAN,
            p->recommended_prune_mode[0] ? p->recommended_prune_mode : "3P",
            ANSI_COLOR_RESET);
    if (p->entropy_enabled)
    {
        fprintf(out, "  * Entropy Search:         %sON%s (-entropy -entropy_gate %.2f)\n",
                ANSI_BOLD_GREEN, ANSI_COLOR_RESET, p->entropy_gate);
    }
    else
    {
        fprintf(out, "  * Entropy Search:         OFF (insufficient gain)\n");
    }

    /* Recommended Parameters */
    fprintf(out, "\n%s=== RECOMMENDED CLUSTERING PRESETS ===%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    fprintf(out, "  * %sBalanced (Default):%s  rlim = %s%.4f%s (-preset balanced / -preset 10%%)\n",
            ANSI_BOLD, ANSI_COLOR_RESET, ANSI_BOLD_GREEN, p->rlim_balanced, ANSI_COLOR_RESET);
    fprintf(out, "  * Ultra-Fine (1%%):     rlim = %.4f (-preset 1%% / -preset ultrafine)\n",
            p->rlim_p01);
    fprintf(out, "  * Very Fine (3%%):      rlim = %.4f (-preset 3%% / -preset vfine)\n",
            p->rlim_p03);
    fprintf(out, "  * Fine (5%%):           rlim = %.4f (-preset fine / -preset 5%%)\n",
            p->rlim_fine);
    fprintf(out, "  * Coarse (25%%):        rlim = %.4f (-preset coarse / -preset 25%%)\n",
            p->rlim_coarse);
    fprintf(out, "  * Cluster Limit:       -maxcl %d\n", p->recommended_maxcl);
    fprintf(out, "  * Spatial Layout:      -tiles %dx%d\n", p->tiles_x, p->tiles_y);
    fprintf(out, "  * Acceleration:        %s %s %s\n",
            p->use_sq16 ? "-sq16" : (p->use_sq8 ? "-sq8" : ""),
            p->te4_enabled ? "-te4" : "",
            p->te5_enabled ? "-te5" : "");
    fprintf(out, "  * Sparse DCC:          %s%s%s\n",
            p->sparse_dcc_enabled ? ANSI_BOLD_GREEN : ANSI_COLOR_YELLOW,
            p->sparse_dcc_enabled ? "ON (-sparse_dcc)" : "OFF",
            ANSI_COLOR_RESET);
    fprintf(out, "  * Soft Bayesian:       %s%s%s\n",
            p->soft_bayesian_enabled ? ANSI_BOLD_GREEN : ANSI_COLOR_YELLOW,
            p->soft_bayesian_enabled ? "ON (-soft_bayesian)" : "OFF",
            ANSI_COLOR_RESET);
    if (p->soft_bayesian_enabled)
    {
        fprintf(out, "  * Soft Sigma Coeff:    %.2f (-soft_sigma %.2f)\n",
                p->soft_bayesian_sigma_coeff, p->soft_bayesian_sigma_coeff);
    }
    fprintf(out, "  * Precision:           %s\n",
            p->recommend_double ? "float64 (-double recommended)" : "float32 (single)");

    /* Quick copy-paste command */
    fprintf(out, "\n%sSuggested gric-cluster Command:%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    fprintf(out, "  gric-cluster %s", p->dataset_path);
    if (p->tiles_x > 1 || p->tiles_y > 1)
    {
        fprintf(out, " -tiles %dx%d", p->tiles_x, p->tiles_y);
    }
    if (p->pred_enabled)
    {
        fprintf(out, " \"-pred[2,%d,2]\"", p->pred_h);
    }
    if (p->tm_mixing_coeff > 0.0)
    {
        fprintf(out, " -tm %.2f", p->tm_mixing_coeff);
    }
    if (p->use_sq16)
    {
        fprintf(out, " -sq16");
    }
    else if (p->use_sq8)
    {
        fprintf(out, " -sq8");
    }
    if (p->te4_enabled)
    {
        fprintf(out, " -te4");
    }
    if (p->te5_enabled)
    {
        fprintf(out, " -te5");
    }
    if (p->sparse_dcc_enabled)
    {
        fprintf(out, " -sparse_dcc");
    }
    if (p->entropy_enabled)
    {
        fprintf(out, " -entropy");
    }
    if (p->soft_bayesian_enabled)
    {
        fprintf(out, " -soft_bayesian");
        if (p->soft_bayesian_sigma_coeff > 1.001)
        {
            fprintf(out, " -soft_sigma %.2f", p->soft_bayesian_sigma_coeff);
        }
    }
    if (p->recommend_double)
    {
        fprintf(out, " -double");
    }
    fprintf(out, "\n\n");
}

/**
 * probe_report_print_env() - Print shell environment variables.
 * @out:     Target output stream.
 * @results: Pointer to completed ProbeResults.
 */
void probe_report_print_env(
    FILE               *out,
    const ProbeResults *results)
{
    if (out == NULL || results == NULL)
    {
        return;
    }

    const GricProfile *p = &results->profile;
    fprintf(out, "export GRIC_RLIM=%.6f\n", p->rlim_recommended);
    fprintf(out, "export GRIC_RLIM_P01=%.6f\n", p->rlim_p01);
    fprintf(out, "export GRIC_RLIM_P03=%.6f\n", p->rlim_p03);
    fprintf(out, "export GRIC_RLIM_FINE=%.6f\n", p->rlim_fine);
    fprintf(out, "export GRIC_RLIM_BALANCED=%.6f\n", p->rlim_balanced);
    fprintf(out, "export GRIC_RLIM_COARSE=%.6f\n", p->rlim_coarse);
    fprintf(out, "export GRIC_MAXCL=%d\n", p->recommended_maxcl);
    fprintf(out, "export GRIC_TILES=\"%dx%d\"\n", p->tiles_x, p->tiles_y);
    fprintf(out, "export GRIC_USE_SQ8=%d\n", p->use_sq8);
    fprintf(out, "export GRIC_USE_SQ16=%d\n", p->use_sq16);
    fprintf(out, "export GRIC_PREDICT=%d\n", p->pred_enabled);
    fprintf(out, "export GRIC_PRED_H=%d\n", p->pred_h);
    fprintf(out, "export GRIC_TM_MIX=%.4f\n", p->tm_mixing_coeff);
    fprintf(out, "export GRIC_PRUNE_MODE=\"%s\"\n",
            p->recommended_prune_mode[0] ? p->recommended_prune_mode : "3P");
    fprintf(out, "export GRIC_TE4=%d\n", p->te4_enabled);
    fprintf(out, "export GRIC_TE5=%d\n", p->te5_enabled);
    fprintf(out, "export GRIC_SPARSE_DCC=%d\n", p->sparse_dcc_enabled);
    fprintf(out, "export GRIC_ENTROPY=%d\n", p->entropy_enabled);
    fprintf(out, "export GRIC_ENTROPY_GATE=%.4f\n", p->entropy_gate);
    fprintf(out, "export GRIC_SOFT_BAYESIAN=%d\n", p->soft_bayesian_enabled);
    fprintf(out, "export GRIC_SOFT_SIGMA=%.4f\n", p->soft_bayesian_sigma_coeff);
    fprintf(out, "export GRIC_RECOMMEND_DOUBLE=%d\n", p->recommend_double);
    fprintf(out, "export GRIC_NOISE_FLOOR=%.6f\n", p->noise_floor_est);
}

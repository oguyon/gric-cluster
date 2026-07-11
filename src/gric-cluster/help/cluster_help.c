/**
 * @file cluster_help.c
 * @brief Implementations of clustering command help screens.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_help.h"
#include "cluster_help_format.h"
#include "cluster_help_content.h"
#include <stdio.h>
#include <string.h>

#include "shared/cli_colors.h"

/**
 * struct help_entry - keyword lookup table entry
 * @keyword: the keyword string (no leading dashes)
 * @desc:    short description shown in ambiguous lists
 */
struct help_entry
{
    const char *keyword;
    const char *desc;
};

static const struct help_entry help_entries[] = {
    /* Input */
    {"stream",     "Input is an ImageStreamIO stream"},
    {"cnt2sync",   "Enable cnt2 synchronization"},
    /* Core */
    {"rlim",
     "Distance threshold for cluster membership"},
    {"auto_rlim",
     "Auto-scaled rlim (a<factor> syntax)"},
    /* Clustering control */
    {"dprob",      "Delta probability"},
    {"maxcl",      "Max number of clusters"},
    {"ncpu",       "Number of CPUs to use"},
    {"maxcl_strategy",
                   "Strategy when maxcl reached"},
    {"discard_frac",
                   "Fraction of oldest clusters to discard"},
    {"maxim",      "Max number of frames"},
    {"gprob",      "Use geometrical probability"},
    {"fmatcha",    "Set fmatch parameter a"},
    {"fmatchb",    "Set fmatch parameter b"},
    {"maxvis",     "Max visitors for gprob history"},
    {"pred",       "Prediction with pattern detection"},
    {"te4",
     "Use 4-point triangle inequality pruning"},
    {"te5",
     "Use 5-point triangle inequality pruning"},
    {"entropy",
     "Use entropy-based target selection"},
    {"entropy_gate",
     "Entropy gating threshold in bits"},
    {"entropy_first_gate",
     "Entropy gate at measurement depth 0"},
    {"entropy_max_targets",
     "Max targets for entropy evaluation"},
    {"entropy_min_prob",
     "Min probability for entropy hypothesis"},
    {"entropy_fast",
     "Popcount-only surrogate (skip Shannon)"},
    {"soft_bayesian",
     "Enable Soft Bayesian update"},
    {"sparse_dcc",
     "Sparse cluster-to-cluster distance matrix"},
    {"sparse_dcc_extra_evals",
     "Extra DCC evaluations per step"},
    {"soft_bayesian_sigma",
     "Sigma coefficient for soft Bayesian update"},
    {"tm",         "Transition matrix mixing"},
    /* Analysis */
    {"scandist",   "Measure distance stats"},
    /* Output */
    {"outdir",     "Specify output directory"},
    {"avg",        "Compute average frame per cluster"},
    {"distall",    "Save all computed distances"},
    {"pngout",     "Write output as PNG images"},
    {"fitsout",    "Force FITS output format"},
    {"dcc",        "Enable dcc.txt output"},
    {"tm_out",
     "Enable transition_matrix.txt output"},
    {"anchors",    "Enable anchors output"},
    {"counts",
     "Enable cluster_counts.txt output"},
    {"membership",
     "Enable frame_membership.txt output"},
    {"no_membership",
     "Disable frame_membership.txt output"},
    {"discarded",
     "Enable discarded_frames.txt output"},
    {"clustered",
     "Enable *.clustered.txt output"},
    {"clusters",
     "Enable individual cluster files"},
    {"no_dcc",
     "Disable dcc.txt output"},
    {"shm",
     "Enable shared-memory status output"},
    {"progress",   "Print progress information"},
    {"conf",       "Read options from configuration file"},
    {"confw",
     "Write options to configuration file"},
    /* Tiling */
    {"tiles",
     "Split image into NxM tile grid"},
    {"tilemap",
     "Load tile map from integer FITS file"},
    {"tileconf",
     "Per-tile configuration overrides"},
    {"retrieval_window",
     "Tuple lookback horizon for trajectory fusion"},
    {"no_xtile",
     "Disable cross-tile trajectory correction"},
    {"no_pass2",
     "(alias for no_xtile)"},
    /* Topics */
    {"intro",      "Getting started with GRIC"},
    {"input",      "Input formats and options"},
    {"output",     "Output files and options"},
    {"clustering", "Clustering control options"},
    {"analysis",   "Analysis and debugging options"},
    {"algorithm",
     "Overview of the GRIC clustering algorithm"},
    {"algorithms",
     "(alias for algorithm)"},
    {"algorithm/gating",
     "Details on the adaptive entropy gating optimization"},
    {"algorithm/gprob",
     "Details on geometric probability learning and visitors"},
    {"algorithm/entropy",
     "Details on Shannon entropy target selection"},
    {"algorithm/pruning",
     "Details on triangle inequality pruning (TE4/TE5)"},
    {"algorithm/sparse_dcc",
     "Details on sparse cluster distance matrix bounds"},
    {"algorithm/soft_bayesian",
     "Details on soft Bayesian updates and gradual fading"},
    {"performance",
     "How to pick options for best performance"},
    {"tuning",
     "(alias for performance)"},
    {"tiling",
     "Image partitioning and multi-tile processing"},
    {"compression",
     "State space compression and trajectory fusion"},
    {"statespace",
     "(alias for compression)"},
};

#define N_HELP_ENTRIES \
    (sizeof(help_entries) / sizeof(help_entries[0]))

void print_usage(
    char *progname)
{
    printf("Usage: %s [options] <rlim> <input_file|stream_name>\n",
           progname);
    printf("Try '%s -h' for more information.\n", progname);
}

void print_help_keyword(
    const char *keyword)
{
    // Normalize keyword (remove leading dashes)
    const char *key = keyword;
    while (*key == '-')
    {
        key++;
    }

    size_t klen = strlen(key);

    // Exact match via table: dispatch directly
    for (size_t i = 0; i < N_HELP_ENTRIES; i++)
    {
        if (strcmp(key, help_entries[i].keyword) == 0)
        {
            printf("\n%sGRIC HELP: %s%s\n\n",
                   ANSI_BOLD_CYAN, key,
                   ANSI_COLOR_RESET);
            print_keyword_content(key);
            return;
        }
    }

    // Prefix match: collect candidates
    size_t matches[N_HELP_ENTRIES];
    int nm = 0;

    for (size_t i = 0; i < N_HELP_ENTRIES; i++)
    {
        if (strncmp(key,
                    help_entries[i].keyword,
                    klen) == 0
            && strncmp(help_entries[i].desc,
                       "(alias", 6) != 0)
        {
            matches[nm++] = i;
        }
    }

    if (nm == 1)
    {
        const char *resolved =
            help_entries[matches[0]].keyword;
        printf("\n%sGRIC HELP: %s%s\n\n",
               ANSI_BOLD_CYAN, resolved,
               ANSI_COLOR_RESET);
        print_keyword_content(resolved);
        return;
    }

    if (nm > 1)
    {
        printf("Ambiguous keyword '%s'."
               " Did you mean:\n\n", key);
        for (int i = 0; i < nm; i++)
        {
            print_see_also_option(
                help_entries[matches[i]].keyword,
                help_entries[matches[i]].desc);
        }
        printf("\n");
        return;
    }

    // No match at all
    printf("No help available for '%s'.\n", keyword);
    printf(
        "Try '%s -h' to see all options,\n"
        "or  '%s -h <topic>' where topic is:\n"
        "  algorithm, performance, input, output,"
        " clustering, analysis\n",
        "gric-cluster", "gric-cluster");
}

static void print_colored_usage(
    const char *usage)
{
    cli_print_colored_usage(usage);
}

static void print_colored_line(
    const char *line)
{
    cli_print_colored_line(line);
}

/**
 * @brief Print a "See Also" topic reference line in Magenta.
 *
 * @param topic Topic name.
 * @param desc  Description text.
 */
static void print_see_also_topic(
    const char *topic,
    const char *desc)
{
    printf("  %s%-24s%s %s\n",
           ANSI_COLOR_MAGENTA, topic,
           ANSI_COLOR_RESET, desc);
} // print_see_also_topic

void print_help(
    char *progname)
{
    printf("\n%sNAME%s\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %sgric-cluster%s"
           " - Clustering tool for image"
           " streams and sequences\n\n",
           ANSI_BOLD_GREEN,
           ANSI_COLOR_RESET);

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    char usage[256];
    snprintf(usage, sizeof(usage), "%s [options] <rlim> <input_file|stream_name>", progname);
    print_colored_usage(usage);
    printf("\n");

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Perform clustering on a stream of images or a pre-recorded file.\n");
    printf("  Supports FITS, MP4 (via ffmpeg), and raw text input.\n\n");

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  (Use '%s%s%s %s-h%s %s<option>%s' for detailed help on a specific option)\n",
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET,
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);

    printf("  Input %s(use '-h input' for details)%s\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);
#ifdef USE_IMAGESTREAMIO
    print_colored_line("    -stream                  Input is an ImageStreamIO stream");
#else
    print_colored_line("    -stream                  Input is an ImageStreamIO stream [DISABLED]");
#endif
    print_colored_line("    -cnt2sync                Enable cnt2 synchronization (increment cnt2 "
                       "after read)");

    printf("  Clustering Control %s(use '-h clustering'"
           " for details)%s\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);

    printf("    %sCore:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -dprob <val>             Delta probability (default: 0.01)");
    print_colored_line("    -maxcl <val>             Max number of clusters (default: 1000)");
    print_colored_line("    -maxim <val>             Max number of frames (default: 100000)");
    print_colored_line("    -ncpu <val>              Number of CPUs to use (default: 1)");

    printf("    %sTiling:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -tiles <NxM>             Split image into NxM tile grid");
    print_colored_line("      -tilemap <file.fits>   Load tile map from integer FITS file");
    print_colored_line("      -tileconf <file.txt>   Per-tile config "
                       "(tile_id rlim maxcl)");
    print_colored_line("      -retrieval_window <N>  Tuple lookback horizon "
                       "(default: 1000)");
    print_colored_line("      -no_xtile              Disable cross-tile trajectory correction");

    printf("    %sPrediction:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -pred[l,h,n]             Pattern detection "
                       "(default: 10,1000,2)");
    print_colored_line("                            l: pattern length  "
                       "h: history size  n: candidates");
    print_colored_line("    -tm <coeff>              Transition matrix mixing "
                       "(0.0 to 1.0)");

    printf("    %sTarget Selection:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -entropy                 Use entropy-based target cluster selection");
    print_colored_line("      -entropy_fast          Popcount-only surrogate "
                       "(skip Shannon eval)");
    print_colored_line("      -entropy_gate <val>    Gating threshold in bits "
                       "(default: 2.0)");
    print_colored_line("      -entropy_first_gate <val>"
                       " Gate at depth 0 "
                       "(default: 4.0)");
    print_colored_line("      -entropy_max_targets <N>"
                       " Max targets for entropy eval "
                       "(default: 15)");
    print_colored_line("      -entropy_min_prob <val>"
                       " Min hypothesis probability "
                       "(default: 0.001)");

    printf("    %sPruning:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -te4                     Use 4-point triangle inequality pruning");
    print_colored_line("    -te5                     Use 5-point triangle inequality pruning");
    print_colored_line("    -sparse_dcc              Enable sparse cluster-to-cluster "
                       "distance matrix");
    print_colored_line("      -sparse_dcc_extra_evals  Extra DCC evals per step "
                       "(default: 0)");

    printf("    %sGeometric Probability:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -gprob                   Use geometrical probability");
    print_colored_line("      -fmatcha <val>         Set fmatch parameter a "
                       "(default: 2.0)");
    print_colored_line("      -fmatchb <val>         Set fmatch parameter b "
                       "(default: 0.5)");
    print_colored_line("      -maxvis <val>          Max visitors for gprob history "
                       "(default: 1000)");

    printf("    %sSoft Bayesian:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -soft_bayesian           Enable Soft Bayesian update approximation");
    print_colored_line("      -soft_bayesian_sigma <val>"
                       " Sigma coefficient "
                       "(default: 1.0)");

    printf("    %sCluster Eviction:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -maxcl_strategy <str>    Strategy when maxcl reached "
                       "(stop|discard|merge) (default: stop)");
    print_colored_line("      -discard_frac <val>    Fraction of clusters to "
                       "candidate (default: 0.5)");

    printf("    %sConfiguration:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line("    -conf <file>             Read options from configuration file");
    print_colored_line("    -confw <file>            Write current options to "
                       "configuration file\n");


    printf("  Analysis & Debugging %s(use '-h analysis'"
           " for details)%s\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);
    print_colored_line("    -scandist                Measure distance stats");
    print_colored_line("    -progress                Print progress (default: enabled)");

    printf("  Output %s(use '-h output' for details)%s\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);
    print_colored_line("    -outdir <name>           Specify output directory "
                       "(default: <filename>.clusterdat)");
    print_colored_line("    -avg                     Compute average frame per cluster");
    print_colored_line("    -distall                 Save all computed distances");
#ifdef USE_PNG
    print_colored_line("    -pngout                  Write output as PNG images");
#else
    print_colored_line("    -pngout                  Write output as PNG images [DISABLED]");
#endif
#ifdef USE_CFITSIO
    print_colored_line("    -fitsout                 Force FITS output format");
#else
    print_colored_line("    -fitsout                 Force FITS output format [DISABLED]");
#endif
    print_colored_line("    -dcc                     Enable dcc.txt output (default: enabled)");
    print_colored_line("    -no_dcc                  Disable dcc.txt output");
    print_colored_line("    -tm_out                  Enable transition_matrix.txt output "
                       "(default: disabled)");
    print_colored_line("    -anchors                 Enable anchors output (default: disabled)");
    print_colored_line("    -counts                  Enable cluster_counts.txt output "
                       "(default: disabled)");
    print_colored_line("    -shm <file>              Enable shared-memory status output file");
    print_colored_line("    -shm-file <file>         Alias for -shm");
    print_colored_line("    -no_membership           Disable frame_membership.txt output");
    print_colored_line("    -membership              Enable frame_membership.txt output "
                       "(default: enabled)");
    print_colored_line("    -discarded               Enable discarded_frames.txt output "
                       "(default: disabled)");
    print_colored_line("    -clustered               Enable *.clustered.txt output "
                       "(default: disabled)");
    print_colored_line("    -clusters                Enable individual cluster files (cluster_X) "
                       "(default: disabled)\n");

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %sgric-cluster%s"
           " -scandist test_walk.txt\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("  %s$%s %sgric-cluster%s"
           " a1.5 test_walk.txt"
           " -clustered %s>%s run.log\n",
           ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
           ANSI_COLOR_GREY, ANSI_COLOR_RESET);
    printf("\n");

    printf("%sCOMPANION TOOLS%s\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Run %sgric-help%s for an"
           " overview of all GRIC programs"
           " and getting-started guidance.\n\n",
           ANSI_BOLD_GREEN,
           ANSI_COLOR_RESET);
    print_see_also_option(
        "gric-help",
        "Suite overview and onboarding"
        " guide");
    print_see_also_option(
        "gric-status",
        "Monitor SHM telemetry in real"
        " time (TUI)");
    print_see_also_option(
        "gric-benchmark",
        "Run performance benchmarks");
    print_see_also_option(
        "gric-plot",
        "Visualize clustering results");
    print_see_also_option(
        "gric-NDmodel",
        "N-D space reconstruction from"
        " DCC matrix");
    print_see_also_option(
        "gric-info",
        "Print build and module support"
        " status");
    print_see_also_option(
        "gric-mktxtseq",
        "Generate synthetic test"
        " sequences");
    print_see_also_option(
        "gric-mkclusteredfile",
        "Reconstruct clustered files"
        " from membership");
    print_see_also_option(
        "gric-stream-to-pipe",
        "Pipe ImageStreamIO data to"
        " stdout");
    print_see_also_option(
        "gric-ascii-spot-2-video",
        "Convert coordinate text to"
        " video/stream");
    printf("\n");

    printf("%sTOPICS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  (Use '%s%s%s %s-h%s %s<topic>%s' for more information)\n",
           ANSI_BOLD_GREEN, progname, ANSI_COLOR_RESET,
           ANSI_COLOR_GREEN, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET);
    print_see_also_topic(
        "intro",
        "Getting started with GRIC");
    print_see_also_topic(
        "algorithm",
        "Overview of the GRIC clustering algorithm");
    print_see_also_topic(
        "performance",
        "How to pick options for best performance");
    print_see_also_topic(
        "input",
        "Input formats and options");
    print_see_also_topic(
        "output",
        "Output files and options");
    print_see_also_topic(
        "clustering",
        "Clustering control options");
    print_see_also_topic(
        "analysis",
        "Analysis and debugging options");
    print_see_also_topic(
        "tiling",
        "Image partitioning and multi-tile processing");
    print_see_also_topic(
        "compression",
        "State space compression and trajectory fusion");
    printf("\n");

    printf("  %sAlgorithm Deep Dives:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_see_also_topic(
        "algorithm/gating",
        "Adaptive entropy gating optimization");
    print_see_also_topic(
        "algorithm/gprob",
        "Geometric probability learning");
    print_see_also_topic(
        "algorithm/entropy",
        "Shannon entropy target selection");
    print_see_also_topic(
        "algorithm/pruning",
        "Triangle inequality pruning (TE4/TE5)");
    print_see_also_topic(
        "algorithm/sparse_dcc",
        "Sparse cluster distance matrix bounds");
    print_see_also_topic(
        "algorithm/soft_bayesian",
        "Soft Bayesian likelihood updates");
    printf("\n");

    printf("\n");
}

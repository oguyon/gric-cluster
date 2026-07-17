/**
 * @file gric-help.c
 * @brief Onboarding/orientation helper utility for the GRIC suite.
 *
 * This file contains the implementation of the gric-help utility, which provides
 * general onboarding orientation, indexes suite tools, and displays formatted
 * help pages for specific programs in the GRIC suite.
 *
 * Main Functions:
 * - print_formatted_help: Helper to output unified usage/description screens.
 * - print_general_help: Displays suite overview, core concepts, program index, and workflows.
 * - print_program_help: Dispatches and displays detailed help for a specific program.
 * - main: Entry point of the help utility.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shared/cli_colors.h"

static void print_header(
    const char *title,
    int         use_color)
{
    (void)use_color;
    printf("\n%s%s%s\n",
           ANSI_BOLD_CYAN, title, ANSI_COLOR_RESET);
} // print_header

static void print_formatted_help(
    const char *banner,
    const char *usage,
    const char *description,
    const char *options,
    const char *examples)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    /* Parse the banner to style the executable name in Bold Green */
    {
        const char *b = banner;
        while (*b == ' ' || *b == '\t')
        {
            b++;
        }
        const char *cmd_start = b;
        while (*b && *b != ' ' && *b != '\t' && *b != '\n' && *b != '-')
        {
            b++;
        }
        if (b > cmd_start)
        {
            printf("  %s%.*s%s", ANSI_BOLD_GREEN, (int)(b - cmd_start), cmd_start, ANSI_COLOR_RESET);
        }
        else
        {
            printf("  ");
        }
        printf("%s\n\n", b);
    }

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    cli_print_colored_usage(usage);
    printf("\n");

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s\n\n", description);

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    cli_print_colored_options(options);
    printf("\n");

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    cli_print_colored_examples(examples);

    cli_print_color_mode();
} // print_formatted_help

/**
 * @brief Prints the gric-help utility's own help screen.
 */
static void print_help_utility_self(void)
{
    const char *banner = "gric-help - Orientation and onboarding helper for the GRIC suite";
    const char *usage = "gric-help [options] [program_name]";
    const char *desc = "Provides onboarding/orientation information for GRIC users and\n"
                       "  detailed summaries of GRIC programs.";
    const char *opts = "  -h, --help           Show this help message\n\n"
                       "  Arguments:\n"
                       "    program_name       Optional: Show detailed help for a program\n"
                       "                       (e.g., gric-cluster, gric-plot, etc.)";
    const char *ex = "  $ gric-help\n"
                     "  $ gric-help gric-cluster";

    print_formatted_help(banner, usage, desc, opts, ex);
} // print_help_utility_self

/**
 * @brief Print general orientation and onboarding help for GRIC.
 *
 * This function prints a high-level summary of the GRIC clustering
 * ecosystem, its core concepts, list of programs, and basic workflows.
 */
static void print_general_help(void)
{
    printf("\n%sGRIC CLUSTER SUITE - ONBOARDING GUIDE%s\n",
           ANSI_BOLD_CYAN, ANSI_COLOR_RESET);

    print_header("1. OVERVIEW", 1);
    printf("  GRIC is a high-speed, distance-based clustering suite designed for processing\n");
    printf("  sequential image streams and high-dimensional data (e.g. coordinates/vectors).\n");
    printf("  It is optimized for low-latency streaming and high-throughput offline reduction.\n");

    print_header("2. CORE CONCEPTS", 1);
    printf("  %sDistance-based Clustering%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Frames are assigned to clusters based on Euclidean distance to cluster anchors.\n");
    printf("    If a frame is further than the radius limit (rlim) from all existing anchors,\n");
    printf("    a new cluster is created with that frame as the anchor.\n\n");

    printf("  %srlim (Radius Limit)%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    The distance threshold for cluster membership. Can be set to a fixed float value\n");
    printf("    or auto-scaled based on the median frame-to-frame distance (e.g., 'a1.5').\n\n");

    printf("  %sGeometric Probability & Transitions%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    - Geometric Probability (-gprob): Uses spatial/temporal trajectory prediction.\n");
    printf("    - Transition Matrix (-tm): Models sequence transition behaviors.\n\n");

    printf("  %sTriangle Inequality Pruning (TE4/TE5)%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Accelerates clustering by skipping expensive distance calculations\n");
    printf("    using geometric constraints on cluster-to-cluster distances.\n\n");

    printf("  %sEntropy-Based Target Selection (-entropy)%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Instead of greedily measuring the most probable cluster, selects the\n");
    printf("    target that maximizes expected information gain (minimizes posterior\n");
    printf("    Shannon entropy).  Uses a multi-stage pipeline: gating, popcount\n");
    printf("    scoring, candidate filtering, and Shannon evaluation.  A lightweight\n");
    printf("    surrogate mode (-entropy_fast) skips the Shannon evaluation entirely.\n");
    printf("    Run 'gric-cluster -h entropy' for a detailed description.\n");

    print_header("3. PROGRAM INDEX", 1);
    printf("  %sCore Tool%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Main clustering executable (offline files or live streams).\n",
           ANSI_BOLD_GREEN, "gric-cluster", ANSI_COLOR_RESET);
    printf("  %sVisualization%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates SVG/PNG diagnostic and summary plots.\n",
           ANSI_BOLD_GREEN, "gric-plot", ANSI_COLOR_RESET);
    printf("  %sUtility & Simulation%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Prints build info, library paths, and enabled features.\n",
           ANSI_BOLD_GREEN, "gric-info", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs N-dimensional coordinates from dcc.txt.\n",
           ANSI_BOLD_GREEN, "gric-NDmodel", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates synthetic coordinate sequences (walk, spiral, etc.).\n",
           ANSI_BOLD_GREEN, "gric-mktxtseq", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Converts 3D coordinates into MP4 video/ImageStreamIO streams.\n",
           ANSI_BOLD_GREEN, "gric-ascii-spot-2-video", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs a full clustered file from input and membership list.\n",
           ANSI_BOLD_GREEN, "gric-mkclusteredfile", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Pipes raw data from live ImageStreamIO shared memory to stdout.\n",
           ANSI_BOLD_GREEN, "gric-stream-to-pipe", ANSI_COLOR_RESET);
    printf("  %sMonitoring & Benchmarking%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Real-time SHM telemetry monitor (TUI dashboard).\n",
           ANSI_BOLD_GREEN, "gric-status", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Performance benchmarking across patterns and options.\n",
           ANSI_BOLD_GREEN, "gric-benchmark", ANSI_COLOR_RESET);

    print_header("4. TYPICAL ONBOARDING WORKFLOW", 1);
    printf("  Follow these steps to familiarize yourself with GRIC:\n\n");
    printf("  %sStep 1: Check Optional Dependencies%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Check which input/output formats are enabled (FITS, PNG, FFmpeg, ImageStreamIO):\n");
    printf("      $ %sgric-info%s\n\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 2: Generate Synthetic Data%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Generate a 2D random walk sequence of 1000 points to act as fake coordinates:\n");
    printf("      $ %sgric-mktxtseq%s 1000 test_walk.txt 2Dwalk\n\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 3: Scan the Sequence Distances%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Measure distance statistics to choose a reasonable radius limit (rlim):\n");
    printf("      $ %sgric-cluster%s -scandist test_walk.txt\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("    Take note of the \"Median distance\" output from the scan.\n\n");

    printf("  %sStep 4: Run the Clustering%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Cluster the points using a radius limit of 1.5x the median distance:\n");
    printf("      $ %sgric-cluster%s a1.5 test_walk.txt -clustered > run.log\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("    This generates an output directory: `test_walk.clusterdat/`.\n\n");

    printf("  %sStep 5: Plot results%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Visualize the clusters and centroids using the plotting tool:\n");
    printf("      $ %sgric-plot%s test_walk.txt run.log plot.png\n\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  For detailed guide on a specific program, run:\n");
    printf("    $ %sgric-help%s %s<program-name>%s   (e.g. %sgric-help%s %sgric-cluster%s)\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    cli_print_color_mode();
} // print_general_help

/**
 * @brief Prints detailed help for a specific program.
 *
 * @param prog Name of the program to show help for.
 */
static int print_program_help(
    const char *prog)
{
    const char *target = prog;

    if (strcmp(prog, "cluster") == 0)
    {
        target = "gric-cluster";
    }
    else if (strcmp(prog, "plot") == 0)
    {
        target = "gric-plot";
    }
    else if (strcmp(prog, "info") == 0)
    {
        target = "gric-info";
    }
    else if (strcmp(prog, "mktxtseq") == 0 || strcmp(prog, "mktestseq") == 0)
    {
        target = "gric-mktxtseq";
    }
    else if (strcmp(prog, "NDmodel") == 0 || strcmp(prog, "ndmodel") == 0 ||
             strcmp(prog, "model_nd") == 0)
    {
        target = "gric-NDmodel";
    }
    else if (strcmp(prog, "ascii-spot-2-video") == 0 || strcmp(prog, "spot2video") == 0)
    {
        target = "gric-ascii-spot-2-video";
    }
    else if (strcmp(prog, "mkclusteredfile") == 0)
    {
        target = "gric-mkclusteredfile";
    }
    else if (strcmp(prog, "stream-to-pipe") == 0 ||
             strcmp(prog, "stream_to_pipe") == 0)
    {
        target = "gric-stream-to-pipe";
    }
    else if (strcmp(prog, "status") == 0)
    {
        target = "gric-status";
    }
    else if (strcmp(prog, "benchmark") == 0)
    {
        target = "gric-benchmark";
    } // if (strcmp(prog, "cluster") == 0)...

    if (strcmp(target, "gric-cluster") == 0)
    {
        /* Help definitions for gric-cluster */
        const char *banner = "gric-cluster - Core clustering tool for image streams\n"
                             "                 and sequences";
        const char *usage = "gric-cluster [options] <rlim> <input_file|stream_name>";
        const char *desc = "Perform clustering on a stream of images or a pre-recorded "
                           "file.\n  Supports FITS, MP4 (via ffmpeg), and raw text "
                           "input.";
        const char *opts =
            "  -rlim <val>          Distance limit (e.g. 0.5, or a1.5 for 1.5x median)\n"
            "  -stream              Enable ImageStreamIO input stream mode\n"
            "  -avg                 Compute and output average frame for each cluster\n"
            "  -gprob               Enable geometric probability path-based clustering\n"
            "  -maxcl_strategy <S>  Strategy when maxcl is hit (stop|discard|merge)\n"
            "  -te4 / -te5          Triangle inequality pruning (4-pt / 5-pt)\n"
            "  -entropy             Entropy-based target selection\n"
            "  -entropy_fast        Popcount-only surrogate (skip Shannon eval)\n"
            "  -soft_bayesian       Smooth Bayesian updates between measurements\n"
            "  -ncpu <N>            Number of OpenMP threads\n"
            "  -scandist            Analyze frame-to-frame distances without clustering\n"
            "  -h <topic>           Detailed help (e.g. -h entropy, -h performance)";
        const char *ex =
            "  $ gric-cluster -scandist test_walk.txt\n"
            "  $ gric-cluster a1.5 test_walk.txt -clustered > run.log";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-plot") == 0)
    {
        /* Help definitions for gric-plot */
        const char *banner = "gric-plot - Visualization tool for clustering results";
        const char *usage = "gric-plot [options] <points_file> <log_file> [output_file]";
        const char *desc = "Generates a diagnostic scatter plot and histograms of the "
                           "clustering\n  results. Outputs PNG (default) or SVG.";
        const char *opts =
            "  -svg                 Output SVG format instead of PNG\n"
            "  -fs <size>           Set font size for labels (default: 18.0)";
        const char *ex =
            "  $ gric-plot input_points.txt gric_run_log.txt summary_plot.png";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-info") == 0)
    {
        /* Help definitions for gric-info */
        const char *banner = "gric-info - Prints support and build status of optional modules";
        const char *usage = "gric-info";
        const char *desc = "Checks and prints status of optional modules: CFITSIO, LibPNG, "
                           "FFmpeg,\n  ImageStreamIO, and OpenMP.";
        const char *opts = "  (None)";
        const char *ex = "  $ gric-info";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-mktxtseq") == 0)
    {
        /* Help definitions for gric-mktxtseq */
        const char *banner = "gric-mktxtseq - Synthetic sequence generator for testing";
        const char *usage = "gric-mktxtseq <N> <output_file> <pattern> [options]";
        const char *desc = "Generates synthetic coordinate sequences (walk, spiral, circle, "
                           "etc.) for testing.";
        const char *opts =
            "  -repeat <M>          Repeat the pattern M times\n"
            "  -noise <R>           Add random noise with radius R to each point\n"
            "  -shuffle             Shuffle the order of generated points\n\n"
            "  Patterns:\n"
            "    [ND]random         Uniform random in unit hypercube/sphere\n"
            "    [ND]sphere         Random points on unit hypersphere surface\n"
            "    [ND]walk[S]        Random walk (S = step size, default 0.1)\n"
            "    [ND]spiral[L]      Spiral (L = loops, default 3.0)\n"
            "    [ND]circle[P]      Circle (P = period)";
        const char *ex = "  $ gric-mktxtseq 1000 test_walk.txt 2Dwalk";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-NDmodel") == 0)
    {
        /* Help definitions for gric-NDmodel */
        const char *banner = "gric-NDmodel - N-Dimensional space reconstruction";
        const char *usage = "gric-NDmodel <dcc_file> <dimensions> <output_file> [options]";
        const char *desc = "Reconstructs N-dimensional coordinates from a cluster distance "
                           "matrix\n  (dcc.txt) using Simulated Annealing optimization.";
        const char *opts =
            "  -temp <val>          Initial temperature (default: 10.0)\n"
            "  -rate <val>          Cooling rate (default: 0.995)\n"
            "  -iter <val>          Number of iterations (default: 100000)";
        const char *ex = "  $ gric-NDmodel dcc.txt 3 coordinates.txt";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-ascii-spot-2-video") == 0)
    {
        /* Help definitions for gric-ascii-spot-2-video */
        const char *banner = "gric-ascii-spot-2-video - Convert coordinate text to video/stream";
        const char *usage =
            "gric-ascii-spot-2-video [options] <pixel_size> <alpha> <input.txt> <output>";
        const char *desc = "Converts a 3D coordinate sequence into a simulated movie (MP4) "
                           "or a live\n  ImageStreamIO shared memory stream containing a "
                           "moving Gaussian spot.";
        const char *opts =
            "  -isio                Write to ImageStreamIO stream instead of a file\n"
            "  -fps <val>           Set playback frame rate\n"
            "  -loop                Loop the coordinates infinitely";
        const char *ex = "  $ gric-ascii-spot-2-video 256 2.0 input.txt output.mp4";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-mkclusteredfile") == 0)
    {
        /* Help definitions for gric-mkclusteredfile */
        const char *banner = "gric-mkclusteredfile - Reconstructs a full clustered file";
        const char *usage =
            "gric-mkclusteredfile <input_file> <membership_file> <output_file> [options]";
        const char *desc = "Reconstructs a fully clustered file from the original coordinates\n"
                           "  and the frame membership output.";
        const char *opts = "  -rlim <val>          Specify radius limit to write to header";
        const char *ex =
            "  $ gric-mkclusteredfile input.txt membership.txt output_clustered.txt";

        print_formatted_help(banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-stream-to-pipe") == 0)
    {
        /* Help definitions for gric-stream-to-pipe */
        const char *banner =
            "gric-stream-to-pipe - Pipes raw"
            " ImageStreamIO stream data";
        const char *usage =
            "gric-stream-to-pipe"
            " <stream_name> [max_frames]";
        const char *desc =
            "Pipes raw floating-point data"
            " from an ImageStreamIO stream"
            " directly to stdout.";
        const char *opts = "  (None)";
        const char *ex =
            "  $ gric-stream-to-pipe"
            " mystream 500";

        print_formatted_help(
            banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target, "gric-status") == 0)
    {
        const char *banner =
            "gric-status - Monitor"
            " shared-memory telemetry"
            " from gric-cluster";
        const char *usage =
            "gric-status [options]"
            " <shm_file>";
        const char *desc =
            "Real-time TUI dashboard that"
            " reads the SHM status struct\n"
            "  written by gric-cluster"
            " (via -shm).  Displays cluster"
            " count,\n  frame rate, entropy"
            " telemetry, and convergence"
            " metrics.";
        const char *opts =
            "  -r <Hz>              "
            "Refresh rate (default: 10)\n"
            "  -1                   "
            "Print once and exit";
        const char *ex =
            "  $ gric-status"
            " /tmp/gric_status.shm";

        print_formatted_help(
            banner, usage, desc, opts, ex);
        return 0;
    }
    else if (strcmp(target,
                    "gric-benchmark") == 0)
    {
        const char *banner =
            "gric-benchmark - Run"
            " performance benchmarks on"
            " gric-cluster";
        const char *usage =
            "gric-benchmark [options]";
        const char *desc =
            "Generates synthetic datasets"
            " and benchmarks gric-cluster\n"
            "  across different patterns,"
            " dimensions, and option"
            " combinations.\n  Reports"
            " timing, distance call counts,"
            " and cluster quality.";
        const char *opts =
            "  -patterns <list>     "
            "Comma-separated pattern"
            " names\n"
            "  -maxim <N>           "
            "Maximum number of frames\n"
            "  -maxcl <N>           "
            "Maximum number of clusters";
        const char *ex =
            "  $ gric-benchmark\n"
            "  $ gric-benchmark"
            " -patterns 3Drand,3Dwalk"
            " -maxim 10000";

        print_formatted_help(
            banner, usage, desc, opts, ex);
        return 0;
    }
    else
    {
        fprintf(stderr, "%sError: Unknown program '%s'.%s\n", ANSI_COLOR_RED, prog, ANSI_COLOR_RESET);
        fprintf(stderr, "Run 'gric-help' to see the list of valid programs.\n");
        return 1;
    } // if (strcmp(target, "gric-cluster") == 0)...
} // print_program_help

static int run_help(
    int   argc,
    char *argv[])
{
    if (argc < 2)
    {
        print_general_help();
        return 0;
    }

    // Check if the user is asking for general help
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "help") == 0)
    {
        print_help_utility_self();
        return 0;
    } // if (strcmp(argv[1], "-h") == 0 ...)

    return print_program_help(argv[1]);
}

int main(
    int   argc,
    char *argv[])
{
    cli_colors_init();

    FILE *tmp = tmpfile();
    if (tmp != NULL)
    {
        int saved_stdout = dup(STDOUT_FILENO);
        int tmp_fd = fileno(tmp);
        dup2(tmp_fd, STDOUT_FILENO);

        int res = run_help(argc, argv);
        fflush(stdout);

        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);

        fseek(tmp, 0, SEEK_END);
        long sz = ftell(tmp);
        fseek(tmp, 0, SEEK_SET);

        char *buf = malloc((size_t)sz + 1);
        if (buf != NULL)
        {
            size_t read_bytes = fread(buf, 1, (size_t)sz, tmp);
            buf[read_bytes] = '\0';
            cli_print_pager(buf);
            free(buf);
        }
        fclose(tmp);
        return res;
    }
    else
    {
        return run_help(argc, argv);
    }
} // main

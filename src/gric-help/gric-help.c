/**
 * @file gric-help.c
 * @brief Onboarding/orientation helper utility for the GRIC suite.
 *
 * Provides general onboarding orientation, indexes suite tools, and displays formatted
 * help pages for specific programs in the GRIC suite.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shared/cli_colors.h"

/**
 * struct ProgramHelpDoc - Detailed documentation record for a GRIC executable.
 * @canonical_name: Primary program binary name (e.g. "gric-cluster").
 * @aliases:        NULL-terminated array of recognized alias strings.
 * @banner:         One-line header description with program name.
 * @usage:          CLI usage prototype string.
 * @description:    Multi-line functional description.
 * @options:        Formatted list of options and flags.
 * @examples:       Example command-line invocations.
 */
typedef struct {
    const char *canonical_name;
    const char *aliases[6];
    const char *banner;
    const char *usage;
    const char *description;
    const char *options;
    const char *examples;
} ProgramHelpDoc;

/**
 * print_header() - Print a colored section header.
 * @title:     Section header title string.
 * @use_color: Color enable flag (unused, retained for compatibility).
 */
static void print_header(
    const char *title,
    int         use_color)
{
    (void)use_color;
    printf("\n%s%s%s\n", ANSI_BOLD_CYAN, title, ANSI_COLOR_RESET);
}

/**
 * print_formatted_help() - Output formatted and syntax-highlighted manual page.
 * @banner:      Header banner.
 * @usage:       Usage synopsis.
 * @description: Functional description.
 * @options:     Option definitions.
 * @examples:    Invocation examples.
 */
static void print_formatted_help(
    const char *banner,
    const char *usage,
    const char *description,
    const char *options,
    const char *examples)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
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
            printf("  %s%.*s%s",
                   ANSI_BOLD_GREEN, (int)(b - cmd_start), cmd_start, ANSI_COLOR_RESET);
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
}

/**
 * print_help_utility_self() - Print the gric-help utility's own help screen.
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
}

/**
 * print_general_help() - Print general orientation and onboarding help for GRIC.
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
    printf("    The distance threshold for cluster membership. Can be a fixed float value\n");
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
    printf("    Shannon entropy). Uses a multi-stage pipeline: gating, popcount\n");
    printf("    scoring, candidate filtering, and Shannon evaluation. A lightweight\n");
    printf("    surrogate mode (-entropy_fast) skips the Shannon evaluation entirely.\n");
    printf("    Run 'gric-cluster -h entropy' for a detailed description.\n");

    print_header("3. PROGRAM INDEX", 1);
    printf("  %sCore Clustering & Indexing%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Main clustering executable (offline files or live streams).\n",
           ANSI_BOLD_GREEN, "gric-cluster", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Out-of-core metric-pruned k-NN search solver.\n",
           ANSI_BOLD_GREEN, "gric-knn", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs frames and feature averages using k-NN graphs.\n",
           ANSI_BOLD_GREEN, "gric-knn-avg", ANSI_COLOR_RESET);
    printf("  %sVisualization & Desktop GUI%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates SVG/PNG diagnostic and summary plots.\n",
           ANSI_BOLD_GREEN, "gric-plot", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Desktop application launcher for interactive simulator.\n",
           ANSI_BOLD_GREEN, "gric-gui", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Native C HTTP micro-server powering the GUI & REST API.\n",
           ANSI_BOLD_GREEN, "gric-server", ANSI_COLOR_RESET);
    printf("  %sBinary Format & Conversion%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Encodes ASCII tables and coordinates into .bin files.\n",
           ANSI_BOLD_GREEN, "gric-ascii2bin", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Decodes and inspects self-describing .bin files.\n",
           ANSI_BOLD_GREEN, "gric-bin2ascii", ANSI_COLOR_RESET);
    printf("  %sBenchmarking, Tuning & Telemetry%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Real-time SHM telemetry monitor (TUI dashboard).\n",
           ANSI_BOLD_GREEN, "gric-status", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Performance benchmarking across synthetic manifolds.\n",
           ANSI_BOLD_GREEN, "gric-benchmark", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Parameter space exploration and hyperparameter tuner.\n",
           ANSI_BOLD_GREEN, "gric-tune", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Offline cluster quality and efficiency analysis tool.\n",
           ANSI_BOLD_GREEN, "gric-cluster-analysis", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Local intrinsic dimensionality (LID) and density estimator.\n",
           ANSI_BOLD_GREEN, "gric-dimdensity", ANSI_COLOR_RESET);
    printf("  %sGenerators, Simulation & Streaming%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Prints build info, library paths, and enabled features.\n",
           ANSI_BOLD_GREEN, "gric-info", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs N-dimensional coordinates from dcc.txt.\n",
           ANSI_BOLD_GREEN, "gric-NDmodel", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates synthetic coordinate sequences (walk, spiral, etc.).\n",
           ANSI_BOLD_GREEN, "gric-mktxtseq", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates multi-body bouncing ball physics simulations.\n",
           ANSI_BOLD_GREEN, "gric-gen-balls", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Converts 3D coordinates into MP4 video/ImageStreamIO streams.\n",
           ANSI_BOLD_GREEN, "gric-ascii-spot-2-video", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Ingests ASCII coordinate streams into ImageStreamIO SHM.\n",
           ANSI_BOLD_GREEN, "gric-txt2stream", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs a full clustered file from input and memberships.\n",
           ANSI_BOLD_GREEN, "gric-mkclusteredfile", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Pipes raw data from live ImageStreamIO shared memory to stdout.\n",
           ANSI_BOLD_GREEN, "gric-stream-to-pipe", ANSI_COLOR_RESET);

    print_header("4. TYPICAL ONBOARDING WORKFLOW", 1);
    printf("  Follow these steps to familiarize yourself with GRIC:\n\n");
    printf("  %sStep 1: Check Optional Dependencies%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Check which I/O formats are enabled (FITS, PNG, FFmpeg, ImageStreamIO):\n");
    printf("      $ %sgric-info%s\n\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 2: Generate Synthetic Data%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Generate a 2D random walk sequence of 1000 points to act as fake coordinates:\n");
    printf("      $ %sgric-mktxtseq%s 1000 test_walk.txt 2Dwalk\n\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 3: Scan the Sequence Distances%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Measure distance statistics to choose a reasonable radius limit (rlim):\n");
    printf("      $ %sgric-cluster%s -scandist test_walk.txt\n", ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("    Take note of the \"Median distance\" output from the scan.\n\n");

    printf("  %sStep 4: Run the Clustering%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Cluster the points using a radius limit of 1.5x the median distance:\n");
    printf("      $ %sgric-cluster%s a1.5 test_walk.txt -clustered > run.log\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("    This generates an output directory: `test_walk.clusterdat/`.\n\n");

    printf("  %sStep 5: Plot results%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Visualize the clusters and centroids using the plotting tool:\n");
    printf("      $ %sgric-plot%s test_walk.txt run.log plot.png\n\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);

    printf("  For detailed guide on a specific program, run:\n");
    printf("    $ %sgric-help%s %s<program-name>%s   (e.g. %sgric-help%s %sgric-cluster%s)\n",
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
           ANSI_COLOR_MAGENTA, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    cli_print_color_mode();
}

static const ProgramHelpDoc PROGRAM_DOCS[] = {
    {
        .canonical_name = "gric-cluster",
        .aliases = {"cluster", "gric-cluster", NULL},
        .banner = "gric-cluster - Core clustering tool for image streams and sequences",
        .usage = "gric-cluster [options] <rlim> <input_file|stream_name>",
        .description = "Perform clustering on a stream of images or a pre-recorded file.\n"
                       "  Supports FITS, MP4 (via ffmpeg), and raw text input.",
        .options =
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
            "  -h <topic>           Detailed help (e.g. -h entropy, -h performance)",
        .examples =
            "  $ gric-cluster -scandist test_walk.txt\n"
            "  $ gric-cluster a1.5 test_walk.txt -clustered > run.log"
    },
    {
        .canonical_name = "gric-plot",
        .aliases = {"plot", "gric-plot", NULL},
        .banner = "gric-plot - Visualization tool for clustering results",
        .usage = "gric-plot [options] <points_file> <log_file> [output_file]",
        .description = "Generates a diagnostic scatter plot and histograms of the clustering\n"
                       "  results. Outputs PNG (default) or SVG.",
        .options =
            "  -svg                 Output SVG format instead of PNG\n"
            "  -fs <size>           Set font size for labels (default: 18.0)",
        .examples =
            "  $ gric-plot input_points.txt gric_run_log.txt summary_plot.png"
    },
    {
        .canonical_name = "gric-gui",
        .aliases = {"gui", "gric-gui", NULL},
        .banner = "gric-gui - Launch the GRIC Interactive Simulator desktop application",
        .usage = "gric-gui [options]",
        .description =
            "Launches the GRIC web simulator in a standalone native desktop window.\n"
            "  Automatically verifies and starts the background HTTP server on the\n"
            "  designated port if not already active.",
        .options =
            "  -h, --help           Show this help message and exit\n"
            "  -p, --port <port>    HTTP port for local simulator server (default: 8080)\n"
            "  -b, --browser <name> Select browser (google-chrome, chromium, firefox, default)\n"
            "  -w, --window-size <W,H> Initial window resolution (default: 1600,1000)\n"
            "  -s, --server-only    Start background HTTP server without launching browser\n"
            "  -k, --kill-server    Terminate running background HTTP server on the port\n"
            "  --status             Display active server process and browser runtime status",
        .examples =
            "  $ gric-gui\n"
            "  $ gric-gui -p 8088 -w 1920,1080\n"
            "  $ gric-gui --server-only\n"
            "  $ gric-gui --kill-server"
    },
    {
        .canonical_name = "gric-info",
        .aliases = {"info", "gric-info", NULL},
        .banner = "gric-info - Prints support and build status of optional modules",
        .usage = "gric-info",
        .description = "Checks and prints status of optional modules: CFITSIO, LibPNG, FFmpeg,\n"
                       "  ImageStreamIO, and OpenMP.",
        .options = "  (None)",
        .examples = "  $ gric-info"
    },
    {
        .canonical_name = "gric-mktxtseq",
        .aliases = {"mktxtseq", "mktestseq", "gric-mktxtseq", NULL},
        .banner = "gric-mktxtseq - Synthetic sequence generator for testing",
        .usage = "gric-mktxtseq <N> <output_file> <pattern> [options]",
        .description = "Generates synthetic coordinate sequences (walk, spiral, circle, etc.).",
        .options =
            "  -repeat <M>          Repeat the pattern M times\n"
            "  -noise <R>           Add random noise with radius R to each point\n"
            "  -shuffle             Shuffle the order of generated points\n\n"
            "  Patterns:\n"
            "    [ND]random         Uniform random in unit hypercube/sphere\n"
            "    [ND]sphere         Random points on unit hypersphere surface\n"
            "    [ND]walk[S]        Random walk (S = step size, default 0.1)\n"
            "    [ND]spiral[L]      Spiral (L = loops, default 3.0)\n"
            "    [ND]circle[P]      Circle (P = period)",
        .examples = "  $ gric-mktxtseq 1000 test_walk.txt 2Dwalk"
    },
    {
        .canonical_name = "gric-NDmodel",
        .aliases = {"NDmodel", "ndmodel", "model_nd", "gric-NDmodel", NULL},
        .banner = "gric-NDmodel - N-Dimensional space reconstruction",
        .usage = "gric-NDmodel <dcc_file> <dimensions> <output_file> [options]",
        .description = "Reconstructs N-dimensional coordinates from a cluster distance matrix\n"
                       "  (dcc.txt) using Simulated Annealing optimization.",
        .options =
            "  -temp <val>          Initial temperature (default: 10.0)\n"
            "  -rate <val>          Cooling rate (default: 0.995)\n"
            "  -iter <val>          Number of iterations (default: 100000)",
        .examples = "  $ gric-NDmodel dcc.txt 3 coordinates.txt"
    },
    {
        .canonical_name = "gric-ascii-spot-2-video",
        .aliases = {"ascii-spot-2-video", "spot2video", "gric-ascii-spot-2-video", NULL},
        .banner = "gric-ascii-spot-2-video - Convert coordinate text to video/stream",
        .usage = "gric-ascii-spot-2-video [options] <pixel_size> <alpha> <input.txt> <output>",
        .description = "Converts a 3D coordinate sequence into a simulated movie (MP4) or a live\n"
                       "  ImageStreamIO shared memory stream containing a moving Gaussian spot.",
        .options =
            "  -isio                Write to ImageStreamIO stream instead of a file\n"
            "  -fps <val>           Set playback frame rate\n"
            "  -loop                Loop the coordinates infinitely",
        .examples = "  $ gric-ascii-spot-2-video 256 2.0 input.txt output.mp4"
    },
    {
        .canonical_name = "gric-mkclusteredfile",
        .aliases = {"mkclusteredfile", "gric-mkclusteredfile", NULL},
        .banner = "gric-mkclusteredfile - Reconstructs a full clustered file",
        .usage = "gric-mkclusteredfile <input_file> <membership_file> <output_file> [options]",
        .description = "Reconstructs a fully clustered file from the original coordinates\n"
                       "  and the frame membership output.",
        .options = "  -rlim <val>          Specify radius limit to write to header",
        .examples = "  $ gric-mkclusteredfile input.txt membership.txt output_clustered.txt"
    },
    {
        .canonical_name = "gric-stream-to-pipe",
        .aliases = {"stream-to-pipe", "stream_to_pipe", "gric-stream-to-pipe", NULL},
        .banner = "gric-stream-to-pipe - Pipes raw ImageStreamIO stream data",
        .usage = "gric-stream-to-pipe <stream_name> [max_frames]",
        .description = "Pipes raw floating-point data from an ImageStreamIO stream to stdout.",
        .options = "  (None)",
        .examples = "  $ gric-stream-to-pipe mystream 500"
    },
    {
        .canonical_name = "gric-status",
        .aliases = {"status", "gric-status", NULL},
        .banner = "gric-status - Monitor shared-memory telemetry from gric-cluster",
        .usage = "gric-status [options] <shm_file>",
        .description = "Real-time TUI dashboard reading the SHM status struct written by\n"
                       "  gric-cluster (via -shm). Displays cluster count, frame rate, entropy\n"
                       "  telemetry, and convergence metrics.",
        .options =
            "  -r <Hz>              Refresh rate (default: 10)\n"
            "  -1                   Print once and exit",
        .examples = "  $ gric-status /tmp/gric_status.shm"
    },
    {
        .canonical_name = "gric-benchmark",
        .aliases = {"benchmark", "gric-benchmark", NULL},
        .banner = "gric-benchmark - Run performance benchmarks on gric-cluster",
        .usage = "gric-benchmark [options]",
        .description = "Generates synthetic datasets and benchmarks gric-cluster across different\n"
                       "  patterns, dimensions, and option combinations.",
        .options =
            "  -patterns <list>     Comma-separated pattern names\n"
            "  -maxim <N>           Maximum number of frames\n"
            "  -maxcl <N>           Maximum number of clusters",
        .examples =
            "  $ gric-benchmark\n"
            "  $ gric-benchmark -patterns 3Drand,3Dwalk -maxim 10000"
    },
    {
        .canonical_name = "gric-knn",
        .aliases = {"knn", "gric-knn", NULL},
        .banner = "gric-knn - Out-of-core metric-pruned k-NN solver",
        .usage = "gric-knn <input_data> <cluster_dir> [options]",
        .description = "Identifies the k-nearest neighbors for each frame in a dataset using\n"
                       "  cluster anchors and metric distance bounds from gric-cluster.",
        .options =
            "  -k <int>             Number of nearest neighbors to find (default: 10)\n"
            "  -o, --output <path>  Output destination file (.txt, .bin, .fits)\n"
            "  -dtmin <int>         Min frame separation |i - j| >= dtmin (default: 1)\n"
            "  -past / -future      Search only in preceding or subsequent frames\n"
            "  -eps <float>         (1+eps)-ANN relaxation slack factor (default: 0.0)\n"
            "  -rlim <float>        Cutoff radius cutoff\n"
            "  -approx              Fast Approximate Graph Search (10-20 evals/query)\n"
            "  -ef-search <int>     Search pool / heap size (default: 2*k in approx)\n"
            "  -nthreads <int>      Number of OpenMP worker threads\n"
            "  -progress            Display live progress bar",
        .examples =
            "  $ gric-knn input.txt cluster_out/ -k 10 -dtmin 5\n"
            "  $ gric-knn dataset.fits cluster_out/ -k 20 -eps 0.05 -nthreads 8"
    },
    {
        .canonical_name = "gric-knn-avg",
        .aliases = {"knn-avg", "knn_avg", "gric-knn-avg", NULL},
        .banner = "gric-knn-avg - Reconstruct frames and averages using k-NN graphs",
        .usage = "gric-knn-avg <input_data> <knn_graph_file> [options]",
        .description = "Computes k-NN locally weighted frame reconstructions and averaged\n"
                       "  feature profiles using k-NN graph connectivity output.",
        .options =
            "  -k <int>             Number of graph neighbors to average\n"
            "  -weights <mode>      Weighting kernel (uniform|gaussian|inverse_dist)\n"
            "  -o, --output <path>  Output destination for averaged frames",
        .examples =
            "  $ gric-knn-avg input.fits knn_graph.bin -k 10 -o reconstructed.fits"
    },
    {
        .canonical_name = "gric-dimdensity",
        .aliases = {"dimdensity", "density", "gric-dimdensity", NULL},
        .banner = "gric-dimdensity - Measure local intrinsic dimensionality and density",
        .usage = "gric-dimdensity <input_file|stream_name> [options]",
        .description = "Estimates local intrinsic dimensionality (LID) and neighborhood density\n"
                       "  profiles across datasets using metric distance distributions.",
        .options =
            "  -k <int>             Number of nearest neighbors to evaluate (default: 20)\n"
            "  -rlim <float>        Cutoff radius cutoff\n"
            "  -o, --output <path>  Output destination for density metrics (.txt or .bin)\n"
            "  -nthreads <int>      Number of OpenMP worker threads",
        .examples =
            "  $ gric-dimdensity input.txt -k 20 -o density_out.txt"
    },
    {
        .canonical_name = "gric-ascii2bin",
        .aliases = {"ascii2bin", "gric-ascii2bin", NULL},
        .banner = "gric-ascii2bin - Convert ASCII tables/coordinates to binary format",
        .usage = "gric-ascii2bin <input.txt> <output.bin> [options]",
        .description =
            "Encodes ASCII coordinate tables and data matrices into the self-describing\n"
            "  GRIC binary (.bin) format for rapid zero-copy mmap.",
        .options =
            "  -type <type>         Semantic type (anchors|dcc|membership|counts|coords)\n"
            "  -double              Encode floats as float64 (default: float32)\n"
            "  -uint32 / -int32     Encode integers as 32-bit unsigned/signed\n"
            "  -dim <D>             Explicit column dimension count\n"
            "  -v, --verbose        Print verbose encoding details",
        .examples =
            "  $ gric-ascii2bin 2Dspiral.txt spiral.bin -type coords\n"
            "  $ gric-ascii2bin dcc.txt dcc.bin -type dcc -double"
    },
    {
        .canonical_name = "gric-bin2ascii",
        .aliases = {"bin2ascii", "gric-bin2ascii", NULL},
        .banner = "gric-bin2ascii - Decode GRIC binary files to ASCII or stdout",
        .usage = "gric-bin2ascii <input.bin> [output.txt] [options]",
        .description = "Decodes self-describing GRIC binary (.bin) files into ASCII format\n"
                       "  or pipes directly to stdout.",
        .options =
            "  -info, -i            Display header metadata without decoding payload\n"
            "  -fmt <specifier>     Custom printf format specifier (e.g. '%.8f')\n"
            "  -v, --verbose        Print decoding summary to stderr",
        .examples =
            "  $ gric-bin2ascii spiral.bin -info\n"
            "  $ gric-bin2ascii dcc.bin - | head -n 10"
    },
    {
        .canonical_name = "gric-server",
        .aliases = {"server", "gric-server", NULL},
        .banner = "gric-server - Native C micro-server for GRIC desktop GUI",
        .usage = "gric-server [options]",
        .description = "Serves the interactive simulator web UI and provides a native REST API\n"
                       "  for workspace file management and CLI execution.",
        .options =
            "  -p, --port <port>    HTTP listen port (default: 8080 or $GRIC_GUI_PORT)\n"
            "  -d, --dir <path>     Workspace directory (default: current directory)\n"
            "  -w, --docs <path>    Documentation directory path\n"
            "  -t <sec>             Inactivity timeout in seconds\n"
            "  --auto-shutdown      Exit when all browser client tabs disconnect\n"
            "  -v, --verbose        Enable verbose HTTP logging",
        .examples =
            "  $ gric-server -p 8080 -d ./workspace\n"
            "  $ gric-server --auto-shutdown"
    },
    {
        .canonical_name = "gric-tune",
        .aliases = {"tune", "gric-tune", NULL},
        .banner = "gric-tune - Parameter search and tuning utility for gric-cluster",
        .usage = "gric-tune <input_file> [options]",
        .description = "Explores parameter spaces across datasets to identify optimal\n"
                       "  hyperparameters for target clustering quality and throughput.",
        .options =
            "  -rlim_range <range>  Range of radius thresholds (min,max,step)\n"
            "  -tm_range <range>    Range of transition matrix weights (min,max,step)\n"
            "  -metric <type>       Optimization metric (speed|quality|balance)",
        .examples = "  $ gric-tune dataset.txt -rlim_range 0.1,1.0,0.05"
    },
    {
        .canonical_name = "gric-cluster-analysis",
        .aliases = {"analysis", "cluster-analysis", "gric-cluster-analysis", NULL},
        .banner = "gric-cluster-analysis - Offline diagnostic and quality analysis",
        .usage = "gric-cluster-analysis <cluster_output_dir> [options]",
        .description = "Analyzes cluster run logs, transition matrices, and memberships to report\n"
                       "  clustering quality, entropy, and trajectory continuity.",
        .options =
            "  -v                   Verbose analysis report\n"
            "  -plot                Generate summary diagnostic plots",
        .examples = "  $ gric-cluster-analysis my_run.clusterdat/"
    },
    {
        .canonical_name = "gric-gen-balls",
        .aliases = {"gen-balls", "gen_balls", "balls", "gric-gen-balls", NULL},
        .banner = "gric-gen-balls - Multi-body bouncing ball simulation generator",
        .usage = "gric-gen-balls [options] <output.fits>",
        .description = "Generates synthetic 3D FITS image cubes containing multi-body elastic\n"
                       "  bouncing ball collisions for multi-tile image tests.",
        .options =
            "  -n <num_balls>       Number of bouncing balls (default: 1)\n"
            "  -r <radius>          Ball radius in pixels (default: 5.0)\n"
            "  -W, -H <size>        Image dimensions (default: 32x32)\n"
            "  -f <frames>          Number of simulation frames (default: 500)\n"
            "  -s <seed>            Random number generator seed",
        .examples = "  $ gric-gen-balls -n 3 -r 5.0 -W 32 -H 32 -f 1000 balls_3.fits"
    },
    {
        .canonical_name = "gric-txt2stream",
        .aliases = {"txt2stream", "gric-txt2stream", NULL},
        .banner = "gric-txt2stream - Ingest ASCII coordinates into ImageStreamIO SHM",
        .usage = "gric-txt2stream <input.txt> <stream_name> [options]",
        .description = "Reads ASCII coordinate sequences from file and writes them into a live\n"
                       "  ImageStreamIO shared-memory circular ring buffer.",
        .options =
            "  -fps <val>           Target frame rate in frames/sec\n"
            "  -loop                Loop coordinate sequence continuously",
        .examples = "  $ gric-txt2stream spiral.txt spiral_stream -fps 100"
    }
};

/**
 * print_program_help() - Find and print detailed help for a specific program name or alias.
 * @prog: Name or alias of the program.
 *
 * Return: 0 if found and displayed, 1 on unknown program name.
 */
static int print_program_help(
    const char *prog)
{
    size_t num_docs = sizeof(PROGRAM_DOCS) / sizeof(PROGRAM_DOCS[0]);

    for (size_t i = 0; i < num_docs; i++)
    {
        const ProgramHelpDoc *doc = &PROGRAM_DOCS[i];
        if (strcmp(prog, doc->canonical_name) == 0)
        {
            print_formatted_help(
                doc->banner, doc->usage, doc->description, doc->options, doc->examples
            );
            return 0;
        }

        for (int a = 0; doc->aliases[a] != NULL; a++)
        {
            if (strcmp(prog, doc->aliases[a]) == 0)
            {
                print_formatted_help(
                    doc->banner, doc->usage, doc->description, doc->options, doc->examples
                );
                return 0;
            }
        }
    }

    fprintf(stderr, "%sError: Unknown program '%s'.%s\n",
            ANSI_COLOR_RED, prog, ANSI_COLOR_RESET);
    fprintf(stderr, "Run 'gric-help' to see the list of valid programs.\n");
    return 1;
}

/**
 * run_help() - Dispatch help request based on arguments.
 * @argc: Command argument count.
 * @argv: Command argument array.
 *
 * Return: 0 on success, 1 on error.
 */
static int run_help(
    int   argc,
    char *argv[])
{
    if (argc < 2)
    {
        print_general_help();
        return 0;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "help") == 0)
    {
        print_help_utility_self();
        return 0;
    }

    return print_program_help(argv[1]);
}

/**
 * main() - Entry point of the gric-help utility.
 * @argc: Command argument count.
 * @argv: Command argument array.
 *
 * Return: 0 on success, 1 on error.
 */
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
}

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

static const char *ansi_color_green = "";
static const char *ansi_color_red = "";
static const char *ansi_color_cyan = "";
static const char *ansi_color_reset = "";
static const char *ansi_bold = "";
static const char *ansi_underline = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_yellow = "";
static const char *ansi_color_grey = "";

#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RED     ansi_color_red
#define ANSI_COLOR_CYAN    ansi_color_cyan
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_UNDERLINE     ansi_underline
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_YELLOW  ansi_color_yellow
#define ANSI_COLOR_GREY    ansi_color_grey

/**
 * @brief Initializes ANSI color strings only if NO_COLOR is not present.
 */
static void init_colors(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_cyan = "\x1b[36m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_underline = "\x1b[4m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_grey = "\x1b[90m";
    }
} // init_colors

/**
 * @brief Prints a formatted section header.
 *
 * @param title     The text of the header.
 * @param use_color Whether to color the header or print plain text.
 */
static void print_header(
    const char *title,
    int         use_color)
{
    if (use_color)
    {
        printf("\n%s--- %s ---%s\n", ANSI_BOLD_CYAN, title, ANSI_COLOR_RESET);
    }
    else
    {
        printf("\n--- %s ---\n", title);
    }
} // print_header

static void print_color_mode(void)
{
    const char *no_color = getenv("NO_COLOR");
    printf("\n%sCOLOR MODE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    if (no_color == NULL)
    {
        printf("  %sENABLED%s (color escape codes are active; disable by setting NO_COLOR=1)\n",
               ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    }
    else
    {
        printf("  %sDISABLED%s (NO_COLOR environment variable is present)\n",
               ANSI_COLOR_RED, ANSI_COLOR_RESET);
    }
} // print_color_mode

static void print_colored_usage(const char *usage)
{
    printf("  ");
    const char *p = usage;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }

    const char *cmd_start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n')
    {
        p++;
    }
    printf("%s%.*s%s", ANSI_BOLD_GREEN, (int)(p - cmd_start), cmd_start, ANSI_COLOR_RESET);

    while (*p)
    {
        if (*p == '<')
        {
            const char *end = strchr(p, '>');
            if (end)
            {
                printf("%s%.*s%s", ANSI_COLOR_MAGENTA, (int)(end - p + 1), p, ANSI_COLOR_RESET);
                p = end + 1;
                continue;
            }
        }
        if (*p == '[')
        {
            const char *end = strchr(p, ']');
            if (end)
            {
                printf("%s%.*s%s", ANSI_COLOR_GREY, (int)(end - p + 1), p, ANSI_COLOR_RESET);
                p = end + 1;
                continue;
            }
        }
        putchar(*p);
        p++;
    }
    printf("\n");
}

static void print_colored_line(const char *line)
{
    const char *start = line;
    while (*start == ' ' || *start == '\t')
    {
        start++;
    }
    if (*start == '-')
    {
        printf("%.*s", (int)(start - line), line);

        const char *end = start;
        while (*end && *end != '\n')
        {
            if ((*end == ' ' && *(end + 1) == ' ') || *end == '\t')
            {
                break;
            }
            end++;
        }

        const char *p = start;
        while (p < end)
        {
            if (*p == '-')
            {
                const char *f_end = p;
                while (f_end < end && *f_end != ' ' && *f_end != '\t' && *f_end != ',' && *f_end != '<' && *f_end != '[')
                {
                    f_end++;
                }
                printf("%s%.*s%s", ANSI_COLOR_GREEN, (int)(f_end - p), p, ANSI_COLOR_RESET);
                p = f_end;
            }
            else if (*p == '<')
            {
                const char *v_end = p;
                while (v_end < end && *v_end != '>')
                {
                    v_end++;
                }
                if (v_end < end)
                {
                    printf("%s%.*s%s", ANSI_COLOR_MAGENTA, (int)(v_end - p + 1), p, ANSI_COLOR_RESET);
                    p = v_end + 1;
                }
                else
                {
                    putchar(*p);
                    p++;
                }
            }
            else
            {
                putchar(*p);
                p++;
            }
        }

        const char *desc = end;
        while (*desc)
        {
            if (strncmp(desc, "default:", 8) == 0 || strncmp(desc, "Default:", 8) == 0)
            {
                printf("%sdefault:%s", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
                desc += 8;
                const char *val_end = desc;
                while (*val_end && *val_end != ')' && *val_end != ',')
                {
                    val_end++;
                }
                printf("%s%.*s%s", ANSI_COLOR_CYAN, (int)(val_end - desc), desc, ANSI_COLOR_RESET);
                desc = val_end;
            }
            else if (strncmp(desc, "caution", 7) == 0 || strncmp(desc, "Caution", 7) == 0)
            {
                printf("%scaution%s", ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
                desc += 7;
            }
            else if (strncmp(desc, "[DISABLED]", 10) == 0)
            {
                printf("%s[DISABLED]%s", ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
                desc += 10;
            }
            else
            {
                putchar(*desc);
                desc++;
            }
        }
        printf("\n");
    }
    else
    {
        int leading_spaces = 0;
        while (line[leading_spaces] == ' ')
        {
            leading_spaces++;
        }
        if (leading_spaces == 2 && line[leading_spaces] != '\0')
        {
            printf("  %s%s%s\n", ANSI_BOLD, line + 2, ANSI_COLOR_RESET);
        }
        else
        {
            printf("%s\n", line);
        }
    }
}

static void print_colored_options(const char *options)
{
    const char *line = options;
    while (line && *line)
    {
        const char *next = strchr(line, '\n');
        int len = next ? (int)(next - line) : (int)strlen(line);
        char buf[1024];
        if (len >= (int)sizeof(buf))
        {
            len = sizeof(buf) - 1;
        }
        strncpy(buf, line, len);
        buf[len] = '\0';

        print_colored_line(buf);
        line = next ? next + 1 : NULL;
    }
}

static void print_colored_examples(const char *examples)
{
    const char *line = examples;
    while (line && *line)
    {
        const char *next = strchr(line, '\n');
        int len = next ? (int)(next - line) : (int)strlen(line);
        char buf[1024];
        if (len >= (int)sizeof(buf))
        {
            len = sizeof(buf) - 1;
        }
        strncpy(buf, line, len);
        buf[len] = '\0';

        const char *dollar = strstr(buf, "$ ");
        if (dollar)
        {
            printf("%.*s", (int)(dollar - buf), buf);
            printf("%s$%s ", ANSI_COLOR_GREY, ANSI_COLOR_RESET);

            const char *cmd = dollar + 2;
            const char *end = cmd;
            while (*end && *end != ' ' && *end != '\t')
            {
                end++;
            }
            printf("%s%.*s%s", ANSI_BOLD_GREEN, (int)(end - cmd), cmd, ANSI_COLOR_RESET);

            const char *p = end;
            while (*p)
            {
                if (*p == '-')
                {
                    const char *f_end = p;
                    while (*f_end && *f_end != ' ' && *f_end != '\t')
                      {
                        f_end++;
                      }
                    printf("%s%.*s%s", ANSI_COLOR_GREEN, (int)(f_end - p), p, ANSI_COLOR_RESET);
                    p = f_end;
                }
                else if (*p == '>' || *p == '<' || *p == '|' || *p == '&')
                {
                    printf("%s%c%s", ANSI_COLOR_GREY, *p, ANSI_COLOR_RESET);
                    p++;
                }
                else
                {
                    putchar(*p);
                    p++;
                }
            }
            printf("\n");
        }
        else
        {
            printf("%s\n", buf);
        }
        line = next ? next + 1 : NULL;
    }
}

/**
 * @brief Prints a standard formatted help screen.
 *
 * @param banner      Program name and brief description.
 * @param usage       Command-line syntax synopsis.
 * @param description Description of functionality and options.
 * @param options     Detailed list of supported flags, defaults, and arguments.
 * @param examples    One to three concrete usage examples.
 */
static void print_formatted_help(
    const char *banner,
    const char *usage,
    const char *description,
    const char *options,
    const char *examples)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s\n\n", banner);

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    print_colored_usage(usage);
    printf("\n");

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s\n\n", description);

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    print_colored_options(options);
    printf("\n");

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    print_colored_examples(examples);

    print_color_mode();
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
    printf("%s====================================================================%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    printf("%s                GRIC CLUSTER SUITE - ONBOARDING GUIDE               %s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    printf("%s====================================================================%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);

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
           ANSI_BOLD, "gric-cluster", ANSI_COLOR_RESET);
    printf("  %sVisualization%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates SVG/PNG diagnostic and summary plots.\n",
           ANSI_BOLD, "gric-plot", ANSI_COLOR_RESET);
    printf("  %sUtility & Simulation%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Prints build info, library paths, and enabled features.\n",
           ANSI_BOLD, "gric-info", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs N-dimensional coordinates from dcc.txt.\n",
           ANSI_BOLD, "gric-NDmodel", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Generates synthetic coordinate sequences (walk, spiral, etc.).\n",
           ANSI_BOLD, "gric-mktxtseq", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Converts 3D coordinates into MP4 video/ImageStreamIO streams.\n",
           ANSI_BOLD, "gric-ascii-spot-2-video", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Reconstructs a full clustered file from input and membership list.\n",
           ANSI_BOLD, "gric-mkclusteredfile", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Pipes raw data from live ImageStreamIO shared memory to stdout.\n",
           ANSI_BOLD, "gric-stream-to-pipe", ANSI_COLOR_RESET);
    printf("  %sMonitoring & Benchmarking%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
    printf("    %s%-24s%s Real-time SHM telemetry monitor (TUI dashboard).\n",
           ANSI_BOLD, "gric-status", ANSI_COLOR_RESET);
    printf("    %s%-24s%s Performance benchmarking across patterns and options.\n",
           ANSI_BOLD, "gric-benchmark", ANSI_COLOR_RESET);

    print_header("4. TYPICAL ONBOARDING WORKFLOW", 1);
    printf("  Follow these steps to familiarize yourself with GRIC:\n\n");
    printf("  %sStep 1: Check Optional Dependencies%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Check which input/output formats are enabled (FITS, PNG, FFmpeg, ImageStreamIO):\n");
    printf("      $ %sgric-info%s\n\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 2: Generate Synthetic Data%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Generate a 2D random walk sequence of 1000 points to act as fake coordinates:\n");
    printf("      $ %sgric-mktxtseq%s 1000 test_walk.txt 2Dwalk\n\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("  %sStep 3: Scan the Sequence Distances%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Measure distance statistics to choose a reasonable radius limit (rlim):\n");
    printf("      $ %sgric-cluster%s -scandist test_walk.txt\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("    Take note of the \"Median distance\" output from the scan.\n\n");

    printf("  %sStep 4: Run the Clustering%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Cluster the points using a radius limit of 1.5x the median distance:\n");
    printf("      $ %sgric-cluster%s a1.5 test_walk.txt -clustered > run.log\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    printf("    This generates an output directory: `test_walk.clusterdat/`.\n\n");

    printf("  %sStep 5: Plot results%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    Visualize the clusters and centroids using the plotting tool:\n");
    printf("      $ %sgric-plot%s test_walk.txt run.log plot.png\n\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);

    printf("  For detailed guide on a specific program, run:\n");
    printf("    $ %sgric-help%s <program-name>   (e.g. gric-help gric-cluster)\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    print_color_mode();
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

int main(
    int   argc,
    char *argv[])
{
    init_colors();

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
} // main

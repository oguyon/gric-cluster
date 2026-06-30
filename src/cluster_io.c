#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef USE_CFITSIO
#include <fitsio.h>
#endif
#include "cluster_io.h"
#include "frameread.h"

// Forward decl for PNG writing
#ifdef USE_PNG
void write_png_frame(const char *filename, double *data, int width, int height);
#endif

#include <unistd.h>

static const char *ansi_color_orange = "";
static const char *ansi_color_green = "";
static const char *ansi_color_red = "";
static const char *ansi_color_blue = "";
static const char *ansi_bg_green = "";
static const char *ansi_color_black = "";
static const char *ansi_color_reset = "";
static const char *ansi_bold = "";
static const char *ansi_underline = "";
static const char *ansi_bold_cyan = "";
static const char *ansi_bold_green = "";
static const char *ansi_color_magenta = "";
static const char *ansi_color_yellow = "";
static const char *ansi_color_grey = "";
static const char *ansi_color_cyan = "";

#define ANSI_COLOR_ORANGE  ansi_color_orange
#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RED     ansi_color_red
#define ANSI_COLOR_BLUE    ansi_color_blue
#define ANSI_BG_GREEN      ansi_bg_green
#define ANSI_COLOR_BLACK   ansi_color_black
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_UNDERLINE     ansi_underline
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_YELLOW  ansi_color_yellow
#define ANSI_COLOR_GREY    ansi_color_grey
#define ANSI_COLOR_CYAN    ansi_color_cyan
void init_colors_io(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
    {
        ansi_color_orange = "\x1b[38;5;208m";
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_blue = "\x1b[34m";
        ansi_bg_green = "\x1b[42m";
        ansi_color_black = "\x1b[30m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_underline = "\x1b[4m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;32m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_cyan = "\x1b[36m";
    }
} // init_colors_io

char *create_output_dir_name(const char *input_file)
{
    const char *base = strrchr(input_file, '/');
    if (base)
    {
        base++;
    }
    else
    {
        base = input_file;
    }
    char *name = strdup(base);
    if (!name)
        return NULL;
    size_t len = strlen(name);
    if (len > 8 && strcmp(name + len - 8, ".fits.fz") == 0)
        name[len - 8] = '\0';
    else if (len > 5 && strcmp(name + len - 5, ".fits") == 0)
        name[len - 5] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".mp4") == 0)
        name[len - 4] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".txt") == 0)
        name[len - 4] = '\0';

    size_t new_len = strlen(name) + strlen(".clusterdat") + 1;
    char *out_dir = (char *)malloc(new_len);
    if (out_dir)
        sprintf(out_dir, "%s.clusterdat", name);
    free(name);
    return out_dir;
}

void print_usage(char *progname)
{
    printf("Usage: %s [options] <rlim> <input_file|stream_name>\n", progname);
    printf("Try '%s -h' for more information.\n", progname);
}

void print_help_keyword(const char *keyword)
{
    int found = 0;

    // Normalize keyword (remove leading dashes)
    const char *key = keyword;
    while (*key == '-')
        key++;

    printf("%sHELP: %s%s\n\n", ANSI_BOLD, keyword, ANSI_COLOR_RESET);

    if (strcmp(key, "stream") == 0)
    {
        printf("%sRole:%s Input Source Selection\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "%sFunction:%s Specifies that the input is a shared memory stream via ImageStreamIO.\n",
            ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Instead of opening a file, the program attaches to an existing "
               "System V shared\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                memory segment and semaphore set managed by the ImageStreamIO "
               "library.\n");
        printf("                It treats the stream as a circular buffer of frames.\n");
        printf("%sUse:%s gric-cluster -stream <stream_name>\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "cnt2sync") == 0)
    {
        printf("%sRole:%s Stream Synchronization\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables synchronization using the 'cnt2' counter in ImageStreamIO.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Standard streaming reads whenever a new frame is available "
               "(cnt0 increments).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                With -cnt2sync, the program waits for the writer to increment "
               "'cnt0', processes\n");
        printf("                the frame, and then increments 'cnt2'. This allows the writer to "
               "wait for the\n");
        printf("                reader (handshake), ensuring no frames are dropped in a tightly "
               "coupled loop.\n");
        printf("%sUse:%s gric-cluster -stream my_stream -cnt2sync\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "dprob") == 0)
    {
        printf("%sRole:%s Cluster Probability Update (Recency Bias)\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sFunction:%s Amount added to a cluster's probability when a frame is assigned to "
               "it (Default: 0.01).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s The algorithm maintains a probability distribution P(c) over all "
               "clusters.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           When frame 'f' is assigned to cluster 'c_k':\n");
        printf("             P(c_k) = P(c_k) + dprob\n");
        printf("           Then all probabilities are re-normalized to sum to 1.0.\n");
        printf("           This creates a 'recency bias': active clusters rise to the top of the "
               "search list,\n");
        printf(
            "           minimizing the number of distance calculations needed to find a match.\n");
        printf("%sUse:%s -dprob 0.05 (Stronger bias, faster adaptation to changing scenes)\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxcl") == 0)
    {
        printf("%sRole:%s Resource Limiting\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Sets the maximum number of clusters allowed (Default: 1000).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Defines the size of static arrays (clusters, visitors) and the "
               "N*N distance\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                cache (dccarray). Affects memory usage (O(N^2) for dccarray).\n");
        printf("                When this limit is reached, the behavior is controlled by "
               "-maxcl_strategy.\n");
        printf("%sUse:%s -maxcl 5000\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "ncpu") == 0)
    {
        printf("%sRole:%s Parallel Processing\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Sets the number of OpenMP threads (Default: 1).\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sImplementation:%s Used to parallelize the 'pruning' loops. When checking if a "
               "candidate cluster\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                is valid, the algorithm checks triangle inequalities against all "
               "other clusters.\n");
        printf("                This loop is split across 'ncpu' threads. Also used in batch "
               "distance calculations.\n");
        printf("%sUse:%s -ncpu 4\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxcl_strategy") == 0)
    {
        printf("%sRole:%s Memory Management Strategy\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Determines behavior when the 'maxcl' limit is reached.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sOptions:%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("  stop    : (Default) Exit program. Ensures dataset integrity.\n");
        printf(
            "  discard : 'Cache Eviction'. Scans the oldest 'discard_frac' clusters and removes\n");
        printf("            the one with the fewest visits. Useful for continuous monitoring.\n");
        printf("  merge   : Merges the two geometrically closest clusters (min d(c_i, c_j)).\n");
        printf("            Computationally expensive (O(N^2) scan) but preserves information.\n");
        printf("%sUse:%s -maxcl 100 -maxcl_strategy discard\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "discard_frac") == 0)
    {
        printf("%sRole:%s Discard Strategy Parameter\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Fraction of oldest clusters to consider for discarding (Default: "
               "0.5).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s When discarding, we don't want to kill a brand new cluster "
               "that hasn't\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                had time to accumulate visitors. This options limits the search to "
               "the first\n");
        printf("                N * discard_frac clusters (the 'oldest' ones by index).\n");
        printf("%sUse:%s -discard_frac 0.2 (Only consider oldest 20%%)\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxim") == 0)
    {
        printf("%sRole:%s Execution Limit\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Process only the first N frames (Default: 100000).\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          Useful for testing on large datasets.\n");
        found = 1;
    }
    else if (strcmp(key, "gprob") == 0)
    {
        printf("%sRole:%s Geometric Probability (Trajectory Learning)\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sFunction:%s Uses historical distance patterns to predict cluster membership.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s For a new frame 'm', the algorithm looks at recent frames 'k' that "
               "share distance\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           measurements to common clusters. It computes a 'Geometrical Match "
               "Coefficient'\n");
        printf("           based on how similar the distance vector of 'm' is to 'k'.\n");
        printf("           If 'm' looks like 'k' geometrically, the probability of 'm' belonging "
               "to the same\n");
        printf("           cluster as 'k' is boosted.\n");
        printf("%sUse:%s -gprob (Highly recommended for continuous drift/trajectory data)\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "fmatcha") == 0)
    {
        printf("%sRole:%s Geometric Matching Parameter A\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Reward factor for exact geometric matches in gprob (Default: 2.0).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sEquation:%s factor = a - (a - b) * (delta_dist / rlim) / 2\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          If delta_dist is 0 (perfect match), factor = a.\n");
        found = 1;
    }
    else if (strcmp(key, "fmatchb") == 0)
    {
        printf("%sRole:%s Geometric Matching Parameter B\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Factor at the pruning limit for gprob (Default: 0.5).\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          If delta_dist is 2*rlim (limit of triangle inequality), factor = b.\n");
        found = 1;
    }
    else if (strcmp(key, "maxvis") == 0)
    {
        printf("%sRole:%s gprob History Limit\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "%sFunction:%s Max number of recent visitors to track per cluster (Default: 1000).\n",
            ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sDetails:%s To compute gprob, we scan past frames ('visitors') of candidate "
               "clusters.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("         This limits how many past frames are stored/scanned to maintain "
               "performance.\n");
        found = 1;
    }
    else if (strcmp(key, "pred") == 0 || strncmp(key, "pred", 4) == 0)
    {
        printf("%sRole:%s Time-Series Prediction\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Predicts next cluster based on sequence history.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sFormat:%s -pred[len,h,n]\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("  len: Length of recent sequence to match (Default: 10).\n");
        printf("  h  : History size to search (Default: 1000).\n");
        printf("  n  : Number of predicted candidates to test first (Default: 2).\n");
        printf("%sAlgorithm:%s Matches the last 'len' cluster assignments against the last 'h' "
               "frames.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "           If the sequence [A, B, C] is found in history followed by D, then D is\n");
        printf("           predicted as a candidate. Predicted candidates are checked *before* "
               "standard sorting.\n");
        printf("%sUse:%s -pred[5,500,1] (For repeating patterns/loops)\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "te4") == 0)
    {
        printf("%sRole:%s 4-Point Pruning\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables aggressive pruning using 4 points.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Standard pruning uses 3 points (Triangle Inequality: d(A,C) <= "
               "d(A,B) + d(B,C)).\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           TE4 uses 2 reference clusters (A, B) + Current Frame (F) + Candidate "
               "(C).\n");
        printf("           It establishes a 2D plane with A, B, F to bound the distance to C more "
               "strictly.\n");
        printf("           Reduces expensive distance calls at the cost of slightly more complex "
               "logic.\n");
        found = 1;
    }
    else if (strcmp(key, "te5") == 0)
    {
        printf("%sRole:%s 5-Point Pruning\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables aggressive pruning using 5 points.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Uses 3 reference clusters + Current Frame + Candidate.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("           It constructs a local 3D coordinate system to strictly bound the "
               "possible\n");
        printf("           distance range. Effective for high-dimensional data where simple "
               "triangle\n");
        printf("           inequalities are loose.\n");
        printf("%sUse:%s -te5 (Recommended for high-dimensional vectors)\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "scandist") == 0)
    {
        printf("%sRole:%s Data Analysis (Pre-run)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Measures distance statistics without clustering.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sImplementation:%s Computes distances between sequential frames (or random pairs) "
               "to build\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "                a histogram. It reports Min, Max, Median, 20%%, 80%% percentiles.\n");
        printf("                Use the Median or 20%% value to choose a good 'rlim'.\n");
        printf("%sUse:%s gric-cluster -scandist input.txt\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "outdir") == 0)
    {
        printf("%sRole:%s Output Management\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Specifies the directory for all output files.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          If not specified, a directory named '<input>.clusterdat' is created.\n");
        found = 1;
    }
    else if (strcmp(key, "avg") == 0)
    {
        printf("%sRole:%s Output Generation\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Computes the average frame for each cluster.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf(
            "%sImplementation:%s Accumulates pixel data for every frame assigned to a cluster.\n",
            ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                At the end, divides by the count. Useful for 'Lucky Imaging' or "
               "noise reduction.\n");
        found = 1;
    }
    else if (strcmp(key, "distall") == 0)
    {
        printf("%sRole:%s Debugging\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Saves every computed distance to 'distall.txt'.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          Format: ID1 ID2 Dist Ratio ClusterIdx Prob GProb\n");
        printf("%sWarning:%s Produces massive files for long runs.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "pngout") == 0)
    {
        printf("%sRole:%s Output Format\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Forces output (anchors, averages, frames) to be written as PNG "
               "images.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Requires libpng support.\n");
        found = 1;
    }
    else if (strcmp(key, "fitsout") == 0)
    {
        printf("%sRole:%s Output Format\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Forces output to be written as FITS (Flexible Image Transport "
               "System) files.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Standard in astronomy.\n");
        found = 1;
    }
    else if (strcmp(key, "dcc") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes the Distance Between Cluster Centers (DCC) matrix to "
               "'dcc.txt'.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format: Cluster_i Cluster_j Distance\n");
        found = 1;
    }
    else if (strcmp(key, "tm_out") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes the Transition Matrix to 'transition_matrix.txt'.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format: From_Cluster To_Cluster Count\n");
        found = 1;
    }
    else if (strcmp(key, "anchors") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "%sFunction:%s Writes the 'anchor' frame (the first frame) of each cluster to disk.\n",
            ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "counts") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'cluster_counts.txt' listing how many frames are in each "
               "cluster.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "membership") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'frame_membership.txt' (Default: Enabled).\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("          Contains a line for every frame: FrameIndex AssignedClusterIndex\n");
        found = 1;
    }
    else if (strcmp(key, "no_membership") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Disables writing 'frame_membership.txt'. Useful to save disk I/O.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "discarded") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "%sFunction:%s Writes list of discarded frames/clusters to 'discarded_frames.txt'.\n",
            ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Lists the frame indices that belonged to deleted clusters.\n");
        found = 1;
    }
    else if (strcmp(key, "clustered") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'filename.clustered.txt' containing ALL data grouped by "
               "cluster.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf(
            "          Format includes comments separating clusters. Good for plotting scripts.\n");
        found = 1;
    }
    else if (strcmp(key, "clusters") == 0)
    {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes individual files (or directories) for each cluster containing "
               "its member frames.\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "tm") == 0)
    {
        printf("%sRole:%s Transition Matrix Mixing\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Uses transition history to predict next cluster.\n", ANSI_BOLD,
               ANSI_COLOR_RESET);
        printf("%sUse:%s -tm <coeff> (0.0 to 1.0)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Mixes the standard probability with the transition probability:\n",
               ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           P_final = (1-coeff)*P_standard + coeff * P(next|prev)\n");
        printf(
            "           where P(next|prev) is derived from the count of transitions prev->next.\n");
        found = 1;
    }

    if (!found)
    {
        printf("No detailed help available for '%s'.\n", keyword);
        printf("Try running '%s -h' to see all options.\n", "gric-cluster");
    }
}

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

void print_help(char *progname)
{
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  gric-cluster - Clustering tool for image streams and sequences\n\n");

    printf("%sUSAGE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    char usage[256];
    snprintf(usage, sizeof(usage), "%s [options] <rlim> <input_file|stream_name>", progname);
    print_colored_usage(usage);
    printf("\n");

    printf("%sDESCRIPTION%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  Perform clustering on a stream of images or a pre-recorded file.\n");
    printf("  Supports FITS, MP4 (via ffmpeg), and raw text input.\n\n");

    printf("%sOPTIONS%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  (Use '%s -h <option>%s' for detailed help on a specific option)\n", progname,
           ANSI_COLOR_RESET);

    print_colored_line("  Input");
#ifdef USE_IMAGESTREAMIO
    print_colored_line("    -stream                  Input is an ImageStreamIO stream");
#else
    print_colored_line("    -stream                  Input is an ImageStreamIO stream [DISABLED]");
#endif
    print_colored_line("    -cnt2sync                Enable cnt2 synchronization (increment cnt2 after read)");

    print_colored_line("  Clustering Control");
    print_colored_line("    -dprob <val>             Delta probability (default: 0.01)");
    print_colored_line("    -maxcl <val>             Max number of clusters (default: 1000)");
    print_colored_line("    -ncpu <val>              Number of CPUs to use (default: 1)");
    print_colored_line("    -maxcl_strategy <str>    Strategy when maxcl reached (stop|discard|merge) (default: stop)");
    print_colored_line("    -discard_frac <val>      Fraction of oldest clusters to candidate for discard (default: 0.5)");
    print_colored_line("    -maxim <val>             Max number of frames (default: 100000)");
    print_colored_line("    -gprob                   Use geometrical probability");
    print_colored_line("    -fmatcha <val>           Set fmatch parameter a (default: 2.0)");
    print_colored_line("    -fmatchb <val>           Set fmatch parameter b (default: 0.5)");
    print_colored_line("    -maxvis <val>            Max visitors for gprob history (default: 1000)");
    print_colored_line("    -pred[l,h,n]             Prediction with pattern detection (default: 10,1000,2)");
    print_colored_line("                            l: length of pattern to match (recent cluster history)");
    print_colored_line("                            h: history size (how far back to search for pattern)");
    print_colored_line("                            n: number of prediction candidates to return");
    print_colored_line("    -te4                     Use 4-point triangle inequality pruning");
    print_colored_line("    -te5                     Use 5-point triangle inequality pruning");
    print_colored_line("    -conf <file>             Read options from configuration file");
    print_colored_line("    -confw <file>            Write current options to configuration file");

    print_colored_line("  Analysis & Debugging");
    print_colored_line("    -scandist                Measure distance stats");
    print_colored_line("    -progress                Print progress (default: enabled)");

    print_colored_line("  Output");
    print_colored_line("    -outdir <name>           Specify output directory (default: <filename>.clusterdat)");
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
    print_colored_line("    -dcc                     Enable dcc.txt output (default: disabled)");
    print_colored_line("    -tm_out                  Enable transition_matrix.txt output (default: disabled)");
    print_colored_line("    -anchors                 Enable anchors output (default: disabled)");
    print_colored_line("    -counts                  Enable cluster_counts.txt output (default: disabled)");
    print_colored_line("    -no_membership           Disable frame_membership.txt output");
    print_colored_line("    -membership              Enable frame_membership.txt output (default: enabled)");
    print_colored_line("    -discarded               Enable discarded_frames.txt output (default: disabled)");
    print_colored_line("    -clustered               Enable *.clustered.txt output (default: disabled)");
    print_colored_line("    -clusters                Enable individual cluster files (cluster_X) (default: disabled)\n");

    printf("%sEXAMPLES%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    printf("  %s$%s %sgric-cluster%s -scandist test_walk.txt\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    printf("  %s$%s %sgric-cluster%s a1.5 test_walk.txt -clustered %s>%s run.log\n", ANSI_COLOR_GREY, ANSI_COLOR_RESET,
           ANSI_BOLD_GREEN, ANSI_COLOR_RESET, ANSI_COLOR_GREY, ANSI_COLOR_RESET);
    print_color_mode();
}

void write_results(ClusterConfig *config, ClusterState *state)
{
    char *out_dir = NULL;
    if (config->user_outdir)
        out_dir = strdup(config->user_outdir);
    else
        out_dir = create_output_dir_name(config->fits_filename);

    if (!out_dir)
        return;

    char out_path[4096];

    // Write dcc.txt
    if (config->output_dcc)
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
                    double d = state->dccarray[i * config->maxnbclust + j];
                    if (d >= 0)
                        fprintf(dcc_out, "%d %d %.6f\n", i, j, d);
                }
            }
            fclose(dcc_out);
        }
    }

    // Write Transition Matrix
    if (config->output_tm && state->transition_matrix)
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
                    long val = state->transition_matrix[i * config->maxnbclust + j];
                    if (val > 0)
                    {
                        fprintf(tm_out, "%d %d %ld\n", i, j, val);
                    }
                }
            }
            fclose(tm_out);
        }
    }

    // Write Anchors
    long width = get_frame_width();
    long height = get_frame_height();
    long nelements = width * height;

    if (config->output_anchors)
    {
        printf("Writing anchors\n");
        if (config->pngout_mode)
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
        else if (is_ascii_input_mode() && !config->fitsout_mode)
        {
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    for (long k = 0; k < nelements; k++)
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
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
                fits_write_pix(afptr, TDOUBLE, fpixel, nelements, state->clusters[i].anchor.data,
                               &status);
            }
            fits_close_file(afptr, &status);
#else
            // Fallback to text if fits disabled but requested?
            fprintf(stderr,
                    "Warning: FITS output requested but not compiled in. Saving as ASCII.\n");
            // Reuse ASCII logic
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr)
            {
                for (int i = 0; i < state->num_clusters; i++)
                {
                    for (long k = 0; k < nelements; k++)
                        fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
#endif
        }
    }

    // Cluster Counts
    int *cluster_counts = (int *)calloc(state->num_clusters, sizeof(int));
    for (long i = 0; i < state->total_frames_processed; i++)
    {
        if (state->assignments[i] >= 0 && state->assignments[i] < state->num_clusters)
            cluster_counts[state->assignments[i]]++;
    }
    if (config->output_counts)
    {
        printf("Writing cluster_counts.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/cluster_counts.txt", out_dir);
        FILE *count_out = fopen(out_path, "w");
        if (count_out)
        {
            for (int c = 0; c < state->num_clusters; c++)
                fprintf(count_out, "Cluster %d: %d frames\n", c, cluster_counts[c]);
            fclose(count_out);
        }
    }

    // Average buffer
    double *avg_buffer = NULL;
    if (config->average_mode)
        avg_buffer = (double *)calloc(nelements, sizeof(double));

    int active_cluster_count = 0;
    for (int c = 0; c < state->num_clusters; c++)
    {
        if (cluster_counts[c] > 0)
            active_cluster_count++;
    }

    if (config->output_clusters)
    {
        printf("Writing cluster files (%d files)\n", active_cluster_count);
    }

    if (config->average_mode)
    {
        printf("Writing average cluster files\n");
    }

    if (config->pngout_mode)
    {
#ifdef USE_PNG
        for (int c = 0; c < state->num_clusters; c++)
        {
            if (cluster_counts[c] == 0)
                continue;

            if (config->output_clusters)
            {
                char cluster_dir[1024];
                snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                mkdir(cluster_dir, 0777);
            }

            if (config->average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;

            for (long f = 0; f < state->total_frames_processed; f++)
            {
                if (state->assignments[f] == c)
                {
                    Frame *fr = getframe_at(f);
                    if (fr)
                    {
                        if (config->output_clusters)
                        {
                            char cluster_dir[1024];
                            snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir,
                                     c);
                            snprintf(out_path, sizeof(out_path), "%s/frame%05ld.png", cluster_dir,
                                     f);
                            write_png_frame(out_path, fr->data, width, height);
                        }
                        if (config->average_mode)
                            for (long k = 0; k < nelements; k++)
                                avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                    }
                }
            }

            if (config->average_mode)
            {
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] /= cluster_counts[c];
                snprintf(out_path, sizeof(out_path), "%s/average_%04d.png", out_dir, c);
                write_png_frame(out_path, avg_buffer, width, height);
            }
        }
#endif
    }
    else if (is_ascii_input_mode() && !config->fitsout_mode)
    {
        FILE *avg_file = NULL;
        if (config->average_mode)
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
                        fprintf(avg_file, "0.0 ");
                    fprintf(avg_file, "\n");
                }
                continue;
            }

            FILE *cfptr = NULL;
            if (config->output_clusters)
            {
                char fname[1024];
                snprintf(fname, sizeof(fname), "%s/cluster_%d.txt", out_dir, c);
                cfptr = fopen(fname, "w");
            }

            if (config->average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;
            for (long f = 0; f < state->total_frames_processed; f++)
            {
                if (state->assignments[f] == c)
                {
                    Frame *fr = getframe_at(f);
                    if (fr)
                    {
                        for (long k = 0; k < nelements; k++)
                        {
                            if (cfptr)
                                fprintf(cfptr, "%f ", fr->data[k]);
                            if (config->average_mode)
                                avg_buffer[k] += fr->data[k];
                        }
                        if (cfptr)
                            fprintf(cfptr, "\n");
                        free_frame(fr);
                    }
                }
            }
            if (cfptr)
                fclose(cfptr);
            if (avg_file)
            {
                for (long k = 0; k < nelements; k++)
                    fprintf(avg_file, "%f ", avg_buffer[k] / cluster_counts[c]);
                fprintf(avg_file, "\n");
            }
        }
        if (avg_file)
            fclose(avg_file);
    }
    else
    {
#ifdef USE_CFITSIO
        int status = 0;
        fitsfile *avg_ptr = NULL;
        if (config->average_mode)
        {
            snprintf(out_path, sizeof(out_path), "!%s/average.fits", out_dir);
            fits_create_file(&avg_ptr, out_path, &status);
            long anaxes[3] = {width, height, state->num_clusters};
            fits_create_img(avg_ptr, DOUBLE_IMG, 3, anaxes, &status);
        }
        for (int c = 0; c < state->num_clusters; c++)
        {
            if (cluster_counts[c] == 0)
                continue;

            fitsfile *cfptr = NULL;
            if (config->output_clusters)
            {
                char fname[1024];
                snprintf(fname, sizeof(fname), "!%s/cluster_%d.fits", out_dir, c);
                fits_create_file(&cfptr, fname, &status);
                long cnaxes[3] = {width, height, cluster_counts[c]};
                fits_create_img(cfptr, DOUBLE_IMG, 3, cnaxes, &status);
            }

            if (config->average_mode)
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] = 0.0;
            int fr_count = 0;
            for (long f = 0; f < state->total_frames_processed; f++)
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
                        if (config->average_mode)
                            for (long k = 0; k < nelements; k++)
                                avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                        fr_count++;
                    }
                }
            }
            if (cfptr)
                fits_close_file(cfptr, &status);
            if (config->average_mode && avg_ptr)
            {
                for (long k = 0; k < nelements; k++)
                    avg_buffer[k] /= cluster_counts[c];
                long fpixel[3] = {1, 1, c + 1};
                fits_write_pix(avg_ptr, TDOUBLE, fpixel, nelements, avg_buffer, &status);
            }
        }
        if (avg_ptr)
            fits_close_file(avg_ptr, &status);
#else
// Fallback ASCII logic if FITS disabled but we reached here
// (Similar to block above)
#endif
    }

    if (avg_buffer)
        free(avg_buffer);

    if (config->output_clustered)
    {
        printf("Writing clustered output file\n");

        const char *base_name_only = strrchr(config->fits_filename, '/');
        if (base_name_only)
            base_name_only++;
        else
            base_name_only = config->fits_filename;

        char *temp_base = strdup(base_name_only);
        char *ext = strrchr(temp_base, '.');
        if (ext && strcmp(ext, ".txt") == 0)
            *ext = '\0';

        char *clustered_fname = (char *)malloc(strlen(out_dir) + strlen(temp_base) + 30);
        sprintf(clustered_fname, "%s/%s.clustered.txt", out_dir, temp_base);
        free(temp_base);

        FILE *clustered_out = fopen(clustered_fname, "w");
        if (clustered_out)
        {
            fprintf(clustered_out, "# Parameters:\n");
            fprintf(clustered_out, "# rlim %.6f\n", config->rlim);
            fprintf(clustered_out, "# dprob %.6f\n", config->deltaprob);
            fprintf(clustered_out, "# maxcl %d\n", config->maxnbclust);
            fprintf(clustered_out, "# maxim %ld\n", config->maxnbfr);
            fprintf(clustered_out, "# gprob_mode %d\n", config->gprob_mode);
            fprintf(clustered_out, "# fmatcha %.2f\n", config->fmatch_a);
            fprintf(clustered_out, "# fmatchb %.2f\n", config->fmatch_b);

            fprintf(clustered_out, "# Stats:\n");
            fprintf(clustered_out, "# Total Clusters %d\n", state->num_clusters);
            fprintf(clustered_out, "# Total Distance Computations %ld\n", state->framedist_calls);
            fprintf(clustered_out, "# Clusters Pruned %ld\n", state->clusters_pruned);
            double avg_dist = (state->total_frames_processed > 0)
                                  ? (double)state->framedist_calls / state->total_frames_processed
                                  : 0.0;
            fprintf(clustered_out, "# Avg Dist/Frame %.2f\n", avg_dist);

            if (state->pruned_fraction_sum && state->step_counts)
            {
                for (int k = 0; k < state->max_steps_recorded; k++)
                {
                    if (state->step_counts[k] > 0)
                    {
                        fprintf(clustered_out, "# Pruning Step %d: %.4f\n", k,
                                state->pruned_fraction_sum[k] / state->step_counts[k]);
                    }
                    else if (k > 0 && state->step_counts[k] == 0)
                    {
                        break;
                    }
                }
            }

            int next_new_cluster = 0;
            for (long i = 0; i < state->total_frames_processed; i++)
            {
                int assigned = state->assignments[i];
                if (assigned == next_new_cluster)
                {
                    fprintf(clustered_out, "# NEWCLUSTER %d %ld ", assigned, i);
                    for (long k = 0; k < nelements; k++)
                        fprintf(clustered_out, "%f ", state->clusters[assigned].anchor.data[k]);
                    fprintf(clustered_out, "\n");
                    next_new_cluster++;
                }
                Frame *fr = getframe_at(i);
                if (fr)
                {
                    fprintf(clustered_out, "%ld %d ", i, assigned);
                    for (long k = 0; k < nelements; k++)
                        fprintf(clustered_out, "%f ", fr->data[k]);
                    fprintf(clustered_out, "\n");
                    free_frame(fr);
                }
            }
            fclose(clustered_out);
        }
        free(clustered_fname);
    }
    free(cluster_counts);
    free(out_dir);
}

void write_run_log(ClusterConfig *config, ClusterState *state, const char *cmdline,
                   struct timespec start_ts, double clust_ms, double out_ms, long max_rss)
{
    char *out_dir = NULL;
    if (config->user_outdir)
        out_dir = strdup(config->user_outdir);
    else
        out_dir = create_output_dir_name(config->fits_filename);

    if (!out_dir)
        return;

    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", out_dir);
    FILE *f = fopen(log_path, "w");
    if (f)
    {
        char time_buf[64];
        struct tm *tm_info = localtime(&start_ts.tv_sec);
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

        fprintf(f, "CMD: %s\n", cmdline);
        fprintf(f, "START_TIME: %s.%09ld\n", time_buf, start_ts.tv_nsec);
        fprintf(f, "TIME_CLUSTERING_MS: %.3f\n", clust_ms);
        fprintf(f, "TIME_OUTPUT_MS: %.3f\n", out_ms);
        fprintf(f, "OUTPUT_DIR: %s\n", out_dir);
        fprintf(f, "PARAM_RLIM: %f\n", config->rlim);
        fprintf(f, "PARAM_DPROB: %f\n", config->deltaprob);
        fprintf(f, "PARAM_MAXCL: %d\n", config->maxnbclust);
        fprintf(f, "PARAM_MAXIM: %ld\n", config->maxnbfr);
        fprintf(f, "PARAM_GPROB: %d\n", config->gprob_mode);
        fprintf(f, "PARAM_FMATCHA: %f\n", config->fmatch_a);
        fprintf(f, "PARAM_FMATCHB: %f\n", config->fmatch_b);
        fprintf(f, "PARAM_TE4: %d\n", config->te4_mode);
        fprintf(f, "PARAM_TE5: %d\n", config->te5_mode);

        if (config->output_dcc)
            fprintf(f, "OUTPUT_FILE: %s/dcc.txt\n", out_dir);
        if (config->output_tm)
            fprintf(f, "OUTPUT_FILE: %s/transition_matrix.txt\n", out_dir);
        if (config->output_anchors)
            fprintf(f, "OUTPUT_FILE: %s/anchors.txt\n", out_dir);
        if (config->output_counts)
            fprintf(f, "OUTPUT_FILE: %s/cluster_counts.txt\n", out_dir);
        if (config->output_membership)
            fprintf(f, "OUTPUT_FILE: %s/frame_membership.txt\n", out_dir);

        if (config->output_clustered)
        {
            const char *base_name_only = strrchr(config->fits_filename, '/');
            if (base_name_only)
                base_name_only++;
            else
                base_name_only = config->fits_filename;
            char *temp_base = strdup(base_name_only);
            char *ext = strrchr(temp_base, '.');
            if (ext && strcmp(ext, ".txt") == 0)
                *ext = '\0';
            fprintf(f, "CLUSTERED_FILE: %s/%s.clustered.txt\n", out_dir, temp_base);
            free(temp_base);
        }

        fprintf(f, "STATS_CLUSTERS: %d\n", state->num_clusters);
        fprintf(f, "STATS_FRAMES: %ld\n", state->total_frames_processed);
        fprintf(f, "STATS_DISTS: %ld\n", state->framedist_calls);
        fprintf(f, "STATS_PRUNED: %ld\n", state->clusters_pruned);
        fprintf(f, "STATS_MAX_RSS_KB: %ld\n", max_rss);

        fprintf(f, "STATS_DIST_HIST_START\n");
        for (int k = 0; k <= config->maxnbclust; k++)
        {
            if (state->dist_counts && state->dist_counts[k] > 0)
            {
                fprintf(f, "%d %ld %ld\n", k, state->dist_counts[k],
                        state->pruned_counts_by_dist[k]);
            }
        }
        fprintf(f, "STATS_DIST_HIST_END\n");

        fclose(f);
        printf("Log written to %s\n", log_path);
    }
    free(out_dir);
}
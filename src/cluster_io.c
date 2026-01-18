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

#define ANSI_COLOR_ORANGE  "\x1b[38;5;208m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_BG_GREEN      "\x1b[42m"
#define ANSI_COLOR_BLACK   "\x1b[30m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"
#define ANSI_UNDERLINE     "\x1b[4m"

char* create_output_dir_name(const char* input_file) {
    const char *base = strrchr(input_file, '/');
    if (base) { base++; } else { base = input_file; }
    char *name = strdup(base);
    if (!name) return NULL;
    size_t len = strlen(name);
    if (len > 8 && strcmp(name + len - 8, ".fits.fz") == 0) name[len - 8] = '\0';
    else if (len > 5 && strcmp(name + len - 5, ".fits") == 0) name[len - 5] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".mp4") == 0) name[len - 4] = '\0';
    else if (len > 4 && strcmp(name + len - 4, ".txt") == 0) name[len - 4] = '\0';

    size_t new_len = strlen(name) + strlen(".clusterdat") + 1;
    char *out_dir = (char *)malloc(new_len);
    if (out_dir) sprintf(out_dir, "%s.clusterdat", name);
    free(name);
    return out_dir;
}

void print_usage(char *progname) {
    printf("Usage: %s [options] <rlim> <input_file|stream_name>\n", progname);
    printf("Try '%s -h' for more information.\n", progname);
}

void print_help_keyword(const char *keyword) {
    int found = 0;
    
    // Normalize keyword (remove leading dashes)
    const char *key = keyword;
    while (*key == '-') key++;

    printf("%sHELP: %s%s\n\n", ANSI_BOLD, keyword, ANSI_COLOR_RESET);

    if (strcmp(key, "stream") == 0) {
        printf("%sRole:%s Input Source Selection\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Specifies that the input is a shared memory stream via ImageStreamIO.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Instead of opening a file, the program attaches to an existing System V shared\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                memory segment and semaphore set managed by the ImageStreamIO library.\n");
        printf("                It treats the stream as a circular buffer of frames.\n");
        printf("%sUse:%s gric-cluster -stream <stream_name>\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "cnt2sync") == 0) {
        printf("%sRole:%s Stream Synchronization\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables synchronization using the 'cnt2' counter in ImageStreamIO.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Standard streaming reads whenever a new frame is available (cnt0 increments).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                With -cnt2sync, the program waits for the writer to increment 'cnt0', processes\n");
        printf("                the frame, and then increments 'cnt2'. This allows the writer to wait for the\n");
        printf("                reader (handshake), ensuring no frames are dropped in a tightly coupled loop.\n");
        printf("%sUse:%s gric-cluster -stream my_stream -cnt2sync\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "dprob") == 0) {
        printf("%sRole:%s Cluster Probability Update (Recency Bias)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Amount added to a cluster's probability when a frame is assigned to it (Default: 0.01).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s The algorithm maintains a probability distribution P(c) over all clusters.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           When frame 'f' is assigned to cluster 'c_k':\n");
        printf("             P(c_k) = P(c_k) + dprob\n");
        printf("           Then all probabilities are re-normalized to sum to 1.0.\n");
        printf("           This creates a 'recency bias': active clusters rise to the top of the search list,\n");
        printf("           minimizing the number of distance calculations needed to find a match.\n");
        printf("%sUse:%s -dprob 0.05 (Stronger bias, faster adaptation to changing scenes)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxcl") == 0) {
        printf("%sRole:%s Resource Limiting\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Sets the maximum number of clusters allowed (Default: 1000).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Defines the size of static arrays (clusters, visitors) and the N*N distance\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                cache (dccarray). Affects memory usage (O(N^2) for dccarray).\n");
        printf("                When this limit is reached, the behavior is controlled by -maxcl_strategy.\n");
        printf("%sUse:%s -maxcl 5000\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "ncpu") == 0) {
        printf("%sRole:%s Parallel Processing\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Sets the number of OpenMP threads (Default: 1).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Used to parallelize the 'pruning' loops. When checking if a candidate cluster\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                is valid, the algorithm checks triangle inequalities against all other clusters.\n");
        printf("                This loop is split across 'ncpu' threads. Also used in batch distance calculations.\n");
        printf("%sUse:%s -ncpu 4\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxcl_strategy") == 0) {
        printf("%sRole:%s Memory Management Strategy\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Determines behavior when the 'maxcl' limit is reached.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sOptions:%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("  stop    : (Default) Exit program. Ensures dataset integrity.\n");
        printf("  discard : 'Cache Eviction'. Scans the oldest 'discard_frac' clusters and removes\n");
        printf("            the one with the fewest visits. Useful for continuous monitoring.\n");
        printf("  merge   : Merges the two geometrically closest clusters (min d(c_i, c_j)).\n");
        printf("            Computationally expensive (O(N^2) scan) but preserves information.\n");
        printf("%sUse:%s -maxcl 100 -maxcl_strategy discard\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "discard_frac") == 0) {
        printf("%sRole:%s Discard Strategy Parameter\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Fraction of oldest clusters to consider for discarding (Default: 0.5).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s When discarding, we don't want to kill a brand new cluster that hasn't\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                had time to accumulate visitors. This options limits the search to the first\n");
        printf("                N * discard_frac clusters (the 'oldest' ones by index).\n");
        printf("%sUse:%s -discard_frac 0.2 (Only consider oldest 20%%)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "maxim") == 0) {
        printf("%sRole:%s Execution Limit\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Process only the first N frames (Default: 100000).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Useful for testing on large datasets.\n");
        found = 1;
    }
    else if (strcmp(key, "gprob") == 0) {
        printf("%sRole:%s Geometric Probability (Trajectory Learning)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Uses historical distance patterns to predict cluster membership.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s For a new frame 'm', the algorithm looks at recent frames 'k' that share distance\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           measurements to common clusters. It computes a 'Geometrical Match Coefficient'\n");
        printf("           based on how similar the distance vector of 'm' is to 'k'.\n");
        printf("           If 'm' looks like 'k' geometrically, the probability of 'm' belonging to the same\n");
        printf("           cluster as 'k' is boosted.\n");
        printf("%sUse:%s -gprob (Highly recommended for continuous drift/trajectory data)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "fmatcha") == 0) {
        printf("%sRole:%s Geometric Matching Parameter A\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Reward factor for exact geometric matches in gprob (Default: 2.0).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sEquation:%s factor = a - (a - b) * (delta_dist / rlim) / 2\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          If delta_dist is 0 (perfect match), factor = a.\n");
        found = 1;
    }
    else if (strcmp(key, "fmatchb") == 0) {
        printf("%sRole:%s Geometric Matching Parameter B\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Factor at the pruning limit for gprob (Default: 0.5).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          If delta_dist is 2*rlim (limit of triangle inequality), factor = b.\n");
        found = 1;
    }
    else if (strcmp(key, "maxvis") == 0) {
        printf("%sRole:%s gprob History Limit\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Max number of recent visitors to track per cluster (Default: 1000).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sDetails:%s To compute gprob, we scan past frames ('visitors') of candidate clusters.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("         This limits how many past frames are stored/scanned to maintain performance.\n");
        found = 1;
    }
    else if (strcmp(key, "pred") == 0 || strncmp(key, "pred", 4) == 0) {
        printf("%sRole:%s Time-Series Prediction\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Predicts next cluster based on sequence history.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFormat:%s -pred[len,h,n]\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("  len: Length of recent sequence to match (Default: 10).\n");
        printf("  h  : History size to search (Default: 1000).\n");
        printf("  n  : Number of predicted candidates to test first (Default: 2).\n");
        printf("%sAlgorithm:%s Matches the last 'len' cluster assignments against the last 'h' frames.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           If the sequence [A, B, C] is found in history followed by D, then D is\n");
        printf("           predicted as a candidate. Predicted candidates are checked *before* standard sorting.\n");
        printf("%sUse:%s -pred[5,500,1] (For repeating patterns/loops)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "te4") == 0) {
        printf("%sRole:%s 4-Point Pruning\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables aggressive pruning using 4 points.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Standard pruning uses 3 points (Triangle Inequality: d(A,C) <= d(A,B) + d(B,C)).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           TE4 uses 2 reference clusters (A, B) + Current Frame (F) + Candidate (C).\n");
        printf("           It establishes a 2D plane with A, B, F to bound the distance to C more strictly.\n");
        printf("           Reduces expensive distance calls at the cost of slightly more complex logic.\n");
        found = 1;
    }
    else if (strcmp(key, "te5") == 0) {
        printf("%sRole:%s 5-Point Pruning\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Enables aggressive pruning using 5 points.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Uses 3 reference clusters + Current Frame + Candidate.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           It constructs a local 3D coordinate system to strictly bound the possible\n");
        printf("           distance range. Effective for high-dimensional data where simple triangle\n");
        printf("           inequalities are loose.\n");
        printf("%sUse:%s -te5 (Recommended for high-dimensional vectors)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "scandist") == 0) {
        printf("%sRole:%s Data Analysis (Pre-run)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Measures distance statistics without clustering.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Computes distances between sequential frames (or random pairs) to build\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                a histogram. It reports Min, Max, Median, 20%%, 80%% percentiles.\n");
        printf("                Use the Median or 20%% value to choose a good 'rlim'.\n");
        printf("%sUse:%s gric-cluster -scandist input.txt\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "outdir") == 0) {
        printf("%sRole:%s Output Management\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Specifies the directory for all output files.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          If not specified, a directory named '<input>.clusterdat' is created.\n");
        found = 1;
    }
    else if (strcmp(key, "avg") == 0) {
        printf("%sRole:%s Output Generation\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Computes the average frame for each cluster.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sImplementation:%s Accumulates pixel data for every frame assigned to a cluster.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("                At the end, divides by the count. Useful for 'Lucky Imaging' or noise reduction.\n");
        found = 1;
    }
    else if (strcmp(key, "distall") == 0) {
        printf("%sRole:%s Debugging\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Saves every computed distance to 'distall.txt'.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format: ID1 ID2 Dist Ratio ClusterIdx Prob GProb\n");
        printf("%sWarning:%s Produces massive files for long runs.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "pngout") == 0) {
        printf("%sRole:%s Output Format\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Forces output (anchors, averages, frames) to be written as PNG images.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Requires libpng support.\n");
        found = 1;
    }
    else if (strcmp(key, "fitsout") == 0) {
        printf("%sRole:%s Output Format\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Forces output to be written as FITS (Flexible Image Transport System) files.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Standard in astronomy.\n");
        found = 1;
    }
    else if (strcmp(key, "dcc") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes the Distance Between Cluster Centers (DCC) matrix to 'dcc.txt'.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format: Cluster_i Cluster_j Distance\n");
        found = 1;
    }
    else if (strcmp(key, "tm_out") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes the Transition Matrix to 'transition_matrix.txt'.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format: From_Cluster To_Cluster Count\n");
        found = 1;
    }
    else if (strcmp(key, "anchors") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes the 'anchor' frame (the first frame) of each cluster to disk.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "counts") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'cluster_counts.txt' listing how many frames are in each cluster.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "membership") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'frame_membership.txt' (Default: Enabled).\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Contains a line for every frame: FrameIndex AssignedClusterIndex\n");
        found = 1;
    }
    else if (strcmp(key, "no_membership") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Disables writing 'frame_membership.txt'. Useful to save disk I/O.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "discarded") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes list of discarded frames/clusters to 'discarded_frames.txt'.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Lists the frame indices that belonged to deleted clusters.\n");
        found = 1;
    }
    else if (strcmp(key, "clustered") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes 'filename.clustered.txt' containing ALL data grouped by cluster.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("          Format includes comments separating clusters. Good for plotting scripts.\n");
        found = 1;
    }
    else if (strcmp(key, "clusters") == 0) {
        printf("%sRole:%s Output Control\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Writes individual files (or directories) for each cluster containing its member frames.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        found = 1;
    }
    else if (strcmp(key, "tm") == 0) {
        printf("%sRole:%s Transition Matrix Mixing\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sFunction:%s Uses transition history to predict next cluster.\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sUse:%s -tm <coeff> (0.0 to 1.0)\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("%sAlgorithm:%s Mixes the standard probability with the transition probability:\n", ANSI_BOLD, ANSI_COLOR_RESET);
        printf("           P_final = (1-coeff)*P_standard + coeff * P(next|prev)\n");
        printf("           where P(next|prev) is derived from the count of transitions prev->next.\n");
        found = 1;
    }
    
    if (!found) {
        printf("No detailed help available for '%s'.\n", keyword);
        printf("Try running '%s -h' to see all options.\n", "gric-cluster");
    }
}

void print_help(char *progname) {
    printf("%sNAME%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  gric-cluster - Clustering tool for image streams and sequences\n\n");

    printf("%sSYNOPSIS%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  %s [options] <rlim> <input_file|stream_name>\n\n", progname);

    printf("%sDESCRIPTION%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  Perform clustering on a stream of images or a pre-recorded file.\n");
    printf("  Supports FITS, MP4 (via ffmpeg), and raw text input.\n");

    printf("\n%sOPTIONS%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("  (Use '%s -h <option>%s' for detailed help on a specific option)\n", progname, ANSI_COLOR_RESET);

    printf("\n  %sInput%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-stream%s                  Input is an ImageStreamIO stream", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    #ifndef USE_IMAGESTREAMIO
    printf(" [DISABLED]");
    #endif
    printf("\n");
    printf("    %s%s-cnt2sync%s                Enable cnt2 synchronization (increment cnt2 after read)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);

    printf("\n  %sClustering Control%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-dprob <val>%s             Delta probability (default: 0.01)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-maxcl <val>%s             Max number of clusters (default: 1000)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-ncpu <val>%s              Number of CPUs to use (default: 1)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-maxcl_strategy <str>%s    Strategy when maxcl reached (stop|discard|merge) (default: stop)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-discard_frac <val>%s      Fraction of oldest clusters to candidate for discard (default: 0.5)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-maxim <val>%s             Max number of frames (default: 100000)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-gprob%s                   Use geometrical probability\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-fmatcha <val>%s           Set fmatch parameter a (default: 2.0)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-fmatchb <val>%s           Set fmatch parameter b (default: 0.5)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-maxvis <val>%s            Max visitors for gprob history (default: 1000)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-pred[l,h,n]%s             Prediction with pattern detection (default: 10,1000,2)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("                            l: length of pattern to match (recent cluster history)\n");
    printf("                            h: history size (how far back to search for pattern)\n");
    printf("                            n: number of prediction candidates to return\n");
    printf("    %s%s-te4%s                     Use 4-point triangle inequality pruning\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-te5%s                     Use 5-point triangle inequality pruning\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-conf <file>%s             Read options from configuration file\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-confw <file>%s            Write current options to configuration file\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);

    printf("\n  %sAnalysis & Debugging%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-scandist%s                Measure distance stats\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-progress%s                Print progress (default: enabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);

    printf("\n  %sOutput%s\n", ANSI_BOLD, ANSI_COLOR_RESET);
    printf("    %s%s-outdir <name>%s           Specify output directory (default: <filename>.clusterdat)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-avg%s                     Compute average frame per cluster\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-distall%s                 Save all computed distances\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-pngout%s                  Write output as PNG images", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    #ifndef USE_PNG
    printf(" [DISABLED]");
    #endif
    printf("\n");
    printf("    %s%s-fitsout%s                 Force FITS output format", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    #ifndef USE_CFITSIO
    printf(" [DISABLED]");
    #endif
    printf("\n");
    printf("    %s%s-dcc%s                     Enable dcc.txt output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-tm_out%s                  Enable transition_matrix.txt output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-anchors%s                 Enable anchors output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-counts%s                  Enable cluster_counts.txt output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-no_membership%s           Disable frame_membership.txt output\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-membership%s              Enable frame_membership.txt output (default: enabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-discarded%s               Enable discarded_frames.txt output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-clustered%s               Enable *.clustered.txt output (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("    %s%s-clusters%s                Enable individual cluster files (cluster_X) (default: disabled)\n", ANSI_BOLD, ANSI_UNDERLINE, ANSI_COLOR_RESET);
    printf("\n");
}

void write_results(ClusterConfig *config, ClusterState *state) {
    char *out_dir = NULL;
    if (config->user_outdir) out_dir = strdup(config->user_outdir);
    else out_dir = create_output_dir_name(config->fits_filename);

    if (!out_dir) return;

    char out_path[4096];

    // Write dcc.txt
    if (config->output_dcc) {
        printf("Writing dcc.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/dcc.txt", out_dir);
        FILE *dcc_out = fopen(out_path, "w");
        if (dcc_out) {
            for (int i = 0; i < state->num_clusters; i++) {
                for (int j = 0; j < state->num_clusters; j++) {
                    double d = state->dccarray[i * config->maxnbclust + j];
                    if (d >= 0) fprintf(dcc_out, "%d %d %.6f\n", i, j, d);
                }
            }
            fclose(dcc_out);
        }
    }

    // Write Transition Matrix
    if (config->output_tm && state->transition_matrix) {
        printf("Writing transition_matrix.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/transition_matrix.txt", out_dir);
        FILE *tm_out = fopen(out_path, "w");
        if (tm_out) {
            for (int i = 0; i < state->num_clusters; i++) {
                for (int j = 0; j < state->num_clusters; j++) {
                    long val = state->transition_matrix[i * config->maxnbclust + j];
                    if (val > 0) {
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

    if (config->output_anchors) {
        printf("Writing anchors\n");
        if (config->pngout_mode) {
            #ifdef USE_PNG
            for (int i = 0; i < state->num_clusters; i++) {
                snprintf(out_path, sizeof(out_path), "%s/anchor_%04d.png", out_dir, i);
                write_png_frame(out_path, state->clusters[i].anchor.data, width, height);
            }
            #else
            fprintf(stderr, "Warning: PNG output requested but not compiled in.\n");
            #endif
        } else if (is_ascii_input_mode() && !config->fitsout_mode) {
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr) {
                for (int i = 0; i < state->num_clusters; i++) {
                    for (long k = 0; k < nelements; k++) fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
        } else {
            #ifdef USE_CFITSIO
            int status = 0;
            fitsfile *afptr;
            snprintf(out_path, sizeof(out_path), "!%s/anchors.fits", out_dir);
            fits_create_file(&afptr, out_path, &status);
            long naxes[3] = { width, height, state->num_clusters };
            fits_create_img(afptr, DOUBLE_IMG, 3, naxes, &status);
            for (int i = 0; i < state->num_clusters; i++) {
                long fpixel[3] = {1, 1, i + 1};
                fits_write_pix(afptr, TDOUBLE, fpixel, nelements, state->clusters[i].anchor.data, &status);
            }
            fits_close_file(afptr, &status);
            #else
            // Fallback to text if fits disabled but requested?
            fprintf(stderr, "Warning: FITS output requested but not compiled in. Saving as ASCII.\n");
            // Reuse ASCII logic
            snprintf(out_path, sizeof(out_path), "%s/anchors.txt", out_dir);
            FILE *afptr = fopen(out_path, "w");
            if (afptr) {
                for (int i = 0; i < state->num_clusters; i++) {
                    for (long k = 0; k < nelements; k++) fprintf(afptr, "%f ", state->clusters[i].anchor.data[k]);
                    fprintf(afptr, "\n");
                }
                fclose(afptr);
            }
            #endif
        }
    }

    // Cluster Counts
    int *cluster_counts = (int *)calloc(state->num_clusters, sizeof(int));
    for (long i = 0; i < state->total_frames_processed; i++) {
        if (state->assignments[i] >= 0 && state->assignments[i] < state->num_clusters)
            cluster_counts[state->assignments[i]]++;
    }
    if (config->output_counts) {
        printf("Writing cluster_counts.txt\n");
        snprintf(out_path, sizeof(out_path), "%s/cluster_counts.txt", out_dir);
        FILE *count_out = fopen(out_path, "w");
        if (count_out) {
            for (int c = 0; c < state->num_clusters; c++) fprintf(count_out, "Cluster %d: %d frames\n", c, cluster_counts[c]);
            fclose(count_out);
        }
    }

    // Average buffer
    double *avg_buffer = NULL;
    if (config->average_mode) avg_buffer = (double *)calloc(nelements, sizeof(double));

    int active_cluster_count = 0;
    for (int c = 0; c < state->num_clusters; c++) {
        if (cluster_counts[c] > 0) active_cluster_count++;
    }

    if (config->output_clusters) {
        printf("Writing cluster files (%d files)\n", active_cluster_count);
    }

    if (config->average_mode) {
        printf("Writing average cluster files\n");
    }

    if (config->pngout_mode) {
        #ifdef USE_PNG
        for (int c = 0; c < state->num_clusters; c++) {
            if (cluster_counts[c] == 0) continue;

            if (config->output_clusters) {
                char cluster_dir[1024];
                snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                mkdir(cluster_dir, 0777);
            }

            if (config->average_mode) for (long k=0; k<nelements; k++) avg_buffer[k] = 0.0;

            for (long f = 0; f < state->total_frames_processed; f++) {
                if (state->assignments[f] == c) {
                    Frame *fr = getframe_at(f);
                    if (fr) {
                        if (config->output_clusters) {
                            char cluster_dir[1024];
                            snprintf(cluster_dir, sizeof(cluster_dir), "%s/cluster_%04d", out_dir, c);
                            snprintf(out_path, sizeof(out_path), "%s/frame%05ld.png", cluster_dir, f);
                            write_png_frame(out_path, fr->data, width, height);
                        }
                        if (config->average_mode) for (long k=0; k<nelements; k++) avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                    }
                }
            }

            if (config->average_mode) {
                for (long k=0; k<nelements; k++) avg_buffer[k] /= cluster_counts[c];
                snprintf(out_path, sizeof(out_path), "%s/average_%04d.png", out_dir, c);
                write_png_frame(out_path, avg_buffer, width, height);
            }
        }
        #endif
    } else if (is_ascii_input_mode() && !config->fitsout_mode) {
        FILE *avg_file = NULL;
        if (config->average_mode) {
            snprintf(out_path, sizeof(out_path), "%s/average.txt", out_dir);
            avg_file = fopen(out_path, "w");
        }
        for (int c = 0; c < state->num_clusters; c++) {
            if (cluster_counts[c] == 0) {
                if (avg_file) { for(long k=0; k<nelements; k++) fprintf(avg_file, "0.0 "); fprintf(avg_file, "\n"); }
                continue;
            }
            
            FILE *cfptr = NULL;
            if (config->output_clusters) {
                char fname[1024];
                snprintf(fname, sizeof(fname), "%s/cluster_%d.txt", out_dir, c);
                cfptr = fopen(fname, "w");
            }
            
            if (config->average_mode) for(long k=0; k<nelements; k++) avg_buffer[k] = 0.0;
            for (long f = 0; f < state->total_frames_processed; f++) {
                if (state->assignments[f] == c) {
                    Frame *fr = getframe_at(f);
                    if (fr) {
                        for(long k=0; k<nelements; k++) {
                            if(cfptr) fprintf(cfptr, "%f ", fr->data[k]);
                            if(config->average_mode) avg_buffer[k] += fr->data[k];
                        }
                        if(cfptr) fprintf(cfptr, "\n");
                        free_frame(fr);
                    }
                }
            }
            if(cfptr) fclose(cfptr);
            if (avg_file) {
                for(long k=0; k<nelements; k++) fprintf(avg_file, "%f ", avg_buffer[k]/cluster_counts[c]);
                fprintf(avg_file, "\n");
            }
        }
        if (avg_file) fclose(avg_file);

    } else {
        #ifdef USE_CFITSIO
        int status = 0;
        fitsfile *avg_ptr = NULL;
        if (config->average_mode) {
            snprintf(out_path, sizeof(out_path), "!%s/average.fits", out_dir);
            fits_create_file(&avg_ptr, out_path, &status);
            long anaxes[3] = { width, height, state->num_clusters };
            fits_create_img(avg_ptr, DOUBLE_IMG, 3, anaxes, &status);
        }
        for (int c = 0; c < state->num_clusters; c++) {
            if (cluster_counts[c] == 0) continue;
            
            fitsfile *cfptr = NULL;
            if (config->output_clusters) {
                char fname[1024];
                snprintf(fname, sizeof(fname), "!%s/cluster_%d.fits", out_dir, c);
                fits_create_file(&cfptr, fname, &status);
                long cnaxes[3] = { width, height, cluster_counts[c] };
                fits_create_img(cfptr, DOUBLE_IMG, 3, cnaxes, &status);
            }

            if (config->average_mode) for(long k=0; k<nelements; k++) avg_buffer[k] = 0.0;
            int fr_count = 0;
            for (long f = 0; f < state->total_frames_processed; f++) {
                if (state->assignments[f] == c) {
                    Frame *fr = getframe_at(f);
                    if (fr) {
                        if (cfptr) {
                            long fpixel[3] = {1, 1, fr_count + 1};
                            fits_write_pix(cfptr, TDOUBLE, fpixel, nelements, fr->data, &status);
                        }
                        if(config->average_mode) for(long k=0; k<nelements; k++) avg_buffer[k] += fr->data[k];
                        free_frame(fr);
                        fr_count++;
                    }
                }
            }
            if (cfptr) fits_close_file(cfptr, &status);
            if (config->average_mode && avg_ptr) {
                for(long k=0; k<nelements; k++) avg_buffer[k] /= cluster_counts[c];
                long fpixel[3] = {1, 1, c + 1};
                fits_write_pix(avg_ptr, TDOUBLE, fpixel, nelements, avg_buffer, &status);
            }
        }
        if (avg_ptr) fits_close_file(avg_ptr, &status);
        #else
        // Fallback ASCII logic if FITS disabled but we reached here
        // (Similar to block above)
        #endif
    }

    if (avg_buffer) free(avg_buffer);

    if (config->output_clustered) {
        printf("Writing clustered output file\n");
        
        const char *base_name_only = strrchr(config->fits_filename, '/');
        if (base_name_only) base_name_only++;
        else base_name_only = config->fits_filename;

        char *temp_base = strdup(base_name_only);
        char *ext = strrchr(temp_base, '.');
        if (ext && strcmp(ext, ".txt") == 0) *ext = '\0';

        char *clustered_fname = (char *)malloc(strlen(out_dir) + strlen(temp_base) + 30);
        sprintf(clustered_fname, "%s/%s.clustered.txt", out_dir, temp_base);
        free(temp_base);

        FILE *clustered_out = fopen(clustered_fname, "w");
        if (clustered_out) {
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
            double avg_dist = (state->total_frames_processed > 0) ? (double)state->framedist_calls / state->total_frames_processed : 0.0;
            fprintf(clustered_out, "# Avg Dist/Frame %.2f\n", avg_dist);

            if (state->pruned_fraction_sum && state->step_counts) {
                for (int k = 0; k < state->max_steps_recorded; k++) {
                    if (state->step_counts[k] > 0) {
                        fprintf(clustered_out, "# Pruning Step %d: %.4f\n", k, state->pruned_fraction_sum[k] / state->step_counts[k]);
                    } else if (k > 0 && state->step_counts[k] == 0) {
                        break;
                    }
                }
            }

            int next_new_cluster = 0;
            for (long i = 0; i < state->total_frames_processed; i++) {
                int assigned = state->assignments[i];
                if (assigned == next_new_cluster) {
                    fprintf(clustered_out, "# NEWCLUSTER %d %ld ", assigned, i);
                    for (long k = 0; k < nelements; k++) fprintf(clustered_out, "%f ", state->clusters[assigned].anchor.data[k]);
                    fprintf(clustered_out, "\n");
                    next_new_cluster++;
                }
                Frame *fr = getframe_at(i);
                if (fr) {
                    fprintf(clustered_out, "%ld %d ", i, assigned);
                    for (long k = 0; k < nelements; k++) fprintf(clustered_out, "%f ", fr->data[k]);
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

void write_run_log(ClusterConfig *config, ClusterState *state, const char *cmdline, struct timespec start_ts, double clust_ms, double out_ms, long max_rss) {
    char *out_dir = NULL;
    if (config->user_outdir) out_dir = strdup(config->user_outdir);
    else out_dir = create_output_dir_name(config->fits_filename);

    if (!out_dir) return;

    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/cluster_run.log", out_dir);
    FILE *f = fopen(log_path, "w");
    if (f) {
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
        
        if (config->output_dcc) fprintf(f, "OUTPUT_FILE: %s/dcc.txt\n", out_dir);
        if (config->output_tm) fprintf(f, "OUTPUT_FILE: %s/transition_matrix.txt\n", out_dir);
        if (config->output_anchors) fprintf(f, "OUTPUT_FILE: %s/anchors.txt\n", out_dir);
        if (config->output_counts) fprintf(f, "OUTPUT_FILE: %s/cluster_counts.txt\n", out_dir);
        if (config->output_membership) fprintf(f, "OUTPUT_FILE: %s/frame_membership.txt\n", out_dir);

        if (config->output_clustered) {
            const char *base_name_only = strrchr(config->fits_filename, '/');
            if (base_name_only) base_name_only++;
            else base_name_only = config->fits_filename;
            char *temp_base = strdup(base_name_only);
            char *ext = strrchr(temp_base, '.');
            if (ext && strcmp(ext, ".txt") == 0) *ext = '\0';
            fprintf(f, "CLUSTERED_FILE: %s/%s.clustered.txt\n", out_dir, temp_base);
            free(temp_base);
        }

        fprintf(f, "STATS_CLUSTERS: %d\n", state->num_clusters);
        fprintf(f, "STATS_FRAMES: %ld\n", state->total_frames_processed);
        fprintf(f, "STATS_DISTS: %ld\n", state->framedist_calls);
        fprintf(f, "STATS_PRUNED: %ld\n", state->clusters_pruned);
        fprintf(f, "STATS_MAX_RSS_KB: %ld\n", max_rss);

        fprintf(f, "STATS_DIST_HIST_START\n");
        for (int k = 0; k <= config->maxnbclust; k++) {
            if (state->dist_counts && state->dist_counts[k] > 0) {
                fprintf(f, "%d %ld %ld\n", k, state->dist_counts[k], state->pruned_counts_by_dist[k]);
            }
        }
        fprintf(f, "STATS_DIST_HIST_END\n");
        
        fclose(f);
        printf("Log written to %s\n", log_path);
    }
    free(out_dir);
}
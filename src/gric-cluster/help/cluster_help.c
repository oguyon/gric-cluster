/**
 * @file cluster_help.c
 * @brief Implementations of clustering command help screens.
 *
 * Implements general program help, keyword detailed descriptions,
 * and colored text formatting for options printout.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_help.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared/cli_colors.h"

void init_colors_help(void)
{
    cli_colors_init();
}

void print_usage(
    char *progname)
{
    printf("Usage: %s [options] <rlim> <input_file|stream_name>\n", progname);
    printf("Try '%s -h' for more information.\n", progname);
}

static void print_see_also_option(
    const char *option,
    const char *desc)
{
    cli_print_see_also_option(option, desc);
}

/**
 * print_rich_segment - print text with inline highlighting
 * @text:    pointer to start of segment
 * @len:     number of bytes to print
 * @is_bold: nonzero if the surrounding line is bold
 *
 * Auto-highlights:
 *   gric-*  executables → yellow
 *   -option flags       → bold green
 *   <placeholder>       → magenta
 */
static void print_rich_segment(
    const char *text,
    int         len,
    int         is_bold)
{
    const char *p = text;
    const char *end = text + len;

    while (p < end)
    {
        /* <placeholder> → magenta */
        if (*p == '<')
        {
            const char *close = memchr(
                p, '>', (size_t)(end - p));
            if (close)
            {
                printf("%s%.*s%s",
                       ANSI_COLOR_MAGENTA,
                       (int)(close - p + 1), p,
                       ANSI_COLOR_RESET);
                if (is_bold)
                {
                    printf("%s", ANSI_BOLD);
                }
                p = close + 1;
                continue;
            }
        }

        /* gric-* executable → yellow */
        if ((p + 5 <= end)
            && memcmp(p, "gric-", 5) == 0
            && (p == text
                || *(p - 1) == ' '
                || *(p - 1) == '\t'))
        {
            const char *we = p + 5;
            while (we < end
                   && *we != ' '
                   && *we != '\t'
                   && *we != '\n')
            {
                we++;
            }
            printf("%s%.*s%s",
                   ANSI_BOLD_GREEN,
                   (int)(we - p), p,
                   ANSI_COLOR_RESET);
            if (is_bold)
            {
                printf("%s", ANSI_BOLD);
            }
            p = we;
            continue;
        }

        /* -option → bold green */
        if (*p == '-'
            && (p == text
                || *(p - 1) == ' '
                || *(p - 1) == '(')
            && (p + 1 < end)
            && (*(p + 1) >= 'a' && *(p + 1) <= 'z'))
        {
            const char *oe = p + 1;
            while (oe < end
                   && *oe != ' '
                   && *oe != '\t'
                   && *oe != ','
                   && *oe != ')'
                   && *oe != '\n')
            {
                oe++;
            }
            printf("%s%.*s%s",
                   ANSI_COLOR_GREEN,
                   (int)(oe - p), p,
                   ANSI_COLOR_RESET);
            if (is_bold)
            {
                printf("%s", ANSI_BOLD);
            }
            p = oe;
            continue;
        }

        putchar(*p);
        p++;
    }
}

static void print_help_section(
    const char *label,
    const char *value)
{
    // Print the colored section label on its own line
    printf("%s%s%s\n",
           ANSI_BOLD_CYAN, label, ANSI_COLOR_RESET);

    // Print each line of value, indented by 2 spaces.
    // A line is a bold sub-header if it starts with
    // non-whitespace AND the next line starts with a
    // space (indented content follows it).
    const char *line_start = value;
    const char *p = value;

    while (*p != '\0')
    {
        if (*p == '\n')
        {
            int ll = (int)(p - line_start);
            const char *next = p + 1;
            int next_indented = (*next == ' ');
            int is_sub = (ll > 0
                          && line_start[0] != ' '
                          && next_indented);
            printf("  ");
            if (is_sub)
            {
                printf("%s", ANSI_BOLD);
            }
            print_rich_segment(
                line_start, ll, is_sub);
            if (is_sub)
            {
                printf("%s", ANSI_COLOR_RESET);
            }
            printf("\n");
            line_start = p + 1;
        }
        p++;
    }

    // Print the last line if there's no trailing newline
    if (p > line_start)
    {
        int ll = (int)(p - line_start);
        printf("  ");
        print_rich_segment(line_start, ll, 0);
        printf("\n");
    }

    // Print a trailing blank line to separate sections
    printf("\n");
}

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

/**
 * print_keyword_content - print the detailed help
 * @key: normalized keyword (no leading dashes)
 *
 * Returns 1 if the keyword matched, 0 otherwise.
 */
static int print_keyword_content(
    const char *key)
{
    if (strcmp(key, "stream") == 0)
    {
        print_help_section("ROLE", "Input Source Selection");
        print_help_section(
            "FUNCTION",
            "Specifies that the input is a shared memory stream via ImageStreamIO.");
        print_help_section(
            "IMPLEMENTATION",
            "Instead of opening a file, the program attaches to an existing System V\n"
            "shared memory segment and semaphore set managed by the ImageStreamIO\n"
            "library. It treats the stream as a circular buffer of frames.");
        print_help_section("USE", "gric-cluster -stream <stream_name>");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-cnt2sync",
            "Enable cnt2 synchronization (increment cnt2 after read)");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "cnt2sync") == 0)
    {
        print_help_section("ROLE", "Stream Synchronization");
        print_help_section(
            "FUNCTION",
            "Enables synchronization using the 'cnt2' counter in ImageStreamIO.");
        print_help_section(
            "IMPLEMENTATION",
            "Standard streaming reads whenever a new frame is available (cnt0 increments).\n"
            "With -cnt2sync, the program waits for the writer to increment 'cnt0', processes\n"
            "the frame, and then increments 'cnt2'. This allows the writer to wait for the\n"
            "reader (handshake), ensuring no frames are dropped in a tightly coupled loop.");
        print_help_section("USE", "gric-cluster -stream my_stream -cnt2sync");
        print_help_section(
            "REQUIRES",
            "-stream\n"
            "  Only meaningful in streaming mode.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-stream",
            "Input is an ImageStreamIO stream");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "dprob") == 0)
    {
        print_help_section("ROLE", "Cluster Probability Update (Recency Bias)");
        print_help_section(
            "FUNCTION",
            "Amount added to a cluster's probability when a frame is assigned\n"
            "to it (Default: 0.01).");
        print_help_section(
            "ALGORITHM",
            "The algorithm maintains a probability distribution P(c) over all clusters.\n"
            "When frame 'f' is assigned to cluster 'c_k':\n"
            "  P(c_k) = P(c_k) + dprob\n"
            "Then all probabilities are re-normalized to sum to 1.0.\n"
            "This creates a 'recency bias': active clusters rise to the top of the\n"
            "search list, minimizing the number of distance calculations needed to find\n"
            "a match.");
        print_help_section(
            "USE",
            "-dprob 0.05 (Stronger bias, faster adaptation to changing scenes)");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        print_see_also_option(
            "-maxcl",
            "Max number of clusters");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "maxcl") == 0)
    {
        print_help_section("ROLE", "Resource Limiting");
        print_help_section(
            "FUNCTION",
            "Sets the maximum number of clusters allowed (Default: 1000).");
        print_help_section(
            "IMPLEMENTATION",
            "Defines the size of static arrays (clusters, visitors) and the N*N distance\n"
            "cache (dccarray). Affects memory usage (O(N^2) for dccarray).\n"
            "When this limit is reached, the behavior is controlled by -maxcl_strategy.");
        print_help_section("USE", "-maxcl 5000");
        print_help_section(
            "INTERACTS WITH",
            "-maxcl_strategy  What happens at the limit\n"
            "-sparse_dcc      Avoids O(maxcl^2) DCC");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-maxcl_strategy",
            "Strategy when maxcl reached");
        print_see_also_option(
            "-discard_frac",
            "Fraction of clusters to discard");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "ncpu") == 0)
    {
        print_help_section("ROLE", "Parallel Processing");
        print_help_section(
            "FUNCTION",
            "Sets the number of OpenMP threads (Default: 1).");
        print_help_section(
            "IMPLEMENTATION",
            "Used to parallelize the 'pruning' loops. When checking if a candidate cluster\n"
            "is valid, the algorithm checks triangle inequalities against all other clusters.\n"
            "This loop is split across 'ncpu' threads. Also used in batch distance\n"
            "calculations.");
        print_help_section("USE", "-ncpu 4");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-te4",
            "Use 4-point triangle inequality pruning");
        print_see_also_option(
            "-te5",
            "Use 5-point triangle inequality pruning");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "maxcl_strategy") == 0)
    {
        print_help_section("ROLE", "Memory Management Strategy");
        print_help_section(
            "FUNCTION",
            "Determines behavior when the 'maxcl' limit is reached.");
        print_help_section(
            "OPTIONS",
            "stop    : (Default) Exit program. Ensures dataset integrity.\n"
            "discard : 'Cache Eviction'. Scans the oldest 'discard_frac' clusters and removes\n"
            "          the one with the fewest visits. Useful for continuous monitoring.\n"
            "merge   : Merges the two geometrically closest clusters (min d(c_i, c_j)).\n"
            "          Computationally expensive (O(N^2) scan) but preserves information.");
        print_help_section("USE", "-maxcl 100 -maxcl_strategy discard");
        print_help_section(
            "ACTIVE WHEN",
            "Cluster count reaches -maxcl.\n"
            "Has no effect before that.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-maxcl",
            "Max number of clusters");
        print_see_also_option(
            "-discard_frac",
            "Fraction of clusters to discard");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "discard_frac") == 0)
    {
        print_help_section("ROLE", "Discard Strategy Parameter");
        print_help_section(
            "FUNCTION",
            "Fraction of oldest clusters to consider for discarding (Default: 0.5).");
        print_help_section(
            "IMPLEMENTATION",
            "When discarding, we don't want to kill a brand new cluster that hasn't had time\n"
            "to accumulate visitors. This options limits the search to the first\n"
            "N * discard_frac clusters (the 'oldest' ones by index).");
        print_help_section("USE", "-discard_frac 0.2 (Only consider oldest 20%)");
        print_help_section(
            "REQUIRES",
            "-maxcl_strategy discard\n"
            "  Has no effect with other strategies.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-maxcl",
            "Max number of clusters");
        print_see_also_option(
            "-maxcl_strategy",
            "Strategy when maxcl reached");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "maxim") == 0)
    {
        print_help_section("ROLE", "Execution Limit");
        print_help_section(
            "FUNCTION",
            "Process only the first N frames (Default: 100000).\n"
            "Useful for testing on large datasets.");
        return 1;
    }
    else if (strcmp(key, "gprob") == 0)
    {
        print_help_section("ROLE", "Geometric Probability (Trajectory Learning)");
        print_help_section(
            "FUNCTION",
            "Uses historical distance patterns to predict cluster membership.");
        print_help_section(
            "ALGORITHM",
            "For a new frame 'm', the algorithm looks at recent frames 'k' that share distance\n"
            "measurements to common clusters. It computes a 'Geometrical Match Coefficient'\n"
            "based on how similar the distance vector of 'm' is to 'k'.\n"
            "If 'm' looks like 'k' geometrically, the probability of 'm' belonging to the same\n"
            "cluster as 'k' is boosted.");
        print_help_section(
            "USE",
            "-gprob (Highly recommended for continuous drift/trajectory data)");
        print_help_section(
            "TUNED BY",
            "-fmatcha <val>   Match reward (default: 2.0)\n"
            "-fmatchb <val>   Pruning factor (default: 0.5)\n"
            "-maxvis <val>    History depth (default: 1000)");
        print_help_section(
            "ENHANCED BY",
            "-entropy         Optimal measurement scheduling\n"
            "-soft_bayesian   Smoother probability updates");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-pred",
            "Prediction with pattern detection");
        print_see_also_option(
            "-tm",
            "Transition matrix mixing");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "fmatcha") == 0)
    {
        print_help_section("ROLE", "Geometric Matching Parameter A");
        print_help_section(
            "FUNCTION",
            "Reward factor for exact geometric matches in gprob (Default: 2.0).");
        print_help_section(
            "EQUATION",
            "factor = a - (a - b) * (delta_dist / rlim) / 2\n"
            "If delta_dist is 0 (perfect match), factor = a.");
        print_help_section(
            "REQUIRES",
            "-gprob (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        print_see_also_option(
            "-fmatchb",
            "Set fmatch parameter b");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "fmatchb") == 0)
    {
        print_help_section("ROLE", "Geometric Matching Parameter B");
        print_help_section(
            "FUNCTION",
            "Factor at the pruning limit for gprob (Default: 0.5).");
        print_help_section(
            "EQUATION",
            "If delta_dist is 2*rlim (limit of triangle inequality), factor = b.");
        print_help_section(
            "REQUIRES",
            "-gprob (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        print_see_also_option(
            "-fmatcha",
            "Set fmatch parameter a");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "maxvis") == 0)
    {
        print_help_section("ROLE", "gprob History Limit");
        print_help_section(
            "FUNCTION",
            "Max number of recent visitors to track per cluster (Default: 1000).");
        print_help_section(
            "DETAILS",
            "To compute gprob, we scan past frames ('visitors') of candidate clusters.\n"
            "This limits how many past frames are stored/scanned to maintain performance.");
        print_help_section(
            "REQUIRES",
            "-gprob (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "pred") == 0 || strncmp(key, "pred", 4) == 0)
    {
        print_help_section("ROLE", "Time-Series Prediction");
        print_help_section(
            "FUNCTION",
            "Predicts next cluster based on sequence history.");
        print_help_section(
            "FORMAT",
            "-pred[len,h,n]\n"
            "  len: Length of recent sequence to match (Default: 10).\n"
            "  h  : History size to search (Default: 1000).\n"
            "  n  : Number of predicted candidates to test first (Default: 2).");
        print_help_section(
            "ALGORITHM",
            "Matches the last 'len' cluster assignments against the last 'h' frames.\n"
            "If the sequence [A, B, C] is found in history followed by D, then D is\n"
            "predicted as a candidate. Predicted candidates are checked *before*\n"
            "standard sorting.");
        print_help_section(
            "USE",
            "-pred[5,500,1] (For repeating patterns/loops)");
        print_help_section(
            "INTERACTS WITH",
            "-gprob   Both contribute to cluster\n"
            "  probability distribution\n"
            "-tm      Transition matrix complements\n"
            "  pattern detection");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        print_see_also_option(
            "-tm",
            "Transition matrix mixing");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "te4") == 0)
    {
        print_help_section("ROLE", "4-Point Pruning");
        print_help_section(
            "FUNCTION",
            "Enables aggressive pruning using 4 points.");
        print_help_section(
            "ALGORITHM",
            "Standard pruning uses 3 points (Triangle Inequality: d(A,C) <= d(A,B) + d(B,C)).\n"
            "TE4 uses 2 reference clusters (A, B) + Current Frame (F) + Candidate (C).\n"
            "It establishes a 2D plane with A, B, F to bound the distance to C more strictly.\n"
            "Reduces expensive distance calls at the cost of slightly more complex logic.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-te5",
            "Use 5-point triangle inequality pruning");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "te5") == 0)
    {
        print_help_section("ROLE", "5-Point Pruning");
        print_help_section(
            "FUNCTION",
            "Enables aggressive pruning using 5 points.");
        print_help_section(
            "ALGORITHM",
            "Uses 3 reference clusters + Current Frame + Candidate.\n"
            "It constructs a local 3D coordinate system to strictly bound the possible\n"
            "distance range. Effective for high-dimensional data where simple triangle\n"
            "inequalities are loose.");
        print_help_section(
            "USE",
            "-te5 (Recommended for high-dimensional vectors)");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-te4",
            "Use 4-point triangle inequality pruning");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "entropy") == 0)
    {
        print_help_section(
            "ROLE",
            "Entropy-Based Target Selection");
        print_help_section(
            "OVERVIEW",
            "In standard (greedy) mode, GRIC"
            " picks the next cluster to"
            " measure by\nchoosing the one"
            " with the highest posterior"
            " probability.  This is fast but\n"
            "ignores the information value of"
            " each measurement: measuring a"
            " cluster\nthat is already very"
            " likely teaches us little,"
            " whereas measuring a cluster\n"
            "that would eliminate many"
            " alternatives can resolve"
            " ambiguity faster.\n\n"
            "With -entropy, GRIC selects the"
            " target that minimizes the"
            " expected\nShannon entropy of"
            " the posterior distribution after"
            " measurement.  This\nmaximizes"
            " expected information gain per"
            " distance evaluation.");
        print_help_section(
            "POSTERIOR DISTRIBUTION",
            "Each frame maintains a"
            " probability vector p(c) over"
            " active clusters.\nThis vector"
            " is initialized from the mixed"
            " priors (gprob + static\npriors)"
            " and updated after each failed"
            " measurement via Bayesian\n"
            "updates (binary pruning or soft"
            " Bayesian likelihood weighting"
            " if\n-soft_bayesian is enabled)."
            "  The entropy H = -sum(p * log2"
            " p) quantifies\nhow spread out"
            " this distribution is: H=0 means"
            " certainty, H=log2(K)\nmeans"
            " uniform over K clusters.");
        print_help_section(
            "MULTI-STAGE PIPELINE",
            "The entropy evaluation runs a"
            " 4-stage pipeline to balance\n"
            "quality against cost:\n\n"
            "1. GATING\n"
            "   Shannon entropy H of the"
            " current posterior is computed."
            "  If H is\n   below the gate"
            " threshold (-entropy_gate at"
            " depth >= 1, or\n   "
            "-entropy_first_gate at depth 0),"
            " the distribution is already\n"
            "   concentrated and greedy argmax"
            " is used.  This skips all\n"
            "   downstream stages.\n\n"
            "2. POPCOUNT SCORING\n"
            "   For each candidate target, a"
            " fast heuristic score is"
            " computed\n   using bitwise AND"
            " + popcount on the consistency"
            " mask bitfield.\n   This score"
            " approximates the expected"
            " support size reduction:\n"
            "   low score = measuring this"
            " target eliminates many"
            " hypotheses.\n   Candidates are"
            " ranked by this score and the"
            " top ones proceed.\n\n"
            "3. CANDIDATE FILTERING\n"
            "   The top candidates are"
            " selected by interleaving two"
            " lists:\n   probability leaders"
            " (high p) and popcount leaders"
            " (low score).\n   The number of"
            " candidates is capped at"
            " -entropy_max_targets,\n"
            "   dynamically reduced based on"
            " the entropy level.\n\n"
            "4. SHANNON EVALUATION\n"
            "   For each candidate target c_i,"
            " compute the expected posterior\n"
            "   entropy if we were to measure"
            " c_i.  Each hypothesis c_j\n"
            "   (with p(c_j) >"
            " -entropy_min_prob) contributes\n"
            "   p(c_j) * H(posterior | measure"
            " c_i, true cluster = c_j).\n"
            "   The candidate with the lowest"
            " expected entropy wins.\n"
            "   Early exit: if the running sum"
            " exceeds the current best,\n"
            "   remaining hypotheses are"
            " skipped.  Hypotheses are"
            " evaluated in\n   descending"
            " probability order to maximize"
            " early exit.");
        print_help_section(
            "FAST SURROGATE MODE",
            "With -entropy_fast, stage 4"
            " (Shannon evaluation) is skipped"
            "\nentirely and the candidate with"
            " the lowest popcount score from"
            "\nstage 2 is returned directly."
            "  This is a first-order"
            " approximation\nof entropy"
            " minimization that uses only"
            " bitwise operations.\nBenchmarks"
            " show near-identical clustering"
            " quality at a fraction\nof the"
            " CPU cost.");
        print_help_section(
            "DIAGNOSTICS",
            "When -entropy is active, the"
            " final summary includes an"
            "\n\"Entropy Diagnostics\" block"
            " reporting:\n"
            "  - Avg/max initial entropy"
            " (uncertainty at frame start)\n"
            "  - Effective candidate count"
            " (2^H)\n"
            "  - Gate ratio (fraction of"
            " frames where gating returned"
            " greedy)\n"
            "  - Contextual guidance (warns if"
            " rlim may need adjustment)\n\n"
            "These metrics are also exported"
            " to the SHM status struct for\n"
            "real-time monitoring via"
            " gric-status.");
        print_help_section(
            "WHEN TO USE",
            "Entropy mode is most valuable"
            " when:\n"
            "  - Clusters overlap"
            " geometrically (high rlim"
            " relative to spacing)\n"
            "  - Frames are randomly ordered"
            " (no temporal coherence)\n"
            "  - The distance function is"
            " expensive (reducing"
            " measurements\n    matters more"
            " than the entropy computation"
            " overhead)\n\n"
            "Entropy mode adds little benefit"
            " when:\n"
            "  - Clusters are well-separated"
            " (gate catches most frames)\n"
            "  - Frames follow a smooth"
            " trajectory (prediction +\n"
            "    gprob already narrow the"
            " candidates)\n"
            "  - The distance function is"
            " trivially cheap");
        print_help_section(
            "WORKS BEST WITH",
            "-gprob          Provides the"
            " probability distribution\n"
            "-soft_bayesian   Smoother Bayesian"
            " updates between measurements\n"
            "-te4 / -te5      Tighter triangle"
            " inequality bounds");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy_fast",
            "Popcount-only surrogate mode");
        print_see_also_option(
            "-entropy_gate",
            "Gating threshold (depth >= 1)");
        print_see_also_option(
            "-entropy_first_gate",
            "Gating threshold (depth 0)");
        print_see_also_option(
            "-entropy_max_targets",
            "Max targets for evaluation");
        print_see_also_option(
            "-entropy_min_prob",
            "Min hypothesis probability");
        print_see_also_option(
            "-soft_bayesian",
            "Soft Bayesian update");
        print_see_also_option(
            "-gprob",
            "Geometrical probability");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "entropy_gate") == 0)
    {
        print_help_section("ROLE", "Entropy Adaptive Gating");
        print_help_section(
            "FUNCTION",
            "Shannon entropy threshold (in bits) below which entropy mode falls back to\n"
            "greedy selection (default: 2.0).");
        print_help_section(
            "RATIONALE",
            "Full entropy evaluation is O(T*H*W) per target selection. When gprob has\n"
            "already concentrated the probability on a few clusters, greedy selection is\n"
            "near-optimal and the expensive evaluation can be skipped. A threshold of\n"
            "2.0 means roughly <=4 effective candidates.");
        print_help_section(
            "USE",
            "-entropy_gate 1.5 (more aggressive gating)");
        print_help_section(
            "REQUIRES",
            "-entropy (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy",
            "Entropy-based target selection");
        print_see_also_option(
            "-soft_bayesian",
            "Soft Bayesian update");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "entropy_fast") == 0)
    {
        print_help_section(
            "ROLE",
            "Popcount-Only Surrogate Mode");
        print_help_section(
            "WHAT IS POPCOUNT",
            "GRIC maintains a consistency"
            " mask: a bitfield where bit k is"
            " set if\ncluster k is"
            " geometrically consistent with"
            " the current frame\n(i.e., not"
            " yet ruled out by triangle"
            " inequality bounds).\n\n"
            "For a candidate target c_i, the"
            " popcount score estimates how"
            " many\nclusters would survive if"
            " we measured c_i.  For each"
            " hypothesis c_j\n(\"what if c_j"
            " is the true cluster?\"), a"
            " bitwise AND of\n"
            "consistency_mask[c_i][c_j] with"
            " the active cluster mask gives\n"
            "the set of clusters still"
            " consistent under that scenario."
            "  The CPU\ninstruction popcount"
            " counts those set bits in a"
            " single cycle.\n\n"
            "Summing over a sample of"
            " hypotheses yields the popcount"
            " score:\n"
            "  Score(c_i) = sum_j popcount("
            "mask[c_i][c_j] & active_mask)\n\n"
            "Low score = measuring c_i leaves"
            " few survivors = high"
            " discriminative\npower."
            "  Since Shannon entropy is"
            " roughly log2(support size),\n"
            "minimizing support size is a"
            " first-order approximation of\n"
            "minimizing entropy, but computed"
            " entirely with fast bitwise\n"
            "operations instead of"
            " floating-point logarithms.");
        print_help_section(
            "FUNCTION",
            "Skips Shannon entropy evaluation"
            " (stage 4 of the entropy"
            " pipeline)\nand returns the"
            " candidate with the lowest"
            " popcount score from\nstage 2"
            " directly.  See -h entropy for"
            " the full pipeline description.");
        print_help_section(
            "RATIONALE",
            "Shannon eval is O(T*H*W) and"
            " dominates Step 3b cost.  The"
            " popcount\nheuristic provides"
            " near-identical target selection"
            " quality at a\nfraction of the"
            " CPU cost.");
        print_help_section(
            "USE",
            "-entropy -entropy_fast");
        print_help_section(
            "REQUIRES",
            "-entropy");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy",
            "Full pipeline description");
        print_see_also_option(
            "-entropy_gate",
            "Gating threshold in bits");
        printf("\n");
        return 1;
    }
    else if (strcmp(key,
                    "entropy_max_targets") == 0)
    {
        print_help_section(
            "ROLE",
            "Entropy Evaluation Budget");
        print_help_section(
            "FUNCTION",
            "Maximum number of candidate"
            " targets to evaluate for"
            " expected Shannon\nentropy"
            " (default: 15).  The actual"
            " count may be reduced"
            " dynamically\nbased on the"
            " current entropy level.");
        print_help_section(
            "USE",
            "-entropy_max_targets 30");
        print_help_section(
            "REQUIRES",
            "-entropy");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy",
            "Entropy-based target selection");
        printf("\n");
        return 1;
    }
    else if (strcmp(key,
                    "entropy_min_prob") == 0)
    {
        print_help_section(
            "ROLE",
            "Entropy Hypothesis Filter");
        print_help_section(
            "FUNCTION",
            "Minimum probability for a"
            " cluster to be considered as"
            " a hypothesis\nin the entropy"
            " evaluation loop (default:"
            " 0.001).  Clusters below\n"
            "this threshold are skipped."
            "  A dynamic floor of 1% of"
            " the leader's\nprobability is"
            " also applied.");
        print_help_section(
            "USE",
            "-entropy_min_prob 0.01");
        print_help_section(
            "REQUIRES",
            "-entropy");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy",
            "Entropy-based target selection");
        printf("\n");
        return 1;
    }
    else if (strcmp(key,
                    "entropy_first_gate") == 0)
    {
        print_help_section(
            "ROLE",
            "Entropy Gate at Depth 0");
        print_help_section(
            "FUNCTION",
            "Entropy gating threshold in bits"
            " for the first measurement\n"
            "attempt (depth 0) of each frame"
            " (default: 4.0).  At depth 0,"
            " the\nposterior is dominated"
            " by the static prior and greedy"
            " argmax is\nnear-optimal."
            "  After at least one failed"
            " measurement, -entropy_gate\n"
            "is used instead.");
        print_help_section(
            "USE",
            "-entropy_first_gate 6.0");
        print_help_section(
            "REQUIRES",
            "-entropy");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy_gate",
            "Gate threshold after depth 0");
        print_see_also_option(
            "-entropy",
            "Entropy-based target selection");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "soft_bayesian") == 0)
    {
        print_help_section("ROLE", "Target Selection Option");
        print_help_section(
            "FUNCTION",
            "Performs smooth Bayesian updates on candidates rather than hard pruning on\n"
            "distance threshold failure.");
        print_help_section(
            "ALGORITHM",
            "When a distance evaluation fails (dist > rlim), we update the target's\n"
            "probability by multiplying it with a sigmoid-like likelihood function\n"
            "approximated using a minimax polynomial. This retains potential matching\n"
            "candidates under noisy distances and leads to faster information gain\n"
            "convergence.");
        print_help_section(
            "REQUIRES",
            "-gprob (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-entropy",
            "Entropy-based target selection");
        print_see_also_option(
            "-entropy_gate",
            "Entropy gating threshold");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "sparse_dcc") == 0)
    {
        print_help_section("ROLE", "Cluster-to-Cluster Distance Optimization");
        print_help_section(
            "FUNCTION",
            "Enables bounded sparse inter-cluster distance matrix tracking to avoid dense\n"
            "O(K^2) anchor distance calls.");
        print_help_section(
            "ALGORITHM",
            "Reuses ongoing sample-to-anchor measurements to calculate triangle inequality\n"
            "bounds, maintaining lower/upper bounds for unmeasured anchor distances.\n"
            "Highly recommended for video input.");
        print_help_section(
            "TUNED BY",
            "-sparse_dcc_extra_evals <val>\n"
            "  Extra DCC entries computed per step.\n"
            "  Higher = tighter bounds, more CPU.");
        print_help_section(
            "INTERACTS WITH",
            "-maxcl   DCC matrix is O(maxcl^2);\n"
            "  sparse DCC most beneficial when\n"
            "  maxcl is large.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-dcc",
            "Enable dcc.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "sparse_dcc_extra_evals") == 0)
    {
        print_help_section("ROLE", "Sparse DCC Parameter");
        print_help_section(
            "FUNCTION",
            "Sets the number of extra inter-cluster distance evaluations to perform when\n"
            "creating a new cluster (default: 0).");
        print_help_section(
            "RATIONALE",
            "Evaluating a small number of extra distances (e.g. 2 or 5) helps tighten the\n"
            "lower/upper bounds of the sparse DCC matrix, improving subsequent triangle\n"
            "inequality pruning efficiency.");
        print_help_section(
            "REQUIRES",
            "-sparse_dcc (has no effect without it)");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN,
               ANSI_COLOR_RESET);
        print_see_also_option(
            "-sparse_dcc",
            "Sparse DCC matrix");
        print_see_also_option(
            "-dcc",
            "Enable dcc.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "scandist") == 0)
    {
        print_help_section("ROLE", "Data Analysis (Pre-run)");
        print_help_section(
            "FUNCTION",
            "Measures distance statistics without clustering.");
        print_help_section(
            "IMPLEMENTATION",
            "Computes distances between sequential frames (or random pairs) to build\n"
            "a histogram. It reports Min, Max, Median, 20%, 80% percentiles.\n"
            "Use the Median or 20% value to choose a good 'rlim'.");
        print_help_section("USE", "gric-cluster -scandist input.txt");
        return 1;
    }
    else if (strcmp(key, "outdir") == 0)
    {
        print_help_section("ROLE", "Output Management");
        print_help_section(
            "FUNCTION",
            "Specifies the directory for all output files.");
        print_help_section(
            "DEFAULT",
            "If not specified, a directory named '<input>.clusterdat' is created.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-avg",
            "Compute average frame per cluster");
        print_see_also_option(
            "-pngout",
            "Write output as PNG images");
        print_see_also_option(
            "-fitsout",
            "Force FITS output format");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "avg") == 0)
    {
        print_help_section("ROLE", "Output Generation");
        print_help_section(
            "FUNCTION",
            "Computes the average frame for each cluster.");
        print_help_section(
            "IMPLEMENTATION",
            "Accumulates pixel data for every frame assigned to a cluster. At the end,\n"
            "divides by the count. Useful for 'Lucky Imaging' or noise reduction.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        print_see_also_option(
            "-pngout",
            "Write output as PNG images");
        print_see_also_option(
            "-fitsout",
            "Force FITS output format");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "distall") == 0)
    {
        print_help_section("ROLE", "Debugging");
        print_help_section(
            "FUNCTION",
            "Saves every computed distance to 'distall.txt'.");
        print_help_section(
            "FORMAT",
            "ID1 ID2 Dist Ratio ClusterIdx Prob GProb");
        print_help_section(
            "WARNING",
            "Produces massive files for long runs.");
        return 1;
    }
    else if (strcmp(key, "pngout") == 0)
    {
        print_help_section("ROLE", "Output Format");
        print_help_section(
            "FUNCTION",
            "Forces output (anchors, averages, frames) to be written as PNG images.\n"
            "Requires libpng support.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-fitsout",
            "Force FITS output format");
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        print_see_also_option(
            "-avg",
            "Compute average frame per cluster");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "fitsout") == 0)
    {
        print_help_section("ROLE", "Output Format");
        print_help_section(
            "FUNCTION",
            "Forces output to be written as FITS (Flexible Image Transport System) files.\n"
            "Standard in astronomy.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-pngout",
            "Write output as PNG images");
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        print_see_also_option(
            "-avg",
            "Compute average frame per cluster");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "dcc") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes the Distance Between Cluster Centers (DCC) matrix to 'dcc.txt'.");
        print_help_section(
            "FORMAT",
            "Cluster_i Cluster_j Distance");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-sparse_dcc",
            "Enable sparse cluster-to-cluster distance matrix");
        print_see_also_option(
            "-tm_out",
            "Enable transition_matrix.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "tm_out") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes the Transition Matrix to 'transition_matrix.txt'.");
        print_help_section(
            "FORMAT",
            "From_Cluster To_Cluster Count");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-tm",
            "Transition matrix mixing coefficient");
        print_see_also_option(
            "-dcc",
            "Enable dcc.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "anchors") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes the 'anchor' frame (the first frame) of each cluster to disk.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "counts") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes 'cluster_counts.txt' listing how many frames are in each cluster.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "membership") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes 'frame_membership.txt' (Default: Enabled).");
        print_help_section(
            "FORMAT",
            "Contains a line for every frame: FrameIndex AssignedClusterIndex");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-no_membership",
            "Disable frame_membership.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "no_membership") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Disables writing 'frame_membership.txt'. Useful to save disk I/O.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-membership",
            "Enable frame_membership.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "discarded") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes list of discarded frames/clusters to 'discarded_frames.txt'.");
        print_help_section(
            "FORMAT",
            "Lists the frame indices that belonged to deleted clusters.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-maxcl_strategy",
            "Strategy when maxcl reached (stop|discard|merge)");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "clustered") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes 'filename.clustered.txt' containing ALL data grouped by cluster.");
        print_help_section(
            "FORMAT",
            "Format includes comments separating clusters. Good for plotting scripts.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "clusters") == 0)
    {
        print_help_section("ROLE", "Output Control");
        print_help_section(
            "FUNCTION",
            "Writes individual files (or directories) for each cluster containing its\n"
            "member frames.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "tm") == 0)
    {
        print_help_section("ROLE", "Transition Matrix Mixing");
        print_help_section(
            "FUNCTION",
            "Uses transition history to predict next cluster.");
        print_help_section(
            "USE",
            "-tm <coeff> (0.0 to 1.0)");
        print_help_section(
            "ALGORITHM",
            "Mixes the standard probability with the transition probability:\n"
            "  P_final = (1-coeff)*P_standard + coeff * P(next|prev)\n"
            "where P(next|prev) is derived from the count of transitions prev->next.");
        printf("%sSEE ALSO%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-gprob",
            "Use geometrical probability");
        print_see_also_option(
            "-pred",
            "Prediction with pattern detection");
        print_see_also_option(
            "-tm_out",
            "Enable transition_matrix.txt output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "input") == 0)
    {
        print_help_section(
            "OVERVIEW",
            "GRIC accepts frames from several input sources.\n"
            "The format is auto-detected from the file extension\n"
            "unless overridden by a flag.");
        print_help_section(
            "SUPPORTED FORMATS",
            "Text (.txt)\n"
            "  One frame per line, space-separated coordinates.\n"
            "  Simplest format; useful for low-dimensional data.\n"
            "\n"
            "FITS cube (.fits, .fits.fz)\n"
            "  3D data cube (width x height x N_frames).\n"
            "  Standard in astronomy. Requires CFITSIO.\n"
            "\n"
            "MP4 video (.mp4)\n"
            "  Video frames extracted as pixel arrays.\n"
            "  Requires FFmpeg (libav*).\n"
            "\n"
            "ImageStreamIO (-stream)\n"
            "  Shared memory circular buffer for real-time\n"
            "  streaming. Use -cnt2sync for handshake mode.");
        print_help_section(
            "THE RLIM PARAMETER",
            "The first positional argument sets the distance\n"
            "threshold for cluster membership:\n"
            "  0.5        Literal value\n"
            "  a1.5       Auto-mode: 1.5 x median distance\n"
            "             (run -scandist first to calibrate)\n"
            "\n"
            "Use -scandist to measure distance statistics and\n"
            "pick a good rlim before a full clustering run.");
        printf("%sOPTIONS%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-stream",
            "Input is an ImageStreamIO stream");
        print_see_also_option(
            "-cnt2sync",
            "Enable cnt2 synchronization");
        print_see_also_option(
            "-scandist",
            "Measure distance stats (pick rlim)");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "output") == 0)
    {
        print_help_section(
            "OVERVIEW",
            "All output files are written to a directory named\n"
            "<input>.clusterdat/ by default (override with\n"
            "-outdir). Most outputs are disabled by default;"
            "\nenable them with the flags below.");
        print_help_section(
            "CORE OUTPUTS (enabled by default)",
            "frame_membership.txt\n"
            "  FrameIndex ClusterIndex per line.\n"
            "  For streams: also includes cnt0 and timestamp.\n"
            "\n"
            "dcc.txt\n"
            "  Inter-cluster distance matrix:\n"
            "  Cluster_i  Cluster_j  Distance");
        print_help_section(
            "OPTIONAL OUTPUTS",
            "transition_matrix.txt  (-tm_out)\n"
            "  From_Cluster  To_Cluster  Count\n"
            "\n"
            "cluster_counts.txt     (-counts)\n"
            "  Number of frames assigned to each cluster\n"
            "\n"
            "anchors.fits/txt/png   (-anchors)\n"
            "  Anchor frame of each cluster\n"
            "\n"
            "average.fits/txt/png   (-avg)\n"
            "  Mean frame per cluster (lucky imaging)\n"
            "\n"
            "*.clustered.txt        (-clustered)\n"
            "  All input data grouped by cluster\n"
            "\n"
            "cluster_X/             (-clusters)\n"
            "  Individual directories with member frames\n"
            "\n"
            "discarded_frames.txt   (-discarded)\n"
            "  Frame indices from evicted clusters\n"
            "\n"
            "distall.txt            (-distall)\n"
            "  Every computed distance with metadata");
        printf("%sOPTIONS%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-outdir",
            "Specify output directory");
        print_see_also_option(
            "-avg",
            "Compute average frame per cluster");
        print_see_also_option(
            "-pngout",
            "Write output as PNG images");
        print_see_also_option(
            "-fitsout",
            "Force FITS output format");
        print_see_also_option(
            "-dcc",
            "Enable dcc.txt output");
        print_see_also_option(
            "-no_dcc",
            "Disable dcc.txt output");
        print_see_also_option(
            "-tm_out",
            "Enable transition_matrix.txt output");
        print_see_also_option(
            "-anchors",
            "Enable anchors output");
        print_see_also_option(
            "-counts",
            "Enable cluster_counts.txt output");
        print_see_also_option(
            "-membership",
            "Enable frame_membership.txt output");
        print_see_also_option(
            "-no_membership",
            "Disable frame_membership.txt output");
        print_see_also_option(
            "-discarded",
            "Enable discarded_frames.txt output");
        print_see_also_option(
            "-clustered",
            "Enable *.clustered.txt output");
        print_see_also_option(
            "-clusters",
            "Enable individual cluster files");
        print_see_also_option(
            "-shm",
            "Enable shared-memory status output");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "clustering") == 0)
    {
        print_help_section(
            "OVERVIEW",
            "These options control how frames are assigned\n"
            "to clusters and how the search is accelerated.");
        print_help_section(
            "CORE PARAMETERS",
            "-dprob <val>   (default: 0.01)\n"
            "  Recency bias: amount added to a cluster's\n"
            "  probability when a frame is assigned to it.\n"
            "  Higher = faster adaptation, more volatile.\n"
            "\n"
            "-maxcl <val>   (default: 1000)\n"
            "  Maximum number of clusters. Controls memory\n"
            "  usage (DCC matrix is O(maxcl^2)).\n"
            "\n"
            "-maxim <val>   (default: 100000)\n"
            "  Maximum number of frames to process.\n"
            "\n"
            "-ncpu <val>    (default: 1)\n"
            "  Number of CPUs for parallel pruning.\n"
            "  Effective when K >= 256.");
        print_help_section(
            "PROBABILITY AND PRUNING",
            "-gprob\n"
            "  Use distance history to estimate match\n"
            "  probability (geometric probability).\n"
            "\n"
            "-soft_bayesian\n"
            "  Gaussian-like soft pruning instead of hard\n"
            "  binary elimination.\n"
            "\n"
            "-te4 / -te5\n"
            "  4-point or 5-point triangle inequality\n"
            "  pruning. Tighter bounds, higher per-step\n"
            "  cost. Best for high-dimensional data.\n"
            "\n"
            "-entropy\n"
            "  Information-theoretic target selection.\n"
            "  Picks the measurement that maximizes\n"
            "  expected entropy reduction.\n"
            "\n"
            "-sparse_dcc\n"
            "  Sparse cluster-to-cluster distance matrix.\n"
            "  Avoids O(K^2) cost at cluster creation.");
        print_help_section(
            "PREDICTION",
            "-pred[l,h,n]   (default: 10,1000,2)\n"
            "  Pattern matching in assignment history.\n"
            "  Tests predicted clusters before standard\n"
            "  ranking. Good for periodic/cyclic data.\n"
            "\n"
            "-tm <coeff>    (0.0 to 1.0)\n"
            "  Transition matrix mixing. Blends prior\n"
            "  probability with transition history.");
        print_help_section(
            "RESOURCE LIMITS",
            "-maxcl_strategy <stop|discard|merge>\n"
            "  What to do when maxcl is reached:\n"
            "  stop:    exit (default, batch mode)\n"
            "  discard: evict least-visited cluster\n"
            "  merge:   merge two closest clusters\n"
            "\n"
            "-discard_frac <val>  (default: 0.5)\n"
            "  Fraction of oldest clusters to consider\n"
            "  for discard eviction.");
        print_help_section(
            "CONFIGURATION FILES",
            "-conf <file>\n"
            "  Read options from a configuration file.\n"
            "\n"
            "-confw <file>\n"
            "  Write current options to a file.");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "analysis") == 0)
    {
        print_help_section(
            "OVERVIEW",
            "Analysis and debugging options help calibrate\n"
            "parameters and monitor the clustering run.");
        print_help_section(
            "PRE-RUN CALIBRATION",
            "-scandist\n"
            "  Reads frames and computes distance statistics\n"
            "  (min, max, median, 20%, 80% percentiles)\n"
            "  without clustering. Use the median or 20%\n"
            "  value to pick a good rlim.\n"
            "\n"
            "  Example:\n"
            "    gric-cluster -scandist input.txt");
        print_help_section(
            "RUNTIME MONITORING",
            "-progress\n"
            "  Print periodic progress information.\n"
            "  Enabled by default.");
        printf("%sOPTIONS%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "-scandist",
            "Measure distance stats");
        print_see_also_option(
            "-progress",
            "Print progress (default: enabled)");
        print_see_also_option(
            "-conf",
            "Read options from configuration file");
        print_see_also_option(
            "-confw",
            "Write current options to file");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "intro") == 0)
    {
        print_help_section(
            "WHAT IS GRIC?",
            "GRIC (Geometric Real-time Image Clustering)\n"
            "groups frames by visual similarity.\n"
            "\n"
            "Each frame is compared against existing\n"
            "cluster anchors. If a close enough match is\n"
            "found (within rlim), the frame joins that\n"
            "cluster. Otherwise a new cluster is created.\n"
            "\n"
            "GRIC is single-pass and sequential: each\n"
            "frame is processed exactly once, in order,\n"
            "making it suitable for real-time streams.");
        print_help_section(
            "TYPICAL WORKFLOW",
            "1. Prepare input\n"
            "   Text file, FITS cube, MP4 video,\n"
            "   or live ImageStreamIO stream.\n"
            "\n"
            "2. Calibrate the distance threshold\n"
            "     gric-cluster -scandist input.txt\n"
            "   Use the reported median distance as\n"
            "   your rlim value.\n"
            "\n"
            "3. Run clustering\n"
            "     gric-cluster a1.5 input.txt\n"
            "   The 'a' prefix means auto-scale:\n"
            "   rlim = 1.5 x median distance.\n"
            "\n"
            "4. Inspect outputs\n"
            "   Results are written to\n"
            "   input.txt.clusterdat/");
        print_help_section(
            "KEY CONCEPTS",
            "rlim\n"
            "  Distance threshold. Frames within this\n"
            "  distance of a cluster anchor are assigned\n"
            "  to that cluster. Smaller rlim = more\n"
            "  clusters; larger rlim = fewer.\n"
            "\n"
            "Anchor\n"
            "  The representative frame that defines a\n"
            "  cluster. All distances are measured to\n"
            "  anchors, not between all frame pairs.\n"
            "\n"
            "DCC (Distance between Cluster Centers)\n"
            "  Matrix of inter-anchor distances. Used\n"
            "  by triangle inequality to skip\n"
            "  unnecessary distance computations.\n"
            "\n"
            "Pruning\n"
            "  Eliminating candidate clusters without\n"
            "  measuring them. The main source of\n"
            "  GRIC's speed advantage.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "algorithm",
            "How the per-frame loop works");
        print_see_also_option(
            "performance",
            "Choosing options for best speed");
        print_see_also_option(
            "input",
            "Supported input formats");
        print_see_also_option(
            "output",
            "What output files are produced");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "algorithm") == 0
          || strcmp(key, "algorithms") == 0)
    {
        print_help_section(
            "GRIC: GEOMETRIC REAL-TIME IMAGE CLUSTERING",
            "GRIC is a single-pass, sequential, distance-based\n"
            "clustering algorithm. It processes frames one at a\n"
            "time and assigns each to a cluster whose anchor is\n"
            "within a maximum Euclidean distance rlim. If no\n"
            "cluster matches, a new one is created.");
        print_help_section(
            "DISTANCE METRIC",
            "Euclidean (L2) distance between frame vectors:\n"
            "  d(a, b) = sqrt( sum( (a[i] - b[i])^2 ) )\n"
            "Computed with AVX2/FMA SIMD when available.");
        print_help_section(
            "PER-FRAME ALGORITHM",
            "For each new frame, GRIC executes:\n"
            "\n"
            "1. PREDICT (optional)\n"
            "   If -pred or -tm is active, generate a ranked\n"
            "   list of likely clusters based on assignment\n"
            "   history or transition counts.\n"
            "\n"
            "2. SELECT TARGET\n"
            "   Pick the cluster to measure next, using one of:\n"
            "   - Greedy: highest prior probability\n"
            "   - Geometric: highest gprob score (-gprob)\n"
            "   - Entropy: max information gain (-entropy)\n"
            "\n"
            "3. MEASURE DISTANCE\n"
            "   Compute d(frame, anchor) via framedist().\n"
            "   This is the expensive step to minimize.\n"
            "\n"
            "4. CHECK MATCH\n"
            "   If d < rlim: assign frame to this cluster.\n"
            "   Otherwise: continue to step 5.\n"
            "\n"
            "5. PRUNE CANDIDATES\n"
            "   Use the measured distance to eliminate clusters\n"
            "   that cannot match (triangle inequality, gprob,\n"
            "   soft Bayesian). Go to step 2.\n"
            "\n"
            "6. CREATE NEW CLUSTER\n"
            "   If all candidates are exhausted, the frame\n"
            "   becomes the anchor of a new cluster.");
        print_help_section(
            "KEY OPTIMIZATIONS",
            "Triangle inequality (always active)\n"
            "  After measuring d(frame, A), eliminate cluster B\n"
            "  if |d(frame,A) - d(A,B)| > rlim. Extends to\n"
            "  4-point (-te4) and 5-point (-te5) pruning for\n"
            "  tighter bounds using coordinate reconstruction.\n"
            "\n"
            "Geometric probability (-gprob)\n"
            "  Learns spatial relationships from measurement\n"
            "  history. When cluster A is measured, examine\n"
            "  past visitors of A and boost/penalize their\n"
            "  assigned clusters based on distance similarity.\n"
            "\n"
            "Entropy-based selection (-entropy)\n"
            "  Information-theoretic target selection. Picks\n"
            "  the measurement that maximally reduces Shannon\n"
            "  entropy of the posterior distribution. Uses a\n"
            "  precomputed consistency mask for efficiency.\n"
            "\n"
            "Sparse DCC (-sparse_dcc)\n"
            "  Maintains upper/lower bounds on inter-cluster\n"
            "  distances instead of computing all O(K^2) pairs.\n"
            "  Critical for large numbers of clusters.\n"
            "\n"
            "Soft Bayesian (-soft_bayesian)\n"
            "  Replaces hard binary pruning with smooth\n"
            "  Gaussian-like likelihood updates. Candidates\n"
            "  fade out gradually instead of being eliminated.\n"
            "\n"
            "Recency bias (-dprob)\n"
            "  Recently active clusters rise in the search\n"
            "  priority, reducing average search depth.");
        print_help_section(
            "SYNERGIES",
            "The optimizations are designed to layer together:\n"
            "  -gprob + -entropy: gprob builds the probability\n"
            "    distribution, entropy schedules measurements\n"
            "    to maximally resolve it.\n"
            "  -sparse_dcc + large -maxcl: sparse DCC avoids\n"
            "    the quadratic cost of dense DCC.\n"
            "  -pred + -tm: pattern detection covers multi-step\n"
            "    sequences; transition matrix handles pairwise\n"
            "    transitions.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "performance",
            "How to pick options for best speed");
        print_see_also_option(
            "clustering",
            "All clustering control options");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "performance") == 0
          || strcmp(key, "tuning") == 0)
    {
        print_help_section(
            "QUICK START",
            "1. Calibrate rlim:\n"
            "     gric-cluster -scandist input.txt\n"
            "   Use the reported median or 20% percentile.\n"
            "\n"
            "2. Run with recommended options:\n"
            "     gric-cluster -gprob -entropy \\\n"
            "       -soft_bayesian -sparse_dcc \\\n"
            "       a1.5 input.txt");
        print_help_section(
            "TEMPORAL / STREAMING DATA",
            "For video, time-series, or ImageStreamIO streams\n"
            "where consecutive frames are correlated:\n"
            "\n"
            "-gprob\n"
            "  Learns spatial relationships from measurement\n"
            "  history. Dramatically reduces search steps.\n"
            "\n"
            "-entropy -soft_bayesian\n"
            "  Information-optimal target selection with smooth\n"
            "  Bayesian updates. Maximizes pruning per step.\n"
            "\n"
            "-sparse_dcc\n"
            "  Avoids O(K^2) distance calls at cluster creation.\n"
            "  Essential when the number of clusters is large.\n"
            "\n"
            "-tm <coeff>  (0.5 to 1.0)\n"
            "  If data has repeating or cyclic patterns, mix\n"
            "  transition probability into the prior.\n"
            "\n"
            "-pred[5,500,2]\n"
            "  For strongly periodic data. Bypasses the full\n"
            "  search by testing predicted clusters first.");
        print_help_section(
            "HIGH-DIMENSIONAL / RANDOM DATA",
            "When framedist() is the bottleneck (large frames,\n"
            "high pixel count):\n"
            "\n"
            "-te4 or -te5\n"
            "  Tighter geometric bounds via coordinate\n"
            "  reconstruction. Reduces distance calls by up\n"
            "  to 45%. Higher per-step CPU cost, so only a\n"
            "  net win for expensive framedist().\n"
            "\n"
            "-entropy -gprob\n"
            "  Synergistic: gprob builds a spatial probability\n"
            "  distribution, entropy schedules measurements\n"
            "  to maximally resolve it.\n"
            "\n"
            "-ncpu <N>\n"
            "  Parallelize pruning loops. Effective when the\n"
            "  number of clusters K >= 256.");
        print_help_section(
            "MEMORY-CONSTRAINED ENVIRONMENTS",
            "-maxcl <val>\n"
            "  DCC matrix is O(maxcl^2). Lower maxcl to\n"
            "  reduce memory usage.\n"
            "\n"
            "-maxcl_strategy discard\n"
            "  For infinite streams: evict least-visited\n"
            "  clusters when maxcl is reached, acting\n"
            "  like a cache.\n"
            "\n"
            "-maxcl_strategy merge\n"
            "  Merge two closest clusters. Preserves\n"
            "  information but is more expensive.\n"
            "\n"
            "-sparse_dcc\n"
            "  Reduces peak DCC computation cost.");
        print_help_section(
            "REDUCING I/O OVERHEAD",
            "Disable outputs you don't need:\n"
            "  -no_membership   Skip frame_membership.txt\n"
            "  -no_dcc          Skip dcc.txt\n"
            "\n"
            "For batch processing, redirect stdout:\n"
            "  gric-cluster ... > run.log");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "algorithm",
            "Overview of the GRIC algorithm");
        print_see_also_option(
            "clustering",
            "All clustering control options");
        print_see_also_option(
            "output",
            "All output files and options");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "tiling") == 0)
    {
        print_help_section(
            "WHAT IS TILING?",
            "Tiling partitions large input images into a grid of independent\n"
            "sub-images (tiles). Each tile runs its own independent GRIC\n"
            "clustering instance in parallel.\n"
            "\n"
            "This yields three major benefits:\n"
            "1. Arithmetic Speedup: Distance metrics check smaller sub-frames\n"
            "   (e.g., a 2x2 tile distance call is 4x cheaper to compute).\n"
            "   Search complexity drops from O(K) to O(K_tile).\n"
            "2. Parallelization: Dispatches tile tasks across OpenMP threads.\n"
            "3. Memory Footprint Reduction: Cluster anchors only store the\n"
            "   pixels of their respective tile (e.g. 1/4 the size for 2x2).\n"
            "   Because local tile environments are simpler, the number of\n"
            "   clusters per tile (K_tile) is much smaller than the monolithic\n"
            "   system. This reduces anchor storage by 10-100x and quadratic\n"
            "   DCC distance matrices by 100-1000x.");
        print_help_section(
            "TILING OPTIONS",
            "-tiles <NxM>\n"
            "  Specify the grid size (e.g. -tiles 2x2).\n"
            "\n"
            "-tileconf <file.txt>\n"
            "  Define per-tile configurations (tile_id rlim maxcl).\n"
            "\n"
            "-retrieval_window <N>\n"
            "  Joint Trajectory Fusion lookback horizon (default: 1000).");
        print_help_section(
            "TUNING GUIDELINES",
            "1. Grid Resolution (The Sweet Spot):\n"
            "   Avoid partitioning too finely (e.g. 4x4 on a 32x32 image).\n"
            "   As grid size M increases, the joint state combinations grow\n"
            "   exponentially (k^M), resulting in combinatorial state explosion\n"
            "   and high OpenMP task scheduling overhead.\n"
            "   * Recommend 2x2 tiling for small/medium scientific sensors.\n"
            "\n"
            "2. Calibrating rlim:\n"
            "   As tile size shrinks, the maximum distance between sub-frames\n"
            "   drops. Scaling rlim proportionally to the square root of the\n"
            "   sub-frame pixel count is a good rule of thumb.\n"
            "\n"
            "3. Joint Trajectory Fusion (Cross-Tile Correction):\n"
            "   When tiling is active, trajectory fusion resolves tile boundary noise\n"
            "   by looking back at global transition history. Use -retrieval_window\n"
            "   to tune memory depth (typically 1,000 to 10,000 frames).");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "performance",
            "How to pick options for best performance");
        print_see_also_option(
            "algorithm",
            "Overview of the GRIC algorithm");
        printf("\n");
        return 1;
    }
    else if (strcmp(key, "compression") == 0
          || strcmp(key, "statespace") == 0)
    {
        print_help_section(
            "WHAT IS STATE SPACE COMPRESSION?",
            "State space compression measures how efficiently GRIC represents\n"
            "the joint spatial-temporal states of a multi-tile system.\n"
            "\n"
            "When clustering tiles independently, local noise near boundaries\n"
            "frequently causes tiles to assign frames to mismatched clusters,\n"
            "creating a massive number of spurious joint state combinations\n"
            "(tuples). State space compression resolves these noisy states\n"
            "by fusing boundary assignments into physically consistent paths.");
        print_help_section(
            "HOW TRAJECTORY FUSION WORKS",
            "1. Independent Spatial Clustering (Pass 1): Each tile clusters\n"
            "   its sub-frame, yielding a raw joint state tuple U = (c_0, c_1, ...).\n"
            "2. Joint Trajectory Fusion (Pass 2): Scans the global history of\n"
            "   resolved states within a lookback window (H). It identifies\n"
            "   frames with spatially similar trajectory patterns.\n"
            "3. Bayesian Correction: Computes transition priors by accumulating\n"
            "   historical transition evidence. A high-contrast prior overrides\n"
            "   local boundary noise, collapsing fragmented assignments\n"
            "   into the same clean physical trajectories.");
        print_help_section(
            "TUNING GUIDELINES",
            "1. Tuning lookback horizon (-retrieval_window <H>):\n"
            "   - Small H (e.g. < 200): Small sample size yields weak transition\n"
            "     statistics, leading to poor error correction and low compression.\n"
            "   - Optimal H (typically 1,000 to 10,000): Collects robust joint\n"
            "     evidence, filtering out random boundary fluctuations.\n"
            "   - Excessive H (e.g. > 20,000): Concepts/clusters undergo drift\n"
            "     and recycling over very long horizons. Stale memory acts as\n"
            "     noise, degrading the compression quality.\n"
            "\n"
            "2. Impact of Tiling Grid (-tiles <NxM>):\n"
            "   Keep grid resolution balanced (recommend 2x2). High grid sizes\n"
            "   (e.g., 4x4) treat ball motion as independent variables, causing\n"
            "   a combinatorial state explosion (k^M unique tuples) that completely\n"
            "   destroys spatial correlations and prevents compression.");
        printf("%sSEE ALSO%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
        print_see_also_option(
            "tiling",
            "Image partitioning and multi-tile processing");
        print_see_also_option(
            "performance",
            "How to pick options for best performance");
        printf("\n");
        return 1;
    }
    return 0;
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
            printf("%sGRIC HELP: %s%s\n\n",
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
        printf("%sGRIC HELP: %s%s\n\n",
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

static void print_color_mode(void)
{
    cli_print_color_mode();
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
    printf("%sNAME%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
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
    print_colored_line("    -dprob <val>             Delta probability (default: 0.01)");
    print_colored_line("    -maxcl <val>             Max number of clusters (default: 1000)");
    print_colored_line("    -ncpu <val>              Number of CPUs to use (default: 1)");
    print_colored_line("    -maxcl_strategy <str>    Strategy when maxcl reached "
                       "(stop|discard|merge) (default: stop)");
    print_colored_line("    -discard_frac <val>      Fraction of oldest clusters to candidate "
                       "for discard (default: 0.5)");
    print_colored_line("    -maxim <val>             Max number of frames (default: 100000)");
    print_colored_line("    -gprob                   Use geometrical probability");
    print_colored_line("    -fmatcha <val>           Set fmatch parameter a (default: 2.0)");
    print_colored_line("    -fmatchb <val>           Set fmatch parameter b (default: 0.5)");
    print_colored_line("    -maxvis <val>            Max visitors for gprob history "
                       "(default: 1000)");
    print_colored_line("    -pred[l,h,n]             Prediction with pattern detection "
                       "(default: 10,1000,2)");
    print_colored_line("                            l: length of pattern to match "
                       "(recent cluster history)");
    print_colored_line("                            h: history size (how far back to search "
                       "for pattern)");
    print_colored_line("                            n: number of prediction candidates "
                       "to return");
    print_colored_line("    -te4                     Use 4-point triangle inequality pruning");
    print_colored_line("    -te5                     Use 5-point triangle inequality pruning");
    print_colored_line("    -entropy                 Use entropy-based target cluster selection");
    print_colored_line("    -entropy_fast            Use popcount-only surrogate "
                       "(skip Shannon eval)");
    print_colored_line("    -entropy_gate <val>       Entropy gating threshold in bits "
                       "(default: 2.0)");
    print_colored_line("    -entropy_first_gate <val> Gate threshold at depth 0 "
                       "(default: 4.0)");
    print_colored_line("    -entropy_max_targets <N>  Max targets for entropy eval "
                       "(default: 15)");
    print_colored_line("    -entropy_min_prob <val>   Min hypothesis probability "
                       "(default: 0.001)");
    print_colored_line("    -soft_bayesian           Enable Soft Bayesian update approximation");
    print_colored_line("    -sparse_dcc              Enable sparse cluster-to-cluster "
                       "distance matrix");
    print_colored_line("    -sparse_dcc_extra_evals  Set number of extra DCC evaluations "
                       "(default: 0)");
    print_colored_line("    -conf <file>             Read options from configuration file");
    print_colored_line("    -confw <file>            Write current options to configuration file");

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

    printf("  %sTiling:%s\n",
           ANSI_BOLD, ANSI_COLOR_RESET);
    print_colored_line(
        "    -tiles <NxM>             "
        "Split image into NxM tile grid");
    print_colored_line(
        "    -tilemap <file.fits>     "
        "Load tile map from integer FITS file");
    print_colored_line(
        "    -tileconf <file.txt>     "
        "Per-tile config (tile_id rlim maxcl)");
    print_colored_line(
        "    -retrieval_window <N>    "
        "Tuple lookback horizon (default: 1000)\n");

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

    print_color_mode();
}

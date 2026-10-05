/**
 * @file plot_internal.h
 * @brief Internal structures and function declarations for plotting clusters.
 */

#ifndef PLOT_INTERNAL_H
#define PLOT_INTERNAL_H

typedef struct
{
    int    id;
    double x;
    double y;
} Anchor;

typedef struct
{
    char text[256];
} HeaderLine;

typedef struct
{
    double     rlim;
    double     dprob;
    int        maxcl;
    long       maxim;
    int        gprob;
    int        te4;
    int        te5;
    char       output_dir[4096];
    char       dcc_filename[4096];
    long       total_frames;
    long       total_clusters;
    long       total_dists;
    long       hist_data[10000];
    long       cluster_query_counts[10000];
    long       samples_per_cluster[10000];
    Anchor     anchors[10000];
    int        num_anchors;
    HeaderLine stats[100];
    int        num_stats;
} PlotData;

/**
 * @brief Parse the coordinate clustering log and membership files.
 *
 * This function opens the log file to extract clustering parameters and statistics.
 * It also opens the membership and points files to load anchors and compute the number
 * of samples per cluster.
 *
 * @param points_filename Path to the points coordinate file.
 * @param log_filename    Path to the clustering log file.
 * @param data            Pointer to the PlotData structure to populate.
 *
 * @return 0 on success, 1 on failure.
 */
int parse_input_data(
    const char *points_filename,
    const char *log_filename,
    PlotData   *data);

/**
 * @brief Render SVG or PNG plots from the parsed data.
 *
 * This function handles all coordinate mappings, color palettes, and SVG/PNG rendering.
 *
 * @param points_filename         Path to the points coordinate file.
 * @param output_filename         Path to the output image file.
 * @param queries_output_filename Path to the queries output image file.
 * @param data                    Pointer to the parsed PlotData.
 * @param png_mode                1 for PNG output, 0 for SVG output.
 * @param font_size               Font size for text rendering.
 *
 * @return 0 on success, 1 on failure.
 */
int render_plots(
    const char     *points_filename,
    const char     *output_filename,
    const char     *queries_output_filename,
    const PlotData *data,
    int             png_mode,
    double          font_size);

#endif // PLOT_INTERNAL_H

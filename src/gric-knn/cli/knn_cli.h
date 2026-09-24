/**
 * @file knn_cli.h
 * @brief Command-line interface parser and help renderer for gric-knn.
 *
 * Declares command-line parsing, argument validation, and terminal formatting
 * functions. These routines parse user flags, initialize search configuration
 * (KnnConfig), render usage synopses, and display the startup configuration banner.
 */

#ifndef KNN_CLI_H
#define KNN_CLI_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_cli_print_usage() - Print concise command-line usage
 * @progname: Name of the executable.
 *
 * Prints a one-line synopsis of required positional arguments and common flags.
 */
void knn_cli_print_usage(
    const char *progname);

/**
 * knn_cli_print_help() - Print detailed colored help and options
 * @progname: Name of the executable.
 *
 * Prints comprehensive option descriptions, algorithmic modes, and usage examples.
 */
void knn_cli_print_help(
    const char *progname);

/**
 * knn_cli_parse() - Parse and validate command-line arguments into KnnConfig
 * @argc:   Argument count.
 * @argv:   Argument vector.
 * @config: Pointer to KnnConfig to initialize and populate.
 *
 * Return: 0 on success, 1 on argument error, or 2 if help was displayed.
 */
int knn_cli_parse(
    int        argc,
    char     **argv,
    KnnConfig *config);

/**
 * knn_cli_print_banner() - Print active configuration parameters
 * @config: Pointer to active KnnConfig.
 *
 * Displays formatted table of active algorithm settings, thresholds, and paths.
 */
void knn_cli_print_banner(
    const KnnConfig *config);

#ifdef __cplusplus
}
#endif

#endif // KNN_CLI_H

/**
 * @file knn_cli.h
 * @brief Command-line interface parser and help renderer for gric-knn.
 */

#ifndef KNN_CLI_H
#define KNN_CLI_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_cli_print_usage() - Print concise command-line usage.
 * @progname: Name of the executable.
 */
void knn_cli_print_usage(
    const char *progname);

/**
 * knn_cli_print_help() - Print detailed colored help and options.
 * @progname: Name of the executable.
 */
void knn_cli_print_help(
    const char *progname);

/**
 * knn_cli_parse() - Parse and validate command-line arguments into KnnConfig.
 * @argc:   Argument count.
 * @argv:   Argument vector.
 * @config: Pointer to KnnConfig to initialize and populate.
 *
 * Return: 0 on success, 1 on argument error, 2 if help was displayed.
 */
int knn_cli_parse(
    int        argc,
    char     **argv,
    KnnConfig *config);

/**
 * knn_cli_print_banner() - Print active configuration parameters.
 * @config: Pointer to active KnnConfig.
 */
void knn_cli_print_banner(
    const KnnConfig *config);

#ifdef __cplusplus
}
#endif

#endif // KNN_CLI_H

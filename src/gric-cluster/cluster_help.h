/**
 * @file cluster_help.h
 * @brief Declarations for clustering command help output.
 *
 * Provides prototypes for outputting usage, detailed keyword help,
 * and general command line help screens for the clustering engine.
 */

#ifndef CLUSTER_HELP_H
#define CLUSTER_HELP_H

void init_colors_help(void);

void print_usage(
    char *progname);

void print_help_keyword(
    const char *keyword);

void print_help(
    char *progname);

#endif // CLUSTER_HELP_H

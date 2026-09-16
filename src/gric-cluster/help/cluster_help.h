/**
 * @file cluster_help.h
 * @brief Declarations for clustering command help output.
 *
 * Provides prototypes for outputting usage, detailed keyword help,
 * and general command line help screens for the clustering engine.
 */

#ifndef CLUSTER_HELP_H
#define CLUSTER_HELP_H

/**
 * init_colors_help() - Initialize ANSI terminal color escapes for help screens.
 */
void init_colors_help(void);

/**
 * print_usage() - Print concise command-line syntax and usage message.
 * @progname: Name of executable program.
 */
void print_usage(
    char *progname);

/**
 * print_help_keyword() - Display help on specific command-line keyword or option.
 * @keyword: Name of keyword or flag to look up.
 */
void print_help_keyword(
    const char *keyword);

/**
 * print_help() - Display complete comprehensive command-line help screen.
 * @progname: Name of executable program.
 */
void print_help(
    char *progname);

#endif // CLUSTER_HELP_H

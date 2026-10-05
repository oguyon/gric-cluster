/**
 * @file cluster_help_format.h
 * @brief Declarations for help text wrapping, colorizing, and section formatting.
 */

#ifndef CLUSTER_HELP_FORMAT_H
#define CLUSTER_HELP_FORMAT_H

/**
 * @brief Initialize colors for help terminal output.
 */
void init_colors_help(
    void);

/**
 * @brief Print a labelled help section with dynamic word wrapping.
 */
void print_help_section(
    const char *label,
    const char *value);

/**
 * @brief Print a "See Also" option reference line.
 */
void print_see_also_option(
    const char *option,
    const char *desc);

#endif // CLUSTER_HELP_FORMAT_H

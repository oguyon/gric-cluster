/**
 * @file cluster_help_format.c
 * @brief Bridge implementations of help text wrapping and formatting calling shared helpers.
 */

#include "cluster_help_format.h"
#include "shared/cli_colors.h"

void init_colors_help(
    void)
{
    cli_colors_init();
}

void print_help_section(
    const char *label,
    const char *value)
{
    cli_print_help_section(label, value);
}

void print_see_also_option(
    const char *option,
    const char *desc)
{
    cli_print_see_also_option(option, desc);
}

/**
 * @file cli_help_view.h
 * @brief High-level CLI help formatting, header boxes, and topic suggestion views.
 */

#ifndef CLI_HELP_VIEW_H
#define CLI_HELP_VIEW_H

/**
 * @brief Print a single line, styling option flags and defaults.
 * @param line The string line to style and print.
 */
void cli_print_colored_line(
    const char *line);

/**
 * @brief Print usage information with styled command and placeholders.
 * @param usage The usage string.
 */
void cli_print_colored_usage(
    const char *usage);

/**
 * @brief Parse and print a block of options, formatting each line.
 * @param options The multiline options block.
 */
void cli_print_colored_options(
    const char *options);

/**
 * @brief Parse and print a block of examples, highlighting commands and symbols.
 * @param examples The multiline examples block.
 */
void cli_print_colored_examples(
    const char *examples);

/**
 * @brief Print a "See Also" option reference line.
 * @param option Option or keyword name.
 * @param desc   Description text.
 */
void cli_print_see_also_option(
    const char *option,
    const char *desc);

/**
 * @brief Print a structured help section heading and formatted content block.
 * @param label Section heading title.
 * @param value Multiline section content.
 */
void cli_print_help_section(
    const char *label,
    const char *value);

/**
 * @brief Print an eye-catching decorative box around a title header.
 * @param title Title string.
 */
void cli_print_header_box(
    const char *title);

/**
 * @brief Find and suggest closest topic match in case of typos.
 * @param topic Target topic name.
 * @param topics Array of valid topic names.
 * @param ntopics Number of topics.
 */
void cli_suggest_similar_topic(
    const char         *topic,
    const char *const  *topics,
    int                 ntopics);

#endif // CLI_HELP_VIEW_H

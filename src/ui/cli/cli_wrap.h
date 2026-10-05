/**
 * @file cli_wrap.h
 * @brief Word wrapping and rich inline syntax formatting for CLI terminal output.
 */

#ifndef CLI_WRAP_H
#define CLI_WRAP_H

/**
 * @brief Highlight and print rich inline syntax elements within a text segment.
 * @param text Pointer to start of text segment.
 * @param len Length of segment in bytes.
 * @param is_bold Flag indicating if surrounding text context is bold.
 */
void cli_print_rich_segment(
    const char *text,
    int         len,
    int         is_bold);

/**
 * @brief Print a line of text word-wrapped to terminal column boundaries.
 * @param text Pointer to text.
 * @param len Length of text.
 * @param indent Left indentation spaces.
 * @param width Target wrapping column width.
 */
void cli_print_wrapped_line(
    const char *text,
    int         len,
    int         indent,
    int         width);

#endif // CLI_WRAP_H

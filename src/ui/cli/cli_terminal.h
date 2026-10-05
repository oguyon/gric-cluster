/**
 * @file cli_terminal.h
 * @brief Terminal dimension detection and pager launch helper functions.
 */

#ifndef CLI_TERMINAL_H
#define CLI_TERMINAL_H

/**
 * @brief Query current terminal width in columns.
 * @return Column count (defaults to 80 if cannot be determined).
 */
int cli_get_terminal_width(
    void);

/**
 * @brief Query current terminal height in rows.
 * @return Row count (defaults to 24 if cannot be determined).
 */
int cli_get_terminal_height(
    void);

/**
 * @brief Display long content in a pager (`less -RF`) if interactive and content overflows.
 * @param content String content to display.
 */
void cli_print_pager(
    const char *content);

#endif // CLI_TERMINAL_H

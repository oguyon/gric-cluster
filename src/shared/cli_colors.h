#ifndef CLI_COLORS_H
#define CLI_COLORS_H

/**
 * @file cli_colors.h
 * @brief Shared ANSI color variable declarations and macros for the GRIC suite.
 */

extern const char *ansi_color_orange;
extern const char *ansi_color_green;
extern const char *ansi_color_red;
extern const char *ansi_color_blue;
extern const char *ansi_bg_green;
extern const char *ansi_color_black;
extern const char *ansi_color_reset;
#define ansi_reset ansi_color_reset
extern const char *ansi_bold;
extern const char *ansi_underline;
extern const char *ansi_bold_cyan;
extern const char *ansi_bold_green;
extern const char *ansi_color_magenta;
extern const char *ansi_color_yellow;
extern const char *ansi_color_grey;
extern const char *ansi_color_cyan;

#define ANSI_COLOR_ORANGE  ansi_color_orange
#define ANSI_COLOR_GREEN   ansi_color_green
#define ANSI_COLOR_RED     ansi_color_red
#define ANSI_COLOR_BLUE    ansi_color_blue
#define ANSI_BG_GREEN      ansi_bg_green
#define ANSI_COLOR_BLACK   ansi_color_black
#define ANSI_COLOR_RESET   ansi_color_reset
#define ANSI_BOLD          ansi_bold
#define ANSI_UNDERLINE     ansi_underline
#define ANSI_BOLD_CYAN     ansi_bold_cyan
#define ANSI_BOLD_GREEN    ansi_bold_green
#define ANSI_COLOR_MAGENTA ansi_color_magenta
#define ANSI_COLOR_YELLOW  ansi_color_yellow
#define ANSI_COLOR_GREY    ansi_color_grey
#define ANSI_COLOR_CYAN    ansi_color_cyan

/**
 * @brief Initialize color variables if NO_COLOR environment variable is not present.
 */
void cli_colors_init(void);

/**
 * @brief Print a message indicating whether color mode is enabled or disabled.
 */
void cli_print_color_mode(void);

/**
 * @brief Print a single line, styling option flags and defaults.
 * @param line The string line to style and print.
 */
void cli_print_colored_line(const char *line);

/**
 * @brief Print usage information with styled command and placeholders.
 * @param usage The usage string.
 */
void cli_print_colored_usage(const char *usage);

/**
 * @brief Parse and print a block of options, formatting each line.
 * @param options The multiline options block.
 */
void cli_print_colored_options(const char *options);

/**
 * @brief Parse and print a block of examples, highlighting commands and symbols.
 * @param examples The multiline examples block.
 */
void cli_print_colored_examples(const char *examples);

/**
 * @brief Print a "See Also" option reference line.
 * @param option Option or keyword name.
 * @param desc   Description text.
 */
void cli_print_see_also_option(const char *option, const char *desc);

void cli_colors_init_force(
    int force_color);

int cli_get_terminal_width(
    void);

int cli_get_terminal_height(
    void);

int cli_is_color_enabled(
    void);

void cli_print_wrapped_line(
    const char *text,
    int         len,
    int         indent,
    int         width);

void cli_print_rich_segment(
    const char *text,
    int         len,
    int         is_bold);

void cli_print_help_section(
    const char *label,
    const char *value);

void cli_print_header_box(
    const char *title);

void cli_print_pager(
    const char *content);

void cli_suggest_similar_topic(
    const char         *topic,
    const char *const  *topics,
    int                 ntopics);

#endif // CLI_COLORS_H


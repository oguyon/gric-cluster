/**
 * @file cli_theme.h
 * @brief ANSI color variable declarations and theme configuration for the GRIC suite.
 */

#ifndef CLI_THEME_H
#define CLI_THEME_H

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
void cli_colors_init(
    void);

/**
 * @brief Force enable or disable color escapes regardless of NO_COLOR environment.
 * @param force_color 1 to enable color, 0 to disable.
 */
void cli_colors_init_force(
    int force_color);

/**
 * @brief Check if color output is currently enabled.
 * @return 1 if enabled, 0 otherwise.
 */
int cli_is_color_enabled(
    void);

/**
 * @brief Print a message indicating whether color mode is enabled or disabled.
 */
void cli_print_color_mode(
    void);

#endif // CLI_THEME_H

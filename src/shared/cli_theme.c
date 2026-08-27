/**
 * @file cli_theme.c
 * @brief Implementation of shared ANSI theme management and color mode detection.
 */

#include "shared/cli_theme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *ansi_color_orange = "";
const char *ansi_color_green = "";
const char *ansi_color_red = "";
const char *ansi_color_blue = "";
const char *ansi_bg_green = "";
const char *ansi_color_black = "";
const char *ansi_color_reset = "";
const char *ansi_bold = "";
const char *ansi_underline = "";
const char *ansi_bold_cyan = "";
const char *ansi_bold_green = "";
const char *ansi_color_magenta = "";
const char *ansi_color_yellow = "";
const char *ansi_color_grey = "";
const char *ansi_color_cyan = "";

void cli_colors_init(
    void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL && isatty(STDOUT_FILENO))
    {
        ansi_color_orange = "\x1b[38;5;208m";
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_blue = "\x1b[34m";
        ansi_bg_green = "\x1b[42m";
        ansi_color_black = "\x1b[30m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_underline = "\x1b[4m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;38;5;154m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_cyan = "\x1b[36m";
    }
} // cli_colors_init

void cli_colors_init_force(
    int force_color)
{
    if (force_color)
    {
        ansi_color_orange = "\x1b[38;5;208m";
        ansi_color_green = "\x1b[32m";
        ansi_color_red = "\x1b[31m";
        ansi_color_blue = "\x1b[34m";
        ansi_bg_green = "\x1b[42m";
        ansi_color_black = "\x1b[30m";
        ansi_color_reset = "\x1b[0m";
        ansi_bold = "\x1b[1m";
        ansi_underline = "\x1b[4m";
        ansi_bold_cyan = "\x1b[1;36m";
        ansi_bold_green = "\x1b[1;38;5;154m";
        ansi_color_magenta = "\x1b[35m";
        ansi_color_yellow = "\x1b[33m";
        ansi_color_grey = "\x1b[90m";
        ansi_color_cyan = "\x1b[36m";
    }
    else
    {
        ansi_color_orange = "";
        ansi_color_green = "";
        ansi_color_red = "";
        ansi_color_blue = "";
        ansi_bg_green = "";
        ansi_color_black = "";
        ansi_color_reset = "";
        ansi_bold = "";
        ansi_underline = "";
        ansi_bold_cyan = "";
        ansi_bold_green = "";
        ansi_color_magenta = "";
        ansi_color_yellow = "";
        ansi_color_grey = "";
        ansi_color_cyan = "";
    }
} // cli_colors_init_force

int cli_is_color_enabled(
    void)
{
    return (ansi_color_reset != NULL && strlen(ansi_color_reset) > 0);
} // cli_is_color_enabled

void cli_print_color_mode(
    void)
{
    const char *no_color = getenv("NO_COLOR");

    printf("\n%sCOLOR MODE%s\n", ANSI_BOLD_CYAN, ANSI_COLOR_RESET);
    if (no_color == NULL)
    {
        printf("  %sENABLED%s (color escape codes are active; disable by setting NO_COLOR=1)\n",
               ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
    }
    else
    {
        printf("  %sDISABLED%s (NO_COLOR environment variable is present)\n",
               ANSI_COLOR_RED, ANSI_COLOR_RESET);
    }
} // cli_print_color_mode

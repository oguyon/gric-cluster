/**
 * @file status_tui.c
 * @brief TUI rendering and color configuration functions for gric-status.
 */

#include "status_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int r;
    int g;
    int b;
} RgbColor;

/**
 * get_rgb_color() - Maps color code to RGB values.
 * @color: The color code constant.
 * @is_bg: Non-zero if mapping for background, zero for foreground.
 *
 * Return: RgbColor structure with mapped RGB values.
 */
static RgbColor get_rgb_color(
    int color,
    int is_bg)
{
    RgbColor rgb;

    if (is_bg)
    {
        rgb.r = 20;
        rgb.g = 22;
        rgb.b = 28;
    }
    else
    {
        rgb.r = 248;
        rgb.g = 249;
        rgb.b = 250;
    }

    switch (color)
    {
        case COLOR_BLACK:
            rgb.r = 20;
            rgb.g = 22;
            rgb.b = 28;
            break;
        case COLOR_RED:
            rgb.r = 220;
            rgb.g = 53;
            rgb.b = 69;
            break;
        case COLOR_GREEN:
            rgb.r = 40;
            rgb.g = 167;
            rgb.b = 69;
            break;
        case COLOR_YELLOW:
            rgb.r = 255;
            rgb.g = 193;
            rgb.b = 7;
            break;
        case COLOR_BLUE:
            rgb.r = 0;
            rgb.g = 123;
            rgb.b = 255;
            break;
        case COLOR_MAGENTA:
            rgb.r = 111;
            rgb.g = 66;
            rgb.b = 193;
            break;
        case COLOR_CYAN:
            rgb.r = 0;
            rgb.g = 180;
            rgb.b = 216;
            break;
        case COLOR_WHITE:
            rgb.r = 248;
            rgb.g = 249;
            rgb.b = 250;
            break;
        case COLOR_GREY:
            if (is_bg)
            {
                rgb.r = 30;
                rgb.g = 34;
                rgb.b = 40;
            }
            else
            {
                rgb.r = 100;
                rgb.g = 110;
                rgb.b = 120;
            }
            break;
        case COLOR_ORANGE:
            rgb.r = 253;
            rgb.g = 126;
            rgb.b = 20;
            break;
        default:
            break;
    }

    return rgb;
}

/**
 * get_256_color() - Maps color code to 256-color index.
 * @color: The color code constant.
 * @is_bg: Non-zero if mapping for background, zero for foreground.
 *
 * Return: 256-color index, or -1 if default/not set.
 */
static int get_256_color(
    int color,
    int is_bg)
{
    switch (color)
    {
        case COLOR_BLACK:   return 234;
        case COLOR_RED:     return 196;
        case COLOR_GREEN:   return 40;
        case COLOR_YELLOW:  return 220;
        case COLOR_BLUE:    return 33;
        case COLOR_MAGENTA: return 129;
        case COLOR_CYAN:    return 38;
        case COLOR_WHITE:   return 255;
        case COLOR_GREY:    return is_bg ? 236 : 244;
        case COLOR_ORANGE:  return 208;
        default:            return -1;
    }
}

/**
 * get_16_color() - Maps color code to 16-color ANSI code.
 * @color: The color code constant.
 * @is_bg: Non-zero if mapping for background, zero for foreground.
 *
 * Return: 16-color ANSI escape suffix code.
 */
static int get_16_color(
    int color,
    int is_bg)
{
    if (is_bg)
    {
        switch (color)
        {
            case COLOR_BLACK:   return 40;
            case COLOR_RED:     return 41;
            case COLOR_GREEN:   return 42;
            case COLOR_YELLOW:  return 43;
            case COLOR_BLUE:    return 44;
            case COLOR_MAGENTA: return 45;
            case COLOR_CYAN:    return 46;
            case COLOR_WHITE:   return 47;
            case COLOR_GREY:    return 100;
            case COLOR_ORANGE:  return 101;
            default:            return 40;
        }
    }
    else
    {
        switch (color)
        {
            case COLOR_BLACK:   return 30;
            case COLOR_RED:     return 31;
            case COLOR_GREEN:   return 32;
            case COLOR_YELLOW:  return 33;
            case COLOR_BLUE:    return 34;
            case COLOR_MAGENTA: return 35;
            case COLOR_CYAN:    return 36;
            case COLOR_WHITE:   return 37;
            case COLOR_GREY:    return 90;
            case COLOR_ORANGE:  return 91;
            default:            return 37;
        }
    }
}

/**
 * ov_detect_color_level() - Auto-detects terminal color capability.
 * @help_mono: If non-zero, forces monochrome (0 colors).
 *
 * Checks NO_COLOR, COLORTERM, and TERM environment variables to set
 * ov__color_level to 0 (monochrome), 1 (16 colors), 2 (256 colors),
 * or 3 (true color).
 */
void ov_detect_color_level(
    int help_mono)
{
    if (help_mono)
    {
        ov__color_level = 0;
        return;
    }

    if (getenv("NO_COLOR") != NULL)
    {
        ov__color_level = 0;
        return;
    }

    const char *colorterm = getenv("COLORTERM");
    if (colorterm &&
        (strcmp(colorterm, "truecolor") == 0 || strcmp(colorterm, "24bit") == 0))
    {
        ov__color_level = 3;
        return;
    }

    const char *term = getenv("TERM");
    if (term && (strstr(term, "256color") || strstr(term, "xterm-256")))
    {
        ov__color_level = 2;
        return;
    }

    ov__color_level = 1;
}

/**
 * tui_apply_colors() - Outputs ANSI sequences to apply foreground/background.
 * @fg:   Foreground color constant.
 * @bg:   Background color constant.
 * @bold: Bold flag (1 to enable bold, 0 for normal).
 *
 * Chooses appropriate ANSI sequences depending on detected ov__color_level.
 */
void tui_apply_colors(
    int fg,
    int bg,
    int bold)
{
    if (bold)
    {
        printf("\033[1m");
    }

    if (ov__color_level == 3) /* TrueColor */
    {
        RgbColor fg_rgb = get_rgb_color(fg, 0);
        RgbColor bg_rgb = get_rgb_color(bg, 1);

        if (fg != COLOR_DEFAULT)
        {
            printf("\033[38;2;%d;%d;%dm", fg_rgb.r, fg_rgb.g, fg_rgb.b);
        }
        if (bg != COLOR_DEFAULT)
        {
            printf("\033[48;2;%d;%d;%dm", bg_rgb.r, bg_rgb.g, bg_rgb.b);
        }
    }
    else if (ov__color_level == 2) /* 256-color */
    {
        int fg_256 = get_256_color(fg, 0);
        int bg_256 = get_256_color(bg, 1);

        if (fg_256 != -1)
        {
            printf("\033[38;5;%dm", fg_256);
        }
        if (bg_256 != -1)
        {
            printf("\033[48;5;%dm", bg_256);
        }
    }
    else if (ov__color_level == 1) /* 16-color */
    {
        if (fg != COLOR_DEFAULT)
        {
            printf("\033[%dm", get_16_color(fg, 0));
        }
        if (bg != COLOR_DEFAULT)
        {
            printf("\033[%dm", get_16_color(bg, 1));
        }
    }
}

/**
 * tui_clear() - Clears the shadow buffer.
 *
 * Fills the shadow buffer with spaces, reset colors, and disables bold.
 */
void tui_clear(void)
{
    for (int r = 0; r < sc_rows; r++)
    {
        for (int c = 0; c < sc_cols; c++)
        {
            shadow_buf[r][c].utf8[0] = ' ';
            shadow_buf[r][c].utf8[1] = '\0';
            shadow_buf[r][c].fg      = COLOR_DEFAULT;
            shadow_buf[r][c].bg      = COLOR_DEFAULT;
            shadow_buf[r][c].bold    = 0;
        }
    }
}

/**
 * tui_draw_string() - Draws a string in the shadow buffer.
 * @r:    Row index.
 * @c:    Column index.
 * @str:  Pointer to the UTF-8 encoded string.
 * @fg:   Foreground color code.
 * @bg:   Background color code.
 * @bold: Bold flag (1 or 0).
 */
void tui_draw_string(
    int         r,
    int         c,
    const char *str,
    int         fg,
    int         bg,
    int         bold)
{
    int col = c;
    const unsigned char *p = (const unsigned char *)str;

    while (*p && col < sc_cols)
    {
        if (r < 0 || r >= sc_rows)
        {
            break;
        }

        ScreenCell *cell = &shadow_buf[r][col];
        cell->fg   = fg;
        cell->bg   = bg;
        cell->bold = bold;

        int len = 1;
        if ((*p & 0x80) == 0) len = 1;
        else if ((*p & 0xE0) == 0xC0) len = 2;
        else if ((*p & 0xF0) == 0xE0) len = 3;
        else if ((*p & 0xF8) == 0xF0) len = 4;

        int i;
        for (i = 0; i < len && *p; i++)
        {
            cell->utf8[i] = *p;
            p++;
        }
        cell->utf8[i] = '\0';
        col++;
    }
}

/**
 * tui_draw_box() - Draws a box with borders and optional title.
 * @r:     Row index of top-left corner.
 * @c:     Column index of top-left corner.
 * @w:     Width of the box.
 * @h:     Height of the box.
 * @title: Optional title string (can be NULL or empty).
 * @fg:    Foreground color code.
 * @bg:    Background color code.
 * @bold:  Bold flag.
 */
void tui_draw_box(
    int         r,
    int         c,
    int         w,
    int         h,
    const char *title,
    int         fg,
    int         bg,
    int         bold)
{
    tui_draw_string(r, c, "┌", fg, bg, bold);
    for (int col = 1; col < w - 1; col++)
    {
        tui_draw_string(r, c + col, "─", fg, bg, bold);
    }
    tui_draw_string(r, c + w - 1, "┐", fg, bg, bold);

    for (int row = 1; row < h - 1; row++)
    {
        tui_draw_string(r + row, c, "│", fg, bg, bold);
        tui_draw_string(r + row, c + w - 1, "│", fg, bg, bold);
    }

    tui_draw_string(r + h - 1, c, "└", fg, bg, bold);
    for (int col = 1; col < w - 1; col++)
    {
        tui_draw_string(r + h - 1, c + col, "─", fg, bg, bold);
    }
    tui_draw_string(r + h - 1, c + w - 1, "┘", fg, bg, bold);

    if (title && strlen(title) > 0)
    {
        char temp_title[128];
        snprintf(temp_title, sizeof(temp_title), " %s ", title);
        tui_draw_string(r, c + 2, temp_title, fg, bg, bold);
    }
}

/**
 * tui_draw_label_button() - Draws a key indicator resembling a graphical button.
 * @r:     Row index.
 * @c:     Column index.
 * @label: Button label text.
 * @is_on: 1 if active (green highlight), 0 if inactive (grey highlight).
 */
void tui_draw_label_button(
    int         r,
    int         c,
    const char *label,
    int         is_on)
{
    char buf[64];
    snprintf(buf, sizeof(buf), " %s ", label);
    tui_draw_string(r, c, "[", COLOR_WHITE, COLOR_DEFAULT, 0);

    if (is_on)
    {
        tui_draw_string(r, c + 1, buf, COLOR_BLACK, COLOR_GREEN, 1);
    }
    else
    {
        tui_draw_string(r, c + 1, buf, COLOR_WHITE, COLOR_GREY, 1);
    }

    tui_draw_string(r, c + 1 + strlen(buf), "]", COLOR_WHITE, COLOR_DEFAULT, 0);
}

/**
 * format_5char_float() - Formats double into exactly 5-character string.
 * @out: Output buffer of size >= 6.
 * @val: Float value to format.
 */
void format_5char_float(
    char   *out,
    double  val)
{
    char temp[64];

    if (val >= 1000.0)
    {
        snprintf(temp, sizeof(temp), "%5.0f", val);
    }
    else if (val >= 100.0)
    {
        snprintf(temp, sizeof(temp), "%5.1f", val);
    }
    else if (val >= 10.0)
    {
        snprintf(temp, sizeof(temp), "%5.2f", val);
    }
    else
    {
        snprintf(temp, sizeof(temp), "%5.3f", val);
    }

    temp[5] = '\0';
    int len = strlen(temp);
    if (len < 5)
    {
        snprintf(out, 6, "%-5s", temp);
    }
    else
    {
        strcpy(out, temp);
    }
}

/**
 * format_5char_int() - Formats uint64_t into exactly 5-character string.
 * @out: Output buffer of size >= 6.
 * @val: Integer value to format.
 */
void format_5char_int(
    char     *out,
    uint64_t  val)
{
    char temp[64];
    snprintf(temp, sizeof(temp), "%5lu", (unsigned long)val);
    temp[5] = '\0';

    int len = strlen(temp);
    if (len < 5)
    {
        snprintf(out, 6, "%-5s", temp);
    }
    else
    {
        strcpy(out, temp);
    }
}

/**
 * tui_flush() - Diff-renders the shadow screen buffer onto the stdout.
 *
 * Emits ANSI sequence to trigger synchronized screen update, checks diffs
 * between shadow_buf and front_buf, writes minimum needed characters, and ends
 * synchronized update.
 */
void tui_flush(void)
{
    printf("\033[?2026h"); /* Start synchronized update */

    int last_fg   = -2;
    int last_bg   = -2;
    int last_bold = -1;

    for (int r = 0; r < sc_rows; r++)
    {
        for (int c = 0; c < sc_cols; c++)
        {
            ScreenCell shadow = shadow_buf[r][c];
            ScreenCell front  = front_buf[r][c];

            if (strcmp(shadow.utf8, front.utf8) != 0 || shadow.fg != front.fg ||
                shadow.bg != front.bg || shadow.bold != front.bold)
            {
                printf("\033[%d;%dH", r + 1, c + 1);

                if (shadow.fg != last_fg || shadow.bg != last_bg || shadow.bold != last_bold)
                {
                    printf("\033[0m");
                    tui_apply_colors(shadow.fg, shadow.bg, shadow.bold);
                    last_fg   = shadow.fg;
                    last_bg   = shadow.bg;
                    last_bold = shadow.bold;
                }

                printf("%s", shadow.utf8);
                front_buf[r][c] = shadow;
            }
        }
    }

    printf("\033[?2026l"); /* End synchronized update */
    fflush(stdout);
}

/**
 * get_state_string() - Translates process state enum to display string.
 * @state: State enum code.
 * @pid:   Target process PID (to check if dead).
 *
 * Return: Status text label.
 */
const char *get_state_string(
    uint32_t state,
    pid_t    pid)
{
    if (state == GRIC_STATUS_INIT || state == GRIC_STATUS_RUNNING)
    {
        if (!is_pid_alive(pid))
        {
            return "DEAD / CRASHED";
        }
    }

    switch (state)
    {
        case GRIC_STATUS_INIT:    return "INITIALIZING";
        case GRIC_STATUS_RUNNING: return "RUNNING";
        case GRIC_STATUS_SUCCESS: return "SUCCESS";
        case GRIC_STATUS_ERROR:   return "ERROR";
        case GRIC_STATUS_ABORTED: return "ABORTED";
        default:                  return "UNKNOWN";
    }
}

/**
 * get_state_color() - Determines color depending on state and PID activity.
 * @state: State enum code.
 * @pid:   Target process PID.
 *
 * Return: UI color code constant.
 */
int get_state_color(
    uint32_t state,
    pid_t    pid)
{
    if (state == GRIC_STATUS_INIT || state == GRIC_STATUS_RUNNING)
    {
        if (!is_pid_alive(pid))
        {
            return COLOR_RED;
        }
    }

    switch (state)
    {
        case GRIC_STATUS_INIT:    return COLOR_YELLOW;
        case GRIC_STATUS_RUNNING: return COLOR_BLUE;
        case GRIC_STATUS_SUCCESS: return COLOR_GREEN;
        case GRIC_STATUS_ERROR:   return COLOR_RED;
        case GRIC_STATUS_ABORTED: return COLOR_RED;
        default:                  return COLOR_WHITE;
    }
}

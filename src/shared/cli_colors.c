/**
 * @file cli_colors.c
 * @brief Implementations of shared ANSI color management and CLI styling helper functions.
 */

#include "shared/cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void cli_colors_init(void)
{
    const char *no_color = getenv("NO_COLOR");

    if (no_color == NULL)
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
}

void cli_print_color_mode(void)
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
}

void cli_print_colored_usage(
    const char *usage)
{
    printf("  ");
    const char *p = usage;

    while (*p == ' ' || *p == '\t')
    {
        p++;
    }

    const char *cmd_start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n')
    {
        p++;
    }
    printf("%s%.*s%s",
           ANSI_BOLD_GREEN,
           (int)(p - cmd_start), cmd_start,
           ANSI_COLOR_RESET);

    while (*p)
    {
        if (*p == '<')
        {
            const char *end = strchr(p, '>');
            if (end)
            {
                printf("%s%.*s%s", ANSI_COLOR_MAGENTA, (int)(end - p + 1), p, ANSI_COLOR_RESET);
                p = end + 1;
                continue;
            }
        }
        if (*p == '[')
        {
            const char *end = strchr(p, ']');
            if (end)
            {
                printf("%s%.*s%s", ANSI_COLOR_GREY, (int)(end - p + 1), p, ANSI_COLOR_RESET);
                p = end + 1;
                continue;
            }
        }
        putchar(*p);
        p++;
    }
    printf("\n");
}

void cli_print_colored_line(
    const char *line)
{
    const char *start = line;

    while (*start == ' ' || *start == '\t')
    {
        start++;
    }

    if (*start == '-')
    {
        printf("%.*s", (int)(start - line), line);

        const char *end = start;
        while (*end && *end != '\n')
        {
            if ((*end == ' ' && *(end + 1) == ' ') || *end == '\t')
            {
                break;
            }
            end++;
        }

        const char *p = start;
        while (p < end)
        {
            if (*p == '-')
            {
                const char *f_end = p;
                while (f_end < end && *f_end != ' ' && *f_end != '\t' && *f_end != ',' &&
                       *f_end != '<' && *f_end != '[')
                {
                    f_end++;
                }
                printf("%s%.*s%s", ANSI_COLOR_GREEN, (int)(f_end - p), p, ANSI_COLOR_RESET);
                p = f_end;
            }
            else if (*p == '<')
            {
                const char *v_end = p;
                while (v_end < end && *v_end != '>')
                {
                    v_end++;
                }
                if (v_end < end)
                {
                    printf("%s%.*s%s", ANSI_COLOR_MAGENTA, (int)(v_end - p + 1), p,
                           ANSI_COLOR_RESET);
                    p = v_end + 1;
                }
                else
                {
                    putchar(*p);
                    p++;
                }
            }
            else
            {
                putchar(*p);
                p++;
            }
        }

        const char *desc = end;
        while (*desc)
        {
            if (strncmp(desc, "default:", 8) == 0 || strncmp(desc, "Default:", 8) == 0)
            {
                printf("%sdefault:%s", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
                desc += 8;
                const char *val_end = desc;
                while (*val_end && *val_end != ')' && *val_end != ',')
                {
                    val_end++;
                }
                printf("%s%.*s%s", ANSI_COLOR_CYAN, (int)(val_end - desc), desc,
                       ANSI_COLOR_RESET);
                desc = val_end;
            }
            else if (strncmp(desc, "caution", 7) == 0 || strncmp(desc, "Caution", 7) == 0)
            {
                printf("%scaution%s", ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
                desc += 7;
            }
            else if (strncmp(desc, "[DISABLED]", 10) == 0)
            {
                printf("%s[DISABLED]%s", ANSI_COLOR_YELLOW, ANSI_COLOR_RESET);
                desc += 10;
            }
            else
            {
                putchar(*desc);
                desc++;
            }
        }
        printf("\n");
    }
    else
    {
        int leading_spaces = 0;
        while (line[leading_spaces] == ' ')
        {
            leading_spaces++;
        }
        if (leading_spaces == 2 && line[leading_spaces] != '\0')
        {
            printf("  %s%s%s\n", ANSI_BOLD, line + 2, ANSI_COLOR_RESET);
        }
        else
        {
            printf("%s\n", line);
        }
    }
}

void cli_print_colored_options(
    const char *options)
{
    const char *line = options;

    while (line && *line)
    {
        const char *next = strchr(line, '\n');
        int len = next ? (int)(next - line) : (int)strlen(line);
        char buf[1024];

        if (len >= (int)sizeof(buf))
        {
            len = sizeof(buf) - 1;
        }
        strncpy(buf, line, len);
        buf[len] = '\0';

        cli_print_colored_line(buf);
        line = next ? next + 1 : NULL;
    }
}

void cli_print_colored_examples(
    const char *examples)
{
    const char *line = examples;

    while (line && *line)
    {
        const char *next = strchr(line, '\n');
        int len = next ? (int)(next - line) : (int)strlen(line);
        char buf[1024];

        if (len >= (int)sizeof(buf))
        {
            len = sizeof(buf) - 1;
        }
        strncpy(buf, line, len);
        buf[len] = '\0';

        const char *dollar = strstr(buf, "$ ");
        if (dollar)
        {
            printf("%.*s", (int)(dollar - buf), buf);
            printf("%s$%s ", ANSI_COLOR_GREY, ANSI_COLOR_RESET);

            const char *cmd = dollar + 2;
            const char *end = cmd;
            while (*end && *end != ' ' && *end != '\t')
            {
                end++;
            }
            printf("%s%.*s%s", ANSI_BOLD_GREEN, (int)(end - cmd), cmd, ANSI_COLOR_RESET);

            const char *p = end;
            while (*p)
            {
                if (*p == '-')
                {
                    const char *f_end = p;
                    while (*f_end && *f_end != ' ' && *f_end != '\t')
                    {
                        f_end++;
                    }
                    printf("%s%.*s%s", ANSI_COLOR_GREEN, (int)(f_end - p), p, ANSI_COLOR_RESET);
                    p = f_end;
                }
                else if (*p == '>' || *p == '<' || *p == '|' || *p == '&')
                {
                    printf("%s%c%s", ANSI_COLOR_GREY, *p, ANSI_COLOR_RESET);
                    p++;
                }
                else
                {
                    putchar(*p);
                    p++;
                }
            }
            printf("\n");
        }
        else
        {
            printf("%s\n", buf);
        }
        line = next ? next + 1 : NULL;
    }
}

void cli_print_see_also_option(
    const char *option,
    const char *desc)
{
    printf("  %s%-24s%s %s\n",
           ANSI_BOLD_GREEN, option,
           ANSI_COLOR_RESET, desc);
}

/**
 * @file cli_wrap.c
 * @brief Implementation of word wrapping and rich inline formatting.
 */

#include "shared/cli_wrap.h"
#include "shared/cli_theme.h"
#include <stdio.h>
#include <string.h>

void cli_print_rich_segment(
    const char *text,
    int         len,
    int         is_bold)
{
    const char *p = text;
    const char *end = text + len;

    while (p < end)
    {
        if (*p == '<')
        {
            const char *close = memchr(p, '>', (size_t)(end - p));
            if (close != NULL)
            {
                printf("%s%.*s%s",
                       ANSI_COLOR_MAGENTA,
                       (int)(close - p + 1), p,
                       ANSI_COLOR_RESET);
                if (is_bold)
                {
                    printf("%s", ANSI_BOLD);
                }
                p = close + 1;
                continue;
            }
        }

        if (*p == '`')
        {
            const char *close = memchr(p + 1, '`', (size_t)(end - (p + 1)));
            if (close != NULL)
            {
                printf("%s%.*s%s",
                       ANSI_COLOR_YELLOW,
                       (int)(close - (p + 1)), p + 1,
                       ANSI_COLOR_RESET);
                if (is_bold)
                {
                    printf("%s", ANSI_BOLD);
                }
                p = close + 1;
                continue;
            }
        }

        if ((p + 5 <= end)
            && memcmp(p, "gric-", 5) == 0
            && (p == text
                || *(p - 1) == ' '
                || *(p - 1) == '\t'))
        {
            const char *we = p + 5;
            while (we < end
                   && *we != ' '
                   && *we != '\t'
                   && *we != '\n')
            {
                we++;
            }
            printf("%s%.*s%s",
                   ANSI_BOLD_GREEN,
                   (int)(we - p), p,
                   ANSI_COLOR_RESET);
            if (is_bold)
            {
                printf("%s", ANSI_BOLD);
            }
            p = we;
            continue;
        }

        if (*p == '-'
            && (p == text
                || *(p - 1) == ' '
                || *(p - 1) == '(')
            && (p + 1 < end)
            && (*(p + 1) >= 'a' && *(p + 1) <= 'z'))
        {
            const char *oe = p + 1;
            while (oe < end
                   && *oe != ' '
                   && *oe != '\t'
                   && *oe != ','
                   && *oe != ')'
                   && *oe != '\n')
            {
                oe++;
            }
            printf("%s%.*s%s",
                   ANSI_COLOR_GREEN,
                   (int)(oe - p), p,
                   ANSI_COLOR_RESET);
            if (is_bold)
            {
                printf("%s", ANSI_BOLD);
            }
            p = oe;
            continue;
        }

        putchar(*p);
        p++;
    }
} // cli_print_rich_segment

void cli_print_wrapped_line(
    const char *text,
    int         len,
    int         indent,
    int         width)
{
    int usable = width - indent;
    if (usable < 20)
    {
        usable = 20;
    }

    const char *p = text;
    const char *end = text + len;

    while (p < end)
    {
        if (p != text)
        {
            while (p < end && *p == ' ')
            {
                p++;
            }
            if (p >= end)
            {
                break;
            }
        }

        int remaining = (int)(end - p);
        if (remaining <= usable)
        {
            for (int i = 0; i < indent; i++)
            {
                putchar(' ');
            }
            cli_print_rich_segment(p, remaining, 0);
            putchar('\n');
            break;
        }

        int brk = usable;
        while (brk > 0 && p[brk] != ' ')
        {
            brk--;
        }
        if (brk == 0)
        {
            brk = usable;
        }

        for (int i = 0; i < indent; i++)
        {
            putchar(' ');
        }
        cli_print_rich_segment(p, brk, 0);
        putchar('\n');
        p += brk;
    }
} // cli_print_wrapped_line

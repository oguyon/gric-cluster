/**
 * @file cli_terminal.c
 * @brief Implementation of terminal dimension detection and pager launching.
 */

#include "shared/cli_terminal.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>

int cli_get_terminal_width(
    void)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    {
        return ws.ws_col;
    }
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    {
        return ws.ws_col;
    }
    if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    {
        return ws.ws_col;
    }
    const char *env = getenv("COLUMNS");
    if (env != NULL)
    {
        int c = atoi(env);
        if (c > 0)
        {
            return c;
        }
    }
    return 80;
} // cli_get_terminal_width

int cli_get_terminal_height(
    void)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
    {
        return ws.ws_row;
    }
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
    {
        return ws.ws_row;
    }
    if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
    {
        return ws.ws_row;
    }
    const char *env = getenv("LINES");
    if (env != NULL)
    {
        int r = atoi(env);
        if (r > 0)
        {
            return r;
        }
    }
    return 24;
} // cli_get_terminal_height

void cli_print_pager(
    const char *content)
{
    int use_pager = 0;
    if (isatty(STDOUT_FILENO))
    {
        int height = cli_get_terminal_height();
        int lines = 0;
        const char *p = content;
        while (*p != '\0')
        {
            if (*p == '\n')
            {
                lines++;
            }
            p++;
        }
        if (lines > height)
        {
            use_pager = 1;
        }
    }

    if (use_pager)
    {
        FILE *pager = popen("less -RF", "w");
        if (pager != NULL)
        {
            fprintf(pager, "%s", content);
            pclose(pager);
            return;
        }
    }
    printf("%s", content);
} // cli_print_pager

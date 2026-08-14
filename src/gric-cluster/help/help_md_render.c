/**
 * @file help_md_render.c
 * @brief Terminal ANSI Markdown parser and renderer implementation.
 */

#define _POSIX_C_SOURCE 200809L
#include "help_md_render.h"
#include "shared/cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/ioctl.h>

/**
 * @brief Get the current terminal width, defaulting to 80 if not available.
 */
static int get_terminal_width(void)
{
    struct winsize ws;
    if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20)
    {
        return (ws.ws_col > 120) ? 120 : (int)ws.ws_col;
    }
    return 80;
}

/**
 * @brief Print a line of text with inline Markdown formatting (`code`, **bold**, -flags, <args>).
 */
static void print_inline_formatted(
    const char *text,
    int         len)
{
    const char *p = text;
    const char *end = text + len;

    while (p < end)
    {
        /* Inline code: `...` */
        if (*p == '`')
        {
            const char *code_end = p + 1;
            while (code_end < end && *code_end != '`')
            {
                code_end++;
            }
            if (code_end < end)
            {
                p++; /* Skip opening backtick */
                const char *tok_start = p;
                int tok_len = (int)(code_end - p);

                /* If token is a CLI flag like `-flag` */
                if (tok_len > 1 && tok_start[0] == '-')
                {
                    printf("%s%.*s%s",
                           ANSI_COLOR_GREEN, tok_len, tok_start,
                           ANSI_COLOR_RESET);
                }
                else
                {
                    printf("%s%.*s%s",
                           ANSI_BOLD, tok_len, tok_start,
                           ANSI_COLOR_RESET);
                }
                p = code_end + 1;
                continue;
            }
        }

        /* Bold text: **...** */
        if (p + 1 < end && *p == '*' && *(p + 1) == '*')
        {
            const char *bold_end = p + 2;
            while (bold_end + 1 < end && !(bold_end[0] == '*' && bold_end[1] == '*'))
            {
                bold_end++;
            }
            if (bold_end + 1 < end)
            {
                p += 2; /* Skip opening ** */
                int bold_len = (int)(bold_end - p);
                printf("%s%.*s%s",
                       ANSI_BOLD, bold_len, p,
                       ANSI_COLOR_RESET);
                p = bold_end + 2;
                continue;
            }
        }

        /* CLI parameters: <param> */
        if (*p == '<')
        {
            const char *param_end = p + 1;
            while (param_end < end && *param_end != '>')
            {
                param_end++;
            }
            if (param_end < end)
            {
                int param_len = (int)(param_end - p + 1);
                printf("%s%.*s%s",
                       ANSI_COLOR_MAGENTA, param_len, p,
                       ANSI_COLOR_RESET);
                p = param_end + 1;
                continue;
            }
        }

        putchar(*p);
        p++;
    }
}

/**
 * @brief Wrap and print a paragraph of text indented by 2 spaces.
 */
static void print_wrapped_paragraph(
    const char *text,
    int         indent,
    int         term_width)
{
    int max_width = term_width - indent - 2;
    if (max_width < 30)
    {
        max_width = 30;
    }

    const char *p = text;
    while (*p)
    {
        /* Skip leading whitespace */
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (!*p)
        {
            break;
        }

        /* Find line break boundary up to max_width */
        const char *line_start = p;
        const char *last_space = NULL;
        int col = 0;

        while (*p && col < max_width)
        {
            if (*p == ' ' || *p == '\t')
            {
                last_space = p;
            }
            col++;
            p++;
        }

        int line_len;
        if (!*p || col < max_width)
        {
            line_len = (int)(p - line_start);
        }
        else if (last_space != NULL && last_space > line_start)
        {
            line_len = (int)(last_space - line_start);
            p = last_space + 1;
        }
        else
        {
            line_len = col;
        }

        /* Print indent */
        for (int ii = 0; ii < indent; ii++)
        {
            putchar(' ');
        }

        print_inline_formatted(line_start, line_len);
        putchar('\n');
    }
}

void render_markdown_help(
    const char *md_text)
{
    if (md_text == NULL)
    {
        return;
    }

    cli_colors_init();
    int term_width = get_terminal_width();

    const char *p = md_text;
    int in_code_block = 0;
    int is_see_also = 0;
    int first_section = 1;
    int pending_blank = 0;

    while (*p)
    {
        /* Extract single line */
        const char *line_start = p;
        while (*p && *p != '\n')
        {
            p++;
        }
        int line_len = (int)(p - line_start);
        if (*p == '\n')
        {
            p++;
        }

        /* Skip trailing carriage return */
        if (line_len > 0 && line_start[line_len - 1] == '\r')
        {
            line_len--;
        }

        /* Skip leading H1 title (# keyword) */
        if (line_len >= 2 && line_start[0] == '#' && line_start[1] == ' ')
        {
            continue;
        }

        /* Code block toggle: ``` */
        if (line_len >= 3 && strncmp(line_start, "```", 3) == 0)
        {
            in_code_block = !in_code_block;
            continue;
        }

        if (in_code_block)
        {
            printf("    %s%.*s%s\n",
                   ANSI_COLOR_GREEN, line_len, line_start,
                   ANSI_COLOR_RESET);
            pending_blank = 0;
            continue;
        }

        /* Blank line in markdown */
        if (line_len == 0)
        {
            pending_blank = 1;
            continue;
        }

        /* Section Heading: ## SECTION */
        if (line_len >= 3 && line_start[0] == '#' && line_start[1] == '#' && line_start[2] == ' ')
        {
            const char *sec = line_start + 3;
            int sec_len = line_len - 3;
            while (sec_len > 0 && *sec == ' ')
            {
                sec++;
                sec_len--;
            }

            if (!first_section)
            {
                putchar('\n');
            }
            first_section = 0;
            pending_blank = 0;

            printf("%s%.*s%s\n",
                   ANSI_BOLD_CYAN, sec_len, sec,
                   ANSI_COLOR_RESET);

            if (sec_len == 8 && strncmp(sec, "SEE ALSO", 8) == 0)
            {
                is_see_also = 1;
            }
            else
            {
                is_see_also = 0;
            }
            continue;
        }

        /* Subsection Heading: ### Subsection */
        if (line_len >= 4 && strncmp(line_start, "### ", 4) == 0)
        {
            const char *sub = line_start + 4;
            int sub_len = line_len - 4;
            printf("\n  %s%.*s%s\n",
                   ANSI_BOLD, sub_len, sub,
                   ANSI_COLOR_RESET);
            pending_blank = 0;
            continue;
        }

        /* Apply pending blank line between elements */
        if (pending_blank)
        {
            putchar('\n');
            pending_blank = 0;
        }

        /* SEE ALSO bullet item: - `-flag`: desc or - `topic`: desc */
        if (is_see_also && (line_start[0] == '-' || line_start[0] == '*'))
        {
            const char *item = line_start + 1;
            while (*item == ' ' || *item == '\t')
            {
                item++;
            }

            /* Extract option `...` */
            if (*item == '`')
            {
                const char *opt_start = item + 1;
                const char *opt_end = strchr(opt_start, '`');
                if (opt_end != NULL)
                {
                    char opt_buf[64];
                    int opt_len = (int)(opt_end - opt_start);
                    if (opt_len > 63)
                    {
                        opt_len = 63;
                    }
                    strncpy(opt_buf, opt_start, (size_t)opt_len);
                    opt_buf[opt_len] = '\0';

                    const char *desc = opt_end + 1;
                    while (*desc == ':' || *desc == ' ' || *desc == '\t')
                    {
                        desc++;
                    }

                    int desc_len = (int)((line_start + line_len) - desc);
                    char desc_buf[256];
                    if (desc_len > 255)
                    {
                        desc_len = 255;
                    }
                    if (desc_len > 0)
                    {
                        strncpy(desc_buf, desc, (size_t)desc_len);
                        desc_buf[desc_len] = '\0';
                    }
                    else
                    {
                        desc_buf[0] = '\0';
                    }

                    cli_print_see_also_option(opt_buf, desc_buf);
                    continue;
                }
            }
        }

        /* Regular bullet item: - item or * item */
        if ((line_start[0] == '-' || line_start[0] == '*') &&
            line_len > 1 && line_start[1] == ' ')
        {
            printf("  %s•%s ", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
            print_inline_formatted(line_start + 2, line_len - 2);
            putchar('\n');
            continue;
        }

        /* Numbered list: 1. item */
        if (isdigit((unsigned char)line_start[0]) && line_len > 3 &&
            line_start[1] == '.' && line_start[2] == ' ')
        {
            printf("  %s%c.%s ", ANSI_BOLD, line_start[0], ANSI_COLOR_RESET);
            print_inline_formatted(line_start + 3, line_len - 3);
            putchar('\n');
            continue;
        }

        /* Indented lines (e.g. preformatted ASCII diagrams or formula lines) */
        if (line_start[0] == ' ' || line_start[0] == '\t')
        {
            printf("  ");
            print_inline_formatted(line_start, line_len);
            putchar('\n');
            continue;
        }

        /* Markdown Table row: | col1 | col2 | ... */
        if (line_start[0] == '|')
        {
            int is_delim = 1;
            for (int ii = 0; ii < line_len; ii++)
            {
                char c = line_start[ii];
                if (c != '|' && c != ':' && c != '-' && c != ' ' && c != '\t')
                {
                    is_delim = 0;
                    break;
                }
            }

            if (is_delim)
            {
                printf("  ----------------------------  "
                       "----------------------------------------\n");
                continue;
            }

            char cells[4][256];
            int ncells = 0;
            const char *cur = line_start;
            const char *l_end = line_start + line_len;

            while (cur < l_end && ncells < 4)
            {
                if (*cur == '|')
                {
                    cur++;
                    const char *c_start = cur;
                    while (c_start < l_end && (*c_start == ' ' || *c_start == '\t'))
                    {
                        c_start++;
                    }
                    const char *c_end = c_start;
                    while (c_end < l_end && *c_end != '|')
                    {
                        c_end++;
                    }
                    const char *c_trim = c_end;
                    while (c_trim > c_start &&
                           (*(c_trim - 1) == ' ' || *(c_trim - 1) == '\t'))
                    {
                        c_trim--;
                    }
                    int clen = (int)(c_trim - c_start);
                    if (clen > 255)
                    {
                        clen = 255;
                    }
                    if (clen > 0)
                    {
                        strncpy(cells[ncells], c_start, (size_t)clen);
                        cells[ncells][clen] = '\0';
                        ncells++;
                    }
                    cur = c_end;
                }
                else
                {
                    cur++;
                }
            }

            if (ncells >= 2)
            {
                int c0_len = (int)strlen(cells[0]);
                const char *c0_p = cells[0];
                int is_bold = 0;

                if (c0_len >= 4 && c0_p[0] == '*' && c0_p[1] == '*' &&
                    c0_p[c0_len - 1] == '*' && c0_p[c0_len - 2] == '*')
                {
                    c0_p += 2;
                    c0_len -= 4;
                    is_bold = 1;
                }

                printf("  %s%.*s%s",
                       is_bold ? ANSI_BOLD : "",
                       c0_len, c0_p,
                       is_bold ? ANSI_COLOR_RESET : "");

                for (int sp = c0_len; sp < 28; sp++)
                {
                    putchar(' ');
                }
                printf("  ");
                print_inline_formatted(cells[1], (int)strlen(cells[1]));
                putchar('\n');
            }
            else if (ncells == 1)
            {
                printf("  ");
                print_inline_formatted(cells[0], (int)strlen(cells[0]));
                putchar('\n');
            }
            continue;
        }

        /* Normal paragraph text: word wrap to terminal */
        char para_buf[4096];
        if (line_len > (int)sizeof(para_buf) - 1)
        {
            line_len = (int)sizeof(para_buf) - 1;
        }
        strncpy(para_buf, line_start, (size_t)line_len);
        para_buf[line_len] = '\0';

        print_wrapped_paragraph(para_buf, 2, term_width);
    }
    putchar('\n');
}

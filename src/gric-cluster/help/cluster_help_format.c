/**
 * @file cluster_help_format.c
 * @brief Implementations of help text wrapping, colorizing, and section formatting.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_help_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "shared/cli_colors.h"

/**
 * get_terminal_width - detect terminal column count
 *
 * Tries ioctl(TIOCGWINSZ) first, falls back to the
 * COLUMNS environment variable, then defaults to 80.
 *
 * Return: terminal width in columns.
 */
static int get_terminal_width(void)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0
        && ws.ws_col > 0)
    {
        return ws.ws_col;
    }
    const char *env = getenv("COLUMNS");
    if (env)
    {
        int c = atoi(env);
        if (c > 0)
        {
            return c;
        }
    }
    return 80;
}

/* Forward declaration */
static void print_rich_segment(
    const char *text,
    int         len,
    int         is_bold);

/**
 * print_wrapped_line - word-wrap a prose line to width
 * @text:   pointer to start of line text
 * @len:    length of the text (no trailing newline)
 * @indent: number of spaces for first and continuation
 * @width:  terminal width in columns
 *
 * Breaks text at word boundaries so that no output line
 * exceeds @width columns. Each output line starts with
 * @indent spaces. The text is rendered through
 * print_rich_segment for syntax highlighting.
 */
static void print_wrapped_line(
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
        /* Skip leading spaces on continuation lines */
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
            /* Fits on one line */
            for (int i = 0; i < indent; i++)
            {
                putchar(' ');
            }
            print_rich_segment(p, remaining, 0);
            putchar('\n');
            break;
        }

        /* Find break point: last space at or before usable */
        int brk = usable;
        while (brk > 0 && p[brk] != ' ')
        {
            brk--;
        }
        if (brk == 0)
        {
            /* No space found; break at usable */
            brk = usable;
        }

        for (int i = 0; i < indent; i++)
        {
            putchar(' ');
        }
        print_rich_segment(p, brk, 0);
        putchar('\n');
        p += brk;
    }
}

/**
 * init_colors_help - initialize CLI colors
 */
void init_colors_help(void)
{
    cli_colors_init();
}

/**
 * print_see_also_option - print option reference in Green
 * @option: option name
 * @desc:   description text
 */
void print_see_also_option(
    const char *option,
    const char *desc)
{
    cli_print_see_also_option(option, desc);
}

/**
 * print_rich_segment - print text with inline highlighting
 * @text:    pointer to start of segment
 * @len:     number of bytes to print
 * @is_bold: nonzero if the surrounding line is bold
 *
 * Auto-highlights:
 *   gric-*  executables → yellow
 *   -option flags       → bold green
 *   <placeholder>       → magenta
 */
static void print_rich_segment(
    const char *text,
    int         len,
    int         is_bold)
{
    const char *p = text;
    const char *end = text + len;

    while (p < end)
    {
        /* <placeholder> → magenta */
        if (*p == '<')
        {
            const char *close = memchr(
                p, '>', (size_t)(end - p));
            if (close)
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

        /* `code` → yellow */
        if (*p == '`')
        {
            const char *close = memchr(
                p + 1, '`', (size_t)(end - (p + 1)));
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

        /* gric-* executable → yellow */
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

        /* -option → bold green */
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
}

/**
 * count_indent - count leading spaces in a line
 * @line: pointer to start of line
 * @len:  length of line
 *
 * Return: number of leading space characters.
 */
static int count_indent(
    const char *line,
    int         len)
{
    int n = 0;
    while (n < len && line[n] == ' ')
    {
        n++;
    }
    return n;
}

/**
 * is_verbatim_line - detect lines that should not reflow
 * @content: pointer to line content (indent stripped)
 * @clen:    length of content
 *
 * Return: 1 if the line must be printed as-is:
 *   - Bullet items:    "- text"
 *   - Numbered items:  "N. text"
 *   - Diagram pipes:   starts with |
 *   - Diagram arrows:  contains -->, <--, --+, ---
 */
static int is_verbatim_line(
    const char *content,
    int         clen)
{
    if (clen <= 0)
    {
        return 0;
    }

    /* Bullet item: "- text" */
    if (content[0] == '-' && clen > 1
        && content[1] == ' ')
    {
        return 1;
    }

    /* Numbered item: "N. text" (N is one or more digits) */
    if (content[0] >= '0' && content[0] <= '9')
    {
        for (int i = 1; i < clen; i++)
        {
            if (content[i] == '.'
                && i + 1 < clen
                && content[i + 1] == ' ')
            {
                return 1;
            }
            if (content[i] < '0'
                || content[i] > '9')
            {
                break;
            }
        }
    }

    /* Diagram pipe at start of line */
    if (content[0] == '|')
    {
        return 1;
    }

    /* Scan for diagram arrows / connectors */
    for (int i = 0; i < clen; i++)
    {
        /* --> or --+ or <-- */
        if (content[i] == '-' && i + 1 < clen
            && (content[i + 1] == '>'
                || content[i + 1] == '+'))
        {
            return 1;
        }
        if (content[i] == '<' && i + 1 < clen
            && content[i + 1] == '-')
        {
            return 1;
        }
        /* --- (3+ dashes = horizontal rule/connector) */
        if (content[i] == '-'
            && i + 2 < clen
            && content[i + 1] == '-'
            && content[i + 2] == '-')
        {
            return 1;
        }
    }
    return 0;
}

/**
 * print_help_section - print a labelled help section
 * @label: section heading (printed in bold cyan)
 * @value: body text with embedded newlines
 *
 * All text is word-wrapped to the terminal width.
 * Indentation is preserved: consecutive lines at the
 * same indent level are joined into paragraphs and
 * reflowed.
 *
 * ASCII diagram lines (containing |, +, -->, etc.)
 * are printed verbatim to preserve alignment.
 *
 * Sub-headers (non-indented line followed by an
 * indented line) are rendered in bold.
 */
void print_help_section(
    const char *label,
    const char *value)
{
    int width = get_terminal_width();

    printf("%s%s%s\n",
           ANSI_BOLD_CYAN, label, ANSI_COLOR_RESET);

    /*
     * Collect lines into an array for look-ahead
     * (sub-header detection).
     */
    const char *lines[512];
    int lens[512];
    int nlines = 0;

    {
        const char *ls = value;
        const char *p = value;
        while (*p != '\0')
        {
            if (*p == '\n')
            {
                if (nlines < 512)
                {
                    lines[nlines] = ls;
                    lens[nlines] = (int)(p - ls);
                    nlines++;
                }
                ls = p + 1;
            }
            p++;
        }
        if (p > ls && nlines < 512)
        {
            lines[nlines] = ls;
            lens[nlines] = (int)(p - ls);
            nlines++;
        }
    }

    /*
     * Paragraph buffer with tracked indent level.
     * Consecutive lines at the same indent are joined
     * into a paragraph and word-wrapped together.
     */
    char para[4096];
    int plen = 0;
    int para_indent = 0;  /* indent of current paragraph */

    /* Helper macro: flush the pending paragraph */
    #define FLUSH_PARA() do {                          \
        if (plen > 0)                                  \
        {                                              \
            print_wrapped_line(                        \
                para, plen,                            \
                2 + para_indent, width);               \
            plen = 0;                                  \
        }                                              \
    } while (0)

    for (int li = 0; li < nlines; li++)
    {
        const char *line = lines[li];
        int ll = lens[li];

        /* Empty line: flush paragraph, emit blank */
        if (ll == 0)
        {
            FLUSH_PARA();
            putchar('\n');
            continue;
        }

        int indent = count_indent(line, ll);
        const char *content = line + indent;
        int clen = ll - indent;

        /*
         * Sub-header detection: a line followed by
         * a more-indented line is rendered bold.
         *
         * Checked BEFORE verbatim so numbered
         * sub-headers like "1. PREDICT PRIORS" get
         * bold.  Bullets and diagram lines are
         * excluded unless they are numbered items
         * with ALL-CAPS text (section headers).
         *
         * Skipped when mid-paragraph at the same
         * indent (continuation prose).
         */
        if (indent == 0
            && !(plen > 0 && para_indent == 0))
        {
            /* Check if verbatim but allow numbered headers */
            int skip_subheader = 0;
            if (is_verbatim_line(content, clen))
            {
                skip_subheader = 1;
                /* Exception: numbered item with ALLCAPS text */
                if (content[0] >= '0'
                    && content[0] <= '9')
                {
                    int allcaps = 1;
                    int saw_alpha = 0;
                    for (int k = 0; k < clen; k++)
                    {
                        char c = content[k];
                        if (c >= 'a' && c <= 'z')
                        {
                            allcaps = 0;
                            break;
                        }
                        if (c >= 'A' && c <= 'Z')
                        {
                            saw_alpha = 1;
                        }
                    }
                    if (allcaps && saw_alpha)
                    {
                        skip_subheader = 0;
                    }
                }
            }

            if (!skip_subheader)
            {
                int next_indented = 0;
                if (li + 1 < nlines
                    && lens[li + 1] > 0)
                {
                    int next_indent =
                        count_indent(
                            lines[li + 1],
                            lens[li + 1]);
                    if (next_indent > indent)
                    {
                        next_indented = 1;
                    }
                }
                if (next_indented)
                {
                    FLUSH_PARA();
                    printf("  %s", ANSI_BOLD);
                    print_rich_segment(line, ll, 1);
                    printf("%s\n",
                           ANSI_COLOR_RESET);
                    continue;
                }
            }
        }

        /*
         * Verbatim line: flush and print as-is.
         */
        if (is_verbatim_line(content, clen))
        {
            FLUSH_PARA();
            printf("  ");
            print_rich_segment(line, ll, 0);
            putchar('\n');
            continue;
        }

        /*
         * Normal line: accumulate into paragraph.
         * Flush if indent level changed.
         */
        if (plen > 0 && indent != para_indent)
        {
            FLUSH_PARA();
        }
        para_indent = indent;

        /* Append content (stripped of leading indent) */
        if (plen > 0
            && plen + 1 + clen
                < (int)sizeof(para) - 1)
        {
            para[plen++] = ' ';
        }
        if (plen + clen < (int)sizeof(para) - 1)
        {
            memcpy(para + plen, content, clen);
            plen += clen;
        }
    } /* for each line */

    FLUSH_PARA();
    #undef FLUSH_PARA

    /* Trailing blank line to separate sections */
    printf("\n");
}

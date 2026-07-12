/**
 * @file cli_colors.c
 * @brief Implementations of shared ANSI color management and CLI styling helper functions.
 */

#include "shared/cli_colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

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
        memcpy(buf, line, len);
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
        memcpy(buf, line, len);
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
}

int cli_is_color_enabled(
    void)
{
    return (ansi_color_reset != NULL && strlen(ansi_color_reset) > 0);
}

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
}

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
}

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
}

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
}

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
}

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

static int is_verbatim_line(
    const char *content,
    int         clen)
{
    if (clen <= 0)
    {
        return 0;
    }

    if (content[0] == '-' && clen > 1 && content[1] == ' ')
    {
        return 1;
    }

    if (content[0] >= '0' && content[0] <= '9')
    {
        for (int i = 1; i < clen; i++)
        {
            if (content[i] == '.' && i + 1 < clen && content[i + 1] == ' ')
            {
                return 1;
            }
            if (content[i] < '0' || content[i] > '9')
            {
                break;
            }
        }
    }

    if (content[0] == '|')
    {
        return 1;
    }

    for (int i = 0; i < clen; i++)
    {
        if (content[i] == '-' && i + 1 < clen && (content[i + 1] == '>' || content[i + 1] == '+'))
        {
            return 1;
        }
        if (content[i] == '<' && i + 1 < clen && content[i + 1] == '-')
        {
            return 1;
        }
        if (content[i] == '-' && i + 2 < clen && content[i + 1] == '-' && content[i + 2] == '-')
        {
            return 1;
        }
    }
    return 0;
}

void cli_print_help_section(
    const char *label,
    const char *value)
{
    int width = cli_get_terminal_width();

    printf("%s%s%s\n", ANSI_BOLD_CYAN, label, ANSI_COLOR_RESET);

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

    char para[4096];
    int plen = 0;
    int para_indent = 0;

    #define FLUSH_PARA() do {                          \
        if (plen > 0)                                  \
        {                                              \
            cli_print_wrapped_line(                    \
                para, plen,                            \
                2 + para_indent, width);               \
            plen = 0;                                  \
        }                                              \
    } while (0)

    for (int li = 0; li < nlines; li++)
    {
        const char *line = lines[li];
        int ll = lens[li];

        if (ll == 0)
        {
            FLUSH_PARA();
            putchar('\n');
            continue;
        }

        int indent = count_indent(line, ll);
        const char *content = line + indent;
        int clen = ll - indent;

        if (indent == 0 && !(plen > 0 && para_indent == 0))
        {
            int skip_subheader = 0;
            if (is_verbatim_line(content, clen))
            {
                skip_subheader = 1;
                if (content[0] >= '0' && content[0] <= '9')
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
                if (li + 1 < nlines && lens[li + 1] > 0)
                {
                    int next_indent = count_indent(lines[li + 1], lens[li + 1]);
                    if (next_indent > indent)
                    {
                        next_indented = 1;
                    }
                }
                if (next_indented)
                {
                    FLUSH_PARA();
                    printf("  %s", ANSI_BOLD);
                    cli_print_rich_segment(line, ll, 1);
                    printf("%s\n", ANSI_COLOR_RESET);
                    continue;
                }
            }
        }

        if (is_verbatim_line(content, clen))
        {
            FLUSH_PARA();
            printf("  ");
            cli_print_rich_segment(line, ll, 0);
            putchar('\n');
            continue;
        }

        if (plen > 0 && indent != para_indent)
        {
            FLUSH_PARA();
        }
        para_indent = indent;

        if (plen > 0 && plen + 1 + clen < (int)sizeof(para) - 1)
        {
            para[plen++] = ' ';
        }
        if (plen + clen < (int)sizeof(para) - 1)
        {
            memcpy(para + plen, content, clen);
            plen += clen;
        }
    }

    FLUSH_PARA();
    #undef FLUSH_PARA

    printf("\n");
}

void cli_print_header_box(
    const char *title)
{
    int utf8 = 0;
    const char *lang = getenv("LANG");
    const char *lc_all = getenv("LC_ALL");
    if ((lang != NULL && (strstr(lang, "UTF-8") != NULL || strstr(lang, "utf-8") != NULL)) ||
        (lc_all != NULL && (strstr(lc_all, "UTF-8") != NULL || strstr(lc_all, "utf-8") != NULL)))
    {
        utf8 = 1;
    }

    int len = (int)strlen(title);
    int width = len + 4;

    if (utf8)
    {
        printf("%s┌", ANSI_BOLD_CYAN);
        for (int i = 0; i < width - 2; i++)
        {
            printf("─");
        }
        printf("┐%s\n", ANSI_COLOR_RESET);

        printf("%s│%s  %s%s%s  %s│%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET,
               ANSI_BOLD, title, ANSI_COLOR_RESET,
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);

        printf("%s└", ANSI_BOLD_CYAN);
        for (int i = 0; i < width - 2; i++)
        {
            printf("─");
        }
        printf("┘%s\n", ANSI_COLOR_RESET);
    }
    else
    {
        printf("%s+", ANSI_BOLD_CYAN);
        for (int i = 0; i < width - 2; i++)
        {
            printf("-");
        }
        printf("+%s\n", ANSI_COLOR_RESET);

        printf("%s|%s  %s%s%s  %s|%s\n",
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET,
               ANSI_BOLD, title, ANSI_COLOR_RESET,
               ANSI_BOLD_CYAN, ANSI_COLOR_RESET);

        printf("%s+", ANSI_BOLD_CYAN);
        for (int i = 0; i < width - 2; i++)
        {
            printf("-");
        }
        printf("+%s\n", ANSI_COLOR_RESET);
    }
}

static int get_levenshtein_distance(
    const char *s1,
    const char *s2)
{
    int len1 = (int)strlen(s1);
    int len2 = (int)strlen(s2);

    int *d = malloc((size_t)((len1 + 1) * (len2 + 1)) * sizeof(int));
    if (d == NULL)
    {
        return 999;
    }

    for (int i = 0; i <= len1; i++)
    {
        d[i * (len2 + 1)] = i;
    }
    for (int j = 0; j <= len2; j++)
    {
        d[j] = j;
    }

    for (int i = 1; i <= len1; i++)
    {
        for (int j = 1; j <= len2; j++)
        {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            int m_inst = d[(i - 1) * (len2 + 1) + j] + 1;
            int m_del = d[i * (len2 + 1) + (j - 1)] + 1;
            int m_sub = d[(i - 1) * (len2 + 1) + (j - 1)] + cost;

            int min = m_inst;
            if (m_del < min)
            {
                min = m_del;
            }
            if (m_sub < min)
            {
                min = m_sub;
            }
            d[i * (len2 + 1) + j] = min;
        }
    }

    int res = d[len1 * (len2 + 1) + len2];
    free(d);
    return res;
}

void cli_suggest_similar_topic(
    const char         *topic,
    const char *const  *topics,
    int                 ntopics)
{
    int best_dist = 999;
    const char *best_suggestion = NULL;

    for (int i = 0; i < ntopics; i++)
    {
        if (strstr(topics[i], topic) != NULL)
        {
            best_dist = 0;
            best_suggestion = topics[i];
            break;
        }

        int dist = get_levenshtein_distance(topic, topics[i]);
        if (dist < best_dist)
        {
            best_dist = dist;
            best_suggestion = topics[i];
        }
    }

    if (best_suggestion != NULL && best_dist <= 4)
    {
        fprintf(stderr, "\nDid you mean?\n  %s%s%s\n",
                ANSI_BOLD_GREEN, best_suggestion, ANSI_COLOR_RESET);
    }
}


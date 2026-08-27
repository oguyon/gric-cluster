/**
 * @file cli_help_view.c
 * @brief Implementation of formatted CLI help, options styling, headers, and topic suggestions.
 */

#include "shared/cli_help_view.h"
#include "shared/cli_theme.h"
#include "shared/cli_terminal.h"
#include "shared/cli_wrap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} // cli_print_colored_usage

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
} // cli_print_colored_line

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
} // cli_print_colored_options

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

        const char *p = buf;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }

        if (*p == '$')
        {
            printf("  %s$%s", ANSI_COLOR_GREY, ANSI_COLOR_RESET);
            p++;
            while (*p == ' ' || *p == '\t')
            {
                putchar(*p);
                p++;
            }

            const char *cmd_start = p;
            while (*p && *p != ' ' && *p != '\t')
            {
                p++;
            }
            printf("%s%.*s%s",
                   ANSI_BOLD_GREEN,
                   (int)(p - cmd_start), cmd_start,
                   ANSI_COLOR_RESET);

            while (*p)
            {
                if (*p == '-' && *(p + 1) != ' ')
                {
                    const char *f_end = p + 1;
                    while (*f_end && *f_end != ' ' && *f_end != '\t')
                    {
                        f_end++;
                    }
                    printf("%s%.*s%s",
                           ANSI_COLOR_GREEN,
                           (int)(f_end - p), p,
                           ANSI_COLOR_RESET);
                    p = f_end;
                }
                else
                {
                    putchar(*p);
                    p++;
                }
            }
            printf("\n");
        }
        else if (*p == '#')
        {
            printf("%s%s%s\n", ANSI_COLOR_GREY, buf, ANSI_COLOR_RESET);
        }
        else
        {
            printf("%s\n", buf);
        }

        line = next ? next + 1 : NULL;
    }
} // cli_print_colored_examples

void cli_print_see_also_option(
    const char *option,
    const char *desc)
{
    printf("  %s%-24s%s %s\n",
           ANSI_BOLD_GREEN, option,
           ANSI_COLOR_RESET, desc);
} // cli_print_see_also_option

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
} // cli_print_help_section

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
} // cli_print_header_box

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
} // cli_suggest_similar_topic

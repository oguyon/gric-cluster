/**
 * @file cluster_help_content.c
 * @brief Help content database dispatcher using embedded Markdown topics.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_help_content.h"
#include "help_md_render.h"
#include <stddef.h>

int print_keyword_content(
    const char *key)
{
    if (key == NULL)
    {
        return 0;
    }

    const char *md_content = help_topic_lookup(key);
    if (md_content == NULL)
    {
        return 0;
    }

    render_markdown_help(md_content);
    return 1;
}

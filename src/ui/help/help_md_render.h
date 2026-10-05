/**
 * @file help_md_render.h
 * @brief Terminal ANSI Markdown parser and renderer declarations.
 *
 * Provides functions to parse Markdown strings and render formatted,
 * syntax-colored, and word-wrapped help screens on the terminal.
 */

#ifndef HELP_MD_RENDER_H
#define HELP_MD_RENDER_H

/**
 * @brief Look up an embedded help topic by keyword or alias.
 * @param keyword The keyword or option name to search for.
 * @return Pointer to raw Markdown string, or NULL if not found.
 */
const char *help_topic_lookup(
    const char *keyword);

/**
 * @brief Render a raw Markdown string to stdout with ANSI formatting and wrapping.
 * @param md_text The raw Markdown string content.
 */
void render_markdown_help(
    const char *md_text);

#endif // HELP_MD_RENDER_H

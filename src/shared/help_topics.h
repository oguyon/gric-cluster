/**
 * @file help_topics.h
 * @brief Header declaring compiled embedded help topics database interface.
 */

#ifndef HELP_TOPICS_H
#define HELP_TOPICS_H

#include <stddef.h>

struct HelpTopicEntry
{
    const char *keyword;
    const char *content;
};

/**
 * help_topic_lookup() - Retrieve markdown content for given keyword.
 * @keyword: Topic keyword or option name.
 *
 * Return: Pointer to markdown string literal, or NULL if not found.
 */
const char *help_topic_lookup(
    const char *keyword);

/**
 * help_topic_get_table() - Retrieve the NULL-terminated help topic table.
 *
 * Return: Pointer to HelpTopicEntry array terminated by {NULL, NULL}.
 */
const struct HelpTopicEntry *help_topic_get_table(void);

/**
 * help_topic_get_count() - Count total number of available topics and aliases.
 *
 * Return: Total number of valid entries.
 */
size_t help_topic_get_count(void);

#endif // HELP_TOPICS_H

/**
 * @file cluster_help_content.h
 * @brief Declaration of the help content database lookup.
 */

#ifndef CLUSTER_HELP_CONTENT_H
#define CLUSTER_HELP_CONTENT_H

/**
 * @brief Print the detailed help content for a given keyword.
 * @param key The normalized option or topic keyword.
 * @return 1 if matched and printed, 0 if not found.
 */
int print_keyword_content(
    const char *key);

#endif // CLUSTER_HELP_CONTENT_H

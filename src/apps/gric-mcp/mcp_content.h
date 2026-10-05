/**
 * @file mcp_content.h
 * @brief Auto-generated recipes and suite manifest data for gric-mcp.
 */

#ifndef MCP_CONTENT_H
#define MCP_CONTENT_H

#include <stddef.h>

struct mcp_recipe
{
    const char *name;
    const char *title;
    const char *body;
};

struct mcp_suite_program
{
    const char *name;
    const char *category;
    const char *summary;
    const char *usage;
    const char *requires;
    const char *alias_of;
};

const struct mcp_recipe        *mcp_recipe_table(size_t *count);

const struct mcp_recipe        *mcp_recipe_find(const char *name);

const struct mcp_suite_program *mcp_suite_program_table(size_t *count);

#endif /* MCP_CONTENT_H */

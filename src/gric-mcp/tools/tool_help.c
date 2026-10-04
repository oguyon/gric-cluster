/**
 * @file tool_help.c
 * @brief MCP knowledge engine tools for documentation, suite catalog, and recipes.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_content.h"
#include "shared/help_topics.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * to_lowercase() - Convert string to lowercase in place.
 * @str: String buffer to convert.
 */
static void to_lowercase(
    char *str)
{
    for (size_t ii = 0; str[ii] != '\0'; ii++)
    {
        str[ii] = (char)tolower((unsigned char)str[ii]);
    }
} // to_lowercase

int mcp_tool_help(
    const cJSON *args,
    cJSON       *res)
{
    const char *topic_str = NULL;
    const char *query_str = NULL;

    if (args != NULL)
    {
        cJSON *t_item = cJSON_GetObjectItemCaseSensitive(args, "topic");
        if (t_item != NULL && cJSON_IsString(t_item))
        {
            topic_str = t_item->valuestring;
        }

        cJSON *q_item = cJSON_GetObjectItemCaseSensitive(args, "query");
        if (q_item != NULL && cJSON_IsString(q_item))
        {
            query_str = q_item->valuestring;
        }
    }

    /* 1. Direct Topic Lookup */
    if (topic_str != NULL && strlen(topic_str) > 0 &&
        strcmp(topic_str, "list") != 0 && strcmp(topic_str, "all") != 0)
    {
        char norm_topic[128];
        strncpy(norm_topic, topic_str, sizeof(norm_topic) - 1);
        norm_topic[sizeof(norm_topic) - 1] = '\0';
        to_lowercase(norm_topic);

        /* Normalize leading dashes if user queried -rlim or --entropy */
        char *search_key = norm_topic;
        while (*search_key == '-')
        {
            search_key++;
        }

        const char *content = help_topic_lookup(search_key);
        if (content != NULL)
        {
            cJSON_AddStringToObject(res, "status", "FOUND");
            cJSON_AddStringToObject(res, "topic", search_key);
            cJSON_AddStringToObject(res, "content", content);
            return 0;
        }

        /* If exact lookup failed, fall back to searching for candidate matches */
        query_str = search_key;
    } // if topic_str

    /* 2. Search / Query / List All */
    const struct HelpTopicEntry *table = help_topic_get_table();
    if (table == NULL)
    {
        cJSON_AddStringToObject(res, "status", "ERROR");
        cJSON_AddStringToObject(res, "message", "Help database unavailable");
        return -1;
    }

    if (query_str != NULL && strlen(query_str) > 0)
    {
        char norm_q[128];
        strncpy(norm_q, query_str, sizeof(norm_q) - 1);
        norm_q[sizeof(norm_q) - 1] = '\0';
        to_lowercase(norm_q);

        cJSON *matches = cJSON_CreateArray();
        int match_count = 0;

        for (size_t ii = 0; table[ii].keyword != NULL; ii++)
        {
            char norm_k[128];
            strncpy(norm_k, table[ii].keyword, sizeof(norm_k) - 1);
            norm_k[sizeof(norm_k) - 1] = '\0';
            to_lowercase(norm_k);

            /* Check keyword substring or content match */
            if (strstr(norm_k, norm_q) != NULL ||
                (table[ii].content != NULL && strcasestr(table[ii].content, norm_q) != NULL))
            {
                cJSON *item = cJSON_CreateObject();
                cJSON_AddStringToObject(item, "topic", table[ii].keyword);

                /* Create concise preview line */
                char preview[160] = "";
                if (table[ii].content != NULL)
                {
                    const char *p = table[ii].content;
                    const char *nl = strchr(p, '\n');
                    if (nl != NULL && (size_t)(nl - p) < sizeof(preview))
                    {
                        strncpy(preview, p, (size_t)(nl - p));
                        preview[nl - p] = '\0';
                    }
                    else
                    {
                        strncpy(preview, p, sizeof(preview) - 1);
                        preview[sizeof(preview) - 1] = '\0';
                    }
                }
                cJSON_AddStringToObject(item, "preview", preview);
                cJSON_AddItemToArray(matches, item);
                match_count++;

                if (match_count >= 20)
                {
                    break;
                }
            } // if match
        } // for ii

        cJSON_AddStringToObject(res, "status", (match_count > 0) ? "MATCHES_FOUND" : "NOT_FOUND");
        cJSON_AddStringToObject(res, "query", query_str);
        cJSON_AddNumberToObject(res, "match_count", match_count);
        cJSON_AddItemToObject(res, "matches", matches);
        return 0;
    } // if query_str

    /* 3. Catalog of all available topics */
    cJSON *topics_arr = cJSON_CreateArray();
    size_t total = 0;

    for (size_t ii = 0; table[ii].keyword != NULL; ii++)
    {
        cJSON_AddItemToArray(topics_arr, cJSON_CreateString(table[ii].keyword));
        total++;
    }

    cJSON_AddStringToObject(res, "status", "CATALOG");
    cJSON_AddNumberToObject(res, "total_topics", (double)total);
    cJSON_AddItemToObject(res, "available_topics", topics_arr);
    return 0;
} // mcp_tool_help

int mcp_tool_list_suite(
    const cJSON *args,
    cJSON       *res)
{
    (void)args;
    size_t nprogs = 0;
    const struct mcp_suite_program *progs = mcp_suite_program_table(&nprogs);
    cJSON *progs_arr = cJSON_CreateArray();

    for (size_t ii = 0; ii < nprogs; ii++)
    {
        cJSON *prog = cJSON_CreateObject();
        cJSON_AddStringToObject(prog, "name", progs[ii].name);
        cJSON_AddStringToObject(prog, "category", progs[ii].category);
        cJSON_AddStringToObject(prog, "summary", progs[ii].summary);
        cJSON_AddStringToObject(prog, "usage", progs[ii].usage);
        if (progs[ii].requires != NULL)
        {
            cJSON_AddStringToObject(prog, "requires", progs[ii].requires);
        }
        if (progs[ii].alias_of != NULL)
        {
            cJSON_AddStringToObject(prog, "alias_of", progs[ii].alias_of);
        }
        cJSON_AddItemToArray(progs_arr, prog);
    } // for ii

    cJSON_AddStringToObject(res, "status", "SUCCESS");
    cJSON_AddNumberToObject(res, "total_programs", (double)nprogs);
    cJSON_AddItemToObject(res, "programs", progs_arr);
    return 0;
} // mcp_tool_list_suite

int mcp_tool_get_recipe(
    const cJSON *args,
    cJSON       *res)
{
    const char *recipe_name = NULL;
    if (args != NULL)
    {
        cJSON *r_item = cJSON_GetObjectItemCaseSensitive(args, "recipe");
        if (r_item != NULL && cJSON_IsString(r_item))
        {
            recipe_name = r_item->valuestring;
        }
    }

    size_t nrecipes = 0;
    const struct mcp_recipe *recipes = mcp_recipe_table(&nrecipes);

    if (recipe_name == NULL || strlen(recipe_name) == 0 || strcmp(recipe_name, "list") == 0)
    {
        cJSON *list = cJSON_CreateArray();
        for (size_t ii = 0; ii < nrecipes; ii++)
        {
            cJSON_AddItemToArray(list, cJSON_CreateString(recipes[ii].name));
        }

        cJSON_AddStringToObject(res, "status", "RECIPES_INDEX");
        cJSON_AddItemToObject(res, "available_recipes", list);
        return 0;
    }

    const struct mcp_recipe *rec = mcp_recipe_find(recipe_name);
    if (rec != NULL)
    {
        cJSON_AddStringToObject(res, "recipe", rec->name);
        cJSON_AddStringToObject(res, "title", rec->title);
        cJSON_AddStringToObject(res, "instructions", rec->body);
        return 0;
    }

    cJSON_AddStringToObject(res, "status", "UNKNOWN_RECIPE");
    cJSON_AddStringToObject(res, "requested", recipe_name);
    return -1;
} // mcp_tool_get_recipe

int mcp_tool_server_info(
    const cJSON *args,
    cJSON       *res)
{
    (void)args;
    const struct mcp_server_config *cfg = mcp_registry_get_config();

    cJSON_AddStringToObject(res, "name", "gric-mcp");
    cJSON_AddStringToObject(res, "version", "1.0.0");
    cJSON_AddBoolToObject(res, "read_only", cfg->read_only);
    cJSON_AddBoolToObject(res, "has_source_tree", cfg->has_source_tree);

    cJSON *ts_array = cJSON_CreateArray();
    if (cfg->toolsets & MCP_TS_KNOWLEDGE)
    {
        cJSON_AddItemToArray(ts_array, cJSON_CreateString("knowledge"));
    }
    if (cfg->toolsets & MCP_TS_ANALYSIS)
    {
        cJSON_AddItemToArray(ts_array, cJSON_CreateString("analysis"));
    }
    if (cfg->toolsets & MCP_TS_RUN)
    {
        cJSON_AddItemToArray(ts_array, cJSON_CreateString("run"));
    }
    if (cfg->toolsets & MCP_TS_OPS)
    {
        cJSON_AddItemToArray(ts_array, cJSON_CreateString("ops"));
    }
    if (cfg->toolsets & MCP_TS_DEV)
    {
        cJSON_AddItemToArray(ts_array, cJSON_CreateString("dev"));
    }
    cJSON_AddItemToObject(res, "toolsets", ts_array);

    return 0;
} // mcp_tool_server_info

const struct mcp_tool_def mcp_tooldef_help = {
    .name         = "gric_help",
    .toolset      = MCP_TS_KNOWLEDGE,
    .side_effects = 0,
    .fn           = mcp_tool_help,
    .description  = "Query GRIC embedded help database (95+ topics) by keyword or fuzzy "
                    "search: algorithms, parameters (rlim, entropy, tiling, milk), flags.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"topic\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Topic keyword or flag name (e.g. 'entropy', 'rlim').\"\n"
        "    },\n"
        "    \"query\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Optional search query to match in documentation.\"\n"
        "    }\n"
        "  }\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_list_suite = {
    .name         = "gric_list_suite",
    .toolset      = MCP_TS_KNOWLEDGE,
    .side_effects = 0,
    .fn           = mcp_tool_list_suite,
    .description  = "List all 18+ programs in the GRIC suite with categories, summaries, "
                    "and CLI usage prototypes.",
    .input_schema = "{\"type\": \"object\"}",
};

const struct mcp_tool_def mcp_tooldef_get_recipe = {
    .name         = "gric_get_recipe",
    .toolset      = MCP_TS_KNOWLEDGE,
    .side_effects = 0,
    .fn           = mcp_tool_get_recipe,
    .description  = "Retrieve step-by-step cookbook instructions for common tasks (e.g. "
                    "'milk_realtime_streaming', 'image_cube_clustering').",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"recipe\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Name of recipe, or 'list' to see available recipes.\"\n"
        "    }\n"
        "  }\n"
        "}",
};

const struct mcp_tool_def mcp_tooldef_server_info = {
    .name         = "gric_server_info",
    .toolset      = MCP_TS_KNOWLEDGE,
    .side_effects = 0,
    .fn           = mcp_tool_server_info,
    .description  = "Retrieve runtime status, version, active toolsets, and read-only mode "
                    "of the gric-mcp server.",
    .input_schema = "{\"type\": \"object\"}",
};

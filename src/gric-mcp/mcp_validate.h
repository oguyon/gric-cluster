/**
 * @file mcp_validate.h
 * @brief Input validation and strict parsing primitives for gric-mcp.
 */

#ifndef MCP_VALIDATE_H
#define MCP_VALIDATE_H

#include "shared/cjson/cJSON.h"
#include <stddef.h>
#include <stdint.h>

/**
 * mcp_valid_identifier() - Validate string consists of [A-Za-z0-9_.-] and is not empty.
 * @s:       String to validate.
 * @max_len: Maximum permissible length.
 *
 * Return: 1 if valid identifier, 0 otherwise.
 */
int mcp_valid_identifier(
    const char *s,
    size_t      max_len);

/**
 * mcp_valid_path() - Validate filesystem path contains no control characters.
 * @s:       Path to validate.
 * @max_len: Maximum permissible length.
 *
 * Return: 1 if valid path, 0 otherwise.
 */
int mcp_valid_path(
    const char *s,
    size_t      max_len);

/**
 * mcp_parse_double_strict() - Parse floating point number consuming full string.
 * @s:   Input string.
 * @out: Pointer to store parsed double.
 *
 * Return: 0 on success, -1 on invalid or unconsumed characters.
 */
int mcp_parse_double_strict(
    const char *s,
    double     *out);

/**
 * mcp_parse_int64_strict() - Parse 64-bit integer consuming full string.
 * @s:   Input string.
 * @out: Pointer to store parsed int64_t.
 *
 * Return: 0 on success, -1 on invalid or unconsumed characters.
 */
int mcp_parse_int64_strict(
    const char *s,
    int64_t    *out);

/**
 * mcp_parse_onoff() - Parse boolean/switch value from JSON element into 0 or 1.
 * @v:   cJSON item (bool, number, or string like "ON"/"OFF", "true"/"false", "1"/"0").
 * @out: Pointer to store 0 or 1.
 *
 * Return: 0 on success, -1 if unrecognized.
 */
int mcp_parse_onoff(
    const cJSON *v,
    int         *out);

#endif // MCP_VALIDATE_H

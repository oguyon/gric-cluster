/**
 * @file mcp_test_util.h
 * @brief Common test verification macros for MCP tests.
 */

#ifndef MCP_TEST_UTIL_H
#define MCP_TEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1); \
        } \
    } while (0)

#endif // MCP_TEST_UTIL_H

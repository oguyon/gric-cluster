/**
 * @file mcp_staleness.h
 * @brief Stale-server detection for gric-mcp daemon.
 */

#ifndef MCP_STALENESS_H
#define MCP_STALENESS_H

#include <stddef.h>

struct mcp_stale_status
{
    int         stale;
    const char *reason;         /* "binary_rebuilt" | "head_moved" | NULL */
    char        repo_head[41];
};

/**
 * mcp_staleness_check() - Check if running server binary or repo HEAD is stale.
 *
 * Return: Pointer to cached stale status (5 second TTL).
 */
const struct mcp_stale_status *mcp_staleness_check(void);

/**
 * mcp_git_resolve_head() - Resolve 40-character commit SHA for git repository.
 * @repo_root: Root directory of repository containing .git (dir or file).
 * @out_sha:   Destination buffer for 40-char SHA plus null terminator.
 * @out_size:  Size of destination buffer (must be at least 41 bytes).
 *
 * Return: 0 on success, -1 on failure.
 */
int mcp_git_resolve_head(
    const char *repo_root,
    char       *out_sha,
    size_t      out_size);

/* Test hooks */
void mcp_staleness_set_exe_path_override(
    const char *path);

void mcp_staleness_set_root_override(
    const char *path);

void mcp_staleness_reset(void);

#endif /* MCP_STALENESS_H */

/**
 * @file mcp_staleness.c
 * @brief Implementation of stale-server detection for gric-mcp daemon.
 */

#include "mcp_staleness.h"
#include "mcp_registry.h"
#include "mcp_tools.h"
#include "gric_build_info.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static struct mcp_stale_status s_status = {0, NULL, ""};
static time_t s_last_check = 0;
static const char *s_exe_override = NULL;
static const char *s_root_override = NULL;

static int is_valid_sha1(
    const char *s)
{
    if (s == NULL || strlen(s) < 40)
    {
        return 0;
    }
    for (int ii = 0; ii < 40; ii++)
    {
        char c = s[ii];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F')))
        {
            return 0;
        }
    }
    return 1;
}

static void trim_trailing_whitespace(
    char *s)
{
    if (s == NULL)
    {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\r' || s[len - 1] == '\n' ||
                       s[len - 1] == ' '  || s[len - 1] == '\t'))
    {
        s[--len] = '\0';
    }
}

static int check_binary_rebuilt(void)
{
    char buf[1024];
    ssize_t len = -1;

    if (s_exe_override != NULL)
    {
        strncpy(buf, s_exe_override, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        len = (ssize_t)strlen(buf);
    }
    else
    {
        len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0)
        {
            buf[len] = '\0';
        }
    }

    if (len > 0)
    {
        const char *deleted_suffix = " (deleted)";
        size_t suffix_len = strlen(deleted_suffix);
        if ((size_t)len >= suffix_len &&
            strcmp(buf + len - suffix_len, deleted_suffix) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int mcp_git_resolve_head(
    const char *repo_root,
    char       *out_sha,
    size_t      out_size)
{
    if (repo_root == NULL || out_sha == NULL || out_size < 41)
    {
        return -1;
    }

    char dot_git[1024];
    snprintf(dot_git, sizeof(dot_git), "%s/.git", repo_root);

    struct stat st;
    if (stat(dot_git, &st) != 0)
    {
        return -1;
    }

    char git_dir[1024];
    if (S_ISDIR(st.st_mode))
    {
        snprintf(git_dir, sizeof(git_dir), "%s", dot_git);
    }
    else if (S_ISREG(st.st_mode))
    {
        FILE *fp = fopen(dot_git, "r");
        if (fp == NULL)
        {
            return -1;
        }
        char line[1024];
        if (fgets(line, sizeof(line), fp) == NULL)
        {
            fclose(fp);
            return -1;
        }
        fclose(fp);

        trim_trailing_whitespace(line);
        if (strncmp(line, "gitdir:", 7) != 0)
        {
            return -1;
        }
        const char *p = line + 7;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '/')
        {
            snprintf(git_dir, sizeof(git_dir), "%s", p);
        }
        else
        {
            snprintf(git_dir, sizeof(git_dir), "%s/%s", repo_root, p);
        }
    }
    else
    {
        return -1;
    }

    char head_file[2048];
    snprintf(head_file, sizeof(head_file), "%s/HEAD", git_dir);

    FILE *fp = fopen(head_file, "r");
    if (fp == NULL)
    {
        return -1;
    }

    char head_line[1024];
    if (fgets(head_line, sizeof(head_line), fp) == NULL)
    {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    trim_trailing_whitespace(head_line);

    if (strncmp(head_line, "ref:", 4) == 0)
    {
        const char *ref_name = head_line + 4;
        while (*ref_name == ' ' || *ref_name == '\t')
        {
            ref_name++;
        }

        /* 1. Check loose ref */
        char ref_file[2048];
        snprintf(ref_file, sizeof(ref_file), "%s/%s", git_dir, ref_name);
        FILE *rfp = fopen(ref_file, "r");
        if (rfp != NULL)
        {
            char sha_buf[128];
            if (fgets(sha_buf, sizeof(sha_buf), rfp) != NULL)
            {
                trim_trailing_whitespace(sha_buf);
                if (is_valid_sha1(sha_buf))
                {
                    snprintf(out_sha, out_size, "%.40s", sha_buf);
                    fclose(rfp);
                    return 0;
                }
            }
            fclose(rfp);
        }

        /* 2. Check packed-refs */
        char packed_file[2048];
        snprintf(packed_file, sizeof(packed_file), "%s/packed-refs", git_dir);
        FILE *pfp = fopen(packed_file, "r");
        if (pfp != NULL)
        {
            char pline[1024];
            while (fgets(pline, sizeof(pline), pfp) != NULL)
            {
                trim_trailing_whitespace(pline);
                if (pline[0] == '#' || pline[0] == '^')
                {
                    continue;
                }
                char *space = strchr(pline, ' ');
                if (space != NULL)
                {
                    *space = '\0';
                    const char *target_ref = space + 1;
                    if (strcmp(target_ref, ref_name) == 0 && is_valid_sha1(pline))
                    {
                        snprintf(out_sha, out_size, "%.40s", pline);
                        fclose(pfp);
                        return 0;
                    }
                }
            } // while fgets
            fclose(pfp);
        }
        return -1;
    }
    else
    {
        /* Detached HEAD */
        if (is_valid_sha1(head_line))
        {
            snprintf(out_sha, out_size, "%.40s", head_line);
            return 0;
        }
        return -1;
    }
} // mcp_git_resolve_head

void mcp_staleness_set_exe_path_override(
    const char *path)
{
    s_exe_override = path;
    s_last_check = 0;
}

void mcp_staleness_set_root_override(
    const char *path)
{
    s_root_override = path;
    s_last_check = 0;
}

void mcp_staleness_reset(void)
{
    s_status.stale = 0;
    s_status.reason = NULL;
    s_status.repo_head[0] = '\0';
    s_exe_override = NULL;
    s_root_override = NULL;
    s_last_check = 0;
}

const struct mcp_stale_status *mcp_staleness_check(void)
{
    time_t now = time(NULL);
    if (s_last_check != 0 && (now - s_last_check) < 5)
    {
        return &s_status;
    }
    s_last_check = now;
    s_status.stale = 0;
    s_status.reason = NULL;

    /* Signal 1: Binary rebuilt */
    if (check_binary_rebuilt())
    {
        s_status.stale = 1;
        s_status.reason = "binary_rebuilt";
        return &s_status;
    }

    /* Signal 2: HEAD moved */
    char root[1024];
    if (s_root_override != NULL)
    {
        snprintf(root, sizeof(root), "%s", s_root_override);
    }
    else
    {
        const struct mcp_server_config *cfg = mcp_registry_get_config();
        if (cfg != NULL && !cfg->has_source_tree)
        {
            return &s_status;
        }
        mcp_get_project_root(root, sizeof(root));
        if (root[0] == '\0')
        {
            return &s_status;
        }
    }

    char head_sha[41];
    if (mcp_git_resolve_head(root, head_sha, sizeof(head_sha)) == 0)
    {
        snprintf(s_status.repo_head, sizeof(s_status.repo_head), "%s", head_sha);
        if (strcmp(GRIC_GIT_HEAD, "unknown") != 0 &&
            strcmp(head_sha, GRIC_GIT_HEAD) != 0)
        {
            s_status.stale = 1;
            s_status.reason = "head_moved";
            return &s_status;
        }
    }

    return &s_status;
} // mcp_staleness_check

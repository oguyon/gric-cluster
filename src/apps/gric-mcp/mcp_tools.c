/**
 * @file mcp_tools.c
 * @brief Path resolution and tools list delegation for GRIC MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

cJSON *mcp_tools_get_list(void)
{
    return mcp_registry_tools_list();
} // mcp_tools_get_list

/**
 * is_project_root() - Check if a candidate directory is the gric repository root.
 * @dir: Directory path string.
 *
 * Return: 1 if dir contains src/gric-mcp/gric-mcp.c, 0 otherwise.
 */
static int is_project_root(
    const char *dir)
{
    if (dir == NULL || dir[0] == '\0')
    {
        return 0;
    }
    char check[2048];
    snprintf(check, sizeof(check), "%s/src/gric-mcp/gric-mcp.c", dir);
    return (access(check, R_OK) == 0) ? 1 : 0;
} // is_project_root

/**
 * get_current_executable() - Discover current process executable path portably.
 * @out_path: Target buffer for executable path.
 * @out_size: Buffer capacity.
 */
static void get_current_executable(
    char   *out_path,
    size_t  out_size)
{
    out_path[0] = '\0';
#if defined(__APPLE__)
    char temp[1024];
    uint32_t bsize = (uint32_t)sizeof(temp);
    if (_NSGetExecutablePath(temp, &bsize) == 0)
    {
        char resolved[1024];
        if (realpath(temp, resolved) != NULL)
        {
            strncpy(out_path, resolved, out_size - 1);
            out_path[out_size - 1] = '\0';
        }
        else
        {
            strncpy(out_path, temp, out_size - 1);
            out_path[out_size - 1] = '\0';
        }
    }
#elif defined(__linux__)
    ssize_t len = readlink("/proc/self/exe", out_path, out_size - 1);
    if (len > 0)
    {
        out_path[len] = '\0';
    }
#endif
} // get_current_executable

/**
 * mcp_get_project_root() - Retrieve the repository root directory.
 * @buf:  Target buffer to store the root directory path.
 * @size: Buffer capacity.
 */
void mcp_get_project_root(
    char   *buf,
    size_t  size)
{
    const char *env_root = getenv("GRIC_PROJECT_ROOT");
    if (env_root != NULL && env_root[0] != '\0')
    {
        strncpy(buf, env_root, size - 1);
        buf[size - 1] = '\0';
        return;
    }

    /* 1. Walk up from current executable path if known */
    char exe_path[1024];
    get_current_executable(exe_path, sizeof(exe_path));
    if (exe_path[0] != '\0')
    {
        char curr[1024];
        strncpy(curr, exe_path, sizeof(curr) - 1);
        curr[sizeof(curr) - 1] = '\0';
        char *last_slash = strrchr(curr, '/');
        if (last_slash != NULL)
        {
            *last_slash = '\0';
        }
        while (curr[0] != '\0')
        {
            if (is_project_root(curr))
            {
                strncpy(buf, curr, size - 1);
                buf[size - 1] = '\0';
                return;
            }
            last_slash = strrchr(curr, '/');
            if (last_slash == NULL || last_slash == curr)
            {
                break;
            }
            *last_slash = '\0';
        }
    }

    /* 2. Walk up from current working directory */
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        while (cwd[0] != '\0')
        {
            if (is_project_root(cwd))
            {
                strncpy(buf, cwd, size - 1);
                buf[size - 1] = '\0';
                return;
            }
            char *last_slash = strrchr(cwd, '/');
            if (last_slash == NULL || last_slash == cwd)
            {
                break;
            }
            *last_slash = '\0';
        }
    }

    /* 3. Check relative fallback locations */
    if (is_project_root("."))
    {
        strncpy(buf, ".", size - 1);
        buf[size - 1] = '\0';
        return;
    }
    if (is_project_root(".."))
    {
        strncpy(buf, "..", size - 1);
        buf[size - 1] = '\0';
        return;
    }

    strncpy(buf, ".", size - 1);
    buf[size - 1] = '\0';
} // mcp_get_project_root

/**
 * mcp_resolve_path() - Resolve a relative or absolute file path within the project.
 * @input_path: Input path string (may be relative to project root or CWD).
 * @out_buf:    Target buffer for resolved path.
 * @out_size:   Buffer capacity.
 */
void mcp_resolve_path(
    const char *input_path,
    char       *out_buf,
    size_t      out_size)
{
    if (input_path == NULL || input_path[0] == '\0')
    {
        out_buf[0] = '\0';
        return;
    }

    /* Check if already accessible directly */
    if (access(input_path, R_OK) == 0)
    {
        strncpy(out_buf, input_path, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* If relative, try prepending project root */
    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char candidate[2048];
    snprintf(candidate, sizeof(candidate), "%s/%s", root, input_path);
    if (access(candidate, R_OK) == 0)
    {
        strncpy(out_buf, candidate, out_size - 1);
        out_buf[out_size - 1] = '\0';
        return;
    }

    /* Fallback to original */
    strncpy(out_buf, input_path, out_size - 1);
    out_buf[out_size - 1] = '\0';
} // mcp_resolve_path

/**
 * mcp_get_build_dir() - Retrieve the active build directory path.
 * @buf:  Target buffer to store the build directory path.
 * @size: Buffer capacity.
 */
void mcp_get_build_dir(
    char   *buf,
    size_t  size)
{
    const char *env_build = getenv("GRIC_BUILD_DIR");
    if (env_build != NULL && env_build[0] != '\0')
    {
        strncpy(buf, env_build, size - 1);
        buf[size - 1] = '\0';
        return;
    }

    char root[512];
    mcp_get_project_root(root, sizeof(root));

    /* Check executable parent directory for CMakeCache.txt */
    char exe_path[512];
    get_current_executable(exe_path, sizeof(exe_path));
    if (exe_path[0] != '\0')
    {
        char exe_dir[512];
        strncpy(exe_dir, exe_path, sizeof(exe_dir) - 1);
        exe_dir[sizeof(exe_dir) - 1] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash != NULL)
        {
            *slash = '\0';
            char check[1024];
            snprintf(check, sizeof(check), "%s/CMakeCache.txt", exe_dir);
            if (access(check, F_OK) == 0)
            {
                strncpy(buf, exe_dir, size - 1);
                buf[size - 1] = '\0';
                return;
            }
            slash = strrchr(exe_dir, '/');
            if (slash != NULL)
            {
                *slash = '\0';
                snprintf(check, sizeof(check), "%s/CMakeCache.txt", exe_dir);
                if (access(check, F_OK) == 0)
                {
                    strncpy(buf, exe_dir, size - 1);
                    buf[size - 1] = '\0';
                    return;
                }
            }
        }
    }

    /* Check standard build directories relative to root */
    const char *candidates[] = {
        "build",
        "build-milk",
        "build-asan",
        "_build",
        NULL
    };
    for (int ii = 0; candidates[ii] != NULL; ii++)
    {
        char cand_path[1024];
        snprintf(cand_path, sizeof(cand_path), "%s/%s", root, candidates[ii]);
        char check_cmake[2048];
        snprintf(check_cmake, sizeof(check_cmake), "%s/CMakeCache.txt", cand_path);
        if (access(check_cmake, F_OK) == 0)
        {
            strncpy(buf, cand_path, size - 1);
            buf[size - 1] = '\0';
            return;
        }
    }

    snprintf(buf, size, "%s/build", root);
} // mcp_get_build_dir

/**
 * mcp_find_executable() - Locate a build target or installed executable.
 * @name:     Base executable filename (e.g. "gric-cluster").
 * @out_path: Target buffer for full executable path.
 * @out_size: Buffer capacity.
 *
 * Return: 0 if found and executable, -1 on failure.
 */
int mcp_find_executable(
    const char *name,
    char       *out_path,
    size_t      out_size)
{
    if (name == NULL || name[0] == '\0')
    {
        out_path[0] = '\0';
        return -1;
    }

    /* Direct path check */
    if (access(name, X_OK) == 0)
    {
        strncpy(out_path, name, out_size - 1);
        out_path[out_size - 1] = '\0';
        return 0;
    }

    char build_dir[1024];
    mcp_get_build_dir(build_dir, sizeof(build_dir));

    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    char exe_dir[1024] = {0};
    char exe_path[1024];
    get_current_executable(exe_path, sizeof(exe_path));
    if (exe_path[0] != '\0')
    {
        strncpy(exe_dir, exe_path, sizeof(exe_dir) - 1);
        exe_dir[sizeof(exe_dir) - 1] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash != NULL)
        {
            *slash = '\0';
        }
    }

    char candidate[2048];

    /* 1. Try active build directory */
    snprintf(candidate, sizeof(candidate), "%s/%s", build_dir, name);
    if (access(candidate, X_OK) == 0)
    {
        strncpy(out_path, candidate, out_size - 1);
        out_path[out_size - 1] = '\0';
        return 0;
    }

    snprintf(candidate, sizeof(candidate), "%s/bin/%s", build_dir, name);
    if (access(candidate, X_OK) == 0)
    {
        strncpy(out_path, candidate, out_size - 1);
        out_path[out_size - 1] = '\0';
        return 0;
    }

    snprintf(candidate, sizeof(candidate), "%s/src/gric-fps/%s", build_dir, name);
    if (access(candidate, X_OK) == 0)
    {
        strncpy(out_path, candidate, out_size - 1);
        out_path[out_size - 1] = '\0';
        return 0;
    }

    /* 2. Try current executable's directory */
    if (exe_dir[0] != '\0')
    {
        snprintf(candidate, sizeof(candidate), "%s/%s", exe_dir, name);
        if (access(candidate, X_OK) == 0)
        {
            strncpy(out_path, candidate, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 0;
        }

        snprintf(candidate, sizeof(candidate), "%s/src/gric-fps/%s", exe_dir, name);
        if (access(candidate, X_OK) == 0)
        {
            strncpy(out_path, candidate, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 0;
        }
    }

    /* 3. Try standard candidate directories relative to root */
    const char *dirs[] = {"build", "build-milk", "build-asan", "bin", NULL};
    for (int ii = 0; dirs[ii] != NULL; ii++)
    {
        snprintf(candidate, sizeof(candidate), "%s/%s/%s", root, dirs[ii], name);
        if (access(candidate, X_OK) == 0)
        {
            strncpy(out_path, candidate, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 0;
        }

        snprintf(
            candidate, sizeof(candidate), "%s/%s/src/gric-fps/%s", root, dirs[ii], name);
        if (access(candidate, X_OK) == 0)
        {
            strncpy(out_path, candidate, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 0;
        }
    }

    /* Default fallback */
    snprintf(out_path, out_size, "%s/build/%s", root, name);
    return -1;
} // mcp_find_executable

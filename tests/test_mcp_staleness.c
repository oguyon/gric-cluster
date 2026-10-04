/**
 * @file test_mcp_staleness.c
 * @brief Unit tests for gric-mcp server staleness and git HEAD parser.
 */

#include "mcp_staleness.h"
#include "tests/mcp/mcp_test_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void run_cmd(
    const char *cmd)
{
    int ret = system(cmd);
    (void)ret;
}

static void cleanup_fixtures(void)
{
    run_cmd("rm -rf /tmp/test_git_loose /tmp/test_git_packed "
            "/tmp/test_git_detached /tmp/test_git_wt /tmp/test_git_wt_main");
}

static void test_staleness_exe_deleted(void)
{
    mcp_staleness_reset();
    mcp_staleness_set_exe_path_override("/path/to/build/gric-mcp (deleted)");

    const struct mcp_stale_status *st = mcp_staleness_check();
    CHECK(st != NULL);
    CHECK(st->stale == 1);
    CHECK(st->reason != NULL && strcmp(st->reason, "binary_rebuilt") == 0);

    mcp_staleness_reset();
    printf("PASS: test_staleness_exe_deleted\n");
}

static void test_git_resolve_head_loose_ref(void)
{
    run_cmd("mkdir -p /tmp/test_git_loose/.git/refs/heads");
    run_cmd("echo 'ref: refs/heads/feat-x' > /tmp/test_git_loose/.git/HEAD");
    run_cmd("echo '1111222233334444555566667777888899990000' > "
            "/tmp/test_git_loose/.git/refs/heads/feat-x");

    char sha[41];
    int ret = mcp_git_resolve_head("/tmp/test_git_loose", sha, sizeof(sha));
    CHECK(ret == 0);
    CHECK(strcmp(sha, "1111222233334444555566667777888899990000") == 0);

    printf("PASS: test_git_resolve_head_loose_ref\n");
}

static void test_git_resolve_head_packed_refs(void)
{
    run_cmd("mkdir -p /tmp/test_git_packed/.git");
    run_cmd("echo 'ref: refs/heads/main' > /tmp/test_git_packed/.git/HEAD");
    FILE *fp = fopen("/tmp/test_git_packed/.git/packed-refs", "w");
    CHECK(fp != NULL);
    fprintf(fp, "# pack-refs with: peeled-tags fully-peeled sorted\n");
    fprintf(fp, "aaaabbbbccccddddeeeeffff0000111122223333 refs/heads/other\n");
    fprintf(fp, "55556666777788889999aaaabbbbccccddddeeee refs/heads/main\n");
    fclose(fp);

    char sha[41];
    int ret = mcp_git_resolve_head("/tmp/test_git_packed", sha, sizeof(sha));
    CHECK(ret == 0);
    CHECK(strcmp(sha, "55556666777788889999aaaabbbbccccddddeeee") == 0);

    printf("PASS: test_git_resolve_head_packed_refs\n");
}

static void test_git_resolve_head_detached(void)
{
    run_cmd("mkdir -p /tmp/test_git_detached/.git");
    run_cmd("echo 'abcdefabcdefabcdefabcdefabcdefabcdefabcd' > "
            "/tmp/test_git_detached/.git/HEAD");

    char sha[41];
    int ret = mcp_git_resolve_head("/tmp/test_git_detached", sha, sizeof(sha));
    CHECK(ret == 0);
    CHECK(strcmp(sha, "abcdefabcdefabcdefabcdefabcdefabcdefabcd") == 0);

    printf("PASS: test_git_resolve_head_detached\n");
}

static void test_git_resolve_head_worktree(void)
{
    run_cmd("mkdir -p /tmp/test_git_wt_main/.git/worktrees/wt1/refs/heads");
    run_cmd("echo 'ref: refs/heads/branch-wt' > "
            "/tmp/test_git_wt_main/.git/worktrees/wt1/HEAD");
    run_cmd("echo '9999888877776666555544443333222211110000' > "
            "/tmp/test_git_wt_main/.git/worktrees/wt1/refs/heads/branch-wt");

    run_cmd("mkdir -p /tmp/test_git_wt");
    run_cmd("echo 'gitdir: /tmp/test_git_wt_main/.git/worktrees/wt1' > "
            "/tmp/test_git_wt/.git");

    char sha[41];
    int ret = mcp_git_resolve_head("/tmp/test_git_wt", sha, sizeof(sha));
    CHECK(ret == 0);
    CHECK(strcmp(sha, "9999888877776666555544443333222211110000") == 0);

    printf("PASS: test_git_resolve_head_worktree\n");
}

static void test_staleness_head_moved(void)
{
    mcp_staleness_reset();
    mcp_staleness_set_root_override("/tmp/test_git_detached");

    const struct mcp_stale_status *st = mcp_staleness_check();
    CHECK(st != NULL);
    CHECK(st->stale == 1);
    CHECK(st->reason != NULL && strcmp(st->reason, "head_moved") == 0);
    CHECK(strcmp(st->repo_head, "abcdefabcdefabcdefabcdefabcdefabcdefabcd") == 0);

    mcp_staleness_reset();
    printf("PASS: test_staleness_head_moved\n");
}

int main(void)
{
    printf("=== Running MCP Server Staleness Tests ===\n");
    cleanup_fixtures();

    test_staleness_exe_deleted();
    test_git_resolve_head_loose_ref();
    test_git_resolve_head_packed_refs();
    test_git_resolve_head_detached();
    test_git_resolve_head_worktree();
    test_staleness_head_moved();

    cleanup_fixtures();
    printf("=== All MCP Staleness Tests Passed ===\n");
    return 0;
}

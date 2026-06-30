---
description: Git worktree management.
---

# Sync Worktree

To review PRs or work on hotfixes without altering your active branch workspace:
1. Add a worktree:
   ```bash
   git worktree add ../gric-cluster-fix main
   ```
2. Perform work in `../gric-cluster-fix`.
3. Prune and remove when finished:
   ```bash
   git worktree remove ../gric-cluster-fix
   ```

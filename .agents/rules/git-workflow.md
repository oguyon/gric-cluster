---
description: Git branching, committing, and worktree workflow.
---

# Git Workflow

The `gric-cluster` repository uses a standard git workflow centered on the `main` branch.

## 1. Branching Strategy
- All development should occur on feature branches branched from `main`.
- Merge feature branches into `main` using Pull Requests after compiling and verifying all changes.

## 2. Commit Guidelines
- Write clear, descriptive commit messages.
- Format: `<component>: <short summary of changes>`
- Example: `core: optimize Euclidean distance vectorization path`

## 3. Worktree Usage
For maintaining multiple branches or reviewing PRs, use Git worktrees rather than
cloning multiple times:
```bash
git worktree add ../gric-review feature-branch
```

## 4. Pull Requests & Agentic Tool Disclosure
When opening a Pull Request:
- Ensure all code compiles cleanly without warnings and passes all tests.
- Disclose the use of agentic tools in the PR description with:
  - An attribution and sign-off message matching the format:
    `Implemented by <model name>. Reviewed and signed off by O. Guyon`
    (e.g., `Implemented by gemini 3.8 flash (high). Reviewed and signed off by O. Guyon`).
  - A short description/summary of what the prompts asked the agent to do.

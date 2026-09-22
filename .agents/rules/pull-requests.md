---
description: Pull request standards, verification requirements, and agentic tool disclosure.
---

# Pull Request Standards

When preparing or submitting Pull Requests (PRs):

## 1. Agentic Tool Disclosure
Any PR implementing changes with the assistance of agentic tools must explicitly
disclose this in the PR description:
- **Attribution and Sign-off**: Include a statement matching the format:
  `Implemented by <model name>. Reviewed and signed off by O. Guyon`
  Example:
  `Implemented by gemini 3.8 flash (high). Reviewed and signed off by O. Guyon`
- **Prompt Summary**: Include a short description/summary of what the prompts asked
  the agent to do.

## 2. Verification Checklist
- **Compilation**: Code builds cleanly with zero warnings or errors.
- **Style**: Adheres to project C style (Allman braces, <= 100 chars,
  parameter alignment).
- **Testing**: All automated tests pass; regression checks on benchmark datasets succeed.
- **Git History**: Clean, focused commit messages following project conventions.

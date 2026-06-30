---
name: pr-preparation
description: Standard check before preparing a Pull Request.
---

# PR Preparation

Before opening a PR or merging code:

## Checklist
1. **Clean compilation:** Check that the code builds with zero warnings or errors.
2. **Format compliance:** Ensure code adheres to the project C style (K&R braces, short lines).
3. **Tests passing:** Run local dataset tests to verify no regressions occur in the clustering outputs.
4. **Clean Git tree:** Keep commit history structured and focus each commit on a single logical change.

---
description: Reproduce, debug, and fix a bug in gric-cluster.
---

# Fix Bug

Follow this sequence to fix bug reports:
1. **Reproduce:** Construct a test case or config that consistently reproduces the issue.
2. **Trace:** Check error outputs and logs. Use sanitizers (ASAN/UBSAN) if necessary by compiling with `-fsanitize=address`.
3. **Fix:** Implement the bugfix using proper conventions (defensive coding, single-label cleanup, K&R braces).
4. **Verify:** Run compile and test validations.

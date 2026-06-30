---
description: Code quality auditing of C sources.
---

# Audit Code Quality

Check for:
- [ ] Correct brace style (K&R/OTBS).
- [ ] No lines exceeding 100 characters.
- [ ] Zero implicit header dependencies (include what is used).
- [ ] Correct loop index type matches (e.g. `uint64_t` index for `uint64_t` bounds).
- [ ] Proper error handling, return value checks, and resource cleanup.

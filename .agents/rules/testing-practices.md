---
description: Testing guidelines for verification of clustering results.
---

# Testing Practices

Ensure all changes are validated for regressions and correctness before completing tasks.

## 1. Local Compile & Test
After making changes to source or CMake configuration:
1. Rebuild the project:
   ```bash
   mkdir -p build && cd build && cmake .. && make -j$(nproc)
   ```
2. Verify by running the clustering on test coordinate datasets:
   ```bash
   ./gric-cluster 0.5 ../test_strat.txt
   ```
3. Run test data generators to verify formatting is correct:
   ```bash
   ./gric-mktxtseq
   ```

## 2. Regression Testing
When fixing a bug:
1. Create a script or configuration that triggers the bug.
2. Confirm the bug reproduces prior to your code changes.
3. Confirm the bug is fully resolved after your changes, without introducing other build warnings or compiler errors.

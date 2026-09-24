## Summary of Changes
<!-- Provide a clear, concise overview of what changed and why. -->

## Changes Checklist
- [ ] Code compiles cleanly with zero warnings (`-Wall -Wextra`)
- [ ] All CTest unit and regression tests pass (`ctest --output-on-failure`)
- [ ] Adheres to C code style guide (Allman braces, <= 100 character lines, column-aligned parameters)
- [ ] No allocations (`malloc`, `calloc`) inside compute/clustering loops
- [ ] No resource/memory leaks (verified with ASan/UBSan or Valgrind)
- [ ] New dependencies reflected in `CMakeLists.txt` and `dependency_graph.md`

## Testing Performed
<!-- Describe how these changes were tested (commands executed, benchmarks run, datasets tested). -->
```bash
# Example:
ctest --test-dir build --output-on-failure
```

---
<!--
If agentic AI tools were used, disclose below per project guidelines:
Implemented by <model name>. Reviewed and signed off by O. Guyon.
Followed by a concise technical summary of what task the model performed.
-->
Implemented by . Reviewed and signed off by O. Guyon.

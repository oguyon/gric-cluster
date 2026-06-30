---
name: refactor-c-source
description: Safely split and reorganize large C source files into smaller modules.
---

# Refactor C Source Files

This skill guides the agent through safely splitting a large `.c` file into multiple smaller files.

## Workflow
1. **Analyze the source file:** Map functions, dependencies, static variables, and headers.
2. **Design the split:** Propose new filenames and assign functions logically (e.g., helpers, computation, input/output).
3. **Implement splitting:**
   - Create the new `.c` files and matching headers.
   - Include only the specific headers required by each new file.
4. **Update CMake:** Add new source files to `CLUSTER_SRCS` in the root `CMakeLists.txt`.
5. **Verify:** Compile the project and run manual dataset tests to ensure no regressions.

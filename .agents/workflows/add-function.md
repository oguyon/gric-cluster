---
description: Add a new function to an existing C file
---

# Add a Function to an Existing C File

Follow this workflow when adding a function to `gric-cluster`.

## Workflow
1. **Define prototype:** Add the prototype to the corresponding header file.
2. **Implement logic:** Implement the function in the `.c` file, using K&R brace style.
3. **Document:** Write Kernel-Doc style comments above the function definition.
4. **Update headers:** Check that all headers required by the new logic are explicitly included in the `.c` file.
5. **Compile and verify:** Compile the project to confirm there are no compiler warnings.

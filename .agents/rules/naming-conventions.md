---
description: Naming conventions for files, variables, functions, and structures.
---

# Naming Conventions

Maintain consistency with the existing `gric-cluster` codebase naming patterns.

## 1. Files & Folders
- Use lowercase `snake_case` with underscores for C source/header files: e.g., `cluster_core.c`, `png_io.c`.
- Keep names descriptive and concise.

## 2. Functions
- Public API functions declared in headers: `<component>_<verb>_<object>` or similar clear style: e.g., `run_clustering`, `framedist`.
- Static helper functions: lowercase `snake_case` descriptive names.

## 3. Variables
- Local variables: short, lowercase `snake_case`.
- Loop indices:
  - Innermost/pixel processing loops: Use doubled letters `ii`, `jj`, `kk`.
  - Outer or general index loops: Use descriptive indices like `arg_idx`, `cl_idx`, `frame_idx`.
  - Avoid single-character indices like `i`, `j`, `k` in non-trivial loops to remain searchable.
- Dimension variables:
  - `width`: image width (axis 0)
  - `height`: image height (axis 1)
  - `size`: total pixels (`width * height`)
- Pointers: Descriptive name, optionally suffixed with `_ptr` if helpful.

## 4. Structs & Types
- Struct names: use `CamelCase` or `UPPER_CASE` typedefs matching the existing style: e.g., `ClusterConfig`, `ClusterState`, `Frame`, `Candidate`.

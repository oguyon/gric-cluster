# GRIC Developer & Agent Guidelines (AGENTS.md)

Welcome to the `gric-cluster` codebase. This document outlines core instructions,
architecture boundaries, and tools for AI coding assistants and developers.

---

## 1. First Steps: Consult `gric-mcp` Before Writing Code

Before implementing new features, CLI commands, or algorithms, always query the
built-in **`gric-mcp`** server to inspect existing functionality and documentation:

- **Check existing tools**: Call `gric_list_suite` to browse all 18+ suite binaries
  and avoid reinventing existing utilities.
- **Query documentation**: Call `gric_help(topic="...")` or `gric_help(query="...")`
  to search 95+ embedded topics (e.g. `rlim`, `entropy`, `quantization`, `milk`).
- **Use cookbook recipes**: Call `gric_get_recipe(recipe="...")` for step-by-step
  instructions (`milk_realtime_streaming`, `high_dim_knn_search`, `image_cube_clustering`).
- **Validate data & invariants**: Use `gric_probe_dataset` and `gric_verify_invariants`.
- **Milk operations**: Use `gric_fps_status`, `gric_fps_run`, `gric_fps_set`, `gric_fps_stop`,
  and `gric_probe_fps_streams` when working with Milk streaming clustering.

---

## 2. Core Architecture Principles

- **Layered Design**:
  - **Level 2 (`libgric`)**: Pure high-performance compute library. Must remain
    completely independent of external frameworks (no Milk, no GUI dependencies).
  - **Level 3 (`libmilkgric.so`, `milk-fpsexec-gric-cluster`)**: Adapters connecting
    `libgric` to Milk Function Parameter Structures (FPS) and ImageStreamIO SHM.
- **Zero Allocations in Compute Loops**:
  - Never call `malloc()`, `calloc()`, or `realloc()` inside core clustering loops
    (`run_clustering`) or distance functions (`framedist`). Pre-allocate all buffers.
- **Resource Management**:
  - Always clean up FITS handles, FFmpeg streams, PNG handles, and file descriptors.
  - Follow the Linux kernel `goto cleanup` pattern in reverse allocation order.

---

## 3. Strict Code Style Rules

All C code and markdown files must strictly comply with:

1. **Brace Style: Allman**:
   Opening braces must be on their own line for both functions and control flow
   (`if`, `for`, `while`, `switch`).
2. **Line Length Limit: $\le 100$ Characters**:
   Applies to all `.c`, `.h`, `.md`, and script files. Do not exceed 100 characters.
3. **Column-Aligned Function Prototypes**:
   Multi-line prototypes and definitions must use column-aligned parameter names
   per `.agents/rules/parameter-alignment.md`.
4. **Header Hygiene**:
   Every `.c` file must include exactly the headers it uses (no implicit includes).
5. **No Implicit Double Promotions**:
   Use single-precision float functions (`sqrtf`, `powf`) and float literals (`0.5f`)
   when working with 32-bit floats.

---

## 4. Specialized Skills (`.agents/skills/`)

Activate these skills when working on specialized areas:
- `milk-adapter-pattern`: Standard architecture and templates for Milk integration.
- `imagestream-internals`: ImageStreamIO shared memory layout and semaphore synchronization.
- `optimize-compute-function`: Checklists and guidelines for hot clustering loops.
- `advanced-math-patterns`: High-performance numerical and SIMD patterns.
- `pr-preparation`: Pull request validation checklist and disclosure note requirements.
- `diagnose-build-failure`: CMake and compiler troubleshooting.

---

## 5. Build & Test Commands

```bash
# Build entire suite (including gric-mcp)
make -C build -j

# Run complete test suite (must pass 100%)
ctest --test-dir build --output-on-failure
```

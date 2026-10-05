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
- **Server status**: Call `gric_server_info` to inspect active toolsets and server capabilities.

### Toolsets & Operational Safety

`gric-mcp` organizes tools into modular toolsets:
- **`knowledge`** (read-only): `gric_help`, `gric_list_suite`, `gric_get_recipe`,
  `gric_server_info`.
- **`analysis`** (read-only): `gric_inspect_run`, `gric_probe_dataset`,
  `gric_verify_invariants`, `gric_probe_shm`.
- **`ops`**: `gric_fps_status`, `gric_probe_fps_streams` (read-only), and
  `gric_fps_run`, `gric_fps_set`, `gric_fps_stop` (side-effecting).
- **`dev`** (read-only): `gric_audit_code_style`, `gric_align_parameters`,
  `gric_inspect_simd` (available when running inside a source tree).
- **`run`**: Solver and clustering execution tools.

**Presets and Safety Flags**:
- **Default**: `--toolsets=knowledge,analysis,run`.
- **Full**: `--toolsets=all` (configured in `.agents/mcp_config.json`).
- **Read-Only**: `--read-only` hides and rejects all side-effecting operations
  (`gric_fps_run`, `gric_fps_set`, `gric_fps_stop`).
- **Synchronization**: Follow [.agents/rules/mcp-sync.md](file://.agents/rules/mcp-sync.md)
  whenever adding CLI flags, FPS parameters, stream layouts, or suite programs.

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
- `advanced-math-patterns`: High-performance numerical and SIMD patterns.
- `diagnose-build-failure`: CMake and compiler troubleshooting.
- `feature-planner`: Guidelines for planning new features or refactorings.
- `imagestream-internals`: ImageStreamIO shared memory layout and semaphore synchronization.
- `optimize-compute-function`: Checklists and guidelines for hot clustering loops.
- `pr-preparation`: Pull request validation checklist and disclosure note requirements.
- `refactor-c-source`: Safely split and reorganize large C source files into smaller modules.
- `simd-optimization`: Methodology for writing, auditing, and benchmarking SIMD compute kernels.

---

## 5. Build & Test Commands

```bash
# Build entire suite (including gric-mcp)
make -C build -j

# Run complete test suite (must pass 100%)
ctest --test-dir build --output-on-failure
```

---

## 6. Pre-Merge Invariants & CI Gates

Never merge a PR based on local testing alone. Before merging any PR, verify:
1. **100% Green CI Status**: Run `gh pr checks <PR_NUM>` and confirm all matrix jobs
   pass (0 failing, 0 pending, 0 cancelled).
2. **POSIX & macOS Compatibility**: No Linux glibc-only calls (e.g. naked `pipe2`) or
   `/proc` assumptions without Darwin fallbacks (`_NSGetExecutablePath()`, directory crawl).
3. **Memory Alignment Sizing**: `aligned_alloc(alignment, size)` requires `size % alignment == 0`
   to avoid ASan aborts (`invalid-aligned-alloc-alignment`) and macOS `EINVAL`/SIGSEGV.
4. **Pointer Lifetimes**: Never store or return pointers to local stack buffers (`char buf[...]`);
   Clang/GCC `-O2`/`-O3` optimization passes pop and overwrite stack frames.
5. **Dynamic Path Resolution**: Always use `mcp_find_executable()` and `mcp_get_build_dir()`
   rather than hardcoding `./build/` paths.


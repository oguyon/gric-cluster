---
name: refactor-c-source
description: Safely split and reorganize large C source files and long functions into clean modules.
---

# Refactoring C Source Files & Long Functions

This skill provides step-by-step guidance for safely decomposing oversized C source files
(> 600 lines) and long functions (> 60 lines) into modular, testable, and maintainable units
without introducing behavioral regressions or performance penalties.

---

## 1. Targeted Metrics & Code Hygiene

When refactoring, aim for the following targets:
- **Source Files**: $\le 600$ lines (hard limit $1000$ lines).
- **Function Bodies**: $\le 60$ lines (hard limit $150$ lines).
- **`main()` Functions**: $\le 40$ lines (hard limit $80$ lines).
- **Brace Style**: Strictly Allman (opening brace `{` on its own line).
- **Documentation**: Kernel-Doc comment above every non-trivial function with a clear,
  single-sentence summary on the first line.

---

## 2. Function Decomposition Patterns

### Pattern A: Pipeline / Phase Extraction (Context Struct)
Long computational pipelines that interweave preparation, multi-pass computation, and cleanup
should be partitioned into sequential phase functions sharing a context structure:

```c
/* Instead of one 400-line monolithic runner: */
int engine_run(struct engine_config *cfg)
{
    struct engine_ctx ctx;

    if (engine_init_context(&ctx, cfg) != 0)
    {
        return -1;
    }

    if (engine_prepare_buffers(&ctx) != 0 ||
        engine_execute_passes(&ctx) != 0 ||
        engine_finalize_output(&ctx) != 0)
    {
        engine_cleanup(&ctx);
        return -1;
    }

    engine_cleanup(&ctx);
    return 0;
}
```

### Pattern B: Table-Driven CLI & Dispatch
Large `main()` or CLI parsing functions with hundreds of lines of `if (strcmp(...) == 0)` should
be converted into static option tables processed by `cli_opt` or dedicated handler tables.

### Pattern C: ISA Strategy Separation (SIMD Kernels)
When a numerical algorithm contains separate AVX2, AVX-512, and scalar variants in a single file:
1. Split into distinct files: `<algo>_scalar.c`, `<algo>_avx2.c`, `<algo>_avx512.c`.
2. Keep the dispatch logic and public entry point in `<algo>.c`.
3. Use a function pointer table or static dispatch populated at initialization via CPUID detection.
4. This keeps files compact and allows CMake to attach target architecture flags per-file.

### Pattern D: Predicate & Inline Extraction
Extract deeply nested boolean conditions into `static inline bool is_<condition>(...)` functions
in internal headers. This flattens control flow and documents the domain intent.

---

## 3. File Partitioning Heuristics

1. **Partition by Responsibility**:
   - **`<module>_core.c`**: Primary data structures and lifecycle.
   - **`<module>_step.c`**: Algorithmic transition and state evolution.
   - **`<module>_math.c`**: Pure numerical formulas and kernels.
   - **`<module>_io.c`**: Serialization, file, and stream I/O.
   - **`<module>_cli.c`**: Argument validation and interactive diagnostics.
2. **Private Internal Headers**:
   - Create `<module>_internal.h` for declarations shared between files of the same module.
   - Do NOT expose internal headers in `include/gric/`.
3. **Avoid Grab-Bag Anti-Patterns**:
   - Never create `utils.c`, `misc.c`, or `helpers.c`.
   - Name files according to the specific domain noun and verb (e.g. `cluster_bounds.c`).

---

## 4. Safe Refactoring Workflow

Follow this ordered protocol when executing a refactoring:

1. **Establish Baseline**:
   - Record test suite status: `ctest --test-dir build --output-on-failure`.
   - If refactoring hot compute loops, capture a performance baseline using
     `scripts/perf_baseline.sh` or `gric-benchmark`.
2. **Move or Split with `git mv`**:
   - Use `git mv` when renaming or relocating files to preserve commit history.
   - Keep pure file moves in a separate commit from internal code changes.
3. **Extract Functions Incrementally**:
   - Extract one helper at a time.
   - Maintain column-aligned parameters and Allman braces.
   - Ensure header hygiene: each new file includes only its required headers.
4. **Update Build Files**:
   - Add new source files to CMake targets (`CMakeLists.txt` or subdirectories).
5. **Verify Correctness & Invariants**:
   - Build: `make -C build -j`.
   - Test: `ctest --test-dir build --output-on-failure`.
   - Verify code size limits: `scripts/check_code_size.sh`.
   - Run invariant checks: `scripts/check_pr_invariants.sh`.
   - For SIMD kernels, run `gric_inspect_simd` to ensure compiler auto-vectorization
     and loop unrolling were not compromised by lost inlining.
6. **Documentation & MCP Sync**:
   - Update `dependency_graph.md` if cross-module dependencies changed.
   - Follow `.agents/rules/mcp-sync.md` if CLI flags or binary interfaces changed.

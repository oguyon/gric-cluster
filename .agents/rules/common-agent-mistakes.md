---
trigger: always_on
---

# Common Agent Mistakes

Consolidated checklist of pitfalls that AI agents frequently hit when generating code for `gric-cluster`. Check this list before finalizing any generated code.

## CMake & Build System
1. **Forgetting to add `.c` to CLUSTER_SRCS.**
   If you add a source file used by the main clustering binary, it must be added to `CLUSTER_SRCS` in the root `CMakeLists.txt`.
2. **Not linking optional dependencies.**
   When compiling new executables, link against standard libraries and optional modules if they are used (e.g., `${IMAGESTREAMIO_LIBRARIES}`, `${PNG_LIBRARIES}`, `${CFITSIO_LIBRARIES}`, `m`).

## Memory & Resource Management
3. **Allocation inside compute loops.**
   Never call `malloc()`, `calloc()`, or `realloc()` inside the core clustering loop (`run_clustering`) or distance calculation functions (`framedist`). Allocate buffers during setup/initialization and reuse them.
4. **Leaking file descriptors or handles.**
   Always close FITS handles, FFmpeg streams, PNG structures, and file pointers on error paths. Use the `goto cleanup` pattern to free resources in reverse allocation order.
5. **Leaving test SHM files.**
   When creating test `ImageStreamIO` streams in tests/debug, clean them up from `/dev/shm/` immediately afterwards.

## Compiler & Math Correctness
6. **Implicit double promotions.**
   Ensure that float math uses float variants (`sqrtf`, `powf`) and float literals (`0.5f`) when working with single-precision floats, or standard double variants when using double-precision floats.
7. **Type mismatches in loop indices.**
   Always match the loop index type to the bound variable type (e.g., `for (uint64_t ii = 0; ii < xysize; ii++)` when `xysize` is `uint64_t`) to prevent breaking compiler SIMD auto-vectorization.
8. **Implicit header includes.**
   Every `.c` file must include exactly the headers it uses. Do not rely on header side-effects.
9. **Lines > 100 characters.**
   Limit line length in C source code, scripts, and documentation files to 100 characters.

## Cross-Platform & OS Portability
10. **Linux-only functions without POSIX fallbacks.**
    Never use glibc-only APIs like `pipe2(..., O_CLOEXEC)` without a portable fallback.
    macOS Darwin lacks `pipe2()`. Always provide standard `pipe()` + `fcntl(..., FD_CLOEXEC)`.
11. **Assuming `/proc` exists.**
    macOS Darwin does not have `/proc`. Never assume `/proc/self/exe` is available.
    Use `_NSGetExecutablePath()` on Darwin and provide hierarchical directory traversal.

## Memory Alignment & Invariants
12. **`aligned_alloc(alignment, size)` sizing rule.**
    POSIX and AddressSanitizer (ASan) mandate that `size` MUST be an integral multiple of
    `alignment` (`size % alignment == 0`). Non-multiples abort under ASan with
    `invalid-aligned-alloc-alignment` and return `EINVAL`/SIGSEGV on macOS. Always pad `size`
    to a multiple of `alignment` (e.g. `bench_aligned_alloc()`).

## Lifetime & Optimization Safety
13. **Dangling pointers to stack buffers.**
    Never assign local stack buffers (`char buf[...]`) to struct pointer fields or return them.
    Under Clang/GCC `-O2`/`-O3` optimizations, stack frames are popped and overwritten,
    causing corrupted data and intermittent test failures. Copy strings into struct buffers.
14. **Hardcoded build paths.**
    Never hardcode `./build/` or `%s/build/` when finding executables. Build directories vary
    (`build-milk`, `build-asan`, custom prefixes). Always use `mcp_find_executable()` and
    `mcp_get_build_dir()`.

## Pre-Merge Verification
15. **Merging before all CI checks pass green.**
    Never merge a PR based solely on local tests. Always verify all GitHub Actions matrix jobs
    (`gh pr checks <PR>`) are 100% successful (0 failing, 0 pending) across Linux, macOS,
    Clang Release, ASan/UBSan Debug, GCC, and Milk Streaming Adapter.

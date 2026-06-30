---
description: Defensive coding practices, buffer safety, pointer discipline, and resource limits.
---

# Defensive Programming Practices

The project structures its code defensively to guarantee robustness and stability in high-performance clustering.

## 1. Buffer and String Safety
- **Ban unbounded functions:** `strcpy()`, `sprintf()`, and `strcat()` are strictly forbidden.
- **Use bounded alternatives:** Always use `strncpy()`, `snprintf()`, and `strncat()`. Ensure the destination size is explicitly passed and that the resulting string is null-terminated.

## 2. Pointer Discipline
- **Initialization:** Always initialize pointers to `NULL` or to valid memory immediately upon declaration.
- **Dereference safety:** Validate pointers (especially incoming arguments) against `NULL` before dereferencing.
- **Dangling pointers:** Immediately after calling `free()` on a pointer, set it to `NULL`.

## 3. Input Validation
- **Untrusted input:** Validate command-line arguments, config file values, and file contents. Check lengths, ranges, and formats prior to processing.
- **Bounds check:** Validate configuration options (like limits, thread counts, threshold weights) during startup configuration checks (`apply_option`), not inside hot computational loops.

## 4. Integer Arithmetic and Bounds Checking
- **Safe arithmetic:** Prevent integer overflow and underflow. Be mindful of signed versus unsigned comparisons.
- **Array access:** Validate array indices before accessing memory.
- **Hoist checks:** Hoist bounds checking and size validation outside of tight compute loops or SIMD regions. Pre-compute loop boundaries based on frame size once during initialization.

## 5. State Initialization
- **Zero-initialization:** Prefer `calloc()` over `malloc()` for allocating structs to avoid uninitialized memory bugs. If `malloc()` is used, explicitly initialize all fields immediately.
- **Pre-allocation:** Allocate state structures (like `ClusterState` or visitor lists) once during initialization and reuse them. Never call allocations inside the frame distance computation or clustering loop.

## 6. Resource Limits
- **Avoid exhaustion:** Apply maximum bounds on loops, file reads, or memory allocations (e.g., using `maxnbclust` or `maxnbfr` settings) to prevent infinite loops or Out-Of-Memory (OOM) scenarios.

## 7. Format String Safety
- **Safe formatting:** Never pass user input or external data directly as the format string to `printf()` or `fprintf()`. Always use a literal format string: `fprintf(stderr, "%s", untrusted_string)`.

## 8. Safe Signal Handling
- Keep signal handlers (for `SIGINT`) strictly minimal.
- Signal handlers should only set a `volatile sig_atomic_t` flag (e.g., `stop_requested = 1`) that the main loop safely polls.

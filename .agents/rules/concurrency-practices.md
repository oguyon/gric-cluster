---
description: Thread safety and concurrency patterns for gric-cluster.
---

# Concurrency Practices

`gric-cluster` supports multi-threading via OpenMP and optional streaming via `ImageStreamIO` shared memory. Follow these patterns to avoid race conditions.

## OpenMP Parallelization
- Ensure loops parallelized via `#pragma omp parallel for` do not have race conditions on shared state.
- Use atomic updates for shared statistics or counters:
  ```c
  #ifdef _OPENMP
  #pragma omp atomic
  #endif
  state->framedist_calls++;
  ```
- Keep loop index variables private to each thread (declaring them inside the `for` statement achieves this automatically in C99/C11/C17).

## Shared Memory Semaphore Protocol
- When reading/writing streams via `ImageStreamIO`, use the library's semaphore functions (`ImageStreamIO_semwait`, `ImageStreamIO_sempost`) for synchronization.
- Wait on the reader semaphore index assigned to your process to receive frame updates without CPU busy-waiting.
- Always check the return values of `ImageStreamIO` calls for failure.

## Volatile Keyword
- Use `volatile sig_atomic_t` for signal handling flags (e.g., `stop_requested`):
  ```c
  extern volatile sig_atomic_t stop_requested;
  ```
- Do not use `volatile` on data arrays or pointers, as it disables compiler optimization and SIMD vectorization entirely.

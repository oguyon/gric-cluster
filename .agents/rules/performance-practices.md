---
description: Guidelines for high-performance loops, vectorization, and thread safety.
---

# Performance Practices

Performance is critical for high-speed clustering. Follow these principles to optimize hot-path functions.

## 1. Compiler Attributes
- Annotate performance-critical helper functions (like `framedist` or pixel-level math) with compiler qualifiers:
  - Use `__attribute__((hot))` to hint that the function is on the hot path.
  - Use `restrict` (or `__restrict__`) on non-aliasing pointers to allow the compiler to assume vectorization safety.

## 2. Element-wise Vectorization (SIMD)
- Ensure loop indices match the bound types. Mismatching type sizes (e.g., `uint32_t` index vs `uint64_t` size) can block vectorization.
- When writing AVX/SIMD intrinsics, ensure data is aligned, or use unaligned load instructions (`_mm256_loadu_pd`) if alignment is not guaranteed.
- Check target flags (`__AVX__`, `__FMA__`) using preprocessor macros before compiling architecture-specific paths.

## 3. Zero-Allocation on Hot Path
- Never call memory allocators (`malloc`, `calloc`, `realloc`, `free`) in per-frame loops. All working space must be allocated during setup (`ClusterState` initialization) and reused.

## 4. Loop Optimizations
- Use `#pragma omp parallel for` with OpenMP to parallelize independent calculations across CPU cores.
- Minimize transcendental functions (`sqrt`, `pow`, `sin`) in loops. For integer exponents, write explicit multiplications (e.g., `diff * diff` instead of `pow(diff, 2)`).

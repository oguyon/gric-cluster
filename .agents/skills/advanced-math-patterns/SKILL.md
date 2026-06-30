---
name: advanced-math-patterns
description: Reference for implementing high-performance mathematical operations.
---

# Advanced Math Patterns

Naive loops can become a performance bottleneck in clustering algorithms. Follow these standard patterns for high-performance math.

## 1. Matrix and Vector Vectorization (SIMD)
When writing operations across large arrays or images, guide the compiler to vectorize:
1. **Restrict Pointers:** Always use `restrict` on local pointers to indicate that memory regions do not overlap.
2. **Type Matching:** Loop index types MUST match the bound variables (e.g. `uint64_t ii = 0; ii < size` where `size` is `uint64_t`). Mismatches block auto-vectorization.
3. **OpenMP + SIMD:** Use OpenMP pragmas to parallelize loops:
   ```c
   #pragma omp parallel for
   for (long ii = 0; ii < size; ii++) {
       out[ii] = in[ii] * factor;
   }
   ```

## 2. Avoiding Transcendentals in Hot Loops
Function calls like `sqrt()` and `pow()` inside loops destroy performance:
- **Integer Powers:** Never use `pow(x, 2)`. Use `x * x` instead.
- **Float vs Double:** If processing floats, use `sqrtf()`, `powf()`, `sinf()`. If doubles, use `sqrt()`, `pow()`, `sin()`. Do not mix types to avoid costly promotions.
- **Algebraic Reductions:** If comparing distances against a threshold (`d < limit`), compare squared distance (`d^2 < limit^2`) to avoid computing the square root (`sqrt`) on every iteration.

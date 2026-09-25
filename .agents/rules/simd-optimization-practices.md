---
description: Mandatory rules and constraints for SIMD intrinsics and compiler vectorization.
---

# SIMD Optimization Practices

Follow these mandatory rules when considering, writing, or refactoring SIMD intrinsics.

## 1. When to Use Explicit SIMD vs. Compiler Auto-Vectorization
* **Use Clean C by Default:** Write idiomatic, portable C with `restrict` pointers and
  `#pragma omp simd` for element-wise operations and loops over small arrays ($N < 64$).
* **Reserve Intrinsics for True Bottlenecks & High-Value Wins:**
  1. 1-to-N batched distances with query register reuse (`framedistance_batch_*`).
  2. 2D register-tiled matrix distance microkernels (`cluster_gemm_dist`).
  3. Single-vector high-throughput distance reductions (`framedist`).
  4. Algorithmic SIMD (FastScan in-register LUTs via `vpshufb`, $E_8$ lattice, VNNI dot products).
  5. Chunked early-cutoff reductions (vectorized 32/64-element bounds checking).
  6. Top-$k$ branchless SIMD heaps and linear scans.

## 2. Forbidden Pseudo-SIMD Anti-Patterns
* **No Scatter-Gather Packing in Branching Loops:** Never use `_mm256_set_pd()` or
  `_mm512_set_pd()` to manually assemble non-contiguous variables into vectors inside loops
  where scalar early-breaks can terminate early (e.g., triangle inequality bounds).
  Packing overhead exceeds scalar math and destroys branch prediction.
* **No Intrinsics on Non-Hotspots:** Never introduce AVX2/AVX-512 variants for initialization
  scratch arrays or outer pipeline steps (e.g. normalizing 100 probabilities once per frame).
  Dispatch overhead and code bloat exceed any microsecond gain.
* **No Hybrid Vector-Compare with Scalar Unpacking:** Do not use SIMD comparison masks
  (`_mm512_cmp_pd_mask`) if each bit is subsequently extracted with scalar bit-shifts and stores.
  Use clean branchless scalar logic instead.
* **No Dead Header Inclusions:** Never `#include <immintrin.h>` or `"gric_simd.h"` in files
  that do not directly invoke SIMD intrinsics.

## 3. Requirements for Explicit SIMD Kernels
* **Latency Hiding (Multi-Accumulator Unrolling):** Modern x86 FMA units have a latency of
  4–5 cycles and 2 execution ports (concurrency requirement = 8). When writing reduction loops,
  always unroll with at least 4 independent vector accumulators (`acc0`..`acc3`) to prevent
  pipeline stalls.
* **Dynamic Hardware Dispatch:** Never rely on static `-march=native` for SIMD code paths in
  shared libraries or core engines. Guard vectorized kernels behind runtime CPUID queries via
  `gric_get_simd_level()` in `gric_simd.h`, and annotate kernels with `GRIC_TARGET_AVX2` or
  `GRIC_TARGET_AVX512`.
* **Mandatory Scalar Fallback:** Every handwritten SIMD function must have a verified,
  portable scalar C fallback to ensure execution on non-x86 hardware (e.g., ARM64, WebAssembly).
* **Verify with Benchmarks:** Every new SIMD kernel must be validated against `gric-simd-bench`
  or a cycle timer to prove that it outperforms compiler-generated code.

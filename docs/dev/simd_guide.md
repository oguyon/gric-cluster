# SIMD & Compute Optimization Guide

This guide details vectorization techniques, hardware microkernels, and mathematical formulation
strategies used to achieve high compute density in **GRIC**.

---

## 1. Vectorization Strategies

GRIC employs two complementary vectorization paradigms:

### A. Coordinate-Parallel SIMD (Single-Pair Distance)
When computing the Euclidean distance between two isolated vectors of dimensionality $D$:

$$\|a - b\|^2 = \sum_{d=0}^{D-1} (a_d - b_d)^2$$

* **AVX-512**: Processes 16 single-precision floats per cycle using `_mm512_sub_ps` and
  `_mm512_fmadd_ps`.
* **AVX2 + FMA**: Processes 8 single-precision floats per cycle using `_mm256_sub_ps` and
  `_mm256_fmadd_ps`.
* **ARM NEON**: Processes 4 single-precision floats per cycle using `vsubq_f32` and `vfmaq_f32`.

### B. Batch-Parallel GEMM Formulation (Multi-Candidate Distance)
When evaluating a batch of $M$ sample frames against $N$ cluster centroids, coordinate subtraction
is mathematically reformulated as a matrix multiplication:

$$\|a_i - b_j\|^2 = \|a_i\|^2 + \|b_j\|^2 - 2 \langle a_i, b_j \rangle$$

1. Precompute squared norms $\|b_j\|^2$ once for all cluster centroids.
2. Compute sample norms $\|a_i\|^2$ once per query.
3. Compute all inner products $A \cdot B^T$ using an optimized GEMM tile microkernel.
4. Add the precomputed norm vectors using SIMD broadcast operations.

---

## 2. Microkernel Architecture (`cluster_gemm_dist.c`)

The matrix distance engine uses register-blocked microkernels designed to keep inner loops
entirely inside vector register files without L1 cache thrashing:

### AVX-512 4&times;4 Microkernel
Maintains 4 query vectors and 4 centroid vectors in 16 vector registers (`zmm0`&ndash;`zmm15`):

```c
/* Accumulate 4x4 inner products across dimension step */
for (int d = 0; d < dim_step; d += 16)
{
    __m512 b0 = _mm512_loadu_ps(&B[0 * ldb + d]);
    __m512 b1 = _mm512_loadu_ps(&B[1 * ldb + d]);

    acc00 = _mm512_fmadd_ps(a0, b0, acc00);
    acc01 = _mm512_fmadd_ps(a0, b1, acc01);
    acc10 = _mm512_fmadd_ps(a1, b0, acc10);
    acc11 = _mm512_fmadd_ps(a1, b1, acc11);
}
```

---

## 3. Quantized Distance Accelerators

| Quantization Method | Storage per Value | SIMD Instruction | Effective Bandwidth |
| :--- | :---: | :--- | :---: |
| **Float32 (Native)** | 32 bits | `vfmadd231ps` | 1.0&times; (baseline) |
| **Float64 (Native)** | 64 bits | `vfmadd231pd` | 0.5&times; |
| **SQ16 (Scalar Int16)** | 16 bits | `_mm256_madd_epi16` | 2.0&times; |
| **SQ8 (Scalar UInt8)** | 8 bits | `_mm256_maddubs_epi16` | 4.0&times; |
| **EQ16 (E8 Lattice)** | 2 bits/dim | Lattice Gosset Decoder | 8.0&times; |
| **RaBitQ (Random Bit)** | 1&ndash;2 bits | POPCNT / Nibble LUT | 16.0&times; |
| **PQ (Product Quant)** | 8 bits/subspace | PSHUFB (`shuffle_epi8`) | 8.0&ndash;12.0&times; |

---

## 4. Optimization Guidelines for New Code

1. **Use `restrict` Keyword**: Guarantee non-aliasing between query pointers and anchor buffers.
   Without `restrict`, compilers assume arrays may overlap, disabling auto-vectorization.
2. **Match Loop Types**: Always declare loop counters with the exact integer type matching
   array bounds (e.g. `uint64_t ii = 0; ii < count` where `count` is `uint64_t`).
3. **Avoid Branching in SIMD Loops**: Replace conditional logic with branchless select
   intrinsics (e.g. `_mm256_blendv_ps` or bitwise masking).
4. **Align Heap Allocations**: When allocating large scratch buffers, align pointers to
   64-byte boundaries (`posix_memalign(&ptr, 64, size)`) to prevent unaligned cache line penalties.

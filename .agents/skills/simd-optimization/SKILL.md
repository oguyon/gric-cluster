---
name: simd-optimization
description: Methodology for writing, auditing, and benchmarking SIMD compute kernels in GRIC.
---

# SIMD Optimization Guide

GRIC is first and foremost a high-performance astronomical clustering engine. Hot compute loops
must saturate CPU vector execution pipelines and cache bandwidth.

This skill documents GRIC's high-performance SIMD architecture, identifying where explicit
intrinsics are essential for maximum throughput and how to structure kernels to achieve
$10\times$ to $20\times$ scaling over compiler-generated code.

---

## 1. High-Value SIMD Opportunities (Where GRIC Wins)

Compilers cannot automatically synthesize high-level vector algorithms. Actively identify and
implement explicit SIMD for these patterns:

1. **1-to-N Batched Distances ($15\times$–$21\times$ speedup):**
   * *Mechanism:* Query vector loaded once into registers, broadcast across 4, 8, or 16 anchors.
   * *Files:* `framedistance_batch_float.c`, `framedistance_batch_double.c`.
2. **2D Register-Tiled GEMM Distances ($16\times$–$20\times$ speedup):**
   * *Mechanism:* $4 \times 4$ and $2 \times 4$ outer-product microkernels holding 16 accumulators.
   * *Files:* `cluster_gemm_dist.c`.
3. **Chunked Early-Cutoff ($5\times$–$12\times$ speedup):**
   * *Mechanism:* 64-element SIMD chunks + single-lane threshold check before horizontal adds.
   * *Files:* `framedistance_cutoff.c`, `scalar_quant_fastscan.c`.
4. **Quantized FastScan LUTs ($10\times$–$12\times$ speedup):**
   * *Mechanism:* In-register centroid LUTs via `vpshufb` across 32 or 64 vectors in parallel.
   * *Files:* `scalar_quant_fastscan.c`.
5. **Lattice Decoding ($8\times$–$10\times$ speedup):**
   * *Mechanism:* Branchless $E_8$ parity bit-twiddling and vector blends.
   * *Files:* `e8_lattice_simd.h`.
6. **Low-Bit Dot Products ($6\times$–$10\times$ speedup):**
   * *Mechanism:* AVX2 `vpmaddwd` (SQ16) and AVX-512 VNNI `vpdpbusd` (SQ8).
   * *Files:* `scalar_quant.c`, `eq16_dist.c`.

---

## 2. Kernel Pattern 1: 1-to-N Batched Distance (Register Reuse)

Evaluating 1 query vector against $N$ candidate anchors 1-by-1 reloads the query vector from cache
$N$ times. The batched engine loads the query chunk **once** into a register and evaluates $N$
candidates concurrently, completely saturating dual FMA pipelines:

```c
/* Process 1 query vs 4 anchors simultaneously in 256-bit AVX2 */
for (long i = 0; i <= size - 16; i += 16)
{
    /* Query chunk 0 loaded ONCE into vq0 */
    __m256 vq0 = _mm256_loadu_ps(&q[i]);
    __m256 d0_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a0[i]));
    __m256 d1_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a1[i]));
    __m256 d2_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a2[i]));
    __m256 d3_0 = _mm256_sub_ps(vq0, _mm256_loadu_ps(&a3[i]));

    a0_0 = _mm256_fmadd_ps(d0_0, d0_0, a0_0);
    a1_0 = _mm256_fmadd_ps(d1_0, d1_0, a1_0);
    a2_0 = _mm256_fmadd_ps(d2_0, d2_0, a2_0);
    a3_0 = _mm256_fmadd_ps(d3_0, d3_0, a3_0);

    /* Query chunk 1 loaded ONCE into vq1 (8 accumulators total in flight) */
    __m256 vq1 = _mm256_loadu_ps(&q[i + 8]);
    __m256 d0_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a0[i + 8]));
    __m256 d1_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a1[i + 8]));
    __m256 d2_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a2[i + 8]));
    __m256 d3_1 = _mm256_sub_ps(vq1, _mm256_loadu_ps(&a3[i + 8]));

    a0_1 = _mm256_fmadd_ps(d0_1, d0_1, a0_1);
    a1_1 = _mm256_fmadd_ps(d1_1, d1_1, a1_1);
    a2_1 = _mm256_fmadd_ps(d2_1, d2_1, a2_1);
    a3_1 = _mm256_fmadd_ps(d3_1, d3_1, a3_1);
}
```

---

## 3. Kernel Pattern 2: Chunked Early-Cutoff with Two-Stage Pruning

When searching for nearest neighbors, computing the entire distance vector is wasteful if the
running distance exceeds a cutoff threshold. Use GRIC's two-stage check:

1. **Stage 1 (Zero horizontal adds):** Check if ANY single lane exceeds `cutoff_sq` using
   `_mm512_max_ps` and `_mm512_cmp_ps_mask`. If one lane alone exceeds the threshold, the total
   must exceed it. Abort immediately.
2. **Stage 2 (Periodic reduction):** Every 64 or 128 elements, do a partial horizontal reduction.
   If partial sum > cutoff, abort.

```c
/* Stage 1: Zero-reduction fast prune */
__m512 max01 = _mm512_max_ps(acc0, acc1);
__m512 max23 = _mm512_max_ps(acc2, acc3);
__m512 max_lane = _mm512_max_ps(max01, max23);
if (_mm512_cmp_ps_mask(max_lane, vcut, _CMP_GT_OQ) != 0)
{
    return cutoff_sq + 1.0; /* Immediate abort */
}

/* Stage 2: Periodic horizontal check every 128 floats */
if ((i + 64) % 128 == 0)
{
    float partial = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
    if ((double)partial > cutoff_sq)
    {
        return cutoff_sq + 1.0;
    }
}
```

---

## 4. Kernel Pattern 3: Small-Dimension Specialization ($D = 2, 3$)

In coordinate clustering and trajectory models, vectors frequently have 2 or 3 dimensions.
AVX2/AVX-512 vector pipelines incur load/mask/reduction overhead that is slower than scalar:

```c
/* Always specialize small dimensions at function entry */
if (dim == 2)
{
    float d0 = a[0] - b[0];
    float d1 = a[1] - b[1];
    return (double)sqrtf(d0 * d0 + d1 * d1);
}
if (dim == 3)
{
    float d0 = a[0] - b[0];
    float d1 = a[1] - b[1];
    float d2 = a[2] - b[2];
    return (double)sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
}
```

---

## 5. False Optimization Anti-Patterns (What to Avoid)

Do not sacrifice code clarity for pseudo-SIMD:
* **No Scatter-Gather Packing in Branching Loops:** Never use `_mm256_set_pd(x[i3], x[i2], ...)`
  to assemble scattered memory addresses into vectors when scalar code could break early after
  1–2 iterations.
* **No SIMD on Non-Hotspots:** Never introduce AVX variants for initialization loops or
  outer-loop array updates ($N < 64$).
* **No Hybrid Compare with Scalar Unpacking:** Do not use SIMD comparison masks if the result
  is immediately unpacked in a scalar loop to perform scalar bitwise operations.

---

## 6. Optimization Checklist

When writing or optimizing a compute kernel:
- [ ] **Can we batch?** If evaluating 1 query against multiple candidates, use 1-to-N batching.
- [ ] **Can we prune?** If comparing against a threshold, use chunked early cutoff.
- [ ] **Are we latency-bound?** Use 4 to 8 independent accumulators to saturate dual FMA ports.
- [ ] **Is data contiguous?** Ensure unit-stride memory access; never pack scattered variables.
- [ ] **Are small dims specialized?** Direct scalar paths for $D \le 3$.
- [ ] **Is dynamic CPUID dispatch used?** Dispatch via `gric_get_simd_level()` in `gric_simd.h`.
- [ ] **Is portable fallback present?** Always provide a verified, clean scalar C fallback.
- [ ] **Validated on benchmark:** Benchmark with `build/gric-simd-bench` to prove scaling.

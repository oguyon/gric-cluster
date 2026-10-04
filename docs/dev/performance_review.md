# Runtime Performance Review & Fix Plan

*Review date: 2026-10-03, against commit `5a0fa21`. Refined 2026-10-04 with hardware-counter
profiling (`perf`, `perf_event_paranoid=-1`), same commit.*

This document collects the findings of a runtime performance review of gric-cluster (clustering
hot path, gric-knn, I/O / FPS daemons / reconstruction pipeline, build configuration), backed by
empirical profiling, followed by a phased plan for fixes.

Status legend for findings:

- **Measured**: confirmed by profiling (numbers below).
- **Verified**: confirmed by reading the code; impact estimated, not measured.

---

## 1. Profiling Summary

### 1.1 Method

Initial pass (2026-10-03, `perf` unavailable): LD_PRELOAD `SIGPROF` sampler, gprof on a `-pg`
build, gdb stack sampling.

Refinement pass (2026-10-04), `perf 7.0.14`:

- **Host:** i9-12950HX (hybrid: 8 P-cores with HT = CPUs 0-15, 8 E-cores = CPUs 16-23), 30 MB L3,
  123 GB RAM, glibc 2.43, THP `madvise` mode.
- **Pinning:** single-threaded runs use `OMP_NUM_THREADS=1 taskset -c 4` (one P-core) so that
  `cpu_core` PMU events and timings are not mixed with E-core samples.
- `perf stat` with `cpu_core` counters and the `TopdownL1`/`TopdownL2` TMA metric groups.
- `perf record -e cpu_core/cycles/ --call-graph lbr` (2 kHz) for exact inclusive attribution
  (e.g. separating glibc `memmove` inside `qsort` from other `memmove` callers).
- PEBS precise memory events (`mem_inst_retired.stlb_miss_stores`, `mem_load_retired.l2_miss`,
  `l3_miss`) and `page-faults` sampling to attribute TLB/cache/page-fault costs to code.
- `kptr_restrict=1` is still set, so kernel symbols are unresolved: kernel time is reported as an
  aggregate ("kernel") and attributed through the user-space code that triggers it.
- Fresh Release build at HEAD in `/tmp/gric-perf` (project flags `-O3 -march=native
  -funroll-loops -g`).

### 1.2 Workloads and results

| ID | Command (from repo root) | Frames → clusters | Wall | RSS |
| :--- | :--- | :--- | :--- | :--- |
| W1 | `gric-cluster 0.01 benchmarks/2Drand.txt -maxcl 20000` | 20k → 16,907 | 14.6 s (sys 2.6) | 8.6 GB |
| W2 | `gric-cluster 0.7 workspace/128Dtorus.bin -maxcl 20000` | 30k → 9,908 | 15.3 s (sys 2.3) | 7.9 GB |
| W3 | `gric-cluster 0.883 workspace/512Dtorus.bin` (tuned flags, see below) | 30k → 3,147 | 1.49 s | 454 MB |
| W4 | `gric-cluster 7.4 benchmarks/balls_coll.fits -maxcl 4000` | 20k (32×32) → 2,626 | 2.41 s | 410 MB |

W3 flags (from `512Dtorus.clusterdat/cluster_run.log`):
`-maxim 30000 -te4 -no-te5 -no-entropy -no-tm -no-pred -gprob -maxvis 20 -no-soft-bayesian
-no-tiles -no-sparse-dcc -sq16 -no-sq8 -no-eq16 -maxcl 4096 -evals`

Hardware counters (single P-core):

| Metric | W1 | W2 | W3 | W4 |
| :--- | :--- | :--- | :--- | :--- |
| IPC | 2.22 | 2.40 | 2.12 | 2.20 |
| Retiring / FE-bound / bad-spec / BE-bound (%) | 30 / 19 / 4 / 47 | 31 / 20 / 5 / 44 | 31 / 11 / 5 / 53 | 36 / 18 / 5 / 41 |
| Page faults | 2.16 M | 1.97 M | 99 k | 118 k |
| dTLB store misses | **597 M** | **219 M** | 21 M | 15 M |
| Kernel share of cycles | 18.5% | 15.3% | 9.4% | 8.0% |
| Exact frame→anchor distances per frame | 0.90 | 0.70 | 9.3 | 0.96 |

The last row (`STATS_DISTS_SAMPLE / STATS_FRAMES`) matters for C1: in the default mode the
search computes less than one exact distance per frame; nearly every candidate is eliminated by
the quant filter, yet all K clusters are fully sorted each frame.

Inclusive attribution (LBR call graphs, % of cycles):

| Component | W1 (2D) | W2 (128D) | W3 (512D tuned) | W4 (image) |
| :--- | :--- | :--- | :--- | :--- |
| Per-frame candidate sort (C1) | **37.1** | **62.4** | n/a (`-gprob`) | **34.5** |
| New-cluster creation, total | 24.1 | 10.7 | 18.1 | 17.2 |
| ↳ column scatter of DCC row (C3) | **21.4** | 6.9 | 6.5 | 3.0 |
| ↳ GEMV computing the new row | 2.2 | 3.5 | 11.2 | 13.7 |
| Dense DCC init in `main` (C2) | 4.0 | 3.6 | 1.9 | 0.7 |
| Kernel (page faults, zeroing) (C2) | 18.5 | 15.3 | 9.4 | 8.0 |
| Quant filter kernels | 8.4 (SQ8) | 3.6 (EQ16) | **41.9** (SQ16) | 22.5 (EQ16) |
| `update_probabilities_and_pruning` | 0.1 | <0.1 | 12.2 | <0.1 |
| `cluster_normalize_probs` (C7) | 0.9 | 1.1 | 4.5 | 1.0 |
| `record_step_assignment` (C16) | <0.1 | <0.1 | 0.3 | 5.0 |
| End-of-run output (C14) | 5.6 | 2.2 | 2.9 | 3.0 |

Corrections to the initial (SIGPROF) pass:

- `main` self-time is only 1-4%, not 16%; the SIGPROF sampler attributed kernel page-fault time
  to `main`. The dense DCC cost (C2) is split between the init loop in `main` (`main.c:274-278`)
  and kernel page-fault handling (8-18%).
- `memmove`, `msort_with_tmp`, `__mempcpy@plt` and `compare_candidates` are 100% inside
  `candidate_sort_descending` (no other significant `memmove` caller).

### 1.3 Thread scaling

| Workload | 1 thread | 4 threads | 12 threads | 24 threads |
| :--- | :--- | :--- | :--- | :--- |
| W2 | 15.3 s | 16.0 s | 16.3 s | 16.8 s (user 32 s, sys 12.6 s) |
| W4 | 2.34 s | 2.09 s | n/a | 2.23 s |

Default invocation (no `OMP_NUM_THREADS`, default `-ncpu 1`, unpinned), refinement pass:

| Workload | Wall | CPU time (user + sys) | Context switches | Threads |
| :--- | :--- | :--- | :--- | :--- |
| W2 | 17.1 s (+12% vs pinned 1 thread) | 41.0 s (28.6 + 14.5) | 980 k | 24 |
| W4 | 2.44 s (+1%) | 7.1 s (3.9 + 3.7) | 586 k | 24 |

In the default W4 run, cycles split as **72.6% kernel** (futex sleep/wake, scheduling), 10.1%
libgomp, 15.7% libgric. The only significant parallel region is `eq16_filter_anchor_matrix`.
Results are bit-identical across thread counts. W3 never crosses the parallel thresholds.

Even with `OMP_NUM_THREADS=1`, `strace` shows ~1 `FUTEX_WAKE_PRIVATE` per frame (19.5k for 20k
frames in W4) from a single thread, consistent with libgomp entering a team-of-one parallel
region every frame.

### 1.4 gric-knn

| Cluster dir | M | 1 thread | IPC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `workspace/128Dtorus.clusterdat` | 3,458 | 0.75 s | 2.14 | matches initial pass |
| W2 output (`-maxcl 20000`) | 9,908 | 1.93 s | 2.19 | bad speculation 19.9% |

- M=3,458: `eq16_dist_asym_cutoff_f32` 29.5%, `knn_search_single_frame` 10.7%,
  `knn_compute_anchor_distance` 8.5%, `cluster_pq_push` 8.0%, `knn_build_cluster_graph` 7.3%,
  `eq16_dist_asym_cutoff_batch_1x8` 7.1%, `gric_get_simd_level` 1.3%.
- M=9,908: **warm-start scan 21.8%** (K4), `knn_build_cluster_graph` 16.3%,
  `knn_parse_dcc_file` 18.1% (393 MB `dcc.bin`): load time is 34% of the run (K8).
- Scaling at M=9,908: 1.93 s (N=1) → 0.49 (4) → 0.32 (8) → 0.26 (16) → 0.26 s (24); P-cores
  only: 0.33 (8), 0.28 (16). Plateau at 16 threads (7.4×); sys time rises from 0.3 to 0.7 s.

The batch CLI path is healthy at moderate M; costs that scale with M (K4, K8) dominate at large M.

### 1.5 Targeted experiments

**Transparent huge pages (C2/C3).** `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (malloc uses
`madvise(MADV_HUGEPAGE)`), no code change, bit-identical assignments:

| Run | Wall | sys | Page faults | dTLB store misses |
| :--- | :--- | :--- | :--- | :--- |
| W1 baseline | 14.65 s | 2.70 s | 2.16 M | 597 M |
| W1 THP | 12.84 s (−12%) | 1.29 s | 12 k | 34 M |
| W2 baseline | 15.25 s | 2.27 s | 1.97 M | 220 M |
| W2 THP | 13.73 s (−10%) | 0.87 s | 11 k | 9.5 M |

With THP, `init_new_cluster_distances` still takes 21.6% of W1 (≈ −10% of its time only): the
column scatter is bound by one cache-line RFO per store, not by the TLB. Kernel time stays at
10.5% (zeroing 2 MB pages of the 8.6 GB matrices). Huge pages are a useful mitigation, but the
layout fixes in C2/C3 remain necessary.

**Build flags (F4).** Min of 3 runs, pinned, bit-identical outputs:

| Build | W3 | W4 | gric-knn (M=3,458) | `sqrt@plt` / `simd@plt` / `framedist*@plt` call sites |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 1.473 s | 2.397 s | 0.715 s | 287 / 86 / 64 |
| `-fno-semantic-interposition -fno-math-errno` | 1.474 s | 2.414 s | 0.709 s | 0 / 90 / 36 |
| above + LTO | 1.464 s | 2.435 s | 0.690 s (−3.5%) | 0 / 0 / 0 |

**Signal delivery (X7).** A 1 kHz `ITIMER_PROF` with an empty handler and no `SA_RESTART`
(LD_PRELOAD) produces bit-identical `frame_membership.bin` for W1 (ASCII), W3 (binary) and W4
(FITS). Default-thread and `perf record` runs of W4 are also bit-identical. X7 is not reproduced.

**Syscalls (C13).** `strace -c` on W4: 243 `openat` in total, so the CLI does not read
`/proc/self/statm` per frame (C13 only applies with `-shm`/FPS). cfitsio issues 56k `read` and
10.8k `lseek` for 20k frames (~45 ms, negligible).

---

## 2. gric-cluster Findings

Paths relative to `src/gric-cluster/` unless otherwise noted.

### C1. Full qsort of all K clusters every frame (Measured, HIGH)

- `steps/compute_priors_and_mixing.c:186` (`candidate_sort_descending`), called at `:481` when
  `gprob_mode` is off (the default).
- `qsort(cands, num_cl, sizeof(Candidate), compare_candidates)`: glibc 2.43 merge sort with an
  indirect comparator and a `__mempcpy` call through the PLT per 16-byte element move.
- **Cost (LBR, inclusive):** 62.4% of cycles in W2 (≈ 316 µs per frame), 37.1% in W1, 34.5% in
  W4; ~50% on a 100k-frame 3D sphere run. It also explains most of the 19-20% front-end-bound
  share in W1/W2 (indirect calls, PLT stubs).
- **Why it is avoidable:**
  - The only consumer (`select_next_measurement_target.c:866-878`) walks `probsortedclindex`
    lazily from the top and skips entries with `clmembflag == 0`. The search measures fewer than
    one exact distance per frame in W1/W2/W4 (Section 1.2), so almost all of the sorted order is
    never read.
  - Per frame, probabilities change by one cluster's `+= deltaprob` plus order-preserving
    normalization (`record_step_assignment.c:118-133` is affine).
- **Ordering contract:** `compare_candidates` has no tie-break, but glibc's merge sort is stable
  and the input is in id order, so the effective order is (p descending, id ascending). Any
  replacement must reproduce exactly that order to stay bit-identical.
- **Fix:**
  - Replace the sort with a lazy iterator: O(K) heapify of (p, id) with an inlined comparator
    (p descending, id ascending), then pop on demand in `select_next_measurement_target`,
    skipping pruned entries as today. Expected cost: O(K + m log K) per frame instead of
    O(K log K) with indirect calls.
  - Longer term: maintain `probsortedclindex` incrementally (bubble the assigned cluster up;
    insert new clusters at their position), falling back to the heap when `-tm` or prediction
    reorders many entries.
  - The entropy selector (`select_next_measurement_target.c`, `_entropy`) also fully qsorts
    `prob_scores` while using only the top `2*limit`; use nth_element or a bounded heap.

### C2. Dense DCC storage sized by maxcl², not by actual cluster count (Measured, HIGH)

- `core/main.c:238-283` allocates `dcc_min`, `dcc_max` (double), `dcc_measured`, `dcc_sq16`
  (19 B per pair × maxcl²) and initializes them serially.
- **Cost:** same 9,908-cluster W2 run:

  | maxcl | Wall | RSS | sys |
  | :--- | :--- | :--- | :--- |
  | 10,000 | 13.2 s | 2.3 GB | 0.8 s |
  | 20,000 | 15.3 s | 7.9 GB | 2.3 s |
  | 40,000 | 22.9 s | 30 GB | 8.0 s |

- **Attribution (perf):** page-fault samples are 86% in `main` (the init loop) and 13% in
  `write_dcc_results`. The init loop (`main.c:274-278`) is scalar: four interleaved stores per
  pair that GCC does not vectorize (possible aliasing between the four arrays). Kernel page-fault
  handling is 15-18% of cycles in W1/W2.
- THP cuts page faults 170× and wall time 10-12% (Section 1.5) as a no-code mitigation.
- **Redundancy:** in dense mode `dcc_max` is a copy of `dcc_min` and `dcc_measured` is all ones
  (`steps/handle_new_cluster_creation.c:206-208`).
- **Fix:**
  - Grow on demand: allocate the capacity in chunks (e.g. 1024 clusters) with row-major
    triangular or chunked square layout, or `mmap` with `MAP_NORESERVE` and let pages fault in
    lazily (no explicit initialization of untouched rows).
  - In dense mode, drop `dcc_max` / `dcc_measured` and store `dcc` as float (keep double only
    under `-double`).
  - Interim: `madvise(MADV_HUGEPAGE)` on the four DCC arrays (allocate them 2 MB-aligned).

### C3. New-cluster DCC writes are column-strided (Measured, MED-HIGH)

- `steps/handle_new_cluster_creation.c:229-236` (`init_new_cluster_distances`) writes
  `k*N + i` across four arrays for every existing k.
- **Cost:** 21.4% of W1 cycles, 6.5-6.9% in W2/W3. `perf annotate`: ~97% of the function's
  samples are on the four scatter stores of this loop; the GEMV computing the row
  (`cluster_gemm_dist_float`) is only 2.2% of W1.
- **Counter evidence:** 99.2% of all STLB store misses (PEBS) are in this function. The count
  matches the model of one miss per store: 4 arrays × K²/2 = 572 M predicted for W1, 597 M
  measured (W2: 196 M predicted, 219 M measured). Every store has a row stride ≥ 20 KB, so each
  one touches a new page and a new cache line.
- With THP the TLB misses disappear but the function is only ~10% faster: the remaining cost is
  one cache-line read-for-ownership per store. Only a layout change fixes it.
- **Fix:** store the symmetric matrix so that the new cluster's distances form a contiguous row
  (lower-triangular row `i` holds `d(i, 0..i-1)`), with an accessor `dcc(i,j)` that swaps
  indices. Combined with C2, this also makes allocation growth trivial.
- **Accuracy note:** the new-cluster row is computed by a float norm-expansion GEMV
  (`math/cluster_gemm_dist.c:1847+`). When `‖a‖² + ‖b‖² ≫ d²`, cancellation can perturb dcc
  values that feed exact-rlim triangle pruning. Consider computing in double, or with
  centered data, or adding an error margin to the pruning test.

### C4. OpenMP: ncpu=1 not honoured, and parallel regions on tiny work (Measured, HIGH)

- `core/cluster_core.c:908`: `omp_set_num_threads` is called only if `ncpu > 1`, so the
  default `-ncpu 1` runs every parallel region on all hardware threads (24 threads observed).
- **Cost (default invocation):** W2 burns 41 s of CPU for 17.1 s wall (2.7× the single-thread
  CPU time, +12% wall); W4 burns 7.1 s CPU for 2.44 s wall, with 72.6% of cycles in the kernel
  and 10% in libgomp, 586k context switches. In an FPS daemon this steals cores from the rest
  of the real-time pipeline.
- Fine-grained parallel regions:
  - `steps/select_next_measurement_target.c:378`: unconditional `omp parallel for` on every
    measurement, with `atomic read` + `critical`, for ~2-15 targets.
  - `select_next_measurement_target.c:247`: `if(M>=16)`.
  - `src/shared/quant/eq16_filter.c:373,632`: `if(num_blocks>=16)` (K≥128 regardless of D);
    at D≤64 the work is < 1 µs, below fork/join cost. This is the region driving W4's overhead.
  - `steps/handle_new_cluster_creation.c:293,320`: `if(b_count>=4)`.
- **Fix:**
  - Always call `omp_set_num_threads(config->optim.ncpu)` (including 1).
  - Replace count-based gates with work-based gates, e.g. `if(K * D >= OMP_MIN_WORK)` with
    `OMP_MIN_WORK` around 2e5 element-ops (tune empirically).
  - Remove the per-measurement parallel region at `:378` (serial is faster for ≤ 16 targets).
  - Consider `OMP_WAIT_POLICY=passive` defaults for the daemons to avoid spinning cores.
  - On this hybrid CPU, default to P-cores only for parallel regions (gric-knn plateaus at 16
    threads; Section 1.4).

### C5. Cluster removal at capacity is O(N²) to O(K³) (Verified, HIGH for streaming at maxcl)

- `core/cluster_mgmt.c:283-330` (`remove_cluster`, DISCARD/MERGE strategies):
  - memmove-shifts rows and columns of `dcc_min`, `dcc_max`, `dcc_measured`, `dcc_sq16`,
    `transition_matrix` (~27 B per pair, ~27 MB at N=1000);
  - memmoves anchor matrices and rebuilds three interleaved matrices (`:~160-210`);
  - rewrites all past assignments, O(frames) (`:~385`);
  - calls `recompute_consistency_mask` (`:411`), which clears N³/8 bytes (125 MB at N=1000)
    and does O(K³) work (`steps/update_consistency_mask.c:58-64`).
- MERGE adds an O(K²) dcc scan (`steps/handle_new_cluster_creation.c:708`).
- Once maxcl is reached in a stream, every new cluster triggers this.
- **Fix:**
  - Swap-remove: move the last cluster into the freed slot; copy one row and one column, O(K).
  - Keep external cluster IDs stable via an indirection table (slot ↔ id) instead of rewriting
    history.
  - Update only the moved slot's mask row/column using the existing
    `update_consistency_mask_for_new_cluster`.

### C6. Consistency mask O(N³) memory, O(K²) per new cluster (Verified, HIGH at large K, entropy mode)

- Allocated as `cluster_pairs * words` (`core/main.c:239,302`), `words = (N+63)/64` with N =
  maxnbclust, so popcount/hypothesis loops scan N bits even when K ≪ N.
- Each new cluster does three O(K²) passes; step 4 (`steps/update_consistency_mask.c:264-300`)
  does a read-modify-write into a different cache line per (i,j).
- **Fix:** size loops by `num_clusters`; store the mask transposed so the new cluster's bits are
  contiguous; replace the branchy bit-set in `update_consistency_mask_range` with
  compare + movemask.

### C7. O(K) scans per measurement instead of per frame (Verified/Measured, MED-HIGH)

- `steps/measure_distance_to_cluster.c:38-46`: counts pruned clusters over all K on every
  distance for telemetry (always on since `max_steps_recorded = maxnbclust`). Use
  `K - num_active_clusters`.
- `core/cluster_step.c:314`: `cluster_normalize_probs` over all K on every measurement that
  needs a prune update (4.5% of W3). The sum loop (`compute_priors_and_mixing.c:48`, 59% of the
  function) is a single FP-add dependency chain (latency-bound, not vectorized without
  reassociation). Normalize once per frame, or carry the normalization factor lazily.
- `steps/update_probabilities_and_pruning.c` (te5 block): `active_before` computed over K
  even with trace off.
- Entropy selector: ~6 passes over K (active count, argmax, H, active_mask, active_indices,
  prob_scores) per measurement. Drive these from `active_clusters`.

### C8. TE5 pruning iterates all K and never updates the active list (Verified, MED-HIGH, also correctness)

- `math/cluster_prune.c:209-211`: loops over all (p,q) measurement pairs and all clusters
  0..K-1 every measurement: O(m²·K) per measurement, O(m³·K) per frame.
- It does not update `active_clusters` or `entropy_p_current`, so later loops visit dead
  entries and renormalization (tail of `update_probabilities_and_pruning.c`) includes
  probability mass from pruned clusters.
- **Fix:** iterate the active list with swap-remove (as te4 already does); cap pairs; update
  probabilities consistently. Add a regression test comparing te5 assignments before/after.

### C9. Fixed per-measurement bookkeeping (Verified/Measured, LOW-MED at D ≤ 32)

- ~6-8 `clock_gettime` calls per measurement (`core/cluster_step.c:425-470`); vDSO is 2.0% of
  W3.
- Three `omp atomic` increments in `get_dist_cutoff` (`core/cluster_core.c:70-88`) and one on
  `cluster_query_counts`, even in serial code.
- `gric_get_simd_level` called via PLT twice per sq8 distance (347M calls in W1);
  `framedistance.c:170,350` call it per distance. Measured: 1.2% of W1 (function + PLT stub),
  1.3% of gric-knn.
- A D=16 distance is ~10 ns, so this overhead dominates at low dimension.
- **Fix:** timing behind a profiling flag (or sample every N frames); plain increments outside
  parallel regions; resolve SIMD level once at startup into function pointers.

### C10. Quant filter not batched after prediction miss (Verified, MED)

- `core/cluster_quant_filter.c` (`cluster_quant_filter_subsequent`, ~600-700) calls scalar
  per-anchor kernels instead of the interleaved batch filters.
- SQ8 (default for D<32) is never batched; `sq8_compute_lower_bound` takes a double `sqrt` per
  candidate (`src/shared/quant/scalar_quant.c:991`); 5.2% in W1 (+3.2% `sq8_dist_squared_u8`).
  Compare SSD to a squared threshold instead.
- `fast_eq16` / `fast_sq16` paths require `gprob_mode` (off by default).

### C11. Geometric probability update chases pointers (Verified/Measured, MED with -gprob)

- `steps/update_geometric_probabilities.c`: per failed measurement, walks up to
  `max_gprob_visitors` (default 1000) visitors, each a random `frame_infos[k]` load plus a
  linear search of `cluster_indices`: 2-3 cache misses per visitor. 3.1% of W3 (with
  `-maxvis 20`).
- **Fix:** store `(frame, dist, assignment)` inline in `VisitorList` at insertion
  (`core/cluster_mgmt.c:33`).

### C12. Sparse-DCC refinement rescans on every new cluster (Verified, LOW-MED)

- `core/cluster_bounds.c` (`refine_sparse_bounds`) rebuilds its queue with an O(K²) scan
  whenever `num_clusters` changes; uses a 16 KB VLA and insertion into a top-1024.
- `update_dcc_bounds` writes four strided columns per k (same pattern as C3).
- **Fix:** incremental queue update with the new row only.

### C13. /proc/self/statm read every frame (Verified, LOW-MED, `-shm` only)

- `core/cluster_shm.c:27,169` (`get_current_rss_kb`): fopen + fscanf + fclose per frame from
  `gric_shm_update` (measured 3.2 µs and 3 syscalls; also takes the mm lock).
- Not active in plain CLI runs (confirmed with `strace`, Section 1.5); affects `-shm` runs and
  the FPS daemons (F3).
- **Fix:** sample every ≥100 ms, or keep the fd open and `pread`.

### C14. End-of-run output O(K·N) with per-value formatting (Verified/Measured, MED for large N/K)

- `io/cluster_io_results.c:640-662` (PNG), `:729-760` (ASCII), `:871` (FITS): for each cluster,
  scan all N assignments (10⁹ comparisons at K=1000, N=10⁶), re-read each frame via
  `getframe_at` (re-parsing ASCII input), write with one `fprintf("%f ")` per value.
- `write_clustered_output_file` (`:1091-1120`) does the same for all N frames.
- Measured: `write_results` is 2.2-5.6% of cycles in W1-W4 (0.35-0.8 s); in W1 most of it is
  `write_dcc_results` (3.2%), which also takes 13% of the run's page faults.
- **Fix:** bucket frame indices per cluster once (counting sort); fast float formatter or
  binary output; process clusters in parallel.

### C15. SQ16/EQ16 batch filter: short accumulator chains and L3 streaming (Measured, MED-HIGH for high D)

New from the refinement pass.

- `src/shared/quant/scalar_quant_filter.c:1666-1740` (`sq16_filter_anchor_matrix_avx_vnni`):
  41.9% of W3; the EQ16 analogue is 22.5% of W4.
- `perf annotate`: hot lines are the `vpdpwssd` accumulations (`:1728`, `:1736`) and the loads
  folded into `vpsubw` (31%), i.e. the main loop beyond the dim-4 checkpoint, not the early
  abort. Only 4 independent accumulators: with a 5-cycle `vpdpwssd` latency this caps the loop
  at ~0.8 VNNI ops/cycle against a throughput of 2.
- TopdownL2 for W3: 28% core-bound, 24% memory-bound. 65% of all L2-miss loads in W3 are in
  this kernel and they hit L3 (L3 misses are negligible): the K×D×2 B matrix (3.2 MB at
  K=3,147, D=512) does not fit in L2 and is re-streamed from L3 every frame. The interleaved
  block layout puts consecutive 8-anchor blocks 8 KB apart, so early-aborted blocks defeat the
  hardware prefetcher.
- **Fix:**
  - 8 accumulators (integer sums are exact, so results stay bit-identical).
  - Split the matrix into a dimension prefix (e.g. first 32 dims of all anchors, 200 KB at
    K=3,147, L2-resident) and a remainder, so blocks that abort at the first checkpoints never
    touch the remainder.

### C16. Prediction-mode probability update walks the Cluster array twice per frame (Measured, LOW-MED)

New from the refinement pass.

- `steps/record_step_assignment.c:118-133`: with `pred_mode` on (W4 uses `binary`), two passes
  over `state->clusters[i].prob` per frame. `Cluster` is 112 B (AoS), so each iteration loads a
  separate cache line. 5.0% of W4; the annotated lines are `:122` (sum) and `:131` (scale).
- **Fix:** keep probabilities in a contiguous array (SoA), or apply the affine update lazily
  (global scale/offset, update only the assigned cluster). Note the lazy form changes rounding
  and must be validated against the bit-identity rule.

---

## 3. gric-knn Findings

Paths relative to `src/`.

### K1. Candidates re-read / ASCII re-parsed per query when the dataset is not in RAM (Verified, HIGH)

- `gric-knn/search/knn_member_search.h:118` (`knn_resolve_candidate_data`) falls back to
  `knn_reader_read_frame` (`gric-knn/io/knn_reader.c:696`) when `model->dataset_buffer` is
  NULL: memcpy, `fseeko`+`fread`, or full ASCII row parse (`:716-724`) per candidate.
- `dataset_buffer` is NULL when the dataset exceeds a hard 2 GB cap
  (`gric-knn/cache/knn_cache.c:1706`), with `--no-cache`, or when `memory_data` is caller
  supplied (e.g. wasm, early return at `knn_cache.c:1697`; IVF layout is then never built).
- **FPS daemon:** `gric-fps/gric_knn_fps_common.c:168` caches the dataset, then `:187` opens
  `cand_reader` from the file with `knn_reader_open` instead of `knn_reader_open_memory`. The
  4-dataset demo uses `.ref_data dataset_A.txt` (`scripts/demo_milk_4dataset.milk:49`), so
  every candidate of every frame is re-parsed from text.
- **Fix:** use `knn_reader_open_memory(&cand_reader, knn_model.dataset_buffer, ...)` as
  `gric-knn/core/knn_engine.c:171` already does; set `dataset_buffer = memory_data` and build
  IVF when memory data is passed; replace the 2 GB cap with an available-RAM check.

### K2. FastScan runs scalar until the heap holds k items (Verified, HIGH)

- While the heap fills, tau is 1e30 and `compute_*_cutoff_thresh` returns `UINT64_MAX`
  (`gric-knn/pruning/knn_pruning.c:285,318,356`). The SQ16/EQ16/RQ8 32× kernels then fall back
  to a scalar 32×D loop with 64-bit math (`shared/quant/scalar_quant_fastscan.c:359-362,
  539-542`; `shared/quant/residual_quant.h:658-661`), which can only return "all pass".
- **Fix:** `if (cutoff == UINT64_MAX) return 0xFFFFFFFF;` before the scalar fallback; use
  64-bit vector accumulators for cutoffs in (INT32_MAX, UINT32_MAX].

### K3. Cross-dataset / streaming path: exact distance per candidate, with a copy (Verified, HIGH for FPS/cross)

- `gric-knn/cross/knn_cross_eval.c:186,473` call `compute_euclidean_distance` with no cutoff
  and no batching; none of the 13 call sites in `cross/*.c` use the cutoff variant. The memo
  (`rep_dists`) stores exact distances, which blocks a cutoff.
- Candidates are copied into `cand_buffer` even when in memory: `knn_cross_route.c:340, 613,
  901, 1204, 1381`; `knn_cross_dataset.c:231, 264, 293`.
- `knn_cross_eval.c:298-303` computes the exact anchor distance for every cluster that survives
  the pivot test before sorting: O(M·D) per query.
- **Fix:** point directly at in-memory data (use `ivf_vectors` where available); batched cutoff
  kernels, memoize only non-cut distances; compute anchor distances lazily or SQ16-first.

### K4. Per-query cluster scans that could be precomputed (Measured, HIGH at large M)

- Warm start (`gric-knn/search/knn_cluster_search.c:221-308`) scans all M clusters per query for
  the 8 nearest by DCC; depends only on `home_cluster_id`. Precompute per cluster.
- **Measured:** 0.6% at M=3,458 but **21.8%** at M=9,908 (N·M scaling). `perf annotate`: 69% of
  its samples are on `:227`, the `model->clusters[c].num_members == 0` test, a strided load
  from the cluster AoS for every c.
- Linear inter-cluster path (no cluster graph):
  - lower-bound refresh at `:1079-1092` loops over all remaining candidate clusters after
    each anchor with random reads into the M×M double `dcc_matrix`: O(Mc²) per query;
  - can't `break`, only `continue`;
  - `is_cluster_pruned_by_pivots` (`:1130`) runs after the D-dim anchor distance it doesn't need.
- **Fix:** precomputed per-cluster warm-start lists (at minimum, a contiguous `num_members` or
  non-empty bitmap); pivot test before anchor distance; min-heap of candidates with re-push on
  bound tightening.

### K5. SIMD kernel inefficiencies (Verified, MED-HIGH)

- AVX-512 SQ16/RQ8 kernels widen to 32-bit and use `mullo_epi32`, ~3-4× more µops per dim than
  the AVX2 `madd_epi16` version (`shared/quant/residual_quant.h:857-891`,
  `shared/quant/scalar_quant_fastscan.c:551-561`). Port the madd approach (`_mm512_madd_epi16`).
  (Not measurable on this host: AVX-512 is disabled on Alder Lake.)
- RaBitQ discards half of each shuffle result (`shared/quant/rabit_quant.c:1036-1049`, 512-bit
  `:927-940`); scalar double epilogue over AoS (`:1110-1125`).
- PQ AVX-512 kernel has no early abort (`shared/quant/product_quant.h:392-414`).
- Sparse modes: scalar block transpose on SQ16 LRU miss (`gric-knn/search/knn_member_quant.c:
  1395-1416`); sparse SDC scores one candidate at a time without prefetch (`:714-741,
  1119-1150`).

### K6. Locking and heap layout (Verified, MED; includes a data race)

- `bucket_locks` always initialized (`gric-knn/core/knn_engine.c:366`); every push into a
  query's own heap takes an `omp_lock` (`gric-knn/pruning/knn_pruning.c:979`) even with
  `use_reciprocal=0`. Lock only in reciprocal mode.
- **Race:** `knn_heap_push` without lock in warm start (`knn_cluster_search.c:196`) while other
  threads push under lock; `tau`/`count` read unlocked during concurrent `memmove`.
- `KnnMaxHeap` always holds 64 slots (`gric-knn/core/knn_defs.h:84-85`), ~512 B per query
  regardless of k, for all N queries. Size by k.
- k > 64: binary-heap fallback scans all entries for duplicates per push (`core/knn_heap.c`
  after `:466`); rely on visited tags.

### K7. GPU paths (Verified, MED)

- Every search re-packs and re-uploads the full IVF index and anchors and creates a cuBLAS
  handle (`gric-knn/gpu/knn_cuda_ivf.cu:1001-1046`; brute force `knn_cuda.cu:431, 446-540`).
  Keep them on `KnnModel`.
- IVF loop uses `malloc` host buffers and blocking `cudaMemcpy` per batch
  (`knn_cuda_ivf.cu:1093,1144,1204`); reuse the pinned two-stream scheme from
  `knn_cuda.cu:625-761`.
- `--gpu-micro-batch` default 64 (`gric-knn/cli/knn_cli.c:1265`) underutilizes the GPU.
- One warp per member (`knn_cuda_ivf.cu:548-654`); score 32 members per warp.
- **Correctness:** IVF vectors zero-filled when `dataset_buffer` is NULL
  (`gpu/cuda_ivf_index.cu:~131`) → silently wrong results.

### K8. Load time and misc (Measured, MED at large M)

- DCC matrix computed single-threaded O(M²·D) (`gric-knn/io/knn_loader.c:1091-1104`).
- **Measured at M=9,908:** `knn_parse_dcc_file` 18.1% (393 MB `dcc.bin`) and
  `knn_build_cluster_graph` 16.3%: 34% of a single-threaded run is model loading.
- `int i*M+j` overflows for M > 46340; search code uses `home_cluster_id * M` as int.
- ASCII anchors parsed with one `fscanf` per element (`knn_loader.c:228-241, 545-558`).
- Per-thread visited tags N_cand×4 B (+N_cand×8 B in memo mode), randomly accessed: TLB pressure
  at large N.

---

## 4. I/O, FPS Daemons, libgric, GPU Clustering, Build

### F1. libgric and cluster FPS are double-only (Verified, HIGH/MED)

- `src/libgric/gric_api.c:392-410`: `gric_cluster_feed_frame` takes `const double*` and forces
  `frame.is_double = 1`.
- `src/gric-fps/gric_fps_common.c:318-327` converts every float32 stream frame to double, so
  `fps_use_double=0` has no effect, anchors are double, and the float batch path
  (`anchor_matrix_float`, gated on `!current_frame->is_double` in
  `steps/cluster_step_prep.c:388`) is never used: 2× bandwidth, half SIMD width.
- **Fix:** add `gric_cluster_feed_frame_f32(const float*)`; set `is_double` from
  `cfg->use_double`; pass the shm slice pointer directly (no copy).

### F2. Reconstruction FPS formats text and flushes per frame (Verified + microbenchmark, HIGH when out_file set)

- `src/gric-fps/gric_recon_fps_common.c:592-600`: `fprintf("%.6f ")` per dim + `fflush` every
  frame; ~180 µs/frame at 1024 dims (measured). The demo sets `out_file`
  (`scripts/demo_milk_4dataset.sh:280`).
- **Fix:** binary `fwrite` into a large `setvbuf` buffer, flush on a timer/at exit; or leave
  writing to a downstream consumer of the output stream.

### F3. /proc/self/statm every frame in all daemons (Verified + microbenchmark, MED)

- `gric_fps_common.c:619`, `gric_knn_fps_common.c:676`, `gric_recon_fps_common.c:714`
  (plus C13). 3.2 µs + 3 syscalls + mm lock per frame; significant for 3D/8D streams where
  compute is a few µs.
- **Fix:** refresh only when the existing ≥0.1 s / 0.5 s FPS timer fires.

### F4. Build flags block cross-TU inlining (Measured, LOW; downgraded from MED)

- `CMakeLists.txt:6,15,232,279`: everything is `-fPIC` into shared `libgric.so`, default
  semantic interposition, no hidden visibility, no LTO. In the HEAD build: 64
  `framedist*@plt` call sites, 86 `gric_get_simd_level@plt`, 287 `sqrt@plt`/`sqrtf@plt`.
- **Measured (Section 1.5):** `-fno-semantic-interposition -fno-math-errno` removes all sqrt
  PLT calls but changes W3/W4/gric-knn by < 1%; adding LTO removes all listed PLT calls and
  gains 3.5% on gric-knn, nothing on W3/W4. Outputs are bit-identical. The hot paths are
  dominated by C1/C3/C15, not by call overhead.
- **Fix (hygiene, low priority):** `-fno-math-errno`; `-fvisibility=hidden` + explicit exports
  or LTO; re-evaluate after C1/C3 when per-call overhead becomes a larger share. Optional PGO.

### F5. GPU clustering pass 1 synchronous and double-copying (Verified, MED for CUDA builds)

- `src/gpu/cuda_anchor_store.cu:304` pageable `h_frames_float`; `:478-491` second copy then
  blocking `cudaMemcpy` on the default stream; `:521-523` blocking copies back.
- `src/gric-cluster/gpu/cluster_cuda.cu:1090-1121`: no read/compute overlap.
- `cuda_anchor_store.cu:438-440`: two synchronous memcpys per new anchor (one of 4 bytes).
- **Fix:** `cudaMallocHost`, two-stream double buffering (as pass 2 does at
  `cluster_cuda.cu:532-690`), fill pinned buffer directly, batch anchor norms on device.

### F6. Cluster FPS publishes twice per frame, with a race (Verified, partly unconfirmed, MED/LOW)

- `gric_fps_common.c:384-387` increments `cnt0` and `sempost`s; `gric_fps_main.c:156` then calls
  `processinfo_update_output_stream`, which likely does the same (not verified against milk
  source). Downstream waiters may wake twice.
- `gric_fps_main.c:153` writes latency into the buffer after publication (race).
- **Fix:** publish exactly once after all fields are written.

### F7. Stream producer flow-control polling (Verified, LOW/MED)

- `src/tools/converters/txt2stream.c:456-458`: `usleep(10)` loop waiting on `cnt2`; real sleep
  ~50-60 µs with default timer slack → caps flow-controlled pipelines near 15 kHz. Rate pacing
  (`:466-472`) drifts (relative timestamps).
- **Fix:** brief spin then semaphore wait; pace with `clock_nanosleep(TIMER_ABSTIME)`.

### F8. libgric auto-grow stalls (Verified, LOW/MED)

- `gric_api.c:84-240`: with `maxnbclust=0` each doubling reallocs/copies several N×N matrices;
  consistency mask N³/64 words (`:338`), ~1 GB at N=2048, inside `feed_frame`.
- **Fix:** pre-size, or adopt the C2/C6 growable layouts.

### F9. Stream ingest copies every frame (Verified, LOW)

- `src/gric-cluster/io/frameread_stream.c:183-200` copies each frame out of shm; borrow the
  slice in place for 3D ring buffers (`is_borrowed` exists).
- Semaphore wait (`:95-120`) never drains posts accumulated while lagging; flush with
  `sem_trywait` once caught up.

### Cleanup

- `src/gric-cluster/io/frameread.c:~600-690`: duplicated, unreachable mmap fast path in
  `getframe_at`.

### Checked and fine

- Ingestion: binary input is zero-copy mmap; ASCII is mmap + custom parser; most result writers
  use a 64 KB `setvbuf`. FITS input via cfitsio costs ~45 ms of syscalls for 20k frames.
- Full-D distance kernels (`math/framedistance_cutoff.c`): 4 accumulators, FMA, periodic
  horizontal sum.
- Frame structs pooled; prediction history scan cheap.
- Branch prediction: bad speculation is 4-5% in all clustering workloads.
- Determinism: assignments are bit-identical across thread counts, under `perf`, and under
  1 kHz SIGPROF delivery (W1, W3, W4).
- gric-knn: per-thread reusable scratch (no per-query malloc), single OpenMP region over queries
  (`schedule(guided,16)`), epoch-based visited tags, binary heap on the cluster-graph path,
  cached quant cutoffs.

---

## 5. Correctness Issues Found Along the Way

| ID | Issue | Location |
| :--- | :--- | :--- |
| X1 | TE5 doesn't update active list; renormalization includes pruned mass | `math/cluster_prune.c:209` (C8) |
| X2 | Unlocked heap push in k-NN warm start vs locked pushes elsewhere | `gric-knn/search/knn_cluster_search.c:196` (K6) |
| X3 | GPU IVF vectors zero-filled when `dataset_buffer` is NULL | `gpu/cuda_ivf_index.cu:~131` (K7) |
| X4 | `int` index overflow for M > 46340 | `gric-knn/io/knn_loader.c:1091`, search code (K8) |
| X5 | Latency field written after publish | `gric-fps/gric_fps_main.c:153` (F6) |
| X6 | Float GEMV cancellation may perturb dcc used in exact pruning | `math/cluster_gemm_dist.c:1847+` (C3) |
| X7 | ~~Run under SIGPROF reached maxcl at a different frame~~: **not reproduced** (Section 1.5); likely a difference in run configuration in the initial pass | closed unless seen again |
| X8 | `int` products `new_cl * N` / `r * N` in the sparse-DCC path overflow for maxcl > 46340 (the dense path casts to `size_t`) | `steps/handle_new_cluster_creation.c:52-115` |
| X10 | **Hang**: `-entropy` with a quant prefilter (SQ8 default, SQ16, EQ16) loops forever. The per-candidate quant prune clears `clmembflag[cj]` but not `entropy_p_current[cj]`; the entropy argmax scans inactive clusters and returns `cj` again. Reproduced with 20 frames | `core/cluster_quant_filter.c:136,155,174`, `steps/select_next_measurement_target.c:526` |
| X11 | gric-knn reports `Completed successfully` and exits 0 after `Failed to write results` | `gric-knn` main / `io/knn_writer.c` |
| X12 | `gric-knn -nthreads 4`: indices reproducible, ~0.4% of distances differ by 1 ULP between runs | k-NN search (with X2) |

---

## 6. Fix Plan

Guiding rules for every phase:

- **Bit-identical assignments** for default configs unless the change is an intended
  correctness fix. Before each phase, record cluster assignments and `-evals` stats for the
  benchmark set (`benchmarks/default_tests.txt` via `gric-benchmark`) and diff after.
- **Measure before/after** with the W1-W4 workloads (Section 1.2) plus `gric-knn` on 128Dtorus
  at M=3,458 and M=9,908, single-threaded pinned to a P-core and at default threads.
- One PR per item or small group; keep tests green (`ctest`).

### Phase 0: Baseline and harness (prerequisite, done)

1. `scripts/perf_baseline.sh run <outdir> [--bin <builddir>] [--reps 3] [--only W1,..]` runs
   W1-W4, K1, K2 pinned to CPU 4 with `OMP_NUM_THREADS=1` (plus D2/D4 at default threads),
   records wall/user/sys/RSS and `perf stat` counters (IPC, page faults, dTLB store misses,
   TopdownL1) when available, and hashes the outputs. `compare <A> <B>` prints a before/after
   table.
2. `scripts/regression_check.sh record <refdir>` / `check <refdir>`: 23 gric-cluster and 5
   gric-knn configurations on small generated datasets (stored in `<refdir>/data`), compared
   by output hashes and `STATS_*` counters. Takes 12 s; deterministic over repeated runs.
   References are machine-specific (`-march=native`) and are kept outside the repository.
3. ~~Investigate X7~~ (not reproduced; signal delivery does not change results).

Typical use for a change:

```bash
scripts/regression_check.sh record /tmp/ref --bin build-before   # once, before the change
scripts/perf_baseline.sh run /tmp/pb-before --bin build-before --reps 3
# ... apply the change, rebuild into build-after ...
scripts/regression_check.sh check /tmp/ref --bin build-after
scripts/perf_baseline.sh run /tmp/pb-after --bin build-after --reps 3
scripts/perf_baseline.sh compare /tmp/pb-before /tmp/pb-after
```

### Phase 1: Quick, low-risk wins (expected ~2× in default mode)

| Item | Change | Risk | Expected effect (measured basis) |
| :--- | :--- | :--- | :--- |
| C1 | Lazy heap iterator (p desc, id asc) instead of full qsort; nth_element in entropy selector | Low (order contract above) | Removes most of 62% (W2), 37% (W1), 35% (W4) |
| C4 | Always set thread count; work-based OMP gates; drop per-measurement parallel region | Low | Default runs: −2.7× CPU (W2), no kernel futex storm; −12% wall (W2) |
| C2 interim | `madvise(MADV_HUGEPAGE)` on 2 MB-aligned DCC arrays | Very low | −10-12% wall on W1/W2 (measured via glibc tunable) |
| K1 | FPS uses `knn_reader_open_memory` | Very low | Eliminates per-candidate text parsing in the demo |
| K2 | Early return on `UINT64_MAX` cutoff in FastScan | Very low | Removes scalar fallback while heaps fill |
| K4 (part) | Precomputed per-cluster warm-start list | Low | −22% gric-knn at M≈10k |
| F3/C13 | RSS sampling on timer | Very low | −3 µs and 3 syscalls per frame (daemons, `-shm`) |
| C9 | Hoist SIMD-level dispatch; timing behind flag; plain counters in serial code | Low | ~1-2% at D ≤ 32 |

### Phase 2: Memory layout of the DCC structures (expected large gains at high maxcl)

| Item | Change | Risk |
| :--- | :--- | :--- |
| C2 + C3 | Introduce a DCC accessor API (`dcc_get(i,j)`, `dcc_row(i)`); switch to growable lower-triangular row storage; drop redundant dense-mode matrices; float storage | Medium: touches many call sites; do the accessor refactor first as a no-op PR, then change the layout behind it. Expected: removes 21% (W1) scatter + 15-18% kernel page-fault time + RSS ∝ K² instead of maxcl² |
| C12 | Incremental sparse-bound queue | Low-medium |
| F8 | libgric auto-grow reuses the growable layout | Low after C2 |

### Phase 3: Algorithmic hot spots in optional modes

| Item | Change | Risk |
| :--- | :--- | :--- |
| C15 | 8 accumulators in SQ16/EQ16 batch filters; L2-resident dimension prefix | Low (accumulators, exact integer sums) / medium (layout) |
| C7 | Drive per-measurement loops from `active_clusters`; normalize once per frame | Low |
| C16 | SoA probability array for the prediction-mode update | Low |
| C8/X1 | TE5 over active list with swap-remove; fix renormalization | Medium (changes results: intended fix, document it) |
| C10 | Batched SQ8 filter; squared-threshold compare | Low-medium |
| C11 | Inline visitor records | Low |
| C5 | Swap-remove cluster removal with stable ID indirection | Medium-high (touches assignment history, outputs) |
| C6 | Mask sized by K, transposed, vectorized updates | Medium |

### Phase 4: Library API, daemons, build

| Item | Change | Risk |
| :--- | :--- | :--- |
| F1 | `gric_cluster_feed_frame_f32`; FPS passes shm slice directly | Low-medium (new API; keep double path) |
| F2 | Binary buffered output in recon FPS | Low (format change: keep ASCII behind an option) |
| F6/X5 | Single publish after all writes | Low (verify against milk `processinfo_update_output_stream`) |
| F7 | Semaphore-based flow control; absolute-time pacing | Low |
| F9 | Borrow shm slices; drain stale semaphore posts | Low-medium |
| C14 | Bucketed output generation, fast formatting, parallel clusters | Low |
| F4 | `-fno-math-errno`, hidden visibility or LTO (hygiene; < 1% measured today, 3.5% on gric-knn) | Low (bit-identical in tests) |

### Phase 5: gric-knn deeper work

| Item | Change | Risk |
| :--- | :--- | :--- |
| K3 | Direct in-memory candidate access; cutoff kernels in cross paths; lazy anchor distances | Medium |
| K4 | Pivot test first; candidate min-heap (warm-start part moved to Phase 1) | Medium |
| K8/X4 | Parallel DCC build; faster `dcc.bin` load (mmap, float); 64-bit indexing; fast ASCII anchor parsing | Low |
| K6/X2 | Locks only in reciprocal mode; fix warm-start race; heap sized by k | Medium |
| K5 | AVX-512 madd kernels; RaBitQ shuffle fix; PQ early abort; sparse-mode SIMD | Medium (verify bit-exactness vs AVX2; needs an AVX-512 host) |

### Phase 6: GPU (CUDA builds only)

| Item | Change | Risk |
| :--- | :--- | :--- |
| K7/X3 | Persistent device index and cuBLAS handle on `KnnModel`; pinned async IVF loop; larger micro-batch; warp-per-32-members kernel; fix zero-fill | Medium |
| F5 | Pinned buffers and two-stream overlap in clustering pass 1; batched anchor norms | Medium |

### Suggested ordering rationale

Phase 1 delivers most of the measured gain for the default configuration with minimal risk: C1
alone accounts for 35-62% of cycles in three of the four workloads, and C4 removes a 2.7× CPU
overhead in default invocations. Phase 2 addresses the memory scaling (C2/C3: up to 40% of W1
including kernel time) and enables simpler implementations of C5, C6 and F8. Build-flag work
(F4) was measured to be negligible and moves to Phase 4. Phases 3-6 target specific modes and
can be scheduled according to which workloads matter most (high-D tuned runs → C15; streaming
at maxcl → C5/C6; real-time demo → F1-F3, K1, K3; large-M k-NN → K4, K8; GPU users → Phase 6).

### Further profiling (not yet done)

- Set `kptr_restrict=0` to resolve kernel symbols and split kernel time between page-fault
  handling, page zeroing, and futex/scheduler work.
- Profile the FPS daemons and the 4-dataset demo pipeline under `perf record -a` to quantify
  F1-F3, F6, K1 and K3 under streaming load.

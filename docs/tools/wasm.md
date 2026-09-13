# WebAssembly Engine & Native C Parity

The **GRIC** software ecosystem provides two runtime execution engines for clustering and
$k$-NN search: the **Native C CLI** binaries (`gric-cluster`, `gric-knn`) and the client-side
**WebAssembly (WASM) Engine** (`gric_cluster.wasm`).

This document details the WebAssembly architecture, explains how algorithmic equivalence is
guaranteed between WASM and native C, and outlines the key technical differences between them.

---

## 1. Overview & Architecture

The WebAssembly engine allows the full GRIC clustering and $k$-NN pipeline to run entirely
client-side inside modern web browsers (Chrome, Firefox, Safari, Edge) without requiring any
server-side compute, Python runtime, or native binary installation.

```
+-------------------------------------------------------------------------------+
|                       Browser Client / Visual Simulator                       |
|                                                                               |
|  [Main Thread]                                                                |
|    - HTML5 / Canvas Rendering (Quad 2D/3D views)                              |
|    - User controls & parameter configuration                                  |
|    - wasm_bridge.js (TypedArray heap marshalling)                             |
|                               |                                               |
|                    Worker RPC | PostMessage                                   |
|                               v                                               |
|  [Web Worker: wasm_worker.js]                                                 |
|    +-----------------------------------------------------------------------+  |
|    | gric_cluster.wasm (Compiled Native C Source)                          |  |
|    |                                                                       |  |
|    |   +-----------------------+      +---------------------------------+  |  |
|    |   | cluster_step.c        |      | knn_engine.c                    |  |  |
|    |   | Steps 1-5 + Pruning   |      | Hierarchical Metric Tree        |  |  |
|    |   +-----------------------+      +---------------------------------+  |  |
|    |   | tile_state.c          |      | framedistance.c                 |  |  |
|    |   | Multi-Tile Axis Dec.  |      | WASM-SIMD128 Vectorized Math    |  |  |
|    |   +-----------------------+      +---------------------------------+  |  |
|    +-----------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

### Shared Source Code Architecture

The WebAssembly module is **not** a JavaScript reimplementation of GRIC. Instead,
`Makefile.wasm` compiles the **exact same C source files** used by the native command-line
binaries via the Emscripten toolchain:

* **Core Clustering Pipeline**:
  - `src/gric-cluster/core/cluster_step.c`
  - `src/gric-cluster/core/cluster_bounds.c`
  - `src/gric-cluster/core/cluster_mgmt.c`
* **Algorithmic Steps (1 through 5)**:
  - `src/gric-cluster/steps/initialize_initial_cluster.c`
  - `src/gric-cluster/steps/compute_priors_and_mixing.c`
  - `src/gric-cluster/steps/select_next_measurement_target.c`
  - `src/gric-cluster/steps/measure_distance_to_cluster.c`
  - `src/gric-cluster/steps/update_probabilities_and_pruning.c`
  - `src/gric-cluster/steps/update_geometric_probabilities.c`
  - `src/gric-cluster/steps/handle_new_cluster_creation.c`
  - `src/gric-cluster/steps/record_step_assignment.c`
  - `src/gric-cluster/steps/update_consistency_mask.c`
* **Multi-Tile Axis Decomposition & Bayesian Fusion**:
  - `src/gric-cluster/core/tile_state.c`
  - `src/gric-cluster/core/tile_map.c`
  - `src/gric-cluster/io/frame_scatter.c`
  - `src/gric-cluster/math/tuple_retrieval.c`
* **Metric Distance & Quantization Kernels**:
  - `src/gric-cluster/math/framedistance.c`
  - `src/gric-cluster/math/cluster_math.c`
  - `src/gric-cluster/math/cluster_prune.c`
  - `src/shared/scalar_quant.c`
  - `src/shared/quant_memo.c`
* **Out-of-Core $k$-NN Search Engine**:
  - `src/gric-knn/knn_engine.c`
  - `src/gric-knn/knn_heap.c`
  - `src/gric-knn/knn_tree.c`
  - `src/gric-knn/knn_reader.c`

A thin wrapper (`src/wasm/gric_wasm_api.c`) exposes Emscripten-exported C functions
(`_wasm_cluster_*`, `_wasm_multitile_*`, `_wasm_knn_*`) that map memory buffers between
JavaScript TypedArrays (`Float64Array`, `Int32Array`) and C pointers.

---

## 2. Algorithmic Equivalence

Because the exact C files are compiled into WebAssembly, the mathematical algorithm executed in
WASM mode is completely identical to the native C engine:

1. **Step 1 (Initialization)**: Anchor initialization and covering radius verification.
2. **Step 2 (Prediction Retrieval)**: Monotonic history tracking and pattern retrieval.
3. **Step 3a (Prior Mixing & Pruning)**: Transition-matrix sequence prior combined with
   empirical cluster frequencies and candidate pruning against `deltaprob`.
4. **Step 3b (Target Selection)**: Identical greedy prior search or entropy-guided mutual
   information maximization, complete with 3-point, 4-point (TE4), and 5-point (TE5)
   triangle inequality lower/upper bounding.
5. **Step 3c (Distance Measurement)**: Full Euclidean metric evaluation with dynamic bounds
   updating, pruning counters, and telemetry tracking.
6. **Step 4 (Cluster Creation)**: Inter-cluster distance matrix (DCC) population, sparse DCC
   relaxation, and 3D consistency bitmask generation.
7. **Step 5 (Assignment & Telemetry)**: State assignment, visitor list updating, and live trace
   event generation.
8. **Multi-Tile Axis Decomposition**: $N$-dimensional coordinates are scattered into 1D tiles,
   clustered independently in Pass 1, and fused via Bayesian joint tuple fusion (Pass 2) with
   incremental Conditional Probability Table (CPT) cross-tile priors.

### Automated Parity Verification

Algorithmic parity is verified by the continuous automated test suite
(`tests/wasm_benchmark_parity.js`). The test runs both engines on identical benchmark
trajectories and verifies zero drift:

| Benchmark | Total Frames | Clusters ($K$) | Assignment Parity | Distance Eval Match |
| :--- | :--- | :--- | :--- | :--- |
| **`2Dspiral`** | 1,000 | 57 | **100.0% match** (1,000/1,000) | Exact parity |
| **`3Dspiral`** | 1,000 | 109 | **100.0% match** (1,000/1,000) | Exact parity |
| **`2Dwalk`** | 1,000 | 9 | **100.0% match** (1,000/1,000) | Exact parity |
| **`3Dtorus`** | 1,000 | 145 | **100.0% match** (1,000/1,000) | Exact parity |

---

## 3. Comparison & Differences: Native C vs. WebAssembly

While the underlying algorithm and source files are shared, operational and hardware environments
differ between a native Linux executable and a WebAssembly sandboxed runtime in the browser.

| Aspect | Native C (`gric-cluster` / `gric-knn`) | WebAssembly (`gric_cluster.wasm`) |
| :--- | :--- | :--- |
| **Precision** | Single (`float32`), `-double` opt-in | Double (`float64`) for JS arrays |
| **Threading** | OpenMP parallel across CPU threads | Single-threaded; concurrency via Worker |
| **SIMD** | x86_64 256-bit AVX2 / FMA intrinsics | 128-bit WASM SIMD (`-msimd128`) auto-vec |
| **$k$-NN Search** | Graph Routing (`-cluster-graph`) | Metric Tree (L1-L3 Annular + Pivot) |
| **2-Hop Injection** | Enabled by default (`-two-hop`) | Disabled in WASM bridge |
| **Quantization** | Optional `-sq8` and `-sq16` pruning | Disabled (runs full float64) |
| **Memoization** | Optional `-memo` hash cache | Disabled |
| **Data I/O** | FITS, ASCII, binary files, SHM, `mmap` | Direct zero-copy memory (`HEAPF64`) |
| **Profiler** | Native `gric-probe` & `gric-dimdensity` | CLI only (via `DesktopBridge`) |

### Detailed Technical Distinctions

#### 1. Floating-Point Precision
* **Native C**: Defaults to single precision (`float32`), reducing cache pressure and doubling
  SIMD throughput on x86_64 AVX2 architectures. Passing `-double` switches the engine to 64-bit
  double precision.
* **WebAssembly**: Forces `use_double = 1`. JavaScript numeric primitives and standard
  `Float64Array` typed arrays use 64-bit IEEE-754 floating-point values. Running the WASM engine
  in double precision eliminates precision conversion overhead when passing coordinate buffers
  between JavaScript and WebAssembly linear memory.

#### 2. Parallelism and Worker Model
* **Native C**: `gric-knn` utilizes OpenMP to process queries concurrently across all available
  CPU threads (`nthreads`). Inter-query reciprocal updates are synchronized via 4,096 bucket
  mutexes.
* **WebAssembly**: Compiled without SharedArrayBuffer / pthreads flags to maximize browser
  compatibility and avoid cross-origin isolation requirements (`COOP`/`COEP`). Concurrency in
  the visual simulator is achieved by running the entire WASM instance inside a dedicated
  `Web Worker` (`wasm_worker.js`), keeping the main rendering thread responsive at 60 FPS.

#### 3. Vector SIMD Acceleration
* **Native C**: Contains explicit AVX2/FMA intrinsic kernels in `framedistance.c` with 4
  independent vector accumulators, unrolling 32 floats (128 bytes) or 16 doubles per loop.
  It also features 1-query vs. 4-anchor multi-vector batch kernels (`framedist_batch_1x4_*`).
* **WebAssembly**: Compiled with `-msimd128`. The Clang LLVM backend automatically vectorizes the
  inner distance calculation loops into 128-bit WebAssembly vector instructions (`v128`),
  delivering near-native vector speed in browser engines supporting WebAssembly SIMD.

#### 4. $k$-NN Search Strategy
* **Native C**: `gric-knn` defaults to **Graph-Guided Cluster Routing** (`use_cluster_graph = 1`),
  where cluster centroids form a $k_{\text{adj}}=48$ proximity graph. Queries navigate the graph
  using a priority queue bounded by `ef_cluster`, combined with 2-Hop candidate injection.
* **WebAssembly**: `wasm_knn_run_search` executes the exact **Hierarchical Metric Tree Search**:
  * *Level 1*: Whole-cluster multi-pivot bounding against active clusters.
  * *Level 2*: Exact query-to-anchor distance evaluation.
  * *Level 3*: Annular shell pruning ($|d(q, a) - r_a| \le \tau$) with sorted member binary search.
  * *Multi-Pivot Bounding*: AESA metric pruning across evaluated anchor pivots.
  * *Reciprocal Push*: Symmetric distance reuse when queries are not past-only or future-only.

---

## 4. Building and Deploying WebAssembly

WebAssembly compilation is managed via `Makefile.wasm`.

### Prerequisites
* **Emscripten SDK** (`emsdk`): `emcc` must be available in your `PATH`, or installed at
  `~/emsdk` (auto-sourced from `~/emsdk/emsdk_env.sh`).

### Build Commands

```bash
# Build and deploy to both site/simulator/wasm and docs/simulator/wasm
make -f Makefile.wasm

# Alternatively, compile using CMake target
mkdir -p build && cd build
cmake ..
make wasm
```

The build produces two artifacts:
* `gric_cluster.js`: Emscripten module loader and runtime environment.
* `gric_cluster.wasm`: Compiled WebAssembly binary (~182 KB).

Both files are mirrored to `docs/simulator/wasm/` (for GitHub Pages documentation hosting) and
`site/simulator/wasm/` (for the native HTTP server).

---

## 5. Automated Verification Suite

The WebAssembly test suite runs in Node.js and verifies all functional layers:

```bash
# Verify basic initialization and single-frame clustering
node tests/wasm_smoke_test.js

# Verify exact 100% mathematical parity against JavaScript reference
node tests/wasm_benchmark_parity.js

# Verify metric-pruned k-NN solver and distance recall
node tests/wasm_knn_test.js

# Verify 2D, 3D multi-tile axis decomposition and CPT priors
node tests/wasm_multitile_test.js

# Verify dynamic geometric memory growth (unlimited cluster mode)
node tests/wasm_growth_test.js

# Verify live trace buffer event recording for the explain view
node tests/wasm_trace_test.js

# Verify high-dimensional raster image clustering (1024-D)
node tests/wasm_image_test.js
```

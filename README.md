<img src="gric.png" alt="GRIC Logo" width="700">

# GRIC: High-Speed Distance-Based Image & Stream Clustering

[![C17](https://img.shields.io/badge/Language-C17-blue.svg)][c17]
[![CMake](https://img.shields.io/badge/Build-CMake-brightgreen.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-indigo.svg)][docs]

[c17]: https://en.wikipedia.org/wiki/C17_(C_standard_revision)
[docs]: https://oguyon.github.io/gric-cluster/

**GRIC** (Geometric Real-Time Image Clustering) is an ultra-fast, distance-based streaming
clustering engine written in optimized C17. It groups data frames (images, camera streams,
sensor logs) into clusters based on Euclidean distance thresholds ($r_{\text{lim}}$), reducing
distance calculations from $O(K)$ to $O(1 \sim 3)$ per frame via active information-theoretic
target selection and multi-point metric space pruning.

---

## 🎬 Video Walkthrough & Interactive Simulator

Experience GRIC through our narrated HD explainer video or explore algorithms in real-time in your
browser:

- 🌐 **[Interactive 2D & 3D Simulator](https://oguyon.github.io/gric-cluster/visual_simulator.html)**
- 📖 **[Read Full MkDocs Documentation](https://oguyon.github.io/gric-cluster/)**
- 📐 **[Visual Guide & Architecture](https://oguyon.github.io/gric-cluster/algorithm/visual_guide/)**
- 🎥 **[Download / Watch Full Narrated HD Video (MP4)](docs/figures/gric_explainer.mp4)**

![GRIC Algorithm Animated Walkthrough](docs/figures/gric_explainer.gif)
*Figure 1: Narrated visual walkthrough demonstrating sequential stream ingestion, boundary anchor
creation, triangle inequality pruning, spiral center Shannon entropy gain, prior layers, and
multi-tile rich joint tuples.*

---

## 🚀 Key Features & Architectural Innovations

### 1. 5-Stage Sequential Pipeline
GRIC clusters incoming frames sequentially in a single pass without storing dense pairwise distance
matrices.

<img src="docs/figures/gric_master_pipeline.svg" alt="GRIC Master Pipeline" width="100%">

1. **Prior Normalization**: Ingests incoming frames from files or shared memory (`ImageStreamIO`)
   and layers Markov transition probabilities (`-tm`) and trajectory predictors (`-pred[len,h,n]`).
2. **Target Selection**: Chooses candidate anchors to evaluate via Greedy priority or expected
   Shannon entropy minimization (`-entropy`).
3. **SIMD Metric Evaluation**: Computes Euclidean distance using AVX2 SIMD intrinsics and OpenMP
   multi-threading (`-ncpu`).
4. **Multi-Point Pruning**: Evaluates triangle inequality and higher-order simplex bounds to
   mathematically eliminate incompatible candidate clusters with **zero distance computations**.
5. **Assignment & Boundary Spawning**: Re-allocates matching frames ($d \le r_{\text{lim}}$) or
   spawns new exemplar cluster anchors on the boundary when all existing cluster matches fail.

---

### 2. Multi-Point Geometric Pruning

<img src="docs/figures/gric_pruning_geometry.svg" alt="GRIC Pruning Geometry" width="100%">

| Mode | CLI Flag | Geometric Principle | Speedup |
| :--- | :--- | :--- | :--- |
| **3-Point** | *Default* | $|d(f, cA) - d(cA, cX)| > r_{\lim}$ | **50-80% drop** |
| **4-Point** | `-te4` | Triangulate 2D baseline offset & height $h_f$ | **+10-20% gain** |
| **5-Point** | `-te5` | Simplex base plane projection & height $h_3$ | **+15-30% gain** |
| **Sparse** | `-sparse_dcc` | Dynamic $[d_{\min}, d_{\max}]$ bounds | **Zero $O(K^2)$ RAM** |

---

### 3. Information-Theoretic Target Selection (`-entropy`)

<img src="docs/figures/gric_target_selection_entropy.svg" alt="Entropy Selection" width="100%">

- **Greedy Mode (Default)**: Tests candidate anchors in descending order of prior probability.
- **Entropy Mode (`-entropy`)**: Schedules the pivot anchor that minimizes expected posterior
  Shannon entropy across candidate true-cluster hypotheses:
  $$\mathbb{E}[H(T)] = \sum_{c_j} P(c_j) \cdot H(T \mid c_j \text{ is true})$$
- **Spiral Center Pivot**: On non-linear manifolds, measuring distance to the center directly
  yields the radius, **unambiguously resolving the exact position in 1 measurement**.

---

### 4. Prior Modeling & Topological Learning

<img src="docs/figures/gric_priors_prediction.svg" alt="Priors and Prediction" width="100%">

- **Markov Transitions (`-tm <coeff>`)**: Learns pairwise transition probabilities over time.
- **Sequence Predictor (`-pred[len,h,n]`)**: Scans historical logs to forecast multi-step paths.
- **Visitor Geometry (`-gprob`)**: Discovers topology by correlating co-measurement visitors.
- **Soft Bayesian Likelihoods (`-soft_bayesian`)**: Gaussian likelihood fading on surviving
  candidates for noisy sensor streams.

---

### 5. Multi-Tile Parallelism & Rich Joint Tuples (`-tiles`, `-jtf`)

<img src="docs/figures/gric_tiling_jtf.svg" alt="Multi-Tile and JTF" width="100%">

- **Spatial Tiling (`-tiles NxM`)**: Partitions high-dimension images into sub-tiles, maximizing CPU
  L1/L2 cache locality and parallelizing across OpenMP threads (`-ncpu`).
- **Rich Joint Tuples**: Generates state tuples (e.g., $(c_0, c_3, c_2, c_1)$) capturing spatial
  cross-entropy across regions.
- **Joint Trajectory Fusion (`-jtf`)**: Pass 2 lookup matching against historical tuple trajectories
  resolves seam flickering while strictly verifying $d \le r_{\text{lim}}$.

---

### 6. Lattice Quantization, FastScan & Hardware Acceleration

- **E8 Gosset Lattice Quantization (`-eq16`)**: Quantizes 8-dimensional blocks onto the optimal
  $E_8$ Gosset root lattice with Asymmetric Distance Computation (ADC).
- **SparseCache Direct SIMD**: Evaluates 32 candidate vectors concurrently without building large
  transposed index structures (0 MB resident index RAM).
- **Multi-Scheme Quantization**: Scalar (`-sq8`, `-sq16`), residual (`-rq8`), product (`-pq`), and
  randomized 1b/2b bit (`-rabitq`) schemes for adaptive accuracy and compression.
- **Hardware Acceleration**: Multi-tier SIMD dispatch (AVX2, AVX-512) and opt-in CUDA GPU
  acceleration (`--gpu`) via cuBLAS and Tensor Core batched operations.

---

## 🛠️ Programs & Tools

The GRIC suite includes several specialized CLI executables:

| Program | Category | Description |
| :--- | :--- | :--- |
| **`gric-cluster`** | **Core** | Main distance-based clustering engine for streams and images. |
| **`gric-knn`** | **Indexing** | Out-of-core metric-pruned k-nearest neighbor search solver. |
| **`gric-knn-avg`** | **Indexing** | Reconstructs frames and feature averages via k-NN graph. |
| **`gric-probe`** | **Profiling**| Single-pass geometric and statistical dataset profiler. |
| **`gric-mcp`** | **AI Tools** | Native C17 Model Context Protocol server for AI coding agents. |
| **`gric-dimdensity`** | **Analysis** | Local intrinsic dimensionality (LID) & density estimator. |
| **`gric-ascii2bin`** | **Format** | Fast encoder from ASCII coordinates/tables to `.bin` format. |
| **`gric-bin2ascii`** | **Format** | Decoder and metadata inspection tool for GRIC `.bin` files. |
| **`gric-server`** | **Server** | Native C HTTP micro-server powering desktop GUI & REST API. |
| **`gric-gui`** | **Desktop** | Standalone desktop launcher for the interactive simulator. |
| **`gric-status`** | **Telemetry** | Real-time TUI dashboard monitoring shared-memory telemetry. |
| **`gric-benchmark`**| **Testing** | Automated benchmarking suite across synthetic manifolds. |
| **`gric-simd-bench`**| **Testing** | Scalar, AVX2, and AVX-512 SIMD scaling micro-benchmark. |
| **`gric-tune`** | **Optimizer**| Parameter exploration and hyperparameter search utility. |
| **`gric-cluster-analysis`**| **Analysis** | Offline cluster quality & efficiency metrics. |
| **`gric-plot`** | **Plotting** | Publication-ready PNG/SVG summary plots of cluster manifolds. |
| **`gric-mktxtseq`** | **Generator** | Generates synthetic coordinate sequences (spirals, walks). |
| **`gric-NDmodel`** | **Modeling** | Reconstructs N-dimensional space via Simulated Annealing. |
| **`gric-gen-balls`** | **Simulation**| Multi-body physics generating bouncing-ball trajectories. |
| **`gric-ascii-spot-2-video`**| **Media** | Converts trajectories to video files or SHM streams. |
| **`gric-mkclusteredfile`** | **Utility** | Reconstructs clustered image cubes from memberships. |
| **`gric-txt2stream`** | **Streaming**| Ingests ASCII coordinate streams into shared memory SHM. |
| **`gric-stream-to-pipe`** | **Utility** | Pipes raw frames from SHM stream to stdout. |
| **`milk-fpsexec-gric-cluster`** | **Milk FPS** | Real-time FPS daemon (`gric-fps-cluster`). |
| **`libmilkgric.so`** | **Milk CLI** | Dynamic shared plugin for interactive Milk CLI sessions. |
| **`gric-info`** | **Diagnostics**| Displays compile-time feature flags, libraries, and paths. |
| **`gric-help`** | **Manual** | Unified CLI onboarding helper and command documentation. |

---

## 📦 Installation & Dependencies

### System Requirements (Debian / Ubuntu)

```bash
# Build essentials
sudo apt update
sudo apt install build-essential cmake pkg-config

# Optional scientific libraries (recommended for full capability)
sudo apt install libcfitsio-dev libpng-dev \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
    libomp-dev
```

### Shared Memory Streaming (`ImageStreamIO` - Optional)

Required for ultra-low latency camera and sensor streaming:

```bash
git clone https://github.com/milk-org/ImageStreamIO.git
cd ImageStreamIO && mkdir build && cd build
cmake .. && make && sudo make install && sudo ldconfig
```

### Milk Framework Integration (`milk` - Optional)

Required for the standalone Function Parameter Structure (FPS) clustering daemon
(`milk-fpsexec-gric-cluster`), interactive Milk CLI plugin (`libmilkgric.so`), and dynamic
real-time parameter tuning (`milk-fps-set`). Installing Milk also provides `ImageStreamIO`.

**Option 1: Automated setup via helper script**

```bash
./scripts/install_milk_dev.sh --deps
```

**Option 2: Manual compilation from source**

```bash
# 1. Install prerequisites
sudo apt install -y libreadline-dev libncurses-dev libgsl-dev

# 2. Clone Milk (framework-dev branch)
git clone --recursive -b framework-dev https://github.com/milk-org/milk.git
cd milk && mkdir _build && cd _build

# 3. Build and install to /usr/local
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install && sudo ldconfig
```

> **Note**: If installed to a custom prefix like `/usr/local/milk`, add
> `export PKG_CONFIG_PATH=/usr/local/milk/lib/pkgconfig:$PKG_CONFIG_PATH`
> to your `~/.bashrc`.

---

## 🔨 Build & Quick Start

```bash
# 1. Clone repository
git clone https://github.com/oguyon/gric-cluster.git
cd gric-cluster

# 2. Build binaries
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Verification & Feature Check

```bash
./gric-info
```

---

## ⚡ Recommended Presets & Examples

<img src="docs/figures/gric_options_map.svg" alt="Options Reference Map" width="100%">

### 1. High-FPS Video Streams / Smooth Tracking
```bash
./gric-cluster a1.5 input.mp4 -tm 0.8 -pred[10,1000,2] -outdir out_video
```

### 2. High-Dimensional / Non-Linear Manifolds
```bash
./gric-cluster 0.45 input.fits -entropy -gprob -te5 -sparse_dcc -outdir out_manifold
```

### 3. Large Scientific Sensor Arrays ($512 \times 512$+)
```bash
./gric-cluster a1.2 input.fits -tiles 2x2 -jtf -ncpu 8 -outdir out_tiled
```

---

## 📚 Documentation

For full theoretical derivations, CLI flag descriptions, and benchmark reports, visit the
**[GRIC Documentation Site](https://oguyon.github.io/gric-cluster/)**:

- 📖 **[Algorithm Overview & Modes](https://oguyon.github.io/gric-cluster/algorithm/)**
- 🎨 **[Visual Architecture & Guide](https://oguyon.github.io/gric-cluster/algorithm/visual_guide/)**
- ⌨️ **[Comprehensive CLI Option Manual](https://oguyon.github.io/gric-cluster/help/)**
- 🛠️ **[Tools & Utilities Reference](https://oguyon.github.io/gric-cluster/tools/)**
- 📊 **[Benchmark Performance Suite](https://oguyon.github.io/gric-cluster/benchmarks/)**
- 🛰️ **[Real-World Earth Observation Demo](https://oguyon.github.io/gric-cluster/satellite_demo/)**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

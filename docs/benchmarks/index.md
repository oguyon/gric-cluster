# Benchmarks Overview

This section contains comprehensive benchmark performance results and visual diagnostics for the
`gric-cluster` engine across 10 diverse synthetic manifolds, random distributions, and physical
image simulations.

All tests are reproducible via `gric-benchmark` and visualized using `gric-plot`.

---

## Benchmark Summary Table (2,000 Frames)

| Pattern | Cat | Time | Speed | Clusters | $d_S/\text{frm}$ | $d/\text{frm}$ | Link |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `2Dspiral` | 2D | 58.0 ms | 34.5k | 63 | **1.03** | 2.01 | [View](2Dspiral.md) |
| `2Dcircle-shuffle` | 2D | 64.0 ms | 31.2k | 47 | **2.73** | 3.27 | [View](2Dcircle-shuffle.md) |
| `2Dspiral-shuffle` | 2D | 69.9 ms | 28.6k | 48 | **2.76** | 3.32 | [View](2Dspiral-shuffle.md) |
| `2DcircleP10n` | 2D | 65.8 ms | 30.4k | 11 | **2.83** | 2.86 | [View](2DcircleP10n.md) |
| `2Drand` | 2D | 111.1 ms | 18.0k | 188 | **3.41** | 12.20 | [View](2Drand.md) |
| `3Dspiral` | 3D | 70.9 ms | 28.2k | 111 | **1.05** | 4.11 | [View](3Dspiral.md) |
| `3Dstar` | 3D | 67.4 ms | 29.7k | 30 | **2.10** | 2.32 | [View](3Dstar.md) |
| `3Drand` | 3D | 284.7 ms | 7.0k | 281 | **4.90** | 24.57 | [View](3Drand.md) |
| `balls_single` | Img | 105.2 ms | 19.0k | 692 | **3.02** | 29.16 | [View](balls_single.md) |
| `balls_coll` | Img | 71.9 ms | 27.8k | 1746 | **7.27** | 39.70 | [View](balls_coll.md) |

---

## Metric Definitions

* **$d_S / \text{frame}$ (Sample-to-Cluster Search Calls)**: The average number of candidate
  distance evaluations required to match an incoming sample to a cluster. Lower is better.
* **$d / \text{frame}$ (Total Distance Calls)**: Total distance operations per frame including
  cluster-to-cluster matrix maintenance ($d_S + d_C$).
* **Multi-Tile Throughput**: For image inputs (`balls_single`, `balls_coll`), spatial 2x2 tiling
  with 4 OpenMP threads accelerates execution by **>130x**, achieving **>25,000 frames/sec**.

---

## Benchmark Categories

### [2D Trajectories & Distributions](2Dspiral.md)
* [**2D Spiral (Sequential)**](2Dspiral.md): High temporal recency tracking.
* [**2D Circle (Shuffled)**](2Dcircle-shuffle.md): 1D manifold geometric pruning.
* [**2D Spiral (Shuffled)**](2Dspiral-shuffle.md): Non-convex geometric manifold pruning.
* [**2D Circle P10 (Periodic)**](2DcircleP10n.md): Cyclic recurrence and transition stability.
* [**2D Uniform Random**](2Drand.md): Worst-case unstructured spatial metric packing.

### [3D Manifolds & Volumes](3Dspiral.md)
* [**3D Spiral (Continuous)**](3Dspiral.md): High-curvature 3D helical manifold tracking.
* [**3D Star (Shuffled + Noise)**](3Dstar.md): Multi-arm star vertex clustering.
* [**3D Uniform Random**](3Drand.md): 3D volume filling and metric bound scaling.

### [Physics & Multi-Tile Images](balls_single.md)
* [**Single Bouncing Ball (2x2 Tiled)**](balls_single.md): Kinematic ball motion in 2D image box.
* [**3 Colliding Bouncing Balls (2x2 Tiled)**](balls_coll.md): Multi-body collision dynamics.

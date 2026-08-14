# Benchmarks Overview

This section contains comprehensive benchmark performance results and visual diagnostics for the
`gric-cluster` engine across 10 diverse synthetic manifolds, random distributions, and physical
image simulations.

All tests are reproducible via `gric-benchmark` and visualized using `gric-plot`.

---

## Benchmark Summary Table (20,000 Frames)

| Pattern | Cat | Time | Speed | Clusters | $d_S/\text{frm}$ | $d/\text{frm}$ | Link |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `2Dspiral` | 2D | 104.4 ms | 191.5k | 64 | **1.01** | 1.11 | [View](2Dspiral.md) |
| `2Dcircle-shuffle` | 2D | 161.3 ms | 124.0k | 46 | **2.72** | 2.77 | [View](2Dcircle-shuffle.md) |
| `2Dspiral-shuffle` | 2D | 161.8 ms | 123.6k | 49 | **2.80** | 2.86 | [View](2Dspiral-shuffle.md) |
| `2DcircleP10n` | 2D | 114.4 ms | 174.8k | 11 | **2.84** | 2.84 | [View](2DcircleP10n.md) |
| `2Drand` | 2D | 418.2 ms | 47.8k | 221 | **3.49** | 4.71 | [View](2Drand.md) |
| `3Dspiral` | 3D | 138.0 ms | 144.9k | 114 | **1.01** | 1.33 | [View](3Dspiral.md) |
| `3Dstar` | 3D | 150.7 ms | 132.7k | 30 | **2.11** | 2.13 | [View](3Dstar.md) |
| `3Drand` | 3D | 3.48 s | 5.7k | 374 | **5.09** | 8.58 | [View](3Drand.md) |
| `balls_single` | Img | 482.8 ms | 41.4k | 695 | **2.88** | 13.67 | [View](balls_single.md) |
| `balls_coll` | Img | 838.2 ms | 23.9k | 16383 | **6.92** | 35.73 | [View](balls_coll.md) |

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

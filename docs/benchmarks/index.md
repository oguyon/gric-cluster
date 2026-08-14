# Benchmarks Overview

This section contains comprehensive benchmark performance results and visual diagnostics for the
`gric-cluster` engine across 10 diverse synthetic manifolds, random distributions, and physical
image simulations.

All tests are reproducible via `gric-benchmark` and visualized using `gric-plot`.

---

## Benchmark Summary Table (20,000 Frames)

| Pattern | Cat | Time | Speed | Clusters | $d_S/\text{frm}$ | $d/\text{frm}$ | Link |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `2Dspiral` | 2D | 110.8 ms | 180.5k | 64 | **1.01** | 1.11 | [View](2Dspiral.md) |
| `2Dcircle-shuffle` | 2D | 150.5 ms | 132.9k | 43 | **2.73** | 2.78 | [View](2Dcircle-shuffle.md) |
| `2Dspiral-shuffle` | 2D | 157.9 ms | 126.6k | 48 | **2.81** | 2.87 | [View](2Dspiral-shuffle.md) |
| `2DcircleP10n` | 2D | 120.3 ms | 166.2k | 12 | **2.88** | 2.88 | [View](2DcircleP10n.md) |
| `2Drand` | 2D | 413.0 ms | 48.4k | 216 | **3.51** | 4.67 | [View](2Drand.md) |
| `3Dspiral` | 3D | 136.8 ms | 146.2k | 114 | **1.01** | 1.33 | [View](3Dspiral.md) |
| `3Dstar` | 3D | 140.2 ms | 142.6k | 30 | **2.11** | 2.13 | [View](3Dstar.md) |
| `3Drand` | 3D | 3.82 s | 5.2k | 371 | **5.11** | 8.54 | [View](3Drand.md) |
| `balls_single` | Img | 499.3 ms | 40.1k | 695 | **2.88** | 13.67 | [View](balls_single.md) |
| `balls_coll` | Img | 284.6 ms | 70.3k | 1175 | **1.84** | 7.97 | [View](balls_coll.md) |

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

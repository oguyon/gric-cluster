# Single Bouncing Ball (2x2 Tiled FITS Image)

**Category**: Physics & Multi-Tile Images  
**Data Type**: `fits` (20,000 frames)  
**Clustering Parameter**: `rlim = 1.5 (per tile)`

---

## Scenario Overview

A 2D physical ball bouncing elastically inside a 32x32 pixel domain, processed with 2x2 spatial
tiling and 4 OpenMP worker threads.

## Execution Commands

### 1. Data Generation
```bash
gric-gen-balls \
    -n \
    1 \
    -r \
    5.0 \
    -W \
    32 \
    -H \
    32 \
    -f \
    20000 \
    -s \
    42 \
    balls_single.fits
```

### 2. Clustering Execution
```bash
gric-cluster \
    1.5 \
    -maxcl \
    2500 \
    -maxim \
    20000 \
    -outdir \
    out_balls_single \
    -clustered \
    -tiles \
    2x2 \
    -ncpu \
    4 \
    balls_single.fits
```

## Performance Measurements

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Frames** | `20,000` | Number of sequential frames processed |
| **Execution Time** | `482.788 ms` | Total wall-clock runtime |
| **Throughput** | `41,426 fps` | Frames processed per second |
| **Active Clusters / States** | `695` | Total distinct clusters created |
| **Total Distance Calls ($d$)** | `273,422` | All distance calls ($d_S + d_C$) |
| **Sample Distances ($d_S$)** | `57,640` | Sample-to-cluster evaluations |
| **$d_S / \text{frame}$** | `**2.88**` | Search calls per frame |
| **Total $d / \text{frame}$** | `**13.67**` | Total distance ops per frame |
| **Peak Memory** | `135,000 KB` | Peak resident set size (RSS) |

---

## Algorithmic Insights

Spatial decomposition into 4 parallel 16x16 quadrants processes 20,000 frames in **~700 ms**
(>28,000 fps) on CPU with high spatial accuracy.

---

[← Back to Benchmarks Overview](index.md)

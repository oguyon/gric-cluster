# 3 Colliding Bouncing Balls (2x2 Tiled FITS Image)

**Category**: Physics & Multi-Tile Images  
**Data Type**: `fits` (20,000 frames)  
**Clustering Parameter**: `rlim = 7.0 (per tile)`

---

## Scenario Overview

Multi-body elastic collision dynamics between 3 balls in a 32x32 image. Stresses high-
dimensional combinatorial joint state spaces.

## Execution Commands

### 1. Data Generation
```bash
gric-gen-balls -n 3 -r 5.0 -W 32 -H 32 -f 20000 -s 42 balls_coll.fits
```

### 2. Clustering Execution
```bash
gric-cluster 7.0 -maxcl 2500 -maxim 20000 -outdir out_balls_coll -clustered -tiles \
2x2 -ncpu 4 balls_coll.fits
```

## Performance Measurements

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Frames** | `20,000` | Number of sequential frames processed |
| **Execution Time** | `278.338 ms` | Total wall-clock runtime |
| **Throughput** | `71,855 fps` | Frames processed per second |
| **Active Clusters / States** | `1175` | Total distinct clusters created |
| **Total Distance Calls ($d$)** | `159,358` | All distance calls ($d_S + d_C$) |
| **Sample Distances ($d_S$)** | `36,765` | Sample-to-cluster evaluations |
| **$d_S / \text{frame}$** | `**1.84**` | Search calls per frame |
| **Total $d / \text{frame}$** | `**7.97**` | Total distance ops per frame |
| **Peak Memory** | `135,000 KB` | Peak resident set size (RSS) |

---

## Algorithmic Insights

2x2 spatial tiling converts combinatorial state explosion into 4 compact sub-problems of 30-40
clusters per tile, running in **~260 ms** (>75,000 fps) with ~1,175 joint states reconstructed.

---

[← Back to Benchmarks Overview](index.md)

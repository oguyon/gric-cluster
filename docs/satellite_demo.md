# Satellite Earth Observation Clustering Demo

This guide demonstrates how to download, prepare, and cluster long-term satellite image
time series of planet Earth (e.g. from **NASA DSCOVR/EPIC**) using `gric-cluster`.

---

## 1. Overview & Objectives

Satellite Earth observations from geostationary orbit or the Sun-Earth $L_1$ Lagrange point
provide continuous multi-year image streams of Earth's rotating disk.

Clustering satellite Earth imagery with `gric-cluster` demonstrates several key capabilities:

1. **Continuous Manifold & Periodic Recurrence**:
   As Earth rotates every 24 hours, continents (Americas, Pacific, Asia, Europe, Africa)
   periodically rotate through the field of view. `gric-cluster` discovers these rotational
   phase states and reconstructs the periodic Markov state transition graph.
2. **Multi-Tile Spatial Decomposition**:
   By subdividing the $128 \times 128$ image into $2 \times 2$ or $4 \times 4$ spatial tiles,
   the engine tracks Northern vs Southern hemisphere weather systems and equatorial cloud bands
   in parallel across CPU cores.
3. **Metric Triangle Inequality Pruning**:
   Eliminates $>95\%$ of distance calculations during image classification, enabling processing
   speeds of **$>50,000\text{ frames/sec}$** on multi-core workstations.

---

## 2. Dataset Downloader Helper Script

The repository includes a dedicated helper tool, `tools/fetch_satellite_dataset.py`, to
automatically query, download, downsample, and package satellite time series into 3D FITS cubes
ready for clustering.

### Key CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--start` | `2023-01-01` | Start date in `YYYY-MM-DD` format |
| `--end` | `2023-12-31` | End date in `YYYY-MM-DD` format (1-year default) |
| `--size` | `128` | Target image size in pixels ($W \times H$) |
| `--max-frames` | `10000` | Maximum number of frames to download/generate |
| `--step` | `1` | Decimation step (e.g. 1 = all frames, 2 = every 2nd frame) |
| `--output`, `-o` | `earth_128x128.fits` | Output 3D FITS cube filepath |
| `--source` | `epic` | `epic` (NASA DSCOVR) or `synthetic` (offline simulation) |
| `--dry-run` | `false` | Inspect catalog counts and dates without downloading |
| `--run-demo` | `false` | Automatically launch `gric-cluster` on the output cube |

---

## 3. Step-by-Step Workflow

### Step 1: Inspect Available Frames (Dry Run)

Run `--dry-run` to inspect catalog dates and estimated file sizes before downloading:

```bash
python3 tools/fetch_satellite_dataset.py \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --size 128 \
    --max-frames 10000 \
    --dry-run
```

### Step 2: Download & Build FITS Cube

Download the 1-year time series and generate a $128 \times 128$ FITS cube:

```bash
python3 tools/fetch_satellite_dataset.py \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --size 128 \
    --max-frames 10000 \
    --output earth_2023_128x128.fits
```

### Step 3: Run Multi-Tile Streaming Clustering

Execute `gric-cluster` on the generated FITS cube with $2 \times 2$ spatial tiling and
4 OpenMP worker threads:

```bash
gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 4 -clustered earth_2023_128x128.fits
```

---

## 4. Analyzing Clustering Results

Clustering produces structured outputs in the `<filename>.clusterdat/` directory:

* `frame_membership.txt`: Active cluster ID assignment for each sequential frame across all tiles.
* `tuple_history.txt`: Global multi-tile joint state tuples $(c_0, c_1, c_2, c_3)$ over time.
* `dcc.txt`: Inter-cluster pairwise metric distance matrix ($D_{CC}$).
* `cluster_run.log`: Execution metrics including throughput (fps), RMS fit, and distance calls.

### Visualizing Discovered States

Generate spatial diagnostics and Markov transition matrices using Gnuplot:

```bash
# Plot cluster discovery timeline and transition probability matrix
python3 -c "
from tools.gen_benchmark_docs import (
    generate_timeline_plot_gp,
    generate_transition_heatmap_gp
)
from pathlib import Path
out_dir = Path('earth_2023_128x128.clusterdat')
generate_timeline_plot_gp(
    out_dir / 'frame_membership.txt',
    Path('earth_timeline.png'),
    'Earth 2023',
    is_tile=True
)
generate_transition_heatmap_gp(
    out_dir / 'frame_membership.txt',
    Path('earth_transitions.png'),
    'Earth 2023',
    is_tile=True
)
"
```

---

## 5. Offline & Synthetic Planet Generation Mode

In isolated environments or without internet access, `tools/fetch_satellite_dataset.py`
provides a procedural 3D rotating planetary Earth generator (`--source synthetic`):

```bash
python3 tools/fetch_satellite_dataset.py \
    --source synthetic \
    --size 128 \
    --max-frames 10000 \
    --output earth_synthetic_128x128.fits \
    --run-demo
```

This generates realistic rotating Earth disks featuring spherical geometry, continental
landmasses, dynamic rotating cloud bands, solar day/night illumination terminators, and
seasonal axial tilt variation.

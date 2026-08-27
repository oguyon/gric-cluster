# Performance Benchmarking, Tuning & Telemetry Tools

This section covers the diagnostic, tuning, and monitoring executables: `gric-benchmark`,
`gric-tune`, `gric-cluster-analysis`, and `gric-status`.

---

## 1. `gric-benchmark` (Automated Benchmarking Suite)

Generates synthetic datasets across diverse topologies (spirals, spheres, random walks, periodic
circles) and runs `gric-cluster` across various option combinations to report timing, throughput,
distance computations, and metric pruning efficiency.

```bash
gric-benchmark [options]
```

### Options
* `-patterns <list>`: Comma-separated list of test patterns (e.g. `2Dspiral,3Drand,2Dwalk`)
* `-n, -maxim <N>`: Number of sample frames to generate (default: `10000`)
* `-maxcl <N>`: Maximum cluster capacity limit
* `-dim <D>`: Coordinate dimensionality
* `-runs <R>`: Number of benchmark repetitions for timing variance

---

## 2. `gric-tune` (Parameter Search & Exploration)

Explores parameter spaces (radius factor, entropy gating thresholds, transition matrix mixing
weights) across datasets to identify optimal hyperparameter combinations for target clustering
quality and throughput.

```bash
gric-tune <input_file> [options]
```

### Options
* `-rlim_range <min,max,step>`: Range of radius thresholds to test
* `-tm_range <min,max,step>`: Range of transition matrix mixing weights
* `-metric <speed|quality|balance>`: Optimization objective function

---

## 3. `gric-cluster-analysis` (Offline Log & Quality Diagnostic)

Analyzes generated clustering run logs (`cluster_run.log`), transition matrices
(`transition_matrix.txt`), and inter-cluster distance graphs to report metric entropy,
cluster radii distributions, and trajectory continuity.

```bash
gric-cluster-analysis <cluster_output_dir> [options]
```

---

## 4. `gric-status` (Real-Time Shared Memory TUI Monitor)

Terminal User Interface (TUI) dashboard that reads the POSIX shared-memory telemetry stream
published by `gric-cluster` (via `-shm <file>`). Displays live frame processing rates, active
cluster counts, entropy gating telemetry, and memory usage.

```bash
gric-status <shm_file> [options]
```

### Options
* `-r <Hz>`: Refresh rate in updates per second (default: `10`)
* `-1`: Print status telemetry once and exit

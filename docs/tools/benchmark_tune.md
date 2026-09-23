# Performance Benchmarking, Tuning & Telemetry Tools

This section covers the diagnostic, tuning, and monitoring executables: `gric-benchmark`,
`gric-simd-bench`, `gric-tune`, `gric-cluster-analysis`, and `gric-status`.

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

## 2. `gric-simd-bench` (SIMD Scaling Micro-Benchmark)

Benchmarks hardware vector instruction scaling across CPU architectures. Evaluates throughput and
latency across Scalar, AVX2, and AVX-512 code paths for core metric loops:

* Single-vector Euclidean L2 distance (Float32 and Float64)
* Batched 1x8 multi-vector Euclidean distance
* 16-bit Scalar Quantization (SQ16) FastScan screening
* E8 Gosset Lattice Quantization (EQ16) FastScan screening

```bash
gric-simd-bench [options]
```

---

## 3. `gric-tune` (Parameter Search & Exploration)

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

## 4. `gric-cluster-analysis` (Offline Log & Quality Diagnostic)

Analyzes generated clustering run logs (`cluster_run.log`), transition matrices
(`transition_matrix.txt`), and inter-cluster distance graphs to report metric entropy,
cluster radii distributions, and trajectory continuity.

```bash
gric-cluster-analysis <cluster_output_dir> [options]
```

---

## 5. `gric-status` (Real-Time Shared Memory TUI Monitor)

Terminal User Interface (TUI) dashboard that reads the POSIX shared-memory telemetry stream
published by `gric-cluster` (via `-shm <file>`). Displays live frame processing rates, active
cluster counts, entropy gating telemetry, RSS memory usage, and OpenMP thread activity.

```bash
gric-status <shm_file> [options]
```

### Options
* `-r, --rate <Hz>`: Refresh rate in updates per second (default: `15.0`)
* `-w, --watch`: Run in non-interactive streaming watch mode (plain text logging)
* `-hm, --help-mono`: Monochrome output without ANSI terminal color codes
* `-h, --help`: Display command options and exit

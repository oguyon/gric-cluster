# Welcome to the GRIC Cluster Wiki

This project provides a high-speed, distance-based clustering tool optimized for sequential
data (images, camera streams, sensor logs, etc.).

## Sections

*   **[Benchmarks](Benchmarks.md)**: Performance analysis and timing results on synthetic datasets.
*   **[Practical Use](Practical-Use.md)**: Real-world scenarios, workflows, and parameter tips.
*   **[Algorithm Details](Algorithm-Details.md)**: Mathematical derivations and optimizations.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run on sample data
./gric-cluster a1.5 input.txt -outdir results
```

See the [README](../README.md) for detailed installation and usage instructions.

# Welcome to the GRIC Documentation

This project provides a high-speed, distance-based clustering tool optimized for sequential
data (images, sensor logs, etc.).

## Sections

*   **[Benchmarks](benchmarks.md)**: Performance analysis and timing results on synthetic
    datasets.
*   **[Practical Use Cases](practical-use.md)**: Real-world scenarios, workflows, and tips
    for getting the most out of the tool.
*   **[Algorithm Details](algorithm.md)**: Detailed description of the clustering algorithm
    and optimizations.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. && make

# Run on sample data
./gric-cluster a1.5 input.txt -outdir results
```

See the [README](https://github.com/oguyon/gric-cluster) for installation requirements
and dependency details.

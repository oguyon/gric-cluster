# Welcome to the GRIC Documentation

<!-- BUILD_TIMESTAMP -->

This project provides a high-speed, distance-based clustering tool optimized for sequential
data (images, sensor logs, etc.).

## Sections

*   **[Visual Architecture & Options Guide](algorithm/visual_guide.md)**: Visual diagrams, animated explainer video, and interactive 2D simulator for understanding GRIC and all its options.
*   **[Algorithm Overview & Modes](algorithm/index.md)**: High-level overview, steps, and
    modes (Greedy vs. Entropy).
*   **[Benchmarks](benchmarks/index.md)**: Performance analysis and timing results on synthetic
    datasets.
*   **[Practical Use Cases](practical-use.md)**: Real-world scenarios, workflows, and tips
    for getting the most out of the tool.

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

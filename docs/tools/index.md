# GRIC Suite Tools & Utilities

The **GRIC** software ecosystem consists of 19 specialized executables and utilities covering the
entire lifecycle of sequential clustering, high-dimensional k-NN indexing, binary conversion,
data synthesis, visualization, and native GUI serving.

---

## Tool Catalog Overview

| Executable | Category | Description | Reference Guide |
| :--- | :--- | :--- | :--- |
| **`gric-cluster`** | Core | Main distance-based clustering engine for streams and images | [CLI Reference](../help/index.md) |
| **`gric-knn`** | Indexing | Metric-pruned out-of-core k-nearest neighbor search solver | [k-NN Guide](knn.md) |
| **`gric-ascii2bin`** | Format | Fast encoder from ASCII coordinates/tables to `.bin` format | [Binary Format](bin_io.md) |
| **`gric-bin2ascii`** | Format | Decoder & inspection tool for self-describing `.bin` files | [Binary Format](bin_io.md) |
| **`gric-server`** | GUI / Web | High-concurrency native C HTTP/REST micro-server | [Server & GUI](server_gui.md) |
| **`gric-gui`** | Desktop | Native desktop app launcher for the GRIC simulator | [Server & GUI](server_gui.md) |
| **`gric-status`** | Monitoring| Real-time TUI dashboard for shared-memory telemetry | [Benchmarks & Telemetry](benchmark_tune.md) |
| **`gric-benchmark`** | Diagnostics| Automated benchmarking suite across synthetic manifolds | [Benchmarks & Telemetry](benchmark_tune.md) |
| **`gric-tune`** | Optimization| Grid and parameter exploration search optimizer | [Benchmarks & Telemetry](benchmark_tune.md) |
| **`gric-cluster-analysis`**| Diagnostics| Offline cluster quality and efficiency analysis tool | [Benchmarks & Telemetry](benchmark_tune.md) |
| **`gric-plot`** | Rendering | Visualization tool for cluster manifolds (PNG/SVG) | [Data Tools & Generators](generators.md) |
| **`gric-mktxtseq`** | Generator | Synthetic coordinate sequence generator (2D/3D/ND) | [Data Tools & Generators](generators.md) |
| **`gric-NDmodel`** | Modeling | N-dimensional space reconstruction via Simulated Annealing | [Data Tools & Generators](generators.md) |
| **`gric-gen-balls`** | Generator | Multi-body kinematic bouncing ball FITS cube generator | [Data Tools & Generators](generators.md) |
| **`gric-ascii-spot-2-video`**| Generator | Converts 2D/3D trajectories into simulated video streams | [Data Tools & Generators](generators.md) |
| **`gric-mkclusteredfile`**| Utility | Reconstructs grouped image cubes from cluster memberships | [Data Tools & Generators](generators.md) |
| **`gric-txt2stream`** | Streaming | Ingests ASCII coordinate streams into `ImageStreamIO` SHM | [Data Tools & Generators](generators.md) |
| **`gric-stream-to-pipe`**| Streaming | Dumps binary frames from `ImageStreamIO` to stdout | [Data Tools & Generators](generators.md) |
| **`gric-info`** | System | Diagnostics utility reporting build flags and libraries | [Data Tools & Generators](generators.md) |
| **`gric-help`** | System | Unified CLI orientation helper and command manual | [Data Tools & Generators](generators.md) |

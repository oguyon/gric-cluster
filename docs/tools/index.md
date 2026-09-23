# GRIC Suite Tools & Utilities

The **GRIC** software ecosystem consists of over 25 specialized executables and utilities covering
the entire lifecycle of sequential clustering, high-dimensional k-NN indexing, binary conversion,
data synthesis, visualization, and native GUI serving.

---

## Tool Catalog Overview

| Executable | Description | Reference Guide |
| :--- | :--- | :--- |
| **`gric-cluster`** | Main sequential clustering engine | [CLI Reference](../help/index.md) |
| **`gric-knn`** | Metric-pruned k-NN search solver | [k-NN Guide](knn.md) |
| **`gric-knn-avg`** | Reconstructs frames via k-NN graph | [k-NN Guide](knn.md) |
| **`gric-probe`** | Dataset profiler & preset generator | [Dataset Profiler](probe.md) |
| **`gric-mcp`** | C17 Model Context Protocol server | [MCP Server](mcp.md) |
| **`gric-dimdensity`** | Local intrinsic dimension & density | [Dim & Density](dimdensity.md) |
| **`gric-ascii2bin`** | ASCII to binary format encoder | [Binary Format](bin_io.md) |
| **`gric-bin2ascii`** | Binary file inspection & decoder | [Binary Format](bin_io.md) |
| **`gric-server`** | Native C HTTP/REST micro-server | [Server & GUI](server_gui.md) |
| **`gric-gui`** | Desktop simulator app launcher | [Server & GUI](server_gui.md) |
| **`gric_cluster.wasm`** | In-browser C clustering engine | [WASM Guide](wasm.md) |
| **`gric-status`** | Real-time shared-memory monitor | [Telemetry](benchmark_tune.md) |
| **`gric-benchmark`** | Synthetic manifold benchmark suite | [Benchmarks](benchmark_tune.md) |
| **`gric-simd-bench`** | Micro-benchmark for SIMD scaling | [Benchmarks](benchmark_tune.md) |
| **`gric-tune`** | Parameter search & tuning utility | [Tuning](benchmark_tune.md) |
| **`gric-cluster-analysis`** | Offline log diagnostic tool | [Analysis](benchmark_tune.md) |
| **`gric-plot`** | PNG/SVG manifold visualization tool | [Generators](generators.md) |
| **`gric-mktxtseq`** | Synthetic coordinate generator | [Generators](generators.md) |
| **`gric-NDmodel`** | N-dimensional space reconstructor | [Generators](generators.md) |
| **`gric-gen-balls`** | Kinematic bouncing ball generator | [Generators](generators.md) |
| **`gric-gen-asteroid`** | 3D asteroid light-curve generator | [Generators](generators.md) |
| **`gric-asteroid-recon-test`** | Asteroid reconstruction tester | [Generators](generators.md) |
| **`gric-ascii-spot-2-video`** | Trajectory to simulated video | [Generators](generators.md) |
| **`gric-mkclusteredfile`** | Reconstructs grouped image cubes | [Generators](generators.md) |
| **`gric-txt2stream`** | Pipes ASCII stream into SHM buffer | [Streaming](generators.md) |
| **`gric-stream-to-pipe`** | Dumps binary SHM frames to stdout | [Streaming](generators.md) |
| **`gric-info`** | Displays build flags & libraries | [Diagnostics](generators.md) |
| **`gric-help`** | Command help & orientation viewer | [Help Manual](generators.md) |

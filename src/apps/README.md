# Applications & Suite Executables (Level 3)

## Purpose
Top-level executable binaries composing engines, UI, I/O adapters, and protocols into runnable
CLI tools, daemons, and analysis utilities.

## Architectural Layer
- **Level**: Level 3 (Applications)
- **Suite Binaries**:
  - `apps/gric-cluster/`: High-dimensional clustering engine binary (`gric-cluster`).
  - `apps/gric-knn/`: Fast k-NN search and trajectory evaluation CLI (`gric-knn`, `gric-knn-avg`).
  - `apps/gric-server/`: IPC socket daemon providing clustering and k-NN query endpoints.
  - `apps/gric-status/`: Real-time interactive terminal status monitor.
  - `apps/gric-plot/`: Frame, metric, and convergence visualizer.
  - `apps/gric-probe/`: Dataset geometry, dimensionality, and invariant inspection tool.
  - `apps/gric-benchmark/`: Algorithmic and hardware throughput benchmarking harness.
  - `apps/gric-cluster-analysis/`: Post-clustering partition quality and silhouette inspector.
  - `apps/gric-mcp/`: Anthropic Model Context Protocol (MCP) server daemon.
- **Permitted Dependencies**:
  - Level 0 (`base/*`, `third_party/*`), Level 1 (`quant/*`, `accel/*`),
    Level 2 (`engine/*`), Level 3 (`ui/*`).

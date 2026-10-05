# Domain Computational Engines (Level 2)

## Purpose
Core algorithmic engines for incremental clustering, multi-probe anchor routing, and k-nearest
neighbor search. Implemented as pure compute libraries (`libgric`) free from UI or framework
dependencies.

## Architectural Layer
- **Level**: Level 2 (Domain Engines)
- **Subdirectories & Components**:
  - `api/`: Top-level computational API (`gric_api.h`).
  - `locate/`: Multi-probe anchor routing and basin selection (`cluster_locator.h`).
  - `cluster/`:
    - `core/`: Clustering orchestrator, multi-tile dispatch, and state management.
    - `steps/`: Isolated, single-responsibility per-frame clustering steps.
    - `math/`: Distance functions (`framedist`), GEMM microkernels, and tuple retrieval.
    - `io/`: Data ingestion (ASCII, FITS, FFmpeg, ImageStreamIO) and artifact writers.
    - `trace/`: Profiling and performance event tracing.
    - `gpu/`: GPU clustering kernels.
  - `knn/`:
    - `core/`: Multi-threaded query dispatch, priority heaps (`knn_heap.h`).
    - `search/`: Candidate scoring, 2-hop graph injection, and member quantization.
    - `cache/`: Sidecar quantization caches (SQ8, SQ16, EQ16, RQ8, PQ) and memory layouts.
    - `cross/`: Trajectory tracking, cross-dataset graph traversal, and basin routing.
    - `pruning/`: Metric triangle inequality pruning filters.
    - `io/`: Model parsers, binary readers, and index serializes.
    - `gpu/`: CUDA k-NN kernels and IVF queries.
- **Permitted Dependencies**:
  - Level 0 (`base/*`), Level 1 (`quant/*`, `accel/*`).
  - Optional I/O libraries (CFITSIO, libpng, FFmpeg, ImageStreamIO) when compiled in.
  - Strictly **no** UI, CLI, cJSON, or Milk FPS dependencies.

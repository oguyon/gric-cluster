# GRIC Developer & Architecture Guide

Welcome to the internal developer documentation and architecture guide for the **GRIC**
(Geometric Real-Time Incremental Clustering) codebase.

This section provides comprehensive technical documentation for developers, contributors,
and system integrators working directly on the C engine, optimization kernels, or tool bridges.

---

## Guide Overview

The developer documentation is organized into five specialized deep-dive manuals:

### 1. [Architecture & Modularity](architecture.md)
* Architectural principles: layered separation of concerns and dependency isolation.
* Logical source code layout (`src/gric-cluster/`, `src/gric-knn/`, `src/shared/`, etc.).
* Module dependency graph and rules for introducing cross-module relationships.

### 2. [Code Style & Conventions](code_style.md)
* Standard C17 conventions: Allman brace formatting and line length constraints (&le; 100 chars).
* Multi-line function prototype column parameter alignment.
* Kernel-Doc function documentation standards with mandatory Purpose & Context.
* Early-exit control flow, RAII-like cleanup patterns (`goto cleanup`), and variable scoping.

### 3. [Engine Internals](engine_internals.md)
* Core configuration and state layouts: `ClusterConfig`, `ClusterState`, `KnnModel`, `Frame`.
* Frame ingestion, metric pruning pipeline (TE4, TE5, Sparse DCC), and Pass 2 fusion.
* Algorithmic invariants and state verification guarantees.

### 4. [SIMD & Compute Optimization](simd_guide.md)
* Vectorization design: AVX-512, AVX2+FMA, and ARM NEON hardware microkernels.
* GEMM-based distance matrix engines and batch coordinate transformations.
* Guidelines for preventing register spills, pointer aliasing (`restrict`), and alignment faults.

### 5. [Shared Memory & IPC](shm_internals.md)
* ImageStreamIO shared-memory circular buffer layout and semaphore synchronization.
* Atomic sequence counters and real-time streaming ingestion protocols.
* IPC safety, lock-free telemetry, and cross-process buffer cleanup.

---

## Contributing Workflow

When modifying or extending GRIC components:

1. **Check Style & Constraints**: Ensure all C source files comply with the 100-character line
   limit and Allman bracing. Use `gric_audit_code_style` to verify compliance.
2. **Document Functions**: Every non-trivial function must include Kernel-Doc comments explaining
   both its interface and its concrete architectural role ("What is this used for?").
3. **Validate Invariants**: Ensure zero dynamic memory allocations (`malloc`, `calloc`) occur
   within inner compute loops.
4. **Run Integration Suite**: All 52 CTest test suites must pass cleanly (`ctest --test-dir build`).
5. **Verify Documentation**: Run `mkdocs build --strict` to verify all links and cross-references.

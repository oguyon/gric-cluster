# CLI Reference & Options

Detailed reference manual for the `gric-cluster` command line interface, options, and algorithmic concepts.
All topics here are also available directly on the terminal using `gric-cluster --help <topic>`.

## Core & Clustering Control
* [`rlim`](rlim.md): Radius threshold for cluster membership
* [`auto_rlim`](auto_rlim.md): Auto-scaled rlim syntax (`a<factor>`)
* [`dprob`](dprob.md): Delta probability recency update bias
* [`maxcl`](maxcl.md): Maximum number of clusters
* [`ncpu`](ncpu.md): Number of OpenMP threads
* [`maxcl_strategy`](maxcl_strategy.md): Strategy when `maxcl` limit is reached
* [`discard_frac`](discard_frac.md): Fraction of oldest clusters to discard
* [`maxim`](maxim.md): Maximum number of input frames to process
* [`pred`](pred.md): Temporal pattern prediction and velocity extrapolation

## Pruning & Search Optimizations
* [`te4`](te4.md): 4-point triangle inequality pruning
* [`te5`](te5.md): 5-point triangle inequality pruning
* [`gprob`](gprob.md): Geometric probability learning from visitor history
* [`entropy`](entropy.md): Shannon entropy-guided candidate selection
* [`entropy_gate`](entropy_gate.md): Adaptive entropy gating threshold
* [`entropy_fast`](entropy_fast.md): Popcount-only surrogate gating
* [`soft_bayesian`](soft_bayesian.md): Soft Bayesian candidate likelihood updates
* [`sparse_dcc`](sparse_dcc.md): Sparse cluster-to-cluster distance matrix

## Input & Output Formats
* [`stream`](stream.md): ImageStreamIO shared-memory stream input
* [`cnt2sync`](cnt2sync.md): Read synchronization counter for ImageStreamIO
* [`outdir`](outdir.md): Output directory for clustering logs and models
* [`avg`](avg.md): Compute average frame per cluster
* [`distall`](distall.md): Save all computed pairwise distances
* [`pngout`](pngout.md): Export cluster centers as PNG images
* [`fitsout`](fitsout.md): Force FITS format for multi-dimensional images
* [`clustered`](clustered.md): Generate clustered output dataset file
* [`shm`](shm.md): Shared memory status stream

## Multi-Tile Processing
* [`tiles`](tiles.md): Spatial NxM tile grid partitioning
* [`tilemap`](tilemap.md): Integer FITS mask for custom tiling
* [`tileconf`](tileconf.md): Per-tile configuration overrides
* [`jtf`](jtf.md): Joint Trajectory Fusion (Pass 2)
* [`xtile`](xtile.md): Live cross-tile prior injection
* [`cpt`](cpt.md): Conditional Probability Table for tile dependencies
* [`retrieval_window`](retrieval_window.md): Tuple lookback horizon for trajectory fusion

## Conceptual Deep Dives
* [`intro`](intro.md): Getting started with GRIC
* [`algorithm`](algorithm.md): Complete algorithmic overview
* [`algorithm/pruning`](algorithm_pruning.md): Multi-point distance geometry pruning
* [`algorithm/gating`](algorithm_gating.md): Entropy gating details
* [`algorithm/entropy`](algorithm_entropy.md): Information-theoretic target selection
* [`tiling`](tiling.md): Multi-tile distributed processing architecture
* [`performance`](performance.md): Performance tuning and optimization guide

# CLI Reference & Options

Detailed reference manual for the `gric-cluster` command line interface, options, and algorithmic
concepts. All topics here are also available on the terminal via `gric-cluster --help <topic>`.

## Overviews & Architectural Guides
* [`intro`](guides/intro.md): Getting started with GRIC architecture and basic principles
* [`performance`](guides/performance.md): Performance tuning guide and optimization decision matrix
* [`algorithm`](guides/algorithm.md): Complete algorithmic overview and execution pipeline
* [`clustering`](guides/clustering.md): Core clustering loop and step assignment workflow
* [`compression`](guides/compression.md): Information compression principles and model extraction
* [`tiling`](guides/tiling.md): Multi-tile distributed processing architecture

## Core Clustering Options
* [`rlim`](core/rlim.md): Radius threshold for cluster membership (`<val>`)
* [`auto_rlim`](core/auto_rlim.md): Auto-scaled rlim syntax (`a<factor>`) based on nearest-neighbors
* [`scandist`](core/scandist.md): Pre-clustering sample distance scan (`-scandist <N>`)
* [`maxcl`](core/maxcl.md): Maximum cluster capacity limit (`-maxcl <N>`)
* [`maxcl_strategy`](core/maxcl_strategy.md): Strategy when limit is reached (`discard` / `merge`)
* [`discard_frac`](core/discard_frac.md): Oldest cluster discard fraction (`-discard_frac <f>`)
* [`discarded`](core/discarded.md): Discarded cluster trajectory log file (`-discarded <fname>`)
* [`maxim`](core/maxim.md): Maximum number of input frames to process (`-maxim <N>`)
* [`ncpu`](core/ncpu.md): Number of OpenMP worker threads (`-ncpu <N>`)
* [`progress`](core/progress.md): Progress report interval (`-progress <N>`)
* [`verbose`](core/verbose.md): Debug logging verbosity (`-verbose`, `-veryverbose`)
* [`conf`](core/conf.md): Load clustering configuration file (`-conf <file>`)
* [`confw`](core/confw.md): Save active runtime configuration to file (`-confw <file>`)

## Quantization & Hardware Acceleration
* [`eq16`](quantization/eq16.md): 16-bit E8 Gosset lattice quantization and FastScan filtering (`-eq16`)
* [`sq16`](quantization/sq16.md): 16-bit scalar quantization filtering (`-sq16`, `-sq16-ratio`)
* [`sq8`](quantization/sq8.md): 8-bit scalar quantization filtering (`-sq8`)
* [`memo`](quantization/memo.md): Quantized distance memoization cache (`-memo`)
* [`batch_dist`](quantization/batch_dist.md): Multi-vector SIMD batch distance evaluation (`-batch-dist`)
* [`double`](quantization/double.md): 64-bit double precision execution mode (`-double`)
* [`gpu`](quantization/gpu.md): CUDA GPU hardware acceleration (`-gpu`, `-cpu`, `-gpu-batch-size`)

## Profiling & Adaptive Presets
* [`prof`](profiling/prof.md): Dataset profile file loading and auto-discovery (`-prof <file>`, `-no-prof`)
* [`preset`](profiling/preset.md): Radius preset selection from profile (`-preset <fine|balanced|coarse>`)

## Pruning & Distance Geometry
* [`te4`](pruning/te4.md): 4-point triangle inequality pruning (`-te4`)
* [`te5`](pruning/te5.md): 5-point triangle inequality pruning (`-te5`)
* [`algorithm/pruning`](pruning/algorithm_pruning.md): Multi-point distance geometry pruning theory
* [`sparse_dcc`](pruning/sparse_dcc.md): Sparse cluster distance matrix (`-sparse_dcc`)
* [`sparse_dcc_extra_evals`](pruning/sparse_dcc_extra_evals.md): Extra DCC bound evaluations (`<N>`)
* [`algorithm/sparse_dcc`](pruning/algorithm_sparse_dcc.md): Sparse DCC lower/upper bound theory
* [`no_dcc`](pruning/no_dcc.md): Disable inter-cluster distance matrix completely (`-no_dcc`)
* [`dcc`](pruning/dcc.md): Write full pairwise cluster distance matrix (`-dcc <fname>`)
* [`dcc_sq16`](pruning/dcc_sq16.md): 16-bit quantized DCC matrix storage (`-dcc-sq16`)

## Entropy Engine & Candidate Gating
* [`entropy`](entropy/entropy.md): Shannon entropy-guided candidate selection mode (`-entropy`)
* [`entropy_fast`](entropy/entropy_fast.md): Popcount-only fast surrogate gating (`-entropy_fast`)
* [`entropy_gate`](entropy/entropy_gate.md): Adaptive entropy gating threshold (`-entropy_gate <thresh>`)
* [`entropy_first_gate`](entropy/entropy_first_gate.md): Min evaluations before gating (`<N>`)
* [`entropy_max_targets`](entropy/entropy_max_targets.md): Max candidate targets evaluated (`<N>`)
* [`entropy_min_prob`](entropy/entropy_min_prob.md): Min cluster probability threshold (`<p>`)
* [`entropy_leader`](entropy/entropy_leader.md): Dominant leader bypass shortcut (`-entropy_leader`)
* [`algorithm/entropy`](entropy/algorithm_entropy.md): Information-theoretic target selection theory
* [`algorithm/gating`](entropy/algorithm_gating.md): Adaptive entropy gating mathematics

## Priors, Transitions & Prediction
* [`gprob`](priors/gprob.md): Geometric probability learning from visitor history (`-gprob`)
* [`algorithm/gprob`](priors/algorithm_gprob.md): Topology and transition graph learning theory
* [`dprob`](priors/dprob.md): Delta probability recency update bias (`-dprob <val>`)
* [`fmatcha`](priors/fmatcha.md): Prior match scaling factor (`-fmatcha <val>`)
* [`fmatchb`](priors/fmatchb.md): Prior distance falloff exponent (`-fmatchb <val>`)
* [`soft_bayesian`](priors/soft_bayesian.md): Soft Bayesian candidate likelihood updates (`-soft_bayesian`)
* [`soft_bayesian_sigma`](priors/soft_bayesian_sigma.md): Gaussian standard deviation (`<val>`)
* [`algorithm/soft_bayesian`](priors/algorithm_soft_bayesian.md): Soft Bayesian update equations
* [`tm`](priors/tm.md): Temporal transition matrix weight (`-tm <val>`)
* [`tm_out`](priors/tm_out.md): Export learned transition matrix to file (`-tm_out <file>`)
* [`pred`](priors/pred.md): Temporal pattern prediction and velocity extrapolation (`-pred`)
* [`pass2nearest`](priors/pass2nearest.md): Second-pass closest anchor reassignment (`-pass2nearest`)

## Multi-Tile Architecture & Joint Trajectory Fusion
* [`tiles`](tiling/tiles.md): Spatial NxM tile grid partitioning (`-tiles <NxM>`)
* [`tilemap`](tiling/tilemap.md): Integer FITS mask for arbitrary custom tiling (`-tilemap <file>`)
* [`tileconf`](tiling/tileconf.md): Per-tile configuration overrides (`-tileconf <file>`)
* [`jtf`](tiling/jtf.md): Joint Trajectory Fusion Pass 2 (`-jtf`)
* [`retrieval_window`](tiling/retrieval_window.md): Lookback horizon for trajectory fusion (`<N>`)
* [`xtile`](tiling/xtile.md): Live cross-tile prior injection (`-xtile`)
* [`no_xtile`](tiling/no_xtile.md): Disable live cross-tile prior updates (`-no_xtile`)
* [`xtile_decay`](tiling/xtile_decay.md): Cross-tile weight decay rate (`-xtile_decay <rate>`)
* [`cpt`](tiling/cpt.md): Conditional Probability Table for inter-tile dependencies (`-cpt`)

## Input & Stream Ingestion
* [`input`](io/input.md): Supported input formats (FITS cubes, text sequences, binary streams)
* [`filelist`](io/filelist.md): Ingest input as list of image filepaths (`-filelist`)
* [`stream`](io/stream.md): ImageStreamIO shared-memory stream input (`-stream <name>`)
* [`cnt2sync`](io/cnt2sync.md): Read synchronization counter for ImageStreamIO (`-cnt2sync <N>`)
* [`shm`](io/shm.md): Shared memory status stream publication (`-shm <name>`)

## Output, Analysis & Diagnostics
* [`outdir`](output/outdir.md): Output directory for clustering logs and models (`-outdir <dir>`)
* [`output`](output/output.md): Overview of all clustering artifact files
* [`txt`](output/txt.md): Write output artifacts in ASCII plain text format (`-txt`, `-no-txt`)
* [`clustered`](output/clustered.md): Generate clustered output dataset file (`-clustered`)
* [`membership`](output/membership.md): Write per-frame cluster assignment log (`-membership <fname>`)
* [`no_membership`](output/no_membership.md): Disable cluster membership logging (`-no_membership`)
* [`anchors`](output/anchors.md): Export exemplar anchor frame references (`-anchors <fname>`)
* [`no_anchors`](output/no_anchors.md): Suppress exemplar anchor frame output (`-no_anchors`)
* [`counts`](output/counts.md): Export cluster visitor counts (`-counts <fname>`)
* [`no_counts`](output/no_counts.md): Suppress cluster visitor counts output (`-no_counts`)
* [`evals`](output/evals.md): Log frame distance evaluation history (`-evals`, `-no_evals`)
* [`avg`](output/avg.md): Compute average frame per cluster (`-avg`)
* [`fitsout`](output/fitsout.md): Force FITS format for multi-dimensional images (`-fitsout`)
* [`pngout`](output/pngout.md): Export cluster centers as PNG images (`-pngout`)
* [`clusters`](output/clusters.md): Export cluster centroid coordinate file (`-clusters <fname>`)
* [`maxvis`](output/maxvis.md): Maximum visitor frames saved per cluster (`-maxvis <N>`)
* [`distall`](output/distall.md): Save all computed pairwise distances to file (`-distall <fname>`)
* [`analysis`](output/analysis.md): Offline cluster log analysis tool (`gric-cluster-analysis`)
* [`dimdensity`](output/dimdensity.md): Local intrinsic dimension & density estimator tool

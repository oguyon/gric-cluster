# CLI Reference & Options

Detailed reference manual for the `gric-cluster` command line interface, options, and algorithmic
concepts. All topics here are also available on the terminal via `gric-cluster --help <topic>`.

## Overviews & Architectural Guides
* [`intro`](intro.md): Getting started with GRIC architecture and basic principles
* [`performance`](performance.md): Performance tuning guide and optimization decision matrix
* [`algorithm`](algorithm.md): Complete algorithmic overview and execution pipeline
* [`clustering`](clustering.md): Core clustering loop and step assignment workflow
* [`compression`](compression.md): Information compression principles and model extraction
* [`tiling`](tiling.md): Multi-tile distributed processing architecture

## Core Clustering Options
* [`rlim`](rlim.md): Radius threshold for cluster membership (`<val>`)
* [`auto_rlim`](auto_rlim.md): Auto-scaled rlim syntax (`a<factor>`) based on nearest-neighbors
* [`scandist`](scandist.md): Pre-clustering sample distance scan (`-scandist <N>`)
* [`maxcl`](maxcl.md): Maximum cluster capacity limit (`-maxcl <N>`)
* [`maxcl_strategy`](maxcl_strategy.md): Strategy when `maxcl` limit is reached (`discard` / `merge`)
* [`discard_frac`](discard_frac.md): Fraction of oldest clusters to discard on limit (`-discard_frac <f>`)
* [`discarded`](discarded.md): Discarded cluster trajectory log file (`-discarded <fname>`)
* [`maxim`](maxim.md): Maximum number of input frames to process (`-maxim <N>`)
* [`ncpu`](ncpu.md): Number of OpenMP worker threads (`-ncpu <N>`)
* [`progress`](progress.md): Progress report interval (`-progress <N>`)
* [`verbose`](verbose.md): Debug logging verbosity (`-verbose`, `-veryverbose`)
* [`conf`](conf.md): Load clustering configuration file (`-conf <file>`)
* [`confw`](confw.md): Save active runtime configuration to file (`-confw <file>`)

## Pruning & Distance Geometry
* [`te4`](te4.md): 4-point triangle inequality pruning (`-te4`)
* [`te5`](te5.md): 5-point triangle inequality pruning (`-te5`)
* [`algorithm/pruning`](algorithm_pruning.md): Multi-point distance geometry pruning theory
* [`sparse_dcc`](sparse_dcc.md): Sparse cluster distance matrix (`-sparse_dcc`)
* [`sparse_dcc_extra_evals`](sparse_dcc_extra_evals.md): Bound evaluations (`-sparse_dcc_extra_evals <N>`)
* [`algorithm/sparse_dcc`](algorithm_sparse_dcc.md): Sparse DCC lower/upper bound theory
* [`no_dcc`](no_dcc.md): Disable inter-cluster distance matrix completely (`-no_dcc`)
* [`dcc`](dcc.md): Write full pairwise cluster distance matrix (`-dcc <fname>`)

## Entropy Engine & Candidate Gating
* [`entropy`](entropy.md): Shannon entropy-guided candidate selection mode (`-entropy`)
* [`entropy_fast`](entropy_fast.md): Popcount-only fast surrogate gating (`-entropy_fast`)
* [`entropy_gate`](entropy_gate.md): Adaptive entropy gating threshold (`-entropy_gate <thresh>`)
* [`entropy_first_gate`](entropy_first_gate.md): Minimum evaluations before gating (`-entropy_first_gate <N>`)
* [`entropy_max_targets`](entropy_max_targets.md): Max candidate targets evaluated (`-entropy_max_targets <N>`)
* [`entropy_min_prob`](entropy_min_prob.md): Minimum cluster probability threshold (`-entropy_min_prob <p>`)
* [`entropy_leader`](entropy_leader.md): Dominant leader bypass shortcut (`-entropy_leader`)
* [`algorithm/entropy`](algorithm_entropy.md): Information-theoretic target selection theory
* [`algorithm/gating`](algorithm_gating.md): Adaptive entropy gating mathematics

## Priors, Transitions & Prediction
* [`gprob`](gprob.md): Geometric probability learning from visitor history (`-gprob`)
* [`algorithm/gprob`](algorithm_gprob.md): Topology and transition graph learning theory
* [`dprob`](dprob.md): Delta probability recency update bias (`-dprob <val>`)
* [`fmatcha`](fmatcha.md): Prior match scaling factor (`-fmatcha <val>`)
* [`fmatchb`](fmatchb.md): Prior distance falloff exponent (`-fmatchb <val>`)
* [`soft_bayesian`](soft_bayesian.md): Soft Bayesian candidate likelihood updates (`-soft_bayesian`)
* [`soft_bayesian_sigma`](soft_bayesian_sigma.md): Gaussian standard deviation (`-soft_bayesian_sigma <val>`)
* [`algorithm/soft_bayesian`](algorithm_soft_bayesian.md): Soft Bayesian update equations
* [`tm`](tm.md): Temporal transition matrix weight (`-tm <val>`)
* [`tm_out`](tm_out.md): Export learned transition matrix to file (`-tm_out <file>`)
* [`pred`](pred.md): Temporal pattern prediction and velocity extrapolation (`-pred`)
* [`pass2nearest`](pass2nearest.md): Second-pass closest anchor reassignment (`-pass2nearest`)

## Multi-Tile Architecture & Joint Trajectory Fusion
* [`tiles`](tiles.md): Spatial NxM tile grid partitioning (`-tiles <NxM>`)
* [`tilemap`](tilemap.md): Integer FITS mask for arbitrary custom tiling (`-tilemap <file>`)
* [`tileconf`](tileconf.md): Per-tile configuration overrides (`-tileconf <file>`)
* [`jtf`](jtf.md): Joint Trajectory Fusion Pass 2 (`-jtf`)
* [`retrieval_window`](retrieval_window.md): Lookback horizon for trajectory fusion (`-retrieval_window <N>`)
* [`xtile`](xtile.md): Live cross-tile prior injection (`-xtile`)
* [`no_xtile`](no_xtile.md): Disable live cross-tile prior updates (`-no_xtile`)
* [`xtile_decay`](xtile_decay.md): Cross-tile weight decay rate (`-xtile_decay <rate>`)
* [`cpt`](cpt.md): Conditional Probability Table for inter-tile dependencies (`-cpt`)

## Input & Stream Ingestion
* [`input`](input.md): Supported input formats (FITS cubes, text sequences, binary streams)
* [`filelist`](filelist.md): Ingest input as list of image filepaths (`-filelist`)
* [`stream`](stream.md): ImageStreamIO shared-memory stream input (`-stream <name>`)
* [`cnt2sync`](cnt2sync.md): Read synchronization counter for ImageStreamIO (`-cnt2sync <N>`)
* [`shm`](shm.md): Shared memory status stream publication (`-shm <name>`)

## Output, Analysis & Diagnostics
* [`outdir`](outdir.md): Output directory for clustering logs and models (`-outdir <dir>`)
* [`output`](output.md): Overview of all clustering artifact files
* [`clustered`](clustered.md): Generate clustered output dataset file (`-clustered`)
* [`membership`](membership.md): Write per-frame cluster assignment log (`-membership <fname>`)
* [`no_membership`](no_membership.md): Disable cluster membership logging (`-no_membership`)
* [`anchors`](anchors.md): Export exemplar anchor frame references (`-anchors <fname>`)
* [`no_anchors`](no_anchors.md): Suppress exemplar anchor frame output (`-no_anchors`)
* [`counts`](counts.md): Export cluster visitor counts (`-counts <fname>`)
* [`no_counts`](no_counts.md): Suppress cluster visitor counts output (`-no_counts`)
* [`evals`](evals.md): Log frame distance evaluation history (`-evals`, `-no_evals`)
* [`avg`](avg.md): Compute average frame per cluster (`-avg`)
* [`fitsout`](fitsout.md): Force FITS format for multi-dimensional images (`-fitsout`)
* [`pngout`](pngout.md): Export cluster centers as PNG images (`-pngout`)
* [`clusters`](clusters.md): Export cluster centroid coordinate file (`-clusters <fname>`)
* [`maxvis`](maxvis.md): Maximum visitor frames saved per cluster (`-maxvis <N>`)
* [`distall`](distall.md): Save all computed pairwise distances to file (`-distall <fname>`)
* [`analysis`](analysis.md): Offline cluster log analysis tool (`gric-cluster-analysis`)

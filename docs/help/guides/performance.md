# performance

## TUNING OVERVIEW
GRIC is designed for online, real-time clustering. Performance tuning
involves configuring three main components based on your data characteristics:
1. Distance threshold (`rlim`) to control cluster granularity.
2. Spatial/temporal prior models (`-gprob`, `-pred`, `-tm`) to order search candidates.
3. Pruning and acceleration engines (`-entropy`, `-te4`/`-te5`, `-tiles`, `-sparse_dcc`).

## STEP 1: CALIBRATE RLIM (DISTANCE THRESHOLD)
`rlim` is the maximum Euclidean distance between a frame and a cluster anchor.
It determines both cluster resolution and pruning aggressiveness.

1. Inspect natural distance scales:
   `gric-cluster -scandist <input_file>`
   Examine the distance histogram and median sequential distance.

2. Auto-scale syntax (`a<factor>`):
   `gric-cluster a1.5 <input_file>`
   Sets `rlim = 1.5 * median_sequential_distance`.
   - Use `a0.8` - `a1.0` for fine-grained clustering (more clusters).
   - Use `a1.5` - `a2.5` for higher compression (fewer clusters, faster search).

## DATA TYPE: TEMPORALLY CORRELATED / STREAMING VIDEO
For video feeds, sensor streams, or continuous physical processes where
consecutive frames are correlated:

- `-gprob`
  Learns spatial transition topology from visitor history. Ensures recent
  and geometrically adjacent clusters are tested first.

- `-entropy` and `-soft_bayesian`
  Information-optimal target selection with smooth Bayesian likelihood updates.
  Maximizes candidate elimination per distance measurement.

- `-sparse_dcc`
  Avoids full $O(K^2)$ inter-cluster matrix calculations at cluster creation.
  Crucial for high cluster counts ($K > 500$).

- `-tm <coeff>` (0.5 to 1.0)
  For cyclic or repeating trajectories: mixes transition matrix history into
  the prior distribution.

- `-pred[len,h,n]` (e.g. `-pred[5,500,2]`)
  For periodic or repeating motions: scans history for matching sub-sequences
  to predict the next cluster directly.

## DATA TYPE: LARGE IMAGES & HIGH DIMENSIONS
When frame dimension $D$ is large (e.g., 256x256 or 512x512 images) and
distance computation (`framedist`) is the CPU bottleneck:

- `-tiles NxM` (e.g. `-tiles 2x2` or `4x4`)
  Splits the full frame into independent spatial sub-frames:
  - Distance computation cost per sub-frame is reduced by $N \times M$.
  - Local complexity per tile is dramatically lower ($K_{\text{tile}} \ll K_{\text{global}}$).
  - Use `-jtf` (Joint Trajectory Fusion) to correct boundary noise across tiles.

- `-te4` or `-te5`
  Multi-point distance geometry pruning (4-point or 5-point triangle inequality).
  Uses 2 or 3 reference anchors to project points into local coordinates,
  eliminating up to 45% of distance evaluations.

- `-ncpu <N>`
  Parallelizes pruning loops and batch operations across $N$ OpenMP CPU threads.

## DATA TYPE: RANDOM / SHUFFLED DATA POINTS
When data points arrive in random order with no temporal correlation:

- `-entropy`
  Schedules measurements purely by expected information gain without relying
  on temporal recency.

- `-te4`
  Uses metric bounds from past measurements to prune candidates.

- `-dprob 0.0`
  Disables temporal recency bias so search priority is not distorted by
  unrelated previous frames.

- `-ncpu <N>`
  Parallelizes triangle inequality pruning loops when cluster count $K \ge 256$.

## CONTINUOUS STREAMS & MEMORY-CONSTRAINED SYSTEMS
For unbounded 24/7 streaming where memory must remain strictly bounded:

- `-maxcl <N>` (e.g. `-maxcl 2000`)
  Sets an absolute cap on the number of active clusters in memory.

- `-maxcl_strategy discard`
  When `maxcl` is reached, evicts the least-frequently visited clusters.
  Acts as an online LRU cluster cache.

- `-maxcl_strategy merge`
  Merges the two closest clusters, preserving topological coverage at the
  cost of occasional merge recalculations.

- `-sparse_dcc`
  Avoids dense $O(\text{maxcl}^2)$ memory allocations for cluster distance matrices.

## PARAMETER DECISION MATRIX

| Goal / Data Profile | Recommended Options |
| :--- | :--- |
| **Real-time video stream** | `-gprob -entropy -soft_bayesian -sparse_dcc a1.5` |
| **Large high-res frames** | `-tiles 2x2 -jtf -te4 -ncpu 4 a1.5` |
| **Periodic / repeating cycles** | `-gprob -tm 0.8 -pred[5,500,2] a1.2` |
| **Random / static dataset** | `-entropy -te4 -ncpu 8 a1.0` |
| **24/7 infinite stream** | `-maxcl 2000 -maxcl_strategy discard -sparse_dcc a1.5` |

## AUTOMATED TUNING TOOLS
- `gric-tune <input_file>`: Automatically runs a parameter sweep comparing
  tile grids, speed, RMS distortion, and cluster entropy.
- `gric-benchmark`: Runs standardized synthetic and FITS benchmarks.
- `gric-cluster-analysis <outdir>`: Analyzes cluster logs, transition
  matrices, and pruning statistics.

## SEE ALSO
- `-scandist`: Calibrate distance histogram and rlim
- `-tiles`: Image partitioning and multi-tile processing
- `-entropy`: Entropy-based target selection
- `-gprob`: Geometric probability learning
- `algorithm`: Complete algorithmic overview

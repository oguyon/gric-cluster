# tiling

## WHAT IS TILING?
Tiling partitions large input images into a grid of
independent sub-images (tiles). Each tile runs its
own GRIC clustering instance in parallel.

This yields three major benefits:
1. Arithmetic Speedup: Distance computations check smaller sub-frames (e.g. a 2x2 tile distance call is 4x cheaper). Per-tile search complexity drops from O(K) to O(K_tile).
2. Parallelization: OpenMP dispatches one task per tile, running them concurrently.
3. Memory Reduction: Anchors store only tile pixels (e.g. 1/4 the size for 2x2). Simpler local environments mean fewer clusters per tile (K_tile << K), reducing anchor storage by 10-100x and DCC matrices by 100-1000x.

## TILE-BOUNDARY NOISE
The main cost of tiling is tile-boundary noise.

Each tile clusters its sub-frame independently, with no knowledge of what happens in neighboring tiles. When a physical change straddles a tile border (e.g. a moving object crossing from tile 0 into tile 1), one tile may see just enough pixel change to switch clusters while the other does not. The result is a spurious joint state: the per-tile assignments are individually plausible but globally inconsistent.

This manifests as rapid cluster flickering in tiles near the boundary of a physical event, inflating the number of distinct joint states far beyond the true number of physical configurations.

Joint Trajectory Fusion (Pass 2) detects and corrects these misassignments by comparing each frame's per-tile assignment tuple against recent history. If the observed tuple has never (or rarely) occurred before but is one tile-flip away from a common historical pattern, JTF overrides the outlier tile's assignment to match the known pattern.

## TWO-PASS PIPELINE
Each frame goes through two passes (Pass 2 is optional and enabled via the -jtf option):

Pass 1: Independent Spatial Clustering (ISC)
  The full-image frame is scattered into per-tile sub-frames (pixel extraction, not interpolation). Each tile independently runs the standard GRIC clustering pipeline (predict priors, select target, measure distance, update & prune, assign). All tile tasks execute in parallel via OpenMP. Result: one cluster assignment per tile.

Pass 2: Joint Trajectory Fusion (JTF)
  After all tiles complete Pass 1, JTF corrects tile-boundary noise by leveraging cross-tile correlations from recent history.

  For each tile, JTF:
  1. Builds a spatial key from the current Pass 1 assignments of all tiles.
  2. Builds a temporal key from the previous frame's assignment tuple.
  3. Scans the last -retrieval_window tuples for matching spatial+temporal patterns.
  4. Multiplies the Pass 1 posterior by the match scores. If the fused argmax differs from the Pass 1 assignment, verifies that the distance to the target cluster anchor is within rlim before overriding.

  JTF never overrides a newly created cluster (Pass 1 produced a new anchor). It only refines existing-cluster assignments, and only if the distance to the target cluster's anchor is within rlim.

  Example (2x2 tiling, 4 tiles):

  A tuple is the vector of per-tile cluster assignments for one frame. With 4 tiles, each frame produces a tuple like (0, 3, 2, 1), meaning tile 0 was assigned to cluster 0, tile 1 to cluster 3, etc.

  Suppose the recent tuple history contains:
  - frame 997: (0, 3, 2, 1)
  - frame 998: (0, 3, 2, 1)
  - frame 999: (0, 3, 2, 1)

  Now frame 1000 arrives. Pass 1 assigns:
    tile 0 -> cluster 0
    tile 1 -> cluster 3
    tile 2 -> cluster 5   (boundary flicker)
    tile 3 -> cluster 1
  Raw tuple: (0, 3, 5, 1)

  JTF processes tile 2:
  - Spatial key: the other tiles' Pass 1 results = (0, 3, -, 1)
  - Temporal key: previous frame's tuple = (0, 3, 2, 1)
  - Scans history for tuples where tiles 0, 1, 3 matched (0, 3, -, 1) spatially and the predecessor matched (0, 3, 2, 1) temporally.
  - Finds frames 998 and 999 match. Both had tile 2 = cluster 2.
  - Match scores (from history): cluster 2 = 0.95, cluster 5 = 0.01, all others ~0.
  - Element-wise multiply: for each cluster k, fused[k] = pass1_posterior[k] * match_score[k]. Pass 1 gave cluster 5 a posterior of 0.6 and cluster 2 a posterior of 0.3. After multiplication: fused[5] = 0.6 * 0.01 = 0.006, fused[2] = 0.3 * 0.95 = 0.285.
  - Argmax of fused posterior: cluster 2 (0.285 > 0.006).
  - Verifies distance to anchor 2: d(frame, anchor_2) = 11.2, which is <= rlim (13.3).
  - JTF overrides tile 2: cluster 5 -> cluster 2.
  Corrected tuple: (0, 3, 2, 1)

Data flow:
  Full frame
    |
    v
  Scatter (pixel extraction per tile)
    |
    +--- Tile 0 ---> Pass 1 (ISC) ---+
    +--- Tile 1 ---> Pass 1 (ISC) ---+
    +--- Tile 2 ---> Pass 1 (ISC) ---+
    +--- Tile 3 ---> Pass 1 (ISC) ---+
                                     |
                            [taskwait barrier]
                                     |
                                     v
                  Joint Trajectory Fusion (Pass 2)
                                     |
                                     v
                        Record assignment tuple

## SEQUENCE PREDICTION (-pred)
The sequence prediction option (-pred) is a proactive search optimization.

Before Pass 1 begins, the system scans recent joint assignment tuple history using the previous frames' tuples (up to len steps, configured via -pred[len,h,n]) as a query sequence. It retrieves historical matching joint trajectories and assigns transitions a similarity score.

For each tile, these joint scores are used to:
1. Bias the search priors by seeding state.scratch.mixed_probs.
2. Prioritize search order by placing predicted clusters on the shortcut list (tuple_pred_candidates).

This proactive optimization guides the individual search loops before clustering starts, speeding up execution without altering the final spatial clustering boundaries.

## JOINT TRAJECTORY FUSION (-jtf)
The Joint Trajectory Fusion option (-jtf) is a reactive trajectory-smoothing correction.

After all tiles complete Pass 1 independent clustering, JTF builds a spatial query key from the current frame's preliminary Pass 1 tile assignments. It scans history for similar joint layout patterns across recent frames, computing a global spatial-temporal match score for each candidate.

If an individual tile's Pass 1 assignment is noisy (e.g. boundary flickering) but is one tile-flip away from a highly probable historical pattern, JTF overrides the noisy assignment in favor of the globally consistent joint trajectory. The override is accepted only if the distance to the target cluster anchor satisfies the rlim hard threshold constraint.

## TILING OPTIONS
-tiles <NxM>
  Grid size. N columns, M rows. Example: -tiles 2x2 creates 4 tiles.

-tileconf <file.txt>
  Per-tile configuration overrides. ASCII file, one line per tile,
  3 space-separated fields: tile_id rlim maxnbclust. Example: '0 0.8 500'
  sets tile 0 to rlim=0.8 with maxnbclust=500.

-xtile [mode]
  Enable live cross-tile prior injection. Mode 1: spatial-only CPT.
  Mode 2: hybrid spatial-temporal (Strategy C, default). Disabled by default.

-xtile_decay <val>
  Exponential decay coefficient for CPT history (default: 1.0).

-jtf
  Enable Joint Trajectory Fusion (Pass 2) to correct tile-boundary noise.
  Disabled by default.

-retrieval_window <N>
  Number of recent frames to scan during JTF pattern matching (default: 1000).
  Larger values improve accuracy but increase scan cost linearly.

## OPTION INTERACTIONS
All per-tile options (-gprob, -entropy, -soft_bayesian, -sparse_dcc, -te4/-te5) work independently within each tile during Pass 1. Each tile has its own ClusterState, DCC matrix, visitor history, and entropy scheduler.

-pred (sequence prediction)
  Operates at the joint tuple level before Pass 1. predict_joint_tuples() scans tuple history for matching multi-tile patterns, then populates mixed priors and candidate lists for all tiles.

-tm (transition matrix)
  Operates independently per tile during Pass 1.

-ncpu <N>
  Tiles run as OpenMP tasks. When tiles < cpus, nested parallelism is enabled so each tile can use intra-tile parallelism (e.g. in framedist).

## TUNING GUIDELINES
1. Grid Resolution:
   Avoid partitioning too finely (e.g. 4x4 on a 32x32 image). As grid size M increases, joint state combinations grow as k^M, causing combinatorial explosion. OpenMP task scheduling overhead also rises. Recommend 2x2 for small/medium scientific sensors.

2. Calibrating rlim:
   As tile size shrinks, the maximum distance between sub-frames drops. Scale rlim proportionally to sqrt(tile_pixel_count). Use -tileconf for per-tile rlim overrides when tiles cover regions with different dynamic ranges.

3. Retrieval Window Sizing:
   The default (1000) works for most data. Increase to 5000-10000 for slowly-varying signals with rare transitions. Decrease to 100-500 for rapidly-switching data where old patterns are irrelevant.

## SEE ALSO
- `performance`: How to pick options for best performance
- `algorithm`: Overview of the GRIC algorithm
- `compression`: State space compression for tiled data

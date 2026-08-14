# compression

## WHAT IS STATE SPACE COMPRESSION?
State space compression measures how efficiently GRIC represents
the joint spatial-temporal states of a multi-tile system.

When clustering tiles independently, local noise near boundaries
frequently causes tiles to assign frames to mismatched clusters,
creating a massive number of spurious joint state combinations
(tuples). State space compression resolves these noisy states
by fusing boundary assignments into physically consistent paths.

## HOW TRAJECTORY FUSION WORKS
1. Independent Spatial Clustering (Pass 1): Each tile clusters its sub-frame, yielding a raw joint state tuple U = (c_0, c_1, ...).
2. Joint Trajectory Fusion (Pass 2): Scans the global history of resolved states within a lookback window (H), identifying frames with spatially similar trajectory patterns.
3. Bayesian Correction: Computes transition priors by accumulating historical transition evidence. A high-contrast prior overrides local boundary noise, collapsing fragmented assignments into the same clean physical trajectories.

## TUNING GUIDELINES
1. Tuning lookback horizon (-retrieval_window <H>):
   - Small H (e.g. < 200): Small sample size yields weak transition statistics, leading to poor error correction and low compression.
   - Optimal H (typically 1,000 to 10,000): Collects robust joint evidence, filtering out random boundary fluctuations.
   - Excessive H (e.g. > 20,000): Concepts/clusters undergo drift and recycling over very long horizons. Stale memory acts as noise, degrading the compression quality.

2. Impact of Tiling Grid (-tiles <NxM>):
   Keep grid resolution balanced (recommend 2x2). High grid sizes
   (e.g., 4x4) treat ball motion as independent variables, causing
   a combinatorial state explosion (k^M unique tuples) that completely
   destroys spatial correlations and prevents compression.

## SEE ALSO
- `tiling`: Image partitioning and multi-tile processing
- `performance`: How to pick options for best performance

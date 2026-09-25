# milk_streams

## ROLE
Milk Streaming Shared Memory Interface & Output Telemetry Layout

## FUNCTION
Defines the real-time shared memory streams created and consumed by the GRIC
Milk FPS integration (`milk-fpsexec-gric-cluster`).

## INPUT STREAM REQUIREMENTS
- Managed by ImageStreamIO in POSIX shared memory (/dev/shm).
- 2D [W, H] or 3D circular buffer [W, H, S] layout.
- Supported datatypes: FLOAT, DOUBLE, UINT16, UINT8 (auto-converted to float32).
- Synchronized using dedicated reader semaphores without multi-reader races.

## OUTPUT ASSIGNMENT STREAM (<out_name>_assign)
Published on every ingested frame. Allocated as a 1D [8, 1] float vector:
- [0] cnt0: Input frame index counter.
- [1] cluster_id: Assigned cluster ID (0 <= k < K).
- [2] distance: Measured Euclidean distance to assigned cluster anchor.
- [3] is_new_anchor: 1.0 if this frame spawned a new anchor, 0.0 if matched.
- [4] cluster_count: Total active clusters currently discovered (K).
- [5] latency_us: Single-frame processing latency in microseconds.
- [6] evals: Number of metric distance calculations evaluated for this frame.
- [7] reserved: Auxiliary telemetry / quality score.

## CENTROIDS & ANCHORS STREAM (<out_name>_anchors)
Published when .stream_anchors is ON.
- Dimensions: 2D [W*H, maxnbclust] or 3D [W, H, maxnbclust].
- Updated and signaled whenever a new cluster anchor is spawned.

## CLUSTER POPULATIONS STREAM (<out_name>_counts)
Published when .stream_counts is ON.
- Dimensions: 1D [maxnbclust, 1] uint32 array.
- Contains the cumulative number of frames assigned to each cluster.

## SEE ALSO
- `milk`: Milk framework overview
- `milk_fpsexec`: Standalone FPS daemon manual
- `shm`: Shared memory telemetry monitoring

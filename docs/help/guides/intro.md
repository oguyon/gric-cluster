# intro

## WHAT IS GRIC?
GRIC (Geometric Real-time Image Clustering)
groups frames by visual similarity.

Each frame is compared against existing
cluster anchors. If a close enough match is
found (within rlim), the frame joins that
cluster. Otherwise a new cluster is created.

GRIC is single-pass and sequential: each
frame is processed exactly once, in order,
making it suitable for real-time streams.

## TYPICAL WORKFLOW
1. Prepare input
   Text file, FITS cube, MP4 video,
   or live ImageStreamIO stream.

2. Calibrate the distance threshold
     gric-cluster -scandist input.txt
   Use the reported median distance as
   the rlim value.

3. Run clustering
     gric-cluster a1.5 input.txt
   The 'a' prefix means auto-scale:
   rlim = 1.5 x median distance.

4. Inspect outputs
   Results are written to
   input.txt.clusterdat/

## KEY CONCEPTS
rlim
  Distance threshold. Frames within this
  distance of a cluster anchor are assigned
  to that cluster. Smaller rlim = more
  clusters; larger rlim = fewer.

Anchor
  The representative frame that defines a
  cluster. All distances are measured to
  anchors, not between all frame pairs.

DCC (Distance between Cluster Centers)
  Matrix of inter-anchor distances. Used
  by triangle inequality to skip
  unnecessary distance computations.

Pruning
  Eliminating candidate clusters without
  measuring them. The main source of
  GRIC's speed advantage.

## SEE ALSO
- `algorithm`: How the per-frame loop works
- `performance`: Choosing options for best speed
- `input`: Supported input formats
- `output`: What output files are produced

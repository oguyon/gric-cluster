# rlim

## ROLE
Distance Threshold

## FUNCTION
Maximum Euclidean distance between a frame
and a cluster anchor for the frame to be
assigned to that cluster.

rlim is the first positional argument:
  gric-cluster 0.5 input.txt

Prefix with 'a' for auto-scaling:
  gric-cluster a1.5 input.txt
  (rlim = 1.5 x median sequential distance)

## CHOOSING RLIM
Too small: every frame creates its own cluster
  (over-fragmentation).
Too large: distinct states merge into one cluster
  (under-segmentation).

Recommended workflow:
  1. gric-cluster -scandist input.txt
     Inspect the distance histogram.
  2. Start with rlim = 0.5 x median distance
     and adjust based on cluster count.
  3. Or use auto-mode: gric-cluster a1.0 input.txt

## ROLE IN PRUNING
rlim also defines the pruning radius. Triangle
inequality eliminates cluster B after measuring A
when |d(frame,A) - d(A,B)| > rlim. A smaller rlim
makes pruning more aggressive (fewer measurements
per frame).

## SEE ALSO
- `auto_rlim`: Auto-scaled rlim syntax
- `-scandist`: Measure distance stats
- `algorithm`: Algorithm overview

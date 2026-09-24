# pass2nearest

## OVERVIEW
`-pass2nearest` (aliases: `-reassign`, `-second_pass`) enables a second pass
over all clustered frames to reallocate each frame to its globally closest
cluster anchor point.

## MOTIVATION
In initial sequential online clustering (Pass 1):
1. Early frames are never tested against clusters created later in the sequence.
2. Search stops upon finding the first cluster anchor within distance `rlim`,
   which may not be the globally nearest anchor.

## ALGORITHM
Pass 2 retains all distances computed during Pass 1 in memory.
For unmeasured candidate anchors, it uses triangle-inequality lower bounding:
  LB = |d(frame_t, anchor_m) - DCC(anchor_m, anchor_u)|
If LB >= d_best(frame_t), evaluation of anchor_u is pruned away.
Otherwise, the distance is evaluated and d_best is updated.

## OUTPUT UPDATES
When enabled:
- `frame_membership.txt` is updated with closest anchor assignments.
- `transition_matrix.txt` is rebuilt from the updated sequence.
- Cluster counts, radii, and quality metrics reflect the reassigned memberships.

## SEE ALSO
- `clustering`: Core clustering options
- `algorithm`: Overview of GRIC pipeline

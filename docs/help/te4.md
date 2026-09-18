# te4

## ROLE
4-Point Pruning

## FUNCTION
Enables aggressive pruning using 4 points.

## ALGORITHM
Standard pruning uses 3 points (Triangle Inequality: d(A,C) <= d(A,B) + d(B,C)).
TE4 uses 2 reference clusters (A, B) + Current Frame (F) + Candidate (C).
It establishes a 2D plane with A, B, F to bound the distance to C more strictly.
Reduces expensive distance calls at the cost of slightly more complex logic.

### ANCHOR PAIR SELECTION CAP
Most of the pruning power of TE4 comes from the 2 or 3 most informative (closest to frame)
and geometrically orthogonal anchors. To eliminate redundant pairwise checks and DCC memory
bandwidth, TE4 checks are capped to the top 2–3 historical anchors by default.

## OPTIONS
- `-te4`: Enable 4-point triangle inequality pruning
- `-te4_max_anchors <N>`: Set max historical anchors checked per frame (default: 3, 0 = uncapped)
- `-no_te4`: Disable 4-point pruning

## SEE ALSO
- `-te5`: Use 5-point triangle inequality pruning

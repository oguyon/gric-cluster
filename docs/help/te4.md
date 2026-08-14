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

## SEE ALSO
- `-te5`: Use 5-point triangle inequality pruning

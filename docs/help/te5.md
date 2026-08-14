# te5

## ROLE
5-Point Pruning

## FUNCTION
Enables aggressive pruning using 5 points.

## ALGORITHM
Uses 3 reference clusters + Current Frame + Candidate.
It constructs a local 3D coordinate system to strictly bound the possible
distance range. Effective for high-dimensional data where simple triangle
inequalities are loose.

## USE
-te5 (Recommended for high-dimensional vectors)

## SEE ALSO
- `-te4`: Use 4-point triangle inequality pruning

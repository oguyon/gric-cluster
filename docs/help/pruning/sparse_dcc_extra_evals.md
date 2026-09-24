# sparse_dcc_extra_evals

## ROLE
Sparse DCC Parameter

## FUNCTION
Sets the number of extra inter-cluster distance evaluations to perform when
creating a new cluster (default: 0).

## RATIONALE
Evaluating a small number of extra distances (e.g. 2 or 5) helps tighten the
lower/upper bounds of the sparse DCC matrix, improving subsequent triangle
inequality pruning efficiency.

## REQUIRES
-sparse_dcc (has no effect without it)

## SEE ALSO
- `-sparse_dcc`: Sparse DCC matrix
- `-dcc`: Enable dcc.txt output

# sparse_dcc

## ROLE
Cluster-to-Cluster Distance Optimization

## FUNCTION
Enables bounded sparse inter-cluster distance matrix tracking to avoid dense
O(K^2) anchor distance calls.

## ALGORITHM
Reuses ongoing sample-to-anchor measurements to calculate triangle inequality
bounds, maintaining lower/upper bounds for unmeasured anchor distances.
Highly recommended for video input.

## TUNED BY
-sparse_dcc_extra_evals <val>
  Extra DCC entries computed per step.
  Higher = tighter bounds, more CPU.

## INTERACTS WITH


## SEE ALSO
- `-dcc`: Enable dcc.txt output

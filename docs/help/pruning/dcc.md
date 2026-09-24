# dcc

## ROLE
Output Control & Inter-Cluster Distance Matrix

## FUNCTION
Writes the pairwise inter-cluster distance matrix to the output directory (Default: Enabled).

Outputs include:
- `dcc.txt`: ASCII list formatted as `Cluster_i Cluster_j Distance`.
- `dcc.bin`: Float32 matrix in row-major GRIC binary format.
- `dccmin.txt`: Generated when `-sparse_dcc` is enabled to log sparse lower bounds.

To disable DCC output completely, pass `-no_dcc`.

## SEE ALSO
- `-no_dcc`: Suppress DCC distance matrix output
- `-sparse_dcc`: Sparse DCC bounding mode
- `-outdir`: Specify output directory

# no_dcc

## ROLE
Output Control

## FUNCTION
Disables writing the inter-cluster distance
matrix to 'dcc.txt'. DCC output is enabled
by default.

## RATIONALE
DCC output can be large for many clusters.
Disable it to reduce disk I/O when the
inter-cluster distance matrix is not needed.

## SEE ALSO
- `-dcc`: Enable dcc.txt output

# counts

## ROLE
Output Control

## FUNCTION
Writes cluster visitor and assignment counts to disk (Default: Enabled).

Outputs include:
- `cluster_counts.txt`: ASCII table listing member count for each cluster.
- `cluster_counts.bin`: GRIC binary representation for rapid binary loading.

To suppress writing cluster counts, pass `-no_counts`.

## SEE ALSO
- `-no_counts`: Suppress cluster counts output
- `-outdir`: Specify output directory
- `-membership`: Frame membership logging

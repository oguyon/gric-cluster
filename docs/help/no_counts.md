# no_counts

## ROLE
Output Control

## FUNCTION
Suppresses writing cluster assignment counts (`cluster_counts.txt` and `cluster_counts.bin`)
to the output directory.

By default, cluster counts output is enabled. Passing `-no_counts` disables writing the
count files.

## USE
```bash
# Cluster without writing cluster_counts.txt
gric-cluster 0.5 input.txt -no_counts
```

## SEE ALSO
- `-counts`: Enable cluster counts output (default)
- `-outdir`: Specify output directory

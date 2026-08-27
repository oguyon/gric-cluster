# no_anchors

## ROLE
Output Control

## FUNCTION
Suppresses writing cluster anchor frames (`anchors.txt`, `anchors.bin`, or `anchors.fits`)
to the output directory.

By default, cluster anchor output is enabled. Passing `-no_anchors` disables saving anchors,
reducing disk I/O when only membership indices or cluster counts are required.

## USE
```bash
# Cluster without saving anchor data to disk
gric-cluster 0.5 input.txt -no_anchors
```

## SEE ALSO
- `-anchors`: Enable anchor output (default)
- `-outdir`: Specify output directory

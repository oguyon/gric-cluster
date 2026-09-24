# txt

## ROLE
ASCII Text Output Format

## FUNCTION
Forces `gric-cluster` and `gric-knn` to write output artifacts as human-readable plain text files
instead of the default binary `.bin` format.

## DETAILS
- By default, GRIC writes typed binary files (`.bin`) for high throughput and zero-copy mapping.
- Specifying `-txt` writes legacy `.txt` files (such as `anchors.txt`, `dcc.txt`,
  `frame_membership.txt`, and `cluster_counts.txt`).
- `-no-txt`: Suppresses ASCII text output (default behavior).

## OPTIONS
- `-txt`: Write output artifacts in ASCII plain text format
- `-no-txt`: Suppress ASCII text output (write binary `.bin` files only)

## SEE ALSO
- `-outdir`: Output directory for clustering artifacts
- `-fitsout`: Force FITS image cube output format

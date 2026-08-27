# filelist

## ROLE
Input Data Ingestion Mode

## FUNCTION
`-filelist` instructs `gric-cluster` to interpret the input file as a plain text list of image
file paths (one file path per line) rather than a single multi-frame data file or coordinate table.

## USE
```bash
# Ingest frames from a text file containing image paths
gric-cluster 0.5 image_paths.txt -filelist -outdir results/
```

## SEE ALSO
- `-input`: Input format overview
- `-stream`: Shared memory streaming input

# avg

## ROLE
Output Generation

## FUNCTION
Computes the average frame for each cluster.

## IMPLEMENTATION
Accumulates pixel data for every frame assigned to a cluster. At the end,
divides by the count. Useful for 'Lucky Imaging' or noise reduction.

## SEE ALSO
- `-outdir`: Specify output directory
- `-pngout`: Write output as PNG images
- `-fitsout`: Force FITS output format

# anchors

## ROLE
Output Control

## FUNCTION
Writes the exemplar anchor frames (the initial spawning frame for each cluster) to disk
in the output directory (Default: Enabled).

Outputs are written in both self-describing binary format (`anchors.bin`) and formatted
media (`anchors.txt`, `anchors.fits`, or `anchor_%04d.png` depending on input type and flags).

To suppress anchor writing, pass `-no_anchors`.

## SEE ALSO
- `-no_anchors`: Suppress anchor output
- `-outdir`: Specify output directory
- `-fitsout`: Force FITS output format
- `-pngout`: Export individual anchor frames as PNG images

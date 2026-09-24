# double

## ROLE
64-Bit Double-Precision Floating-Point Mode

## FUNCTION
Forces metric computations, distance calculations, anchor vectors, and intermediate accumulators to
use IEEE 754 64-bit double precision (`double`) instead of 32-bit single precision (`float`).

## DETAILS
- Useful for datasets where coordinate values span wide dynamic ranges or where sub-nanometer
  precision is required.
- Automatically selects double-precision AVX2 / AVX-512 vector kernels when available.
- Binary outputs (`anchors.bin`, `dcc.bin`, `membership.bin`) write `FLOAT64` data type headers.

## OPTIONS
- `-double`: Force 64-bit double-precision mode (default: single-precision `float32`)

## SEE ALSO
- `-rlim`: Distance threshold for cluster membership
- `-fitsout`: Force FITS output format

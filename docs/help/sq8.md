# sq8

## ROLE
8-Bit Scalar Quantization Filtering

## FUNCTION
Compresses vector dimensions into unsigned 8-bit bytes (`uint8_t`), achieving $4\times$ memory
compression and maximum SIMD lane occupancy.

## ALGORITHM
Maps vector coordinates into 256 discrete bins:
- Best suited for low-to-medium dimensional spaces ($D \le 32$) or datasets with bounded dynamic
  range.
- Enables 32-way concurrent evaluation on 256-bit AVX2 registers and 64-way on 512-bit AVX-512
  registers.
- Pruning bounds account for the wider 8-bit quantization granularity to preserve metric recall.

## OPTIONS
- `-sq8`: Enable 8-bit scalar quantization filtering
- `-no-sq8`: Disable 8-bit scalar quantization filtering

## SEE ALSO
- `-sq16`: 16-bit scalar quantization filtering
- `-eq16`: 16-bit E8 lattice quantization filtering

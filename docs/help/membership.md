# membership

## ROLE
Output Control

## FUNCTION
Writes per-frame cluster assignments to disk (Default: Enabled).

Outputs include:
- `frame_membership.txt`: ASCII list of `FrameIndex AssignedClusterIndex`.
- `frame_membership.bin`: UINT32 array in GRIC binary format for high-speed indexing.

To suppress frame membership output, pass `-no_membership`.

## SEE ALSO
- `-no_membership`: Disable frame_membership output
- `-counts`: Export cluster assignment counts
- `-outdir`: Specify output directory

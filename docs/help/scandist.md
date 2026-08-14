# scandist

## ROLE
Data Analysis (Pre-run)

## FUNCTION
Measures distance statistics without
clustering. Reports Min, Max, Median,
20th and 80th percentile distances.
Use this to calibrate rlim.

## AUTO-RLIM
Instead of running -scandist manually,
use the 'a' prefix for auto-scaling:
  gric-cluster a1.5 input.txt
This runs scandist internally and sets
rlim = 1.5 x median distance.

## USE
gric-cluster -scandist input.txt

## SEE ALSO
- `rlim`: Distance threshold details
- `auto_rlim`: Auto-scaled rlim syntax

# auto_rlim

## ROLE
Automatic Distance Threshold

## FUNCTION
When the first positional argument starts with
'a', gric-cluster runs a scandist pass first,
then sets rlim = factor x median distance.

  gric-cluster a1.5 input.txt
  equivalent to:
    1. gric-cluster -scandist input.txt
    2. rlim = 1.5 x reported median
    3. gric-cluster <rlim> input.txt

## GUIDELINES
a0.5   Tight:  many small clusters
a1.0   Medium: balanced segmentation
a1.5   Loose:  fewer, broader clusters
a2.0+  Very loose: coarse grouping only

## USE
gric-cluster a1.2 input.txt

## SEE ALSO
- `rlim`: Distance threshold details
- `-scandist`: Measure distance stats

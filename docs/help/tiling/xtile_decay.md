# xtile_decay

## ROLE
CPT Trajectory Decay Control

## FUNCTION
Sets the decay coefficient for historical co-occurrences
in the CPT table. Values should be in range (0.0 to 1.0].
1.0 means no decay (full memory).

## RATIONALE
Allows the system to discount old history and adjust to
transient trajectories or evolving target behavior.

## USE
-xtile_decay <val>

## REQUIRES
-tiles NxM -xtile

## SEE ALSO
- `-xtile`: Cross-tile prior injection

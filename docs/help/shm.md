# shm

## ROLE
Real-Time Monitoring

## FUNCTION
Exports clustering telemetry to a shared
memory status file for real-time monitoring
by gric-status or other consumers.

## DETAILS
The SHM file contains counters, timing data,
entropy diagnostics, and per-frame statistics
updated continuously during the clustering
run.

## USE
-shm /tmp/gric_status.shm

## SEE ALSO
- `gric-status`: Monitor SHM telemetry (TUI)

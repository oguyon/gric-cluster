# milk

## ROLE
Milk Framework Integration and CLI Plugin

## FUNCTION
Integrates GRIC with the Milk high-performance astronomical adaptive optics
framework (framework-dev). Provides shared memory stream processing and the
gric_cluster shell command inside the Milk CLI via libmilkgric.so.

## ARCHITECTURE
- libgric: Pure Level 2 clustering compute library (zero Milk dependency).
- libmilkgric.so: Level 3 shared module plugin for the Milk interactive CLI.
- milk-fpsexec-gric-cluster: Standalone Function Parameter Structure daemon.
- ImageStreamIO: Real-time POSIX shared memory ring buffers and semaphores.
- libprocessinfo: Heartbeat monitoring, loop timing, and milk-procCTRL TUI.

## MILK CLI USAGE
Within the Milk interactive shell:
  milk> loadmodule "milkgric"
  milk> gric_cluster -in_name imrec1 -out_name clust -rlim 0.45

## SEE ALSO
- `milk_fpsexec`: Standalone FPS clustering daemon manual
- `milk_streams`: Shared memory stream output specifications
- `shm`: ImageStreamIO shared memory interface

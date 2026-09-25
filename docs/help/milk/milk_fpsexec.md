# milk_fpsexec

## ROLE
Standalone Function Parameter Structure (FPS) Streaming Clustering Daemon

## FUNCTION
Runs GRIC as a real-time, zero-copy streaming clustering process synchronized to
input ImageStreamIO frames via POSIX semaphores. Exposes runtime parameters in
shared memory for dynamic adjustment via milk-fps-set or milk-fpsCTRL.

## BINARY NAMES
- milk-fpsexec-gric-cluster (canonical FPS executable)
- gric-fps-cluster (convenience symlink)

## SUBCOMMANDS
- fpsinit [name]          Initialize FPS parameter structure in shared memory
- fps [name]              Display current FPS parameters and active values
- fpslist                 List all known FPS instances on the system
- confstart [name]        Enter background configuration loop
- confstop [name]         Stop configuration loop (sent to tmux ctrl window)
- runstart [name]         Enter main real-time frame processing loop
- runstop [name]          Stop real-time processing loop (sent to ctrl window)
- set [name] <args>       Set positional parameters (. to skip unchanged)
- exec [name] <args>      Auto-init + configure arguments + start run loop

## OPTIONS
- -n <name>               Set custom FPS instance name (default: gric_cluster)
- -tmux                   Execute process inside an isolated tmux session
- -procinfo               Enable ProcessInfo heartbeat and rate telemetry
- -loops                  Infinite loop triggered by input stream semaphores
- -loopd <sec>            Infinite loop triggered on a fixed delay timer

## PARAMETERS
- .in_name                Input ImageStreamIO stream name (TRIGGER)
- .out_name               Output assignment stream name (<out>_assign)
- .out_anchors            Output cluster centroids stream name
- .out_counts             Output cluster population counts stream name
- .stream_anchors         Publish anchors stream live (ON/OFF)
- .stream_counts          Publish counts stream live (ON/OFF)
- .allow_frame_drop       Lag policy: 1=jump to latest frame, 0=lossless
- .rlim                   Cluster radius threshold (Euclidean distance)
- .deltaprob              Neighbor search probability cutoff
- .maxnbclust             Maximum cluster allocation ceiling (default: 256)
- .maxcl_strategy         Capacity policy: 0=stop clustering, 1=discard
- .ncpu                   OpenMP worker thread count (0 = auto-detect)
- .use_double             Use 64-bit float precision (default: OFF, 32-bit)
- .use_sq16               Enable 16-bit scalar quantization filter
- .entropy_mode           Enable Shannon entropy target selection
- .reset_state            Dynamic trigger: clears existing clusters on-the-fly

## DYNAMIC TUNING
Parameters can be modified live without stopping the streaming daemon:
  milk-fps-set gric_cluster.rlim 0.35
  milk-fps-set gric_cluster.allow_frame_drop ON
  milk-fps-set gric_cluster.reset_state ON

## SEE ALSO
- `milk`: Milk framework integration overview
- `milk_streams`: Output telemetry vector and image stream layout

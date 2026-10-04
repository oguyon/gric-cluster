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
- .cnt2sync               Flow-control synchronization handshake toggle
- .query_mode             Run in fixed-cluster classification query mode
- .load_anchors           Pre-existing cluster anchor file (.bin) to load
- .rlim                   Cluster radius threshold (Euclidean distance)
- .deltaprob              Neighbor search probability cutoff
- .maxnbclust             Maximum cluster allocation ceiling (default: 256)
- .maxcl_strategy         Capacity policy: 0=stop clustering, 1=discard
- .max_frames             Upper limit on frames to process (0 = infinite)
- .ncpu                   OpenMP worker thread count (0 = auto-detect)
- .use_double             Use 64-bit float precision (default: OFF, 32-bit)
- .use_sq16               Enable 16-bit scalar quantization filter
- .entropy_mode           Enable Shannon entropy target selection
- .reset_state            Dynamic trigger: clears existing clusters on-the-fly
- .shm_status_file        Bridge status SHM file path (blank = default)
- .save_dir               Destination directory to save cluster results
- .status.frames_processed Total frames ingested and processed
- .status.num_clusters    Active clusters discovered
- .status.new_clusters    Total newly created clusters
- .status.distance_evals  Total metric distance computations
- .status.pruning_ratio   Metric bounding pruning ratio (skipped fraction)
- .status.latency_us      Per-frame clustering latency in microseconds
- .status.fps             Instantaneous processing rate in frames per second
- .status.stream_lag      Input ImageStreamIO write - read latency lag
- .status.memory_rss_mb   Process resident set memory size in megabytes

## USAGE EXAMPLES
Initialize FPS with ProcessInfo telemetry:
  $ milk-fpsexec-gric-cluster -procinfo demo:fpsinit

Inspect active parameters and ProcessInfo telemetry:
  $ milk-fpsexec-gric-cluster -procinfo demo:fps

Start real-time frame processing loop:
  $ milk-fpsexec-gric-cluster -tmux -procinfo -loops demo:runstart

Stop processing loop:
  $ milk-fpsexec-gric-cluster -procinfo demo:runstop

## DYNAMIC TUNING
Parameters can be modified live without stopping the streaming daemon:
  milk-fps-set gric_cluster.rlim 0.35
  milk-fps-set gric_cluster.allow_frame_drop ON
  milk-fps-set gric_cluster.reset_state ON

## SEE ALSO
- `milk`: Milk framework integration overview
- `milk_streams`: Output telemetry vector and image stream layout

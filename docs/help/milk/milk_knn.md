# milk_knn

## ROLE
Standalone Function Parameter Structure (FPS) Streaming k-NN Search Daemon

## FUNCTION
Runs real-time nearest neighbor queries on streaming ImageStreamIO vector inputs
against pre-computed Pass 1 cluster anchor codebooks and reference datasets.

## BINARY NAMES
- milk-fpsexec-gric-knn (canonical FPS executable)
- gric-fps-knn (convenience symlink)

## PARAMETERS
- .in_name                Input ImageStreamIO query stream name (TRIGGER)
- .out_name               Output k-NN matches stream name
- .cnt2sync               Flow-control synchronization handshake toggle
- .allow_frame_drop       Lag policy: 1=jump to latest frame, 0=lossless
- .cluster_dir            Pass 1 cluster directory (.clusterdat)
- .ref_data               Reference dataset file (.bin, .fits, .txt)
- .k                      Target number of nearest neighbors (k)
- .rlim                   Max distance cutoff (0 = unbounded)
- .eps                    Relaxation factor for (1+eps)-ANN
- .max_frames             Upper limit on queries to process (0 = infinite)
- .ncpu                   OpenMP worker thread count (0 = auto-detect)
- .use_double             Use 64-bit float precision
- .use_sq8                Enable 8-bit scalar quantization filter
- .use_sq16               Enable 16-bit scalar quantization filter
- .use_eq16               Enable 16-bit E8 lattice quantization filter
- .use_rq8                Enable 8-bit residual quantization filter
- .approx_mode            Fast approximate graph search toggle
- .use_cluster_graph      Graph-guided cluster routing toggle
- .shm_status_file        Bridge status SHM file path (blank = default)
- .status.queries_processed Total queries processed
- .status.k               Active neighbor count k
- .status.distance_evals  Total metric distance computations
- .status.latency_us      Per-frame query latency in microseconds
- .status.fps             Instantaneous processing rate in frames per second
- .status.min_dist        Distance to 1-NN nearest neighbor
- .status.max_dist        Distance to kth neighbor (tau)
- .status.stream_lag      Input stream lag (write - read)
- .status.memory_rss_mb   Process resident set memory size in megabytes

## SEE ALSO
- `milk`: Milk framework integration overview
- `milk_fpsexec`: Standalone FPS daemon manual
- `milk_streams`: Output stream layout and telemetry

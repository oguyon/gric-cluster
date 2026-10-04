# milk_recon

## ROLE
Standalone Function Parameter Structure (FPS) Streaming Dataset Reconstruction Daemon

## FUNCTION
Performs real-time dataset reconstruction and inverse distance weighting (IDW)
averaging across streaming k-NN neighbor indices and target feature spaces.

## BINARY NAMES
- milk-fpsexec-gric-reconstruct (canonical FPS executable)
- gric-fps-reconstruct (convenience symlink)

## PARAMETERS
- .in_name                Input k-NN matches stream [k x 2] (TRIGGER)
- .out_name               Output reconstructed stream name
- .cnt2sync               Flow-control synchronization handshake toggle
- .allow_frame_drop       Lag policy: 1=jump to latest frame, 0=lossless
- .target_data            Target dataset B file (.bin, .fits, .txt)
- .weight_mode            Weighting mode: uniform or idw
- .alpha                  Inverse distance weighting exponent alpha
- .k                      Neighbors to average (0 = all in stream)
- .max_frames             Upper limit on frames to reconstruct (0 = infinite)
- .out_file               Optional output reconstructed ASCII file (.txt)
- .shm_status_file        Bridge status SHM file path (blank = default)
- .status.frames_processed Total frames reconstructed
- .status.latency_us      Per-frame reconstruction latency in microseconds
- .status.fps             Instantaneous processing rate in frames per second
- .status.variance        Target reconstruction dispersion and variance
- .status.k_eff           Effective number of NN points averaged
- .status.stream_lag      Input stream lag (write - read)
- .status.memory_rss_mb   Process resident set memory size in megabytes

## SEE ALSO
- `milk`: Milk framework integration overview
- `milk_fpsexec`: Standalone FPS daemon manual
- `milk_knn`: Streaming k-NN search manual

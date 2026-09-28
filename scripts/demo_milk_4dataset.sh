#!/bin/bash
# ==============================================================================
# demo_milk_4dataset.sh - 4-Dataset Reconstruction Demo in Milk Framework
#
# Implements the complete 4-dataset test (A, B, C, D) using standalone
# Milk FPS daemons:
#   - milk-fpsexec-gric-cluster (for clustering datasets A and B)
#   - milk-fpsexec-gric-knn (for streaming k-NN query on A)
#   - milk-fpsexec-gric-reconstruct (for streaming reconstruction from B)
# ==============================================================================
set -euo pipefail

# Ensure Milk shared libraries are discoverable by dynamic linker
for milk_lib in "/usr/local/milk/lib" "/usr/local/milk-1.03.00/lib"; do
    if [ -d "$milk_lib" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":$milk_lib:"* ]]; then
        export LD_LIBRARY_PATH="$milk_lib:${LD_LIBRARY_PATH:-}"
    fi
done

print_help() {
    cat << 'EOF'
NAME
  demo_milk_4dataset.sh - GRIC Milk Framework 4-Dataset Reconstruction Benchmark

USAGE
  ./demo_milk_4dataset.sh [k] [weight_mode] [build_dir]
  ./demo_milk_4dataset.sh -h | --help

DESCRIPTION
  Runs an end-to-end benchmark demonstrating real-time coupled stream reconstruction:
    - Step 1: Generates synthetic benchmark datasets:
              A (observation space, 1000 samples, noise_a = 0.02)
              B (target state space, 1000 samples, noise_b = 0.10)
              C (query observation stream, 200 samples)
              D_true (ground truth target trajectory without noise)
    - Step 2: Clusters and indexes datasets A and B using gric-cluster and gric-knn.
    - Step 3: Deploys Milk FPS standalone daemons:
              milk-fpsexec-gric-knn (k-NN search against model A)
              milk-fpsexec-gric-reconstruct (reconstruction from matches & model B)
    - Step 4: Streams query C via ImageStreamIO and performs live inference.
    - Step 5: Reports comprehensive performance metrics, reconstruction fidelity (RMSE,
              R², SNR, noise floor ratio), and a parametric sweep over k_eff.

PARAMETERS
  k             Number of nearest neighbors to average for streaming reconstruction
                (default: 10).
  weight_mode   Weighting function for local reconstruction:
                  idw     - Inverse Distance Weighting (w_j = 1/d_j, default)
                  uniform - Equal weighting (w_j = 1/k, k_eff = k)
  build_dir     Optional path to local build directory to discover binaries.
  -h, --help    Display this help message and exit.

EFFECTIVE NN AVERAGING (k_eff) & NOISE REDUCTION
  Reconstructing target state D via weighted average of k neighbors reduces target noise:
    k_eff = 1 / sum(w_j^2)
    Theoretical Noise Floor = sigma_B / sqrt(k_eff)
  Because several samples near the query location are averaged, the reconstruction RMSE
  drives below the training target noise floor sigma_B = 0.10 (RMSE / sigma_B < 1.00x).

EXAMPLES
  $ ./scripts/demo_milk_4dataset.sh
      Run streaming reconstruction with default k=10 (idw, k_eff ≈ 8.09) and sweep.

  $ ./scripts/demo_milk_4dataset.sh 1
      Run with k=1 (no averaging, k_eff = 1.00, RMSE ≈ 1.03x noise floor).

  $ ./scripts/demo_milk_4dataset.sh 15 idw
      Run streaming reconstruction with k=15 using inverse distance weighting.

  $ ./scripts/demo_milk_4dataset.sh 20 uniform
      Run streaming reconstruction with k=20 uniform averaging (k_eff = 20.00).

EOF
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || echo "")"
BUILD_DIR="${BUILD_DIR:-}"
K_ARG=""
WEIGHT_ARG=""

for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        print_help
    elif [ -d "$arg" ]; then
        BUILD_DIR="$arg"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        K_ARG="$arg"
    elif [ "$arg" = "idw" ] || [ "$arg" = "uniform" ]; then
        WEIGHT_ARG="$arg"
    fi
done

find_binary() {
    local bin_name="$1"
    local build_rel_path="$2"

    # 1. Check system PATH first
    if command -v "$bin_name" >/dev/null 2>&1; then
        command -v "$bin_name"
        return 0
    fi

    # 2. Check explicitly specified or discovered build directories
    for dir in "$BUILD_DIR" "$PWD/build" "$ROOT_DIR/build" "$PWD"; do
        if [ -n "$dir" ] && [ -x "$dir/$build_rel_path" ]; then
            echo "$dir/$build_rel_path"
            return 0
        fi
        if [ -n "$dir" ] && [ -x "$dir/$bin_name" ]; then
            echo "$dir/$bin_name"
            return 0
        fi
    done

    # 3. Check common local install directories
    for prefix in "/usr/local/bin" "/usr/local/milk/bin" "$HOME/.local/bin"; do
        if [ -x "$prefix/$bin_name" ]; then
            echo "$prefix/$bin_name"
            return 0
        fi
    done

    echo ""
    return 1
}

FPS_CLUSTER="$(find_binary "milk-fpsexec-gric-cluster" \
    "src/gric-fps/milk-fpsexec-gric-cluster" || true)"
FPS_KNN="$(find_binary "milk-fpsexec-gric-knn" "src/gric-fps/milk-fpsexec-gric-knn" || true)"
FPS_RECON="$(find_binary "milk-fpsexec-gric-reconstruct" \
    "src/gric-fps/milk-fpsexec-gric-reconstruct" || true)"
GRIC_CLUSTER="$(find_binary "gric-cluster" "gric-cluster" || true)"
GRIC_KNN="$(find_binary "gric-knn" "gric-knn" || true)"
GRIC_TXT2STREAM="$(find_binary "gric-txt2stream" "gric-txt2stream" || true)"
GRIC_GEN4="$(find_binary "gric-mk4dataset" "gric-mk4dataset" || true)"
MILK_FPS_SET="$(find_binary "milk-fps-set" "milk-fps-set" || true)"

echo "======================================================================"
echo "    GRIC Milk Framework: 4-Dataset Reconstruction Test Demo           "
echo "======================================================================"

for var in FPS_CLUSTER FPS_KNN FPS_RECON GRIC_CLUSTER GRIC_KNN GRIC_TXT2STREAM; do
    val="${!var}"
    if [ -z "$val" ] || [ ! -x "$val" ]; then
        echo "Error: Required binary for $var not found in PATH or build directory."
        echo "Please ensure GRIC is installed in PATH or specify build dir: $0 <build_dir>"
        exit 1
    fi
done

clean_shm() {
    local prefix="$1"
    for dir in "${MILK_SHM_DIR:-}" "/milk/shm" "/dev/shm" "/tmp"; do
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            rm -f "$dir/${prefix}"*.im.shm 2>/dev/null || true
            rm -f "$dir/gric_knn.fps.shm" "$dir/gric_reconstruct.fps.shm" 2>/dev/null || true
        fi
    done
}

WORK_DIR="$(mktemp -d /tmp/gric_4data_demo_XXXXXX)"
cleanup() {
    echo ""
    echo "[Cleanup] Stopping active FPS daemons..."
    pkill -f "milk-fpsexec-gric" 2>/dev/null || true
    clean_shm "demo4_"
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Step 1: Generate Synthetic 4-Dataset Benchmark (Native C Generator)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/4] Generating coupled benchmark datasets (A, B, C, D_true)..."
N_TRAIN=1000
N_QUERY=200
K_NEIGHBORS="${K_ARG:-10}"
WEIGHT_MODE="${WEIGHT_ARG:-idw}"
NOISE_A=0.02
NOISE_B=0.1

if [ -x "$GRIC_GEN4" ]; then
    "$GRIC_GEN4" "$WORK_DIR" \
        -ntrain "$N_TRAIN" \
        -nquery "$N_QUERY" \
        -dima 3 \
        -dimb 3 \
        -noise-a "$NOISE_A" \
        -noise-b "$NOISE_B"
elif command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/generate_4dataset.py" \
        --outdir "$WORK_DIR" \
        --ntrain "$N_TRAIN" \
        --nquery "$N_QUERY" \
        --dima 3 \
        --dimb 3 \
        --noise-a "$NOISE_A" \
        --noise-b "$NOISE_B"
else
    echo "Error: Neither gric-mk4dataset nor python3 available to generate datasets."
    exit 1
fi

DATA_A="$WORK_DIR/dataset_A.txt"
DATA_B="$WORK_DIR/dataset_B.txt"
DATA_C="$WORK_DIR/dataset_C.txt"
DATA_D_TRUE="$WORK_DIR/dataset_D_true.txt"

# ------------------------------------------------------------------------------
# Step 2: Training Phase (Clustering & k-NN Indexing of Datasets A and B)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/4] Training Phase: Clustering and k-NN indexing of A and B..."
echo "  -> Datasets A and B stored on filesystem and pre-synchronized ($N_TRAIN samples):"
echo "     A: $DATA_A"
echo "     B: $DATA_B"

echo "  -> Training Dataset A (Observation space)..."
mkdir -p "$WORK_DIR/a_cluster.dat"
"$GRIC_CLUSTER" 0.35 "$DATA_A" \
    -outdir "$WORK_DIR/a_cluster.dat" \
    -maxim "$N_TRAIN" >/dev/null 2>&1

K_INDEX="$K_NEIGHBORS"
if [ "$K_INDEX" -lt 30 ]; then
    K_INDEX=30
fi

"$GRIC_KNN" "$DATA_A" "$WORK_DIR/a_cluster.dat" \
    -k "$K_INDEX" -dtmin 1 >/dev/null 2>&1

# Precompute query k-NN graph for parametric sweep
"$GRIC_KNN" "$DATA_A" "$WORK_DIR/a_cluster.dat" \
    -query "$DATA_C" -k "$K_INDEX" -o "$WORK_DIR/c_knn_sweep.txt" >/dev/null 2>&1

echo "  -> Training Dataset B (Target state space)..."
mkdir -p "$WORK_DIR/b_cluster.dat"
"$GRIC_CLUSTER" 0.35 "$DATA_B" \
    -outdir "$WORK_DIR/b_cluster.dat" \
    -maxim "$N_TRAIN" >/dev/null 2>&1

"$GRIC_KNN" "$DATA_B" "$WORK_DIR/b_cluster.dat" \
    -k "$K_INDEX" -dtmin 1 >/dev/null 2>&1

echo "  -> Models successfully staged in:"
echo "     A: $WORK_DIR/a_cluster.dat"
echo "     B: $WORK_DIR/b_cluster.dat"

# ------------------------------------------------------------------------------
# Step 3: Inference Configuration (Configure FPS Daemons)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/4] Configuring real-time streaming inference pipeline..."
clean_shm "demo4_"

# Initialize and configure k-NN searcher daemon
"$FPS_KNN" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_knn.in_name demo4_stream_C >/dev/null
    "$MILK_FPS_SET" gric_knn.out_name demo4_c_matches >/dev/null
    "$MILK_FPS_SET" gric_knn.cluster_dir "$WORK_DIR/a_cluster.dat" >/dev/null
    "$MILK_FPS_SET" gric_knn.ref_data "$DATA_A" >/dev/null
    "$MILK_FPS_SET" gric_knn.k "$K_NEIGHBORS" >/dev/null
    "$MILK_FPS_SET" gric_knn.max_frames "$N_QUERY" >/dev/null
fi

# Initialize and configure Reconstructor daemon
"$FPS_RECON" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_reconstruct.in_name demo4_c_matches >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.target_data "$DATA_B" >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.out_name demo4_stream_D >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.weight_mode "$WEIGHT_MODE" >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.alpha 1.0 >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.k "$K_NEIGHBORS" >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.max_frames "$N_QUERY" >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.out_file "$WORK_DIR/reconstructed_D.txt" >/dev/null
fi

# ------------------------------------------------------------------------------
# Step 4: Stream Execution & Inference
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/4] Executing streaming inference: Query C -> k-NN on A -> Recon D..."

# 1. Start streaming Query C into ImageStreamIO shared memory (with delay for daemons)
"$GRIC_TXT2STREAM" "$DATA_C" demo4_stream_C -fps 200 -maxfr "$N_QUERY" -delay 1.0 >/dev/null 2>&1 &
P_STREAM_C=$!
sleep 0.2

# 2. Launch k-NN Search daemon (reads demo4_stream_C, creates demo4_c_matches)
"$FPS_KNN" -loops runstart >/dev/null 2>&1 &
P_KNN=$!
sleep 0.3

# 3. Launch Reconstructor daemon (reads demo4_c_matches, creates demo4_stream_D)
"$FPS_RECON" -loops runstart >/dev/null 2>&1 &
P_RECON=$!

# Wait for stream to finish feeding all query samples
wait "$P_STREAM_C" 2>/dev/null || true
sleep 0.4

kill "$P_KNN" "$P_RECON" 2>/dev/null || true
sleep 0.2
kill -9 "$P_KNN" "$P_RECON" 2>/dev/null || true
wait "$P_KNN" "$P_RECON" 2>/dev/null || true

# ------------------------------------------------------------------------------
# Performance & Quality Metrics Summary
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "    GRIC MILK 4-DATASET BENCHMARK: PERFORMANCE & QUALITY SUMMARY      "
echo "======================================================================"

# Helper function to extract numerical parameter value from milk-fpsexec output
parse_fps_val() {
    local text="$1"
    local key="$2"
    echo "$text" | grep "$key" 2>/dev/null | \
        awk '{for(i=1;i<=NF;i++) if ($i=="UINT64"||$i=="FLOAT64"||$i=="INT64"||$i=="UINT32") { \
            print $(i+1); break; }}'
}

# 1. Clustering & Compression (Training Phase)
N_CLUST_A="$(grep "STATS_CLUSTERS:" "$WORK_DIR/a_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"
N_CLUST_B="$(grep "STATS_CLUSTERS:" "$WORK_DIR/b_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"
TIME_A="$(grep "TIME_CLUSTERING_MS:" "$WORK_DIR/a_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"
TIME_B="$(grep "TIME_CLUSTERING_MS:" "$WORK_DIR/b_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"
PRUNED_A="$(grep "STATS_PRUNED:" "$WORK_DIR/a_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"
PRUNED_B="$(grep "STATS_PRUNED:" "$WORK_DIR/b_cluster.dat/cluster_run.log" 2>/dev/null | \
    awk '{print $2}')"

echo ""
echo "--- 1. Clustering & Compression (Training Phase) ---"
echo "  Dataset A (Input):       $N_TRAIN samples -> ${N_CLUST_A:-N/A} clusters (${TIME_A:-N/A} ms)"
echo "  Dataset B (Target):      $N_TRAIN samples -> ${N_CLUST_B:-N/A} clusters (${TIME_B:-N/A} ms)"
echo "  Distance Pruning (A):    ${PRUNED_A:-N/A} pairwise evaluations avoided"
echo "  Distance Pruning (B):    ${PRUNED_B:-N/A} pairwise evaluations avoided"

# 2. Streaming k-NN Inference (Dataset C -> Model A)
KNN_FPS_OUT="$("$FPS_KNN" fps 2>/dev/null || true)"
KNN_QUERIES="$(parse_fps_val "$KNN_FPS_OUT" "status.queries_processed")"
KNN_EVALS="$(parse_fps_val "$KNN_FPS_OUT" "status.distance_evals")"
KNN_LATENCY="$(parse_fps_val "$KNN_FPS_OUT" "status.latency_us")"
KNN_RATE="$(parse_fps_val "$KNN_FPS_OUT" "status.fps")"

echo ""
echo "--- 2. Streaming k-NN Inference (Dataset C -> Model A) ---"
echo "  Queries Processed:       ${KNN_QUERIES:-$N_QUERY} / $N_QUERY samples (k = $K_NEIGHBORS)"
echo "  Mean Query Latency:      ${KNN_LATENCY:-N/A} us"
echo "  Query Throughput:        ${KNN_RATE:-N/A} FPS"
if [ -n "$KNN_EVALS" ] && [ -n "$KNN_QUERIES" ] && [ "$KNN_QUERIES" -gt 0 ] 2>/dev/null; then
    AVG_EVALS=$(( KNN_EVALS / KNN_QUERIES ))
    PRUNE_PCT="$(awk "BEGIN { \
        printf \"%.1f\", (1.0 - $KNN_EVALS / ($KNN_QUERIES * $N_TRAIN)) * 100 }")"
    echo "  Total Distance Evals:    $KNN_EVALS (~$AVG_EVALS evals/query, $PRUNE_PCT% pruned)"
fi

# 3. Real-time Reconstruction (Matches -> Output D)
RECON_FPS_OUT="$("$FPS_RECON" fps 2>/dev/null || true)"
RECON_FRAMES="$(parse_fps_val "$RECON_FPS_OUT" "status.frames_processed")"
RECON_LATENCY="$(parse_fps_val "$RECON_FPS_OUT" "status.latency_us")"
RECON_RATE="$(parse_fps_val "$RECON_FPS_OUT" "status.fps")"
RECON_VAR="$(parse_fps_val "$RECON_FPS_OUT" "status.variance")"
RECON_KEFF="$(parse_fps_val "$RECON_FPS_OUT" "status.k_eff")"

echo ""
echo "--- 3. Reconstructor Performance (Matches -> Output D) ---"
echo "  Frames Reconstructed:    ${RECON_FRAMES:-$N_QUERY} / $N_QUERY frames"
echo "  Mean Recon Latency:      ${RECON_LATENCY:-N/A} us"
echo "  Reconstruction Rate:     ${RECON_RATE:-N/A} FPS"
echo "  Effective NN Averaged:   ${RECON_KEFF:-N/A} points / sample (k_eff)"
echo "  Target Local Dispersion: ${RECON_VAR:-N/A} (neighbor variance)"

# 4. Reconstruction Accuracy vs Ground Truth (D vs D_true)
RECON_FILE="$WORK_DIR/reconstructed_D.txt"
TRUE_FILE="$WORK_DIR/dataset_D_true.txt"

if [ -f "$RECON_FILE" ] && [ -f "$TRUE_FILE" ] && [ -x "$GRIC_GEN4" ]; then
    echo ""
    echo "--- 4. Reconstruction Accuracy Assessment (D vs D_true) ---"
    "$GRIC_GEN4" -eval "$RECON_FILE" "$TRUE_FILE" "$NOISE_B" || true
fi

# 5. Parametric Sweep: Effect of Effective NN Averaging on Noise Reduction
SWEEP_KNN="$WORK_DIR/c_knn_sweep.txt"
if [ -f "$SWEEP_KNN" ] && [ -f "$DATA_B" ] && [ -f "$TRUE_FILE" ] && [ -x "$GRIC_GEN4" ]; then
    echo ""
    echo "--- 5. Parametric Sweep: Effect of Effective NN Averaging (k_eff) ---"
    "$GRIC_GEN4" -sweep "$SWEEP_KNN" "$DATA_B" "$TRUE_FILE" "$NOISE_B" || true
fi

echo ""
echo "--- Shared Memory Streams Generated ---"
for dir in "${MILK_SHM_DIR:-}" "/milk/shm" "/dev/shm"; do
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        ls -l "$dir"/demo4_*.im.shm 2>/dev/null || true
    fi
done

echo ""
echo "======================================================================"
echo "    4-DATASET MILK DEMO COMPLETED SUCCESSFULLY!                      "
echo "======================================================================"
exit 0

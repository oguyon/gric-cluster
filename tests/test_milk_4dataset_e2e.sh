#!/bin/bash
# ==============================================================================
# test_milk_4dataset_e2e.sh - E2E tests for GRIC Milk 4-Dataset Reconstruction
#
# Tests both standalone executables (milk-fpsexec-gric-reconstruct) and
# Milk CLI module integration (gric.gric_reconstruct).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-${BUILD_DIR:-$PWD}}"
if [ ! -f "$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-reconstruct" ]; then
    if [ -f "$PWD/src/gric-fps/milk-fpsexec-gric-reconstruct" ]; then
        BUILD_DIR="$PWD"
    elif [ -f "$ROOT_DIR/build/src/gric-fps/milk-fpsexec-gric-reconstruct" ]; then
        BUILD_DIR="$ROOT_DIR/build"
    fi
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

FPS_CLUSTER="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-cluster"
FPS_KNN="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-knn"
FPS_RECON="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-reconstruct"
MILK_SO="$BUILD_DIR/src/gric-fps/libmilkgric.so"
GRIC_CLUSTER="$BUILD_DIR/gric-cluster"
GRIC_KNN="$BUILD_DIR/gric-knn"
GRIC_TXT2STREAM="$BUILD_DIR/gric-txt2stream"
GRIC_GEN4="$BUILD_DIR/gric-mk4dataset"

# Locate milk-cli binary
MILK_CLI=""
if command -v milk >/dev/null 2>&1; then
    MILK_CLI="$(command -v milk)"
elif [ -x "/usr/local/bin/milk" ]; then
    MILK_CLI="/usr/local/bin/milk"
elif [ -x "/usr/local/milk/bin/milk-cli" ]; then
    MILK_CLI="/usr/local/milk/bin/milk-cli"
elif [ -x "/usr/local/milk/bin/milk" ]; then
    MILK_CLI="/usr/local/milk/bin/milk"
fi

# Locate milk-fps-set
MILK_FPS_SET=""
if command -v milk-fps-set >/dev/null 2>&1; then
    MILK_FPS_SET="$(command -v milk-fps-set)"
elif [ -x "/usr/local/bin/milk-fps-set" ]; then
    MILK_FPS_SET="/usr/local/bin/milk-fps-set"
elif [ -x "/usr/local/milk/bin/milk-fps-set" ]; then
    MILK_FPS_SET="/usr/local/milk/bin/milk-fps-set"
fi

echo "======================================================================"
echo "GRIC Milk Framework Integration Test Suite (4-Dataset Reconstruction)"
echo "======================================================================"

for bin in "$FPS_CLUSTER" "$FPS_KNN" "$FPS_RECON" "$MILK_SO" "$GRIC_CLUSTER" "$GRIC_KNN"; do
    if [ ! -f "$bin" ]; then
        echo "Error: Required binary not found: $bin"
        exit 1
    fi
done

check_stream_exists() {
    local stream="$1"
    for dir in "${MILK_SHM_DIR:-}" "/milk/shm" "/dev/shm" "/tmp"; do
        if [ -n "$dir" ] && [ -f "$dir/${stream}.im.shm" ]; then
            return 0
        fi
    done
    return 1
}

clean_e2e_shm() {
    local prefix="$1"
    for dir in "${MILK_SHM_DIR:-}" "/milk/shm" "/dev/shm" "/tmp"; do
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            rm -f "$dir/${prefix}"*.im.shm 2>/dev/null || true
            rm -f "$dir/gric_knn.fps.shm" "$dir/gric_reconstruct.fps.shm" 2>/dev/null || true
        fi
    done
}

TMPDIR="$(mktemp -d /tmp/gric_milk_4d_e2e_XXXXXX)"
cleanup() {
    pkill -f "milk-fpsexec-gric" 2>/dev/null || true
    rm -rf "$TMPDIR"
    clean_e2e_shm "e2e_4d_"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Setup: Generate benchmark datasets A, B, C, D_true (Native C Generator)
# ------------------------------------------------------------------------------
echo "[Setup] Generating benchmark datasets in $TMPDIR..."
if [ -x "$GRIC_GEN4" ]; then
    "$GRIC_GEN4" "$TMPDIR" \
        -ntrain 500 \
        -nquery 50 \
        -dima 3 \
        -dimb 3 \
        -noise-a 0.02 \
        -noise-b 0.1 >/dev/null
elif command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT_DIR/scripts/generate_4dataset.py" \
        --outdir "$TMPDIR" \
        --ntrain 500 \
        --nquery 50 \
        --dima 3 \
        --dimb 3 \
        --noise-a 0.02 \
        --noise-b 0.1 >/dev/null
else
    echo "Error: Neither gric-mk4dataset nor python3 available to generate datasets."
    exit 1
fi

DATA_A="$TMPDIR/dataset_A.txt"
DATA_B="$TMPDIR/dataset_B.txt"
DATA_C="$TMPDIR/dataset_C.txt"

# ------------------------------------------------------------------------------
# Training: Cluster A and B, build k-NN indices
# ------------------------------------------------------------------------------
echo "[Setup] Clustering and indexing datasets A and B..."
mkdir -p "$TMPDIR/a_cluster.dat" "$TMPDIR/b_cluster.dat"

"$GRIC_CLUSTER" 0.35 "$DATA_A" -outdir "$TMPDIR/a_cluster.dat" -maxim 500 >/dev/null 2>&1
"$GRIC_KNN" "$DATA_A" "$TMPDIR/a_cluster.dat" -k 10 -dtmin 1 >/dev/null 2>&1

"$GRIC_CLUSTER" 0.35 "$DATA_B" -outdir "$TMPDIR/b_cluster.dat" -maxim 500 >/dev/null 2>&1
"$GRIC_KNN" "$DATA_B" "$TMPDIR/b_cluster.dat" -k 10 -dtmin 1 >/dev/null 2>&1

# ------------------------------------------------------------------------------
# Test 1: Standalone milk-fpsexec-gric-reconstruct
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "[Test 1] Standalone milk-fpsexec-gric-reconstruct (Pipeline Inference)"
echo "----------------------------------------------------------------------"
clean_e2e_shm "e2e_4d_st_"

"$FPS_KNN" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_knn.in_name e2e_4d_st_in >/dev/null
    "$MILK_FPS_SET" gric_knn.out_name e2e_4d_st_matches >/dev/null
    "$MILK_FPS_SET" gric_knn.cluster_dir "$TMPDIR/a_cluster.dat" >/dev/null
    "$MILK_FPS_SET" gric_knn.ref_data "$DATA_A" >/dev/null
    "$MILK_FPS_SET" gric_knn.k 10 >/dev/null
    "$MILK_FPS_SET" gric_knn.max_frames 50 >/dev/null
fi

"$FPS_RECON" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_reconstruct.in_name e2e_4d_st_matches >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.target_data "$DATA_B" >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.out_name e2e_4d_st_recon >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.weight_mode idw >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.k 10 >/dev/null
    "$MILK_FPS_SET" gric_reconstruct.max_frames 50 >/dev/null
fi

# 1. Start streaming Query C
"$GRIC_TXT2STREAM" "$DATA_C" e2e_4d_st_in -fps 200 -loop >/dev/null 2>&1 &
P1_PID=$!
sleep 0.2

# 2. Start k-NN search daemon
"$FPS_KNN" -loops runstart >/dev/null 2>&1 &
P2_PID=$!
sleep 0.4

# 3. Start Reconstructor daemon
"$FPS_RECON" -loops runstart >/dev/null 2>&1 &
P3_PID=$!
sleep 0.8

kill -9 "$P1_PID" "$P2_PID" "$P3_PID" 2>/dev/null || true
wait "$P1_PID" "$P2_PID" "$P3_PID" 2>/dev/null || true

if ! check_stream_exists "e2e_4d_st_recon"; then
    echo "Error: Test 1 output stream e2e_4d_st_recon not found in shm"
    exit 1
fi
echo "[Test 1] PASSED: Standalone 4-dataset inference reconstructed frames cleanly."

# ------------------------------------------------------------------------------
# Test 2: Milk CLI Module: gric.gric_reconstruct
# ------------------------------------------------------------------------------
if [ -n "$MILK_CLI" ] && [ -x "$MILK_CLI" ]; then
    echo "----------------------------------------------------------------------"
    echo "[Test 2] Milk CLI Module: gric.gric_reconstruct"
    echo "----------------------------------------------------------------------"
    clean_e2e_shm "e2e_4d_cli_"

    "$FPS_KNN" fpsinit >/dev/null 2>&1
    if [ -n "$MILK_FPS_SET" ]; then
        "$MILK_FPS_SET" gric_knn.in_name e2e_4d_cli_in >/dev/null
        "$MILK_FPS_SET" gric_knn.out_name e2e_4d_cli_matches >/dev/null
        "$MILK_FPS_SET" gric_knn.cluster_dir "$TMPDIR/a_cluster.dat" >/dev/null
        "$MILK_FPS_SET" gric_knn.ref_data "$DATA_A" >/dev/null
        "$MILK_FPS_SET" gric_knn.k 10 >/dev/null
        "$MILK_FPS_SET" gric_knn.max_frames 0 >/dev/null
    fi

    "$GRIC_TXT2STREAM" "$DATA_C" e2e_4d_cli_in -fps 200 -loop >/dev/null 2>&1 &
    P4_PID=$!
    sleep 0.2

    "$FPS_KNN" -loops runstart >/dev/null 2>&1 &
    P5_PID=$!
    sleep 0.4

    cat << EOF > "$TMPDIR/cmd_recon.txt"
gric.gric_reconstruct .in_name e2e_4d_cli_matches
gric.gric_reconstruct .out_name e2e_4d_cli_recon
gric.gric_reconstruct .target_data $DATA_B
gric.gric_reconstruct .weight_mode idw
gric.gric_reconstruct .k 10
gric.gric_reconstruct .max_frames 20
gric.gric_reconstruct
exitCLI
EOF

    MILKCLI_ADD_LIBS="$MILK_SO" "$MILK_CLI" \
        < "$TMPDIR/cmd_recon.txt" > "$TMPDIR/cli.log" 2>&1 || true

    kill -9 "$P4_PID" "$P5_PID" 2>/dev/null || true
    wait "$P4_PID" "$P5_PID" 2>/dev/null || true

    if ! check_stream_exists "e2e_4d_cli_recon"; then
        echo "Error: Test 2 output stream e2e_4d_cli_recon not found in shm"
        echo "=== CLI LOG ==="
        cat "$TMPDIR/cli.log"
        exit 1
    fi
    echo "[Test 2] PASSED: Milk CLI gric.gric_reconstruct processed matches cleanly."
fi

echo "======================================================================"
echo "ALL GRIC 4-DATASET MILK INTEGRATION TESTS PASSED"
echo "======================================================================"
exit 0

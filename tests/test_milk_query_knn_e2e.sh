#!/bin/bash
# ==============================================================================
# test_milk_query_knn_e2e.sh - E2E tests for GRIC Milk Query Mode and k-NN
#
# Tests both standalone executables (milk-fpsexec-gric-cluster,
# milk-fpsexec-gric-knn) and Milk CLI integration (gric.gric_cluster,
# gric.gric_knn).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-${BUILD_DIR:-$PWD}}"
if [ ! -f "$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-cluster" ]; then
    if [ -f "$PWD/src/gric-fps/milk-fpsexec-gric-cluster" ]; then
        BUILD_DIR="$PWD"
    elif [ -f "$ROOT_DIR/build/src/gric-fps/milk-fpsexec-gric-cluster" ]; then
        BUILD_DIR="$ROOT_DIR/build"
    elif [ -f "$ROOT_DIR/build-milk/src/gric-fps/milk-fpsexec-gric-cluster" ]; then
        BUILD_DIR="$ROOT_DIR/build-milk"
    fi
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

FPS_CLUSTER="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-cluster"
FPS_KNN="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-knn"
MILK_SO="$BUILD_DIR/src/gric-fps/libmilkgric.so"
GRIC_CLUSTER="$BUILD_DIR/gric-cluster"
GRIC_KNN="$BUILD_DIR/gric-knn"
GRIC_TXT2STREAM="$BUILD_DIR/gric-txt2stream"

# Locate milk-cli binary
MILK_CLI=""
if command -v milk-cli >/dev/null 2>&1; then
    MILK_CLI="$(command -v milk-cli)"
elif [ -n "${MILK_ROOT:-}" ] && [ -x "${MILK_ROOT}/bin/milk-cli" ]; then
    MILK_CLI="${MILK_ROOT}/bin/milk-cli"
elif [ -x "/usr/local/milk/bin/milk-cli" ]; then
    MILK_CLI="/usr/local/milk/bin/milk-cli"
elif [ -x "/usr/local/bin/milk-cli" ]; then
    MILK_CLI="/usr/local/bin/milk-cli"
elif command -v milk >/dev/null 2>&1; then
    MILK_CLI="$(command -v milk)"
fi
if [ -z "$MILK_CLI" ]; then
    MILK_CLI="$(find /usr/local /usr -name "milk-cli" -type f -perm -111 \
        2>/dev/null | grep -E '/bin/milk-cli$' | head -n 1 || true)"
fi

# Locate milk-fps-set
MILK_FPS_SET=""
if command -v milk-fps-set >/dev/null 2>&1; then
    MILK_FPS_SET="$(command -v milk-fps-set)"
elif [ -n "${MILK_ROOT:-}" ] && [ -x "${MILK_ROOT}/bin/milk-fps-set" ]; then
    MILK_FPS_SET="${MILK_ROOT}/bin/milk-fps-set"
elif [ -x "/usr/local/milk/bin/milk-fps-set" ]; then
    MILK_FPS_SET="/usr/local/milk/bin/milk-fps-set"
elif [ -x "/usr/local/bin/milk-fps-set" ]; then
    MILK_FPS_SET="/usr/local/bin/milk-fps-set"
fi
if [ -z "$MILK_FPS_SET" ]; then
    MILK_FPS_SET="$(find /usr/local /usr -name "milk-fps-set" -type f -perm -111 \
        2>/dev/null | head -n 1 || true)"
fi

echo "======================================================================"
echo "GRIC Milk Framework Integration Test Suite (Query Mode & k-NN)"
echo "======================================================================"

# Check binaries and verify zero libCLIcore linkage
for bin in "$FPS_CLUSTER" "$FPS_KNN" "$MILK_SO" "$GRIC_CLUSTER" \
           "$GRIC_KNN" "$GRIC_TXT2STREAM"; do
    if [ ! -f "$bin" ]; then
        echo "Error: Required binary not found: $bin"
        exit 1
    fi
done
for mod in "$FPS_CLUSTER" "$FPS_KNN" "$MILK_SO"; do
    if ldd "$mod" | grep -q "libCLIcore"; then
        echo "Error: $mod must not link libCLIcore!"
        exit 1
    fi
done

TEST_SHM="$(mktemp -d /dev/shm/gric_milk_e2e_shm_XXXXXX)"
export MILK_SHM_DIR="$TEST_SHM"
TMPDIR="$(mktemp -d /tmp/gric_milk_e2e_XXXXXX)"

check_stream_exists() {
    local stream="$1"
    [ -f "$TEST_SHM/${stream}.im.shm" ]
}

wait_for_stream() {
    local stream="$1"
    for _ in $(seq 1 50); do
        if check_stream_exists "$stream"; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

cleanup() {
    "$FPS_CLUSTER" runstop >/dev/null 2>&1 || true
    "$FPS_KNN" runstop >/dev/null 2>&1 || true
    pkill -f "milk-fpsexec-gric" 2>/dev/null || true
    rm -rf "$TMPDIR" "$TEST_SHM"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Step 0: Generate baseline clustering model for tests
# ------------------------------------------------------------------------------
echo "[Setup] Generating baseline cluster model in $TMPDIR/clusterdat..."
mkdir -p "$TMPDIR/clusterdat"
TEST_DATA="$TMPDIR/ref.txt"
cp "$ROOT_DIR/tests/test_strat.txt" "$TEST_DATA"

"$GRIC_CLUSTER" 0.5 "$TEST_DATA" \
    -outdir "$TMPDIR/clusterdat" \
    -maxim 1000 >/dev/null 2>&1

"$GRIC_KNN" "$TEST_DATA" "$TMPDIR/clusterdat" \
    -k 5 -dtmin 1 >/dev/null 2>&1

if [ ! -f "$TMPDIR/clusterdat/anchors.bin" ]; then
    echo "Error: Failed to generate anchors.bin"
    exit 1
fi
if [ ! -f "$TMPDIR/clusterdat/knn_mutual_dists.bin" ]; then
    echo "Error: Failed to generate k-NN graph in clusterdat"
    exit 1
fi
echo "[Setup] Baseline model generated successfully."

# ------------------------------------------------------------------------------
# Test 1: Standalone milk-fpsexec-gric-cluster in Query Mode
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "[Test 1] Standalone milk-fpsexec-gric-cluster (Query Mode)"
echo "----------------------------------------------------------------------"

"$FPS_CLUSTER" fpsinit
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_cluster.in_name e2e_st_q_in
    "$MILK_FPS_SET" gric_cluster.out_name e2e_st_q_out
    "$MILK_FPS_SET" gric_cluster.out_anchors e2e_st_q_anc
    "$MILK_FPS_SET" gric_cluster.out_counts e2e_st_q_cnt
    "$MILK_FPS_SET" gric_cluster.query_mode 1
    "$MILK_FPS_SET" gric_cluster.load_anchors "$TMPDIR/clusterdat/anchors.bin"
    "$MILK_FPS_SET" gric_cluster.max_frames 40
fi

"$GRIC_TXT2STREAM" "$TEST_DATA" e2e_st_q_in -fps 500 -loop >/dev/null 2>&1 &
P1_PID=$!
wait_for_stream "e2e_st_q_in"

"$FPS_CLUSTER" -loops runstart
wait_for_stream "e2e_st_q_out" || true
sleep 0.5
"$FPS_CLUSTER" runstop >/dev/null 2>&1 || true

kill -9 "$P1_PID" 2>/dev/null || true
wait "$P1_PID" 2>/dev/null || true

if ! check_stream_exists "e2e_st_q_out"; then
    echo "Error: Test 1 output stream e2e_st_q_out not found in $TEST_SHM"
    exit 1
fi
echo "[Test 1] PASSED: Standalone query-mode clustering processed frames cleanly."

# ------------------------------------------------------------------------------
# Test 2: Standalone milk-fpsexec-gric-knn Streaming Search
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "[Test 2] Standalone milk-fpsexec-gric-knn (Streaming k-NN)"
echo "----------------------------------------------------------------------"

"$FPS_KNN" fpsinit
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_knn.in_name e2e_st_knn_in
    "$MILK_FPS_SET" gric_knn.out_name e2e_st_knn_out
    "$MILK_FPS_SET" gric_knn.cluster_dir "$TMPDIR/clusterdat"
    "$MILK_FPS_SET" gric_knn.ref_data "$TEST_DATA"
    "$MILK_FPS_SET" gric_knn.k 5
    "$MILK_FPS_SET" gric_knn.max_frames 40
fi

"$GRIC_TXT2STREAM" "$TEST_DATA" e2e_st_knn_in \
    -fps 500 -loop >/dev/null 2>&1 &
P2_PID=$!
wait_for_stream "e2e_st_knn_in"

"$FPS_KNN" -loops runstart
wait_for_stream "e2e_st_knn_out" || true
sleep 0.5
"$FPS_KNN" runstop >/dev/null 2>&1 || true

kill -9 "$P2_PID" 2>/dev/null || true
wait "$P2_PID" 2>/dev/null || true

if ! check_stream_exists "e2e_st_knn_out"; then
    echo "Error: Test 2 output stream e2e_st_knn_out not found in $TEST_SHM"
    exit 1
fi
echo "[Test 2] PASSED: Standalone streaming k-NN search processed queries."

# ------------------------------------------------------------------------------
# Test 3 & 4: Milk CLI Integration Tests (if milk-cli is present)
# ------------------------------------------------------------------------------
if [ -n "$MILK_CLI" ] && [ -x "$MILK_CLI" ]; then
    echo "----------------------------------------------------------------------"
    echo "[Test 3] Milk CLI Module: gric.gric_cluster (Query Mode)"
    echo "----------------------------------------------------------------------"

    "$GRIC_TXT2STREAM" "$TEST_DATA" e2e_cli_q_in \
        -fps 500 -loop >/dev/null 2>&1 &
    P3_PID=$!
    wait_for_stream "e2e_cli_q_in"

    CMD3="@module load $MILK_SO;"
    CMD3="$CMD3 gric.gric_cluster .in_name=e2e_cli_q_in .out_name=e2e_cli_q_out"
    CMD3="$CMD3 .out_anchors=e2e_cli_q_anc .out_counts=e2e_cli_q_cnt"
    CMD3="$CMD3 .query_mode=ON .load_anchors=$TMPDIR/clusterdat/anchors.bin"
    CMD3="$CMD3 .max_frames=30"
    "$MILK_CLI" -q -c "$CMD3" > "$TMPDIR/cli_cluster.log" 2>&1 || true

    kill -9 "$P3_PID" 2>/dev/null || true
    wait "$P3_PID" 2>/dev/null || true

    if ! check_stream_exists "e2e_cli_q_out"; then
        echo "Error: Test 3 output stream e2e_cli_q_out not found in $TEST_SHM"
        cat "$TMPDIR/cli_cluster.log" || true
        exit 1
    fi
    echo "[Test 3] PASSED: Milk CLI gric.gric_cluster in query mode completed."

    echo "----------------------------------------------------------------------"
    echo "[Test 4] Milk CLI Module: gric.gric_knn (Streaming k-NN)"
    echo "----------------------------------------------------------------------"

    "$GRIC_TXT2STREAM" "$TEST_DATA" e2e_cli_knn_in \
        -fps 500 -loop >/dev/null 2>&1 &
    P4_PID=$!
    wait_for_stream "e2e_cli_knn_in"

    CMD4="@module load $MILK_SO;"
    CMD4="$CMD4 gric.gric_knn .in_name=e2e_cli_knn_in .out_name=e2e_cli_knn_out"
    CMD4="$CMD4 .cluster_dir=$TMPDIR/clusterdat .ref_data=$TEST_DATA"
    CMD4="$CMD4 .k=5 .max_frames=30"
    "$MILK_CLI" -q -c "$CMD4" > "$TMPDIR/cli_knn.log" 2>&1 || true

    kill -9 "$P4_PID" 2>/dev/null || true
    wait "$P4_PID" 2>/dev/null || true

    if ! check_stream_exists "e2e_cli_knn_out"; then
        echo "Error: Test 4 output stream e2e_cli_knn_out not found in $TEST_SHM"
        cat "$TMPDIR/cli_knn.log" || true
        exit 1
    fi
    echo "[Test 4] PASSED: Milk CLI gric.gric_knn streaming search completed."
    echo "======================================================================"
    echo "ALL GRIC MILK INTEGRATION TESTS PASSED (4/4)"
    echo "======================================================================"
else
    if [ "${GRIC_REQUIRE_MILK_CLI:-0}" = "1" ]; then
        echo "Error: milk-cli binary required (GRIC_REQUIRE_MILK_CLI=1)!"
        exit 1
    fi
    echo "----------------------------------------------------------------------"
    echo "[Info] milk-cli not found; skipping CLI interactive tests 3 and 4."
    echo "======================================================================"
    echo "GRIC MILK STANDALONE TESTS PASSED (2/2 standalone, 2 skipped)"
    echo "======================================================================"
fi
exit 0

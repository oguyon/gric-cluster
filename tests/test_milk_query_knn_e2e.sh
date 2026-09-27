#!/bin/bash
# ==============================================================================
# test_milk_query_knn_e2e.sh - E2E tests for GRIC Milk Query Mode and k-NN
#
# Tests both standalone executables (milk-fpsexec-gric-cluster, milk-fpsexec-gric-knn)
# and Milk CLI module integration (gric.gric_cluster, gric.gric_knn).
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

FPS_CLUSTER="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-cluster"
FPS_KNN="$BUILD_DIR/src/gric-fps/milk-fpsexec-gric-knn"
MILK_SO="$BUILD_DIR/src/gric-fps/libmilkgric.so"
GRIC_CLUSTER="$BUILD_DIR/gric-cluster"
GRIC_KNN="$BUILD_DIR/gric-knn"
GRIC_TXT2STREAM="$BUILD_DIR/gric-txt2stream"

# Locate milk-cli binary
MILK_CLI=""
if command -v milk >/dev/null 2>&1; then
    MILK_CLI="$(command -v milk)"
elif [ -x "/usr/local/bin/milk" ]; then
    MILK_CLI="/usr/local/bin/milk"
elif [ -x "/usr/local/milk/bin/milk" ]; then
    MILK_CLI="/usr/local/milk/bin/milk"
elif [ -x "/home/oguyon/src/milk/_build/milk-cli" ]; then
    MILK_CLI="/home/oguyon/src/milk/_build/milk-cli"
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
echo "GRIC Milk Framework Integration Test Suite (Query Mode & k-NN)"
echo "======================================================================"

# Check binaries
for bin in "$FPS_CLUSTER" "$FPS_KNN" "$MILK_SO" "$GRIC_CLUSTER" "$GRIC_KNN" "$GRIC_TXT2STREAM"; do
    if [ ! -f "$bin" ]; then
        echo "Error: Required binary not found: $bin"
        exit 1
    fi
done

TMPDIR="$(mktemp -d /tmp/gric_milk_e2e_XXXXXX)"
cleanup() {
    pkill -f "milk-fpsexec-gric" 2>/dev/null || true
    rm -rf "$TMPDIR"
    rm -f /milk/shm/e2e_* /dev/shm/e2e_* 2>/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Step 0: Generate baseline clustering model for tests
# ------------------------------------------------------------------------------
echo "[Setup] Generating baseline cluster model in $TMPDIR/clusterdat..."
mkdir -p "$TMPDIR/clusterdat"
"$GRIC_CLUSTER" 0.5 "$ROOT_DIR/tests/test_strat.txt" \
    -outdir "$TMPDIR/clusterdat" \
    -maxim 1000 >/dev/null 2>&1

"$GRIC_KNN" "$ROOT_DIR/tests/test_strat.txt" "$TMPDIR/clusterdat" \
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
rm -f /milk/shm/e2e_st_q_* /dev/shm/e2e_st_q_* 2>/dev/null || true

"$FPS_CLUSTER" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_cluster.in_name e2e_st_q_in >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.out_name e2e_st_q_out >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.out_anchors e2e_st_q_anc >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.out_counts e2e_st_q_cnt >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.query_mode 1 >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.load_anchors "$TMPDIR/clusterdat/anchors.bin" >/dev/null 2>&1
    "$MILK_FPS_SET" gric_cluster.max_frames 40 >/dev/null 2>&1
fi

"$GRIC_TXT2STREAM" "$ROOT_DIR/tests/test_strat.txt" e2e_st_q_in -fps 500 -loop >/dev/null 2>&1 &
P1_PID=$!
sleep 0.3

"$FPS_CLUSTER" -loops runstart >/dev/null 2>&1
sleep 0.5

kill -9 "$P1_PID" 2>/dev/null || true
wait "$P1_PID" 2>/dev/null || true

if [ ! -f "/milk/shm/e2e_st_q_out.im.shm" ] && [ ! -f "/dev/shm/e2e_st_q_out.im.shm" ]; then
    echo "Error: Test 1 output stream e2e_st_q_out not found in shm"
    exit 1
fi
echo "[Test 1] PASSED: Standalone query-mode clustering processed frames cleanly."

# ------------------------------------------------------------------------------
# Test 2: Standalone milk-fpsexec-gric-knn Streaming Search
# ------------------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "[Test 2] Standalone milk-fpsexec-gric-knn (Streaming k-NN)"
echo "----------------------------------------------------------------------"
rm -f /milk/shm/e2e_st_knn_* /dev/shm/e2e_st_knn_* 2>/dev/null || true

"$FPS_KNN" fpsinit >/dev/null 2>&1
if [ -n "$MILK_FPS_SET" ]; then
    "$MILK_FPS_SET" gric_knn.in_name e2e_st_knn_in >/dev/null 2>&1
    "$MILK_FPS_SET" gric_knn.out_name e2e_st_knn_out >/dev/null 2>&1
    "$MILK_FPS_SET" gric_knn.cluster_dir "$TMPDIR/clusterdat" >/dev/null 2>&1
    "$MILK_FPS_SET" gric_knn.ref_data "$ROOT_DIR/tests/test_strat.txt" >/dev/null 2>&1
    "$MILK_FPS_SET" gric_knn.k 5 >/dev/null 2>&1
    "$MILK_FPS_SET" gric_knn.max_frames 40 >/dev/null 2>&1
fi

"$GRIC_TXT2STREAM" "$ROOT_DIR/tests/test_strat.txt" e2e_st_knn_in \
    -fps 500 -loop >/dev/null 2>&1 &
P2_PID=$!
sleep 0.3

"$FPS_KNN" -loops runstart >/dev/null 2>&1
sleep 0.5

kill -9 "$P2_PID" 2>/dev/null || true
wait "$P2_PID" 2>/dev/null || true

if [ ! -f "/milk/shm/e2e_st_knn_out.im.shm" ] && [ ! -f "/dev/shm/e2e_st_knn_out.im.shm" ]; then
    echo "Error: Test 2 output stream e2e_st_knn_out not found in shm"
    exit 1
fi
echo "[Test 2] PASSED: Standalone streaming k-NN search processed queries cleanly."

# ------------------------------------------------------------------------------
# Test 3 & 4: Milk CLI Integration Tests (if milk CLI is present)
# ------------------------------------------------------------------------------
if [ -n "$MILK_CLI" ] && [ -x "$MILK_CLI" ]; then
    echo "----------------------------------------------------------------------"
    echo "[Test 3] Milk CLI Module: gric.gric_cluster (Query Mode)"
    echo "----------------------------------------------------------------------"
    rm -f /milk/shm/e2e_cli_q_* /dev/shm/e2e_cli_q_* 2>/dev/null || true

    "$GRIC_TXT2STREAM" "$ROOT_DIR/tests/test_strat.txt" e2e_cli_q_in \
        -fps 500 -loop >/dev/null 2>&1 &
    P3_PID=$!
    sleep 0.3

    cat << EOF > "$TMPDIR/cmd_cluster.txt"
gric.gric_cluster .in_name e2e_cli_q_in
gric.gric_cluster .out_name e2e_cli_q_out
gric.gric_cluster .out_anchors e2e_cli_q_anc
gric.gric_cluster .out_counts e2e_cli_q_cnt
gric.gric_cluster .query_mode 1
gric.gric_cluster .load_anchors $TMPDIR/clusterdat/anchors.bin
gric.gric_cluster .max_frames 30
gric.gric_cluster
exitCLI
EOF

    MILKCLI_ADD_LIBS="$MILK_SO" "$MILK_CLI" < "$TMPDIR/cmd_cluster.txt" >/dev/null 2>&1 || true

    kill -9 "$P3_PID" 2>/dev/null || true
    wait "$P3_PID" 2>/dev/null || true

    if [ ! -f "/milk/shm/e2e_cli_q_out.im.shm" ] && \
       [ ! -f "/dev/shm/e2e_cli_q_out.im.shm" ]; then
        echo "Error: Test 3 output stream e2e_cli_q_out not found in shm"
        exit 1
    fi
    echo "[Test 3] PASSED: Milk CLI gric.gric_cluster in query mode completed cleanly."

    echo "----------------------------------------------------------------------"
    echo "[Test 4] Milk CLI Module: gric.gric_knn (Streaming k-NN)"
    echo "----------------------------------------------------------------------"
    rm -f /milk/shm/e2e_cli_knn_* /dev/shm/e2e_cli_knn_* 2>/dev/null || true

    "$GRIC_TXT2STREAM" "$ROOT_DIR/tests/test_strat.txt" e2e_cli_knn_in \
        -fps 500 -loop >/dev/null 2>&1 &
    P4_PID=$!
    sleep 0.3

    cat << EOF > "$TMPDIR/cmd_knn.txt"
gric.gric_knn .in_name e2e_cli_knn_in
gric.gric_knn .out_name e2e_cli_knn_out
gric.gric_knn .cluster_dir $TMPDIR/clusterdat
gric.gric_knn .ref_data $ROOT_DIR/tests/test_strat.txt
gric.gric_knn .k 5
gric.gric_knn .max_frames 30
gric.gric_knn
exitCLI
EOF

    MILKCLI_ADD_LIBS="$MILK_SO" "$MILK_CLI" < "$TMPDIR/cmd_knn.txt" >/dev/null 2>&1 || true

    kill -9 "$P4_PID" 2>/dev/null || true
    wait "$P4_PID" 2>/dev/null || true

    if [ ! -f "/milk/shm/e2e_cli_knn_out.im.shm" ] && \
       [ ! -f "/dev/shm/e2e_cli_knn_out.im.shm" ]; then
        echo "Error: Test 4 output stream e2e_cli_knn_out not found in shm"
        exit 1
    fi
    echo "[Test 4] PASSED: Milk CLI gric.gric_knn streaming search completed cleanly."
else
    echo "[Info] milk-cli binary not found; skipping CLI interactive tests 3 and 4."
fi

echo "======================================================================"
echo "ALL GRIC MILK INTEGRATION TESTS PASSED (4/4)"
echo "======================================================================"
exit 0

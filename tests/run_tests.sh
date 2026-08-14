#!/usr/bin/env bash
set -e

# Determine directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "=================================================="
echo " Running GRIC Test Suite"
echo "=================================================="

# Check binaries
if [ ! -f "${BUILD_DIR}/gric-cluster" ]; then
    echo "Building project in ${BUILD_DIR}..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)"
fi

cd "${BUILD_DIR}"

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    shift
    echo -n "Running ${test_name}... "
    if "$@" > /dev/null 2>&1; then
        echo -e "\033[32mPASSED\033[0m"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "\033[31mFAILED\033[0m"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# 1. Test coordinate dataset clustering
run_test "Coordinate Clustering (test_strat.txt)" \
    ./gric-cluster 0.5 "${SCRIPT_DIR}/test_strat.txt" -maxim 1000 -outdir /tmp/test_strat_out

# 2. Test synthetic sequence generator
run_test "Sequence Generator (gric-mktxtseq)" \
    ./gric-mktxtseq 1000 /tmp/test_spiral.txt 2Dspiral

# 3. Test clustering on generated sequence
run_test "Spiral Clustering (2Dspiral)" \
    ./gric-cluster 0.1 /tmp/test_spiral.txt -maxim 1000 -outdir /tmp/test_spiral_out

# 4. Test bouncing balls generator and clustering
if [ -f "./gric-gen-balls" ]; then
    run_test "Bouncing Balls Generator (1 ball)" \
        ./gric-gen-balls -n 1 -r 5.0 -W 32 -H 32 -f 500 -s 42 /tmp/test_balls_1.fits

    run_test "Bouncing Balls Clustering (1 ball)" \
        ./gric-cluster 3.0 /tmp/test_balls_1.fits -outdir /tmp/test_balls_1_out

    run_test "Bouncing Balls Generator (3 colliding balls)" \
        ./gric-gen-balls -n 3 -r 5.0 -W 32 -H 32 -f 500 -s 42 /tmp/test_balls_3.fits

    run_test "Bouncing Balls Clustering (3 colliding balls)" \
        ./gric-cluster 3.0 /tmp/test_balls_3.fits -outdir /tmp/test_balls_3_out
else
    echo "Skipping bouncing balls tests (gric-gen-balls not built / CFITSIO not enabled)"
fi

# 5. Test Python wrapper
if command -v python3 > /dev/null 2>&1; then
    run_test "Python Wrapper Unit Tests" \
        python3 "${ROOT_DIR}/python/test_wrapper.py"
fi

# 6. Test benchmark utility smoke test
if [ -f "./gric-benchmark" ]; then
    run_test "Benchmark Runner (smoke test 2Dspiral & balls_single)" \
        ./gric-benchmark -p 2Dspiral -p balls_single -n 500
fi

# Cleanup temporary test files
rm -rf /tmp/test_strat_out /tmp/test_spiral.txt /tmp/test_spiral_out \
       /tmp/test_balls_1.fits /tmp/test_balls_1_out \
       /tmp/test_balls_3.fits /tmp/test_balls_3_out

echo "=================================================="
echo " Test Results: ${TESTS_PASSED} Passed, ${TESTS_FAILED} Failed"
echo "=================================================="

if [ ${TESTS_FAILED} -ne 0 ]; then
    exit 1
fi
exit 0

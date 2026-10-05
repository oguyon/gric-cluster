#!/usr/bin/env bash
set -euo pipefail

# scripts/check_pr_invariants.sh
# Verifies cross-platform, memory alignment, and style invariants before PR merge.

ERRORS=0

echo "=== Checking PR Invariants ==="

# 1. Line lengths <= 100 columns
echo "[1/5] Checking line lengths (<= 100 columns)..."
if git grep -n '.\{101,\}' -- \
    'src/**.[ch]' 'include/**.[ch]*' 'tests/**.[ch]*' \
    'tools/**.[ch]' 'benchmarks/**.[ch]' \
    ':!*font8x16.h' ':!*cjson/*' ':!src/third_party/*'; then
    echo "ERROR: Lines exceeding 100 characters found!" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  PASS: Line lengths within 100 columns."
fi

# 2. Portability: pipe2 calls must be guarded for Darwin/macOS
echo "[2/5] Checking POSIX pipe portability..."
PIPE2_COUNT=$(git grep -n -E '(^|[^a-zA-Z0-9_])pipe2\(' -- 'src/**.[ch]' 'tools/**.[ch]' | \
    grep -v "mcp_exec.c" | wc -l || true)
if [ "$PIPE2_COUNT" -gt 0 ]; then
    echo "ERROR: Naked pipe2() call found without Darwin fallback!" >&2
    git grep -n -E '(^|[^a-zA-Z0-9_])pipe2\(' -- 'src/**.[ch]' 'tools/**.[ch]' | \
        grep -v "mcp_exec.c" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  PASS: No naked pipe2() calls."
fi

# 3. Memory alignment: raw aligned_alloc calls must use safe rounded wrapper
echo "[3/5] Checking aligned_alloc sizing safety..."
RAW_ALIGNED=$(git grep -n -E '(^|[^a-zA-Z0-9_])aligned_alloc\(' -- 'src/**.[ch]' 'tools/**.[ch]' | \
    grep -v "bench_simd_scaling.c:.*aligned_alloc(alignment, size)" | wc -l || true)
if [ "$RAW_ALIGNED" -gt 0 ]; then
    echo "ERROR: Raw aligned_alloc() call found without alignment rounding!" >&2
    git grep -n -E '(^|[^a-zA-Z0-9_])aligned_alloc\(' -- 'src/**.[ch]' 'tools/**.[ch]' | \
        grep -v "bench_simd_scaling.c:.*aligned_alloc(alignment, size)" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  PASS: No unprotected aligned_alloc() calls."
fi

# 4. Portability: /proc/self/exe must have Darwin fallback
echo "[4/5] Checking /proc/self/exe portability..."
PROC_EXE=$(git grep -n "/proc/self/exe" -- 'src/**.[ch]' 'tools/**.[ch]' | \
    grep -vE "(mcp_tools\.c|mcp_staleness\.c|server_main\.c)" | wc -l || true)
if [ "$PROC_EXE" -gt 0 ]; then
    echo "ERROR: /proc/self/exe assumed without Darwin fallback!" >&2
    git grep -n "/proc/self/exe" -- 'src/**.[ch]' 'tools/**.[ch]' | \
        grep -vE "(mcp_tools\.c|mcp_staleness\.c|server_main\.c)" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  PASS: No naked /proc/self/exe assumptions."
fi

# 5. Code size and function scope limits (ratchet baseline)
echo "[5/5] Checking code size and function scope limits..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! "$SCRIPT_DIR/check_code_size.sh"; then
    echo "ERROR: Code size constraints or baseline regression detected!" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "  PASS: Code size and function scope within baseline."
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "FAILED: $ERRORS invariant check(s) failed!" >&2
    exit 1
fi

echo "=== All PR Invariant Checks PASSED! ==="

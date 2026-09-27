#!/usr/bin/env bash
# ==============================================================================
# scripts/demo_milk_fps_clustering.sh
#
# Demonstration of GRIC stream clustering in Milk framework mode:
# - Auto mode: runs from start to finish, streams a 3D spiral dataset (N=1000,
#   M=20 repeats, noise=0.1 -> 20,000 samples), displays real-time telemetry,
#   stops cleanly per milk-fpsexec conventions, and saves results in local
#   directory in GRIC binary format.
# - Interactive mode: step-by-step setup with interactive control & telemetry.
# ==============================================================================

set -euo pipefail

# ANSI color codes
BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
MAGENTA="\033[1;35m"
RED="\033[1;31m"
RESET="\033[0m"

# Default configuration parameters
AUTO_MODE=0
FPS_NAME="demo"
STREAM_IN="demo_in"
STREAM_OUT="demo_assign"
PATTERN_FILE="3Dspiral.txt"
PATTERN_TYPE="3Dspiral"
NB_POINTS=1000
NB_REPEATS=20
NOISE_RADIUS=0.1
STREAM_FPS=""
DEFAULT_RLIM=0.30
MAX_FRAMES=20000
SAVE_DIR="."

FEED_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# ------------------------------------------------------------------------------
# CLI Option Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--auto)
            AUTO_MODE=1
            shift
            ;;
        --fps)
            STREAM_FPS="$2"
            shift 2
            ;;
        -n|--frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --points)
            NB_POINTS="$2"
            shift 2
            ;;
        -m|--repeats)
            NB_REPEATS="$2"
            shift 2
            ;;
        --noise)
            NOISE_RADIUS="$2"
            shift 2
            ;;
        --rlim)
            DEFAULT_RLIM="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --name)
            FPS_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -a, --auto        Run automatically from start to finish"
            echo "  -n, --frames <N>  Number of frames to cluster (default: 20000)"
            echo "  --points <N>      Base points per pattern cycle (default: 1000)"
            echo "  -m, --repeats <M> Number of pattern repeats (default: 20)"
            echo "  --noise <val>     Random noise radius (default: 0.1)"
            echo "  --fps <N>         Streaming frame rate (default: 2000 auto / 100 interactive)"
            echo "  --rlim <val>      Clustering radius threshold (default: 0.30)"
            echo "  --save-dir <dir>  Directory to save binary results (default: .)"
            echo "  --name <fpsname>  FPS instance name (default: demo)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use -h for help)"
            exit 1
            ;;
    esac
done

if [[ "$AUTO_MODE" -eq 1 ]]; then
    STREAM_FPS="${STREAM_FPS:-2000}"
else
    STREAM_FPS="${STREAM_FPS:-100}"
fi
RUN_ELAPSED_MS="0"

# ------------------------------------------------------------------------------
# Resolve Executables (PATH or local build/ directory)
# ------------------------------------------------------------------------------
resolve_bin() {
    local bin_name="$1"
    if command -v "$bin_name" >/dev/null 2>&1; then
        echo "$bin_name"
    elif [[ -x "${BUILD_DIR}/${bin_name}" ]]; then
        echo "${BUILD_DIR}/${bin_name}"
    elif [[ -x "${BUILD_DIR}/src/gric-fps/${bin_name}" ]]; then
        echo "${BUILD_DIR}/src/gric-fps/${bin_name}"
    else
        echo ""
    fi
}

BIN_MKTXTSEQ="$(resolve_bin gric-mktxtseq)"
BIN_TXT2STREAM="$(resolve_bin gric-txt2stream)"
BIN_FPSEXEC="$(resolve_bin milk-fpsexec-gric-cluster)"
BIN_FPS_SET="$(resolve_bin milk-fps-set)"
BIN_STREAM2PIPE="$(resolve_bin gric-stream-to-pipe)"
BIN_PROCCTRL="$(resolve_bin milk-procCTRL)"
BIN_GRIC_STATUS="$(resolve_bin gric-status)"
BIN_BIN2ASCII="$(resolve_bin gric-bin2ascii)"

# ------------------------------------------------------------------------------
# Command Logging and Execution Helpers (Prominent Bold/Color for Replication)
# ------------------------------------------------------------------------------
log_cmd() {
    echo -e "  ${BOLD}${GREEN}\$${RESET} ${BOLD}${YELLOW}$*${RESET}"
}

run_cmd() {
    log_cmd "$@"
    "$@"
}

# ------------------------------------------------------------------------------
# Final Results and Output Files Summary
# ------------------------------------------------------------------------------
display_final_summary() {
    # Check if any clustering output files exist
    if [[ ! -f "${SAVE_DIR}/frame_membership.bin" && \
          ! -f "${SAVE_DIR}/frame_membership.txt" && \
          ! -f "${SAVE_DIR}/anchors.bin" && \
          ! -f "${SAVE_DIR}/anchors.txt" ]]; then
        return 0
    fi

    # Ensure all existing .bin files have corresponding decoded .txt files
    for artifact in anchors cluster_counts cluster_radii dcc frame_membership; do
        if [[ -f "${SAVE_DIR}/${artifact}.bin" ]]; then
            if [[ ! -f "${SAVE_DIR}/${artifact}.txt" || \
                  "${SAVE_DIR}/${artifact}.bin" -nt "${SAVE_DIR}/${artifact}.txt" ]]; then
                if [[ -n "${BIN_BIN2ASCII}" ]]; then
                    "${BIN_BIN2ASCII}" -header \
                            "${SAVE_DIR}/${artifact}.bin" \
                            "${SAVE_DIR}/${artifact}.txt" >/dev/null 2>&1 || true
                fi
            fi
        fi
    done

    # Run Python summary formatter
    python3 - "${SAVE_DIR}" "${RUN_ELAPSED_MS:-0}" << 'EOF'
import os, re, sys
from collections import Counter

save_dir = sys.argv[1] if len(sys.argv) > 1 else "."
ms_str = sys.argv[2] if len(sys.argv) > 2 else "0"

def fmt_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:5.1f} {unit}" if unit != "B" else f"{num_bytes:5d} B "
        num_bytes /= 1024.0
    return f"{num_bytes:5.1f} TB"

# 1. Parse frame memberships and cluster distributions
tot_samples = 0
cluster_counts = Counter()
fm_file = os.path.join(save_dir, "frame_membership.txt")
if os.path.exists(fm_file):
    with open(fm_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                tot_samples += 1
                try:
                    cluster_counts[int(parts[1])] += 1
                except ValueError:
                    pass

# 2. Parse anchors metadata (dimension and clusters)
n_clusters = len(cluster_counts)
n_dim = 0
anchors_file = os.path.join(save_dir, "anchors.txt")
if os.path.exists(anchors_file):
    with open(anchors_file, "r") as f:
        for line in f:
            line = line.strip()
            if "Shape" in line:
                m = re.search(r"\[(\d+)\s+rows\s+x\s+(\d+)\s+columns\]", line)
                if m:
                    n_clusters = int(m.group(1))
                    n_dim = int(m.group(2))
            elif not line.startswith("#") and line:
                parts = line.split()
                if len(parts) > 1 and n_dim == 0:
                    n_dim = len(parts) - 1

# 3. Parse cluster radii
rlim = 0.0
radii_file = os.path.join(save_dir, "cluster_radii.txt")
if os.path.exists(radii_file):
    with open(radii_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#") and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rlim = float(parts[1])
                        break
                    except ValueError:
                        pass

# 4. Performance metrics
try:
    elapsed_ms = float(ms_str)
except ValueError:
    elapsed_ms = 0.0

time_fps_str = ""
if elapsed_ms > 0:
    elapsed_sec = elapsed_ms / 1000.0
    fps = tot_samples / elapsed_sec if elapsed_sec > 0 else 0
    time_fps_str = f"  Total Elapsed Time      : {elapsed_sec:.2f} s ({int(fps):,d} FPS)\n"

# 5. Cluster distribution statistics
stats_str = ""
top5_str = ""
if cluster_counts:
    counts = list(cluster_counts.values())
    min_c = min(counts)
    max_c = max(counts)
    mean_c = sum(counts) / len(counts)
    counts_sorted = sorted(counts)
    median_c = counts_sorted[len(counts_sorted) // 2]
    stats_str = (
        f"  Cluster Size Statistics : "
        f"Min: {min_c:,d} | Max: {max_c:,d} | Mean: {mean_c:.1f} | Median: {median_c:,d}\n"
    )
    top5 = cluster_counts.most_common(5)
    top5_lines = []
    for cid, cnt in top5:
        pct = (cnt * 100.0) / tot_samples if tot_samples > 0 else 0
        top5_lines.append(f"    - Cluster #{cid:<2d} : {cnt:6,d} samples ({pct:4.1f}%)")
    top5_str = "  Top Largest Clusters    :\n" + "\n".join(top5_lines) + "\n"

# 6. Artifact file table
artifacts = [
    ("anchors.bin", "Binary", f"Cluster centroids (float32 [{n_clusters} x {n_dim}])"),
    ("anchors.txt", "ASCII", f"Decoded centroids with index [{n_clusters} x {n_dim}]"),
    ("cluster_counts.bin", "Binary", f"Sample counts per cluster (uint32 [{n_clusters}])"),
    ("cluster_counts.txt", "ASCII", "Decoded sample counts per cluster"),
    ("cluster_radii.bin", "Binary", f"Cluster radius thresholds (float32 [{n_clusters}])"),
    ("cluster_radii.txt", "ASCII", "Decoded cluster radii with index"),
    ("dcc.bin", "Binary", f"Inter-cluster distances (float64 [{n_clusters} x {n_clusters}])"),
    ("dcc.txt", "ASCII", "Decoded pairwise DCC distance matrix"),
    ("frame_membership.bin", "Binary", f"Sample assignments (uint32 [{tot_samples}])"),
    ("frame_membership.txt", "ASCII", "Decoded sample assignments (sample_idx, cluster_id)"),
]

art_rows = []
for fname, fmt, desc in artifacts:
    fpath = os.path.join(save_dir, fname)
    if os.path.exists(fpath):
        sz = fmt_size(os.path.getsize(fpath))
        art_rows.append(f"  {fname:<22} {fmt:<8} {sz}   {desc}")

sep_80 = "=" * 80
subsep_80 = "-" * 80
print("\n" + sep_80)
print("                    GRIC CLUSTERING RUN & OUTPUT SUMMARY")
print(sep_80)
print("\n1. Clustering Performance & Results")
print(subsep_80)
print(f"  Total Samples Clustered : {tot_samples:,d} frames")
print(f"  Clusters Discovered     : {n_clusters:,d} clusters")
if n_dim > 0:
    print(f"  Feature Dimensionality  : {n_dim}D")
if rlim > 0:
    print(f"  Cluster Radius Limit    : rlim = {rlim:.6f}")
if time_fps_str:
    sys.stdout.write(time_fps_str)
if stats_str:
    sys.stdout.write(stats_str)
if top5_str:
    sys.stdout.write(top5_str)

abs_dir = os.path.abspath(save_dir)
print(f"\n2. Generated Output Files (in {abs_dir})")
print(subsep_80)
print("  File Name              Format   Size      Description")
print("  " + "-" * 76)
for row in art_rows:
    print(row)

print("\n3. Quick Inspection Commands")
print(subsep_80)
print("  # View decoded cluster centroids (first 25):")
print(f"  head -n 25 {save_dir}/anchors.txt\n")
print("  # View sample-to-cluster assignments (first 25):")
print(f"  head -n 25 {save_dir}/frame_membership.txt\n")
print("  # View cluster member counts:")
print(f"  cat {save_dir}/cluster_counts.txt\n")
print("  # Inspect binary file header metadata:")
print(f"  gric-bin2ascii -i {save_dir}/anchors.bin\n")
print("  # View inter-cluster distance matrix:")
print(f"  head -n 20 {save_dir}/dcc.txt")
print(sep_80 + "\n")
EOF
}

# ------------------------------------------------------------------------------
# Cleanup Handler
# ------------------------------------------------------------------------------
cleanup() {
    echo -e "\n${BOLD}${YELLOW}[Cleanup] Stopping processes and freeing shared memory...${RESET}"

    # Preserve elapsed execution time if not already captured
    if [[ -z "${RUN_ELAPSED_MS:-}" || "${RUN_ELAPSED_MS}" == "0" ]]; then
        local sf="${MILK_SHM_DIR:-/milk/shm}/fps.${FPS_NAME}.status.shm"
        [[ ! -f "$sf" ]] && sf="/dev/shm/fps.${FPS_NAME}.status.shm"
        if [[ -f "$sf" ]]; then
            RUN_ELAPSED_MS="$(python3 -c '
import struct, sys
try:
    with open(sys.argv[1], "rb") as f:
        print(f"{struct.unpack_from(\"d\", f.read(128), 80)[0]:.1f}")
except Exception:
    print("0")
' "$sf" 2>/dev/null || echo "0")"
        fi
    fi

    if [[ -n "${BIN_FPSEXEC}" ]]; then
        log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:runstop"
        "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:runstop" >/dev/null 2>&1 || true

        log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:confstop"
        "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:confstop" >/dev/null 2>&1 || true
    fi

    if [[ -n "${FEED_PID:-}" ]] && kill -0 "${FEED_PID}" 2>/dev/null; then
        log_cmd "kill -INT ${FEED_PID}"
        kill -INT "${FEED_PID}" 2>/dev/null || true
        wait "${FEED_PID}" 2>/dev/null || true
        FEED_PID=""
    fi

    log_cmd "tmux kill-session -t ${FPS_NAME}"
    tmux kill-session -t "${FPS_NAME}" 2>/dev/null || true

    # Clean up SHM streams (preserved binary results on disk are untouched)
    local shm_dirs=("/dev/shm" "${MILK_SHM_DIR:-/milk/shm}")
    for d in "${shm_dirs[@]}"; do
        if [[ -d "$d" ]]; then
            rm -f "${d}/${STREAM_IN}"* "${d}/${STREAM_OUT}"* \
                  "${d}/fps.${FPS_NAME}"* "${d}/proc.${FPS_NAME}"* \
                  "${d}/${FPS_NAME}."* 2>/dev/null || true
        fi
    done
    echo -e "${BOLD}${GREEN}[Cleanup] Done.${RESET}"

    # Display final summary of clustering results and output files
    display_final_summary
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# Dependency Verification
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}=== GRIC Milk Framework Clustering Demo ===${RESET}\n"

check_dep() {
    local name="$1"
    local bin="$2"
    if [[ -z "$bin" ]]; then
        echo -e "${RED}Error:${RESET} Required binary '${name}' not found in PATH or ${BUILD_DIR}."
        echo "Please build the project with Milk support enabled:"
        echo "  cmake -B build -DUSE_MILK=ON && cmake --build build -j\$(nproc)"
        exit 1
    fi
    local loc
    loc="$(command -v "$bin" 2>/dev/null || echo "$bin")"
    echo -e "  [x] Found ${GREEN}${name}${RESET} -> ${loc}"
}

check_dep "gric-mktxtseq" "${BIN_MKTXTSEQ}"
check_dep "gric-txt2stream" "${BIN_TXT2STREAM}"
check_dep "milk-fpsexec-gric-cluster" "${BIN_FPSEXEC}"
check_dep "milk-fps-set" "${BIN_FPS_SET}"
check_dep "gric-stream-to-pipe" "${BIN_STREAM2PIPE}"
check_dep "gric-status" "${BIN_GRIC_STATUS}"
check_dep "gric-bin2ascii" "${BIN_BIN2ASCII}"
echo ""

# ==============================================================================
# AUTO MODE EXECUTION
# ==============================================================================
if [[ "$AUTO_MODE" -eq 1 ]]; then
    echo -e "${BOLD}${MAGENTA}>>> Running in AUTO MODE (${MAX_FRAMES} samples) <<<${RESET}\n"

    # Step 1: Generate dataset
    local_tot=$((NB_POINTS * NB_REPEATS))
    echo -e "${BOLD}${CYAN}[1/5] Generating dataset:${RESET} ${PATTERN_FILE}"
    echo -e "  (${NB_POINTS} pts x ${NB_REPEATS} repeats, noise=${NOISE_RADIUS})"
    run_cmd "${BIN_MKTXTSEQ}" "${NB_POINTS}" "${PATTERN_FILE}" "${PATTERN_TYPE}" \
            -repeat "${NB_REPEATS}" -noise "${NOISE_RADIUS}"

    # Step 2: Initialize & configure FPS daemon
    echo -e "${BOLD}${CYAN}[2/5] Initializing FPS daemon in procinfo mode:${RESET} ${FPS_NAME}..."
    run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fpsinit"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.in_name" "${STREAM_IN}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.out_name" "${STREAM_OUT}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${DEFAULT_RLIM}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.max_frames" "${MAX_FRAMES}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.cnt2sync" "ON"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.save_dir" "${SAVE_DIR}"

    # Step 3: Spin up looping feeder stream (cnt2 flow-control with max FPS cap)
    echo -e "${BOLD}${CYAN}[3/5] Feeder stream (cnt2 gated, max ${STREAM_FPS} FPS):${RESET} " \
            "${STREAM_IN}..."
    log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -cnt2sync -fps ${STREAM_FPS} -loop &"
    "${BIN_TXT2STREAM}" "${PATTERN_FILE}" "${STREAM_IN}" -cnt2sync -fps "${STREAM_FPS}" -loop &
    FEED_PID=$!
    sleep 0.1

    # Step 4: Launch clustering daemon in tmux
    echo -e "${BOLD}${CYAN}[4/5] Launching daemon in tmux session '${FPS_NAME}'...${RESET}"
    run_cmd "${BIN_FPSEXEC}" -tmux -procinfo -loops "${FPS_NAME}:runstart"
    sleep 0.3

    # Step 5: Real-time telemetry status display loop
    echo -e "${BOLD}${CYAN}[5/5] Real-time clustering status:${RESET}"
    echo -e "  To monitor live telemetry in another terminal:"
    log_cmd "${BIN_GRIC_STATUS} ${FPS_NAME} -w"
    echo ""
    SHM_FILE=""

    while true; do
        if [[ -z "${SHM_FILE}" || ! -f "${SHM_FILE}" ]]; then
            for cand in "${MILK_SHM_DIR:-/milk/shm}/fps.${FPS_NAME}.status.shm" \
                        "/dev/shm/fps.${FPS_NAME}.status.shm" \
                        "/tmp/fps.${FPS_NAME}.status.shm"; do
                if [[ -f "$cand" ]]; then
                    SHM_FILE="$cand"
                    break
                fi
            done
        fi

        if [[ -n "${SHM_FILE}" && -f "${SHM_FILE}" ]]; then
            read -r proc tot nclust ms state < <(python3 -c '
import struct, sys
try:
    with open(sys.argv[1], "rb") as f:
        data = f.read(128)
        magic, ver, pid, state = struct.unpack_from("4I", data, 0)
        tot, proc = struct.unpack_from("2Q", data, 16)
        nclust = struct.unpack_from("I", data, 32)[0]
        ms = struct.unpack_from("d", data, 80)[0]
        print(f"{proc} {tot} {nclust} {ms:.1f} {state}")
except Exception:
    print("0 0 0 0.0 0")
' "${SHM_FILE}" 2>/dev/null || echo "0 0 0 0.0 0")

            if [[ "$tot" -gt 0 ]]; then
                pct=$(( (proc * 100) / tot ))
                rate=0
                if (( $(echo "$ms > 0" | bc -l 2>/dev/null || echo 0) )); then
                    rate=$(python3 -c \
                        "print(int(${proc} / (${ms} / 1000.0)))" 2>/dev/null || echo 0)
                fi
                sec=$(python3 -c "print(f'{${ms}/1000.0:.2f}')" 2>/dev/null || echo "0.00")
                st_label="RUNNING"
                [[ "$state" -eq 2 ]] && st_label="SUCCESS"
                [[ "$state" -eq 3 ]] && st_label="ERROR"

                fmt="\r${BOLD}${CYAN}[%s]${RESET} %-7s | "
                fmt+="Frames: ${BOLD}%'d / %'d${RESET} (%2d%%) | "
                fmt+="Clusters: ${YELLOW}%'d${RESET} | Rate: %'d FPS | Time: %ss "
                printf "$fmt" \
                       "${FPS_NAME}" "${st_label}" "$proc" "$tot" "$pct" "$nclust" "$rate" "$sec"
            fi

            if [[ "$state" -ge 2 ]] || { [[ "$tot" -gt 0 ]] && [[ "$proc" -ge "$tot" ]]; }; then
                RUN_ELAPSED_MS="${ms:-0}"
                break
            fi
        fi
        sleep 0.1
    done

    # Stop background feeder immediately so extra frames are not streamed during post-proc
    if [[ -n "${FEED_PID:-}" ]] && kill -0 "${FEED_PID}" 2>/dev/null; then
        echo -e "\n\n${BOLD}${CYAN}[Stream Feeder]${RESET} Stopping background camera feeder..."
        log_cmd "kill -INT ${FEED_PID}"
        kill -INT "${FEED_PID}" 2>/dev/null || true
        wait "${FEED_PID}" 2>/dev/null || true
        FEED_PID=""
    fi

    echo -e "\n${BOLD}${GREEN}Clustering completed successfully!${RESET}"
    echo -e "  Clustered samples : ${BOLD}${proc}${RESET} / ${tot} (requested: ${MAX_FRAMES})"
    echo -e "  (Feeder was gated via cnt2 flow control at the exact rate of gric-cluster)\n"

    # Display final status telemetry
    echo -e "${BOLD}${CYAN}--- Final Status Telemetry ---${RESET}"
    if [[ -n "${BIN_GRIC_STATUS}" ]]; then
        run_cmd "${BIN_GRIC_STATUS}" "${FPS_NAME}" || true
    fi

    # Display saved binary results
    echo -e "\n${BOLD}${CYAN}--- Saved Binary Clustering Results ---${RESET}"
    echo -e "Artifacts in ${GREEN}${SAVE_DIR}${RESET}:"
    log_cmd "ls -lh ${SAVE_DIR}/anchors.bin ${SAVE_DIR}/dcc.bin" \
            "${SAVE_DIR}/cluster_counts.bin ${SAVE_DIR}/cluster_radii.bin" \
            "${SAVE_DIR}/frame_membership.bin"
    ls -lh "${SAVE_DIR}"/anchors.bin "${SAVE_DIR}"/dcc.bin \
           "${SAVE_DIR}"/cluster_counts.bin "${SAVE_DIR}"/cluster_radii.bin \
           "${SAVE_DIR}"/frame_membership.bin 2>/dev/null || true
    echo ""

    if [[ -f "${SAVE_DIR}/anchors.bin" ]]; then
        echo -e "${BOLD}Anchors Binary Header Inspection:${RESET}"
        run_cmd "${BIN_BIN2ASCII}" -i "${SAVE_DIR}/anchors.bin"
    fi

    # Decode binary files into ASCII files for all outputs
    echo -e "\n${BOLD}${CYAN}--- Decoding Binary Outputs to ASCII ---${RESET}"
    for artifact in anchors cluster_counts cluster_radii dcc frame_membership; do
        if [[ -f "${SAVE_DIR}/${artifact}.bin" ]]; then
            run_cmd "${BIN_BIN2ASCII}" -header \
                    "${SAVE_DIR}/${artifact}.bin" \
                    "${SAVE_DIR}/${artifact}.txt"
        fi
    done

    echo -e "\n${BOLD}${CYAN}Decoded ASCII Artifacts in ${GREEN}${SAVE_DIR}${RESET}:"
    log_cmd "ls -lh ${SAVE_DIR}/anchors.txt ${SAVE_DIR}/dcc.txt" \
            "${SAVE_DIR}/cluster_counts.txt ${SAVE_DIR}/cluster_radii.txt" \
            "${SAVE_DIR}/frame_membership.txt"
    ls -lh "${SAVE_DIR}"/anchors.txt "${SAVE_DIR}"/dcc.txt \
           "${SAVE_DIR}"/cluster_counts.txt "${SAVE_DIR}"/cluster_radii.txt \
           "${SAVE_DIR}"/frame_membership.txt 2>/dev/null || true
    echo ""

    if [[ -f "${SAVE_DIR}/anchors.txt" ]]; then
        echo -e "${BOLD}Sample Decoded Anchors Header & Centroids:${RESET}"
        head -n 16 "${SAVE_DIR}/anchors.txt"
        echo ""
    fi

    if [[ -f "${SAVE_DIR}/cluster_radii.txt" ]]; then
        echo -e "${BOLD}Sample Decoded Cluster Radii Header & Values:${RESET}"
        head -n 16 "${SAVE_DIR}/cluster_radii.txt"
        echo ""
    fi

    if [[ -f "${SAVE_DIR}/frame_membership.txt" ]]; then
        echo -e "${BOLD}Sample Decoded Frame Memberships Header & Assignments:${RESET}"
        head -n 16 "${SAVE_DIR}/frame_membership.txt"
        echo ""
    fi

    echo -e "\n${BOLD}${CYAN}====================================================${RESET}"
    echo -e "${BOLD}${CYAN}       Summary of Step-by-Step Replication Commands ${RESET}"
    echo -e "${BOLD}${CYAN}====================================================${RESET}"
    echo -e "# 1. Generate pattern dataset:"
    log_cmd "${BIN_MKTXTSEQ} ${NB_POINTS} ${PATTERN_FILE} ${PATTERN_TYPE}" \
            "-repeat ${NB_REPEATS} -noise ${NOISE_RADIUS}"
    echo -e "\n# 2. Initialize FPS daemon and configure parameters:"
    log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:fpsinit"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.in_name ${STREAM_IN}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.out_name ${STREAM_OUT}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.rlim ${DEFAULT_RLIM}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.max_frames ${MAX_FRAMES}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.cnt2sync ON"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.save_dir ${SAVE_DIR}"
    echo -e "\n# 3. Feed stream into shared memory (flow-controlled by consumer cnt2, max FPS):"
    log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -cnt2sync -fps ${STREAM_FPS} -loop &"
    echo -e "\n# 4. Launch clustering daemon in tmux session:"
    log_cmd "${BIN_FPSEXEC} -tmux -procinfo -loops ${FPS_NAME}:runstart"
    echo -e "\n# 5. Monitor status (live watch):"
    log_cmd "${BIN_GRIC_STATUS} ${FPS_NAME} -w"
    echo -e "\n# 6. Stop stream feeder (clustering complete):"
    log_cmd "kill -INT <feeder_pid>"
    echo -e "\n# 7. Inspect binary output header:"
    log_cmd "${BIN_BIN2ASCII} -i ${SAVE_DIR}/anchors.bin"
    echo -e "\n# 8. Decode binary outputs into ASCII files (with headers):"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/anchors.bin ${SAVE_DIR}/anchors.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/cluster_counts.bin ${SAVE_DIR}/cluster_counts.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/cluster_radii.bin ${SAVE_DIR}/cluster_radii.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/frame_membership.bin ${SAVE_DIR}/frame_membership.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/dcc.bin ${SAVE_DIR}/dcc.txt"
    echo -e "\n# 9. Stop clustering daemon and clean up:"
    log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:runstop"
    log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:confstop"
    log_cmd "tmux kill-session -t ${FPS_NAME}"
    echo -e "${BOLD}${CYAN}====================================================${RESET}"

    echo -e "\n${GREEN}Auto mode completed. Cleaning up processes and shared memory...${RESET}"
    exit 0
fi

# ==============================================================================
# INTERACTIVE MODE EXECUTION
# ==============================================================================

# Step 1: Generate Pattern Dataset
echo -e "${BOLD}${CYAN}--- Step 1: Generate Synthetic Pattern ---${RESET}"
echo -e "Default pattern: ${GREEN}${PATTERN_TYPE}${RESET} " \
        "(${NB_POINTS} pts, ${NB_REPEATS} repeats, noise=${NOISE_RADIUS})"
read -r -p "Enter base points per cycle [default: ${NB_POINTS}]: " user_pts
if [[ -n "${user_pts}" ]]; then
    NB_POINTS="${user_pts}"
fi
read -r -p "Enter number of repeats [default: ${NB_REPEATS}]: " user_rep
if [[ -n "${user_rep}" ]]; then
    NB_REPEATS="${user_rep}"
fi
read -r -p "Enter noise radius [default: ${NOISE_RADIUS}]: " user_noise
if [[ -n "${user_noise}" ]]; then
    NOISE_RADIUS="${user_noise}"
fi

TOTAL_DATASET_PTS=$((NB_POINTS * NB_REPEATS))
echo -e "Generating ${GREEN}${PATTERN_FILE}${RESET} (${TOTAL_DATASET_PTS} total points)..."
run_cmd "${BIN_MKTXTSEQ}" "${NB_POINTS}" "${PATTERN_FILE}" "${PATTERN_TYPE}" \
        -repeat "${NB_REPEATS}" -noise "${NOISE_RADIUS}"
echo -e "Sample data (first 3 points):"
head -n 3 "${PATTERN_FILE}"
echo ""

# Step 2: Initialize & Configure FPS Instance
echo -e "${BOLD}${CYAN}--- Step 2: Initialize FPS in Procinfo Mode ---${RESET}"
run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fpsinit"

if [[ "$MAX_FRAMES" -eq 20000 && "$TOTAL_DATASET_PTS" -ne 20000 ]]; then
    MAX_FRAMES="$TOTAL_DATASET_PTS"
fi

echo -e "Binding parameters via ${GREEN}milk-fps-set${RESET}:"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.in_name" "${STREAM_IN}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.out_name" "${STREAM_OUT}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${DEFAULT_RLIM}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.max_frames" "${MAX_FRAMES}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.cnt2sync" "ON"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.save_dir" "${SAVE_DIR}"

echo -e "\n${BOLD}Active FPS Configuration:${RESET}"
run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fps"
echo ""

# Step 3: Feed Input Stream
echo -e "${BOLD}${CYAN}--- Step 3: Feed ImageStreamIO Ring Buffer (Flow-Controlled) ---${RESET}"
echo -e "Streaming ${GREEN}${PATTERN_FILE}${RESET} (cnt2 gated, max ${STREAM_FPS} FPS)..."
log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -cnt2sync -fps ${STREAM_FPS} -loop &"
"${BIN_TXT2STREAM}" "${PATTERN_FILE}" "${STREAM_IN}" -cnt2sync -fps "${STREAM_FPS}" -loop &
FEED_PID=$!
sleep 0.3
echo -e "Feeder started (PID: ${FEED_PID}, stream: ${GREEN}${STREAM_IN}${RESET}, gated by cnt2).\n"

# Step 4: Launch Clustering Daemon
echo -e "${BOLD}${CYAN}--- Step 4: Launch Clustering Daemon ---${RESET}"
echo -e "Starting daemon in isolated tmux session with semaphore triggering..."
run_cmd "${BIN_FPSEXEC}" -tmux -procinfo -loops "${FPS_NAME}:runstart"
sleep 1.0

if tmux has-session -t "${FPS_NAME}" 2>/dev/null; then
    echo -e "${GREEN}Daemon successfully running in tmux session '${FPS_NAME}'.${RESET}\n"
else
    echo -e "${RED}Warning: tmux session '${FPS_NAME}' not found.${RESET}\n"
fi

# Step 5: Interactive Control & Telemetry Loop
print_menu() {
    echo -e "${BOLD}${CYAN}=== Interactive Control Menu ===${RESET}"
    echo "  [1] Sample live clustering output (5 frames)"
    echo "  [2] Inspect live FPS status (.status.* parameters)"
    echo "  [3] Launch gric-status interactive dashboard (gric-status demo -w)"
    echo "  [4] Adjust clustering radius (rlim) live"
    echo "  [5] Reset cluster state dynamically (reset_state ON)"
    echo "  [6] Open milk-procCTRL telemetry dashboard"
    echo "  [7] View tmux daemon log"
    echo "  [8] Decode binary outputs to ASCII (anchors, counts, radii, dcc)"
    echo "  [q] Quit and clean up"
    echo ""
}

read_telemetry_frames() {
    echo -e "\n${MAGENTA}Sampling 5 frames from '${STREAM_OUT}'...${RESET}"
    "${BIN_STREAM2PIPE}" "${STREAM_OUT}" 5 | python3 -c '
import sys, struct
labels = ["cnt0", "cluster_id", "dist", "new_anchor", "nclust", "lat_us", "evals", "aux"]
for i in range(5):
    raw = sys.stdin.buffer.read(32)
    if not raw: break
    vals = struct.unpack("8f", raw)
    print(f"  Frame {i:2d} -> " + " | ".join(f"{k}: {v:6.2f}" for k, v in zip(labels, vals)))
' || echo -e "${YELLOW}Waiting for frames...${RESET}"
    echo ""
}

while true; do
    print_menu
    read -r -p "Select an option [1-8, q]: " choice
    case "$choice" in
        1)
            log_cmd "${BIN_STREAM2PIPE} ${STREAM_OUT} 5"
            read_telemetry_frames
            ;;
        2)
            echo -e "\n${BOLD}${CYAN}--- Live FPS Status Parameters (${FPS_NAME}:fps) ---${RESET}"
            log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:fps"
            "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fps" | \
                grep -E "status\.|rlim|in_name|out_name|max_frames|save_dir" || true
            echo ""
            ;;
        3)
            if [[ -n "${BIN_GRIC_STATUS}" ]]; then
                echo -e "${CYAN}Launching gric-status dashboard (press 'q' to return)...${RESET}"
                run_cmd "${BIN_GRIC_STATUS}" "${FPS_NAME}" -w || true
            else
                echo -e "${YELLOW}gric-status not found.${RESET}\n"
            fi
            ;;
        4)
            read -r -p "Enter new clustering radius (rlim) [e.g. 0.20, 0.60]: " new_rlim
            if [[ -n "${new_rlim}" ]]; then
                run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${new_rlim}"
                echo -e "${GREEN}Parameter ${FPS_NAME}.rlim updated to ${new_rlim}.${RESET}\n"
            fi
            ;;
        5)
            echo -e "${YELLOW}Triggering dynamic cluster reset...${RESET}"
            run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.reset_state" "ON"
            echo -e "${GREEN}Cluster state reset command sent.${RESET}\n"
            ;;
        6)
            if [[ -n "${BIN_PROCCTRL}" ]]; then
                echo -e "${CYAN}Launching milk-procCTRL (press 'q' inside TUI to return)...${RESET}"
                run_cmd "${BIN_PROCCTRL}" || true
            else
                echo -e "${YELLOW}milk-procCTRL not found in PATH.${RESET}\n"
            fi
            ;;
        7)
            echo -e "\n${CYAN}--- tmux pane ${FPS_NAME}:2 output ---${RESET}"
            log_cmd "tmux capture-pane -p -t ${FPS_NAME}:2"
            tmux capture-pane -p -t "${FPS_NAME}:2" 2>/dev/null | tail -n 15 || \
                echo "Unable to read tmux pane."
            echo -e "${CYAN}-------------------------------${RESET}\n"
            ;;
        8)
            echo -e "\n${BOLD}${CYAN}--- Decoding Binary Outputs to ASCII ---${RESET}"
            for artifact in anchors cluster_counts cluster_radii dcc frame_membership; do
                if [[ -f "${SAVE_DIR}/${artifact}.bin" ]]; then
                    run_cmd "${BIN_BIN2ASCII}" -header \
                            "${SAVE_DIR}/${artifact}.bin" \
                            "${SAVE_DIR}/${artifact}.txt"
                else
                    echo -e "${YELLOW}File '${SAVE_DIR}/${artifact}.bin' not found.${RESET}"
                fi
            done
            echo ""
            ;;
        q|Q)
            echo -e "${GREEN}Exiting...${RESET}"
            break
            ;;
        *)
            echo -e "${RED}Invalid option '${choice}'. Please select 1-8 or q.${RESET}\n"
            ;;
    esac
done

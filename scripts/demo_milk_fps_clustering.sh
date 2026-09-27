#!/usr/bin/env bash
# ==============================================================================
# scripts/demo_milk_fps_clustering.sh
#
# Demonstration of GRIC stream clustering in Milk framework mode:
# - Auto mode: runs from start to finish, streams a looping dataset, clusters
#   10,000 samples, displays real-time telemetry, stops cleanly per milk-fpsexec
#   conventions, and saves results in local directory in GRIC binary format.
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
PATTERN_FILE="2Dspiral.txt"
PATTERN_TYPE="2Dspiral"
NB_POINTS=2000
STREAM_FPS=""
DEFAULT_RLIM=0.40
MAX_FRAMES=10000
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
            echo "  -n, --frames <N>  Number of frames to cluster (default: 10000)"
            echo "  --fps <N>         Streaming frame rate (default: 2000 auto / 100 interactive)"
            echo "  --rlim <val>      Clustering radius threshold (default: 0.40)"
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
    NB_POINTS="${MAX_FRAMES}"
    STREAM_FPS="${STREAM_FPS:-2000}"
else
    STREAM_FPS="${STREAM_FPS:-100}"
fi

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
# Cleanup Handler
# ------------------------------------------------------------------------------
cleanup() {
    echo -e "\n${BOLD}${YELLOW}[Cleanup] Stopping processes and freeing shared memory...${RESET}"
    if [[ -n "${BIN_FPSEXEC}" ]]; then
        log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:runstop"
        "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:runstop" >/dev/null 2>&1 || true

        log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:confstop"
        "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:confstop" >/dev/null 2>&1 || true
    fi

    if [[ -n "${FEED_PID}" ]] && kill -0 "${FEED_PID}" 2>/dev/null; then
        log_cmd "kill -INT ${FEED_PID}"
        kill -INT "${FEED_PID}" 2>/dev/null || true
        wait "${FEED_PID}" 2>/dev/null || true
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
    echo -e "${BOLD}${CYAN}[1/5] Generating dataset:${RESET} ${PATTERN_FILE} (${NB_POINTS} pts)..."
    run_cmd "${BIN_MKTXTSEQ}" "${NB_POINTS}" "${PATTERN_FILE}" "${PATTERN_TYPE}"

    # Step 2: Spin up looping feeder stream
    echo -e "${BOLD}${CYAN}[2/5] Spinning looping feeder stream:${RESET} " \
            "${STREAM_IN} (${STREAM_FPS} FPS)..."
    log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -fps ${STREAM_FPS} -loop &"
    "${BIN_TXT2STREAM}" "${PATTERN_FILE}" "${STREAM_IN}" -fps "${STREAM_FPS}" -loop &
    FEED_PID=$!
    sleep 0.3

    # Step 3: Initialize & configure FPS daemon
    echo -e "${BOLD}${CYAN}[3/5] Initializing FPS daemon in procinfo mode:${RESET} ${FPS_NAME}..."
    run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fpsinit"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.in_name" "${STREAM_IN}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.out_name" "${STREAM_OUT}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${DEFAULT_RLIM}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.max_frames" "${MAX_FRAMES}"
    run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.save_dir" "${SAVE_DIR}"

    # Step 4: Launch clustering daemon in tmux
    echo -e "${BOLD}${CYAN}[4/5] Launching daemon in tmux session '${FPS_NAME}'...${RESET}"
    run_cmd "${BIN_FPSEXEC}" -tmux -procinfo -loops "${FPS_NAME}:runstart"
    sleep 0.5

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
        ms = struct.unpack_from("d", data, 72)[0]
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
                break
            fi
        fi
        sleep 0.1
    done

    echo -e "\n\n${BOLD}${GREEN}Clustering completed successfully!${RESET}\n"

    # Display final status telemetry
    echo -e "${BOLD}${CYAN}--- Final Status Telemetry ---${RESET}"
    if [[ -n "${BIN_GRIC_STATUS}" ]]; then
        run_cmd "${BIN_GRIC_STATUS}" "${FPS_NAME}" || true
    fi

    # Display saved binary results
    echo -e "\n${BOLD}${CYAN}--- Saved Binary Clustering Results ---${RESET}"
    echo -e "Artifacts in ${GREEN}${SAVE_DIR}${RESET}:"
    log_cmd "ls -lh ${SAVE_DIR}/anchors.bin ${SAVE_DIR}/dcc.bin" \
            "${SAVE_DIR}/cluster_counts.bin ${SAVE_DIR}/cluster_radii.bin"
    ls -lh "${SAVE_DIR}"/anchors.bin "${SAVE_DIR}"/dcc.bin \
           "${SAVE_DIR}"/cluster_counts.bin "${SAVE_DIR}"/cluster_radii.bin 2>/dev/null || true
    echo ""

    if [[ -f "${SAVE_DIR}/anchors.bin" ]]; then
        echo -e "${BOLD}Anchors Binary Header Inspection:${RESET}"
        run_cmd "${BIN_BIN2ASCII}" -i "${SAVE_DIR}/anchors.bin"
    fi

    # Decode binary files into ASCII files for all outputs
    echo -e "\n${BOLD}${CYAN}--- Decoding Binary Outputs to ASCII ---${RESET}"
    for artifact in anchors cluster_counts cluster_radii dcc; do
        if [[ -f "${SAVE_DIR}/${artifact}.bin" ]]; then
            run_cmd "${BIN_BIN2ASCII}" -header \
                    "${SAVE_DIR}/${artifact}.bin" \
                    "${SAVE_DIR}/${artifact}.txt"
        fi
    done

    echo -e "\n${BOLD}${CYAN}Decoded ASCII Artifacts in ${GREEN}${SAVE_DIR}${RESET}:"
    log_cmd "ls -lh ${SAVE_DIR}/anchors.txt ${SAVE_DIR}/dcc.txt" \
            "${SAVE_DIR}/cluster_counts.txt ${SAVE_DIR}/cluster_radii.txt"
    ls -lh "${SAVE_DIR}"/anchors.txt "${SAVE_DIR}"/dcc.txt \
           "${SAVE_DIR}"/cluster_counts.txt "${SAVE_DIR}"/cluster_radii.txt 2>/dev/null || true
    echo ""

    if [[ -f "${SAVE_DIR}/anchors.txt" ]]; then
        echo -e "${BOLD}Sample Decoded Anchors Header & Centroids:${RESET}"
        head -n 12 "${SAVE_DIR}/anchors.txt"
        echo ""
    fi

    echo -e "\n${BOLD}${CYAN}====================================================${RESET}"
    echo -e "${BOLD}${CYAN}       Summary of Step-by-Step Replication Commands ${RESET}"
    echo -e "${BOLD}${CYAN}====================================================${RESET}"
    echo -e "# 1. Generate pattern dataset:"
    log_cmd "${BIN_MKTXTSEQ} ${NB_POINTS} ${PATTERN_FILE} ${PATTERN_TYPE}"
    echo -e "\n# 2. Feed stream into shared memory in background (looping):"
    log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -fps ${STREAM_FPS} -loop &"
    echo -e "\n# 3. Initialize FPS daemon and configure parameters:"
    log_cmd "${BIN_FPSEXEC} -procinfo ${FPS_NAME}:fpsinit"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.in_name ${STREAM_IN}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.out_name ${STREAM_OUT}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.rlim ${DEFAULT_RLIM}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.max_frames ${MAX_FRAMES}"
    log_cmd "${BIN_FPS_SET} ${FPS_NAME}.save_dir ${SAVE_DIR}"
    echo -e "\n# 4. Launch clustering daemon in tmux session:"
    log_cmd "${BIN_FPSEXEC} -tmux -procinfo -loops ${FPS_NAME}:runstart"
    echo -e "\n# 5. Monitor status (live watch):"
    log_cmd "${BIN_GRIC_STATUS} ${FPS_NAME} -w"
    echo -e "\n# 6. Inspect binary output header:"
    log_cmd "${BIN_BIN2ASCII} -i ${SAVE_DIR}/anchors.bin"
    echo -e "\n# 7. Decode binary outputs into ASCII files (with headers):"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/anchors.bin ${SAVE_DIR}/anchors.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/cluster_counts.bin ${SAVE_DIR}/cluster_counts.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/cluster_radii.bin ${SAVE_DIR}/cluster_radii.txt"
    log_cmd "${BIN_BIN2ASCII} -header" \
            "${SAVE_DIR}/dcc.bin ${SAVE_DIR}/dcc.txt"
    echo -e "\n# 8. Stop clustering daemon and clean up:"
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
echo -e "Default pattern: ${GREEN}${PATTERN_TYPE}${RESET} (${NB_POINTS} points)"
read -r -p "Enter number of points [default: ${NB_POINTS}]: " user_pts
if [[ -n "${user_pts}" ]]; then
    NB_POINTS="${user_pts}"
fi

echo -e "Generating ${GREEN}${PATTERN_FILE}${RESET} (${NB_POINTS} points)..."
run_cmd "${BIN_MKTXTSEQ}" "${NB_POINTS}" "${PATTERN_FILE}" "${PATTERN_TYPE}"
echo -e "Sample data (first 3 points):"
head -n 3 "${PATTERN_FILE}"
echo ""

# Step 2: Feed Input Stream
echo -e "${BOLD}${CYAN}--- Step 2: Feed ImageStreamIO Ring Buffer ---${RESET}"
echo -e "Streaming ${GREEN}${PATTERN_FILE}${RESET} to shared memory stream '${STREAM_IN}'."
read -r -p "Streaming frame rate (FPS) [default: ${STREAM_FPS}]: " user_fps
if [[ -n "${user_fps}" ]]; then
    STREAM_FPS="${user_fps}"
fi

log_cmd "${BIN_TXT2STREAM} ${PATTERN_FILE} ${STREAM_IN} -fps ${STREAM_FPS} -loop &"
"${BIN_TXT2STREAM}" "${PATTERN_FILE}" "${STREAM_IN}" -fps "${STREAM_FPS}" -loop &
FEED_PID=$!
sleep 0.5
echo -e "Feeder started (PID: ${FEED_PID}, stream: ${GREEN}${STREAM_IN}${RESET}).\n"

# Step 3: Initialize & Configure FPS Instance
echo -e "${BOLD}${CYAN}--- Step 3: Initialize FPS in Procinfo Mode ---${RESET}"
run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fpsinit"

echo -e "Binding parameters via ${GREEN}milk-fps-set${RESET}:"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.in_name" "${STREAM_IN}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.out_name" "${STREAM_OUT}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${DEFAULT_RLIM}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.max_frames" "${MAX_FRAMES}"
run_cmd "${BIN_FPS_SET}" "${FPS_NAME}.save_dir" "${SAVE_DIR}"

echo -e "\n${BOLD}Active FPS Configuration:${RESET}"
run_cmd "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fps"
echo ""

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
            for artifact in anchors cluster_counts cluster_radii dcc; do
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

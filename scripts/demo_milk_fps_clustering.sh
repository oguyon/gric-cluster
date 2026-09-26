#!/usr/bin/env bash
# ==============================================================================
# scripts/demo_milk_fps_clustering.sh
#
# Interactive demonstration of GRIC stream clustering in Milk framework mode:
# 1. Generates a synthetic dataset (2Dspiral) using gric-mktxtseq.
# 2. Streams coordinates to ImageStreamIO shared memory using gric-txt2stream.
# 3. Initializes and configures the FPS instance with procinfo mode enabled.
# 4. Runs the real-time clustering daemon in an isolated tmux session.
# 5. Provides an interactive menu to read telemetry, live-tune rlim, and exit.
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
FPS_NAME="demo"
STREAM_IN="demo_in"
STREAM_OUT="demo_assign"
PATTERN_FILE="2Dspiral.txt"
PATTERN_TYPE="2Dspiral"
NB_POINTS=2000
STREAM_FPS=100
DEFAULT_RLIM=0.40

FEED_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# ------------------------------------------------------------------------------
# Resolve Executables (PATH or local build/ directory)
# ------------------------------------------------------------------------------
resolve_bin() {
    local bin_name="$1"
    if command -v "$bin_name" >/dev/null 2>&1; then
        command -v "$bin_name"
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

# ------------------------------------------------------------------------------
# Cleanup Handler
# ------------------------------------------------------------------------------
cleanup() {
    echo -e "\n${YELLOW}[Cleanup] Stopping processes and freeing shared memory...${RESET}"
    if [[ -n "${BIN_FPSEXEC}" ]]; then
        "${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:runstop" >/dev/null 2>&1 || true
    fi

    if [[ -n "${FEED_PID}" ]] && kill -0 "${FEED_PID}" 2>/dev/null; then
        kill -INT "${FEED_PID}" 2>/dev/null || true
        wait "${FEED_PID}" 2>/dev/null || true
    fi

    tmux kill-session -t "${FPS_NAME}" 2>/dev/null || true

    # Clean up SHM files created during demo
    local shm_dirs=("/dev/shm" "${MILK_SHM_DIR:-/milk/shm}")
    for d in "${shm_dirs[@]}"; do
        if [[ -d "$d" ]]; then
            rm -f "${d}/${STREAM_IN}"* "${d}/${STREAM_OUT}"* \
                  "${d}/fps.${FPS_NAME}"* "${d}/proc.${FPS_NAME}"* 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}[Cleanup] Done.${RESET}"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# Dependency Verification
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}=== GRIC Milk Framework Clustering Demo ===${RESET}\n"

check_dep() {
    local name="$1"
    local path="$2"
    if [[ -z "$path" ]]; then
        echo -e "${RED}Error:${RESET} Required binary '${name}' not found in PATH or ${BUILD_DIR}."
        echo "Please build the project with Milk support enabled:"
        echo "  cmake -B build -DUSE_MILK=ON && cmake --build build -j\$(nproc)"
        exit 1
    fi
    echo -e "  [x] Found ${GREEN}${name}${RESET} -> ${path}"
}

check_dep "gric-mktxtseq" "${BIN_MKTXTSEQ}"
check_dep "gric-txt2stream" "${BIN_TXT2STREAM}"
check_dep "milk-fpsexec-gric-cluster" "${BIN_FPSEXEC}"
check_dep "milk-fps-set" "${BIN_FPS_SET}"
check_dep "gric-stream-to-pipe" "${BIN_STREAM2PIPE}"
echo ""

# ------------------------------------------------------------------------------
# Step 1: Generate Pattern Dataset
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}--- Step 1: Generate Synthetic Pattern ---${RESET}"
echo -e "Default pattern: ${GREEN}${PATTERN_TYPE}${RESET} (${NB_POINTS} points)"
read -r -p "Enter number of points [default: ${NB_POINTS}]: " user_pts
if [[ -n "${user_pts}" ]]; then
    NB_POINTS="${user_pts}"
fi

echo -e "Generating ${GREEN}${PATTERN_FILE}${RESET} (${NB_POINTS} points)..."
"${BIN_MKTXTSEQ}" "${NB_POINTS}" "${PATTERN_FILE}" "${PATTERN_TYPE}"
echo -e "Sample data (first 3 points):"
head -n 3 "${PATTERN_FILE}"
echo ""

# ------------------------------------------------------------------------------
# Step 2: Feed Input Stream
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}--- Step 2: Feed ImageStreamIO Ring Buffer ---${RESET}"
echo -e "Streaming ${GREEN}${PATTERN_FILE}${RESET} to shared memory stream '${STREAM_IN}'."
read -r -p "Streaming frame rate (FPS) [default: ${STREAM_FPS}]: " user_fps
if [[ -n "${user_fps}" ]]; then
    STREAM_FPS="${user_fps}"
fi

"${BIN_TXT2STREAM}" "${PATTERN_FILE}" "${STREAM_IN}" -fps "${STREAM_FPS}" -loop &
FEED_PID=$!
sleep 0.5
echo -e "Feeder started (PID: ${FEED_PID}, stream: ${GREEN}${STREAM_IN}${RESET}).\n"

# ------------------------------------------------------------------------------
# Step 3: Initialize & Configure FPS Instance
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}--- Step 3: Initialize FPS in Procinfo Mode ---${RESET}"
echo -e "Executing: ${GREEN}${BIN_FPSEXEC} -procinfo ${FPS_NAME}:fpsinit${RESET}"
"${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fpsinit"

echo -e "Binding parameters via ${GREEN}milk-fps-set${RESET}:"
"${BIN_FPS_SET}" "${FPS_NAME}.in_name" "${STREAM_IN}"
"${BIN_FPS_SET}" "${FPS_NAME}.out_name" "${STREAM_OUT}"
"${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${DEFAULT_RLIM}"

echo -e "\n${BOLD}Active FPS Configuration:${RESET}"
"${BIN_FPSEXEC}" -procinfo "${FPS_NAME}:fps"
echo ""

# ------------------------------------------------------------------------------
# Step 4: Launch Clustering Daemon
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}--- Step 4: Launch Clustering Daemon ---${RESET}"
echo -e "Starting daemon in isolated tmux session with semaphore triggering..."
"${BIN_FPSEXEC}" -tmux -procinfo -loops "${FPS_NAME}:runstart"
sleep 1.0

if tmux has-session -t "${FPS_NAME}" 2>/dev/null; then
    echo -e "${GREEN}Daemon successfully running in tmux session '${FPS_NAME}'.${RESET}\n"
else
    echo -e "${RED}Warning: tmux session '${FPS_NAME}' not found.${RESET}\n"
fi

# ------------------------------------------------------------------------------
# Step 5: Interactive Control & Telemetry Loop
# ------------------------------------------------------------------------------
print_menu() {
    echo -e "${BOLD}${CYAN}=== Interactive Control Menu ===${RESET}"
    echo "  [1] Sample live clustering output (5 frames)"
    echo "  [2] Adjust clustering radius (rlim) live"
    echo "  [3] Reset cluster state dynamically (reset_state ON)"
    echo "  [4] Open milk-procCTRL telemetry dashboard"
    echo "  [5] View tmux daemon log"
    echo "  [q] Quit and clean up"
    echo ""
}

read_telemetry_frames() {
    echo -e "\n${MAGENTA}Sampling 5 frames from '${STREAM_OUT}'...${RESET}"
    "${BIN_STREAM2PIPE}" "${STREAM_OUT}" 5 | python3 -c '
import sys, struct
labels = ["cnt0", "cluster_id", "dist", "new_anchor", "nclust", "lat_us", "evals", "aux"]
for i in range(5):
    raw = sys.stdin.buffer.read(64)
    if not raw: break
    vals = struct.unpack("8d", raw)
    print(f"  Frame {i:2d} -> " + " | ".join(f"{k}: {v:6.2f}" for k, v in zip(labels, vals)))
' || echo -e "${YELLOW}Waiting for frames...${RESET}"
    echo ""
}

while true; do
    print_menu
    read -r -p "Select an option [1-5, q]: " choice
    case "$choice" in
        1)
            read_telemetry_frames
            ;;
        2)
            read -r -p "Enter new clustering radius (rlim) [e.g. 0.20, 0.60]: " new_rlim
            if [[ -n "${new_rlim}" ]]; then
                "${BIN_FPS_SET}" "${FPS_NAME}.rlim" "${new_rlim}"
                echo -e "${GREEN}Parameter ${FPS_NAME}.rlim updated to ${new_rlim}.${RESET}\n"
            fi
            ;;
        3)
            echo -e "${YELLOW}Triggering dynamic cluster reset...${RESET}"
            "${BIN_FPS_SET}" "${FPS_NAME}.reset_state" "ON"
            echo -e "${GREEN}Cluster state reset command sent.${RESET}\n"
            ;;
        4)
            if [[ -n "${BIN_PROCCTRL}" ]]; then
                echo -e "${CYAN}Launching milk-procCTRL (press 'q' inside TUI to return)...${RESET}"
                "${BIN_PROCCTRL}" || true
            else
                echo -e "${YELLOW}milk-procCTRL not found in PATH.${RESET}\n"
            fi
            ;;
        5)
            echo -e "\n${CYAN}--- tmux pane demo:2 output ---${RESET}"
            tmux capture-pane -p -t "${FPS_NAME}:2" 2>/dev/null | tail -n 15 || \
                echo "Unable to read tmux pane."
            echo -e "${CYAN}-------------------------------${RESET}\n"
            ;;
        q|Q)
            echo -e "${GREEN}Exiting...${RESET}"
            break
            ;;
        *)
            echo -e "${RED}Invalid option '${choice}'. Please select 1-5 or q.${RESET}\n"
            ;;
    esac
done

#!/usr/bin/env bash
# ==============================================================================
# scripts/perf_baseline.sh
#
# Runtime performance baseline for gric-cluster and gric-knn (workloads W1-W4
# and K1-K2 of docs/dev/performance_review.md).
#
#   scripts/perf_baseline.sh run <outdir> [--bin <builddir>] [--cpu <N>]
#                                [--reps <N>] [--only W1,W2,...] [--no-perf]
#   scripts/perf_baseline.sh compare <outdir_before> <outdir_after>
#
# run:     executes each workload pinned to one CPU (default 4, a P-core on
#          hybrid Intel CPUs) with OMP_NUM_THREADS=1, keeps the fastest of
#          --reps runs (default 1), and adds one `perf stat` run when perf is
#          usable (IPC, page faults, dTLB store misses, TopdownL1). D2/D4 are
#          W2/W4 at default threads (unpinned, OMP_NUM_THREADS unset).
#          Writes <outdir>/summary.tsv and <outdir>/hashes.txt.
# compare: prints a markdown table of wall times and output-hash equality.
#
# Datasets are local (gitignored). Override locations with GRIC_PERF_BENCH
# (default <repo>/benchmarks) and GRIC_PERF_WORKSPACE (default
# <repo>/workspace). Missing datasets are skipped. Existing .gricprof files
# next to the datasets are used, as in normal runs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${REPO_DIR}/build"
BENCH="${GRIC_PERF_BENCH:-${REPO_DIR}/benchmarks}"
WS="${GRIC_PERF_WORKSPACE:-${REPO_DIR}/workspace}"
CPU=4
REPS=1
ONLY=""
USE_PERF=1

usage()
{
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

# ------------------------------------------------------------------------------
# compare mode
# ------------------------------------------------------------------------------
compare_summaries()
{
    local a="$1/summary.tsv"
    local b="$2/summary.tsv"
    [[ -f "${a}" && -f "${b}" ]] || { echo "Missing summary.tsv" >&2; exit 2; }

    echo "| ID | Wall before (s) | Wall after (s) | Speedup | CPU before (s) | CPU after (s)" \
         "| RSS before (MB) | RSS after (MB) | Output |"
    echo "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"
    # summary.tsv columns: id wall user sys rss_mb ipc pf dtlb_st ret fe bs be hash
    awk -F'\t' '
        NR == FNR { if (FNR > 1) { w[$1] = $2; c[$1] = $3 + $4; r[$1] = $5; h[$1] = $13 }
                    next }
        FNR > 1 && ($1 in w) {
            sp = ($2 > 0) ? w[$1] / $2 : 0
            same = (h[$1] == $13) ? "identical" : "**differs**"
            printf "| %s | %.3f | %.3f | %.2fx | %.2f | %.2f | %.0f | %.0f | %s |\n",
                   $1, w[$1], $2, sp, c[$1], $3 + $4, r[$1], $5, same
        }' "${a}" "${b}"
}

MODE="${1:-}"
case "${MODE}" in
    compare)
        [[ $# -eq 3 ]] || usage
        compare_summaries "$2" "$3"
        exit 0
        ;;
    run)
        OUT_DIR="${2:-}"
        [[ -n "${OUT_DIR}" ]] || usage
        shift 2
        ;;
    *)
        usage
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)     BIN_DIR="$2"; shift 2 ;;
        --cpu)     CPU="$2"; shift 2 ;;
        --reps)    REPS="$2"; shift 2 ;;
        --only)    ONLY="$2"; shift 2 ;;
        --no-perf) USE_PERF=0; shift ;;
        *)         echo "Unknown argument: $1" >&2; usage ;;
    esac
done

CLUSTER="${BIN_DIR}/gric-cluster"
KNN="${BIN_DIR}/gric-knn"
for exe in "${CLUSTER}" "${KNN}"; do
    [[ -x "${exe}" ]] || { echo "Missing executable: ${exe}" >&2; exit 2; }
done
[[ -x /usr/bin/time ]] || { echo "GNU time (/usr/bin/time) is required" >&2; exit 2; }

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
RUN_DIR="${OUT_DIR}/run"
mkdir -p "${RUN_DIR}"

# ------------------------------------------------------------------------------
# perf availability and event names (cpu_core/ prefix on hybrid CPUs)
# ------------------------------------------------------------------------------
PMU=""
if [[ -d /sys/bus/event_source/devices/cpu_core ]]; then
    PMU="cpu_core/"
fi
ev()
{
    if [[ -n "${PMU}" ]]; then
        echo "${PMU}$1/"
    else
        echo "$1"
    fi
}
PERF_EVENTS="task-clock,page-faults,$(ev cycles),$(ev instructions),$(ev dTLB-store-misses)"
PERF_TOPDOWN=(-M TopdownL1)

if [[ ${USE_PERF} -eq 1 ]]; then
    if ! command -v perf > /dev/null 2>&1 ||
       ! perf stat -e "$(ev cycles)" true > /dev/null 2>&1; then
        echo "Note: perf not usable (check perf_event_paranoid); counters skipped" >&2
        USE_PERF=0
    elif ! perf stat "${PERF_TOPDOWN[@]}" true > /dev/null 2>&1; then
        PERF_TOPDOWN=()
    fi
fi

# ------------------------------------------------------------------------------
# Workloads: <id> <pinned:1|0> <kind:cluster|knn> <args...>
# K2 uses the cluster output of W2 (run W2 first).
# ------------------------------------------------------------------------------
W3FLAGS="-maxim 30000 -te4 -no-te5 -no-entropy -no-tm -no-pred -gprob -maxvis 20"
W3FLAGS+=" -no-soft-bayesian -no-tiles -no-sparse-dcc -sq16 -no-sq8 -no-eq16 -maxcl 4096 -evals"

WORKLOADS=(
    "W1 1 cluster ${BENCH}/2Drand.txt|-maxcl 20000 -outdir ${RUN_DIR}/W1 0.01"
    "W2 1 cluster ${WS}/128Dtorus.bin|-maxcl 20000 -outdir ${RUN_DIR}/W2 0.7"
    "W3 1 cluster ${WS}/512Dtorus.bin|${W3FLAGS} -outdir ${RUN_DIR}/W3 0.883"
    "W4 1 cluster ${BENCH}/balls_coll.fits|-maxcl 4000 -outdir ${RUN_DIR}/W4 7.4"
    "K1 1 knn ${WS}/128Dtorus.bin|${WS}/128Dtorus.clusterdat -k 10 -nthreads 1"
    "K2 1 knn ${WS}/128Dtorus.bin|${RUN_DIR}/W2 -k 10 -nthreads 1"
    "D2 0 cluster ${WS}/128Dtorus.bin|-maxcl 20000 -outdir ${RUN_DIR}/D2 0.7"
    "D4 0 cluster ${BENCH}/balls_coll.fits|-maxcl 4000 -outdir ${RUN_DIR}/D4 7.4"
)

selected()
{
    [[ -z "${ONLY}" ]] && return 0
    [[ ",${ONLY}," == *",$1,"* ]]
}

# build_cmd <kind> <data> <args> <id>: prints the command line
build_cmd()
{
    local kind="$1" data="$2" args="$3" id="$4"
    if [[ "${kind}" == "cluster" ]]; then
        echo "${CLUSTER} ${args% *} ${args##* } ${data}"
    else
        echo "${KNN} ${data} ${args} -o ${RUN_DIR}/${id}.knn.bin"
    fi
}

# output_hash <kind> <id>
output_hash()
{
    local f
    if [[ "$1" == "cluster" ]]; then
        f="${RUN_DIR}/$2/frame_membership.bin"
    else
        f="${RUN_DIR}/$2.knn.bin_indices.bin"
    fi
    if [[ -f "${f}" ]]; then
        md5sum < "${f}" | cut -c1-12
    else
        echo "missing"
    fi
}

# perf_metrics <csv>: prints "ipc pf dtlb ret fe bs be"
perf_metrics()
{
    awk -F';' '
        $3 ~ /instructions/       && $1 ~ /^[0-9]/ { ins = $1 }
        $3 ~ /cycles/             && $3 !~ /UNHALTED/ && $1 ~ /^[0-9]/ { cyc = $1 }
        $3 == "page-faults"       { pf = $1 }
        $3 ~ /dTLB-store-misses/  && $1 ~ /^[0-9]/ { dt = $1 }
        $6 != "" && $7 ~ /tma_retiring/        && ret == "" { ret = $6 }
        $6 != "" && $7 ~ /tma_frontend_bound/  && fe == ""  { fe = $6 }
        $6 != "" && $7 ~ /tma_bad_speculation/ && bs == ""  { bs = $6 }
        $6 != "" && $7 ~ /tma_backend_bound/   && be == ""  { be = $6 }
        END {
            ipc = (cyc > 0) ? sprintf("%.2f", ins / cyc) : "na"
            printf "%s %s %s %s %s %s %s\n", ipc, (pf == "" ? "na" : pf),
                   (dt == "" ? "na" : dt), (ret == "" ? "na" : ret),
                   (fe == "" ? "na" : fe), (bs == "" ? "na" : bs), (be == "" ? "na" : be)
        }' "$1"
}

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.tsv"
printf "id\twall_s\tuser_s\tsys_s\trss_mb\tipc\tpage_faults\tdtlb_st_miss" > "${SUMMARY}"
printf "\tretiring\tfe_bound\tbad_spec\tbe_bound\thash\n" >> "${SUMMARY}"
: > "${OUT_DIR}/hashes.txt"

{
    echo "date: $(date -Iseconds)"
    echo "git: $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "bin: ${BIN_DIR}"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "pin: ${CPU}, reps: ${REPS}, perf: ${USE_PERF}"
} > "${OUT_DIR}/meta.txt"

for w in "${WORKLOADS[@]}"; do
    read -r id pinned kind rest <<< "${w}"
    selected "${id}" || continue
    data="${rest%%|*}"
    args="${rest#*|}"
    if [[ ! -e "${data}" ]]; then
        echo "${id}: skipped (missing ${data})"
        continue
    fi
    if [[ "${id}" == "K2" && ! -d "${RUN_DIR}/W2" ]]; then
        echo "K2: skipped (needs W2 output)"
        continue
    fi

    cmd=$(build_cmd "${kind}" "${data}" "${args}" "${id}")
    if [[ "${pinned}" == "1" ]]; then
        prefix=(env OMP_NUM_THREADS=1 taskset -c "${CPU}")
    else
        prefix=(env -u OMP_NUM_THREADS)
    fi

    # Timed runs: keep the fastest
    best_wall="" best_line=""
    for ((r = 0; r < REPS; r++)); do
        # shellcheck disable=SC2086
        "${prefix[@]}" /usr/bin/time -f "%e %U %S %M" -o "${RUN_DIR}/${id}.time" \
            ${cmd} > "${RUN_DIR}/${id}.log" 2>&1 || { echo "${id}: FAILED"; continue 2; }
        line=$(tail -1 "${RUN_DIR}/${id}.time")
        wall=${line%% *}
        if [[ -z "${best_wall}" ]] || awk "BEGIN{exit !(${wall} < ${best_wall})}"; then
            best_wall="${wall}"
            best_line="${line}"
        fi
    done
    read -r wall user sys rss_kb <<< "${best_line}"
    rss_mb=$(awk "BEGIN{printf \"%.0f\", ${rss_kb} / 1024}")

    # Counter run
    metrics="na na na na na na na"
    if [[ ${USE_PERF} -eq 1 ]]; then
        csv="${RUN_DIR}/${id}.perf.csv"
        # shellcheck disable=SC2086
        if "${prefix[@]}" perf stat -x';' "${PERF_TOPDOWN[@]}" -e "${PERF_EVENTS}" \
            -o "${csv}" ${cmd} > /dev/null 2>&1; then
            metrics=$(perf_metrics "${csv}")
        fi
    fi

    hash=$(output_hash "${kind}" "${id}")
    echo "${id} ${hash}" >> "${OUT_DIR}/hashes.txt"
    read -r ipc pf dt ret fe bs be <<< "${metrics}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${id}" "${wall}" "${user}" \
        "${sys}" "${rss_mb}" "${ipc}" "${pf}" "${dt}" "${ret}" "${fe}" "${bs}" "${be}" \
        "${hash}" >> "${SUMMARY}"
    printf "%-3s wall %7.3f s  user %7.2f  sys %6.2f  rss %6s MB  ipc %s  hash %s\n" \
        "${id}" "${wall}" "${user}" "${sys}" "${rss_mb}" "${ipc}" "${hash}"
done

echo "Summary: ${SUMMARY}"

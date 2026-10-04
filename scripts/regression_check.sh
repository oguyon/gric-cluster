#!/usr/bin/env bash
# ==============================================================================
# scripts/regression_check.sh
#
# Bit-identity regression check for gric-cluster and gric-knn outputs.
#
# Record a reference once (before a change), then check after the change:
#
#   scripts/regression_check.sh record <refdir> [--bin <builddir>]
#   scripts/regression_check.sh check  <refdir> [--bin <builddir>] [--keep]
#
# record: generates small datasets into <refdir>/data (stored, not regenerated:
#         gric-mktxtseq seeds from the clock), runs the case matrix and writes
#         <refdir>/reference.txt (output hashes + selected STATS_* lines).
# check:  re-runs the matrix on the stored datasets and diffs against the
#         reference. Exit status 1 on any mismatch.
#
# The reference is machine-specific (builds use -march=native): keep <refdir>
# outside the repository. Runs use OMP_NUM_THREADS=1 unless a case sets -ncpu.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${REPO_DIR}/build"
MODE=""
REF_DIR=""
KEEP=0

usage()
{
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        record|check) MODE="$1"; REF_DIR="${2:-}"; shift 2 || usage ;;
        --bin)        BIN_DIR="$2"; shift 2 ;;
        --keep)       KEEP=1; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown argument: $1" >&2; usage ;;
    esac
done
[[ -n "${MODE}" && -n "${REF_DIR}" ]] || usage

CLUSTER="${BIN_DIR}/gric-cluster"
KNN="${BIN_DIR}/gric-knn"
MKSEQ="${BIN_DIR}/gric-mktxtseq"
GENBALLS="${BIN_DIR}/gric-gen-balls"
for exe in "${CLUSTER}" "${KNN}"; do
    [[ -x "${exe}" ]] || { echo "Missing executable: ${exe}" >&2; exit 2; }
done

export OMP_NUM_THREADS=1
DATA_DIR="${REF_DIR}/data"
# Per-case wall-clock limit: a hang shows up as FAILED instead of stalling the run.
CASE_TIMEOUT="${GRIC_REG_TIMEOUT:-120}"

# ------------------------------------------------------------------------------
# Case matrix
#   cluster cases: <name> <dataset> <rlim> <flags...>
#   "auto" cases omit -no-prof and use the dataset profile stored in data/.
# ------------------------------------------------------------------------------
CLUSTER_CASES=(
    "c01_2d_default      2Drand.txt     0.05 -no-prof -maxcl 4000"
    "c02_2d_discard      2Drand.txt     0.05 -no-prof -maxcl 300 -maxcl_strategy discard"
    "c03_2d_merge        2Drand.txt     0.05 -no-prof -maxcl 300 -maxcl_strategy merge"
    "c04_2d_ncpu4        2Drand.txt     0.05 -no-prof -maxcl 4000 -ncpu 4"
    "c05_3d_default      3Dspiral.txt   0.02 -no-prof"
    "c06_3d_pred         3Dspiral.txt   0.02 -no-prof -pred"
    "c07_3d_tm           3Dspiral.txt   0.02 -no-prof -tm 0.3"
    "c08_3d_gprob        3Dspiral.txt   0.02 -no-prof -gprob"
    "c09_3d_te5          3Dspiral.txt   0.02 -no-prof -te5"
    "c10_3d_entropy      3Dspiral.txt   0.02 -no-prof -entropy -no-sq8"
    "c11_3d_sparse       3Dspiral.txt   0.02 -no-prof -sparse_dcc"
    "c12_3d_double       3Dspiral.txt   0.02 -no-prof -double"
    "c13_32d_default     32Dwalk.txt    0.3  -no-prof"
    "c14_32d_sq16        32Dwalk.txt    0.3  -no-prof -sq16"
    "c15_32d_eq16        32Dwalk.txt    0.3  -no-prof -eq16"
    "c16_32d_nosq8       32Dwalk.txt    0.3  -no-prof -no-sq8"
    "c17_32d_ncpu4       32Dwalk.txt    0.3  -no-prof -ncpu 4"
    "c18_32d_auto        32Dwalk.txt    0.3  -sq16"
    "c19_128d_sq16gprob  128Dspiral.txt 0.45 -no-prof -sq16 -gprob -maxvis 20"
    "c20_128d_eq16       128Dspiral.txt 0.45 -no-prof -eq16"
    "c21_balls_default   balls.fits     7.4  -no-prof -maxcl 2000"
    "c22_balls_ncpu8     balls.fits     7.4  -no-prof -maxcl 2000 -ncpu 8"
    "c23_balls_auto      balls.fits     7.4  -maxcl 2000 -eq16"
)

#   knn cases: <name> <cluster case> <dataset> <flags...>
KNN_CASES=(
    "k01_3d_default      c05_3d_default     3Dspiral.txt -k 10"
    "k02_32d_sq16        c14_32d_sq16       32Dwalk.txt  -k 10 -sq16"
    "k03_32d_eq16        c15_32d_eq16       32Dwalk.txt  -k 10 -eq16"
    "k04_2d_threads4     c01_2d_default     2Drand.txt   -k 10 -nthreads 4"
    "k05_balls           c21_balls_default  balls.fits   -k 5 -dtmin 2"
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
hash_file()
{
    if [[ -f "$1" ]]; then
        md5sum < "$1" | cut -c1-12
    else
        echo "missing"
    fi
}

stats_of()
{
    local log="$1/cluster_run.log"
    local out=""
    for key in CLUSTERS FRAMES DISTS DISTS_SAMPLE DISTS_INTERCLUSTER PRUNED; do
        local val
        val=$(grep -m1 "^STATS_${key}:" "${log}" 2>/dev/null | awk '{print $2}')
        out+=" ${key}=${val:-na}"
    done
    echo "${out}"
}

generate_datasets()
{
    mkdir -p "${DATA_DIR}"
    [[ -x "${MKSEQ}" ]] || { echo "Missing executable: ${MKSEQ}" >&2; exit 2; }
    "${MKSEQ}" 4000 "${DATA_DIR}/2Drand.txt" 2Drandom > /dev/null
    "${MKSEQ}" 4000 "${DATA_DIR}/3Dspiral.txt" 3Dspiral -noise 0.02 > /dev/null
    "${MKSEQ}" 3000 "${DATA_DIR}/32Dwalk.txt" 32Dwalk > /dev/null
    "${MKSEQ}" 2000 "${DATA_DIR}/128Dspiral.txt" 128Dspiral -noise 0.05 > /dev/null
    if [[ -x "${GENBALLS}" ]]; then
        "${GENBALLS}" -n 3 -r 5.0 -W 32 -H 32 -f 1500 -s 42 "${DATA_DIR}/balls.fits" \
            > /dev/null 2>&1
    else
        echo "Note: ${GENBALLS} not found (no CFITSIO?): balls cases skipped" >&2
    fi
}

# run_cluster_case <work_dir> <case line>
run_cluster_case()
{
    local work="$1"
    read -r name dataset rlim flags <<< "$2"
    local data="${DATA_DIR}/${dataset}"
    if [[ ! -f "${data}" ]]; then
        echo "${name} skipped (no ${dataset})"
        return
    fi

    local outdir="${work}/${name}"
    # shellcheck disable=SC2086
    if ! timeout "${CASE_TIMEOUT}" "${CLUSTER}" ${flags} -outdir "${outdir}" "${rlim}" "${data}" \
        > "${work}/${name}.log" 2>&1; then
        echo "${name} FAILED (see ${work}/${name}.log)"
        return
    fi

    local fm anchors
    fm=$(hash_file "${outdir}/frame_membership.bin")
    anchors=$(hash_file "${outdir}/anchors.bin")
    echo "${name} fm=${fm} anchors=${anchors}$(stats_of "${outdir}")"
}

# run_knn_case <work_dir> <case line>
run_knn_case()
{
    local work="$1"
    read -r name ccase dataset flags <<< "$2"
    local data="${DATA_DIR}/${dataset}"
    local cdir="${work}/${ccase}"
    if [[ ! -f "${data}" || ! -d "${cdir}" ]]; then
        echo "${name} skipped (no ${dataset} or ${ccase})"
        return
    fi

    # A ".txt" output name selects the text+binary writer for every input type (FITS inputs
    # otherwise default to FITS output); binary files are named <base>_indices.bin etc.
    local base="${work}/${name}"
    # shellcheck disable=SC2086
    if ! timeout "${CASE_TIMEOUT}" "${KNN}" "${data}" "${cdir}" ${flags} -o "${base}.txt" \
        > "${work}/${name}.log" 2>&1; then
        echo "${name} FAILED (see ${work}/${name}.log)"
        return
    fi

    # Multithreaded k-NN distances are reproducible only to ~1 ULP (the value for a pair
    # depends on which thread computes it first), so only the indices are hashed there.
    if [[ " ${flags} " == *" -nthreads "* ]]; then
        echo "${name} idx=$(hash_file "${base}_indices.bin")" \
             "mutual=$(hash_file "${base}_mutual_dists.bin")"
        return
    fi
    echo "${name} idx=$(hash_file "${base}_indices.bin")" \
         "dist=$(hash_file "${base}_distances.bin")" \
         "mutual=$(hash_file "${base}_mutual_dists.bin")" \
         "txt=$(hash_file "${base}.txt")"
}

# run_matrix <work_dir> <results_file>
run_matrix()
{
    local work="$1"
    local results="$2"
    : > "${results}"
    for c in "${CLUSTER_CASES[@]}"; do
        run_cluster_case "${work}" "${c}" | tee -a "${results}"
    done
    for c in "${KNN_CASES[@]}"; do
        run_knn_case "${work}" "${c}" | tee -a "${results}"
    done
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if [[ "${MODE}" == "record" ]]; then
    if [[ -e "${REF_DIR}/reference.txt" ]]; then
        echo "Reference already exists: ${REF_DIR}/reference.txt (remove it first)" >&2
        exit 2
    fi
    mkdir -p "${REF_DIR}"
    generate_datasets

    # Warm-up pass: lets "auto" cases write their dataset profiles, so that the
    # recorded pass loads them from disk exactly as later check runs will.
    WARM=$(mktemp -d)
    run_matrix "${WARM}" "${WARM}/results.txt" > /dev/null
    rm -rf "${WARM}"

    WORK="${REF_DIR}/work"
    rm -rf "${WORK}"
    mkdir -p "${WORK}"
    run_matrix "${WORK}" "${REF_DIR}/reference.txt"
    rm -rf "${WORK}"

    {
        echo "date: $(date -Iseconds)"
        echo "git: $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "bin: ${BIN_DIR}"
        echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    } > "${REF_DIR}/meta.txt"
    echo "Reference recorded in ${REF_DIR}/reference.txt"
    exit 0
fi

# check mode
[[ -f "${REF_DIR}/reference.txt" ]] || { echo "No reference in ${REF_DIR}" >&2; exit 2; }
WORK=$(mktemp -d)
run_matrix "${WORK}" "${WORK}/results.txt" > /dev/null

if diff -u "${REF_DIR}/reference.txt" "${WORK}/results.txt" > "${WORK}/diff.txt"; then
    echo "PASS: $(wc -l < "${WORK}/results.txt") cases identical to reference"
    STATUS=0
else
    echo "FAIL: outputs differ from reference ($(grep -c '^+[a-z]' "${WORK}/diff.txt") cases)"
    cat "${WORK}/diff.txt"
    STATUS=1
fi

if [[ ${KEEP} -eq 1 || ${STATUS} -ne 0 ]]; then
    echo "Work directory kept: ${WORK}"
else
    rm -rf "${WORK}"
fi
exit ${STATUS}

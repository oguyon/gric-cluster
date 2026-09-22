#!/usr/bin/env python3
"""
benchmark_rq8_e8.py - Benchmark comparing RQ8 variants (Cubic vs E8 SDC vs E8 ADC vs SparseCache)
                      and EQ16 across dimensions and on the 512D Torus Knot dataset.

Evaluates:
  1. Footprint (Index RAM and bytes/dim)
  2. Search speed (ms, FPS)
  3. Metric pruning efficiency and framedist computations
  4. Exact recall@k against brute-force ground truth
"""

import os
import sys
import re
import subprocess
import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
TMP_DIR = "/tmp/benchmark_rq8_e8"
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))

os.makedirs(TMP_DIR, exist_ok=True)


def run_cmd(cmd):
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        print(f"Command failed: {cmd}\nStderr: {p.stderr}", file=sys.stderr)
    return p.stdout, p.stderr, p.returncode


def parse_knn_output(stdout):
    telem = {
        "load_ms": 0.0,
        "search_ms": 0.0,
        "fps": 0.0,
        "framedists": 0,
        "prune_pct": 0.0,
        "index_mb": 0.0,
        "rq8_evals": 0,
        "rq8_pruned": 0,
    }
    m_load = re.search(r"Loaded Pass 1 Model in\s+([\d\.]+)\s+ms", stdout)
    if m_load:
        telem["load_ms"] = float(m_load.group(1))

    m_mem = re.search(r"Built RQ8 SIMD transposed blocks:\s+([\d\.]+)\s+MB", stdout)
    if m_mem:
        telem["index_mb"] = float(m_mem.group(1))
    elif "RQ8 SparseCache Active: 0.00 MB resident index" in stdout:
        telem["index_mb"] = 0.0

    m_time = re.search(
        r"Search Wall Time:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+fps\)", stdout
    )
    if m_time:
        telem["search_ms"] = float(m_time.group(1))
        telem["fps"] = float(m_time.group(2))

    m_dist = re.search(r"Framedist Computations:\s+(\d+)", stdout)
    if m_dist:
        telem["framedists"] = int(m_dist.group(1))

    m_prune = re.search(r"Metric Pruning Efficiency:\s+([\d\.]+)%", stdout)
    if m_prune:
        telem["prune_pct"] = float(m_prune.group(1))

    m_rq8 = re.search(r"RQ8 Evaluations:\s+(\d+)", stdout)
    if m_rq8:
        telem["rq8_evals"] = int(m_rq8.group(1))

    m_pruned = re.search(r"RQ8 Lower-Bound Pruned:\s+(\d+)", stdout)
    if m_pruned:
        telem["rq8_pruned"] = int(m_pruned.group(1))

    return telem


def load_knn_results(path):
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = int(parts[0])
                n_ids = [int(parts[i]) for i in range(1, len(parts), 2)]
                results[qid] = n_ids
    return results


def compute_recall(gt_dict, cand_dict, k=10):
    if not gt_dict or not cand_dict:
        return 0.0
    common = set(gt_dict.keys()).intersection(set(cand_dict.keys()))
    if not common:
        return 0.0
    total_match = 0
    total_needed = 0
    for q in common:
        s_gt = set(gt_dict[q][:k])
        s_cand = set(cand_dict[q][:k])
        total_match += len(s_gt.intersection(s_cand))
        total_needed += len(s_gt)
    return (total_match / total_needed * 100.0) if total_needed > 0 else 0.0


def benchmark_dataset(name, dataset_path, cluster_dir, dim, n_samples, k=10, dtmin=5):
    print("=" * 96)
    print(f" BENCHMARK: {name} (D = {dim}, N = {n_samples}, k = {k})")
    print("=" * 96)

    gt_out = f"{TMP_DIR}/out_gt_{dim}D.txt"
    cubic_out = f"{TMP_DIR}/out_cubic_{dim}D.txt"
    sdc_out = f"{TMP_DIR}/out_sdc_{dim}D.txt"
    adc_fs_out = f"{TMP_DIR}/out_adc_fs_{dim}D.txt"
    adc_sp_out = f"{TMP_DIR}/out_adc_sp_{dim}D.txt"

    # 1. Exact Ground Truth
    print("  [1/5] Exact Ground Truth...")
    run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -no-rq8 -no-eq16 -o {gt_out}"
    )
    gt_res = load_knn_results(gt_out)

    # 2. Cubic RQ8
    print("  [2/5] Cubic RQ8 (Transposed FastScan)...")
    stdout_c, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -rq8 -no-e8 -no-rq8-adc -no-rq8-sparse -o {cubic_out}"
    )
    t_c = parse_knn_output(stdout_c)
    rec_c = compute_recall(gt_res, load_knn_results(cubic_out), k)

    # 3. RQ8-E8 SDC FastScan
    print("  [3/5] RQ8-E8 SDC (Transposed FastScan)...")
    stdout_sdc, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -rq8 -e8 -no-rq8-adc -no-rq8-sparse -o {sdc_out}"
    )
    t_sdc = parse_knn_output(stdout_sdc)
    rec_sdc = compute_recall(gt_res, load_knn_results(sdc_out), k)

    # 4. RQ8-E8 ADC FastScan
    print("  [4/5] RQ8-E8 ADC (Transposed FastScan)...")
    stdout_adc_fs, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -rq8 -e8 -rq8-adc -no-rq8-sparse -o {adc_fs_out}"
    )
    t_adc_fs = parse_knn_output(stdout_adc_fs)
    rec_adc_fs = compute_recall(gt_res, load_knn_results(adc_fs_out), k)

    # 5. RQ8-E8 ADC SparseCache
    print("  [5/5] RQ8-E8 ADC SparseCache (0 MB resident index)...")
    stdout_adc_sp, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -rq8 -e8 -rq8-adc -rq8-sparse -o {adc_sp_out}"
    )
    t_adc_sp = parse_knn_output(stdout_adc_sp)
    rec_adc_sp = compute_recall(gt_res, load_knn_results(adc_sp_out), k)

    # Summary Table
    print("\n  COMPARISON:")
    print("  " + "-" * 92)
    print(
        f"  {'Method':<24} | {'RAM (MB)':<10} | {'Time (ms)':<10} | {'FPS':<9} | "
        f"{'Framedists':<11} | {'Pruned':<10} | {'Recall':<8}"
    )
    print("  " + "-" * 92)
    methods = [
        ("Cubic RQ8 (FS)", t_c, rec_c),
        ("RQ8-E8 SDC (FS)", t_sdc, rec_sdc),
        ("RQ8-E8 ADC (FS)", t_adc_fs, rec_adc_fs),
        ("RQ8-E8 ADC (Sparse)", t_adc_sp, rec_adc_sp),
    ]
    for mname, t, rec in methods:
        print(
            f"  {mname:<24} | {t['index_mb']:>8.2f} MB | {t['search_ms']:>8.2f} ms | "
            f"{t['fps']:>7.0f} | {t['framedists']:>11} | {t['rq8_pruned']:>10} | {rec:>7.2f}%"
        )
    print("  " + "-" * 92)
    speedup = t_sdc["search_ms"] / t_adc_sp["search_ms"] if t_adc_sp["search_ms"] > 0 else 1.0
    prune_gain = t_adc_sp["rq8_pruned"] - t_c["rq8_pruned"]
    print(f"  >> ADC + Sparse Speedup vs SDC: {speedup:.2f}x")
    print(f"  >> Extra candidates pruned vs Cubic: +{prune_gain:,d}")
    print()

    return {
        "name": name,
        "dim": dim,
        "n_samples": n_samples,
        "cubic": t_c,
        "sdc": t_sdc,
        "adc_fs": t_adc_fs,
        "adc_sp": t_adc_sp,
    }


def main():
    print("Starting RQ8-E8 Benchmark Suite (Cubic vs E8 SDC vs E8 ADC vs SparseCache)...\n")
    results = []

    # Test synthetic dimensions: 8D, 16D, 32D, 64D
    for dim in [8, 16, 32, 64]:
        n_samples = 3000
        dataset_path = f"{TMP_DIR}/data_{dim}D.txt"
        cluster_dir = f"{TMP_DIR}/cluster_rq8_{dim}D"

        run_cmd(f"{BUILD_DIR}/gric-mktxtseq {n_samples} {dataset_path} {dim}Dwalk")
        rlim = 0.20 * np.sqrt(dim / 2.0)
        run_cmd(
            f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -outdir {cluster_dir}"
        )

        res = benchmark_dataset(
            f"Synthetic {dim}D Walk",
            dataset_path,
            cluster_dir,
            dim,
            n_samples,
            k=10,
            dtmin=5,
        )
        results.append(res)

    # Test 512D Torus Knot
    torus_bin = f"{WORKSPACE_DIR}/512Dtorus.bin"
    torus_cl = f"{WORKSPACE_DIR}/512Dtorus.clusterdat"
    if os.path.exists(torus_bin) and os.path.exists(torus_cl):
        res = benchmark_dataset(
            "512D Torus Knot Dataset",
            torus_bin,
            torus_cl,
            512,
            30000,
            k=10,
            dtmin=10,
        )
        results.append(res)

    print("\n" + "#" * 96)
    print(" GRAND BENCHMARK SUMMARY: RQ8-E8 ADVANCEMENTS")
    print("#" * 96)
    print(
        f"{'Dataset / Dim':<24} | {'Cubic Pruned':<14} | {'E8 ADC Pruned':<14} | "
        f"{'Sparse RAM':<11} | {'E8 ADC Speedup':<14}"
    )
    print("-" * 96)
    for r in results:
        prune_c = f"{r['cubic']['rq8_pruned']:,}"
        prune_adc = f"{r['adc_sp']['rq8_pruned']:,}"
        ram_sp = f"{r['adc_sp']['index_mb']:.1f} MB"
        t_sdc = r['sdc']['search_ms']
        t_sp = r['adc_sp']['search_ms']
        spd = t_sdc / t_sp if t_sp > 0 else 1.0
        spd_str = f"{spd:.2f}x ({r['adc_sp']['fps']:.0f} fps)"
        print(
            f"{r['name']:<24} | {prune_c:<14} | {prune_adc:<14} | {ram_sp:<11} | {spd_str:<14}"
        )
    print("-" * 96)


if __name__ == "__main__":
    main()
